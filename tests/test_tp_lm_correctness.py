"""
Phase 3 prerequisite: the full TPMambaLM (stacked TP blocks + cache) must
match the dense MambaLM end-to-end, using the real mamba-130m-hf checkpoint,
loaded shard-by-shard directly from the safetensors file (no full dense
model ever materialized in either process).

Still gloo/CPU/2-process here — this is what a `torchrun --nproc_per_node=2`
run with `nccl` will do on Kaggle's 2xT4, minus the device and backend.
"""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download

from mamba_lite.model import MambaLM
from mamba_lite.tp_model import TPMambaLM

WORLD_SIZE = 2
MODEL_ID = "state-spaces/mamba-130m-hf"


def _worker(rank, world_size, init_file, model_path, result_queue):
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    tp_model = TPMambaLM.from_pretrained_shard(model_path, rank=rank, world_size=world_size).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, tp_model.cfg.vocab_size, (1, 12))

    with torch.no_grad():
        tp_logits, _ = tp_model.prefill(ids)
        tp_tokens = tp_model.generate(ids, max_new_tokens=8)
        comm_count = tp_model.layers[0].mixer.comm.count

    if rank == 0:
        result_queue.put((tp_logits, tp_tokens, comm_count))
    dist.destroy_process_group()


def test_tp_lm_matches_dense_lm():
    model_path = snapshot_download(MODEL_ID)
    dense = MambaLM.from_pretrained(model_path).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, dense.cfg.vocab_size, (1, 12))
    with torch.no_grad():
        dense_logits, _ = dense.prefill(ids)
        dense_tokens = dense.generate(ids, max_new_tokens=8)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "rendezvous")
        procs = [
            ctx.Process(target=_worker, args=(r, WORLD_SIZE, init_file, model_path, result_queue))
            for r in range(WORLD_SIZE)
        ]
        for p in procs:
            p.start()
        tp_logits, tp_tokens, comm_count = result_queue.get(timeout=120)
        for p in procs:
            p.join(timeout=120)

    torch.testing.assert_close(tp_logits, dense_logits, atol=1e-3, rtol=1e-3)
    assert torch.equal(tp_tokens, dense_tokens), (
        f"tp: {tp_tokens.tolist()}\ndense: {dense_tokens.tolist()}"
    )
    # Each prefill() call does 2 AllReduces per layer (batched over the whole
    # sequence, not per token). We call prefill() once directly, then
    # generate() internally calls prefill() once more and step() 7 times
    # (max_new_tokens=8 -> 1 prefill-derived token + 7 decode steps):
    # 2*(1 + 1 + 7) = 18, counted on layer 0's mixer only.
    assert comm_count == 18, f"expected 18 AllReduces on layer 0, got {comm_count}"

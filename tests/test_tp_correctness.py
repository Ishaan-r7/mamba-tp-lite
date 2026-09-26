"""
Phase 2 correctness test. Spawns 2 real processes (gloo backend, CPU —
works anywhere, no GPU needed) and checks:

  1. TPMambaMixer's output matches a single-process dense MambaMixer exactly
     (up to float tolerance) — the sharded math is equivalent to the original.
  2. It uses exactly 2 AllReduces per forward call.
  3. NaiveTPMambaMixer, which slices the packed in_proj output the "obvious"
     but wrong way, produces a WRONG result — demonstrating the packed-tensor
     pitfall from the paper's Section IV-C.

This mirrors what `torchrun --nproc_per_node=2 ...` with `nccl` will do on
real GPUs (Kaggle 2xT4) — only the backend and device change.
"""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mamba_lite.model import MambaConfig, MambaMixer
from mamba_lite.tp_model import NaiveTPMambaMixer, TPMambaMixer

WORLD_SIZE = 2


def _small_config() -> MambaConfig:
    # small but not tiny: exercises real shard math without being slow on CPU
    return MambaConfig(d_model=32, n_layer=1, vocab_size=100, d_state=8, d_conv=4, expand=2)


def _worker(rank, world_size, init_file, result_queue, variant: str):
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    torch.manual_seed(0)  # same seed on every rank -> same dense reference weights
    cfg = _small_config()
    dense = MambaMixer(cfg, layer_idx=0)
    dense.eval()

    cls = TPMambaMixer if variant == "correct" else NaiveTPMambaMixer
    tp = cls(cfg, rank=rank, world_size=world_size)
    tp.load_shard_from_dense(dense)
    tp.eval()

    torch.manual_seed(123)  # same seed on every rank -> same input tensor
    x = torch.randn(2, 5, cfg.d_model)

    with torch.no_grad():
        dense_out = dense(x)  # HF-style forward from mamba_lite.model.MambaMixer... see note below
        tp_out = tp(x)

    if rank == 0:
        result_queue.put((dense_out, tp_out, tp.comm.count))
    dist.destroy_process_group()


def _run(variant: str):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "rendezvous")
        procs = [
            ctx.Process(target=_worker, args=(r, WORLD_SIZE, init_file, result_queue, variant))
            for r in range(WORLD_SIZE)
        ]
        for p in procs:
            p.start()
        dense_out, tp_out, comm_count = result_queue.get(timeout=60)
        for p in procs:
            p.join(timeout=60)
    return dense_out, tp_out, comm_count


def test_correct_tp_matches_dense_with_two_allreduces():
    dense_out, tp_out, comm_count = _run("correct")
    torch.testing.assert_close(tp_out, dense_out, atol=1e-4, rtol=1e-4)
    assert comm_count == 2, f"expected exactly 2 AllReduces per block, got {comm_count}"


def test_naive_tp_gives_wrong_result():
    dense_out, tp_out, _ = _run("naive")
    assert not torch.allclose(tp_out, dense_out, atol=1e-2), (
        "naive packed-tensor split was expected to break correctness, but matched anyway"
    )

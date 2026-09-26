"""
Phase 4: quantized AllReduce. Two things to check, in the same spirit as the
paper's Table I:

  1. Quantized TP output is CLOSE to (not exactly equal to) the fp32
     reference -- quantization is lossy by design, so this uses a looser
     tolerance than the Phase 2/3 exact-match tests.
  2. Greedy next-token agreement (Top-1) and Top-k candidate overlap between
     fp32 and quantized generation, over a real prompt on mamba-130m-hf --
     the same style of accuracy metric the paper reports (its Table I).

Still gloo/CPU/2-process; the actual speedup only shows up on real GPU
bandwidth (see scripts/run_tp.py --allreduce-dtype and
benchmarks/PHASE3_RESULTS.md's bandwidth-vs-latency prediction for Phase 4).
"""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download

from mamba_lite.model import MambaConfig, MambaMixer
from mamba_lite.tp_model import TPMambaLM, TPMambaMixer

WORLD_SIZE = 2
MODEL_ID = "state-spaces/mamba-130m-hf"


def _small_config() -> MambaConfig:
    return MambaConfig(d_model=32, n_layer=1, vocab_size=100, d_state=8, d_conv=4, expand=2)


def _mixer_worker(rank, world_size, init_file, result_queue, allreduce_dtype):
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    cfg = _small_config()
    dense = MambaMixer(cfg, layer_idx=0).eval()

    tp = TPMambaMixer(cfg, rank=rank, world_size=world_size, allreduce_dtype=allreduce_dtype)
    tp.load_shard_from_dense(dense)
    tp.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 5, cfg.d_model)
    with torch.no_grad():
        dense_out = dense(x)
        tp_out = tp(x)

    if rank == 0:
        result_queue.put((dense_out, tp_out))
    dist.destroy_process_group()


def _run_mixer(allreduce_dtype: str):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "rendezvous")
        procs = [
            ctx.Process(target=_mixer_worker, args=(r, WORLD_SIZE, init_file, q, allreduce_dtype))
            for r in range(WORLD_SIZE)
        ]
        for p in procs:
            p.start()
        dense_out, tp_out = q.get(timeout=60)
        for p in procs:
            p.join(timeout=60)
    return dense_out, tp_out


def test_fp16_allreduce_close_but_not_exact():
    dense_out, tp_out = _run_mixer("fp16")
    # close: fp16 has ~3 decimal digits of precision -> allow a looser tolerance
    torch.testing.assert_close(tp_out, dense_out, atol=1e-2, rtol=1e-2)
    # but NOT exact -- if this ever passes at fp32 tolerance, quantization silently no-op'd
    assert not torch.allclose(tp_out, dense_out, atol=1e-6), (
        "fp16 quantized output matched fp32 reference to fp32 precision -- "
        "quantization path likely isn't actually being used"
    )


def test_int8_allreduce_close_but_noticeably_lossy():
    dense_out, tp_out = _run_mixer("int8")
    # int8 is much coarser (~2 significant digits); just check it's in the right ballpark
    torch.testing.assert_close(tp_out, dense_out, atol=0.1, rtol=0.2)


def _lm_worker(rank, world_size, init_file, model_path, ids, result_queue, allreduce_dtype):
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    model = TPMambaLM.from_pretrained_shard(
        model_path, rank=rank, world_size=world_size, allreduce_dtype=allreduce_dtype
    ).eval()

    with torch.no_grad():
        logits, _ = model.prefill(ids)

    if rank == 0:
        result_queue.put(logits)
    dist.destroy_process_group()


def _run_lm(model_path, ids, allreduce_dtype: str):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "rendezvous")
        procs = [
            ctx.Process(target=_lm_worker, args=(r, WORLD_SIZE, init_file, model_path, ids, q, allreduce_dtype))
            for r in range(WORLD_SIZE)
        ]
        for p in procs:
            p.start()
        logits = q.get(timeout=120)
        for p in procs:
            p.join(timeout=120)
    return logits


def test_quantization_accuracy_table_like_paper():
    """Reports Top-1 / Top-5 (unordered) / Top-5 (ordered) agreement between
    fp32 and quantized AllReduce on real mamba-130m weights, over real
    tokenized text (not random token ids -- random ids produce
    out-of-distribution activations with an unrealistically wide dynamic
    range, which especially inflates int8's error) -- the same metrics as
    the paper's Table I. Prints the table. fp16 asserts a tight floor
    (it should be near-lossless); int8 is diagnostic only, since how lossy
    naive per-tensor int8 AllReduce is *after compounding over every layer*
    is exactly the open question this test exists to answer."""
    from transformers import AutoTokenizer

    model_path = snapshot_download(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(model_path)
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "State space models process sequences token by token, maintaining "
        "a compact hidden state instead of attending over the full history. "
        "This makes inference cost scale linearly with sequence length."
    )
    ids = tok(text, return_tensors="pt").input_ids  # real text -> realistic activation ranges

    fp32_logits = _run_lm(model_path, ids, "fp32")
    k = 5
    fp32_top1 = fp32_logits[0].argmax(dim=-1)  # [seq_len]
    fp32_topk = fp32_logits[0].topk(k, dim=-1).indices  # [seq_len, k]

    print(f"\n(prompt length: {ids.shape[1]} real tokens)")
    print(f"{'precision':>10} | {'top1 %':>8} | {'top5 unordered %':>18} | {'top5 ordered %':>16}")
    for dtype in ["fp16", "int8"]:
        q_logits = _run_lm(model_path, ids, dtype)
        q_top1 = q_logits[0].argmax(dim=-1)
        q_topk = q_logits[0].topk(k, dim=-1).indices

        top1_match = (q_top1 == fp32_top1).float().mean().item() * 100

        unordered = torch.tensor(
            [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(fp32_topk, q_topk)]
        ).mean().item() * 100

        ordered_match = (q_topk == fp32_topk).all(dim=-1).float().mean().item() * 100

        print(f"{dtype:>10} | {top1_match:>7.2f}% | {unordered:>17.2f}% | {ordered_match:>15.2f}%")

        if dtype == "fp16":
            # fp16 should be close to lossless -- this IS a tight bound
            assert top1_match > 90, f"fp16 Top-1 agreement unexpectedly low: {top1_match:.1f}%"
        # int8: no assertion. 24 layers x 2 AllReduces = 48 compounded
        # quantization events per forward pass; whether that's still usable
        # is the finding this test reports, not something to assume.

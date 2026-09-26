# Mamba-TP Lite

A from-scratch, learning-focused reimplementation of the ideas in
["Scaling State-Space Models on Multiple GPUs with Tensor Parallelism"](https://arxiv.org/abs/2602.21144)
(Dutt, Shah, Masarani, Gandhi — Stony Brook University).

Not benchmarked against the paper's numbers — this is for understanding SSM
inference and tensor-parallel systems design, not a reproduction. See
[PLAN.md](PLAN.md) for the phased plan and what each phase teaches, and
[NOTES.md](NOTES.md) for a per-phase goal/action/result/unexpected/fix log.

## What's here

| Phase | File | What it does |
|---|---|---|
| 1 | [`mamba_lite/model.py`](mamba_lite/model.py) | Mamba from scratch (loads real HF weights). `prefill()` = full scan, `step()` = cached recurrent decode. |
| 2 | [`mamba_lite/tp_model.py`](mamba_lite/tp_model.py) | Tensor-parallel mixer: channel-sharded, exactly 2 AllReduces/block. Includes a deliberately broken `NaiveTPMambaMixer` for comparison. |
| 3 | [`scripts/run_tp.py`](scripts/run_tp.py) | `torchrun`-launched benchmark/profiler: TP vs single-GPU throughput, `torch.profiler` traces. |
| 4 | [`mamba_lite/tp_utils.py`](mamba_lite/tp_utils.py) | Quantized AllReduce (FP16, INT8) + a paper-Table-I-style accuracy check. |

## Results

**Phase 1 — the SSM cache keeps decode cost flat, not growing with prompt length** (CPU, mamba-130m):

<img src="assets/phase1_cache_speedup.png" width="500">

**Phase 3 — single-token decode is communication-bound; TP loses to a single GPU** (Kaggle 2xT4, batch=1):

<img src="assets/phase3_decode_tp_vs_1gpu.png" width="500">

**...but batched prefill flips this once a single GPU's compute saturates:**

<img src="assets/phase3_prefill_crossover.png" width="700">

**Phase 4 — FP16 AllReduce is nearly lossless; naive INT8 wrecks fine-grained ranking after compounding over every layer:**

<img src="assets/phase4_quant_accuracy.png" width="500">

**Same-session controlled result: quantization helps the bandwidth-relevant case and actively hurts the latency-bound one:**

<img src="assets/phase4_fp16_speed_change.png" width="500">

Full numbers and analysis: [`benchmarks/PHASE3_RESULTS.md`](benchmarks/PHASE3_RESULTS.md), [`benchmarks/PHASE4_RESULTS.md`](benchmarks/PHASE4_RESULTS.md). Regenerate these charts with `.venv/bin/python scripts/make_plots.py`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install torch transformers safetensors huggingface_hub pytest
```

## Tests (all run on CPU, no GPU needed)

```bash
.venv/bin/python -m pytest tests/ -v
```

- `test_correctness.py` — our Mamba vs HF's `MambaForCausalLM`.
- `test_tp_correctness.py` — 2-process (`gloo`) mixer-level TP correctness + AllReduce count + the naive-split failure demo.
- `test_tp_lm_correctness.py` — same, for the full stacked model.
- `test_quantized_allreduce.py` — quantized AllReduce correctness + the accuracy table above.

## Benchmarks

```bash
# Phase 1: cache vs no-cache decode latency (CPU is fine)
.venv/bin/python benchmarks/bench_cache.py --prompt-lens 16 64 256

# Phase 3/4: TP throughput, optional profiler trace, optional quantized AllReduce
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend gloo --device cpu   # local smoke test
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend nccl --device cuda \
  --compare-single-gpu --profile --allreduce-dtype fp16   # on a real 2-GPU box (e.g. Kaggle "GPU T4 x2")
```

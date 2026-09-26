# Mamba-TP Lite

A from-scratch, learning-focused reimplementation of the ideas in
["Scaling State-Space Models on Multiple GPUs with Tensor Parallelism"](https://arxiv.org/abs/2602.21144)
(Dutt, Shah, Masarani, Gandhi — Stony Brook University).

Not benchmarked against the paper's numbers — this is for understanding SSM
inference and tensor-parallel systems design, not a reproduction. See
[PLAN.md](PLAN.md) for the phased plan and what each phase teaches.

## What's here

| Phase | File | What it does |
|---|---|---|
| 1 | [`mamba_lite/model.py`](mamba_lite/model.py) | Mamba from scratch (loads real HF weights). `prefill()` = full scan, `step()` = cached recurrent decode. |
| 2 | [`mamba_lite/tp_model.py`](mamba_lite/tp_model.py) | Tensor-parallel mixer: channel-sharded, exactly 2 AllReduces/block. Includes a deliberately broken `NaiveTPMambaMixer` for comparison. |
| 3 | [`scripts/run_tp.py`](scripts/run_tp.py) | `torchrun`-launched benchmark/profiler: TP vs single-GPU throughput, `torch.profiler` traces. |
| 4 | *(next)* | Quantized AllReduce (FP16, INT8) + accuracy check. |

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

## Benchmarks

```bash
# Phase 1: cache vs no-cache decode latency (CPU is fine)
.venv/bin/python benchmarks/bench_cache.py --prompt-lens 16 64 256

# Phase 3: TP throughput + optional profiler trace
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend gloo --device cpu   # local smoke test
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend nccl --device cuda --compare-single-gpu --profile  # on a real 2-GPU box (e.g. Kaggle "GPU T4 x2")
```

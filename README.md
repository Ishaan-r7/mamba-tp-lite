# Mamba-TP Lite

Tensor-parallel inference for the Mamba state-space model, implemented from
scratch in PyTorch: a Mamba forward pass verified against Hugging Face's
reference implementation, channel-sharded tensor parallelism with exactly 2
AllReduces per block, quantized communication (FP16/INT8), and CUDA
graph-captured decode. Benchmarked on real 2xT4 GPUs (Kaggle).

## Results

**CUDA graph decode: 2.5–3.7x faster, verified bit-exact against eager execution**

<img src="assets/phase5_cuda_graph_speedup.png" width="700">

**Tensor parallelism: ~1.7–1.8x prefill throughput once batch size saturates a single GPU's compute**

<img src="assets/phase3_prefill_crossover.png" width="700">

Single-token decode is a different story — communication overhead dominates at low batch and tensor parallelism is slower than one GPU:

<img src="assets/phase3_decode_tp_vs_1gpu.png" width="500">

**SSM cache: decode cost stays flat regardless of prompt length (up to 73x faster than recomputing the prefix from scratch)**

<img src="assets/phase1_cache_speedup.png" width="500">

**Mixed-precision AllReduce: 43 of 48 collectives in INT8 at KL 0.0005 vs fp32 (Top-1 98.8%), with per-token scales and int8 on the wire. On 2x T4 over PCIe the speed gain isn't there: FP16 is about +5% prefill / -5% decode, INT8 is within noise on prefill and ~30% slower on decode**

<img src="assets/phase6_site_sensitivity.png" width="700">
<img src="assets/phase6_pareto.png" width="500">
<img src="assets/phase6_gpu_speed.png" width="600">

Full numbers: [`benchmarks/PHASE3_RESULTS.md`](benchmarks/PHASE3_RESULTS.md), [`benchmarks/PHASE4_RESULTS.md`](benchmarks/PHASE4_RESULTS.md), [`benchmarks/PHASE5_RESULTS.md`](benchmarks/PHASE5_RESULTS.md), [`benchmarks/PHASE6_RESULTS.md`](benchmarks/PHASE6_RESULTS.md). Regenerate charts with `.venv/bin/python scripts/make_plots.py`.

## Components

| | File | |
|---|---|---|
| Model | [`mamba_lite/model.py`](mamba_lite/model.py) | Mamba forward pass: `prefill()` for the full scan, `step()` for cached recurrent decode. |
| Tensor parallelism | [`mamba_lite/tp_model.py`](mamba_lite/tp_model.py) | Channel-sharded mixer, exactly 2 AllReduces per block. Includes a deliberately incorrect `NaiveTPMambaMixer` used in tests to catch a packed-tensor sharding bug. |
| Benchmark/profiling | [`scripts/run_tp.py`](scripts/run_tp.py) | `torchrun`-launched throughput and `torch.profiler` harness. |
| Quantized communication | [`mamba_lite/tp_utils.py`](mamba_lite/tp_utils.py) | FP16 and INT8 collectives (INT8 = per-token-scaled all-gather, int8 on the wire), selectable per collective site; KL/Top-k eval harness in [`mamba_lite/quant_eval.py`](mamba_lite/quant_eval.py). |
| CUDA graphs | [`mamba_lite/cuda_graph.py`](mamba_lite/cuda_graph.py) | Graph-captured decode with a numerical correctness check run before any speedup is reported. |

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

- `test_correctness.py` — Mamba forward pass vs. Hugging Face's `MambaForCausalLM`.
- `test_tp_correctness.py` — 2-process (`gloo`) tensor-parallel correctness, AllReduce count, and the naive-sharding failure case.
- `test_tp_lm_correctness.py` — same, for the full stacked model.
- `test_quantized_allreduce.py` — quantized AllReduce correctness and a Top-1/Top-5 accuracy table.
- `test_per_site_allreduce.py` — per-collective dtype lookup, collective counts, and the INT8 variants (real int8 on the wire, identical results on all ranks).

## Benchmarks

```bash
# SSM cache: decode latency, cached vs. no cache (CPU is fine)
.venv/bin/python benchmarks/bench_cache.py --prompt-lens 16 64 256

# Tensor-parallel throughput, profiler trace, quantized AllReduce
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend gloo --device cpu   # local smoke test
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend nccl --device cuda \
  --compare-single-gpu --profile --allreduce-dtype fp16   # real 2-GPU box (e.g. Kaggle "GPU T4 x2")

# Mixed-precision AllReduce accuracy (CPU) and speed (2 GPUs); see benchmarks/PHASE6_RESULTS.md
.venv/bin/python scripts/phase6_eval.py --n 100 --len 256
torchrun --standalone --nproc_per_node=2 scripts/phase6_gpu_bench.py

# CUDA graph decode speedup, verified correct, same-session A/B (needs a real GPU)
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend nccl --device cuda \
  --compare-single-gpu --verify-cuda-graph --compare-cuda-graph
```

## Reference

Sharding and communication design follows ["Scaling State-Space Models on Multiple GPUs with Tensor Parallelism"](https://arxiv.org/abs/2602.21144) (Dutt, Shah, Masarani, Gandhi — Stony Brook University).

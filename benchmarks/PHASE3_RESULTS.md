# Phase 3 results (Kaggle "GPU T4 x2", mamba-130m-hf, batch=1)

```
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py \
  --backend nccl --device cuda \
  --prompt-lens 16 64 256 --n-new-tokens 32 \
  --compare-single-gpu --profile
```

| prompt_len | TP tok/s (decode) | 1-GPU tok/s (decode) | TP vs 1-GPU |
|---|---|---|---|
| 16  | 37.5 | 48.6 | 0.77x (TP slower) |
| 64  | 35.1 | 47.8 | 0.73x (TP slower) |
| 256 | 36.3 | 47.3 | 0.77x (TP slower) |

`allreduces_per_layer0 == 76` in every row, matching the expected count exactly
(warmup: 1 prefill + 4 steps -> 2+8=10 AllReduces; timed: 1 prefill + 32 steps
-> 2+64=66; total 76) — confirms the sharding math ran correctly on real
NCCL/CUDA, not just in the CPU/gloo tests.

## Why TP is slower here

`torch.profiler` on rank 0: `record_param_comms` (the AllReduce) is **89% of
total self-CUDA time** (522ms / 584ms). At 130M params each mixer block's
local matmuls are tiny, so splitting them across 2 GPUs barely reduces
per-GPU compute, while every token still pays for 2 AllReduces x 24 layers
over Kaggle's PCIe interconnect. Communication cost dominates and TP loses
to single-GPU. This is the diminishing-returns regime the paper's own
Section V-C3 describes -- it just shows up immediately here because the
model and batch size are both too small for TP to have anything to
amortize its fixed collective-launch overhead against.

Also notable: rank 0 and rank 1 report very different self-CUDA time for the
same collectives (584ms vs 146ms) -- a profiler-attribution artifact of NCCL
wait time landing on whichever rank's timeline is sampled. Worth inspecting
`profiles/decode_rank{0,1}.json` in a trace viewer (ui.perfetto.dev) to see
the actual stall pattern between ranks.

## Next: find where TP actually wins

Two independent levers to test (run each separately):

```bash
# Lever 1: bigger model (more per-GPU compute per AllReduce)
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py \
  --backend nccl --device cuda --model-id state-spaces/mamba-1.4b-hf \
  --prompt-lens 64 256 --n-new-tokens 32 --compare-single-gpu

# Lever 2: bigger batch (amortize the fixed AllReduce launch cost)
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py \
  --backend nccl --device cuda \
  --prompt-lens 64 256 --n-new-tokens 32 --batch-size 32 --compare-single-gpu
```

Expectation: as batch size or model size grows, the TP/1-GPU ratio should
rise toward and past 1.0x -- that crossover point *is* the result worth
understanding, more so than either single number alone.

## Follow-up results (still Kaggle 2xT4)

### Lever 1: bigger model (mamba-1.4b-hf, batch=1 decode)

| | TP tok/s | 1-GPU tok/s | ratio |
|---|---|---|---|
| decode | ~18.3 | ~24.5 | 0.75x |

Same conclusion as 130m, ~10x more parameters later: decode is dominated by
48 blocking AllReduces per token (2/layer x 24 layers), a count that is
fixed by depth, not size. Going from 130m to 1.4b didn't change the ratio at
all (0.73-0.77x -> 0.75x). At these model sizes on a PCIe-connected T4 pair,
single-token decode looks structurally comm-bound; it would likely take a
much larger model (where compute-per-token finally rivals the collective
overhead) or NVLink (lower per-collective fixed cost) to flip this.

### Lever 2: prefill throughput (compute-bound regime), mamba-130m

```bash
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py \
  --backend nccl --device cuda \
  --prompt-lens 256 1024 --prefill-batch-sizes 1 8 32 \
  --prefill-throughput --compare-single-gpu
```

| prompt_len | batch | TP tok/s | 1-GPU tok/s | TP speedup |
|---|---|---|---|---|
| 256  | 1  | 361  | 395  | 0.91x |
| 256  | 8  | 2837 | 3006 | 0.94x |
| 256  | 32 | 7050 | 4150 | **1.70x** |
| 1024 | 1  | 371  | 396  | 0.94x |
| 1024 | 8  | 2814 | 2936 | 0.96x |
| 1024 | 32 | 6492 | 3626 | **1.79x** |

**This is the crossover.** Prefill has only 2 AllReduces total per layer for
the *entire* batched sequence (not per token), so the fixed collective
overhead barely matters here even at batch=1 -- TP is close to parity
(0.91-0.94x) instead of decode's 0.75x. At batch=32, TP pulls decisively
ahead (~1.7-1.8x): each rank's local matmuls are half the width, so its
FLOPs roughly halve, while the single T4's own throughput is clearly
saturating (going batch 8->32 is only a 1.4x gain for 4x more work: 3006 ->
4150 tok/s) -- the single GPU is compute-bound and out of headroom, while
each TP shard still has room to scale. This is the same "GPU utilization
saturates, TP still has headroom" effect the paper attributes its
diminishing-returns discussion to (Section V-C3 / Fig. 8), reproduced here
independently at small scale.

**Takeaway for this project:** TP's benefit is workload-shape-dependent, not
just model-size-dependent. It loses for low-batch, single-token decode
(latency-bound, fixed per-token collective count dominates, unaffected by
model size up to 1.4B here) and wins clearly for high-batch prefill
(compute-bound, collective count is fixed per *call* not per token, and per-GPU
FLOPs genuinely drop). Real serving systems handle this by using TP mainly
for prefill/large-batch scenarios and treating decode-time collectives as
the thing to minimize hardest -- which is exactly Phase 4's job.

### Where quantized AllReduce (Phase 4) should matter

At batch=32, prompt_len=1024, `out_proj`'s AllReduce payload is
`32 * 1024 * 768 * 4 bytes` ~= 100MB *per layer*, ~2.4GB total across 24
layers for one prefill call -- large enough that this AllReduce is plausibly
bandwidth-bound (not just latency-bound), which is exactly the regime where
FP32->FP16 quantization should show a real, measurable speedup. Decode's
per-token AllReduce payloads, by contrast, are tiny (`batch=1` x `d_model` or
smaller) and latency-bound -- quantizing those should do close to nothing,
which is itself worth confirming rather than assuming.

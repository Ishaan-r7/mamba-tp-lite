# Phase 5 results: CUDA graphs for decode

Kaggle "GPU T4 x2", mamba-130m, batch=1, `--allreduce-dtype fp32`. Command:

```bash
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py \
  --backend nccl --device cuda \
  --prompt-lens 64 256 1024 --n-new-tokens 32 \
  --compare-single-gpu --verify-cuda-graph --compare-cuda-graph
```

`--compare-cuda-graph` runs eager and graphed decode back-to-back in the
same process, at the same prompt length, from a seeded (identical) prompt --
same-session, apples-to-apples, learned the hard way in Phase 4 that
cross-session Kaggle comparisons aren't trustworthy.

## Two real bugs found before trusting any number

Both dense and TP+NCCL graph capture *succeeded* on the very first attempt
(`cuda_graph_status: captured`, no fallback) -- capturing collectives inside
a CUDA graph is a famously finicky area, so that alone was a pleasant
surprise. But `--verify-cuda-graph` (same prompt, two independent caches,
one stepped eagerly, one via captured replay, check every step's logits
match) immediately caught that the fast numbers were wrong:

1. **Warmup contamination.** `GraphedStepper.capture()`'s warmup loop called
   `model.step()` 3 times against the *real* cache before the actual capture
   call. `step()` isn't idempotent under a repeated fixed input -- it
   advances the SSM recurrent state every call regardless of the token
   value -- so warmup silently advanced the cache 3 real steps before
   capture ever recorded anything. Fixed by warming up on a disposable deep
   copy of the cache instead (`_clone_cache` in `mamba_lite/cuda_graph.py`).

2. **The actual root cause: cache updates via reassignment, not in-place
   writes.** `cache.ssm_state = (cache.ssm_state.float() * dA + dBx).to(dtype)`
   rebinds the Python attribute to a brand-new tensor rather than writing
   into the existing one. CUDA graph replay never re-runs Python -- it only
   replays the recorded kernel launches against the exact memory addresses
   seen during the ONE capture call. So the captured graph permanently read
   from "whatever address `cache.ssm_state` pointed to at capture time" and
   wrote to a different, freshly-allocated address on every replay --
   meaning **the graph never advanced the recurrence at all**; every
   `replay()` call silently recomputed "step 1 from the original prefilled
   state," forever. Fixed in both `mamba_lite/model.py` and
   `mamba_lite/tp_model.py`: cache updates now use `.copy_()` into the
   existing tensor instead of reassignment, so read and write addresses stay
   identical across calls.

Applying fix #1 alone made the mismatch *worse* (max abs logit diff went
from 33.4 to 182), which was the signal that #1 wasn't the real cause --
just a real bug that happened to partially mask a bigger one. After fix #2,
`verify_cuda_graph_correctness` reports `match=True, detail=0.0` -- an exact
bit-level match, not just "close" -- for both TP and dense.

Both fixes are value-preserving for eager execution (same computed values,
different tensor identity/timing of when it's produced), confirmed by the
full existing test suite (9/9) passing unchanged throughout.

## Verified speedup (same-session, same prompt, bit-exact correctness)

| prompt_len | TP eager | TP graphed | TP speedup | 1-GPU eager | 1-GPU graphed | 1-GPU speedup |
|---|---|---|---|---|---|---|
| 64   | 37.8 tok/s | 112–113 tok/s | **~2.98x** | 55.3 tok/s | 175.2 tok/s | **3.17x** |
| 256  | 36.3 tok/s | 132–134 tok/s | **~3.67x** | 55.3 tok/s | 137.0 tok/s | **2.48x** |
| 1024 | 37.5–37.8 tok/s | 122.5–122.6 tok/s | **~3.25x** | 52.2 tok/s | 145.3 tok/s | **2.78x** |

CUDA graphs give a consistent **~2.5-3.7x decode speedup**, for both TP and
single-GPU, across every prompt length tested -- directly confirming Phase
3's finding that decode is dominated by per-op launch overhead (its
profiler showed ~19% of decode CPU time was just `cudaLaunchKernel`), not
compute. Removing that overhead via graph capture recovers a large,
consistent multiple, with zero change to correctness.

## Open follow-up (not required to close this phase, but a natural next check)

Phase 4 found FP16-quantized AllReduce makes eager decode consistently
*worse* (-9% to -16%), attributed to two extra cast kernels per collective
adding pure launch overhead with nothing to offset it at decode's tiny
payload size. Once CUDA graphs remove launch overhead from the equation
entirely, does that regression disappear or even flip positive? Testable
with `--allreduce-dtype fp16 --cuda-graph` (note: NOT `--allreduce-dtype
int8`, which is explicitly rejected -- its scale-sync `.item()` call is a
GPU->CPU sync that can't be captured).

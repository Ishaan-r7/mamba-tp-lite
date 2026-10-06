# Phase 4 results: quantized AllReduce

> **Superseded for INT8.** The INT8 path measured in this document summed `int32` values, so it did not reduce bytes on the wire. The corrected int8 collectives and their results are in [`mixed_precision_results.md`](mixed_precision_results.md). The FP16 results here still stand.

Correctness/accuracy tests: `tests/test_quantized_allreduce.py` (gloo/CPU, real
mamba-130m-hf weights, 42-token real text prompt -- not random token ids, see
below for why that matters).

## Accuracy table (paper's Table I style)

| precision | Top-1 % | Top-5 unordered % | Top-5 ordered % |
|---|---|---|---|
| fp16 | 100.00% | 100.00% | 97.62% |
| int8 | 64.29%  | 66.19%  | 2.38%  |

FP16 (the paper's own choice, Section IV-D) is close to lossless here --
consistent with the paper's reported ~98-99% Top-1 for Mamba. INT8 (our own
extension) keeps a majority of top-1 predictions right but nearly destroys
the fine-grained ranking among close candidates (2.4% *ordered* top-5 match,
vs 66% *unordered* overlap -- the right tokens are often still in the top 5,
just reshuffled). This compounds over every AllReduce: mamba-130m has 24
layers x 2 AllReduces = 48 quantization events in a single forward pass, so
even a "small" per-call error compounds substantially by the final logits.

**Methodology note, and a real mistake caught along the way:** the first
version of this test fed random token ids as the prompt (`torch.randint(...)`)
instead of real tokenized text. That gave a much worse, and misleading,
int8 Top-1 of 43.8% -- random-id "sentences" don't look like real language to
the model, so activations land further out of the distribution the model
(and the int8 quantizer's dynamic range) actually expects. Switching to a
real tokenized paragraph raised int8's Top-1 to 64.3% with everything else
unchanged. Lesson: an accuracy eval is only as good as its input
distribution, even for a quick sanity check.

## What's still open (needs a real GPU to answer)

Phase 3 found that AllReduce cost is latency-bound at low batch (decode,
batch=1 prefill) and starts becoming bandwidth-relevant at high batch/long
prompt (batch=32, len=1024 prefill: `out_proj`'s payload is ~100MB/layer).
The prediction from that analysis: quantized AllReduce should give close to
no speedup on decode (nothing to save on tiny latency-bound messages) and a
real speedup on large-batch prefill (fewer bytes to move when bandwidth
actually matters). This needs measuring on Kaggle's 2xT4:

```bash
# Decode: expect ~no change from fp16 (latency-bound, tiny payload)
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py \
  --backend nccl --device cuda --allreduce-dtype fp16 \
  --prompt-lens 64 256 --n-new-tokens 32 --compare-single-gpu

# Prefill at the batch=32 crossover point: expect a real speedup from fp16
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py \
  --backend nccl --device cuda --allreduce-dtype fp16 \
  --prompt-lens 1024 --prefill-batch-sizes 32 --prefill-throughput

# int8: much lossier (see accuracy table above) -- only worth it if the
# bandwidth savings are large; also costs an EXTRA collective per call
# (scale-sync AllReduce(MAX) before the sum), which could erase gains
# entirely at small message sizes
torchrun --standalone --nproc_per_node=2 scripts/run_tp.py \
  --backend nccl --device cuda --allreduce-dtype int8 \
  --prompt-lens 1024 --prefill-batch-sizes 32 --prefill-throughput
```

Compare each `prefill_tokens_per_s` / `tokens_per_s_decode` against the
matching `--allreduce-dtype fp32` run from Phase 3 to get the speedup.

## Real GPU results (Kaggle 2xT4, mamba-130m, FP16 AllReduce)

**Methodology note first:** a naive cross-session comparison (fp32 numbers
from one Kaggle notebook session, fp16 from a separate later session)
initially looked dramatic in both directions -- decode looked ~28% slower
and the single-GPU *dense* baseline (which fp16-AllReduce can't possibly
affect) also swung by >20% between sessions. That's shared-infra variance
(thermal throttling / other tenants / clock state), not our code, and it's
large enough to swamp a real effect if you're not careful. All numbers below
are same-session, back-to-back runs on identical hardware state -- the only
way to trust a comparison here.

| workload | fp32 | fp16 | change |
|---|---|---|---|
| decode, len=64  | 32.5 tok/s | 28.5 tok/s | **-12%** |
| decode, len=256 | 33.3 tok/s | 30.1 tok/s | **-9%**  |
| decode, len=1024| 33.1 tok/s | 27.7 tok/s | **-16%** |
| prefill, batch=32, len=1024 | 2526 tok/s | 2675 tok/s | **+6%** |

**Decode gets consistently worse with fp16 AllReduce, across every prompt
length tested.** This is sharper than the Phase 3 prediction of "no
benefit" -- it's an active regression. Cause: `all_reduce_sum_fp16` adds two
extra elementwise cast kernels per call (fp32->fp16 before, fp16->fp32
after). Decode's payload is already tiny (latency-bound, nothing to save by
halving a few KB), so those two extra kernel launches are pure added
overhead on top of an already kernel-launch-dominated workload (Phase 3's
profiler: ~19% of decode's CPU time is just `cudaLaunchKernel`). **Lesson:
quantizing a latency-bound collective can cost more than it saves** -- a
real reason production systems don't blindly quantize every AllReduce.

**Prefill at the batch=32/long-prompt crossover point gets a modest but
real +6%.** Smaller than the ~1.7-1.8x envelope Phase 3's raw payload-size
math suggested, most likely because this session's overall GPU throughput
was running ~2.5x slower than the original Phase 3 session (2526-2675 tok/s
here vs. 6490 tok/s for the identical fp32 config, originally) -- with
everything slower, the AllReduce itself is a smaller share of total time,
so there's proportionally less for quantization to save. The direction is
still right and still consistent with the bandwidth-vs-latency framing:
fp16 helps exactly where Phase 3 said the collective becomes bandwidth-
relevant, and hurts exactly where Phase 3 said it's latency-bound.

**int8 was not benchmarked for speed on GPU** -- given the accuracy table
above (2.4% Top-5 ordered match after 24 layers), the accuracy cost is
severe enough that a speed number alone wouldn't make it a reasonable
choice for this model at this depth without a smarter scheme (per-channel
scales, error feedback, or quantizing only a subset of layers/collectives).

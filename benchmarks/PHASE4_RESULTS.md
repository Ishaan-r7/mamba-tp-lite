# Phase 4 results: quantized AllReduce

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

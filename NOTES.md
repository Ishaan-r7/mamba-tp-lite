# Project notes

Brief, per-phase: goal, what we did, what happened, anything unexpected, and
the fix/contribution. Kept updated as we go, not reconstructed at the end.

## Phase 1 — Mamba from scratch + SSM cache

- **Goal:** write Mamba's forward pass ourselves (not HF's `generate()`), with
  an explicit per-layer cache, and confirm the cache actually helps.
- **Action:** implemented `prefill()`/`step()` in `mamba_lite/model.py`;
  verified against HF's `MambaForCausalLM`; benchmarked cached vs.
  from-scratch-every-token decode.
- **Result:** exact match with HF (prefill logits, greedy tokens, cached
  decode reproducing full-prefill logits at every position). Cache keeps
  decode flat (~12ms/token) regardless of prompt length; no-cache decode
  grows with prompt length (38x-73x slower at len 256, CPU).
- **Unexpected:** `A_log`/`D` were declared as raw `nn.Parameter(torch.empty(...))`
  with no default init (unlike `nn.Linear`'s auto-init) — invisible until a
  freshly-constructed `MambaMixer` (not loaded from a checkpoint) was needed
  for Phase 2's tests, where it caused a real, non-deterministic test failure.
- **Fix:** added `_reset_ssm_params()` (same S4D-real init HF uses) so a
  fresh `MambaMixer` is deterministic without loading weights.

## Phase 2 — Tensor-parallel mixer (channel splitter + packed params)

- **Goal:** shard the mixer across ranks the way the paper describes (channel
  splitter + explicit packed-param handling), with exactly 2 AllReduces/block.
- **Action:** `TPMambaMixer` (correct sharding) + `NaiveTPMambaMixer`
  (deliberately splits the packed `in_proj` `[x|z]` output straight through,
  the Section IV-C pitfall) in `mamba_lite/tp_model.py`. 2-process `gloo`
  (CPU) test comparing both against a dense reference.
- **Result:** `TPMambaMixer` matches the dense reference exactly, with
  exactly 2 AllReduces counted per forward call. `NaiveTPMambaMixer`
  confirmed wrong, as predicted.
- **Unexpected:** none beyond the Phase 1 init bug (caught here, fixed there).
- **Contribution:** the `CommCounter` pattern (wrap `dist.all_reduce`, count
  calls) — a simple, reusable way to assert communication topology in tests
  instead of just trusting the code.

## Phase 3 — Full TP model, real 2xT4 benchmarks

- **Goal:** stack TP blocks into a full model with cache-aware decode, then
  measure it on real 2-GPU hardware (Kaggle "GPU T4 x2").
- **Action:** `TPMambaLM` with `from_pretrained_shard()` (slices each tensor
  into its shard directly from the checkpoint, never materializes a full
  dense model per rank); `scripts/run_tp.py` for `torchrun`-launched
  benchmarking + `torch.profiler`.
- **Result (mamba-130m, batch=1, decode):** TP is ~25-30% *slower* than
  single-GPU (~36 vs ~48 tok/s). Confirmed correct (exact AllReduce count),
  not a bug. Profiler: AllReduce is 89% of rank 0's GPU time.
  **Same 130m result held at mamba-1.4b** (~0.75x ratio, unchanged) and
  **didn't improve with batch=32 decode either** — decode pays 48 fixed
  AllReduces per token (2 x n_layer) regardless of model size or batch,
  so it's structurally comm-bound on PCIe in this range.
  **Prefill throughput flips this**: at batch=32, TP hits ~1.7-1.8x speedup
  over single-GPU, because prefill only pays 2 AllReduces per layer for the
  *whole* batched sequence (not per token), and the single T4 saturates its
  own compute before each (half-width) TP shard does.
- **Unexpected:**
  1. `torchrun` hung indefinitely in the local dev sandbox during rendezvous
     (IPv6 reverse-DNS lookup failures, reproduced with a bare "hello world"
     script) — a sandbox networking quirk, not a code bug. Correctness was
     already proven via the `gloo`+`mp.spawn` tests (identical forward code,
     different process-launch mechanism), so this didn't block progress —
     just moved all real-hardware runs to Kaggle.
  2. `mamba-1.4b-hf` ships as a sharded checkpoint (multiple `.safetensors`
     files + an index), not one `model.safetensors` — both loaders crashed
     with `FileNotFoundError` on the first 1.4b run.
  3. Batching didn't help decode at all (expected some improvement) — turned
     out the AllReduce *count* (not payload size) is the bottleneck for
     decode, and batching only grows payload, not count.
- **Fix:** `load_hf_state_dict()` in `mamba_lite/model.py` handles both
  single-file and sharded (`*.index.json`) checkpoints, shared by both
  `MambaLM.from_pretrained` and `TPMambaLM.from_pretrained_shard`.
- **Contribution:** the compute-bound-vs-latency-bound framing itself — full
  writeup in `benchmarks/PHASE3_RESULTS.md`. This wasn't just "reproduce the
  paper's number," it's an independently-found explanation of *when* TP
  helps, backed by a controlled sweep (model size x batch size x workload
  phase) rather than a single headline number.

## Phase 4 — Quantized AllReduce

- **Goal:** reduce AllReduce cost by quantizing the collective's payload:
  FP32->FP16 (the paper's own choice) plus an INT8 version as our own
  extension, and check what it costs in accuracy.
- **Action:** `all_reduce_sum_fp16`/`all_reduce_sum_int8` in
  `mamba_lite/tp_utils.py`, threaded through `TPMambaMixer`/`TPMambaBlock`/
  `TPMambaLM` as a selectable `allreduce_dtype`. Correctness tests (loose
  tolerance, not exact -- quantization is lossy by design) plus a paper-
  Table-I-style accuracy table (Top-1 / Top-5 unordered / Top-5 ordered)
  on real mamba-130m-hf weights.
- **Result:** FP16 is close to lossless (100% Top-1, 97.6% Top-5 ordered) --
  matches the paper. INT8 keeps 64% Top-1 but nearly destroys fine-grained
  ranking (2.4% Top-5 ordered) once compounded over 24 layers x 2
  AllReduces = 48 quantization events per forward pass. Full table and the
  planned GPU speedup test in `benchmarks/PHASE4_RESULTS.md`.
- **Unexpected:** the first version of the accuracy test used random token
  ids as the prompt instead of real text. That alone dropped INT8's Top-1
  from 64.3% to 43.8% -- out-of-distribution activations widen the tensor's
  dynamic range and inflate quantization error, which the model never sees
  in normal use. INT8 AllReduce also structurally costs *two* collectives
  per call (an AllReduce(MAX) to sync a shared quantization scale, then the
  AllReduce(SUM) itself) -- naive INT8 isn't just "4x fewer bytes," it trades
  bandwidth for an extra latency hop.
- **Fix:** rewrote the accuracy test to tokenize a real paragraph instead of
  `torch.randint`, and made the test's own pass/fail bar honest: a tight
  floor for FP16 (should be near-lossless) and diagnostic-only reporting for
  INT8 (how lossy it ends up is the finding, not an assumption to encode as
  a pass/fail threshold).

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
  writeup in `benchmarks/tensor_parallel_results.md`. This wasn't just "reproduce the
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
  planned GPU speedup test in `benchmarks/quantized_allreduce_initial_results.md`.
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

- **GPU follow-up (Kaggle 2xT4, same-session controlled runs -- see
  `benchmarks/quantized_allreduce_initial_results.md`):** FP16 AllReduce makes decode
  consistently *worse* (-9% to -16% across prompt lengths 64/256/1024) --
  sharper than predicted; the two extra cast kernels per call are pure
  overhead when the payload is already tiny and latency-bound. FP16
  AllReduce gives a modest but real +6% on the batch=32/len=1024 prefill
  case, the bandwidth-relevant regime Phase 3 identified -- direction
  confirmed, magnitude smaller than the raw payload-size math suggested.
- **Unexpected (methodology):** an initial cross-session comparison (fp32
  numbers from one Kaggle notebook session, fp16 from a later one) showed
  a much bigger swing in both directions -- and so did the single-GPU dense
  baseline, which quantization cannot possibly touch. That is shared-infra
  variance (thermal/tenant/clock state), not a code effect, and it can be
  as large as the thing being measured. Fixed by re-running fp32 and fp16
  back-to-back in the same notebook session before trusting any comparison.

## Phase 5 — CUDA graphs for decode

- **Goal:** decode was found (Phase 3) to be dominated by per-op kernel-launch
  overhead, not compute -- CUDA graphs capture the whole decode step's
  kernel launches once and replay them as a single launch, testing that
  hypothesis directly. Unlike every earlier phase, this needs a real GPU:
  no CPU/gloo fallback exists for CUDA graphs.
- **Action:** `mamba_lite/cuda_graph.py`'s `GraphedStepper` (warmup on a side
  stream, capture, replay -- both for the dense model and the TP+NCCL path),
  wired into `scripts/run_tp.py` via `--cuda-graph`, plus
  `--verify-cuda-graph` (numerical correctness check) and
  `--compare-cuda-graph` (same-session eager-vs-graphed A/B).
- **Result:** verified correct (`match=True, detail=0.0` -- exact, not just
  close) and a consistent **~2.5-3.7x decode speedup**, both TP and
  single-GPU, across all three prompt lengths tested. Confirms Phase 3's
  launch-overhead hypothesis directly. Full numbers in
  `benchmarks/cuda_graph_results.md`.
- **Unexpected -- two real bugs, caught specifically because a correctness
  check was built before trusting any speed number:**
  1. Warmup calls ran against the *real* cache before capture, and `step()`
     isn't idempotent under a repeated input -- it silently advanced the
     cache 3 real steps before anything was even captured.
  2. The actual root cause: `step()` updated cache state via Python
     reassignment (`cache.ssm_state = new_tensor`) instead of writing into
     the existing tensor in place. CUDA graph replay never re-runs Python --
     it only replays recorded kernel launches against the exact memory
     addresses seen during the one capture call -- so the captured graph
     permanently read from the pre-capture address and wrote to a different
     one every time. **The graph never actually advanced the recurrence at
     all**; every replay silently recomputed "step 1 from the original
     state," forever. This would have shipped a model that produces wrong
     text while reporting an exciting (and honestly still directionally
     correct, since kernel-launch cost doesn't depend on tensor values)
     speedup number.
  3. Fixing bug #1 alone made the measured mismatch *larger* (33.4 -> 182.2
     max abs logit diff), which was the tell that #1 wasn't the real cause
     -- a genuine but secondary bug partially masking a bigger one.
- **Fix:** `_clone_cache()` for warmup (bug #1); `.copy_()` instead of
  reassignment for cache updates in both `mamba_lite/model.py` and
  `mamba_lite/tp_model.py` (bug #2, the real fix). Both changes are
  value-preserving for eager execution -- confirmed by the full existing
  test suite (9/9) passing unchanged after each fix.
- **Contribution:** this is the clearest example in the whole project of why
  a fast number and a correct number are different claims that both need
  checking, not just one. The verification harness (`verify_cuda_graph_correctness`)
  was built *before* trusting the first "captured, 2-3x faster" result, and
  it caught a bug that a napkin-math sanity check ("the number looks
  plausible") would never have surfaced.

## Phase 6 — Mixed-precision AllReduce

- **Goal:** find which of the 48 collectives (2 per layer x 24 layers) tolerate
  INT8, put only those in INT8, and see whether it makes inference faster.
- **Action:** per-site dtype hook (`site_dtype`, keyed by layer + x_proj/out_proj);
  new INT8 variants (`int8_sum`, `int8_gather`, `int8_gather_tok`); eval harness
  (`mamba_lite/quant_eval.py`: KL / Top-1 / Top-5 vs fp32 on Simple English
  Wikipedia); single-site sweeps, Pareto curve and greedy search; one
  back-to-back 2xT4 benchmark (`scripts/quant_gpu_bench.py`).
- **Result:** accuracy yes, speed no. 43/48 collectives in INT8 (5 out_proj kept
  fp16) gives KL 0.0005, Top-1 98.8%; layer 23 out_proj alone is ~80% of the
  all-INT8 damage. On 2xT4 nothing beats fp32 by more than FP16's ~+5% prefill;
  INT8 variants are ~30% slower on decode. Full numbers: `benchmarks/mixed_precision_results.md`.
- **Unexpected:**
  1. Phase 4's INT8 sent int32 through the SUM collective, so it saved no
     bandwidth at all. Caught while planning the speed test, before measuring.
  2. INT8's bad accuracy was the scale, not the 8 bits: one scale per tensor
     gave KL 0.65; one scale per token gave 0.013.
  3. Phase 4's accuracy numbers (42 tokens) reproduced almost exactly on 100
     passages x 256 tokens, so the small test was fine.
  4. Prefill is only ~5-9% communication at this size, which caps any gain.
     The prediction from Phase 3 ("quantization helps where bandwidth-bound")
     held directionally but the effect is small for a 130m model.
  5. Graphed fp32 decode was 5.7x faster than eager in this session vs 2.5-3.7x
     in Phase 5: the graphed side was faster; likely host variance.
- **Fix:** int8 on the wire via all-gather of packed int8 + per-row scales
  (also removes the scale-sync collective and the `.item()` sync, so INT8
  becomes CUDA-graph capturable; verified bit-exact vs eager on NCCL).
- **Contribution:** the per-site sensitivity map and the honest negative speed
  result; static (calibrated) scales were deliberately not built since the
  decode cost is kernel count, not just the scale step (untested judgment).

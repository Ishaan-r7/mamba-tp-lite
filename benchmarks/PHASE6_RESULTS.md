# Phase 6: mixed-precision AllReduce

Question: the 48 AllReduces per forward pass (2 per layer x 24 layers) are not
equally sensitive to quantization. Can the tolerant ones go to INT8 and the
fragile ones stay in higher precision, and does that make the model faster?

Short answer: **accuracy yes, speed no (on this model and hardware).** 43 of 48
collectives run in INT8 at KL 0.0005 vs fp32, but on 2x T4 over PCIe the
quantized variants give no measurable prefill gain and slower decode.

Raw data: `benchmarks/phase6/*.json`. Charts: `assets/phase6_*.png`.
Everything below was measured with `mamba-130m-hf`.

## 1. A bug in the Phase 4 INT8 path

The old INT8 AllReduce sent `int32` through the SUM collective, so it moved the
same bytes as fp32 and saved no bandwidth (its docstring claimed otherwise).
Replaced with INT8 that is really int8 on the wire. Measured wire traffic per
prefill forward (rank 0, batch 32 x 1024):

| config | MB sent | vs fp32 |
|---|---|---|
| fp32 | 2668 | 1.00x |
| fp16 | 1334 | 0.50x |
| INT8 (all 48) | 673 | 0.25x |
| mixed, 1 site fp16 | 698 | 0.26x |
| mixed, 5 sites fp16 | 798 | 0.30x |

Variants compared (`mamba_lite/tp_utils.py`), 100 Simple English Wikipedia
passages x 256 tokens, all-48-sites quantized, reference = fp32 TP logits:

| variant | wire | KL (mean) | Top-1 | Top-5 ordered |
|---|---|---|---|---|
| fp16 | fp16 | 0.00001 | 99.9% | 97.5% |
| `int8_legacy` (Phase 4) | int32 | 0.81 | 64.3% | 2.6% |
| `int8_sum` (+-63, one global scale) | int8 | 3.82 | 37.0% | 0.3% |
| `int8_gather` (one scale per tensor, per rank) | int8 | 0.65 | 67.5% | 3.6% |
| **`int8_gather_tok` (one scale per token)** | int8 | **0.013** | **92.9%** | **36.8%** |

- The Phase 4 numbers reproduce on the larger data (FP16 99.9/97.5 vs 100/97.6;
  legacy INT8 64.3% Top-1 both times), so the original 42-token measurement was
  representative.
- INT8's error was mostly the scale granularity, not the 8 bits: giving each
  token its own scale cuts KL about 50x. Outlier tokens/channels no longer set
  the quantization grid for the whole tensor.
- `int8_sum` is worse than the legacy path because +-63 at 2 ranks halves the
  resolution (about 4x the KL) on top of the global scale.
- `int8_gather_tok` (now the `int8` default) quantizes per token, all-gathers
  the packed int8 + scale bytes, and dequantizes and sums locally. One
  collective per site, no scale-sync AllReduce, no GPU->CPU sync, so unlike
  Phase 4's INT8 it can be captured in a CUDA graph.

## 2. Which collectives are fragile (INT8 at one site, rest fp32)

64 passages x 256 tokens. Chart: `assets/phase6_site_sensitivity.png`.

- **One site dominates:** layer 23 `out_proj` alone gives KL 0.0101, about 80% of
  the all-INT8 damage (0.0127). Next worst: layer 22/21/0/20 `out_proj`
  (0.0007 / 0.0007 / 0.0006 / 0.0003).
- `out_proj` (AllReduce #2) is the fragile site type: mean single-site KL 0.0005
  vs 0.000007 for `x_proj` (about 75x lower). The median site is 8e-6.
- FP16 at any single site is harmless (worst: 1e-5).

## 3. Mixed precision

Chart: `assets/phase6_pareto.png`. Each combination was measured directly
(single-site errors do not add up exactly).

- Adding the least-sensitive sites to INT8 in order: KL stays below 4e-4 up to
  40 INT8 sites and only climbs steeply for the last few.
- Greedy (start all-INT8, repeatedly move the most damaging site to fp16):

| INT8 sites | fp16 sites (cumulative) | KL | Top-1 | Top-5 ordered |
|---|---|---|---|---|
| 48 | none | 0.0127 | 92.9% | 36.8% |
| 47 | L23 out_proj | 0.0027 | 96.9% | 60.8% |
| 46 | + L22 out_proj | 0.0019 | 97.5% | 65.7% |
| 45 | + L21 out_proj | 0.0013 | 98.1% | 72.1% |
| 44 | + L0 out_proj | 0.0007 | 98.5% | 77.0% |
| **43** | **+ L20 out_proj** | **0.0005** | **98.8%** | **80.2%** |
| 42 | + L23 x_proj | 0.0004 | 98.9% | 81.5% |

Two configs go to hardware: `mixed_1fp16` (47 INT8) and `mixed_5fp16` (43 INT8).

## 4. Speed on 2x Tesla T4 (Kaggle, same session)

fp32, fp16, INT8 and both mixed configs run back-to-back in one process, 3
interleaved rounds, median reported. Chart: `assets/phase6_gpu_speed.png`.

Before timing, each config was checked on NCCL: finite logits, KL vs fp32, and
CUDA-graph output vs eager.

| config | KL vs fp32 (42 tokens) | Top-1 | graph == eager |
|---|---|---|---|
| fp32 | 0 | 100% | exact |
| fp16 | 0.000007 | 100% | exact |
| INT8 | 0.0151 | 92.1% | exact |
| mixed_1fp16 | 0.0033 | 98.4% | exact |
| mixed_5fp16 | 0.0007 | 100% | exact |

INT8's accuracy on real GPUs matches the CPU harness (0.015 vs 0.013), and
INT8 all-gather captures and replays correctly under NCCL.

| config | prefill tok/s (batch 32 x 1024) | decode eager ms/tok | decode graphed ms/tok |
|---|---|---|---|
| fp32 | 6553 (1.000x) | 25.07 (1.000x) | 4.38 (1.000x) |
| fp16 | 6875 (1.049x) | 26.83 (0.934x) | 4.59 (0.954x) |
| INT8 | 6698 (1.022x) | 44.22 (0.567x) | 6.45 (0.678x) |
| mixed_1fp16 | 6665 (1.017x) | 43.26 (0.579x) | 6.41 (0.682x) |
| mixed_5fp16 | 6661 (1.017x) | 41.95 (0.597x) | 6.27 (0.698x) |

Reading it:

- **Step 0 (does graph capture remove FP16's decode regression?)** Partly. FP16
  decode is 0.934x eager and 0.954x graphed, so the gap shrinks but stays. The
  remaining ~0.2 ms per token over 48 sites (about 4 us per site) is the
  actual runtime of the two cast kernels, not launch overhead. A fused
  cast+collective kernel could recover at most that ~4%, so it is not worth
  building for decode.
- **Prefill:** FP16 is faster than fp32 in all three rounds (+3.2%, +7.0%,
  +4.9%). INT8 and mixed are not distinguishable from fp32: per-round
  differences range from -4.2% to +4.7%, inside the ~7% round-to-round spread
  fp32 itself shows (6395-6884 tok/s).
- **Decode:** INT8 variants are 40-43% slower eager and 30-32% slower graphed. The
  graphed gap is about 2.1 ms per token over 48 sites (about 43 us per site):
  the quantize / pack / all-gather / dequantize path is a dozen small kernels
  per collective, against one cast pair for FP16. Decode payloads are tiny, so
  there are no bytes worth saving.
- **Why INT8 does not win prefill despite 4x fewer bytes (not profiled, a
  likely explanation):** a prefill forward takes about 5 s for 32,768 tokens
  and sends 2.7 GB. At any plausible PCIe throughput (roughly 6-12 GB/s) that
  is only ~5-9% of the time, so even free communication would cap the gain
  near that. INT8's extra elementwise passes over 100 MB activations per site
  eat most of what the smaller payload saves. At this model size prefill is
  dominated by compute and the sequential scan, not by communication.
- fp32 graphed decode is 5.7x faster than eager in this session (25.07 vs
  4.38 ms), above Phase 5's 2.5-3.7x. The eager baseline is the same; the
  graphed side was faster here (4.4 ms vs ~7.5-8.2 ms in Phase 5). Likely a
  different Kaggle host/clock state (the cross-session variance seen in
  Phase 4); not investigated further. All ratios within this table come from
  one session and are comparable with each other.

## 5. Conclusions

1. INT8 collectives are accurate enough when each token gets its own scale: 43 of
   48 collectives in INT8 with 5 `out_proj` sites kept in fp16 gives KL 0.0005,
   Top-1 98.8%, 70% less communication than fp32.
2. On mamba-130m / 2x T4 / PCIe none of this makes inference faster. FP16 is the
   only practical choice: about +5% prefill, about -5% to -7% decode. INT8 only
   makes sense where communication is a much larger share of runtime
   (larger models, slower interconnects), which this setup is not.
3. A bug found along the way (INT8 carrying int32) was fixed; INT8 now really
   sends a quarter of fp32's bytes (673 vs 2668 MB measured).

## Not done / limits

- **Static (calibrated) scales** were not built. The measured INT8 decode cost
  comes from the number of small kernels per collective rather than from the
  scale reduction alone, so removing the scale step is unlikely to change the
  outcome; this is an untested judgment, not a measurement.
- Speed was measured on mamba-130m only, one Kaggle session, 3 rounds; the
  accuracy sweeps are also 130m only. Prefill differences under ~5% are within
  noise.
- Sensitivity was measured with greedy search and a Pareto ordering, not an
  exhaustive search over all 2^48 mixes.

## Reproduce

```bash
# accuracy (CPU)
.venv/bin/python scripts/phase6_eval.py --n 100 --len 256
.venv/bin/python scripts/phase6_sweep.py --dtype int8   # and --dtype fp16
.venv/bin/python scripts/phase6_pareto.py
# speed (2 GPUs)
torchrun --standalone --nproc_per_node=2 scripts/phase6_gpu_bench.py
```

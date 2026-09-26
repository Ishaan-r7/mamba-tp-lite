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

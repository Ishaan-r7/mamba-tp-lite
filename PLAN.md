# Mamba-TP Lite

A learning project: build tensor-parallel Mamba inference from scratch, following the design in
"Scaling State-Space Models on Multiple GPUs with Tensor Parallelism" (Dutt et al., arXiv 2602.21144).

Goal is understanding, not matching the paper's numbers. Every benchmark here compares *my own*
implementations against each other (e.g. cache vs no cache, fp32 vs fp16 all-reduce).

## Hardware (all free / already have)

- **Colab L4 (24 GB)** — main dev box: single-GPU work, profiling.
- **Kaggle "GPU T4 x2"** — free real 2-GPU box for NCCL tensor parallelism. Check weekly quota.
- **gloo backend on CPU** — 2-process TP correctness tests anywhere, no GPU needed.

Models: `state-spaces/mamba-130m-hf` for correctness, `mamba-1.4b-hf` when you want TP to actually pay off.
Start in pure PyTorch; `mamba-ssm` fused kernels are optional (slow to install on Colab).

## Phase 1 — Mamba from scratch + SSM cache

Build: a minimal Mamba forward in PyTorch that loads HF weights (in_proj, conv1d, x_proj, dt_proj,
A_log, D, out_proj, RMSNorm). Two paths:
- `prefill(prompt)` — full selective scan, returns logits + cache (per layer: SSM state
  `[d_inner, d_state]` and last `d_conv-1` conv inputs).
- `step(token, cache)` — one-token recurrent update.

Check: greedy tokens identical to HF `generate()`; logits `allclose`.
Measure: time per output token, cached vs re-running the full prefix every step, for 16/64/256 outputs.

Learn: prefill (compute-bound) vs decode (memory-bound), why SSM state is O(1) vs a KV cache's O(L),
`torch.cuda.Event` timing, warmup, `torch.cuda.synchronize`.

## Phase 2 — Tensor-parallel mixer (channel splitter + packed params)

Build: shard each Mamba block across `world_size` ranks using `torch.distributed`:

| Op | Sharding | Comm |
|---|---|---|
| `in_proj` -> `[x \| z]` | column-parallel, split x and z **separately** | none |
| depthwise conv1d | by channel | none |
| `x_proj` -> `[dt \| B \| C]` | row-parallel | **AllReduce #1** |
| `dt_proj`, `A_log`, `D`, scan, gate | by channel | none |
| `out_proj` | row-parallel | **AllReduce #2** |

Also build the *naive* version (split the packed `in_proj` output contiguously) and watch it break.
Cache is sharded by channel too: each rank only holds its channels' state.

Check: TP output `allclose` to Phase 1 output. Wrap `dist.all_reduce` in a counter, assert 2 per block.
Run with `gloo` (2 CPU procs) first, then `nccl` on Kaggle 2xT4 via `torchrun --nproc_per_node=2`.

Learn: Megatron column/row-parallel pattern, why depthwise ops shard for free, process groups,
`torchrun`, NCCL env vars (`NCCL_DEBUG=INFO`), what a collective costs.

## Phase 3 — Profile the real 2-GPU run

Measure on Kaggle 2xT4: tokens/sec for 1 GPU vs 2-GPU TP, max prompt length before OOM for each,
and time spent in AllReduce vs compute.

Tools: `torch.profiler` (export Chrome trace), optionally Nsight Systems (`nsys profile`).

Learn: reading a GPU timeline, compute/comm overlap (or lack of it), PCIe vs NVLink,
latency-bound vs bandwidth-bound collectives (AllReduce #1 is ~80 cols wide, #2 is `d_model` wide).

## Phase 4 — Quantized AllReduce

Build: cast the tensor to lower precision before `all_reduce`, cast back after.
- fp32 -> fp16 (paper's version; run the model in fp32 for this).
- fp16 -> int8 with a per-token scale (own extension).
- Try quantizing only AllReduce #2 vs both.

Measure: speedup vs unquantized TP, and accuracy vs unquantized: Top-1 agreement, Top-5 overlap.

Learn: bytes-on-the-wire vs latency, the alpha-beta cost model, why tiny collectives don't benefit.

## Stretch ideas

- CUDA Graphs for the decode step (kills kernel-launch overhead at batch 1).
- Write the selective-scan decode step as a Triton kernel.
- Swap in `mamba-ssm` fused kernels and see what changes.
- Batched serving loop with requests of different lengths.

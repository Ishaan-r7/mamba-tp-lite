"""
Phase 5: CUDA graph capture for the decode step.

Decode is dominated by per-op launch overhead: every token re-launches
dozens of small kernels (matmuls, convs, elementwise ops), and for the TP
model, 2 x n_layer separate NCCL AllReduce calls on top of that (see Phase
3's profiler: ~19% of decode's CPU time was just `cudaLaunchKernel`). A CUDA
graph records that whole sequence of launches ONCE and replays it as a
single launch on every subsequent token -- it doesn't reduce compute or
communication volume, only the fixed launch overhead around it.

Works for both MambaLM and TPMambaLM, since both share the same
`step(ids, cache) -> logits` signature. Follows PyTorch's standard capture
recipe (warmup on a side stream, then one capture call, then replay):
https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs

Requires a real CUDA device -- there is no CPU/gloo fallback for this,
unlike every earlier phase. Also requires the model's step() to contain no
GPU->CPU syncs (no `.item()`, no data-dependent Python control flow) in the
captured region -- our int8 quantized AllReduce violates this (its scale
sync calls `.item()`), so it is explicitly rejected below rather than
silently miscapturing.
"""
from __future__ import annotations

import torch


class CUDAGraphUnsupported(RuntimeError):
    """Raised when graph capture can't proceed -- caller should catch this
    and fall back to eager execution rather than crash the whole benchmark."""


def _clone_cache(cache):
    """Deep-copies a MambaCache (used by both MambaLM and TPMambaLM) so
    warmup can exercise the model's step() -- which mutates cache state on
    every call, since the recurrence advances regardless of what token value
    is fed in -- without advancing the REAL cache the caller intends to
    capture from. Warming up on the real cache silently shifts capture to
    "step n_warmup+1 onward" instead of "step 1 onward", which any code
    comparing graphed vs eager output from the same starting cache would
    (correctly) flag as a mismatch -- this is exactly the bug that
    `verify_cuda_graph_correctness` caught the first time this shipped."""
    from .model import LayerCache, MambaCache

    return MambaCache(
        [LayerCache(conv_state=l.conv_state.clone(), ssm_state=l.ssm_state.clone()) for l in cache.layers]
    )


class GraphedStepper:
    """Captures `model.step(static_ids, cache)` once, then replays it.

    Usage:
        stepper = GraphedStepper(model, cache, batch_size=1, device="cuda")
        stepper.capture()
        for _ in range(n_new_tokens):
            logits = stepper.replay(next_id)   # next_id: [batch, 1] long tensor
    """

    def __init__(self, model, cache, batch_size: int, device: str, n_warmup: int = 3):
        if device != "cuda" or not torch.cuda.is_available():
            raise CUDAGraphUnsupported(
                "CUDA graphs require a real CUDA device; there is no CPU/gloo fallback."
            )
        allreduce_dtype = getattr(model, "allreduce_dtype", "fp32")
        if allreduce_dtype == "int8":
            raise CUDAGraphUnsupported(
                "int8 AllReduce calls .item() to sync its quantization scale (a GPU->CPU "
                "sync), which cannot be captured in a CUDA graph. Use --allreduce-dtype "
                "fp32 or fp16 with --cuda-graph."
            )

        self.model = model
        self.cache = cache
        self.device = device
        self.static_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.static_logits = None
        self.graph = None
        self.n_warmup = n_warmup

    def capture(self):
        # 1. Warmup on a side stream, synced with the current stream both
        #    before and after -- this is the part of the recipe that lets
        #    the caching allocator settle on stable addresses before we
        #    record anything. Skipping this is the most common way to get
        #    silently wrong results from a "successful" capture.
        #
        #    Warmup runs against a DISPOSABLE CLONE of self.cache, not
        #    self.cache itself: step() advances the recurrent state on every
        #    call regardless of the (fixed, dummy) input token, so warming
        #    up on the real cache would silently advance it n_warmup steps
        #    before capture ever records anything -- capture would then
        #    represent "step n_warmup+1 onward", not "step 1 onward" from
        #    the cache the caller handed us. The clone has identical shapes/
        #    dtypes/device, so it exercises the same allocator size classes
        #    just as effectively.
        warmup_cache = _clone_cache(self.cache)
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            with torch.no_grad():
                for _ in range(self.n_warmup):
                    self.model.step(self.static_ids, warmup_cache)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        # 2. Capture. Whatever tensor objects get created for cache state
        #    and the output logits during THIS call become the fixed
        #    addresses that every future replay() reads/writes -- there is
        #    no need to manually pre-allocate or .copy_() into cache
        #    buffers; PyTorch's graph pool allocator handles that as long as
        #    we don't touch self.cache from anywhere else afterward.
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph):
                with torch.no_grad():
                    self.static_logits = self.model.step(self.static_ids, self.cache)
        except RuntimeError as e:
            raise CUDAGraphUnsupported(
                f"CUDA graph capture failed: {e}\n"
                "This is often a version-specific NCCL/CUDA graph interaction for the TP "
                "path (capturing collectives inside a graph is genuinely finicky). Try "
                "--allreduce-dtype fp32 (no cast kernels to capture), or run without "
                "--cuda-graph to fall back to eager decode."
            ) from e

    def replay(self, next_id: torch.Tensor) -> torch.Tensor:
        self.static_ids.copy_(next_id)
        self.graph.replay()
        return self.static_logits

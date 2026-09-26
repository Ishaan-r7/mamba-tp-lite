"""Small helpers shared by the tensor-parallel mixer: a counted AllReduce
(so tests can assert exactly how many collectives run per block) and the
channel-sharding math used to split each weight tensor across ranks.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


class CommCounter:
    """Counts collectives so a test can assert 'exactly 2 AllReduces per block'."""

    def __init__(self):
        self.count = 0
        self.bytes = 0

    def reset(self):
        self.count = 0
        self.bytes = 0


def all_reduce_sum(tensor: torch.Tensor, group=None, counter: CommCounter | None = None) -> torch.Tensor:
    if counter is not None:
        counter.count += 1
        counter.bytes += tensor.numel() * tensor.element_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def all_reduce_sum_fp16(tensor: torch.Tensor, group=None, counter: CommCounter | None = None) -> torch.Tensor:
    """Paper's Section IV-D: cast FP32 -> FP16 for the collective, reduce,
    cast back. Halves the bytes on the wire, at FP16's precision cost.
    No calibration/scale needed -- FP16 already covers activation-sized
    values in this model without special-casing."""
    orig_dtype = tensor.dtype
    low = tensor.to(torch.float16).contiguous()
    if counter is not None:
        counter.count += 1
        counter.bytes += low.numel() * low.element_size()
    dist.all_reduce(low, op=dist.ReduceOp.SUM, group=group)
    return low.to(orig_dtype)


def all_reduce_sum_int8(tensor: torch.Tensor, group=None, counter: CommCounter | None = None) -> torch.Tensor:
    """Our own extension beyond the paper: symmetric per-tensor INT8
    quantization for the collective. Needs a scale shared by every rank
    (otherwise summing differently-scaled integers is meaningless), so this
    is TWO collectives, not one: an AllReduce(MAX) over |tensor| to agree on
    a scale, then an AllReduce(SUM) over the quantized int32 values. That
    extra round-trip is a real cost worth knowing about -- naive INT8
    AllReduce is not simply 4x fewer bytes than FP32 for free; it trades
    bandwidth for an extra latency hop, which can lose at small message
    sizes (e.g. single-token decode) even though it wins at large ones."""
    orig_dtype = tensor.dtype
    tensor = tensor.contiguous()

    local_max = tensor.abs().max().clamp(min=1e-8).reshape(1)
    if counter is not None:
        counter.count += 1
        counter.bytes += local_max.numel() * local_max.element_size()
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)  # scale-sync collective
    scale = local_max.item() / 127.0

    q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    q32 = q.to(torch.int32)
    if counter is not None:
        counter.count += 1
        counter.bytes += q32.numel() * q32.element_size()
    dist.all_reduce(q32, op=dist.ReduceOp.SUM, group=group)  # sum collective

    return (q32.to(torch.float32) * scale).to(orig_dtype)


ALLREDUCE_FNS = {
    "fp32": all_reduce_sum,
    "fp16": all_reduce_sum_fp16,
    "int8": all_reduce_sum_int8,
}


def channel_slice(rank: int, world_size: int, total: int) -> slice:
    assert total % world_size == 0, f"{total} not divisible by world_size={world_size}"
    shard = total // world_size
    return slice(rank * shard, (rank + 1) * shard)

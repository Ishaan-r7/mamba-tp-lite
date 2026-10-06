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


def all_reduce_sum_int8_legacy(tensor: torch.Tensor, group=None, counter: CommCounter | None = None) -> torch.Tensor:
    """Phase 4 baseline, kept to reproduce the old accuracy numbers. NOTE: the
    sum collective carries int32, so this moves as many bytes as fp32 and
    saves no bandwidth; see all_reduce_sum_int8_sum / _gather for real int8
    on the wire. Symmetric per-tensor INT8
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


def all_reduce_sum_int8_sum(tensor: torch.Tensor, group=None, counter: CommCounter | None = None) -> torch.Tensor:
    """INT8 on the wire via AllReduce(SUM). A scale shared by all ranks is
    agreed with a tiny AllReduce(MAX); values are quantized to
    +-(127 // world_size) so the int8 sum across ranks cannot overflow. Two
    collectives per call, but no GPU->CPU sync (the scale stays a tensor)."""
    orig_dtype = tensor.dtype
    ws = dist.get_world_size(group)
    qmax = 127 // ws
    t = tensor.contiguous().float()

    scale = t.abs().max().clamp(min=1e-8).reshape(1)
    if counter is not None:
        counter.count += 1
        counter.bytes += scale.numel() * scale.element_size()
    dist.all_reduce(scale, op=dist.ReduceOp.MAX, group=group)
    scale = scale / qmax

    q = (t / scale).round().clamp(-qmax, qmax).to(torch.int8)
    if counter is not None:
        counter.count += 1
        counter.bytes += q.numel() * q.element_size()
    dist.all_reduce(q, op=dist.ReduceOp.SUM, group=group)
    return (q.float() * scale).to(orig_dtype)


def _int8_gather(tensor: torch.Tensor, group, counter, per_token: bool) -> torch.Tensor:
    orig_dtype = tensor.dtype
    ws = dist.get_world_size(group)
    shape = tensor.shape
    cols = shape[-1] if per_token else tensor.numel()
    t = tensor.contiguous().float().reshape(-1, cols)  # [rows, cols]; one row total when per-tensor
    rows = t.shape[0]

    scale = (t.abs().amax(dim=1).clamp(min=1e-8) / 127.0)  # [rows]
    q = (t / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    payload = torch.cat([scale.view(torch.uint8), q.view(torch.uint8).reshape(-1)])  # scales first: 4-byte aligned
    if counter is not None:
        counter.count += 1
        counter.bytes += payload.numel()

    if dist.get_backend(group) == "nccl":
        gathered = torch.empty(ws * payload.numel(), dtype=torch.uint8, device=payload.device)
        dist.all_gather_into_tensor(gathered, payload, group=group)
        pieces = gathered.view(ws, -1)
    else:
        bufs = [torch.empty_like(payload) for _ in range(ws)]
        dist.all_gather(bufs, payload, group=group)
        pieces = torch.stack(bufs)

    out = torch.zeros(rows, cols, dtype=torch.float32, device=t.device)
    for r in range(ws):
        s_r = pieces[r, : 4 * rows].contiguous().view(torch.float32)
        out += pieces[r, 4 * rows:].view(torch.int8).reshape(rows, cols).float() * s_r[:, None]
    return out.reshape(shape).to(orig_dtype)


def all_reduce_sum_int8_gather(tensor: torch.Tensor, group=None, counter: CommCounter | None = None) -> torch.Tensor:
    """INT8 on the wire via all-gather: each rank quantizes with its OWN
    per-tensor scale (full +-127), packs the 4 scale bytes in front of its
    int8 payload, one all-gather moves the packed bytes, and every rank
    dequantizes and sums the gathered pieces locally in rank order (so all
    ranks get an identical result). One collective per call, no scale-sync
    AllReduce, no GPU->CPU sync. Traffic matches an AllReduce at
    world_size=2 but grows with more ranks."""
    return _int8_gather(tensor, group, counter, per_token=False)


def all_reduce_sum_int8_gather_tok(tensor: torch.Tensor, group=None, counter: CommCounter | None = None) -> torch.Tensor:
    """Same as int8_gather but one scale per token (per row of the last dim),
    so a few outlier tokens/channels no longer set the quantization grid for
    the whole tensor. Costs 4 extra bytes per row on the wire."""
    return _int8_gather(tensor, group, counter, per_token=True)


ALLREDUCE_FNS = {
    "fp32": all_reduce_sum,
    "fp16": all_reduce_sum_fp16,
    "int8": all_reduce_sum_int8_gather_tok,  # Phase 6 winner: per-token scales, int8 on the wire, one collective
    "int8_legacy": all_reduce_sum_int8_legacy,
    "int8_sum": all_reduce_sum_int8_sum,
    "int8_gather": all_reduce_sum_int8_gather,
    "int8_gather_tok": all_reduce_sum_int8_gather_tok,
}


def channel_slice(rank: int, world_size: int, total: int) -> slice:
    assert total % world_size == 0, f"{total} not divisible by world_size={world_size}"
    shard = total // world_size
    return slice(rank * shard, (rank + 1) * shard)


SITES = ("x_proj", "out_proj")


def site_dtype(spec, layer_idx: int, site: str) -> str:
    """Resolves the AllReduce dtype for one collective site. `spec` is either
    a dtype name applied everywhere, or a {(layer_idx, site): dtype} map
    (sites missing from the map stay fp32). `site` is "x_proj" (AllReduce #1)
    or "out_proj" (AllReduce #2)."""
    assert site in SITES, site
    dtype = spec if isinstance(spec, str) else spec.get((layer_idx, site), "fp32")
    assert dtype in ALLREDUCE_FNS, f"unknown allreduce dtype {dtype!r}"
    return dtype

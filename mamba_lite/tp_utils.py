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


def channel_slice(rank: int, world_size: int, total: int) -> slice:
    assert total % world_size == 0, f"{total} not divisible by world_size={world_size}"
    shard = total // world_size
    return slice(rank * shard, (rank + 1) * shard)

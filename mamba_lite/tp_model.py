"""
Phase 2: tensor-parallel Mamba mixer, sharded across `world_size` ranks by
channel (the d_inner dimension). Two versions:

  TPMambaMixer        - correct sharding, exactly 2 AllReduces per block
  NaiveTPMambaMixer    - the "obvious" but wrong way to shard in_proj's packed
                         [x | z] output; produces incorrect results because it
                         slices straight through the x/z boundary instead of
                         splitting each half by channel first. Kept here on
                         purpose so the failure can be demonstrated in tests.

Sharding plan (matches the paper's Section IV-B/C, restated per-op):

  in_proj  [2*d_inner, d_model]  -> split x's d_inner rows AND z's d_inner
                                     rows separately by channel, then each
                                     rank concatenates its own x-shard + z-shard.
                                     (no comm)
  conv1d   depthwise, groups=d_inner -> slice channels.               (no comm)
  x_proj   [dt_rank+2*d_state, d_inner] -> split columns (contraction dim)
                                            by channel; each rank computes a
                                            partial sum of the FULL dt/B/C
                                            vector -> AllReduce #1 (sum).
  dt_proj, A_log, D, scan, gate -> all indexed by channel, local only.  (no comm)
  out_proj [d_model, d_inner] -> split columns (contraction dim) by channel;
                                  each rank computes a partial sum of the
                                  FULL d_model output -> AllReduce #2 (sum).

After AllReduce #1, every rank holds the identical, correct dt/B/C — that's
the "packed parameter" fix from Section IV-C: only the channel-sharded ∆
(via dt_proj) stays local per rank; B and C are the same on every rank.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MambaConfig, MambaMixer
from .tp_utils import CommCounter, all_reduce_sum, channel_slice


class _TPMambaMixerBase(nn.Module):
    """Shared structure/forward for the correct and naive variants; they only
    differ in how `load_shard_from_dense` slices in_proj."""

    def __init__(self, cfg: MambaConfig, rank: int, world_size: int, group=None):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.comm = CommCounter()

        assert cfg.d_inner % world_size == 0
        self.shard = cfg.d_inner // world_size
        shard, dt_rank, d_state, d_conv = self.shard, cfg.dt_rank, cfg.d_state, cfg.d_conv

        self.in_proj = nn.Linear(cfg.d_model, 2 * shard, bias=cfg.bias)
        self.conv1d = nn.Conv1d(shard, shard, kernel_size=d_conv, groups=shard, padding=d_conv - 1, bias=cfg.conv_bias)
        # x_proj/out_proj keep FULL output width; only their input (contraction) dim is sharded.
        self.x_proj = nn.Linear(shard, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, shard, bias=True)
        self.A_log = nn.Parameter(torch.empty(shard, d_state))
        self.D = nn.Parameter(torch.empty(shard))
        self.out_proj = nn.Linear(shard, cfg.d_model, bias=False)  # bias handled manually (rank 0 only)
        self.out_bias = nn.Parameter(torch.zeros(cfg.d_model)) if cfg.bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model], identical (replicated) on every rank."""
        b, l, _ = x.shape
        shard, d_state, dt_rank, d_conv = self.shard, self.cfg.d_state, self.cfg.dt_rank, self.cfg.d_conv

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)  # each [b, l, shard]
        x_ssm = x_ssm.transpose(1, 2)  # [b, shard, l]

        x_padded = F.pad(x_ssm, (d_conv - 1, 0))
        conv_out = F.conv1d(x_padded, self.conv1d.weight, self.conv1d.bias, groups=shard)[..., :l]
        x_ssm = F.silu(conv_out)  # [b, shard, l]

        # partial sum over this rank's channel-shard of d_inner -> needs AllReduce to be correct
        x_dbl_partial = self.x_proj(x_ssm.transpose(1, 2))  # [b, l, dt_rank+2*d_state]
        x_dbl = all_reduce_sum(x_dbl_partial.contiguous(), self.group, self.comm)  # AllReduce #1
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)

        dt = self.dt_proj.weight @ dt.transpose(1, 2) + self.dt_proj.bias[None, :, None]  # [b, shard, l]
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log.float())  # [shard, d_state]
        dA = torch.exp(dt[..., None] * A[None, :, None, :])  # [b, shard, l, d_state]
        dB = dt[..., None] * B[:, None, :, :]  # [b, shard, l, d_state]
        dBx = dB * x_ssm[..., None]

        ssm_state = torch.zeros(b, shard, d_state, device=x.device, dtype=torch.float32)
        ys = []
        for t in range(l):
            ssm_state = dA[:, :, t] * ssm_state + dBx[:, :, t]
            ys.append(torch.einsum("bdn,bn->bd", ssm_state, C[:, t]))
        y = torch.stack(ys, dim=1)  # [b, l, shard]
        y = y + x_ssm.transpose(1, 2) * self.D
        y = y * F.silu(z)

        out_partial = self.out_proj(y)  # [b, l, d_model], partial sum over this rank's channels
        out = all_reduce_sum(out_partial.contiguous(), self.group, self.comm)  # AllReduce #2
        if self.out_bias is not None:
            out = out + self.out_bias
        return out

    def load_shard_from_dense(self, dense: MambaMixer):
        raise NotImplementedError


class TPMambaMixer(_TPMambaMixerBase):
    """Correct sharding: in_proj's x-block and z-block are each split by
    channel independently, so no rank ever gets a slice that straddles the
    x/z boundary of the packed [x | z] tensor."""

    def load_shard_from_dense(self, dense: MambaMixer):
        r, ws = self.rank, self.world_size
        sl = channel_slice(r, ws, self.cfg.d_inner)

        w = dense.in_proj.weight  # [2*d_inner, d_model]
        d_inner = self.cfg.d_inner
        x_rows, z_rows = w[:d_inner][sl], w[d_inner:][sl]
        self.in_proj.weight.data.copy_(torch.cat([x_rows, z_rows], dim=0))
        if dense.in_proj.bias is not None:
            bx, bz = dense.in_proj.bias[:d_inner][sl], dense.in_proj.bias[d_inner:][sl]
            self.in_proj.bias.data.copy_(torch.cat([bx, bz], dim=0))

        self.conv1d.weight.data.copy_(dense.conv1d.weight[sl])
        if dense.conv1d.bias is not None:
            self.conv1d.bias.data.copy_(dense.conv1d.bias[sl])

        self.x_proj.weight.data.copy_(dense.x_proj.weight[:, sl])  # columns = input dim
        self.dt_proj.weight.data.copy_(dense.dt_proj.weight[sl])  # rows = output dim
        self.dt_proj.bias.data.copy_(dense.dt_proj.bias[sl])
        self.A_log.data.copy_(dense.A_log[sl])
        self.D.data.copy_(dense.D[sl])
        self.out_proj.weight.data.copy_(dense.out_proj.weight[:, sl])  # columns = input dim
        if self.out_bias is not None:
            # bias is added once after the AllReduce, not once per rank -> only rank 0 holds it
            self.out_bias.data.copy_(dense.out_proj.bias if r == 0 else torch.zeros_like(dense.out_proj.bias))


class NaiveTPMambaMixer(_TPMambaMixerBase):
    """WRONG sharding, kept to demonstrate the packed-tensor pitfall
    (Section IV-C): slices straight through the full 2*d_inner in_proj
    output contiguously, so for world_size=2, rank 0 gets all of `x` (and
    none of `z`) while rank 1 gets all of `z` (and none of `x`). Every other
    op is sharded the same (correct) way as TPMambaMixer, so this isolates
    exactly what goes wrong when the packed tensor is split naively."""

    def load_shard_from_dense(self, dense: MambaMixer):
        r, ws = self.rank, self.world_size
        sl = channel_slice(r, ws, self.cfg.d_inner)
        full_sl = channel_slice(r, ws, 2 * self.cfg.d_inner)  # <-- the bug: split the packed tensor directly

        self.in_proj.weight.data.copy_(dense.in_proj.weight[full_sl])
        if dense.in_proj.bias is not None:
            self.in_proj.bias.data.copy_(dense.in_proj.bias[full_sl])

        self.conv1d.weight.data.copy_(dense.conv1d.weight[sl])
        if dense.conv1d.bias is not None:
            self.conv1d.bias.data.copy_(dense.conv1d.bias[sl])
        self.x_proj.weight.data.copy_(dense.x_proj.weight[:, sl])
        self.dt_proj.weight.data.copy_(dense.dt_proj.weight[sl])
        self.dt_proj.bias.data.copy_(dense.dt_proj.bias[sl])
        self.A_log.data.copy_(dense.A_log[sl])
        self.D.data.copy_(dense.D[sl])
        self.out_proj.weight.data.copy_(dense.out_proj.weight[:, sl])
        if self.out_bias is not None:
            self.out_bias.data.copy_(dense.out_proj.bias if r == 0 else torch.zeros_like(dense.out_proj.bias))

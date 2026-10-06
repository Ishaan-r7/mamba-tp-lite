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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LayerCache, MambaConfig, MambaMixer
from .tp_utils import ALLREDUCE_FNS, CommCounter, channel_slice, site_dtype


class _TPMambaMixerBase(nn.Module):
    """Shared structure/prefill/step for the correct and naive variants; they
    only differ in how `load_shard_from_dense` slices in_proj. Each rank's
    cache (LayerCache) is naturally sharded: conv_state/ssm_state only ever
    hold this rank's channel-shard, since they're built from local tensors."""

    def __init__(self, cfg: MambaConfig, rank: int, world_size: int, group=None, allreduce_dtype="fp32",
                 layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.comm = CommCounter()
        self.layer_idx = layer_idx
        self.set_allreduce_dtype(allreduce_dtype)

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

    def set_allreduce_dtype(self, allreduce_dtype) -> None:
        self.allreduce_dtype = allreduce_dtype
        self._ar_x = ALLREDUCE_FNS[site_dtype(allreduce_dtype, self.layer_idx, "x_proj")]
        self._ar_out = ALLREDUCE_FNS[site_dtype(allreduce_dtype, self.layer_idx, "out_proj")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for the mixer-only tests: full sequence, no cache."""
        out, _, _ = self.prefill(x, cache=None)
        return out

    def prefill(self, x: torch.Tensor, cache: LayerCache | None = None):
        """x: [batch, seq_len, d_model], identical (replicated) on every rank.
        Returns (out [b,l,d_model], conv_state [b,shard,d_conv-1], ssm_state [b,shard,d_state])."""
        b, l, _ = x.shape
        shard, d_state, dt_rank, d_conv = self.shard, self.cfg.d_state, self.cfg.dt_rank, self.cfg.d_conv

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)  # each [b, l, shard]
        x_ssm = x_ssm.transpose(1, 2)  # [b, shard, l]

        if cache is not None and cache.conv_state is not None:
            x_padded = torch.cat([cache.conv_state, x_ssm], dim=-1)
        else:
            x_padded = F.pad(x_ssm, (d_conv - 1, 0))
        conv_out = F.conv1d(x_padded, self.conv1d.weight, self.conv1d.bias, groups=shard)[..., :l]
        x_ssm = F.silu(conv_out)  # [b, shard, l]
        final_conv_state = x_padded[..., -(d_conv - 1):].clone() if d_conv > 1 else torch.zeros(
            b, shard, 0, device=x.device, dtype=x.dtype
        )

        # partial sum over this rank's channel-shard of d_inner -> needs AllReduce to be correct
        x_dbl_partial = self.x_proj(x_ssm.transpose(1, 2))  # [b, l, dt_rank+2*d_state]
        x_dbl = self._ar_x(x_dbl_partial.contiguous(), self.group, self.comm)  # AllReduce #1
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)

        dt = self.dt_proj.weight @ dt.transpose(1, 2) + self.dt_proj.bias[None, :, None]  # [b, shard, l]
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log.float())  # [shard, d_state]
        dA = torch.exp(dt[..., None] * A[None, :, None, :])  # [b, shard, l, d_state]
        dB = dt[..., None] * B[:, None, :, :]  # [b, shard, l, d_state]
        dBx = dB * x_ssm[..., None]

        ssm_state = (
            cache.ssm_state.float()
            if cache is not None
            else torch.zeros(b, shard, d_state, device=x.device, dtype=torch.float32)
        )
        ys = []
        for t in range(l):
            ssm_state = dA[:, :, t] * ssm_state + dBx[:, :, t]
            ys.append(torch.einsum("bdn,bn->bd", ssm_state, C[:, t]))
        y = torch.stack(ys, dim=1)  # [b, l, shard]
        y = y + x_ssm.transpose(1, 2) * self.D
        y = y * F.silu(z)

        out_partial = self.out_proj(y)  # [b, l, d_model], partial sum over this rank's channels
        out = self._ar_out(out_partial.contiguous(), self.group, self.comm)  # AllReduce #2
        if self.out_bias is not None:
            out = out + self.out_bias
        return out, final_conv_state, ssm_state.to(x.dtype)

    def step(self, x_t: torch.Tensor, cache: LayerCache):
        """x_t: [batch, d_model] single token. Mutates cache in place (this
        rank's channel-shard only). Returns out: [batch, d_model]."""
        shard, d_state, dt_rank, d_conv = self.shard, self.cfg.d_state, self.cfg.dt_rank, self.cfg.d_conv

        xz = self.in_proj(x_t)
        x_ssm, z = xz.chunk(2, dim=-1)  # [b, shard] each

        if d_conv > 1:
            conv_in = torch.cat([cache.conv_state, x_ssm.unsqueeze(-1)], dim=-1)  # [b, shard, d_conv]
            w = self.conv1d.weight.squeeze(1)  # [shard, d_conv]
            conv_out = (conv_in * w[None]).sum(dim=-1)
            if self.conv1d.bias is not None:
                conv_out = conv_out + self.conv1d.bias
            # in-place: CUDA graph replay reads/writes fixed addresses, so the
            # cache tensor's identity must stay the same across steps, not be
            # replaced by a new tensor object each call (see cuda_graph.py)
            cache.conv_state.copy_(conv_in[..., 1:])
        else:
            conv_out = x_ssm
        x_ssm = F.silu(conv_out)  # [b, shard]

        x_dbl_partial = self.x_proj(x_ssm)  # [b, dt_rank+2*d_state]
        x_dbl = self._ar_x(x_dbl_partial.contiguous(), self.group, self.comm)  # AllReduce #1
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight, self.dt_proj.bias)  # [b, shard]
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log.float())  # [shard, d_state]
        dA = torch.exp(dt[..., None] * A[None])  # [b, shard, d_state]
        dB = dt[..., None] * B[:, None, :]  # [b, shard, d_state]
        dBx = dB * x_ssm[..., None]

        cache.ssm_state.copy_((cache.ssm_state.float() * dA + dBx).to(cache.ssm_state.dtype))
        y = torch.einsum("bdn,bn->bd", cache.ssm_state.float(), C)  # [b, shard]
        y = y + x_ssm * self.D
        y = y * F.silu(z)

        out_partial = self.out_proj(y)  # [b, d_model]
        out = self._ar_out(out_partial.contiguous(), self.group, self.comm)  # AllReduce #2
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


class TPMambaBlock(nn.Module):
    def __init__(self, cfg: MambaConfig, layer_idx: int, rank: int, world_size: int, group=None,
                 allreduce_dtype="fp32"):
        super().__init__()
        self.norm = _import_rmsnorm(cfg)
        self.mixer = TPMambaMixer(cfg, rank, world_size, group, allreduce_dtype, layer_idx=layer_idx)

    def prefill(self, x, cache: LayerCache | None):
        residual = x
        y, conv_state, ssm_state = self.mixer.prefill(self.norm(x), cache)
        return residual + y, conv_state, ssm_state

    def step(self, x_t, cache: LayerCache):
        residual = x_t
        y = self.mixer.step(self.norm(x_t), cache)
        return residual + y


def _import_rmsnorm(cfg: MambaConfig):
    from .model import RMSNorm

    return RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)


class TPMambaLM(nn.Module):
    """
    Tensor-parallel full model. Only the mixer is sharded (matching the
    paper's scope); the embedding/lm_head are small relative to the mixer
    stack and are simply replicated on every rank, no vocab-parallelism.

    Build with `TPMambaLM.from_pretrained_shard(...)`, which reads the
    checkpoint's state dict directly and slices each tensor into this rank's
    shard *before* copying it into a parameter — it never materializes a
    full dense model in this process, so memory scales down with world_size
    (the actual point of TP), not just compute.
    """

    def __init__(self, cfg: MambaConfig, rank: int, world_size: int, group=None, allreduce_dtype="fp32"):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.allreduce_dtype = allreduce_dtype
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [TPMambaBlock(cfg, i, rank, world_size, group, allreduce_dtype) for i in range(cfg.n_layer)]
        )
        self.norm_f = _import_rmsnorm(cfg)

    def set_allreduce_dtype(self, allreduce_dtype) -> None:
        self.allreduce_dtype = allreduce_dtype
        for layer in self.layers:
            layer.mixer.set_allreduce_dtype(allreduce_dtype)

    def lm_head(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.weight)

    def prefill(self, input_ids: torch.Tensor, cache=None):
        from .model import MambaCache

        x = self.embedding(input_ids)
        new_layers = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache.layers[i] if cache is not None else None
            x, conv_state, ssm_state = layer.prefill(x, layer_cache)
            new_layers.append(LayerCache(conv_state=conv_state, ssm_state=ssm_state))
        x = self.norm_f(x)
        return self.lm_head(x), MambaCache(new_layers)

    def step(self, input_ids: torch.Tensor, cache):
        x = self.embedding(input_ids).squeeze(1)
        for i, layer in enumerate(self.layers):
            x = layer.step(x, cache.layers[i])
        x = self.norm_f(x)
        return self.lm_head(x).unsqueeze(1)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int):
        logits, cache = self.prefill(input_ids)
        next_id = logits[:, -1:].argmax(dim=-1)
        tokens = [input_ids, next_id]
        for _ in range(max_new_tokens - 1):
            logits = self.step(next_id, cache)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_id)
        return torch.cat(tokens, dim=1)

    @classmethod
    def from_pretrained_shard(
        cls, hf_path: str, rank: int, world_size: int, group=None, dtype=torch.float32,
        allreduce_dtype="fp32",
    ) -> "TPMambaLM":
        """Loads a HF `state-spaces/mamba-*-hf` checkpoint (single-file or
        sharded safetensors), keeping in memory (per rank) only this rank's
        channel-shard of every mixer tensor. `allreduce_dtype` picks the
        precision used for the *collective* only ("fp32"/"fp16"/"int8") --
        model weights/activations stay in `dtype` throughout."""
        from .model import load_hf_state_dict

        cfg = MambaConfig.from_hf_config(f"{hf_path}/config.json")
        model = cls(cfg, rank, world_size, group, allreduce_dtype).to(dtype)
        sd = load_hf_state_dict(hf_path)

        model.embedding.weight.data.copy_(sd["backbone.embeddings.weight"].to(dtype))
        model.norm_f.weight.data.copy_(sd["backbone.norm_f.weight"].to(dtype))

        sl = channel_slice(rank, world_size, cfg.d_inner)
        d_inner = cfg.d_inner
        for i in range(cfg.n_layer):
            p = f"backbone.layers.{i}."
            block = model.layers[i]
            block.norm.weight.data.copy_(sd[p + "norm.weight"].to(dtype))
            mixer = block.mixer

            w = sd[p + "mixer.in_proj.weight"]
            x_rows, z_rows = w[:d_inner][sl], w[d_inner:][sl]
            mixer.in_proj.weight.data.copy_(torch.cat([x_rows, z_rows], dim=0).to(dtype))
            if cfg.bias:
                b = sd[p + "mixer.in_proj.bias"]
                bx, bz = b[:d_inner][sl], b[d_inner:][sl]
                mixer.in_proj.bias.data.copy_(torch.cat([bx, bz], dim=0).to(dtype))

            mixer.conv1d.weight.data.copy_(sd[p + "mixer.conv1d.weight"][sl].to(dtype))
            if cfg.conv_bias:
                mixer.conv1d.bias.data.copy_(sd[p + "mixer.conv1d.bias"][sl].to(dtype))

            mixer.x_proj.weight.data.copy_(sd[p + "mixer.x_proj.weight"][:, sl].to(dtype))
            mixer.dt_proj.weight.data.copy_(sd[p + "mixer.dt_proj.weight"][sl].to(dtype))
            mixer.dt_proj.bias.data.copy_(sd[p + "mixer.dt_proj.bias"][sl].to(dtype))
            mixer.A_log.data.copy_(sd[p + "mixer.A_log"][sl].to(dtype))
            mixer.D.data.copy_(sd[p + "mixer.D"][sl].to(dtype))
            mixer.out_proj.weight.data.copy_(sd[p + "mixer.out_proj.weight"][:, sl].to(dtype))
            if cfg.bias and mixer.out_bias is not None:
                ob = sd[p + "mixer.out_proj.bias"]
                mixer.out_bias.data.copy_(ob.to(dtype) if rank == 0 else torch.zeros_like(ob, dtype=dtype))

        return model

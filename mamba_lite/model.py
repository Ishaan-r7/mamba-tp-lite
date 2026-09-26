"""
Minimal from-scratch Mamba (v1) forward pass, in plain PyTorch.

Loads weights from a HF `state-spaces/mamba-*-hf` checkpoint and reproduces
its numerics exactly (verified against HF in tests/test_correctness.py).

Two entry points per mixer:
  - prefill(x)        : full-sequence selective scan, returns output + cache
  - step(x_t, cache)   : single-token recurrent update using the cache

Shapes (mamba-130m defaults):
  d_model    = 768   (hidden_size)
  d_inner    = 1536  (intermediate_size, = expand * d_model)
  d_state    = 16    (state_size, "N")
  dt_rank    = 48    (time_step_rank)
  d_conv     = 4     (conv_kernel)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_hf_state_dict(hf_path: str) -> dict[str, torch.Tensor]:
    """Loads a HF checkpoint's weights, whether it's a single model.safetensors
    file (e.g. mamba-130m-hf) or sharded across multiple files with a
    model.safetensors.index.json (e.g. mamba-1.4b-hf, mamba-2.8b-hf)."""
    import os

    from safetensors.torch import load_file

    single = os.path.join(hf_path, "model.safetensors")
    index = os.path.join(hf_path, "model.safetensors.index.json")
    if os.path.exists(single):
        return load_file(single)
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            state_dict.update(load_file(os.path.join(hf_path, shard_file)))
        return state_dict
    raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json under {hf_path}")


@dataclass
class MambaConfig:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    conv_bias: bool = True
    bias: bool = False
    pad_vocab_size_multiple: int = 8
    layer_norm_epsilon: float = 1e-5

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    def __post_init__(self):
        if self.dt_rank is None:
            self.dt_rank = max(1, self.d_model // 16)
        # HF pads vocab size up to a multiple; if loading from HF config, the
        # config.json already carries the padded vocab_size, so this is a no-op there.
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple
            )

    @classmethod
    def from_hf_config(cls, config_path: str) -> "MambaConfig":
        cfg = json.load(open(config_path))
        return cls(
            d_model=cfg["hidden_size"],
            n_layer=cfg["num_hidden_layers"],
            vocab_size=cfg["vocab_size"],
            d_state=cfg["state_size"],
            d_conv=cfg["conv_kernel"],
            expand=cfg["expand"],
            dt_rank=cfg["time_step_rank"],
            conv_bias=cfg["use_conv_bias"],
            bias=cfg["use_bias"],
            pad_vocab_size_multiple=cfg.get("pad_vocab_size_multiple", 8),
            layer_norm_epsilon=cfg.get("layer_norm_epsilon", 1e-5),
        )


@dataclass
class LayerCache:
    """Per-layer recurrent state, sized so it can be sharded by channel later (Phase 2)."""

    conv_state: torch.Tensor  # [batch, d_inner, d_conv - 1]
    ssm_state: torch.Tensor  # [batch, d_inner, d_state]


class MambaCache:
    def __init__(self, layers: list[LayerCache]):
        self.layers = layers

    @staticmethod
    def allocate(config: MambaConfig, batch_size: int, device, dtype) -> "MambaCache":
        layers = [
            LayerCache(
                conv_state=torch.zeros(
                    batch_size, config.d_inner, config.d_conv - 1, device=device, dtype=dtype
                ),
                ssm_state=torch.zeros(
                    batch_size, config.d_inner, config.d_state, device=device, dtype=dtype
                ),
            )
            for _ in range(config.n_layer)
        ]
        return MambaCache(layers)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype))


class MambaMixer(nn.Module):
    """
    One Mamba mixer block. This is the unit that Phase 2 will shard across GPUs:
      in_proj  -> [x | z]                (column-parallel candidate)
      conv1d (depthwise, groups=d_inner) (shards for free, by channel)
      x_proj   -> [dt | B | C]            (row-parallel -> AllReduce #1)
      dt_proj, A_log, D, scan, gate       (all per-channel, shard for free)
      out_proj                            (row-parallel -> AllReduce #2)
    """

    def __init__(self, cfg: MambaConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        d_inner, d_model, d_state, dt_rank, d_conv = (
            cfg.d_inner,
            cfg.d_model,
            cfg.d_state,
            cfg.dt_rank,
            cfg.d_conv,
        )

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=cfg.bias)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1, bias=cfg.conv_bias
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        self.A_log = nn.Parameter(torch.empty(d_inner, d_state))
        self.D = nn.Parameter(torch.empty(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=cfg.bias)
        self._reset_ssm_params()

    def _reset_ssm_params(self):
        """A_log/D are plain nn.Parameter(torch.empty(...)) so, unlike in_proj/x_proj/etc,
        they get no default init from nn.Linear/nn.Conv1d. from_pretrained() overwrites
        these anyway, but a freshly-constructed MambaMixer (e.g. in the TP tests) needs
        them deterministic, not leftover uninitialized memory. Same S4D-real init as HF."""
        with torch.no_grad():
            A = torch.arange(1, self.A_log.shape[1] + 1, dtype=torch.float32)[None, :]
            A = A.expand(self.A_log.shape[0], -1).contiguous()
            self.A_log.copy_(torch.log(A))
            self.D.fill_(1.0)

    # ---- prefill: full-sequence selective scan (sequential reference form) ----
    def prefill(self, x: torch.Tensor, cache: LayerCache | None = None):
        """
        x: [batch, seq_len, d_model]
        Returns: (y [batch, seq_len, d_model], final_conv_state, final_ssm_state)
        """
        b, l, _ = x.shape
        d_inner, d_state, dt_rank, d_conv = (
            self.cfg.d_inner,
            self.cfg.d_state,
            self.cfg.dt_rank,
            self.cfg.d_conv,
        )

        xz = self.in_proj(x)  # [b, l, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)  # each [b, l, d_inner]
        x_ssm = x_ssm.transpose(1, 2)  # [b, d_inner, l]

        # depthwise causal conv: left-pad with prior conv_state (zeros if none)
        if cache is not None and cache.conv_state is not None:
            x_padded = torch.cat([cache.conv_state, x_ssm], dim=-1)
        else:
            x_padded = F.pad(x_ssm, (d_conv - 1, 0))
        conv_out = F.conv1d(
            x_padded, self.conv1d.weight, self.conv1d.bias, groups=d_inner
        )[..., :l]
        x_ssm = F.silu(conv_out)  # [b, d_inner, l]
        final_conv_state = x_padded[..., -(d_conv - 1):].clone() if d_conv > 1 else torch.zeros(
            b, d_inner, 0, device=x.device, dtype=x.dtype
        )

        x_dbl = self.x_proj(x_ssm.transpose(1, 2))  # [b, l, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.transpose(1, 2)  # [b, d_inner, l] (bias added below)
        dt = dt + self.dt_proj.bias[None, :, None]
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        # discretize over the whole sequence
        dA = torch.exp(dt[..., None] * A[None, :, None, :])  # [b, d_inner, l, d_state]
        dB = dt[..., None] * B[:, None, :, :]  # [b, d_inner, l, d_state]
        dBx = dB * x_ssm[..., None]  # [b, d_inner, l, d_state]

        ssm_state = (
            cache.ssm_state.float()
            if cache is not None
            else torch.zeros(b, d_inner, d_state, device=x.device, dtype=torch.float32)
        )
        ys = []
        for t in range(l):
            ssm_state = dA[:, :, t] * ssm_state + dBx[:, :, t]
            y_t = torch.einsum("bdn,bn->bd", ssm_state, C[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # [b, l, d_inner]
        y = y + x_ssm.transpose(1, 2) * self.D  # D skip
        y = y * F.silu(z)
        out = self.out_proj(y)  # [b, l, d_model]

        return out, final_conv_state, ssm_state.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain no-cache forward, used as the dense reference in TP correctness tests."""
        out, _, _ = self.prefill(x, cache=None)
        return out

    # ---- step: single-token recurrent update (decode) ----
    def step(self, x_t: torch.Tensor, cache: LayerCache):
        """
        x_t: [batch, d_model] (single token)
        Mutates cache in place. Returns y_t: [batch, d_model].
        """
        d_inner, d_state, dt_rank, d_conv = (
            self.cfg.d_inner,
            self.cfg.d_state,
            self.cfg.dt_rank,
            self.cfg.d_conv,
        )

        xz = self.in_proj(x_t)  # [b, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)  # [b, d_inner] each

        # depthwise conv over the last d_conv inputs (d_conv-1 history + current)
        if d_conv > 1:
            conv_in = torch.cat([cache.conv_state, x_ssm.unsqueeze(-1)], dim=-1)  # [b, d_inner, d_conv]
            w = self.conv1d.weight.squeeze(1)  # [d_inner, d_conv]
            conv_out = (conv_in * w[None]).sum(dim=-1)  # [b, d_inner]
            if self.conv1d.bias is not None:
                conv_out = conv_out + self.conv1d.bias
            # in-place: CUDA graph replay reads/writes fixed addresses, so the
            # cache tensor's identity must stay the same across steps, not be
            # replaced by a new tensor object each call (see cuda_graph.py)
            cache.conv_state.copy_(conv_in[..., 1:])
        else:
            conv_out = x_ssm
        x_ssm = F.silu(conv_out)  # [b, d_inner]

        x_dbl = self.x_proj(x_ssm)  # [b, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight, self.dt_proj.bias)  # [b, d_inner]
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        dA = torch.exp(dt[..., None] * A[None])  # [b, d_inner, d_state]
        dB = dt[..., None] * B[:, None, :]  # [b, d_inner, d_state]
        dBx = dB * x_ssm[..., None]  # [b, d_inner, d_state]

        cache.ssm_state.copy_((cache.ssm_state.float() * dA + dBx).to(cache.ssm_state.dtype))
        y = torch.einsum("bdn,bn->bd", cache.ssm_state.float(), C)  # [b, d_inner]
        y = y + x_ssm * self.D
        y = y * F.silu(z)
        out = self.out_proj(y)  # [b, d_model]
        return out


class MambaBlock(nn.Module):
    def __init__(self, cfg: MambaConfig, layer_idx: int):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.mixer = MambaMixer(cfg, layer_idx)

    def prefill(self, x, cache: LayerCache | None):
        residual = x
        y, conv_state, ssm_state = self.mixer.prefill(self.norm(x), cache)
        return residual + y, conv_state, ssm_state

    def step(self, x_t, cache: LayerCache):
        residual = x_t
        y = self.mixer.step(self.norm(x_t), cache)
        return residual + y


class MambaLM(nn.Module):
    """Full model: embedding -> n_layer MambaBlocks -> final norm -> tied lm_head."""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([MambaBlock(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        # lm_head is tied to the embedding in mamba-*-hf checkpoints.

    def lm_head(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.weight)

    def prefill(self, input_ids: torch.Tensor, cache: MambaCache | None = None):
        """
        input_ids: [batch, seq_len]
        Returns: (logits [batch, seq_len, vocab], cache)
        """
        x = self.embedding(input_ids)
        new_cache_layers = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache.layers[i] if cache is not None else None
            x, conv_state, ssm_state = layer.prefill(x, layer_cache)
            new_cache_layers.append(LayerCache(conv_state=conv_state, ssm_state=ssm_state))
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, MambaCache(new_cache_layers)

    def step(self, input_ids: torch.Tensor, cache: MambaCache):
        """
        input_ids: [batch, 1] (single new token per sequence)
        Mutates cache in place. Returns logits: [batch, 1, vocab].
        """
        x = self.embedding(input_ids).squeeze(1)  # [b, d_model]
        for i, layer in enumerate(self.layers):
            x = layer.step(x, cache.layers[i])
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits.unsqueeze(1)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, greedy: bool = True):
        """Prefill once, then decode token-by-token using the cache. Returns full token sequence."""
        logits, cache = self.prefill(input_ids)
        next_id = logits[:, -1:].argmax(dim=-1) if greedy else None
        tokens = [input_ids, next_id]
        for _ in range(max_new_tokens - 1):
            logits = self.step(next_id, cache)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_id)
        return torch.cat(tokens, dim=1)

    @classmethod
    def from_pretrained(cls, hf_path: str, dtype=torch.float32) -> "MambaLM":
        """Load weights from a local HF `state-spaces/mamba-*-hf` snapshot directory
        (single-file or sharded safetensors, see load_hf_state_dict)."""
        cfg = MambaConfig.from_hf_config(f"{hf_path}/config.json")
        model = cls(cfg).to(dtype)
        state_dict = load_hf_state_dict(hf_path)

        remapped = {}
        remapped["embedding.weight"] = state_dict["backbone.embeddings.weight"]
        remapped["norm_f.weight"] = state_dict["backbone.norm_f.weight"]
        for i in range(cfg.n_layer):
            p = f"backbone.layers.{i}."
            q = f"layers.{i}."
            remapped[q + "norm.weight"] = state_dict[p + "norm.weight"]
            remapped[q + "mixer.in_proj.weight"] = state_dict[p + "mixer.in_proj.weight"]
            remapped[q + "mixer.conv1d.weight"] = state_dict[p + "mixer.conv1d.weight"]
            if cfg.conv_bias:
                remapped[q + "mixer.conv1d.bias"] = state_dict[p + "mixer.conv1d.bias"]
            remapped[q + "mixer.x_proj.weight"] = state_dict[p + "mixer.x_proj.weight"]
            remapped[q + "mixer.dt_proj.weight"] = state_dict[p + "mixer.dt_proj.weight"]
            remapped[q + "mixer.dt_proj.bias"] = state_dict[p + "mixer.dt_proj.bias"]
            remapped[q + "mixer.A_log"] = state_dict[p + "mixer.A_log"]
            remapped[q + "mixer.D"] = state_dict[p + "mixer.D"]
            remapped[q + "mixer.out_proj.weight"] = state_dict[p + "mixer.out_proj.weight"]
            if cfg.bias:
                remapped[q + "mixer.out_proj.bias"] = state_dict[p + "mixer.out_proj.bias"]

        missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in remapped.items()}, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        return model

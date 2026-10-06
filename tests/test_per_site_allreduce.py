"""
Per-collective AllReduce dtype hook (Phase 6): the dtype is looked up per
(layer_idx, site) instead of one global setting, and the new INT8 variants
put real int8 on the wire. All checks run on 2-process gloo / CPU.
"""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mamba_lite.model import MambaConfig, MambaMixer
from mamba_lite.tp_model import TPMambaMixer
from mamba_lite.tp_utils import site_dtype

WORLD_SIZE = 2
LAYER = 3

CASES = {
    "fp32_str": ("fp32", LAYER),
    "fp32_map": ({(LAYER, "x_proj"): "fp32", (LAYER, "out_proj"): "fp32"}, LAYER),
    "empty_map": ({}, LAYER),
    "int8_out_only_hit": ({(LAYER, "out_proj"): "int8_gather"}, LAYER),
    "int8_out_only_miss": ({(LAYER, "out_proj"): "int8_gather"}, LAYER + 1),
    "int8_x_only": ({(LAYER, "x_proj"): "int8_gather"}, LAYER),
    "int8_gather_all": ("int8_gather", LAYER),
    "int8_gather_tok_all": ("int8_gather_tok", LAYER),
    "int8_sum_all": ("int8_sum", LAYER),
    "int8_legacy_all": ("int8_legacy", LAYER),
    "fp16_all": ("fp16", LAYER),
}


def _worker(rank, world_size, init_file, q):
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    cfg = MambaConfig(d_model=32, n_layer=1, vocab_size=100, d_state=8, d_conv=4, expand=2)
    dense = MambaMixer(cfg, layer_idx=0).eval()
    torch.manual_seed(123)
    x = torch.randn(2, 5, cfg.d_model)

    results = {"dense": dense(x).detach()}
    for name, (spec, layer_idx) in CASES.items():
        tp = TPMambaMixer(cfg, rank, world_size, allreduce_dtype=spec, layer_idx=layer_idx)
        tp.load_shard_from_dense(dense)
        tp.eval()
        with torch.no_grad():
            out = tp(x)
        results[name] = (out, tp.comm.count, tp.comm.bytes)
    q.put((rank, results))
    dist.destroy_process_group()


def _run():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "rendezvous")
        procs = [ctx.Process(target=_worker, args=(r, WORLD_SIZE, init_file, q)) for r in range(WORLD_SIZE)]
        for p in procs:
            p.start()
        res = dict(q.get(timeout=120) for _ in range(WORLD_SIZE))
        for p in procs:
            p.join(timeout=120)
    return res


def test_site_dtype_lookup():
    assert site_dtype("fp16", 5, "x_proj") == "fp16"
    m = {(2, "out_proj"): "int8_gather"}
    assert site_dtype(m, 2, "out_proj") == "int8_gather"
    assert site_dtype(m, 2, "x_proj") == "fp32"
    assert site_dtype(m, 3, "out_proj") == "fp32"


def test_per_site_hook_and_int8_variants():
    res = _run()
    r0 = res[0]
    dense = r0["dense"]

    base = r0["fp32_str"][0]
    # all-fp32 map (full, partial or empty) is bit-identical to the old global-string path
    assert torch.equal(r0["fp32_map"][0], base)
    assert torch.equal(r0["empty_map"][0], base)
    assert torch.allclose(base, dense, atol=1e-5)

    # CommCounter: exactly 2 per forward for fp32/fp16/int8_gather; sum-variant and legacy pay a scale collective per site
    for name in ("fp32_str", "fp32_map", "empty_map", "fp16_all", "int8_gather_all", "int8_gather_tok_all",
                 "int8_out_only_hit", "int8_x_only", "int8_out_only_miss"):
        assert r0[name][1] == 2, (name, r0[name][1])
    assert r0["int8_sum_all"][1] == 4 and r0["int8_legacy_all"][1] == 4

    # site keying: the map entry hits only its own (layer, site)
    assert not torch.equal(r0["int8_out_only_hit"][0], base)
    assert torch.equal(r0["int8_out_only_miss"][0], base)
    assert not torch.equal(r0["int8_x_only"][0], base)

    # real int8 on the wire: payload bytes per call shrink vs fp32, legacy (int32) does not
    assert r0["int8_gather_all"][2] < r0["fp32_str"][2] / 3
    assert r0["int8_gather_tok_all"][2] < r0["fp32_str"][2] / 3
    assert r0["int8_legacy_all"][2] >= r0["fp32_str"][2]

    # all ranks must end up with the identical (post-collective) output
    for name in CASES:
        assert torch.equal(res[0][name][0], res[1][name][0]), name

    # lossy but close
    for name in ("int8_gather_all", "int8_gather_tok_all", "int8_sum_all", "int8_legacy_all", "fp16_all"):
        err = (res[0][name][0] - dense).abs().max().item()
        scale = dense.abs().max().item()
        assert err < 0.15 * scale, (name, err, scale)

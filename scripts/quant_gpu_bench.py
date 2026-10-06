"""Phase 6 GPU benchmark: fp32 vs fp16 vs int8 vs mixed AllReduce precision,
all in ONE process / ONE session, interleaved over several rounds so shared-GPU
drift hits every config equally (the Phase 4 lesson).

Per config it measures
  - prefill throughput (batch 32, len 1024 by default)
  - decode ms/token, eager and CUDA-graph captured (batch 1)
  - bytes sent per prefill forward (from CommCounter)
and first sanity-checks that int8 / graph capture are numerically sane on NCCL.

Run on a 2-GPU box:
  torchrun --standalone --nproc_per_node=2 scripts/quant_gpu_bench.py \
      --mixed-json benchmarks/quant_mixed/mixed_configs.json
Rank 0 prints a final line starting with QUANT_GPU_RESULT_JSON: -- paste that back.
"""
import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

from mamba_lite.quant_eval import quant_metrics
from mamba_lite.tp_model import TPMambaLM
from run_tp import bench_prefill_decode, bench_prefill_throughput, verify_cuda_graph_correctness


def build_spec(cfg_def, n_layer):
    """cfg_def: a dtype name, or {"default": dtype, "overrides": {"L:site": dtype}}."""
    if isinstance(cfg_def, str):
        return cfg_def
    spec = {}
    for layer in range(n_layer):
        for site in ("x_proj", "out_proj"):
            spec[(layer, site)] = cfg_def["overrides"].get(f"{layer}:{site}", cfg_def["default"])
    return spec


def total_bytes(model):
    return sum(layer.mixer.comm.bytes for layer in model.layers)


def reset_comm(model):
    for layer in model.layers:
        layer.mixer.comm.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--mixed-json", default="benchmarks/quant_mixed/mixed_configs.json")
    ap.add_argument("--configs", nargs="+", default=None, help="subset/order of config names")
    ap.add_argument("--prefill-batch", type=int, default=32)
    ap.add_argument("--prefill-len", type=int, default=1024)
    ap.add_argument("--decode-prompt-len", type=int, default=256)
    ap.add_argument("--n-new-tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="quant_gpu_results.json")
    ap.add_argument("--cpu-smoke", action="store_true", help="gloo/CPU dry run to check the script logic (no timing value)")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = "cpu" if args.cpu_smoke else "cuda"
    if not args.cpu_smoke:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo" if args.cpu_smoke else "nccl", rank=rank, world_size=world_size)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    model_path = snapshot_download(args.model_id)
    model = TPMambaLM.from_pretrained_shard(model_path, rank=rank, world_size=world_size).to(device).eval()
    n_layer = model.cfg.n_layer

    defs = {"fp32": "fp32", "fp16": "fp16", "int8": "int8"}
    if os.path.exists(args.mixed_json):
        defs.update(json.load(open(args.mixed_json)))
    names = args.configs or list(defs)
    specs = {n: build_spec(defs[n], n_layer) for n in names}
    log(f"configs: {names}")

    # --- sanity: logits vs fp32 on NCCL, and graph correctness for each lossy config ---
    sanity = {}
    from transformers import AutoTokenizer

    text = ("The quick brown fox jumps over the lazy dog. State space models process sequences token by "
            "token, maintaining a compact hidden state instead of attending over the full history. This "
            "makes inference cost scale linearly with sequence length. Tensor parallelism splits each "
            "layer across several GPUs, which must then exchange partial results after every block.")
    ids = AutoTokenizer.from_pretrained(model_path)(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        model.set_allreduce_dtype("fp32")
        ref, _ = model.prefill(ids)
        for n in names:
            model.set_allreduce_dtype(specs[n])
            out, _ = model.prefill(ids)
            m = quant_metrics(ref.float().cpu(), out.float().cpu())
            kl, top1 = m["kl"].mean().item(), m["top1"].mean().item()
            finite = bool(torch.isfinite(out).all())
            graph_ok, detail = None, None
            try:
                graph_ok, detail = verify_cuda_graph_correctness(model, 64, device)
            except Exception as e:  # capture can fail for reasons other than CUDAGraphUnsupported
                detail = f"{type(e).__name__}: {str(e)[:200]}"
            sanity[n] = {"kl_vs_fp32": kl, "top1_vs_fp32": top1, "finite": finite, "graph_match": graph_ok,
                         "graph_detail": detail if isinstance(detail, str) else float(detail)}
            log(f"[sanity] {n:12s} KL vs fp32 {kl:.5f} top1 {top1 * 100:.1f}% finite={finite} graph_match={graph_ok} detail={detail}")

    # --- timed rounds, interleaved across configs ---
    rows = {n: {"prefill_tok_s": [], "decode_eager_ms": [], "decode_graph_ms": [], "graph_status": None,
                "bytes_per_prefill": None} for n in names}
    for rnd in range(args.rounds):
        for n in names:
            model.set_allreduce_dtype(specs[n])
            reset_comm(model)
            r = bench_prefill_throughput(model, args.prefill_batch, args.prefill_len, device, n_repeats=3)
            if rows[n]["bytes_per_prefill"] is None:
                rows[n]["bytes_per_prefill"] = total_bytes(model) / 4  # warmup + 3 repeats = 4 forwards
            rows[n]["prefill_tok_s"].append(r["prefill_tokens_per_s"])

            bench_prefill_decode(model, args.decode_prompt_len, 4, device)
            e = bench_prefill_decode(model, args.decode_prompt_len, args.n_new_tokens, device)
            rows[n]["decode_eager_ms"].append(e["decode_ms_per_token"])
            try:
                bench_prefill_decode(model, args.decode_prompt_len, 4, device, use_cuda_graph=True)
                g = bench_prefill_decode(model, args.decode_prompt_len, args.n_new_tokens, device, use_cuda_graph=True)
                rows[n]["graph_status"] = g["cuda_graph_status"]
                if g["cuda_graph_status"] == "captured":
                    rows[n]["decode_graph_ms"].append(g["decode_ms_per_token"])
            except Exception as ex:
                rows[n]["graph_status"] = f"error:{type(ex).__name__}: {str(ex)[:160]}"
            log(f"round {rnd + 1} {n:12s} prefill {rows[n]['prefill_tok_s'][-1]:9.0f} tok/s  "
                f"decode eager {rows[n]['decode_eager_ms'][-1]:.2f} ms  "
                f"graph {rows[n]['decode_graph_ms'][-1] if rows[n]['decode_graph_ms'] else 'n/a'}")

    summary = {}
    for n in names:
        r = rows[n]
        summary[n] = {
            "prefill_tok_s_median": statistics.median(r["prefill_tok_s"]),
            "decode_eager_ms_median": statistics.median(r["decode_eager_ms"]),
            "decode_graph_ms_median": statistics.median(r["decode_graph_ms"]) if r["decode_graph_ms"] else None,
            "graph_status": r["graph_status"],
            "bytes_per_prefill_rank0": r["bytes_per_prefill"],
            "raw": r,
        }
    base = summary["fp32"] if "fp32" in summary else None
    if base:
        log("\n=== vs fp32 (same session, median of rounds) ===")
        for n in names:
            s = summary[n]
            pf = s["prefill_tok_s_median"] / base["prefill_tok_s_median"]
            de = base["decode_eager_ms_median"] / s["decode_eager_ms_median"]
            dg = (base["decode_graph_ms_median"] / s["decode_graph_ms_median"]
                  if s["decode_graph_ms_median"] and base["decode_graph_ms_median"] else None)
            log(f"{n:12s} prefill x{pf:.3f}  decode-eager x{de:.3f}  decode-graph x{dg if dg is None else round(dg, 3)}  "
                f"wire MB/prefill {s['bytes_per_prefill_rank0'] / 1e6:.0f}")

    if rank == 0:
        result = {"model": args.model_id, "world_size": world_size, "device": "cpu" if args.cpu_smoke else torch.cuda.get_device_name(0),
                  "args": vars(args), "sanity": sanity, "summary": summary}
        json.dump(result, open(args.out, "w"))
        print("QUANT_GPU_RESULT_JSON:" + json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

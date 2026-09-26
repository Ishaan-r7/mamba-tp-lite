"""
Phase 3: run the tensor-parallel Mamba model as real separate processes and
measure it. Same script for two situations:

  # Local smoke test, no GPU, 2 CPU processes, gloo backend:
  torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend gloo --device cpu

  # Real 2-GPU run on Kaggle's "GPU T4 x2" (or any 2-GPU box), nccl backend:
  torchrun --standalone --nproc_per_node=2 scripts/run_tp.py --backend nccl --device cuda

`torchrun` sets RANK/WORLD_SIZE/LOCAL_RANK env vars and launches one process
per GPU; this script reads those instead of doing its own multiprocessing
(that's only needed in the pytest tests, where we don't have torchrun).

With --profile, wraps the decode loop in torch.profiler and writes a Chrome
trace per rank to --profile-dir (open at chrome://tracing or with
Perfetto UI at https://ui.perfetto.dev).
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

from mamba_lite.model import MambaLM
from mamba_lite.tp_model import TPMambaLM


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def bench_prefill_decode(model, prompt_len, n_new_tokens, device, batch_size=1):
    ids = torch.randint(0, model.cfg.vocab_size, (batch_size, prompt_len), device=device)

    with torch.no_grad():
        sync(device)
        t0 = time.perf_counter()
        _, cache = model.prefill(ids)
        sync(device)
        t1 = time.perf_counter()

        next_id = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        for _ in range(n_new_tokens):
            model.step(next_id, cache)
        sync(device)
        t2 = time.perf_counter()

    return {
        "prefill_s": t1 - t0,
        "decode_ms_per_token": (t2 - t1) / n_new_tokens * 1000,
        "tokens_per_s_decode": n_new_tokens / (t2 - t1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None, help="local HF snapshot dir; overrides --model-id if set")
    ap.add_argument("--model-id", default="state-spaces/mamba-130m-hf",
                     help="HF hub id to download, e.g. state-spaces/mamba-1.4b-hf")
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--n-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--compare-single-gpu", action="store_true",
                     help="rank 0 also times the dense (non-TP) model on its own device, for a TP-vs-1-GPU comparison")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-dir", default="profiles")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)

    model_path = args.model_path or snapshot_download(args.model_id)
    dtype = torch.float32
    tp_model = TPMambaLM.from_pretrained_shard(model_path, rank=rank, world_size=world_size, dtype=dtype)
    tp_model = tp_model.to(args.device).eval()
    log(rank, f"loaded TP shard on {args.device}:{local_rank}, world_size={world_size}")

    results = {}
    for L in args.prompt_lens:
        # warmup (build kernels / cuDNN autotune / first-touch memory)
        bench_prefill_decode(tp_model, L, min(4, args.n_new_tokens), args.device, args.batch_size)
        stats = bench_prefill_decode(tp_model, L, args.n_new_tokens, args.device, args.batch_size)
        comm = tp_model.layers[0].mixer.comm
        stats["allreduces_per_layer0"] = comm.count
        comm.reset()
        results[L] = stats
        log(rank, f"prompt_len={L}: {stats}")

    if args.compare_single_gpu and rank == 0:
        dense = MambaLM.from_pretrained(model_path, dtype=dtype).to(args.device).eval()
        log(rank, "-- single-GPU (no TP) comparison, rank 0 only --")
        for L in args.prompt_lens:
            bench_prefill_decode(dense, L, min(4, args.n_new_tokens), args.device, args.batch_size)
            stats = bench_prefill_decode(dense, L, args.n_new_tokens, args.device, args.batch_size)
            log(rank, f"[1-GPU dense] prompt_len={L}: {stats}")
            results.setdefault("single_gpu", {})[L] = stats

    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        ids = torch.randint(0, tp_model.cfg.vocab_size, (args.batch_size, 64), device=args.device)
        with torch.no_grad():
            _, cache = tp_model.prefill(ids)
            next_id = torch.zeros((args.batch_size, 1), dtype=torch.long, device=args.device)
            with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                for _ in range(16):
                    tp_model.step(next_id, cache)
                    sync(args.device)
        trace_path = os.path.join(args.profile_dir, f"decode_rank{rank}.json")
        prof.export_chrome_trace(trace_path)
        log(rank, f"wrote profiler trace to {trace_path}")
        log(rank, prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    if rank == 0:
        out_path = "bench_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log(rank, f"wrote {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Phase 1 benchmark: does the SSM cache actually reduce per-token decode latency?

Compares two ways of generating N new tokens after a prompt of length L:
  - "cached":   prefill once, then step() once per new token (O(1) state, no re-scan)
  - "no_cache": re-run prefill() on the whole growing sequence for every new token
                (recomputes the full scan from scratch each time -> O(L) per token)

Reports average time per generated token for both, at a few prompt lengths.
Run on CPU by default; pass --device cuda on a GPU box (Colab/Kaggle).
"""
import argparse
import time

import torch

from mamba_lite.model import MambaLM


def time_cached_decode(model, prompt_ids, n_new_tokens, device):
    ids = prompt_ids.to(device)
    with torch.no_grad():
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        _, cache = model.prefill(ids)
        next_id = torch.zeros((ids.shape[0], 1), dtype=torch.long, device=device)  # dummy fixed token
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.perf_counter()  # end of prefill, start of decode timing

        for _ in range(n_new_tokens):
            model.step(next_id, cache)
        torch.cuda.synchronize() if device == "cuda" else None
        t2 = time.perf_counter()

    prefill_s = t1 - t0
    decode_s = t2 - t1
    return prefill_s, decode_s / n_new_tokens


def time_no_cache_decode(model, prompt_ids, n_new_tokens, device):
    ids = prompt_ids.to(device)
    fixed_token = torch.zeros((ids.shape[0], 1), dtype=torch.long, device=device)
    with torch.no_grad():
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        model.prefill(ids)
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.perf_counter()

        cur = ids
        for _ in range(n_new_tokens):
            cur = torch.cat([cur, fixed_token], dim=1)
            model.prefill(cur)  # re-scan the whole (growing) sequence from scratch
        torch.cuda.synchronize() if device == "cuda" else None
        t2 = time.perf_counter()

    prefill_s = t1 - t0
    decode_s = t2 - t1
    return prefill_s, decode_s / n_new_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None, help="local HF snapshot dir; default downloads mamba-130m-hf")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--n-new-tokens", type=int, default=16)
    args = ap.parse_args()

    path = args.model_path or __import__("huggingface_hub").snapshot_download("state-spaces/mamba-130m-hf")
    model = MambaLM.from_pretrained(path).to(args.device).eval()

    print(f"{'prompt_len':>10} | {'cached ms/tok':>14} | {'no_cache ms/tok':>16} | {'speedup':>8}")
    print("-" * 60)
    for L in args.prompt_lens:
        prompt = torch.randint(0, model.cfg.vocab_size, (1, L))

        # warmup
        time_cached_decode(model, prompt, 2, args.device)
        time_no_cache_decode(model, prompt, 2, args.device)

        _, cached_ms = time_cached_decode(model, prompt, args.n_new_tokens, args.device)
        _, nocache_ms = time_no_cache_decode(model, prompt, args.n_new_tokens, args.device)
        cached_ms, nocache_ms = cached_ms * 1000, nocache_ms * 1000

        print(f"{L:>10} | {cached_ms:>14.3f} | {nocache_ms:>16.3f} | {nocache_ms / cached_ms:>7.1f}x")


if __name__ == "__main__":
    main()

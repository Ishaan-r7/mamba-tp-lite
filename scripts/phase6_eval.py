"""Whole-model quantization error on Simple English Wikipedia passages
(fp32 TP logits as the reference). Usage:
  python scripts/phase6_eval.py --n 64 --len 128 --out benchmarks/phase6/eval_global.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from mamba_lite.quant_eval import EvalPool, load_passages, summarize

CONFIGS = ["fp32", "fp16", "int8_legacy", "int8_sum", "int8_gather"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--out", default="benchmarks/phase6/eval_global.json")
    args = ap.parse_args()

    path = snapshot_download(args.model_id)
    ids = load_passages(AutoTokenizer.from_pretrained(path), args.n, args.len)
    print(f"{args.n} passages x {args.len} tokens", flush=True)

    pool = EvalPool(path, ids, batch=args.batch)
    results = {}
    try:
        for name in args.configs:
            r = pool.evaluate(name)
            s = summarize({k: v for k, v in r.items() if k != "seconds"})
            results[name] = {"per_passage": {k: v for k, v in r.items() if k != "seconds"}, "summary": s,
                             "seconds": r["seconds"]}
            print(f"{name:12s} KL {s['kl']['mean']:.5f}±{s['kl']['std']:.5f}  top1 {s['top1']['mean']*100:5.1f}%  "
                  f"top5u {s['top5_unordered']['mean']*100:5.1f}%  top5o {s['top5_ordered']['mean']*100:5.1f}%  "
                  f"({r['seconds']:.0f}s)", flush=True)
    finally:
        pool.close()
    with open(args.out, "w") as f:
        json.dump({"n": args.n, "len": args.len, "results": results}, f)


if __name__ == "__main__":
    main()

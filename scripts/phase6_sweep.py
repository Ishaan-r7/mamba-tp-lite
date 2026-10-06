"""Per-site sensitivity sweep: quantize ONE of the 48 AllReduce sites
(layer x {x_proj, out_proj}) at a time, everything else fp32, and measure
error vs. the fp32 reference. Results are saved after every config so an
interrupted run resumes where it stopped.
  python scripts/phase6_sweep.py --dtype int8 --n 64 --len 256
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from mamba_lite.model import MambaConfig
from mamba_lite.quant_eval import EvalPool, load_passages, summarize
from mamba_lite.tp_utils import SITES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="int8")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"benchmarks/phase6/sweep_{args.dtype}.json"

    path = snapshot_download(args.model_id)
    n_layer = MambaConfig.from_hf_config(f"{path}/config.json").n_layer
    done = json.load(open(out)) if os.path.exists(out) else {"n": args.n, "len": args.len, "dtype": args.dtype, "sites": {}}
    assert (done["n"], done["len"], done["dtype"]) == (args.n, args.len, args.dtype), "existing file used other settings"

    ids = load_passages(AutoTokenizer.from_pretrained(path), args.n, args.len)
    pool = EvalPool(path, ids, batch=args.batch)
    try:
        for layer in range(n_layer):
            for site in SITES:
                key = f"{layer}:{site}"
                if key in done["sites"]:
                    continue
                r = pool.evaluate({(layer, site): args.dtype})
                per = {k: v for k, v in r.items() if k != "seconds"}
                done["sites"][key] = {"summary": summarize(per), "seconds": r["seconds"]}
                s = done["sites"][key]["summary"]
                print(f"{key:14s} KL {s['kl']['mean']:.6f}  top1 {s['top1']['mean']*100:5.2f}%  "
                      f"top5o {s['top5_ordered']['mean']*100:5.2f}%  ({r['seconds']:.0f}s)", flush=True)
                json.dump(done, open(out, "w"))
    finally:
        pool.close()


if __name__ == "__main__":
    main()

"""Mixed-precision search using the single-site INT8 sensitivities.

1. Pareto: put the k least-sensitive sites in INT8 (k = 0..48), the rest in
   fp32 (curve A) or fp16 (curve B); every combination is measured directly,
   since single-site errors don't add up exactly.
2. Greedy backward: start from all-INT8 and repeatedly move to fp16 the site
   (among the most sensitive candidates) whose move lowers KL the most.
Results are cached per config so an interrupted run resumes.
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

KS = [0, 8, 16, 24, 32, 36, 40, 42, 44, 45, 46, 47, 48]


def key_of(spec: dict) -> str:
    return json.dumps(sorted([f"{l}:{s}={d}" for (l, s), d in spec.items()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--greedy-candidates", type=int, default=10)
    ap.add_argument("--greedy-steps", type=int, default=6)
    ap.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--sweep", default="benchmarks/phase6/sweep_int8.json")
    ap.add_argument("--out", default="benchmarks/phase6/pareto.json")
    args = ap.parse_args()

    path = snapshot_download(args.model_id)
    n_layer = MambaConfig.from_hf_config(f"{path}/config.json").n_layer
    all_sites = [(l, s) for l in range(n_layer) for s in SITES]

    sweep = json.load(open(args.sweep))["sites"]
    single = {(int(k.split(":")[0]), k.split(":")[1]): v["summary"]["kl"]["mean"] for k, v in sweep.items()}
    order = sorted(all_sites, key=lambda x: single[x])  # least sensitive first

    state = json.load(open(args.out)) if os.path.exists(args.out) else {
        "n": args.n, "len": args.len, "order": [f"{l}:{s}" for l, s in order], "evals": {},
        "pareto_fp32_rest": [], "pareto_fp16_rest": [], "greedy": []}
    assert (state["n"], state["len"]) == (args.n, args.len)

    ids = load_passages(AutoTokenizer.from_pretrained(path), args.n, args.len)
    pool = EvalPool(path, ids, batch=args.batch)

    def ev(spec: dict, label: str):
        k = key_of(spec)
        if k not in state["evals"]:
            r = pool.evaluate(spec)
            state["evals"][k] = {"summary": summarize({a: b for a, b in r.items() if a != "seconds"}),
                                 "seconds": r["seconds"]}
            json.dump(state, open(args.out, "w"))
        s = state["evals"][k]["summary"]
        print(f"{label:34s} KL {s['kl']['mean']:.6f}  top1 {s['top1']['mean']*100:5.2f}%  "
              f"top5o {s['top5_ordered']['mean']*100:5.2f}%", flush=True)
        return s

    try:
        for rest, name in (("fp32", "pareto_fp32_rest"), ("fp16", "pareto_fp16_rest")):
            state[name] = []
            for k in KS:
                spec = {site: "int8" for site in order[:k]}
                if rest == "fp16":
                    spec.update({site: "fp16" for site in order[k:]})
                s = ev(spec, f"pareto rest={rest} k_int8={k}")
                state[name].append({"k": k, "summary": s})
            json.dump(state, open(args.out, "w"))

        cands = order[::-1][: args.greedy_candidates]  # most sensitive first
        spec = {site: "int8" for site in all_sites}
        cur = ev(spec, "greedy start: all int8")
        state["greedy"] = [{"moved": None, "summary": cur}]
        for step in range(args.greedy_steps):
            best = None
            for c in cands:
                if spec[c] == "fp16":
                    continue
                trial = dict(spec)
                trial[c] = "fp16"
                s = ev(trial, f"greedy step {step + 1} try {c[0]}:{c[1]}")
                if best is None or s["kl"]["mean"] < best[1]["kl"]["mean"]:
                    best = (c, s)
            spec[best[0]] = "fp16"
            state["greedy"].append({"moved": f"{best[0][0]}:{best[0][1]}", "summary": best[1]})
            print(f"==> step {step + 1}: moved {best[0][0]}:{best[0][1]} to fp16, KL {best[1]['kl']['mean']:.6f}", flush=True)
            json.dump(state, open(args.out, "w"))
    finally:
        pool.close()


if __name__ == "__main__":
    main()

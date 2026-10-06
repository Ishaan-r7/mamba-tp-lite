"""
Generates the static charts embedded in README.md, from the real numbers
recorded in NOTES.md / benchmarks/tensor_parallel_results.md / quantized_allreduce_initial_results.md.
Not a live benchmark runner -- just plots already-measured results so the
README doesn't need a notebook to render.

Run: .venv/bin/python scripts/make_plots.py
Writes PNGs into assets/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.8,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

COLOR_CACHED = "#2E7D32"
COLOR_NOCACHE = "#C62828"
COLOR_TP = "#1565C0"
COLOR_1GPU = "#EF6C00"
COLOR_FP16 = "#6A1B9A"
COLOR_INT8 = "#AD1457"


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Phase 1: SSM cache -- ms/token, cached vs from-scratch-every-token (CPU)
# ---------------------------------------------------------------------------
prompt_lens = [16, 64, 256]
cached_ms = [12.067, 12.352, 12.244]
nocache_ms = [466.642, 559.176, 898.936]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(prompt_lens))
w = 0.35
ax.bar(x - w / 2, cached_ms, w, label="with cache", color=COLOR_CACHED)
ax.bar(x + w / 2, nocache_ms, w, label="no cache (rescans prefix)", color=COLOR_NOCACHE)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in prompt_lens])
ax.set_xlabel("prompt length (tokens)")
ax.set_ylabel("ms / generated token (log scale)")
ax.set_title("Phase 1: SSM cache keeps decode O(1), not O(prompt length)")
for xi, c, n in zip(x, cached_ms, nocache_ms):
    ax.annotate(f"{n/c:.0f}x", (xi, n), ha="center", va="bottom", fontsize=10, color="#333")
ax.legend()
savefig(fig, "cache_speedup.png")

# ---------------------------------------------------------------------------
# 2. Phase 3: decode throughput, TP vs 1-GPU (mamba-130m, batch=1, Kaggle 2xT4)
# ---------------------------------------------------------------------------
prompt_lens = [16, 64, 256]
tp_decode = [37.46, 35.13, 36.18]  # avg of both ranks
gpu1_decode = [48.62, 47.78, 47.34]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(prompt_lens))
ax.bar(x - w / 2, tp_decode, w, label="TP (2xT4)", color=COLOR_TP)
ax.bar(x + w / 2, gpu1_decode, w, label="1 GPU (no TP)", color=COLOR_1GPU)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in prompt_lens])
ax.set_xlabel("prompt length (tokens)")
ax.set_ylabel("decode tokens / sec")
ax.set_title("Phase 3: single-token decode -- TP loses (comm-bound)")
ax.legend()
savefig(fig, "tp_decode_vs_1gpu.png")

# ---------------------------------------------------------------------------
# 3. Phase 3: prefill throughput crossover as batch size grows
# ---------------------------------------------------------------------------
batches = [1, 8, 32]
tp_prefill_256 = [361.4, 2836.8, 7049.8]
gpu1_prefill_256 = [395.1, 3006.0, 4150.0]
tp_prefill_1024 = [371.25, 2813.9, 6492.1]
gpu1_prefill_1024 = [395.6, 2935.9, 3625.9]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, tp_vals, gpu_vals, L in zip(axes, [tp_prefill_256, tp_prefill_1024], [gpu1_prefill_256, gpu1_prefill_1024], [256, 1024]):
    ax.plot(batches, tp_vals, "o-", label="TP (2xT4)", color=COLOR_TP, linewidth=2.5, markersize=7)
    ax.plot(batches, gpu_vals, "s-", label="1 GPU (no TP)", color=COLOR_1GPU, linewidth=2.5, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("batch size")
    ax.set_title(f"prompt_len={L}")
    ax.legend()
axes[0].set_ylabel("prefill tokens / sec")
fig.suptitle("Phase 3: prefill throughput -- TP wins once batch saturates a single GPU", y=1.02, fontweight="bold")
savefig(fig, "tp_prefill_crossover.png")

# ---------------------------------------------------------------------------
# 4. Phase 4: quantization accuracy table (Top-1 / Top-5 unordered / ordered)
# ---------------------------------------------------------------------------
metrics = ["Top-1", "Top-5\n(unordered)", "Top-5\n(ordered)"]
fp16_vals = [100.00, 100.00, 97.62]
int8_vals = [64.29, 66.19, 2.38]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(metrics))
ax.bar(x - w / 2, fp16_vals, w, label="fp16 AllReduce", color=COLOR_FP16)
ax.bar(x + w / 2, int8_vals, w, label="int8 AllReduce", color=COLOR_INT8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("agreement with fp32 (%)")
ax.set_ylim(0, 108)
ax.set_title("Phase 4: quantized AllReduce accuracy (mamba-130m)")
for xi, v in zip(x - w / 2, fp16_vals):
    ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=9)
for xi, v in zip(x + w / 2, int8_vals):
    ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=9)
ax.legend(loc="lower left")
savefig(fig, "quant_initial_accuracy.png")

# ---------------------------------------------------------------------------
# 5. Phase 4: fp16 AllReduce speed change, same-session controlled
# ---------------------------------------------------------------------------
labels = ["decode\nlen=64", "decode\nlen=256", "decode\nlen=1024", "prefill\nbatch=32,len=1024"]
pct_change = [-12.3, -9.4, -16.3, +6.0]
colors = [COLOR_INT8 if v < 0 else COLOR_CACHED for v in pct_change]

fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(labels, pct_change, color=colors)
ax.axhline(0, color="#333", linewidth=1)
ax.set_ylabel("% change vs fp32 AllReduce\n(same Kaggle session, controlled)")
ax.set_title("Phase 4: fp16 AllReduce helps compute-bound prefill,\nhurts latency-bound decode", pad=14)
for b, v in zip(bars, pct_change):
    ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom" if v > 0 else "top", fontsize=10, fontweight="bold")
savefig(fig, "quant_initial_fp16_speed.png")

# ---------------------------------------------------------------------------
# 6. Phase 5: CUDA graph decode speedup, verified correct (same-session)
# ---------------------------------------------------------------------------
prompt_lens_labels = ["64", "256", "1024"]
tp_eager = [37.8, 36.3, 37.6]
tp_graphed = [112.5, 133.0, 122.6]
gpu1_eager = [55.3, 55.3, 52.2]
gpu1_graphed = [175.2, 137.0, 145.3]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, eager, graphed, title in zip(axes, [tp_eager, gpu1_eager], [tp_graphed, gpu1_graphed], ["TP (2xT4)", "1 GPU (no TP)"]):
    x = np.arange(len(prompt_lens_labels))
    ax.bar(x - w / 2, eager, w, label="eager", color=COLOR_1GPU)
    ax.bar(x + w / 2, graphed, w, label="CUDA graph", color=COLOR_CACHED)
    for xi, e, g in zip(x, eager, graphed):
        ax.annotate(f"{g/e:.2f}x", (xi, g), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_lens_labels)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_title(title)
    ax.legend()
axes[0].set_ylabel("decode tokens / sec")
fig.suptitle("Phase 5: CUDA graphs speed up decode ~2.5-3.7x\n(verified bit-exact correct, not just faster)", y=1.06, fontweight="bold")
savefig(fig, "cuda_graph_speedup.png")

print("done")


# ---------------------------------------------------------------------------
# 7. Phase 6: per-site INT8 sensitivity (48 collectives), Pareto, GPU speed
# ---------------------------------------------------------------------------
import json

P6 = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "quant_mixed")
COLOR_FP32 = "#616161"
COLOR_MIXED = "#1565C0"


def _load6(name):
    with open(os.path.join(P6, name)) as f:
        return json.load(f)


sweep = _load6("sweep_int8.json")["sites"]
items = sorted(((k, v["summary"]["kl"]["mean"]) for k, v in sweep.items()), key=lambda kv: -kv[1])
fig, ax = plt.subplots(figsize=(11, 4.4))
colors = [COLOR_INT8 if k.endswith("out_proj") else COLOR_TP for k, _ in items]
ax.bar(range(len(items)), [v for _, v in items], color=colors)
ax.set_yscale("log")
ax.set_xticks(range(len(items)))
ax.set_xticklabels([f"L{k.split(':')[0]} {'out' if k.endswith('out_proj') else 'x'}" for k, _ in items],
                   rotation=90, fontsize=7.5)
ax.set_ylabel("KL(fp32 || INT8 at this site only)")
ax.set_title("Single-site INT8 damage: 48 AllReduce collectives, sorted (mamba-130m)")
ax.bar([0], [0], color=COLOR_INT8, label="out_proj (AllReduce #2)")
ax.bar([0], [0], color=COLOR_TP, label="x_proj (AllReduce #1)")
ax.legend(loc="upper right")
ax.set_xlim(-0.8, len(items) - 0.2)
savefig(fig, "quant_site_sensitivity.png")

par = _load6("pareto.json")
fig, ax = plt.subplots(figsize=(7, 4.4))
for key, label, col in (("pareto_fp32_rest", "rest of sites fp32", COLOR_FP32),
                        ("pareto_fp16_rest", "rest of sites fp16", COLOR_FP16)):
    pts = [(p["k"], p["summary"]["kl"]["mean"]) for p in par[key] if p["k"] > 0]
    ax.plot([a for a, _ in pts], [b for _, b in pts], marker="o", ms=4, color=col, label=label, lw=1.6)
g = par["greedy"]
ax.plot([48 - i for i in range(len(g))], [x["summary"]["kl"]["mean"] for x in g], marker="s", ms=6,
        color=COLOR_MIXED, lw=1.6, label="greedy: move most-damaging site to fp16")
ax.annotate("all 48 INT8", (48, g[0]["summary"]["kl"]["mean"]), xytext=(38.5, 1.0e-2), textcoords="data",
            fontsize=10, va="center", arrowprops=dict(arrowstyle="-", color="#888"))
ax.annotate("43 INT8 + 5 fp16\n(mixed_5fp16)", (43, g[5]["summary"]["kl"]["mean"]), xytext=(33, 6e-5),
            textcoords="data", fontsize=10, va="center", arrowprops=dict(arrowstyle="-", color="#888"))
ax.set_yscale("log")
ax.set_ylim(7e-6, 3e-2)
ax.set_xlabel("number of collectives in INT8 (of 48)")
ax.set_ylabel("KL vs fp32 logits")
ax.set_title("Mixed precision: error vs INT8 coverage")
ax.legend(fontsize=9, loc="upper left")
savefig(fig, "quant_pareto.png")

gpu = _load6("gpu_results.json")["summary"]
names = ["fp32", "fp16", "int8", "mixed_1fp16", "mixed_5fp16"]
labels = ["fp32", "fp16", "INT8\n(all 48)", "mixed\n1 fp16", "mixed\n5 fp16"]
base = gpu["fp32"]
series = [
    ("prefill, batch 32 x 1024", [gpu[n]["prefill_tok_s_median"] / base["prefill_tok_s_median"] for n in names], COLOR_TP),
    ("decode, eager", [base["decode_eager_ms_median"] / gpu[n]["decode_eager_ms_median"] for n in names], COLOR_1GPU),
    ("decode, CUDA graph", [base["decode_graph_ms_median"] / gpu[n]["decode_graph_ms_median"] for n in names], COLOR_CACHED),
]
fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(len(names))
w = 0.26
for i, (lab, vals, col) in enumerate(series):
    ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(1.0, color="#444", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("speed vs fp32 (>1 = faster)")
ax.set_title("AllReduce precision on 2x T4, same session (mamba-130m)")
ax.set_ylim(0, 1.25)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=9, frameon=False)
savefig(fig, "quant_gpu_speed.png")

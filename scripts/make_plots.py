"""
Generates the static charts embedded in README.md, from the real numbers
recorded in NOTES.md / benchmarks/PHASE3_RESULTS.md / PHASE4_RESULTS.md.
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
savefig(fig, "phase1_cache_speedup.png")

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
savefig(fig, "phase3_decode_tp_vs_1gpu.png")

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
savefig(fig, "phase3_prefill_crossover.png")

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
savefig(fig, "phase4_quant_accuracy.png")

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
savefig(fig, "phase4_fp16_speed_change.png")

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
savefig(fig, "phase5_cuda_graph_speedup.png")

print("done")

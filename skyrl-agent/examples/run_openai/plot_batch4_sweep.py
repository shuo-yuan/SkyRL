"""
Plot the batch4 pool sweep results.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs/bfcl_pool_batch4_sweep"
POOL_SIZES = [4, 64, 128, 256, 512]

# Load summary
summary_file = Path(OUTPUT_DIR) / "pool_batch4_sweep_summary.json"
with open(summary_file) as f:
    summary = json.load(f)

# Convert pool sizes to int keys
by_pool = {int(k): v for k, v in summary.items()}

# Plot configuration
plot_configs = [
    ("In-Context",  "#4CAF50", "o",  "-"),
    ("TST-BM25-k1", "#E91E63", "v",  "--"),
    ("TST-BM25-k2", "#9C27B0", "D",  "--"),
    ("TST-BM25-k4", "#673AB7", "s",  "--"),
    ("TST-LLM",     "#2196F3", "P",  "-"),
    ("TST-GT",      "#FF5722", "^",  "-"),
]

fig, ax = plt.subplots(figsize=(11, 6))

for lbl, color, marker, ls in plot_configs:
    xs = [ps for ps in POOL_SIZES if lbl in by_pool.get(ps, {})]
    ys = [by_pool[ps][lbl] for ps in xs]
    if not xs:
        continue
    ax.plot(xs, ys, marker=marker, linewidth=2.2, markersize=8,
            color=color, label=lbl, linestyle=ls)
    ax.fill_between(xs, ys, alpha=0.05, color=color)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color=color, fontweight="bold")

ax.set_xscale("log")
ax.set_xticks(POOL_SIZES)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("Tool pool size", fontsize=13)
ax.set_ylabel("Accuracy (avg reward)", fontsize=13)
ax.set_title("Batch-4 Accuracy vs. Tool Pool Size", fontsize=14, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(fontsize=10, loc="lower left")
fig.tight_layout()

plot_file = Path(OUTPUT_DIR) / "pool_batch4_sweep_plot.png"
fig.savefig(plot_file, dpi=150)
plt.close(fig)
print(f"Plot saved → {plot_file}")

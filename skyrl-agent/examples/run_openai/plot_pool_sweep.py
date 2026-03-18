"""
Plot the pool sweep results from individual result files.
Aggregates all conditions: GT, TST-BM25-k1/2/4, TST-LLM, TST-LLM-mini, In-Context
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

OUTPUT_DIR = "outputs/bfcl_pool_sweep"
POOL_SIZES = [1, 64, 128, 256, 512]

# Load all result files and aggregate
by_pool = defaultdict(dict)

for pool_size in POOL_SIZES:
    # Map of label -> filename suffix
    label_to_suffix = {
        "In-Context": "incontext",
        "TST-GT": "tst_gt",
        "TST-BM25-k1": "tst_bm25_k1",
        "TST-BM25-k2": "tst_bm25_k2",
        "TST-BM25-k4": "tst_bm25",  # Note: k4 is stored as tst_bm25
        "TST-LLM": "tst_llm",
        "TST-LLM-mini": "tst_llm_mini",
    }
    
    for label, suffix in label_to_suffix.items():
        result_file = Path(OUTPUT_DIR) / f"pool{pool_size}_{suffix}_results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                avg_reward = data.get("avg_reward", 0.0)
                by_pool[pool_size][label] = avg_reward

# Plot configuration - matching plot_batch4_sweep.py style
plot_configs = [
    ("In-Context",  "#4CAF50", "o",  "-"),
    ("TST-BM25-k1", "#E91E63", "v",  "--"),
    ("TST-BM25-k2", "#9C27B0", "D",  "--"),
    ("TST-BM25-k4", "#673AB7", "s",  "--"),
    ("TST-LLM",     "#2196F3", "P",  "-"),
    ("TST-LLM-mini", "#00BCD4", "X", "-"),
    ("TST-GT",      "#FF5722", "^",  "-"),
]

fig, ax = plt.subplots(figsize=(12, 7))

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
ax.set_title("Accuracy vs. Tool Pool Size (All Conditions)", fontsize=14, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(fontsize=9, loc="best", ncol=2)
fig.tight_layout()

plot_file = Path(OUTPUT_DIR) / "pool_sweep_plot.png"
fig.savefig(plot_file, dpi=150)
plt.close(fig)
print(f"Plot saved → {plot_file}")

# Also save updated summary
summary = {str(ps): by_pool[ps] for ps in POOL_SIZES}
summary_file = Path(OUTPUT_DIR) / "pool_sweep_summary.json"
summary_file.write_text(json.dumps(summary, indent=2))
print(f"Summary saved → {summary_file}")

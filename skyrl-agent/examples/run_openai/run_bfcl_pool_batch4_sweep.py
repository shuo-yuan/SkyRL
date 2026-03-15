"""
Pool-size sweep with batch_size=4.

For each (pool_size, method):
  - Slide a window of pool_size tasks over the dataset
  - Run the FIRST 4 tasks as a batch sharing the pool_size-task tool pool
  - Evaluate all 4 tasks and average accuracy

Methods: In-Context, TST-BM25-k1/k2/k4, TST-LLM, TST-GT

Usage:
    cd /home/ec2-user/shuo/SkyRL/skyrl-agent
    uv run --env-file .env examples/run_openai/run_bfcl_pool_batch4_sweep.py
"""

POOL_SIZES   = [4, 64, 128, 256, 512]
BATCH_SIZE   = 4
MAX_PARALLEL = 200
OUTPUT_DIR   = "outputs/bfcl_pool_batch4_sweep"

import asyncio, datasets, json, os, time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import skyrl_agent.tools.bfcl_tool_search

from skyrl_agent.agents.react.react_agent import ReActAgent
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.openai import OpenAIAPIBackend, OpenAIAPIBackendConfig
from skyrl_agent.tasks.bfcl_eval_task import (
    BFCLEvalTask, _normalise_function_list, _parse_json_field,
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def combine_tools(insts):
    seen, out = set(), []
    for inst in insts:
        for fn in _normalise_function_list(_parse_json_field(inst.get("function", []))):
            n = fn.get("name", "")
            if n and n not in seen:
                seen.add(n); p = dict(fn.get("parameters", {}))
                if p.get("type") == "dict": p["type"] = "object"
                out.append({"type": "function", "function": {
                    "name": n, "description": fn.get("description", ""), "parameters": p,
                }})
    return out


def make_agent(idx, use_search, fold=False):
    tools = ["bfcl_tool_search"] if use_search else []
    cfg = TrajectoryConfig(
        instance_id=idx, trajectory_id=0,
        sampling_params={"temperature": 1.0, "max_tokens": 32768},
        max_prompt_length=32768,
        qwen3_enable_thinking=False, qwen3_acc_thinking=False,
        tools=tools,
        max_iterations=10 * BATCH_SIZE,
        agent_cls="skyrl_agent.agents.react.ReActAgent",
        enable_turn_reminder=False, early_step_threshold=0,
        fold_tool_info=fold, debug_log=False, profile_tools=False,
    )
    engine = OpenAIAPIBackend(infer_engine=None, cfg=OpenAIAPIBackendConfig(
        model_name="gpt-5-nano", api_url="https://api.openai.com",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    ))
    return ReActAgent(traj_config=cfg, infer_engine=engine, tokenizer=None)


async def run_one_batch(sem, batch_insts, pool_insts, batch_idx,
                        use_search, retrieval, bm25_k):
    async with sem:
        combined = combine_tools(pool_insts)
        tool_search_mode = use_search
        instructions = [BFCLEvalTask.get_instruction(inst, tool_search_mode=tool_search_mode)
                        for inst in batch_insts]
        agent = make_agent(batch_idx, use_search)
        if not use_search:
            agent._active_bfcl_tool_params = combined
        try:
            _, per_calls = await agent.run_batch(
                instructions=instructions,
                instances=batch_insts,
                combined_tool_params=combined,
                use_tool_search=use_search,
                tool_search_k=3,
                bm25_k=bm25_k,
                tool_search_retrieval=retrieval,
                tool_search_llm_model="gpt-5-nano",
            )
        except Exception as e:
            return [0.0] * len(batch_insts)

        rewards = []
        for calls, inst in zip(per_calls, batch_insts):
            try:
                r = await BFCLEvalTask.evaluate_result(
                    result=calls, instance=inst,
                    data_source="", instance_id=inst.get("id", ""), trajectory_id=0,
                )
            except Exception:
                r = 0.0
            rewards.append(float(r))
        return rewards


async def run_condition(all_inst, pool_size, use_search, retrieval, bm25_k, label, sem):
    total = len(all_inst)
    print(f"  [{label} pool={pool_size}] starting …")
    t0 = time.time()

    # For each starting index i, pool = [i..i+pool_size), batch = [i..i+BATCH_SIZE)
    coros = []
    batch_starts = list(range(0, total, BATCH_SIZE))
    for start in batch_starts:
        batch = [all_inst[(start + j) % total] for j in range(BATCH_SIZE)]
        pool  = [all_inst[(start + j) % total] for j in range(pool_size)]
        coros.append(run_one_batch(sem, batch, pool, start // BATCH_SIZE,
                                   use_search, retrieval, bm25_k))

    batch_rewards = list(await asyncio.gather(*coros))
    all_rewards = [r for batch in batch_rewards for r in batch]

    elapsed = time.time() - t0
    avg = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    cat_rewards: Dict[str, List[float]] = defaultdict(list)
    for i, r in enumerate(all_rewards):
        inst = all_inst[i % total]
        cat = str(inst.get("id", "")).rsplit("_", 1)[0] or "unknown"
        cat_rewards[cat].append(r)

    print(f"  [{label} pool={pool_size}] avg={avg:.4f} "
          f"({sum(all_rewards):.0f}/{len(all_rewards)}) {elapsed:.1f}s")

    result = {
        "mode": label, "pool_size": pool_size, "batch_size": BATCH_SIZE,
        "retrieval": retrieval, "bm25_k": bm25_k if retrieval == "bm25" else None,
        "avg_reward": avg, "n_tasks": len(all_rewards),
        "per_category": {c: {"avg": sum(v)/len(v), "count": len(v)}
                         for c, v in cat_rewards.items()},
    }
    suffix_map = {
        "In-Context":  "incontext",
        "TST-BM25-k1": "tst_bm25_k1",
        "TST-BM25-k2": "tst_bm25_k2",
        "TST-BM25-k4": "tst_bm25_k4",
        "TST-LLM":     "tst_llm",
        "TST-GT":      "tst_gt",
    }
    suffix = suffix_map.get(label, label.lower().replace(" ", "_").replace("-", "_"))
    fname = Path(OUTPUT_DIR) / f"pool{pool_size}_{suffix}_results.json"
    fname.write_text(json.dumps(result, indent=2))
    return pool_size, label, avg


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    print(f"Loading dataset …")
    ds = datasets.load_dataset("parquet", data_files="data/BFCL_single_turn.parquet")["train"]
    all_inst = [dict(r) for r in ds]
    total = len(all_inst)
    print(f"Total tasks: {total}  |  batch_size: {BATCH_SIZE}  |  pool sizes: {POOL_SIZES}")

    sem = asyncio.Semaphore(MAX_PARALLEL)
    t_start = time.time()

    # (use_search, retrieval, bm25_k, label)
    conditions = [
        (False, "bm25",         4, "In-Context"),
        (True,  "bm25",         1, "TST-BM25-k1"),
        (True,  "bm25",         2, "TST-BM25-k2"),
        (True,  "bm25",         4, "TST-BM25-k4"),
        (True,  "llm",          4, "TST-LLM"),
        (True,  "ground_truth", 4, "TST-GT"),
    ]

    coros = [
        run_condition(all_inst, ps, use_search, retrieval, bm25_k, label, sem)
        for ps in POOL_SIZES
        for use_search, retrieval, bm25_k, label in conditions
    ]
    results = list(await asyncio.gather(*coros))
    total_elapsed = time.time() - t_start

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"batch_size={BATCH_SIZE}  wall-clock: {total_elapsed:.0f}s")
    print(f"{'='*70}")
    labels = [c[3] for c in conditions]
    print(f"{'Pool':>6}  " + "  ".join(f"{l:>14}" for l in labels))
    print("-" * (8 + 16 * len(labels)))
    by_pool: Dict[int, Dict[str, float]] = defaultdict(dict)
    for ps, lbl, avg in results:
        by_pool[ps][lbl] = avg
    for ps in POOL_SIZES:
        row = "  ".join(f"{by_pool[ps].get(l, float('nan')):>14.4f}" for l in labels)
        print(f"  {ps:>4}  {row}")

    summary = {str(ps): {l: by_pool[ps].get(l) for l in labels} for ps in POOL_SIZES}
    summary_file = Path(OUTPUT_DIR) / "pool_batch4_sweep_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → {summary_file}")

    # ── Plot ───────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
            xs = [ps for ps in POOL_SIZES if lbl in by_pool[ps]]
            ys = [by_pool[ps][lbl] for ps in xs]
            if not xs: continue
            ax.plot(xs, ys, marker=marker, linewidth=2.2, markersize=8,
                    color=color, label=lbl, linestyle=ls)
            ax.fill_between(xs, ys, alpha=0.05, color=color)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 9),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=color, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(POOL_SIZES); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Tool pool size", fontsize=13)
        ax.set_ylabel("Accuracy (avg reward)", fontsize=13)
        ax.set_title(f"Batch-{BATCH_SIZE} Accuracy vs. Tool Pool Size", fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=10, loc="lower left")
        fig.tight_layout()
        plot_file = Path(OUTPUT_DIR) / "pool_batch4_sweep_plot.png"
        fig.savefig(plot_file, dpi=150); plt.close(fig)
        print(f"Plot saved → {plot_file}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())

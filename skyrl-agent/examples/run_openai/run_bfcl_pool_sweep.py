"""
Sweep over tool-pool sizes [1, 4, 16, 64, 256] for two conditions:
  - In-Context: all pool tools in the system prompt
  - TST (LLM):  only bfcl_tool_search in system prompt; LLM retrieval

Each task is evaluated as the "first task" in a sliding-window pool of
POOL_SIZE tools (wraps around at the end of the dataset).

Usage:
    cd /home/ec2-user/shuo/SkyRL/skyrl-agent
    uv run --env-file .env examples/run_openai/run_bfcl_pool_sweep.py
"""

# ─── configure ───────────────────────────────────────────────────
POOL_SIZES    = [1, 64, 128, 256, 512]
MAX_PARALLEL  = 200          # concurrent API calls shared across all runs
BM25_K        = 4            # k for BM25 retrieval (number of tools returned)
OUTPUT_DIR    = "outputs/bfcl_pool_sweep"
# ─────────────────────────────────────────────────────────────────

import asyncio, datasets, json, os, time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import skyrl_agent.tools.bfcl_tool_search          # force-register bfcl_tool_search

from skyrl_agent.agents.react.react_agent import ReActAgent
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.openai import OpenAIAPIBackend, OpenAIAPIBackendConfig
from skyrl_agent.tasks.bfcl_eval_task import (
    BFCLEvalTask, _normalise_function_list, _parse_json_field,
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────

def combine_tools(insts: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for inst in insts:
        for fn in _normalise_function_list(_parse_json_field(inst.get("function", []))):
            n = fn.get("name", "")
            if n and n not in seen:
                seen.add(n)
                p = dict(fn.get("parameters", {}))
                if p.get("type") == "dict":
                    p["type"] = "object"
                out.append({"type": "function", "function": {
                    "name": n, "description": fn.get("description", ""),
                    "parameters": p,
                }})
    return out


def make_agent(idx: int, use_search: bool) -> ReActAgent:
    tools = ["bfcl_tool_search"] if use_search else []
    cfg = TrajectoryConfig(
        instance_id=idx, trajectory_id=0,
        sampling_params={"temperature": 1.0, "max_tokens": 32768},
        max_prompt_length=32768,
        qwen3_enable_thinking=False, qwen3_acc_thinking=False,
        tools=tools, max_iterations=10,
        agent_cls="skyrl_agent.agents.react.ReActAgent",
        enable_turn_reminder=False, early_step_threshold=0,
        fold_tool_info=False, debug_log=False, profile_tools=False,
    )
    engine = OpenAIAPIBackend(
        infer_engine=None,
        cfg=OpenAIAPIBackendConfig(
            model_name="gpt-5-nano",
            api_url="https://api.openai.com",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    )
    return ReActAgent(traj_config=cfg, infer_engine=engine, tokenizer=None)


async def run_one(sem, inst, pool, idx, use_search, retrieval="llm"):
    async with sem:
        combined = combine_tools(pool)
        instr = BFCLEvalTask.get_instruction(inst, tool_search_mode=use_search)
        agent = make_agent(idx, use_search)
        if not use_search:
            agent._active_bfcl_tool_params = combined  # upfront for In-Context
        try:
            _, per_calls = await agent.run_batch(
                instructions=[instr], instances=[inst],
                combined_tool_params=combined,
                use_tool_search=use_search,
                tool_search_k=3,
                bm25_k=BM25_K,
                tool_search_retrieval=retrieval,
                tool_search_llm_model="gpt-5-nano",
            )
        except Exception as e:
            return 0.0
        calls = per_calls[0] if per_calls else []
        try:
            r = await BFCLEvalTask.evaluate_result(
                result=calls, instance=inst,
                data_source="", instance_id=inst.get("id", ""), trajectory_id=0,
            )
        except Exception:
            r = 0.0
        return float(r)


async def run_condition(all_inst, pool_size, use_search, sem,
                        retrieval="llm", label=None):
    if label is None:
        label = "TST-LLM" if use_search else "In-Context"
    total = len(all_inst)
    print(f"  [{label} pool={pool_size}] starting {total} tasks …")
    t0 = time.time()

    coros = [
        run_one(
            sem, all_inst[i],
            [all_inst[(i + j) % total] for j in range(pool_size)],
            i, use_search, retrieval,
        )
        for i in range(total)
    ]
    rewards = list(await asyncio.gather(*coros))
    elapsed = time.time() - t0

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    cat_rewards: Dict[str, List[float]] = defaultdict(list)
    for i, r in enumerate(rewards):
        cat = str(all_inst[i].get("id", "")).rsplit("_", 1)[0] or "unknown"
        cat_rewards[cat].append(r)

    print(f"  [{label} pool={pool_size}] avg={avg:.4f}  "
          f"({sum(rewards):.0f}/{len(rewards)})  {elapsed:.1f}s")

    result = {
        "mode": label, "pool_size": pool_size,
        "retrieval": retrieval,
        "avg_reward": avg, "n_tasks": len(rewards),
        "per_category": {
            c: {"avg": sum(v) / len(v), "count": len(v)}
            for c, v in cat_rewards.items()
        },
    }
    suffix = {"In-Context": "incontext", "TST-LLM": "tst_llm",
              "TST-GT": "tst_gt"}.get(label, label.lower().replace(" ", "_"))
    fname = Path(OUTPUT_DIR) / f"pool{pool_size}_{suffix}_results.json"
    fname.write_text(json.dumps(result, indent=2))
    return pool_size, label, avg


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    print("Loading dataset …")
    ds = datasets.load_dataset("parquet", data_files="data/BFCL_single_turn.parquet")["train"]
    all_inst = [dict(r) for r in ds]
    total = len(all_inst)
    print(f"Total tasks: {total}  |  pool sizes: {POOL_SIZES}  |  max_parallel: {MAX_PARALLEL}")

    # Shared semaphore across ALL concurrent runs
    sem = asyncio.Semaphore(MAX_PARALLEL)
    t_start = time.time()

    # Launch all (pool_size × condition) combos concurrently
    # conditions: (use_search, retrieval_method, label_suffix)
    conditions = [
        (False, "bm25",         "In-Context"),
        (True,  "llm",          "TST-LLM"),
        (True,  "ground_truth", "TST-GT"),
    ]
    coros = [
        run_condition(all_inst, ps, use_search, sem, retrieval, label)
        for ps in POOL_SIZES
        for use_search, retrieval, label in conditions
    ]
    results = await asyncio.gather(*coros)
    total_elapsed = time.time() - t_start

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Summary  (total wall-clock: {total_elapsed:.0f}s)")
    print(f"{'='*55}")
    print(f"{'Pool':>6}  {'In-Context':>12}  {'TST-LLM':>12}  {'TST-GT':>10}")
    print("-" * 48)
    by_pool: Dict[int, Dict[str, float]] = defaultdict(dict)
    for ps, label, avg in results:
        by_pool[ps][label] = avg
    for ps in POOL_SIZES:
        ic  = by_pool[ps].get("In-Context", float("nan"))
        tst = by_pool[ps].get("TST-LLM",    float("nan"))
        gt  = by_pool[ps].get("TST-GT",      float("nan"))
        print(f"  {ps:>4}  {ic:>12.4f}  {tst:>12.4f}  {gt:>10.4f}")

    # ── Save combined summary ──────────────────────────────────────
    summary = {
        str(ps): {
            "In-Context": by_pool[ps].get("In-Context"),
            "TST-LLM":    by_pool[ps].get("TST-LLM"),
        }
        for ps in POOL_SIZES
    }
    summary_file = Path(OUTPUT_DIR) / "pool_sweep_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → {summary_file}")

    # ── Plot ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        configs = [
            ("In-Context", "#4CAF50", "o"),
            ("TST-LLM",    "#2196F3", "s"),
            ("TST-GT",     "#FF5722", "^"),
        ]
        for label, color, marker in configs:
            xs = [ps for ps in POOL_SIZES if label in by_pool[ps]]
            ys = [by_pool[ps][label] for ps in xs]
            ax.plot(xs, ys, marker=marker, linewidth=2.5, markersize=9,
                    color=color, label=label)
            ax.fill_between(xs, ys, alpha=0.08, color=color)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 10),
                            textcoords="offset points", ha="center",
                            fontsize=8.5, color=color, fontweight="bold")

        ax.set_xscale("log")
        ax.set_xticks(POOL_SIZES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Tool pool size", fontsize=13)
        ax.set_ylabel("Accuracy (avg reward)", fontsize=13)
        ax.set_title("Task-1 Accuracy vs. Tool Pool Size", fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=11)
        fig.tight_layout()
        plot_file = Path(OUTPUT_DIR) / "pool_sweep_plot.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved → {plot_file}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())

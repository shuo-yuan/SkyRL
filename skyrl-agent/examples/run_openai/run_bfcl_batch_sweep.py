"""
Sweep TASKS_PER_BATCH ∈ {1, 4, 8, 12, 16} and plot accuracy vs batch size.

Usage:
    cd /home/ec2-user/shuo/SkyRL/skyrl-agent
    uv run --env-file .env examples/run_openai/run_bfcl_batch_sweep.py

Results are saved to outputs/bfcl_batch_sweep/ and a PNG plot is generated.
"""

# ─────────────────────────────────────────────────────────────────
BATCH_SIZES = [1, 4, 8, 12, 16]    # ← batch sizes to sweep

# Tool-search mode: if True, only `bfcl_tool_search` is in the system prompt
# and the model must call it to retrieve tools.  Set False for upfront-tools mode.
USE_TOOL_SEARCH: bool = True
TOOL_SEARCH_K:   int  = 3
TOOL_SEARCH_RETRIEVAL: str = "llm"    # "bm25" or "llm"
TOOL_SEARCH_LLM_MODEL: str = "gpt-5-nano"
BM25_K:               int  = 4       # k for BM25 retrieval

# When True, each completed task's tool calls and responses are collapsed into
# a single summary line before the next task begins (keeps context short).
FOLD_TOOL_INFO: bool = False

# Optional: limit dataset size for a quick test (None = full 3641)
N_SAMPLES = None

# Model / API
MODEL            = "Qwen/Qwen3-32B"
API_MODEL_NAME   = "gpt-5-nano"
API_URL          = "https://api.openai.com"
DATASET_PATH     = "data/BFCL_single_turn.parquet"

# Agent settings
MAX_ITERATIONS_PER_TASK = 10
MAX_PROMPT_LENGTH       = 32768
TEMPERATURE             = 1.0
MAX_TOKENS              = 32768
MAX_PARALLEL_AGENTS     = 200
# ─────────────────────────────────────────────────────────────────

import asyncio, json, os, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import datasets
from transformers import AutoTokenizer

import skyrl_agent.tools.bfcl_tool_search  # force-register bfcl_tool_search
from skyrl_agent.agents.react.react_agent import ReActAgent
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.openai import OpenAIAPIBackend, OpenAIAPIBackendConfig
from skyrl_agent.tasks.bfcl_eval_task import (
    BFCLEvalTask, _normalise_function_list, _parse_json_field,
)

OUTPUT_DIR = Path("outputs/bfcl_batch_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers (same as run_openai_bfcl_batch.py) ───────────────────

def combine_tools(instances):
    seen, combined = set(), []
    for inst in instances:
        for func in _normalise_function_list(_parse_json_field(inst.get("function", []))):
            name = func.get("name", "")
            if name and name not in seen:
                seen.add(name)
                params = dict(func.get("parameters", {}))
                if params.get("type") == "dict":
                    params["type"] = "object"
                combined.append({"type": "function", "function": {
                    "name": name,
                    "description": func.get("description", ""),
                    "parameters": params,
                }})
    return combined


def make_agent(tokenizer, batch_idx, tasks_per_batch, use_tool_search=False):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    traj_config = TrajectoryConfig(
        instance_id=batch_idx, trajectory_id=0,
        sampling_params={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        max_prompt_length=MAX_PROMPT_LENGTH,
        qwen3_enable_thinking=False, qwen3_acc_thinking=False,
        tools=["bfcl_tool_search"] if use_tool_search else [],
        fold_tool_info=FOLD_TOOL_INFO,
        max_iterations=MAX_ITERATIONS_PER_TASK * tasks_per_batch,
        agent_cls="skyrl_agent.agents.react.ReActAgent",
        enable_turn_reminder=False, early_step_threshold=0,
        debug_log=False, profile_tools=False,
    )
    backend_cfg = OpenAIAPIBackendConfig(
        model_name=API_MODEL_NAME, api_url=API_URL, api_key=api_key,
    )
    infer_engine = OpenAIAPIBackend(infer_engine=None, cfg=backend_cfg)
    return ReActAgent(traj_config=traj_config, infer_engine=infer_engine, tokenizer=tokenizer)


async def run_one_batch(tokenizer, batch_instances, batch_idx, semaphore, tasks_per_batch):
    async with semaphore:
        n = len(batch_instances)
        instructions = [
            BFCLEvalTask.get_instruction(inst, tool_search_mode=USE_TOOL_SEARCH)
            for inst in batch_instances
        ]
        combined_tools = combine_tools(batch_instances)
        agent = make_agent(tokenizer, batch_idx, tasks_per_batch,
                           use_tool_search=USE_TOOL_SEARCH)
        if not USE_TOOL_SEARCH:
            agent._active_bfcl_tool_params = combined_tools
        try:
            _, per_task_calls = await agent.run_batch(
                instructions=instructions,
                instances=batch_instances,
                combined_tool_params=combined_tools,
                use_tool_search=USE_TOOL_SEARCH,
                tool_search_k=TOOL_SEARCH_K,
                bm25_k=BM25_K,
                tool_search_retrieval=TOOL_SEARCH_RETRIEVAL,
                tool_search_llm_model=TOOL_SEARCH_LLM_MODEL,
            )
        except Exception as e:
            print(f"[Batch {batch_idx}] error: {e}")
            return [0.0] * n

        rewards = []
        for task_calls, instance in zip(per_task_calls, batch_instances):
            try:
                r = await BFCLEvalTask.evaluate_result(
                    result=task_calls, instance=instance,
                    data_source=instance.get("data_source", ""),
                    instance_id=instance.get("id", ""), trajectory_id=0,
                )
            except Exception:
                r = 0.0
            rewards.append(float(r))
        return rewards


async def run_sweep_for_batch_size(
    all_instances, tokenizer, tasks_per_batch, global_semaphore
):
    """Run the full dataset with the given batch size and return per-sample rewards."""
    batches = [
        all_instances[i: i + tasks_per_batch]
        for i in range(0, len(all_instances), tasks_per_batch)
    ]
    tasks = [
        run_one_batch(tokenizer, batch, idx, global_semaphore, tasks_per_batch)
        for idx, batch in enumerate(batches)
    ]
    batch_rewards = await asyncio.gather(*tasks)

    all_rewards = []
    for br in batch_rewards:
        all_rewards.extend(br)
    return all_rewards


async def run_one_sweep(all_instances, tokenizer, tasks_per_batch, global_semaphore):
    """Wrapper that runs one sweep and collects its result dict."""
    print(f"[Batch={tasks_per_batch:2d}] Starting sweep …")
    t0 = time.time()

    rewards = await run_sweep_for_batch_size(
        all_instances, tokenizer, tasks_per_batch, global_semaphore
    )

    elapsed   = time.time() - t0
    avg       = sum(rewards) / len(rewards) if rewards else 0.0
    n_correct = sum(rewards)

    cat_rewards: Dict[str, List[float]] = defaultdict(list)
    for inst, r in zip(all_instances[: len(rewards)], rewards):
        cat = str(inst.get("id", "")).rsplit("_", 1)[0] or "unknown"
        cat_rewards[cat].append(r)

    cat_summary = {
        cat: {"avg": sum(v)/len(v), "count": len(v), "correct": sum(v)}
        for cat, v in cat_rewards.items()
    }

    result = {
        "tasks_per_batch": tasks_per_batch,
        "total_samples":   len(rewards),
        "avg_reward":      avg,
        "n_correct":       n_correct,
        "time_seconds":    elapsed,
        "per_category":    cat_summary,
    }

    # Save immediately when this sweep finishes
    result_file = OUTPUT_DIR / f"batch_{tasks_per_batch}_results.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"[Batch={tasks_per_batch:2d}] DONE  avg={avg:.4f}  "
        f"({n_correct:.0f}/{len(rewards)})  {elapsed:.1f}s  → {result_file}"
    )
    return tasks_per_batch, result


# ── Main sweep ────────────────────────────────────────────────────

async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    print(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print(f"Loading dataset: {DATASET_PATH}")
    full_ds = datasets.load_dataset("parquet", data_files=DATASET_PATH)["train"]
    if N_SAMPLES:
        full_ds = full_ds.select(range(min(N_SAMPLES, len(full_ds))))
    all_instances = [dict(row) for row in full_ds]
    print(f"Total samples : {len(all_instances)}")
    print(f"Batch sizes   : {BATCH_SIZES}  (all run in parallel)")
    print(f"Concurrency   : {MAX_PARALLEL_AGENTS} total API calls shared across sweeps")

    # One shared semaphore limits the total number of concurrent API calls
    # across ALL batch-size sweeps running simultaneously.
    global_semaphore = asyncio.Semaphore(MAX_PARALLEL_AGENTS)

    t_start = time.time()

    # Launch all batch-size sweeps concurrently
    sweep_tasks = [
        run_one_sweep(all_instances, tokenizer, bs, global_semaphore)
        for bs in BATCH_SIZES
    ]
    completed = await asyncio.gather(*sweep_tasks)

    sweep_results: Dict[int, Dict] = {bs: res for bs, res in completed}
    total_elapsed = time.time() - t_start

    # ── Save combined sweep results ───────────────────────────────
    combined_file = OUTPUT_DIR / "sweep_summary.json"
    with open(combined_file, "w") as f:
        json.dump(
            {str(k): v for k, v in sweep_results.items()},
            f, indent=2,
        )
    print(f"\nSweep summary saved to {combined_file}")

    # ── Print summary table ───────────────────────────────────────
    print(f"\n{'Batch Size':>12} {'Avg Accuracy':>14} {'Time (s)':>10}")
    print("-" * 40)
    for bs in BATCH_SIZES:
        if bs in sweep_results:
            r = sweep_results[bs]
            print(f"{bs:>12} {r['avg_reward']:>14.4f} {r['time_seconds']:>10.1f}")
    print(f"\nTotal wall-clock time: {total_elapsed:.1f}s  "
          f"(vs ~{total_elapsed * len(BATCH_SIZES):.0f}s if sequential)")

    # ── Plot ──────────────────────────────────────────────────────
    _plot_sweep(sweep_results)


def _plot_sweep(sweep_results: Dict[int, Dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        batch_sizes = sorted(sweep_results.keys())
        accuracies  = [sweep_results[bs]["avg_reward"] for bs in batch_sizes]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(batch_sizes, accuracies, marker="o", linewidth=2,
                markersize=8, color="#2196F3", label="Overall accuracy")
        ax.fill_between(batch_sizes, accuracies,
                         alpha=0.12, color="#2196F3")

        # Annotate each point
        for bs, acc in zip(batch_sizes, accuracies):
            ax.annotate(
                f"{acc:.3f}",
                xy=(bs, acc), xytext=(0, 10),
                textcoords="offset points",
                ha="center", fontsize=9, color="#1565C0",
            )

        ax.set_xlabel("Tasks per batch (TASKS_PER_BATCH)", fontsize=12)
        ax.set_ylabel("Accuracy (avg reward)", fontsize=12)
        ax.set_title("BFCL Single-Turn Accuracy vs. Batch Size", fontsize=13, fontweight="bold")
        ax.set_xticks(batch_sizes)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_ylim(
            max(0, min(accuracies) - 0.05),
            min(1.0, max(accuracies) + 0.05),
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)

        plot_file = OUTPUT_DIR / "accuracy_vs_batch_size.png"
        fig.tight_layout()
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_file}")

    except ImportError:
        print("matplotlib not installed – skipping plot. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Plot error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

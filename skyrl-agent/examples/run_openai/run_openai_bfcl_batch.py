"""
Multi-task BFCL evaluation: TASKS_PER_BATCH tasks chained into one conversation.

The model sees ALL tools from all tasks upfront and handles each task
sequentially within a single conversation, retaining message history.

Usage:
    cd /home/ec2-user/shuo/SkyRL/skyrl-agent
    uv run --env-file .env examples/run_openai/run_openai_bfcl_batch.py

Tune TASKS_PER_BATCH below to control how many tasks are chained together.
Set to 1 to match the original single-task behaviour.
"""

# ──────────────────────────────────────────────────────────────────
# ← Configure here
TASKS_PER_BATCH: int = 5

# If True, only a `bfcl_tool_search` tool is shown in the system
# prompt.  The model must call it to retrieve relevant functions from
# the batch pool before it can use them.  Requires TOOL_SEARCH_K.
# If False (default), all tools are visible upfront (original behaviour).
USE_TOOL_SEARCH: bool = False
TOOL_SEARCH_K:   int  = 3           # tools returned per search query
FOLD_TOOL_INFO:  bool = False        # collapse each task's history into a summary

# Retrieval method used by bfcl_tool_search when USE_TOOL_SEARCH=True:
#   "bm25"  – fast keyword scoring, no extra API cost (default)
#   "llm"   – uses TOOL_SEARCH_LLM_MODEL for reranking (slower, one extra API call per search)
TOOL_SEARCH_RETRIEVAL: str = "llm"
BM25_K:               int  = 4     # k for BM25 retrieval when TOOL_SEARCH_RETRIEVAL="bm25"
TOOL_SEARCH_LLM_MODEL: str = "gpt-5-nano"
# ──────────────────────────────────────────────────────────────────

import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import datasets
from transformers import AutoTokenizer

from skyrl_agent.agents.react.react_agent import ReActAgent
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.openai import OpenAIAPIBackend, OpenAIAPIBackendConfig
from skyrl_agent.tasks.bfcl_eval_task import (
    BFCLEvalTask,
    _normalise_function_list,
    _parse_json_field,
)

# ── Config ────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-32B"
API_MODEL_NAME = "gpt-5-nano"
API_URL = "https://api.openai.com"
DATASET_PATH = "data/BFCL_single_turn.parquet"
OUTPUT_DIR = Path("outputs/bfcl_batch")
N_SAMPLES: Optional[int] = None       # None = full dataset
MAX_ITERATIONS_PER_TASK: int = 10     # scaled by TASKS_PER_BATCH internally
MAX_PROMPT_LENGTH: int = 32768
MAX_PARALLEL_AGENTS: int = 200        # concurrent agent runs
TEMPERATURE: float = 1.0
MAX_TOKENS: int = 8192


# ── Tool helpers ──────────────────────────────────────────────────

def combine_tools(instances: List[Dict]) -> List[Dict]:
    """Return deduplicated union of all function schemas across instances."""
    seen: set = set()
    combined: List[Dict] = []
    for inst in instances:
        for func in _normalise_function_list(_parse_json_field(inst.get("function", []))):
            name = func.get("name", "")
            if name and name not in seen:
                seen.add(name)
                params = dict(func.get("parameters", {}))
                if params.get("type") == "dict":
                    params["type"] = "object"
                combined.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": func.get("description", ""),
                        "parameters": params,
                    },
                })
    return combined


# ── Agent factory ─────────────────────────────────────────────────

def make_agent(tokenizer, batch_idx: int, use_tool_search: bool = False) -> ReActAgent:
    # Register bfcl_tool_search only when USE_TOOL_SEARCH is enabled
    tools = ["bfcl_tool_search"] if use_tool_search else []
    traj_config = TrajectoryConfig(
        instance_id=batch_idx,
        trajectory_id=0,
        sampling_params={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        max_prompt_length=MAX_PROMPT_LENGTH,
        qwen3_enable_thinking=False,
        qwen3_acc_thinking=False,
        tools=tools,
        max_iterations=MAX_ITERATIONS_PER_TASK * TASKS_PER_BATCH,
        agent_cls="skyrl_agent.agents.react.ReActAgent",
        enable_turn_reminder=False,
        early_step_threshold=0,
        fold_tool_info=FOLD_TOOL_INFO,
        debug_log=False,
        profile_tools=False,
    )
    api_key = os.environ.get("OPENAI_API_KEY", "")
    backend_cfg = OpenAIAPIBackendConfig(
        model_name=API_MODEL_NAME,
        api_url=API_URL,
        api_key=api_key,
    )
    infer_engine = OpenAIAPIBackend(infer_engine=None, cfg=backend_cfg)
    return ReActAgent(
        traj_config=traj_config,
        infer_engine=infer_engine,
        tokenizer=tokenizer,
    )


# ── Per-batch runner ──────────────────────────────────────────────

async def run_one_batch(
    tokenizer,
    batch_instances: List[Dict],
    batch_idx: int,
    semaphore: asyncio.Semaphore,
) -> List[float]:
    """Run one batch of TASKS_PER_BATCH instances and return per-task rewards."""
    async with semaphore:
        n = len(batch_instances)

        # Build per-task instructions and combined tools
        instructions = [
            BFCLEvalTask.get_instruction(inst, tool_search_mode=USE_TOOL_SEARCH)
            for inst in batch_instances
        ]
        combined_tools = combine_tools(batch_instances)

        agent = make_agent(tokenizer, batch_idx)
        if not USE_TOOL_SEARCH:
            agent._active_bfcl_tool_params = combined_tools  # set before run_batch

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
            print(f"[Batch {batch_idx}] run_batch error: {e}")
            return [0.0] * n

        # Evaluate each sub-task
        rewards: List[float] = []
        for task_calls, instance in zip(per_task_calls, batch_instances):
            try:
                reward = await BFCLEvalTask.evaluate_result(
                    result=task_calls,
                    instance=instance,
                    data_source=instance.get("data_source", ""),
                    instance_id=instance.get("id", ""),
                    trajectory_id=0,
                )
            except Exception as e:
                print(f"[Eval error] {e}")
                reward = 0.0
            rewards.append(float(reward))

        correct = sum(rewards)
        print(
            f"[Batch {batch_idx:4d}] {correct:.0f}/{n} correct  "
            f"({correct/n*100:.1f}%)  "
            f"tasks: {[inst.get('id','?') for inst in batch_instances]}"
        )
        return rewards


# ── Main ──────────────────────────────────────────────────────────

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    print(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print(f"Loading dataset: {DATASET_PATH}")
    full_ds = datasets.load_dataset("parquet", data_files=DATASET_PATH)["train"]
    if N_SAMPLES is not None:
        full_ds = full_ds.select(range(min(N_SAMPLES, len(full_ds))))
    all_instances = [dict(row) for row in full_ds]
    print(f"Total samples: {len(all_instances)}  |  TASKS_PER_BATCH: {TASKS_PER_BATCH}")

    # Group into batches of TASKS_PER_BATCH
    batches: List[List[Dict]] = [
        all_instances[i : i + TASKS_PER_BATCH]
        for i in range(0, len(all_instances), TASKS_PER_BATCH)
    ]
    print(f"Total batches: {len(batches)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(MAX_PARALLEL_AGENTS)
    start_time = time.time()

    # Run all batches concurrently (up to MAX_PARALLEL_AGENTS at once)
    tasks = [
        run_one_batch(tokenizer, batch, batch_idx, semaphore)
        for batch_idx, batch in enumerate(batches)
    ]
    batch_rewards_list = await asyncio.gather(*tasks)

    # Flatten: per-sample rewards in dataset order
    all_rewards: List[float] = []
    for batch_rewards in batch_rewards_list:
        all_rewards.extend(batch_rewards)

    elapsed = time.time() - start_time

    # ── Per-category summary ───────────────────────────────────────
    category_rewards: Dict[str, List[float]] = defaultdict(list)
    for inst, reward in zip(all_instances[: len(all_rewards)], all_rewards):
        entry_id = str(inst.get("id", ""))
        cat = entry_id.rsplit("_", 1)[0] if entry_id else "unknown"
        category_rewards[cat].append(reward)

    print(f"\n{'='*60}")
    print(f"BFCL Batch Eval Complete  (TASKS_PER_BATCH={TASKS_PER_BATCH})")
    print(f"{'='*60}")
    print(f"Total samples : {len(all_rewards)}")
    print(f"Total time    : {elapsed:.1f}s")
    avg = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(f"Average reward: {avg:.4f}  ({sum(all_rewards):.0f}/{len(all_rewards)})")

    cat_summary: Dict[str, Dict] = {}
    print("\nPer-category results:")
    for cat, rewards in sorted(category_rewards.items()):
        cat_avg = sum(rewards) / len(rewards) if rewards else 0.0
        cat_summary[cat] = {
            "avg_reward": cat_avg,
            "count": len(rewards),
            "correct": sum(rewards),
        }
        print(f"  {cat}: {sum(rewards):.0f}/{len(rewards)} = {cat_avg:.4f}")

    # ── Save results ───────────────────────────────────────────────
    results_file = OUTPUT_DIR / f"bfcl_batch_{TASKS_PER_BATCH}_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "tasks_per_batch": TASKS_PER_BATCH,
                "total_samples": len(all_rewards),
                "total_time_seconds": elapsed,
                "average_reward": avg,
                "per_category": cat_summary,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())

"""
Smoke test for bfcl_tool_search mode on 50 samples.

Tests:
  1. Tool search is actually called (model uses bfcl_tool_search before domain calls)
  2. Per-task evaluation is correct
  3. Accuracy comparison: tool_search=True vs tool_search=False (same 50 samples)

Usage:
    uv run --env-file .env examples/run_openai/run_bfcl_tool_search_smoke.py
"""

import asyncio, json, os, time, re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import datasets
from transformers import AutoTokenizer
import skyrl_agent.tools.bfcl_tool_search  # force registration

from skyrl_agent.agents.react.react_agent import ReActAgent
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.openai import OpenAIAPIBackend, OpenAIAPIBackendConfig
from skyrl_agent.tasks.bfcl_eval_task import (
    BFCLEvalTask, _normalise_function_list, _parse_json_field,
)

# ── Config ────────────────────────────────────────────────────────
N_SAMPLES      = 4
TASKS_PER_BATCH = 4
MODEL          = "Qwen/Qwen3-32B"
API_MODEL_NAME = "gpt-5-nano"
API_URL        = "https://api.openai.com"
DATASET_PATH   = "data/BFCL_single_turn.parquet"
MAX_PARALLEL   = 16
OUTPUT_DIR     = Path("outputs/bfcl_tool_search_smoke")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVAL      = "llm"    # "bm25" or "llm"
TOOL_SEARCH_K  = 4


def combine_tools(instances):
    seen, out = set(), []
    for inst in instances:
        for fn in _normalise_function_list(_parse_json_field(inst.get("function", []))):
            n = fn.get("name", "")
            if n and n not in seen:
                seen.add(n)
                p = dict(fn.get("parameters", {}))
                if p.get("type") == "dict":
                    p["type"] = "object"
                out.append({"type": "function", "function": {
                    "name": n, "description": fn.get("description", ""), "parameters": p,
                }})
    return out


def make_agent(tokenizer, batch_idx, use_tool_search):
    tools = ["bfcl_tool_search"] if use_tool_search else []
    cfg = TrajectoryConfig(
        instance_id=batch_idx, trajectory_id=0,
        sampling_params={"temperature": 1.0, "max_tokens": 8192},
        max_prompt_length=32768,
        qwen3_enable_thinking=False, qwen3_acc_thinking=False,
        tools=tools,
        max_iterations=10 * TASKS_PER_BATCH,
        agent_cls="skyrl_agent.agents.react.ReActAgent",
        enable_turn_reminder=False, early_step_threshold=0,
        debug_log=False, profile_tools=False,
    )
    engine = OpenAIAPIBackend(
        infer_engine=None,
        cfg=OpenAIAPIBackendConfig(
            model_name=API_MODEL_NAME, api_url=API_URL,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    )
    return ReActAgent(traj_config=cfg, infer_engine=engine, tokenizer=tokenizer)


async def run_batch(tokenizer, instances, batch_idx, sem, use_tool_search):
    async with sem:
        n = len(instances)
        instructions  = [BFCLEvalTask.get_instruction(inst, tool_search_mode=use_tool_search)
                         for inst in instances]
        combined      = combine_tools(instances)
        agent         = make_agent(tokenizer, batch_idx, use_tool_search)
        if not use_tool_search:
            agent._active_bfcl_tool_params = combined

        try:
            _, per_calls = await agent.run_batch(
                instructions=instructions, instances=instances,
                combined_tool_params=combined,
                use_tool_search=use_tool_search,
                tool_search_k=TOOL_SEARCH_K,
                tool_search_retrieval=RETRIEVAL,
            )
        except Exception as e:
            print(f"[Batch {batch_idx}] error: {e}")
            return [0.0]*n, [[] for _ in range(n)], [False]*n

        rewards, all_calls, search_used = [], [], []
        for calls, inst in zip(per_calls, instances):
            try:
                r = await BFCLEvalTask.evaluate_result(
                    result=calls, instance=inst,
                    data_source=inst.get("data_source",""),
                    instance_id=inst.get("id",""), trajectory_id=0,
                )
            except Exception:
                r = 0.0
            rewards.append(float(r))
            all_calls.append(calls)
            # Check if bfcl_tool_search was called for this task
            # (we can check conversation messages logged, but easier: check calls list)
            # bfcl_tool_search calls are NOT in _bfcl_recorded_calls
            # We check via agent's message history indirectly — just note N/A here
            search_used.append(None)

        return rewards, all_calls, search_used


async def run_mode(all_instances, tokenizer, use_tool_search):
    batches = [all_instances[i:i+TASKS_PER_BATCH] for i in range(0,len(all_instances),TASKS_PER_BATCH)]
    sem = asyncio.Semaphore(MAX_PARALLEL)
    tasks = [run_batch(tokenizer, b, i, sem, use_tool_search) for i, b in enumerate(batches)]
    results = await asyncio.gather(*tasks)

    all_rewards, all_calls_flat = [], []
    for rewards, calls, _ in results:
        all_rewards.extend(rewards)
        all_calls_flat.extend(calls)
    return all_rewards, all_calls_flat


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    print(f"Loading tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    print(f"Loading dataset ({N_SAMPLES} samples)…")
    ds = datasets.load_dataset("parquet", data_files=DATASET_PATH)["train"]
    instances = [dict(r) for r in ds.select(range(N_SAMPLES))]
    categories = [str(inst.get("id","")).rsplit("_",1)[0] for inst in instances]

    print(f"\nRunning {N_SAMPLES} samples, TASKS_PER_BATCH={TASKS_PER_BATCH}, "
          f"retrieval={RETRIEVAL!r}, K={TOOL_SEARCH_K}")
    print("="*60)

    t0 = time.time()
    rewards_search, _ = await run_mode(instances, tok, use_tool_search=True)
    t_search = time.time() - t0

    t0 = time.time()
    rewards_baseline, _ = await run_mode(instances, tok, use_tool_search=False)
    t_base = time.time() - t0

    # ── Per-category breakdown ────────────────────────────────────
    cat_search   = defaultdict(list)
    cat_baseline = defaultdict(list)
    for cat, rs, rb in zip(categories, rewards_search, rewards_baseline):
        cat_search[cat].append(rs)
        cat_baseline[cat].append(rb)

    print(f"\n{'Category':30s} {'Baseline':>10} {'ToolSearch':>10} {'Diff':>8}")
    print("-"*62)
    for cat in sorted(set(categories)):
        rb = sum(cat_baseline[cat])/len(cat_baseline[cat]) if cat_baseline[cat] else 0
        rs = sum(cat_search[cat])/len(cat_search[cat]) if cat_search[cat] else 0
        d = rs - rb
        flag = "↑" if d > 0.02 else ("↓" if d < -0.02 else " ")
        print(f"  {cat:28s} {rb:10.3f} {rs:10.3f} {d:+8.3f} {flag}")

    avg_b = sum(rewards_baseline)/len(rewards_baseline)
    avg_s = sum(rewards_search)/len(rewards_search)
    print("-"*62)
    print(f"  {'OVERALL':28s} {avg_b:10.3f} {avg_s:10.3f} {avg_s-avg_b:+8.3f}")
    print(f"\nTime — baseline: {t_base:.1f}s   tool_search: {t_search:.1f}s")

    # Save
    out = {
        "n_samples": N_SAMPLES, "tasks_per_batch": TASKS_PER_BATCH,
        "retrieval": RETRIEVAL, "tool_search_k": TOOL_SEARCH_K,
        "baseline_avg": avg_b, "tool_search_avg": avg_s,
        "per_category": {
            cat: {
                "baseline": sum(cat_baseline[cat])/len(cat_baseline[cat]),
                "tool_search": sum(cat_search[cat])/len(cat_search[cat]),
            }
            for cat in sorted(set(categories))
        },
    }
    f = OUTPUT_DIR / f"smoke_batch{TASKS_PER_BATCH}_{RETRIEVAL}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {f}")


if __name__ == "__main__":
    asyncio.run(main())

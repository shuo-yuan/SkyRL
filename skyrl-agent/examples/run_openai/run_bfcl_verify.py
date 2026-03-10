"""
Quick verification: run a few single-turn and multi-turn BFCL tasks,
print full trajectories and evaluation scores.

Usage:
    cd /home/ec2-user/shuo/SkyRL/skyrl-agent
    uv run --env-file .env python examples/run_openai/run_bfcl_verify.py
"""

import os, json, asyncio, re
from pathlib import Path

from skyrl_agent import AutoAgentRunner
from transformers import AutoTokenizer
import datasets

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

TOKENIZER_MODEL = "Qwen/Qwen3-32B"
YAML_PATH = str(Path(__file__).parents[1] / "inference" / "openai_api_gpt_load_finish.yaml")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

# ── 1. Select samples ──────────────────────────────────────────────────────
SINGLE_TURN_IDS = [
    "simple_python_0",       # simple single function call
    "multiple_0",            # choose from multiple functions
    "parallel_0",            # call multiple functions in parallel
    "parallel_multiple_0",   # parallel + multiple
]
MULTI_TURN_IDS = [
    "multi_turn_base_0",     # 2-turn base task
    "multi_turn_base_1",     # another base task
]

ds_ast = datasets.load_dataset("parquet", data_files="data/BFCL_single_turn_ast.parquet")["train"]
ds_mt  = datasets.load_dataset("parquet", data_files="data/BFCL_multi_turn.parquet")["train"]

id_to_row = {}
for row in ds_ast:
    id_to_row[row["id"]] = dict(row)
for row in ds_mt:
    id_to_row[row["id"]] = dict(row)

wanted = SINGLE_TURN_IDS + MULTI_TURN_IDS
rows = [id_to_row[eid] for eid in wanted if eid in id_to_row]
print(f"Running {len(rows)} samples: {[r['id'] for r in rows]}")

subset = datasets.Dataset.from_list(rows)

# ── 2. Run ─────────────────────────────────────────────────────────────────
runner = AutoAgentRunner.from_task(YAML_PATH, infer_engine=None, tokenizer=tokenizer)
output = asyncio.run(runner.run(subset, val_mode=True))

# ── 3. Print trajectories ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("DETAILED TRAJECTORIES")
print(f"{'='*70}")

for iid, trajs in sorted(runner.trajectories.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
    for tid, traj in trajs.items():
        r = traj.result or {}
        entry_id = rows[int(iid)]["id"] if str(iid).isdigit() else str(iid)
        messages  = r.get("messages", [])
        finish    = r.get("finish_reason", "N/A")
        reward    = r.get("reward", "N/A")
        results   = r.get("results", None)

        print(f"\n{'─'*70}")
        print(f"  ID            : {entry_id}")
        print(f"  finish_reason : {finish}")
        print(f"  reward        : {reward}")

        # Show recorded BFCL calls (the agent's result payload)
        if results is not None:
            if isinstance(results, list):
                # single-turn: flat list of call dicts
                # multi-turn: list[list[call dicts]]
                if results and isinstance(results[0], list):
                    print(f"  recorded_calls (multi-turn, {len(results)} turns):")
                    for ti, turn_calls in enumerate(results):
                        for call in turn_calls:
                            print(f"    turn {ti}: {call.get('function','')}({json.dumps(call.get('arguments',{}), ensure_ascii=False)})")
                else:
                    print(f"  recorded_calls ({len(results)} calls):")
                    for call in results:
                        if isinstance(call, dict):
                            print(f"    {call.get('function','')}({json.dumps(call.get('arguments',{}), ensure_ascii=False)})")
            else:
                print(f"  result        : {str(results)[:200]}")

        # Show conversation turns
        print(f"  conversation ({len(messages)} msgs):")
        for msg in messages:
            role    = msg.get("role", "?")
            content = msg.get("content", "") or ""
            # Truncate long messages
            snippet = content[:300].replace("\n", " ")
            if len(content) > 300:
                snippet += "…"
            print(f"    [{role}] {snippet}")

# ── 4. Summary ─────────────────────────────────────────────────────────────
metrics = output.get("rollout_metrics", {})
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  raw_reward        : {metrics.get('rollout_metrics/raw_reward', 'N/A')}")
print(f"  num_all_resolved  : {metrics.get('rollout_metrics/num_all_resolved', 'N/A')}")
print(f"  num_none_resolved : {metrics.get('rollout_metrics/num_none_resolved', 'N/A')}")
print(f"  avg_turns         : {metrics.get('rollout_metrics/avg_turn_assistant', 'N/A')}")
print(f"  error_runtime     : {metrics.get('rollout_metrics/error_runtime', 'N/A')}")
print(f"  error_evaluation  : {metrics.get('rollout_metrics/error_evaluation', 'N/A')}")

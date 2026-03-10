"""
Run all BFCL scoring tasks through the skyrl-agent ReAct pipeline.

Usage:
    cd /home/ec2-user/shuo/SkyRL/skyrl-agent
    uv run --env-file .env examples/run_openai/run_openai_bfcl_all.py
"""

import os
import json
import time
from pathlib import Path
import asyncio

from skyrl_agent import AutoAgentRunner
from transformers import AutoTokenizer
import datasets

# --- Config ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = api_key

MODEL = "Qwen/Qwen3-32B"
DATASET_PATH = "data/BFCL_single_turn.parquet"
YAML_PATH = str(Path(__file__).parents[1] / "inference" / "openai_api_gpt_load_finish.yaml")
OUTPUT_DIR = Path("outputs/bfcl_all")
BATCH_SIZE = 1024  # Process in batches to avoid memory issues
N_SAMPLES = None   # Set to None to run the full dataset

# --- Load ---
print(f"Loading tokenizer: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print(f"Loading dataset: {DATASET_PATH}")
full_ds = datasets.load_dataset("parquet", data_files=DATASET_PATH)["train"]
if N_SAMPLES is not None:
    full_ds = full_ds.select(range(min(N_SAMPLES, len(full_ds))))
print(f"Total samples: {len(full_ds)}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Run in batches ---
all_rewards = []
all_finish_reasons = []
category_rewards = {}
start_time = time.time()

num_batches = (len(full_ds) + BATCH_SIZE - 1) // BATCH_SIZE
for batch_idx in range(num_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(full_ds))
    batch_ds = full_ds.select(range(batch_start, batch_end))

    print(f"\n{'='*60}")
    print(f"Batch {batch_idx+1}/{num_batches} (samples {batch_start}-{batch_end-1})")
    print(f"{'='*60}")

    agent_generator = AutoAgentRunner.from_task(
        YAML_PATH,
        infer_engine=None,
        tokenizer=tokenizer,
    )

    try:
        output = asyncio.run(agent_generator.run(batch_ds, val_mode=True))
    except Exception as e:
        print(f"[ERROR] Batch {batch_idx+1} failed: {e}")
        # Fill with zeros for this batch
        all_rewards.extend([0.0] * (batch_end - batch_start))
        continue

    metrics = output.get("rollout_metrics", {})
    batch_reward = metrics.get("rollout_metrics/raw_reward", 0.0)

    # Collect per-sample rewards from traj_rewards (list of per-trajectory floats/bools)
    # base.py._post_process_results() puts per-trajectory rewards in output["traj_rewards"]
    traj_rewards_raw = output.get("traj_rewards", [])
    if traj_rewards_raw:
        sample_rewards = [float(r) for r in traj_rewards_raw]
    else:
        # Fallback: broadcast batch average (should not happen in normal runs)
        sample_rewards = [batch_reward] * (batch_end - batch_start)
    all_rewards.extend(sample_rewards)

    # Track per-category rewards using per-sample rewards
    for i, reward in zip(range(batch_start, batch_end), sample_rewards):
        entry_id = str(full_ds[i].get("id", ""))
        cat = entry_id.rsplit("_", 1)[0] if entry_id else "unknown"
        category_rewards.setdefault(cat, []).append(reward)

    elapsed = time.time() - start_time
    print(f"Batch {batch_idx+1} reward: {batch_reward:.4f} | Elapsed: {elapsed:.1f}s")

# --- Summary ---
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"BFCL All Tasks Complete")
print(f"{'='*60}")
print(f"Total samples: {len(full_ds)}")
print(f"Total time: {elapsed:.1f}s")
if all_rewards:
    avg_reward = sum(all_rewards) / len(all_rewards)
    print(f"Average reward: {avg_reward:.4f} ({sum(all_rewards):.0f}/{len(all_rewards)})")

# Per-category summary
print(f"\nPer-category results:")
cat_summary = {}
for cat, rewards in sorted(category_rewards.items()):
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    cat_summary[cat] = {"avg_reward": avg, "count": len(rewards), "correct": sum(rewards)}
    print(f"  {cat}: {sum(rewards):.0f}/{len(rewards)} = {avg:.4f}")

# Save results
results_file = OUTPUT_DIR / "bfcl_all_results.json"
with open(results_file, "w") as f:
    json.dump({
        "total_samples": len(full_ds),
        "total_time_seconds": elapsed,
        "average_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "per_category": cat_summary,
    }, f, indent=2)
print(f"\nResults saved to {results_file}")

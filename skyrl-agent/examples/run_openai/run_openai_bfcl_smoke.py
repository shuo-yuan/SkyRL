"""
Smoke test: run 2 examples from each BFCL category (except web_search)
to verify trajectory execution and evaluation both work correctly.

Usage:
    cd /home/ec2-user/shuo/SkyRL/skyrl-agent
    uv run --env-file .env examples/run_openai/run_openai_bfcl_smoke.py
"""

import os, json, time, asyncio
from pathlib import Path
from collections import defaultdict

from skyrl_agent import AutoAgentRunner
from transformers import AutoTokenizer
import datasets

# ---------------------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = api_key

MODEL = "Qwen/Qwen3-32B"
DATASET_PATH = "data/BFCL_all_scoring_tasks.parquet"
YAML_PATH = str(Path(__file__).parents[1] / "inference" / "openai_api_gpt_load_finish.yaml")
OUTPUT_DIR = Path("outputs/bfcl_smoke")
N_PER_CATEGORY = 2  # samples per category

SKIP_CATEGORIES = {
    "web_search_base",
    "web_search_no_snippet",
    # memory prereq entries are not scored separately
    "memory_kv_prereq",
    "memory_rec_sum_prereq",
    "memory_vector_prereq",
}

# ---------------------------------------------------------------------------
print(f"Loading tokenizer: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print(f"Loading dataset: {DATASET_PATH}")
full_ds = datasets.load_dataset("parquet", data_files=DATASET_PATH)["train"]
print(f"Total samples: {len(full_ds)}")

# Pick N_PER_CATEGORY examples from each category
cat_to_indices: dict = defaultdict(list)
for i, eid in enumerate(full_ds["id"]):
    cat = str(eid).rsplit("_", 1)[0]
    if cat not in SKIP_CATEGORIES:
        cat_to_indices[cat].append(i)

selected_indices = []
for cat in sorted(cat_to_indices):
    idxs = cat_to_indices[cat][:N_PER_CATEGORY]
    selected_indices.extend(idxs)
    print(f"  {cat}: {[full_ds[i]['id'] for i in idxs]}")

subset = full_ds.select(selected_indices)
print(f"\nTotal samples to run: {len(subset)} ({len(cat_to_indices)} categories × {N_PER_CATEGORY})")

# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
start_time = time.time()

agent_generator = AutoAgentRunner.from_task(YAML_PATH, infer_engine=None, tokenizer=tokenizer)
output = asyncio.run(agent_generator.run(subset, val_mode=True))

# Print per-sample trajectory summary
print(f"\n{'='*60}")
print("TRAJECTORY SUMMARY")
print(f"{'='*60}")
for iid, trajs in sorted(agent_generator.trajectories.items(), key=lambda x: str(x[0])):
    for tid, traj in trajs.items():
        r = traj.result or {}
        entry_id = subset[int(iid)]["id"] if str(iid).isdigit() else iid
        messages = r.get("messages", [])
        # Collect assistant turns
        turns = [m for m in messages if m.get("role") == "assistant"]
        fn_calls = []
        for m in turns:
            content = m.get("content", "")
            if "<function=" in content:
                # Extract function name
                import re
                fns = re.findall(r"<function=([^>]+)>", content)
                fn_calls.extend(fns)
        reward = r.get("reward", "N/A")
        finish = r.get("finish_reason", "N/A")
        print(f"\n[{entry_id}]")
        print(f"  finish_reason : {finish}")
        print(f"  reward        : {reward}")
        print(f"  #turns        : {len(turns)}")
        print(f"  fn_calls      : {fn_calls}")

elapsed = time.time() - start_time
metrics = output.get("rollout_metrics", {})

# Per-sample reward requires us to re-run per-category;
# here we do a single batch and report aggregate + category breakdown from IDs.
print(f"\n{'='*60}")
print(f"Smoke Test Complete  ({elapsed:.1f}s)")
print(f"{'='*60}")
print(f"raw_reward:        {metrics.get('rollout_metrics/raw_reward', 'N/A'):.3f}")
print(f"num_all_resolved:  {metrics.get('rollout_metrics/num_all_resolved', 'N/A')}")
print(f"num_none_resolved: {metrics.get('rollout_metrics/num_none_resolved', 'N/A')}")
print(f"error_runtime:     {metrics.get('rollout_metrics/error_runtime', 'N/A')}")
print(f"error_evaluation:  {metrics.get('rollout_metrics/error_evaluation', 'N/A')}")
print(f"avg_turns:         {metrics.get('rollout_metrics/avg_turn_assistant', 'N/A'):.1f}")

# Save results
results_file = OUTPUT_DIR / "smoke_results.json"
with open(results_file, "w") as f:
    json.dump({"elapsed_seconds": elapsed, "metrics": {k: v for k, v in metrics.items()}}, f, indent=2)
print(f"\nFull metrics saved to {results_file}")

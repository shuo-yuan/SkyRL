"""
Test run on BFCL_multiple_parallel dataset:
- LLM retriever (k=0, gpt-5-nano)
- Pool contains all tools from all tasks in the batch (not in system prompt, only searchable)
- Batch size=10
- Run 10 tasks
"""
import asyncio, datasets, json, os, time
from pathlib import Path
from typing import List, Dict

import skyrl_agent.tools.bfcl_tool_search

from skyrl_agent.agents.react.react_agent import ReActAgent
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.openai import OpenAIAPIBackend, OpenAIAPIBackendConfig
from skyrl_agent.tasks.bfcl_eval_task import (
    BFCLEvalTask, _normalise_function_list, _parse_json_field,
)

# Configuration
DATASET_PATH = "data/BFCL_multiple_parallel.parquet"
POOL_SIZE = 1  # Note: This is not used anymore - pool now contains all batch tools
BATCH_SIZE = 8  # Test with batch_size=8
N_TASKS = 16  # Run 16 tasks
SEARCH_K = 0  # LLM decides how many tools to return
LLM_MODEL = "gpt-5-nano"
OUTPUT_DIR = "outputs/bfcl_multiple_parallel_test"
LOG_FILE = "outputs/bfcl_multiple_parallel_test/run_debug.log"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


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


def make_agent(idx: int) -> ReActAgent:
    cfg = TrajectoryConfig(
        instance_id=idx, trajectory_id=0,
        sampling_params={"temperature": 1.0, "max_tokens": 32768},
        max_prompt_length=32768,
        qwen3_enable_thinking=False, qwen3_acc_thinking=False,
        tools=["bfcl_tool_search"], max_iterations=10 * BATCH_SIZE,
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


async def run_batch_test():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    print(f"Loading dataset: {DATASET_PATH}")
    ds = datasets.load_dataset("parquet", data_files=DATASET_PATH)["train"]
    all_inst = [dict(r) for r in ds]
    
    # Take first N_TASKS
    test_instances = all_inst[:N_TASKS]
    print(f"Running {len(test_instances)} tasks")
    print(f"Configuration: LLM retriever, k={SEARCH_K}, pool_size={POOL_SIZE}, batch_size={BATCH_SIZE}")
    
    # Note: Pool will be built from all tasks in each batch
    # The pool contains all tools from all tasks, but these tools are NOT in system prompt
    # They are only available for search via bfcl_tool_search
    total = len(all_inst)
    
    t_start = time.time()
    all_rewards = []
    all_results = []
    
    # Process in batches
    for batch_idx in range(0, len(test_instances), BATCH_SIZE):
        batch_instances = test_instances[batch_idx:batch_idx + BATCH_SIZE]
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx // BATCH_SIZE + 1}: Processing {len(batch_instances)} tasks")
        print(f"{'='*60}")
        
        # Build instructions and combined tools for the batch
        batch_instructions = []
        for inst in batch_instances:
            instr = BFCLEvalTask.get_instruction(inst, tool_search_mode=True)
            batch_instructions.append(instr)
        
        # Combine tools from ALL tasks in the batch
        # This creates a pool containing all tools from all tasks in the batch
        # These tools are NOT in the system prompt, but are available for search
        combined_tools = combine_tools(batch_instances)
        
        # Create agent for this batch
        agent = make_agent(0)
        
        # Run batch
        try:
            _, per_calls = await agent.run_batch(
                instructions=batch_instructions,
                instances=batch_instances,
                combined_tool_params=combined_tools,
                use_tool_search=True,
                search_k=SEARCH_K,
                tool_search_retrieval="llm",
                tool_search_llm_model=LLM_MODEL,
            )
            
            # Evaluate results
            batch_rewards = []
            for i, (calls, inst) in enumerate(zip(per_calls, batch_instances)):
                try:
                    r = await BFCLEvalTask.evaluate_result(
                        result=calls, instance=inst,
                        data_source="", instance_id=inst['id'], trajectory_id=0
                    )
                    batch_rewards.append(float(r))
                    all_rewards.append(float(r))
                    
                    result = {
                        "instance_id": inst['id'],
                        "reward": float(r),
                        "calls": calls,
                    }
                    all_results.append(result)
                    
                    print(f"  Task {batch_idx + i + 1}: {inst['id']} → reward={r:.3f}")
                except Exception as e:
                    print(f"  Task {batch_idx + i + 1}: {inst['id']} → ERROR: {e}")
                    batch_rewards.append(0.0)
                    all_rewards.append(0.0)
            
            batch_avg = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            print(f"  Batch avg: {batch_avg:.4f} ({sum(batch_rewards):.0f}/{len(batch_rewards)})")
            
        except Exception as e:
            print(f"  Batch ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - t_start
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    
    # Save results
    results_file = Path(OUTPUT_DIR) / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "dataset": DATASET_PATH,
                "pool_size": POOL_SIZE,
                "batch_size": BATCH_SIZE,
                "n_tasks": N_TASKS,
                "search_k": SEARCH_K,
                "llm_model": LLM_MODEL,
                "retrieval": "llm",
            },
            "results": all_results,
            "summary": {
                "total_tasks": len(all_rewards),
                "avg_reward": avg_reward,
                "total_correct": sum(all_rewards),
                "time_seconds": elapsed,
            }
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total tasks: {len(all_rewards)}")
    print(f"Avg reward: {avg_reward:.4f}")
    print(f"Total correct: {sum(all_rewards):.0f}/{len(all_rewards)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(run_batch_test())

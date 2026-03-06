import os
from pathlib import Path
import asyncio

from skyrl_agent import AutoAgentRunner
from transformers import AutoTokenizer
import datasets

os.environ["OPENAI_API_KEY"] = ""

model = "Qwen/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(model)

dataset = "data/BFCL_all_scoring_tasks.parquet"
dataset = datasets.load_dataset("parquet", data_files=dataset)["train"].select(range(10))
print(dataset[0])

yaml_path = str(Path(__file__).parents[1] / "inference" / "openai_api_gpt_load_finish.yaml")

agent_generator = AutoAgentRunner.from_task(
    yaml_path,
    infer_engine=None,
    tokenizer=tokenizer,
)

output = asyncio.run(agent_generator.run(dataset, val_mode=True))
print(output["rewards"])

from typing import Any, Dict, List, Optional

from skyrl_agent.tasks.base import BaseTask

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response


class BFCLEvalTask(BaseTask):
    """
    Task wrapper that evaluates BFCL multi-turn outputs using BFCL's checker logic.

    Expected instance fields (minimal):
    - ground_truth_list: list[list[str]] (ground truth function calls per turn)
    - question or prompt: list of messages or list of turns (each turn is a list of messages)
    - model_name: name registered in BFCL MODEL_CONFIG_MAPPING (for decoding raw outputs)

    For evaluation:
    - result can be a raw model_result_list (list of turns -> list of step strings),
      or a decoded model_result_list_decoded (list of turns -> list of list[str]).
    - If result is not provided, instance may include model_result_list or
      model_result_list_decoded.
    """

    @classmethod
    async def initialize_runtime(cls, *args, **kwargs) -> Any:
        return {}

    @classmethod
    def get_instruction(cls, instance: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = instance.get("question") or instance.get("prompt") or instance.get("raw_prompt") or ""
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            prompt = prompt[0]
        if isinstance(prompt, dict):
            prompt = [prompt]
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        return prompt

    @classmethod
    def complete_runtime(cls, *args, **kwargs) -> Dict[str, Any]:
        return {}

    @classmethod
    async def evaluate_result(
        cls, result: Any, instance: Any, data_source: str = None, instance_id: int = None, trajectory_id: int = None
    ) -> float:
        if not isinstance(instance, dict):
            return 0.0

        model_result_list = result or instance.get("model_result_list") or instance.get("model_result")
        model_result_list_decoded = instance.get("model_result_list_decoded")
        ground_truth_list = instance.get("ground_truth_list") or instance.get("possible_answer") or instance.get(
            "ground_truth"
        )
        prompt_entry = instance.get("prompt_entry") or instance
        model_name = instance.get("model_name") or instance.get("bfcl_model_name") or "unknown"
        test_category = instance.get("test_category") or data_source or str(instance.get("id", ""))

        if ground_truth_list is None:
            return 0.0

        if model_result_list_decoded is None:
            if _looks_decoded_multi_turn(model_result_list):
                model_result_list_decoded = model_result_list
            else:
                model_result_list_decoded = _decode_multi_turn_results(model_result_list, model_name)

        accuracy_checker_result = multi_turn_checker(
            model_result_list_decoded,
            ground_truth_list,
            prompt_entry,
            test_category,
            model_name,
        )
        if not accuracy_checker_result.get("valid"):
            return 0.0

        if any(isinstance(turn, list) and len(turn) == 0 for turn in ground_truth_list):
            irrelevance_result = multi_turn_irrelevance_checker(model_result_list_decoded, ground_truth_list)
            if not irrelevance_result.get("valid"):
                return 0.0

        return 1.0


def _decode_multi_turn_results(model_result_list: Any, model_name: str) -> List[List[List[str]]]:
    if model_name not in MODEL_CONFIG_MAPPING:
        raise ValueError(f"Unknown BFCL model name '{model_name}'. Provide model_result_list_decoded instead.")

    config = MODEL_CONFIG_MAPPING[model_name]
    handler = config.model_handler(
        model_name=config.model_name,
        temperature=0,
        registry_name=model_name,
        is_fc_model=config.is_fc_model,
    )

    if not isinstance(model_result_list, list):
        raise ValueError("model_result_list must be a list of turns.")

    multi_turn_model_result_list_decoded: List[List[List[str]]] = []
    for single_turn_model_result_list in model_result_list:
        single_turn_model_result_list_decoded: List[List[str]] = []
        for model_result_item in single_turn_model_result_list:
            try:
                decoded_result: List[str] = handler.decode_execute(model_result_item, has_tool_call_tag=False)
                if is_empty_execute_response(decoded_result):
                    continue
                single_turn_model_result_list_decoded.append(decoded_result)
            except Exception:
                continue
        multi_turn_model_result_list_decoded.append(single_turn_model_result_list_decoded)

    return multi_turn_model_result_list_decoded


def _looks_decoded_multi_turn(model_result_list: Any) -> bool:
    if not isinstance(model_result_list, list):
        return False
    if not model_result_list:
        return False
    for turn in model_result_list:
        if not isinstance(turn, list):
            return False
        for step in turn:
            if not isinstance(step, list):
                return False
            for item in step:
                if not isinstance(item, str):
                    return False
    return True

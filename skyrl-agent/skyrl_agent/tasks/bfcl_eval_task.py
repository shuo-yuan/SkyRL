"""
BFCLEvalTask – single-turn BFCL evaluation.

Supports single-turn AST categories (simple, multiple, parallel, live_*, …)
as well as relevance and irrelevance tasks.

Key data contract (per instance / dataset row):
  - id            : str   – e.g. "simple_python_0", "parallel_0"
  - question      : list  – [{role, content}, …]
  - function      : list  – function schemas for this task
  - ground_truth / possible_answer : the expected answer (format varies by category)
"""

import json
from typing import Any, Dict, List, Optional

from skyrl_agent.tasks.base import BaseTask

# ---------------------------------------------------------------------------
# Model name to use when calling BFCL ast_checker.
# Must be a key in bfcl_eval.constants.model_config.MODEL_CONFIG_MAPPING.
#
# We use "Qwen/Qwen3-32B-FC" (underscore_to_dot=False) because:
#  - We use a TEXT-based function-calling format (not native OpenAI tool API)
#  - In text format the model outputs function names verbatim, including "."
#  - underscore_to_dot=False means the checker does NOT convert "." → "_"
#    in ground-truth function names, so the comparison is like-for-like.
# ---------------------------------------------------------------------------
_BFCL_MODEL_NAME = "Qwen/Qwen3-32B-FC"

# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------
_IRRELEVANCE_KEYWORDS = ("irrelevance",)
_RELEVANCE_KEYWORDS = ("relevance",)


def _test_category_from_id(entry_id: str) -> str:
    """Derive the test category from an entry id like 'simple_python_0'."""
    return entry_id.rsplit("_", 1)[0] if entry_id else ""


def _is_irrelevance(cat: str) -> bool:
    return any(k in cat for k in _IRRELEVANCE_KEYWORDS)


def _is_relevance_only(cat: str) -> bool:
    return any(k in cat for k in _RELEVANCE_KEYWORDS) and not _is_irrelevance(cat)


# ---------------------------------------------------------------------------
# Helpers to normalise instance data coming from different dataset formats
# ---------------------------------------------------------------------------

def _to_dict(instance: Any) -> dict:
    """Robustly convert pandas Series / dict-like to a plain dict."""
    if isinstance(instance, dict):
        return instance
    # pandas.Series
    if hasattr(instance, "to_dict"):
        return instance.to_dict()
    return dict(instance) if instance else {}


def _parse_json_field(value: Any) -> Any:
    """If *value* is a JSON-encoded string, parse it; otherwise return as-is."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in ("[", "{"):
            try:
                return json.loads(stripped)
            except Exception:
                pass
    return value


def _normalise_question_to_messages(question: Any) -> List[Dict[str, str]]:
    """
    Normalise the ``question`` field into a flat list of OpenAI-style messages.
    """
    question = _parse_json_field(question)
    if isinstance(question, str):
        return [{"role": "user", "content": question}]
    if isinstance(question, dict):
        return [question]
    if isinstance(question, list):
        msgs: List[Dict[str, str]] = []
        for item in question:
            if isinstance(item, list):
                # First turn only for initial instruction
                for sub in item:
                    if isinstance(sub, dict) and "role" in sub and "content" in sub:
                        msgs.append(sub)
                if msgs:
                    return msgs
            elif isinstance(item, dict) and "role" in item and "content" in item:
                msgs.append(item)
        if msgs:
            return msgs
    return [{"role": "user", "content": str(question)}]


def _normalise_function_list(raw: Any) -> List[dict]:
    """Parse function schemas; handles JSON-encoded strings and nested formats."""
    raw = _parse_json_field(raw)
    if not isinstance(raw, list):
        return []
    result: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # OpenAI ChatCompletionToolParam: {"type":"function","function":{…}}
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            fn = item["function"]
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        # Flat function doc: {"name", "description", "parameters", …}
        elif isinstance(item.get("name"), str):
            result.append(item)
    return result


def _function_list_to_tool_params(funcs: List[dict]) -> List[dict]:
    """Convert flat function docs → OpenAI ChatCompletionToolParam format."""
    result = []
    for fn in funcs:
        params = fn.get("parameters", {})
        if isinstance(params, dict):
            p = dict(params)
            if p.get("type") == "dict":
                p["type"] = "object"
        else:
            p = {"type": "object", "properties": {}, "required": []}
        result.append({
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": p,
            },
        })
    return result


# ---------------------------------------------------------------------------
# Cached ast_checker import (avoid repeated noisy warnings)
# ---------------------------------------------------------------------------
_ast_checker_cache = {"checked": False, "checker": None, "Language": None, "ReturnFormat": None}


def _get_ast_checker():
    """Return ast_checker function or None. Warn only once on failure."""
    if not _ast_checker_cache["checked"]:
        _ast_checker_cache["checked"] = True
        try:
            from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker as _ac
            from bfcl_eval.constants.enums import Language, ReturnFormat
            _ast_checker_cache["checker"] = _ac
            _ast_checker_cache["Language"] = Language
            _ast_checker_cache["ReturnFormat"] = ReturnFormat
        except Exception as e:
            print(f"[BFCLEvalTask] ast_checker unavailable ({e}), will use fallback checker for all AST tasks.")
    return _ast_checker_cache["checker"]


def _get_ast_enums():
    """Return (Language, ReturnFormat) or (None, None)."""
    _get_ast_checker()  # ensure loaded
    return _ast_checker_cache["Language"], _ast_checker_cache["ReturnFormat"]


# ---------------------------------------------------------------------------
# BFCLEvalTask
# ---------------------------------------------------------------------------

class BFCLEvalTask(BaseTask):
    """
    Single-turn BFCL evaluation task.

    ``get_instruction`` exposes the first (and only) turn's user messages.
    The instance's ``function`` list is injected into the agent as BFCL tool
    params (via ``_active_bfcl_tool_params``).

    ``evaluate_result`` routes to AST checker or relevance/irrelevance checker
    based on the task category inferred from ``instance["id"]``.
    """

    @classmethod
    async def initialize_runtime(cls, *args, **kwargs) -> Any:
        return {}

    @classmethod
    def get_instruction(cls, instance: Any) -> List[Dict[str, str]]:
        inst = _to_dict(instance)
        question = inst.get("question") or inst.get("prompt") or inst.get("raw_prompt") or ""
        msgs = _normalise_question_to_messages(question)

        # Prepend a system message following the official BFCL prompt style,
        # but using our <function=...> tool call format (injected later by
        # convert_fncall_messages_to_non_fncall_messages).
        system_prompt = (
            "You are an expert in composing functions. "
            "You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
            "If none of the functions can be used, point it out. "
            "If the given question lacks the parameters required by the function, also point it out. "
            "You should only return the function calls in your response. "
            "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
            "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
            "Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task. "
            "If the task requires multiple parallel function calls, you may output all of them in a single response, "
            "one after another in the specified format."
        )
        # If there's already a system message, prepend to its content.
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = system_prompt + "\n\n" + msgs[0]["content"]
        else:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        return msgs

    @classmethod
    def complete_runtime(cls, *args, **kwargs) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # Evaluation – route to the right BFCL checker
    # ------------------------------------------------------------------

    @classmethod
    async def evaluate_result(
        cls,
        result: Any,
        instance: Any,
        data_source: str = None,
        instance_id: int = None,
        trajectory_id: int = None,
    ) -> float:
        inst = _to_dict(instance)
        entry_id: str = str(inst.get("id", ""))
        if not entry_id:
            print(f"[BFCLEvalTask] Warning: instance has no 'id' field, cannot evaluate.")
            return 0.0

        test_category = _test_category_from_id(entry_id)

        if _is_irrelevance(test_category) or _is_relevance_only(test_category):
            return cls._evaluate_relevance(result, inst, test_category)
        else:
            # Single-turn AST check (simple_python, multiple, parallel, live_*, …)
            return cls._evaluate_ast(result, inst, test_category)

    # ------------------------------------------------------------------
    # Single-turn AST evaluation (照搬 ast_checker)
    # ------------------------------------------------------------------

    @classmethod
    def _evaluate_ast(cls, result: Any, inst: dict, test_category: str) -> float:
        prompt_function = _normalise_function_list(_parse_json_field(inst.get("function", [])))
        entry_id = str(inst.get("id", ""))
        ground_truth = _get_ground_truth(inst, entry_id)
        if ground_truth is None:
            return 0.0

        model_result_decoded = _decode_result_to_ast(result)
        if not model_result_decoded:
            return 0.0

        # Try the official BFCL ast_checker first.
        ast_checker = _get_ast_checker()

        # ast_checker requires at least one function description; if the instance
        # has no function field (e.g. in unit tests), fall through to fallback.
        if ast_checker is not None and prompt_function:
            Language, ReturnFormat = _get_ast_enums()
            try:
                if "java" in test_category:
                    language = Language.JAVA
                elif "javascript" in test_category:
                    language = Language.JAVASCRIPT
                else:
                    language = Language.PYTHON

                checker_result = ast_checker(
                    prompt_function,
                    model_result_decoded,
                    ground_truth,
                    language,
                    test_category,
                    _BFCL_MODEL_NAME,
                )
                return 1.0 if checker_result.get("valid") else 0.0
            except Exception as e:
                print(f"[BFCLEvalTask] ast_checker error for {entry_id} ({test_category}): {e}")
                # Fall through to fallback

        # Fallback: simple function-name + argument matching when ast_checker
        # cannot be imported (e.g. tree-sitter version mismatch).
        return _fallback_ast_check(model_result_decoded, ground_truth)

    # ------------------------------------------------------------------
    # Relevance / Irrelevance evaluation
    # ------------------------------------------------------------------

    @classmethod
    def _evaluate_relevance(cls, result: Any, inst: dict, test_category: str) -> float:
        """
        Relevance: model should output a valid function call → 1.0
        Irrelevance: model should NOT output a valid function call → 1.0
        """
        has_function_call = _agent_made_bfcl_call(result)

        if _is_irrelevance(test_category):
            return 1.0 if not has_function_call else 0.0
        else:
            return 1.0 if has_function_call else 0.0


# ---------------------------------------------------------------------------
# Result decoding helpers
# ---------------------------------------------------------------------------

def _get_ground_truth(inst: dict, entry_id: str = "") -> Any:
    """Extract ground truth from instance or load from BFCL library."""
    # Try instance fields first
    for key in ("ground_truth", "possible_answer", "ground_truth_list"):
        gt = inst.get(key)
        if gt is not None:
            return _parse_json_field(gt)

    # Fall back to loading from BFCL library by entry id
    if not entry_id:
        entry_id = str(inst.get("id", ""))
    if not entry_id:
        return None
    test_category = _test_category_from_id(entry_id)
    if not test_category:
        return None

    try:
        from bfcl_eval.utils import load_ground_truth_entry
        gt_entries = load_ground_truth_entry(test_category)
        for gt_entry in gt_entries:
            if isinstance(gt_entry, dict) and gt_entry.get("id") == entry_id:
                return gt_entry.get("ground_truth")
    except Exception as e:
        print(f"[BFCLEvalTask] Failed to load ground truth from BFCL lib for {entry_id}: {e}")

    return None


def _decode_result_to_ast(result: Any) -> Optional[List[dict]]:
    """
    Decode agent result for AST checker.

    The result from the agent for BFCL single-turn tasks is ``_bfcl_recorded_calls``:
    a list of ``{"function": name, "arguments": {…}}`` dicts.

    AST checker expects: ``[{func_name: {param: value, …}}, …]``.
    """
    calls = result
    if not isinstance(calls, list):
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except Exception:
                return None
        if not isinstance(calls, list):
            return None

    decoded: List[dict] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn_name = call.get("function", "")
        fn_args = call.get("arguments", {})
        if not fn_name or fn_name in ("finish", "load", "evict"):
            continue
        if isinstance(fn_args, str):
            try:
                fn_args = json.loads(fn_args)
            except Exception:
                fn_args = {}
        if not isinstance(fn_args, dict):
            fn_args = {}
        decoded.append({fn_name: fn_args})

    return decoded if decoded else None


def _agent_made_bfcl_call(result: Any) -> bool:
    """
    Check if the agent made any BFCL-domain function calls.
    result is _bfcl_recorded_calls: list of {"function": ..., "arguments": ...}.
    """
    if not isinstance(result, list) or len(result) == 0:
        return False
    for call in result:
        if isinstance(call, dict):
            fn = call.get("function", "")
            if fn and fn not in ("finish", "load", "evict"):
                return True
    return False


def _fallback_ast_check(model_calls: Any, ground_truth: Any) -> float:
    """
    Lightweight AST check when the official ``ast_checker`` is unavailable.

    BFCL ground truth for single-turn AST tasks is a list of dicts like:
      [{"func_name": {"param": [accepted_values], ...}}, ...]

    The model_calls should be decoded into a similar structure.
    We check:
      1) Every ground-truth function name appears in model calls.
      2) Required parameters match one of the accepted values.
    """
    if not isinstance(ground_truth, list) or not isinstance(model_calls, list):
        return 0.0

    # Build a lookup of model calls by function name
    model_by_name: dict = {}
    for call in model_calls:
        if isinstance(call, dict):
            for fn_name, fn_args in call.items():
                model_by_name.setdefault(fn_name, []).append(fn_args if isinstance(fn_args, dict) else {})

    for gt_item in ground_truth:
        if not isinstance(gt_item, dict):
            continue
        for fn_name, expected_args in gt_item.items():
            if fn_name not in model_by_name:
                return 0.0
            if not isinstance(expected_args, dict):
                continue
            # Check at least one model call matches all expected args
            matched = False
            for model_args in model_by_name[fn_name]:
                all_match = True
                for param_name, accepted_values in expected_args.items():
                    if not isinstance(accepted_values, list):
                        accepted_values = [accepted_values]
                    model_val = model_args.get(param_name)
                    if not any(_values_match(model_val, av) for av in accepted_values):
                        all_match = False
                        break
                if all_match:
                    matched = True
                    break
            if not matched:
                return 0.0

    return 1.0


def _values_match(model_val: Any, expected_val: Any) -> bool:
    """Flexible value comparison for BFCL AST checking."""
    if model_val == expected_val:
        return True
    # Empty string matches None/missing
    if expected_val == "" and model_val is None:
        return True
    if model_val == "" and expected_val is None:
        return True
    # Numeric comparison
    try:
        if float(model_val) == float(expected_val):
            return True
    except (TypeError, ValueError):
        pass
    # String comparison (case-insensitive for some fields)
    if str(model_val).strip().lower() == str(expected_val).strip().lower():
        return True
    return False

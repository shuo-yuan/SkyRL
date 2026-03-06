import copy
import importlib
import inspect
import json
import re
from typing import Any

from .bfcl_backend_config import CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES


def execute_multi_turn_func_call(
    func_call_list: list[str],
    initial_config: dict,
    involved_classes: list[str],
    model_name: str,
    test_entry_id: str,
    long_context: bool = False,
    is_evaL_run: bool = False,
    instance_store: dict[str, Any] | None = None,
) -> tuple[list[str], dict]:
    if is_evaL_run:
        model_name += "_eval"

    if instance_store is None:
        instance_store = {}

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        if class_name not in CLASS_FILE_PATH_MAPPING:
            continue
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        instance_name = f"{model_name}_{test_entry_id}_{class_name}_instance"
        instance_name = re.sub(r"[-./]", "_", instance_name)
        if instance_name not in instance_store:
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                class_instance._load_scenario(copy.deepcopy(class_initial_config), long_context=long_context)
            instance_store[instance_name] = class_instance
        else:
            class_instance = instance_store[instance_name]

        involved_instances[class_name] = class_instance

        for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            class_method_name_mapping[method_name] = instance_name

    execution_results = []
    for func_call in func_call_list:
        func_call = _process_method_calls(func_call, class_method_name_mapping)

        try:
            func_call_copy = func_call
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            if func_call_copy in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            eval_globals = {"__builtins__": __builtins__}
            eval_globals.update(instance_store)
            func_call_result = eval(func_call, eval_globals)

            if isinstance(func_call_result, str):
                pass
            elif isinstance(func_call_result, dict):
                try:
                    func_call_result = json.dumps(func_call_result)
                except Exception:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)

            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances


def _process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"
    processed_string = re.sub(pattern, replace_function, function_call_string)

    return processed_string

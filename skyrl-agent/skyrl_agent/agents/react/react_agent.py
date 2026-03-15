import json
import re
import copy
from typing import Any, List, Dict, Tuple, Optional
from collections import defaultdict
from uuid import uuid4
import traceback
import os
from skyrl_agent.functional.utils import (
    Transition,
    record_transition,
    StepResult,
    StepException,
    ContextWindowExceeded,
    ParseError,
    NoToolCall,
    ToolExecutionFailed,
)

from skyrl_agent.functional.history import (
    MessageHistory,
    MessageEncoder,
    parse_tool_call,
    extract_tool_info,
    check_truncated_tool_call,
    format_output_preview,
)
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.base import AsyncInferBackend
from skyrl_agent.tools.base import TOOL_REGISTRY
from skyrl_agent.dispatcher.async_utils import call_sync_from_async
from .messages import (
    TOOL_CALL_PARSE_ERROR_GUIDANCE,
    NO_TOOL_CALL_DETECTED_GUIDANCE,
    TOOL_INVOCATION_ERROR_GUIDANCE,
    get_turn_reminder_text,
)
from skyrl_agent.functional.function_calling import (
    convert_fncall_messages_to_non_fncall_messages,
    _extract_and_validate_params,
    FunctionCallValidationError,
)


class ReActAgent:
    def __init__(
        self,
        traj_config: TrajectoryConfig,
        infer_engine: AsyncInferBackend,
        tokenizer: Any,
    ) -> None:
        self.cfg = traj_config
        self.tokenizer = tokenizer
        self.infer_engine = infer_engine
        self.sampling_params = traj_config.sampling_params

        self.max_prompt_length = traj_config.max_prompt_length
        self.qwen3_enable_thinking = traj_config.qwen3_enable_thinking
        self.qwen3_acc_thinking = traj_config.qwen3_acc_thinking

        self.instance_id = traj_config.instance_id
        self.trajectory_id = traj_config.trajectory_id
        self.max_iterations = traj_config.max_iterations
        self.early_step_threshold = traj_config.early_step_threshold
        self.enable_turn_reminder = getattr(traj_config, "enable_turn_reminder", False)

        self.step_count = 0
        self.history = MessageHistory()
        self.tools = {}
        self.tool_params = []
        self.transitions: List[Transition] = []  # Record transitions per LLM call

        self.agent_id = uuid4().hex
        self._register_tools(traj_config.tools)

        # Message encoder
        self.message_encoder = MessageEncoder(
            tokenizer, qwen3_enable_thinking=self.qwen3_enable_thinking, qwen3_acc_thinking=self.qwen3_acc_thinking
        )

        self._active_bfcl_tool_params: List[Dict[str, Any]] = []
        self._bfcl_recorded_calls: List[Dict[str, Any]] = []  # Track BFCL function calls for evaluation
        self.bfcl_tool_params: List[Dict[str, Any]] = self._load_bfcl_tool_params(
            getattr(traj_config, "bfcl_tool_params_path", None)
        )
        self._active_bfcl_tool_params = self.bfcl_tool_params

        # fold_tool_info: collapse completed-task history into a one-line summary
        self._fold_tool_info: bool = getattr(traj_config, "fold_tool_info", False)
        # Index of the current task's question message in history.
        # Used by the folding logic to know which messages to keep.
        self._bfcl_task_question_msg_idx: int = 1   # default: user msg after system
        # Set by _try_advance_bfcl_batch when folding occurs; tells step() to
        # skip _append_tool_output for the just-completed domain call.
        self._fold_skip_append: bool = False

        # Batch BFCL mode state (used by run_batch())
        self._bfcl_batch_mode: bool = False
        self._bfcl_batch_pending: List[Tuple[Dict, List[Dict]]] = []  # (instance, instruction) pairs
        self._bfcl_all_task_results: List[List[Dict]] = []  # completed per-task call lists
        self._bfcl_task_idx: int = 0  # current task index (0-based)

        # Tool-search mode state (used when use_tool_search=True in run_batch())
        # When enabled, combined tools are held in _bfcl_tool_pool instead of
        # being placed in _active_bfcl_tool_params upfront.  The model must call
        # bfcl_tool_search to retrieve tools before using them.
        self._bfcl_tool_search_enabled: bool = False
        self._bfcl_tool_pool: List[Dict] = []       # all available tools for this batch
        self._bfcl_tool_search_k: int = 3           # k used as fallback inside _llm_retrieve
        self._bfcl_bm25_k: int = 4                  # k specifically for BM25 retrieval
        # Tools retrieved in the CURRENT task only (reset on task transition).
        # Used for parse-validation so the model cannot accidentally call a tool
        # from a prior task.  _active_bfcl_tool_params keeps the full history
        # (for future direct reuse without re-searching).
        self._bfcl_current_task_tools: List[Dict] = []

        # In-context
        self.put_tools_in_context = True

        self.prompt_token_len = 0
        self.response_token_len = 0

        # Debug and profiling flags/counters
        self._debug = bool(traj_config.debug_log)
        self._profile_enabled = bool(traj_config.profile_tools)
        self._tool_calls_total: int = 0
        self._tool_calls_by_name: Dict[str, int] = defaultdict(int)

    def _register_tools(self, tools: List[str]) -> None:
        """Register a list of tool instances."""
        print(f"[Register Tools] {tools}")
        for name in tools:
            if name not in TOOL_REGISTRY:
                raise ValueError(f"Unknown tool '{name}'. Must be one of: {list(TOOL_REGISTRY)}")
            tool = TOOL_REGISTRY[name]()
            # Enforce unique function names per agent for ALL tools
            if tool.name in self.tools:
                raise ValueError(
                    f"Duplicate tool function name '{tool.name}' for this agent. "
                    f"Tool function names must be unique per agent."
                )
            self.tools[tool.name] = tool
            self.tool_params.append(tool.get_tool_param())

    @record_transition
    async def _generate_with_recording(self, input_ids, sampling_params, request_id, messages=None):
        """LLM generation wrapper that records transitions.

        This method is decorated to automatically capture:
        - input_ids: tokens fed to the LLM
        - output_tokens: tokens generated by the LLM
        - logprobs: log probabilities of generated tokens
        """
        if getattr(self.infer_engine, "use_chat_api", False):
            return await self.infer_engine.async_generate_prompts(
                prompts=messages,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        return await self.infer_engine.async_generate_ids(
            input_ids=input_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        )

    def _prepare_llm_input(self) -> tuple[List[int], Dict]:
        """Prepare input_ids and sampling params for LLM using incremental encoding.

        When history is reset, retokenizes everything. Otherwise, performs incremental
        encoding by appending new messages to existing tokens.

        Returns:
            Tuple of (input_ids, sampling_params)
        """

        # Check if history was reset - store flag before clearing it
        history_was_reset = self.history.was_reset()
        if history_was_reset:
            self.prompt_token_len = 0
            self.history.clear_reset_flag()

        # Add turn reminder to history (optional via config)
        if self.enable_turn_reminder:
            remaining_steps = self.max_iterations - self.step_count + 1
            reminder_text = get_turn_reminder_text(
                self.step_count,
                remaining_steps,
                early_step_threshold=self.early_step_threshold,
            )
            self.history.add_turn_reminder(reminder_text)

        # Determine if we should retokenize everything or use incremental encoding
        is_prompt = len(self.history) == 2  # system + user (with reminder appended)
        should_retokenize = is_prompt or history_was_reset or not self.transitions
        # Use display params (hides retrieved tools in tool-search mode).
        active_tool_params = self._get_display_tool_params()

        if should_retokenize:
            # Retokenize everything (first message or after history reset)
            input_ids = self.message_encoder.encode_messages(
                self.history.messages,
                active_tool_params,
                is_first_message=True,
            )
            self.prompt_token_len = len(input_ids)
        else:
            # Incremental encoding: append new messages to existing tokens
            last_transition = self.transitions[-1]
            if not last_transition.ac.token_ids:
                # Retokenize the action response_str if serving endpoints do not return token_ids
                message = [{"role": "assistant", "content": last_transition.ac.text}]
                last_transition.ac.token_ids = self.message_encoder.encode_messages(
                    message, active_tool_params, add_generation=False
                )

            # Encode only the new observation message(s)
            new_obs_ids = self.message_encoder.encode_messages(
                [self.history.messages[-1]], active_tool_params, add_generation=True
            )

            # Build input_ids incrementally: previous observation + previous action + new observation
            input_ids = last_transition.ob.input_ids + last_transition.ac.token_ids + new_obs_ids

        self.response_token_len = len(input_ids) - self.prompt_token_len

        # Prepare sampling params
        sampling_params = copy.deepcopy(self.sampling_params)
        sampling_params["max_tokens"] = self.max_prompt_length - self.response_token_len

        return input_ids, sampling_params

    def _handle_parse_error(self, error: str) -> None:
        """Handle tool call parsing error and raise ParseError."""
        print(f"[Agent Step Error] Converter failed to parse tool call: {error}")
        guidance = TOOL_CALL_PARSE_ERROR_GUIDANCE.format(error=error)

        self.history.add_tool_error(error)
        self.history.add_user_guidance(guidance)

        raise ParseError()

    def _handle_no_tool_call(self, response_str: str) -> None:
        """Handle case when no tool call is detected and raise NoToolCall."""
        print(f"[Agent Step {self.step_count}] No tool call found in response")

        # Check if response was likely truncated during a tool call
        if check_truncated_tool_call(response_str):
            print("[ERROR] Tool call appears incomplete - likely truncated!")
            print(f"[ERROR] Last 500 chars: {response_str[-500:]}")

        self.history.add_user_guidance(NO_TOOL_CALL_DETECTED_GUIDANCE)
        raise NoToolCall()

    async def _execute_tool(self, tool_name: str, tool_args: Dict, tool_call_id: str) -> Any:
        """Execute a tool and return output.

        Raises:
            ToolExecutionFailed: If tool execution fails
        """
        tool = self.tools[tool_name]

        try:
            output = await call_sync_from_async(
                tool.call,
                tool_args,
                agent=self,
                trajectory_id=self.trajectory_id,
            )

            # Record profiling stats if enabled
            if self._profile_enabled:
                try:
                    self._tool_calls_total += 1
                    if tool_name:
                        self._tool_calls_by_name[tool_name] += 1
                except Exception:
                    pass

            return output

        except Exception as e:
            # Tool invocation failed
            error_str = str(e)
            try:
                self.history.add_tool_error(error_str, tool_call_id)
            except Exception:
                self.history.add_tool_error("Tool failed with an exception.", tool_call_id)

            self.history.add_user_guidance(TOOL_INVOCATION_ERROR_GUIDANCE)
            raise ToolExecutionFailed()

    def _append_tool_output(self, output: Any, tool_call_id: str) -> None:
        """Append tool output to message history.

        Args:
            output: Tool output to append
            tool_call_id: ID of the tool call
        """
        try:
            self.history.add_tool_response(output, tool_call_id)

            if self._debug:
                preview = format_output_preview(output)
                print(f"[Tool Output Preview] {preview}")

        except Exception as e:
            print(f"[Agent Step Error] Error appending tool output to messages: {str(e)}")
            self.history.add_tool_error(str(e), tool_call_id)

    def _load_bfcl_tool_params(self, path: str | None) -> List[Dict[str, Any]]:
        default_path = os.path.join(os.path.dirname(__file__), "..", "tools", "bfcl_tool_params.json")
        if not path:
            path = default_path
        path = os.path.abspath(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _normalize_bfcl_function_entry(self, entry: Dict[str, Any]) -> Dict[str, Any] | None:
        """Normalize BFCL function schema into OpenAI function-tool format."""
        if not isinstance(entry, dict):
            return None

        # Already in OpenAI function-tool format.
        if entry.get("type") == "function" and isinstance(entry.get("function"), dict):
            fn = entry["function"]
            if isinstance(fn.get("name"), str):
                return entry
            return None

        name = entry.get("name")
        if not isinstance(name, str) or not name:
            return None
        params = entry.get("parameters", {})
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}, "required": []}
        if params.get("type") == "dict" or not isinstance(params.get("type"), str):
            params = copy.deepcopy(params)
            params["type"] = "object"
        if not isinstance(params.get("properties"), dict):
            params = copy.deepcopy(params)
            params["properties"] = {}
        if not isinstance(params.get("required"), list):
            params = copy.deepcopy(params)
            params["required"] = []

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": entry.get("description", ""),
                "parameters": params,
            },
        }

    def _resolve_bfcl_tool_params_from_instance(self, instance: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        """Prefer per-instance BFCL function schema when available."""
        if instance is None or not hasattr(instance, "get"):
            return self.bfcl_tool_params

        raw = instance.get("function")
        if raw is None:
            return self.bfcl_tool_params
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return self.bfcl_tool_params

        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return self.bfcl_tool_params

        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in raw:
            norm = self._normalize_bfcl_function_entry(item)
            if not norm:
                continue
            name = norm["function"]["name"]
            if name in seen:
                continue
            seen.add(name)
            normalized.append(norm)

        return normalized if normalized else self.bfcl_tool_params

    def _get_active_tool_params(self) -> List[Dict[str, Any]]:
        """Return tool schema used for PARSING model responses.

        Includes registered tools (e.g. bfcl_tool_search) plus:
          - In tool-search mode: ONLY tools retrieved in the CURRENT task
            (_bfcl_current_task_tools).  Using the full history
            (_active_bfcl_tool_params) would allow the model to accidentally
            call tools from prior tasks and pass parse validation silently.
          - Otherwise: all retrieved BFCL tool params.
        """
        merged: List[Dict[str, Any]] = list(self.tool_params)
        seen_names = set()
        for tp in merged:
            fn = tp.get("function") if isinstance(tp, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str):
                seen_names.add(name)

        # Use full accumulated tools for parse validation.
        # (Restricting to current-task only would prevent correct tool reuse.)
        source = self._active_bfcl_tool_params
        for tp in source:
            if not isinstance(tp, dict):
                continue
            fn = tp.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            if not isinstance(name, str) or name in seen_names:
                continue
            seen_names.add(name)
            merged.append(tp)
        return merged

    def _get_display_tool_params(self) -> List[Dict[str, Any]]:
        """Return tool schema used for building the system prompt (shown to model).

        In tool-search mode the system prompt always shows ONLY bfcl_tool_search,
        regardless of what has been retrieved so far.  Previously retrieved tools
        are communicated via the search result guidance message (not the system
        prompt), and are available in _get_active_tool_params() for parse
        validation so the model can call them directly based on prior guidance.

        Outside tool-search mode this is identical to _get_active_tool_params().
        """
        if self._bfcl_tool_search_enabled:
            # System prompt: only the registered support tools (bfcl_tool_search).
            return list(self.tool_params)
        return self._get_active_tool_params()

    def _is_parallel_bfcl(self) -> bool:
        """Return True if this is a parallel BFCL category (needs multiple calls per turn)."""
        inst = getattr(self, "instance", None)
        if inst is None:
            return False
        _get = inst.get if hasattr(inst, "get") else lambda k, d=None: getattr(inst, k, d)
        entry_id = str(_get("id", "") or "")
        cat = entry_id.rsplit("_", 1)[0] if entry_id else ""
        _PARALLEL_CATEGORIES = {"parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"}
        return cat in _PARALLEL_CATEGORIES

    # Regex patterns for BFCL function call parsing (reuse same format as function_calling.py)
    _BFCL_FN_REGEX = re.compile(r"<function=([^>]+)>\n(.*?)</function>", re.DOTALL)
    _BFCL_PARAM_REGEX = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

    def _extract_bfcl_calls_from_response(
        self, response_str: str
    ) -> List[Tuple[str, Dict]]:
        """
        Extract ALL <function=...> calls from a BFCL response string.

        Used for parallel BFCL tasks where the model may output multiple function
        calls in a single response. Returns a list of (fn_name, fn_args) tuples
        for every valid <function=...> block found in the response.

        Safe for non-BFCL contexts: called only when has_bfcl_tools=True and
        has_registered_tools=False.
        """
        active_tools = self._get_active_tool_params()
        calls: List[Tuple[str, Dict]] = []

        for fn_match in self._BFCL_FN_REGEX.finditer(response_str):
            fn_name = fn_match.group(1)
            fn_body = fn_match.group(2)

            # Skip system tools
            if fn_name in {"finish", "load", "evict"}:
                continue

            # Look up schema
            matching_tool = next(
                (
                    t["function"]
                    for t in active_tools
                    if t.get("type") == "function" and t["function"]["name"] == fn_name
                ),
                None,
            )
            if not matching_tool:
                continue

            # Parse parameters
            param_matches = self._BFCL_PARAM_REGEX.finditer(fn_body)
            try:
                args = _extract_and_validate_params(matching_tool, param_matches, fn_name)
            except (FunctionCallValidationError, Exception):
                # Malformed call — skip rather than fail the whole step
                continue

            calls.append((fn_name, args))

        return calls

    def _try_advance_bfcl_batch(self) -> bool:
        """Save current task's calls and advance to the next task in batch mode.

        Returns True if successfully advanced (more tasks remain),
        False if all tasks are done.
        """
        # Save current task result before clearing
        self._bfcl_all_task_results.append(list(self._bfcl_recorded_calls))
        self._bfcl_recorded_calls = []

        if not self._bfcl_batch_pending:
            return False  # all tasks done

        self._bfcl_task_idx += 1
        next_instance, next_instruction = self._bfcl_batch_pending.pop(0)

        # If fold_tool_info is enabled, collapse the completed task's messages
        # (search calls, search results, domain calls, tool responses, etc.)
        # into a single summary line.  This keeps the context short.
        # Also set _fold_skip_append so step() skips _append_tool_output for
        # the domain call that just completed (its result is in the summary).
        if self._fold_tool_info:
            completed_calls = self._bfcl_all_task_results[-1]  # just saved
            fn_names = [c.get("function", "?") for c in completed_calls]
            if fn_names:
                summary = (
                    f"[Task {self._bfcl_task_idx} complete. "
                    f"Called: {', '.join(fn_names)}.]"
                )
            else:
                summary = f"[Task {self._bfcl_task_idx} complete. No function called.]"
            # Keep everything up to and including the task's question message,
            # then append the summary.  Everything in between (tool calls, search
            # results, intermediate assistant messages) is discarded.
            self.history.messages = (
                self.history.messages[: self._bfcl_task_question_msg_idx + 1]
                + [{"role": "user", "content": summary}]
            )
            self._fold_skip_append = True   # signal step() to skip _append_tool_output

        # Update instance so _is_parallel_bfcl() uses the correct category
        self.instance = next_instance
        # Normal mode: keep combined _active_bfcl_tool_params (all upfront).

        # Extract user-role content from the next task's instruction
        next_user_msgs = [m for m in next_instruction if m.get("role") == "user"]
        next_question = next_user_msgs[0]["content"] if next_user_msgs else ""

        # Inject transition message into history
        self.history.add_user_guidance(
            f"Task {self._bfcl_task_idx} done.\n\n"
            f"Task {self._bfcl_task_idx + 1}:\n{next_question}"
        )
        # Track where the new task's question is so folding knows what to keep.
        self._bfcl_task_question_msg_idx = len(self.history.messages) - 1
        print(
            f"[Batch] Saved task {self._bfcl_task_idx} result "
            f"({len(self._bfcl_all_task_results[-1])} calls), "
            f"advancing to task {self._bfcl_task_idx + 1}."
        )
        return True

    def _add_bfcl_tools_to_active(self, new_tools: List[Dict]) -> None:
        """Add tools to _active_bfcl_tool_params (deduplicated by function name).

        Called by BFCLToolSearchTool after retrieval.
        Also adds tools to _bfcl_current_task_tools (current-task scope,
        reset on task transition) so parse-validation stays scoped to the
        current task and does not accidentally accept calls to prior-task tools.
        """
        existing_names = {
            t.get("function", {}).get("name", "")
            for t in self._active_bfcl_tool_params
            if isinstance(t, dict)
        }
        current_names = {
            t.get("function", {}).get("name", "")
            for t in self._bfcl_current_task_tools
            if isinstance(t, dict)
        }
        for tool in new_tools:
            name = tool.get("function", {}).get("name", "") if isinstance(tool, dict) else ""
            if name:
                if name not in existing_names:
                    self._active_bfcl_tool_params.append(tool)
                    existing_names.add(name)
                if name not in current_names:
                    self._bfcl_current_task_tools.append(tool)
                    current_names.add(name)

    async def _execute_bfcl_tool(self, tool_name: str, tool_args: Any, tool_call_id: str) -> Any:
        """Execute a BFCL single-turn function call.

        For single-turn tasks the function call format is what matters for evaluation
        (AST checker), not the execution result. We record the call and return a
        success acknowledgement so the model can proceed or stop.
        """
        try:
            args_display = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
            return json.dumps(
                {
                    "status": "success",
                    "function": tool_name,
                    "arguments": tool_args if isinstance(tool_args, dict) else str(tool_args),
                    "message": (
                        f"Function {tool_name}({args_display}) called successfully. "
                        "If there are more required calls, make them now. Otherwise stop."
                    ),
                }
            )
        except Exception as e:
            error_str = str(e)
            try:
                self.history.add_tool_error(error_str, tool_call_id)
            except Exception:
                self.history.add_tool_error("Tool failed with an exception.", tool_call_id)
            self.history.add_user_guidance(TOOL_INVOCATION_ERROR_GUIDANCE)
            raise ToolExecutionFailed()

    def _add_loaded_tools_to_buffer(self, output: Any) -> None:
        return

    def _evict_tools_from_buffer(self, output: Any) -> None:
        return

    async def step(self):
        """Execute one agent step: LLM generation -> tool call -> tool execution.

        Returns:
            Tuple of (done, finish_reason, result)
        """
        self.step_count += 1
        print(f"[Agent Step {self.step_count}] instance={self.instance_id} traj={self.trajectory_id}")

        result = None

        try:
            # 1. Prepare LLM input
            if getattr(self.infer_engine, "use_chat_api", False):
                sampling_params = copy.deepcopy(self.sampling_params)
                # Use display params: in tool-search mode this hides retrieved
                # domain tools from the system prompt (they appear only in the
                # search result guidance message).
                messages = convert_fncall_messages_to_non_fncall_messages(
                    self.history.messages, self._get_display_tool_params()
                )
                input_ids = []
            else:
                input_ids, sampling_params = self._prepare_llm_input()
                messages = None

            # Check context window
            if self.response_token_len >= self.max_prompt_length:
                print("[Agent Step] Stopping reason: context_window_exceeded. Stopping agent.")
                raise ContextWindowExceeded()

            # 2. Generate LLM response
            response_str, meta_info = await self._generate_with_recording(
                input_ids=input_ids,
                sampling_params=sampling_params,
                request_id=self.agent_id,
                messages=messages,
            )
            stop_reason = meta_info["finish_reason"]
            print(f"[Agent Step {self.step_count}] LLM response: {response_str}. Stop reason: {stop_reason}")

            # Add assistant message to history
            self.history.add_assistant(response_str)

            # Check if generation stopped due to length
            if stop_reason == "length":
                print(f"[Agent Step] Stopping reason: {stop_reason}. Stopping agent.")
                raise ContextWindowExceeded()

            # 3. Parse tool call from response
            parse_tool_params = self._get_active_tool_params()
            tool_call, parse_error = parse_tool_call(response_str, parse_tool_params)

            # Determine if this is a BFCL-only run (no registered tools like finish).
            # bfcl_tool_search is a support tool, not a "real" registered tool for
            # the purpose of BFCL detection — exclude it from has_registered_tools.
            _BFCL_SUPPORT_TOOLS = frozenset({"bfcl_tool_search"})
            has_bfcl_tools = bool(self._active_bfcl_tool_params) or self._bfcl_tool_search_enabled
            has_registered_tools = bool(set(self.tools) - _BFCL_SUPPORT_TOOLS)

            # Handle parse error
            if parse_error:
                # For BFCL tasks without registered tools, parse errors on non-function-call
                # output just mean the model is done (BFCL style: no function call = turn over).
                if has_bfcl_tools and not has_registered_tools:
                    print(f"[Agent Step {self.step_count}] BFCL: model output is not a function call, stopping.")
                    if self._bfcl_batch_mode and self._try_advance_bfcl_batch():
                        result = StepResult.continuing(response_str)
                    else:
                        result = StepResult.finished("BFCL_TURN_DONE", self._bfcl_recorded_calls)
                else:
                    self._handle_parse_error(parse_error)

            # Handle no tool call detected
            elif tool_call is None:
                if has_bfcl_tools and not has_registered_tools:
                    # BFCL style: no function call output = turn/task is done
                    print(f"[Agent Step {self.step_count}] BFCL: no function call detected, stopping.")
                    if self._bfcl_batch_mode and self._try_advance_bfcl_batch():
                        result = StepResult.continuing(response_str)
                    else:
                        result = StepResult.finished("BFCL_TURN_DONE", self._bfcl_recorded_calls)
                elif not has_registered_tools and not has_bfcl_tools:
                    print(f"[Agent Step {self.step_count}] No tools provided, returning response.")
                    result = StepResult.finished("FINISH", response_str)
                else:
                    self._handle_no_tool_call(response_str)

            else:
                # 4. Extract tool information
                tool_name, tool_args = extract_tool_info(tool_call)
                tool_call_id = tool_call.get("id")
                # 5. Execute tool
                # bfcl_tool_search is treated as a support / registered tool:
                # it updates _active_bfcl_tool_params and then we continue.
                if tool_name in {"load", "evict", "finish", "bfcl_tool_search"}:
                    if tool_name not in self.tools:
                        self.history.add_user_guidance(json.dumps({"error": f"Tool '{tool_name}' not found."}))
                        result = StepResult.continuing(response_str)
                    else:
                        output = await self._execute_tool(tool_name, tool_args, tool_call_id)
                else:
                    output = await self._execute_bfcl_tool(tool_name, tool_args, tool_call_id)
                    # Record the BFCL function call for evaluation
                    recorded_args = tool_args
                    if isinstance(recorded_args, str):
                        try:
                            recorded_args = json.loads(recorded_args)
                        except Exception:
                            recorded_args = {}
                    if not isinstance(recorded_args, dict):
                        recorded_args = {}
                    self._bfcl_recorded_calls.append({
                        "function": tool_name,
                        "arguments": recorded_args,
                    })

                    # Also capture any additional <function=...> calls in the same
                    # response (batch parallel mode: model outputs all calls at once).
                    # _extract_bfcl_calls_from_response returns ALL calls in order;
                    # index 0 is the call we already recorded above, so we skip it.
                    # Only applies to BFCL context; safe no-op for non-BFCL tasks.
                    _extra_parallel_calls: List[Tuple[str, Dict]] = []
                    if has_bfcl_tools and not has_registered_tools:
                        all_calls = self._extract_bfcl_calls_from_response(response_str)
                        _extra_parallel_calls = all_calls[1:]
                        for extra_fn_name, extra_fn_args in _extra_parallel_calls:
                            self._bfcl_recorded_calls.append({
                                "function": extra_fn_name,
                                "arguments": extra_fn_args,
                            })
                            print(
                                f"[Agent Step {self.step_count}] BFCL: captured additional "
                                f"parallel call {extra_fn_name} from same response."
                            )

                if tool_name in {"load", "evict", "finish"} and tool_name not in self.tools:
                    pass
                # 6. Check if finish tool was called
                elif tool_name == "finish":
                    print(f"[Agent Step {self.step_count}] Finish tool called. Stopping agent.")
                    result = StepResult.finished("FINISH_TOOL", output)
                elif tool_name == "bfcl_tool_search":
                    # Search / retrieval tool: always continue so the model can
                    # call the domain function it just retrieved.  Never triggers
                    # BFCL stop logic.
                    # We use add_user_guidance (not _append_tool_output) to avoid
                    # double-JSON-encoding the output string.
                    print(f"[Agent Step {self.step_count}] bfcl_tool_search executed, continuing.")
                    if output is not None:
                        print(f"[Tool Output step {self.step_count}] (bfcl_tool_search) {output[:200]}…")
                        # Parse the output and produce a readable plain-text message
                        try:
                            parsed = json.loads(output)
                            found = parsed.get("found", 0)
                            desc  = parsed.get("tool_descriptions", "")
                            if found > 0 and desc:
                                guidance = (
                                    f"Search complete. Found {found} function(s):\n\n"
                                    f"{desc}\n"
                                    "Now call the appropriate function above using the standard format."
                                )
                            else:
                                guidance = parsed.get("message", str(output))
                        except Exception:
                            guidance = str(output)
                        print(f"[Guidance to model step {self.step_count}] {guidance[:300]}…")
                        self.history.add_user_guidance(guidance)
                    result = StepResult.continuing(response_str)
                else:
                    # For non-parallel BFCL (simple, multiple, …): stop after first call.
                    # For parallel BFCL:
                    #   - Batch mode (model output N calls at once): all already recorded,
                    #     stop immediately to avoid the model re-outputting the same calls.
                    #   - Sequential mode (model output 1 call): continue until model stops.
                    if has_bfcl_tools and not has_registered_tools and not self._is_parallel_bfcl():
                        # Non-parallel: stop after first call (or advance to next batch task)
                        if self._bfcl_batch_mode and self._try_advance_bfcl_batch():
                            print(f"[Agent Step {self.step_count}] BFCL batch: task done, advancing.")
                            result = StepResult.continuing(response_str)
                        else:
                            print(f"[Agent Step {self.step_count}] BFCL single-step: stopping after first call.")
                            result = StepResult.finished("BFCL_SINGLE_TURN_DONE", self._bfcl_recorded_calls)
                    elif has_bfcl_tools and not has_registered_tools and _extra_parallel_calls:
                        # Batch parallel: all calls captured in one shot — stop (or advance).
                        if self._bfcl_batch_mode and self._try_advance_bfcl_batch():
                            print(f"[Agent Step {self.step_count}] BFCL batch parallel: task done, advancing.")
                            result = StepResult.continuing(response_str)
                        else:
                            print(
                                f"[Agent Step {self.step_count}] BFCL batch parallel: "
                                f"captured {1 + len(_extra_parallel_calls)} calls, stopping."
                            )
                            result = StepResult.finished("BFCL_PARALLEL_BATCH_DONE", self._bfcl_recorded_calls)
                    else:
                        result = StepResult.continuing(response_str)

                    # 7. Append tool output to history only if output is not None
                    # and fold did not already clean up this task's history.
                    if output is not None and not self._fold_skip_append:
                        print(f"[Tool Output step {self.step_count}] {output}")
                        self._append_tool_output(output, tool_call_id)
                        if tool_name == "load":
                            self._add_loaded_tools_to_buffer(output)
                        elif tool_name == "evict":
                            self._evict_tools_from_buffer(output)
                    else:
                        if not self._fold_skip_append:
                            print(f"[Tool Output step {self.step_count}] No output (feedback embedded in user message)")
                    self._fold_skip_append = False   # reset after each step

        except StepException as e:
            # Handle expected control flow exceptions
            result = e.step_result

        except Exception as e:
            # Handle unexpected errors
            print(f"[Agent Step Error] Error during step: {str(e)}")
            result = StepResult.finished(f"error: {str(e)}", None)

        # Single exit point
        return result.to_tuple()

    async def run(self, instruction: List[Dict], instance: Dict | None = None) -> List[str]:
        """Run the agent to completion with the provided user input.

        Optionally accepts an instance payload for tools (stored on self.instance).
        Only single-turn execution is supported.
        """
        self.instance = instance
        self._active_bfcl_tool_params = self._resolve_bfcl_tool_params_from_instance(instance)

        _BFCL_SUPPORT_TOOLS = frozenset({"bfcl_tool_search"})
        has_bfcl_tools = bool(self._active_bfcl_tool_params) or self._bfcl_tool_search_enabled
        has_registered_tools = bool(set(self.tools) - _BFCL_SUPPORT_TOOLS)
        is_bfcl_single_turn = has_bfcl_tools and not has_registered_tools

        self._init_message(instruction)
        result = None
        finish_reason = None
        while self.step_count < self.max_iterations:
            try:
                done, finish_reason, result = await self.step()
                if done:
                    break
            except Exception as e:
                finish_reason = f"error: {str(e)}"
                print(f"[Agent Run Error] Exception during step: {str(e)}")
                print(traceback.format_exc())
                break
        else:
            finish_reason = "max_iterations_reached"

        # Normalise BFCL finish reasons
        if is_bfcl_single_turn and finish_reason in (
            "BFCL_TURN_DONE", "BFCL_SINGLE_TURN_DONE", "BFCL_PARALLEL_BATCH_DONE"
        ):
            finish_reason = "BFCL_SINGLE_TURN_DONE"
            result = self._bfcl_recorded_calls

        print("[Agent Run] Final messages:", self.get_messages())
        return finish_reason, result

    async def run_batch(
        self,
        instructions: List[List[Dict]],
        instances: List[Dict],
        combined_tool_params: Optional[List[Dict]] = None,
        use_tool_search: bool = False,
        tool_search_k: int = 3,
        bm25_k: int = 4,
        tool_search_retrieval: str = "bm25",
        tool_search_llm_model: str = "gpt-5-nano",
        tool_search_llm_base_url: str = "https://api.openai.com/v1",
    ) -> Tuple[str, List[List[Dict]]]:
        """Run N BFCL tasks sequentially in one conversation.

        Args:
            instructions:             Per-task instruction message lists.
            instances:                Per-task dataset row dicts.
            combined_tool_params:     Pre-combined tool params; None = first instance only.
            use_tool_search:          False (default) = all tools upfront (original behaviour).
                                      True = only bfcl_tool_search in system prompt; model
                                      must call it to retrieve tools.
            tool_search_k:            Tools returned per search query (default 3).
            tool_search_retrieval:    "bm25" (default) or "llm".
            tool_search_llm_model:    LLM model for retrieval (default "gpt-5-nano").
            tool_search_llm_base_url: Base URL for LLM retrieval API.

        Returns:
            ("BFCL_BATCH_DONE", [task0_calls, task1_calls, ..., taskN_calls])
        """
        if not instances:
            return "BFCL_BATCH_DONE", []

        # Initialise batch state
        self._bfcl_batch_mode = True
        self._bfcl_batch_pending = list(zip(instances[1:], instructions[1:]))
        self._bfcl_all_task_results = []
        self._bfcl_task_idx = 0
        self.instance = instances[0]

        # ── Tool-search mode vs. upfront-tools mode ──────────────
        all_tools = combined_tool_params or self._resolve_bfcl_tool_params_from_instance(instances[0])

        if use_tool_search:
            # Store all tools in the retrieval pool; start with empty active params.
            # The model must call bfcl_tool_search before using any tool.
            self._bfcl_tool_search_enabled = True
            self._bfcl_tool_pool = list(all_tools)
            self._bfcl_tool_search_k = tool_search_k
            self._bfcl_bm25_k = bm25_k
            # Retrieval settings read by BFCLToolSearchTool.call()
            self._bfcl_tool_search_retrieval = tool_search_retrieval
            self._bfcl_tool_search_llm_model = tool_search_llm_model
            self._bfcl_tool_search_llm_base_url = tool_search_llm_base_url
            self._active_bfcl_tool_params = []
            self._bfcl_current_task_tools = []   # reset for new batch
        else:
            # Original behaviour: all tools visible upfront.
            self._bfcl_tool_search_enabled = False
            self._bfcl_tool_pool = []
            self._active_bfcl_tool_params = list(all_tools)

        # Bootstrap with first task's instruction
        self._init_message(instructions[0])
        # After _init_message the history is [system, user].
        # The user message (index 1) is the first task's question.
        self._bfcl_task_question_msg_idx = 1

        finish_reason = "max_iterations_reached"
        while self.step_count < self.max_iterations:
            try:
                done, finish_reason, result = await self.step()
                if done:
                    break
            except Exception as e:
                print(f"[Batch Run Error] {e}")
                print(traceback.format_exc())
                finish_reason = f"error: {e}"
                break

        # If the last task ended via max_iterations (not via stop signal),
        # save whatever calls were recorded.
        if len(self._bfcl_all_task_results) < len(instances):
            self._bfcl_all_task_results.append(list(self._bfcl_recorded_calls))

        # Pad missing tasks with empty call lists
        while len(self._bfcl_all_task_results) < len(instances):
            self._bfcl_all_task_results.append([])

        self._bfcl_batch_mode = False
        self._bfcl_tool_search_enabled = False
        self._bfcl_tool_pool = []
        self._bfcl_current_task_tools = []
        final_results = self._bfcl_all_task_results[: len(instances)]
        print(f"[Agent Batch] Done. {len(instances)} tasks, finish_reason={finish_reason}")
        print(f"[Agent Batch] Per-task call counts: "
              f"{[len(r) for r in final_results]}")
        for i, (task_calls, inst) in enumerate(zip(final_results, instances)):
            entry_id = str(inst.get('id', f'task{i}')) if isinstance(inst, dict) else f'task{i}'
            fn_names = [c.get('function','?') for c in task_calls]
            print(f"[Agent Batch]   task {i} ({entry_id}): {fn_names}")
        return "BFCL_BATCH_DONE", final_results

    def get_messages(self) -> List[dict]:
        return convert_fncall_messages_to_non_fncall_messages(self.history.messages, self._get_display_tool_params())

    def get_transitions(self) -> List[Transition]:
        """Return the list of transitions recorded during agent execution.

        Each transition contains:
        - ob: Observation with input_ids (tokens fed to LLM)
        - ac: TokensWithLogprobs with output_tokens, logprobs, and generated text
        - reward: Float reward value (default 0.0, can be updated based on outcomes)
        - episode_done: Boolean indicating if episode finished
        - metrics: Dict with finish_reason, response_length, and other metadata
        """
        return self.transitions

    def _init_message(self, instruction: List[Dict]) -> None:
        """Initialize the agent's message history with the provided instruction.

        Automatically collects system prompt prefixes from registered tools and prepends them
        to the system message if present.
        """
        if not isinstance(instruction, list):
            raise ValueError("Instruction must be a list of messages.")

        for msg in instruction:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must be a dictionary with 'role' and 'content'.")

        # Collect system prompt prefixes from registered tools
        tool_prefixes = []
        for tool_name, tool in self.tools.items():
            prefix = tool.get_system_prompt_prefix()
            if prefix:
                tool_prefixes.append(prefix)

        # Prepend tool prefixes to system message if any exist
        if tool_prefixes:
            processed_instruction = copy.deepcopy(instruction)
            # Combine all tool prefixes
            combined_prefix = "\n\n---\n\n".join(tool_prefixes)

            # Find the first system message
            system_msg_found = False
            for msg in processed_instruction:
                if msg.get("role") == "system":
                    # Prepend tool prefixes to existing system message
                    msg["content"] = combined_prefix + "\n\n---\n\n" + msg["content"]
                    system_msg_found = True
                    break

            # If no system message exists, create one at the beginning
            if not system_msg_found:
                processed_instruction.insert(0, {"role": "system", "content": combined_prefix})

            self.history.initialize(processed_instruction)
        else:
            # Ensure a system message exists so tool descriptions can be appended later
            processed_instruction = copy.deepcopy(instruction)
            if not any(msg.get("role") == "system" for msg in processed_instruction):
                processed_instruction.insert(0, {"role": "system", "content": ""})
            self.history.initialize(processed_instruction)

    # Expose profiling snapshot for upstream aggregation
    def get_tool_profile(self) -> Dict[str, Any]:
        if not self._profile_enabled:
            return None
        try:
            return {
                "tool_calls_total": int(self._tool_calls_total),
                "tool_calls_by_name": dict(self._tool_calls_by_name),
            }
        except Exception:
            return None


if __name__ == "__main__":
    # Example usage for testing
    from skyrl_agent.config.configuration_utils import TrajectoryConfig
    from skyrl_agent.integrations.openai import OpenAIBackend, OpenAIBackendConfig
    from transformers import AutoTokenizer
    import asyncio

    # Load tokenizer and model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define trajectory configuration
    traj_config = TrajectoryConfig(
        instance_id="test_instance",
        trajectory_id="test_trajectory",
        sampling_params={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048,
        },
        max_prompt_length=12048,
        qwen3_enable_thinking=True,
        tools=["finish", "code_interpreter"],
        max_iterations=5,
        agent_cls="skyrl_agent.agents.react.ReActAgent",  # Use ReActAgent for testing
    )

    backend_config = OpenAIBackendConfig(
        model_name=model_name,
        # change this to your desired url and port
        api_url="http://localhost:8000",
    )
    # TODO: model_name need not be in config
    infer_engine = OpenAIBackend(infer_engine=None, cfg=backend_config)

    # Create the ReAct agent
    # Test for with tools
    agent = ReActAgent(
        traj_config=traj_config,
        infer_engine=infer_engine,
        tokenizer=tokenizer,
    )

    # Define a sample instruction
    instruction = [
        {"content": "Please reason step by step, and put your final answer within \\boxed{}.", "role": "system"},
        {
            "content": "Points $A,B,C,D,E$ and $F$ lie, in that order, on $\\overline{AF}$, dividing it into five segments, each of length 1. Point $G$ is not on line $AF$. Point $H$ lies on $\\overline{GD}$, and point $J$ lies on $\\overline{GF}$. The line segments $\\overline{HC}, \\overline{JE},$ and $\\overline{AG}$ are parallel. Find $HC/JE$.",
            "role": "user",
        },
    ]

    # Run the agent
    finish_reason, result = asyncio.run(agent.run(instruction))

    print(agent.get_messages())
    print(f"Finish Reason: {finish_reason}")
    print(f"Result: {result}")

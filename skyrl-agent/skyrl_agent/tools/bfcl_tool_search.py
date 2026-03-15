"""
BFCLToolSearchTool – in-context tool retrieval for batch BFCL.

Supports two retrieval methods (set via cfg["retrieval_method"]):

  "bm25" (default)
      Fast keyword / BM25-style scoring over tool names and descriptions.
      No extra API call needed.

  "llm"
      Uses an LLM (default: gpt-5-nano, configurable via cfg["llm_model"])
      to pick the most relevant tools.  Requires OPENAI_API_KEY.
      The LLM is called synchronously inside the tool's call() method.

Usage pattern (both modes):
  1. System prompt: only `bfcl_tool_search` is listed
  2. Model calls `bfcl_tool_search(query="...")` to find relevant tools
  3. Found tools are added to active params; model can call them normally
  4. Pool resets between batch tasks

Configuration (via cfg dict or agent attributes):
  retrieval_method  "bm25" | "llm"         (default "bm25")
  llm_model         any OpenAI model name  (default "gpt-5-nano")
  llm_base_url      OpenAI API base URL    (default https://api.openai.com/v1)

K (how many tools are returned) is set by agent._bfcl_tool_search_k,
which in turn comes from TOOL_SEARCH_K in the runner script.

Registration: "bfcl_tool_search"
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import BaseTool, register_tool


@register_tool("bfcl_tool_search")
class BFCLToolSearchTool(BaseTool):
    """Search the batch's combined function pool by keyword query.

    Returns the top-k most relevant function schemas and makes them available
    for the model to call in subsequent steps.

    cfg keys (all optional):
        retrieval_method  "bm25" (default) or "llm"
        llm_model         OpenAI model name (default "gpt-5-nano")
        llm_base_url      Base URL for OpenAI-compatible API
                          (default https://api.openai.com/v1)
    """

    name = "bfcl_tool_search"
    description = (
        "Search for available functions by keyword. "
        "Call this tool whenever you need to use a function that is not yet available "
        "in your current tool list — for example, at the start of a new task or "
        "when you realize you need an additional function to complete the current task. "
        "After the search, call the function(s) returned by the search. "
        "If a task requires multiple function calls, search for each function you need "
        "before calling it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Short noun-phrase describing what the function does, "
                    "using the key domain words. "
                    "Good: 'product of prime numbers', 'triangle area from base height'. "
                    "Avoid generic words like 'calculate' or 'find' — focus on the subject matter."
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.retrieval_method: str = self.cfg.get("retrieval_method", "bm25")
        self.llm_model: str = self.cfg.get("llm_model", "gpt-5-nano")
        self.llm_base_url: str = self.cfg.get(
            "llm_base_url", "https://api.openai.com/v1"
        )

    def get_system_prompt_prefix(self) -> Optional[str]:
        return None

    def call(
        self,
        tool_args: Dict[str, Any],
        agent: Any = None,
        trajectory_id: Any = None,
        **kwargs,
    ) -> str:
        import json as _json
        # If tool_args arrived as a JSON string (double-encoded), parse it first.
        if isinstance(tool_args, str):
            try:
                tool_args = _json.loads(tool_args)
            except Exception:
                return _json.dumps({"error": f"tool_args is a string and could not be parsed: {tool_args!r}"})
        query = str(tool_args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query must be non-empty."})

        if agent is None or not getattr(agent, "_bfcl_tool_search_enabled", False):
            return json.dumps({"error": "BFCLToolSearch not configured on this agent."})

        pool: List[Dict] = getattr(agent, "_bfcl_tool_pool", [])
        if not pool:
            return json.dumps({
                "found": 0,
                "message": (
                    "No functions are available for this task. "
                    "If the task cannot be completed with available tools, "
                    "explain this clearly and do NOT call any function."
                ),
            })

        # General k (used as BM25 fallback inside _llm_retrieve if LLM fails)
        k = int(getattr(agent, "_bfcl_tool_search_k", 3))
        # BM25-specific k — controls how many tools BM25 retrieves directly.
        # Default is 4; can be overridden via agent._bfcl_bm25_k or run_batch(bm25_k=...).
        bm25_k = int(getattr(agent, "_bfcl_bm25_k", 4))

        # ── Dispatch to retrieval method ──────────────────────────
        # Agent-level attributes override the tool's own cfg so the
        # runner can inject settings without touching TOOL_REGISTRY.
        method   = getattr(agent, "_bfcl_tool_search_retrieval", self.retrieval_method)
        llm_mdl  = getattr(agent, "_bfcl_tool_search_llm_model", self.llm_model)
        llm_url  = getattr(agent, "_bfcl_tool_search_llm_base_url", self.llm_base_url)

        if method == "ground_truth":
            retrieved = _ground_truth_retrieve(pool, agent)
        elif method == "llm":
            retrieved = _llm_retrieve(pool, query, k, model=llm_mdl, base_url=llm_url)
        elif method == "bm25":
            retrieved = _bm25_retrieve(pool, query, bm25_k)
        else:
            # Fallback: treat unknown methods as bm25
            retrieved = _bm25_retrieve(pool, query, bm25_k)

        if not retrieved:
            return json.dumps({"found": 0, "message": "No matching tools found."})

        # Add retrieved tools to the agent's active BFCL params
        agent._add_bfcl_tools_to_active(retrieved)

        # Build human-readable description for the model
        from skyrl_agent.functional.function_calling import convert_tools_to_description
        desc = convert_tools_to_description(retrieved)

        return json.dumps({
            "found": len(retrieved),
            "retrieval_method": method,   # actual method used (not the default)
            "message": (
                f"Found {len(retrieved)} function(s). "
                "Call the appropriate function now."
            ),
            "tool_descriptions": desc,
        })


# ── Ground-truth retrieval ────────────────────────────────────────

def _ground_truth_retrieve(pool: List[Dict], agent: Any) -> List[Dict]:
    """Return the tools that are actually required for the current task.

    Reads the ground-truth function names from ``agent.instance["function"]``
    and returns exactly those tools from the pool.  This simulates perfect
    retrieval and provides an accuracy upper bound for the TST setting.

    Falls back to BM25 (k=3) if the instance or its function field cannot
    be resolved.
    """
    inst = getattr(agent, "instance", None)
    if inst is None:
        return _bm25_retrieve(pool, "", 3)

    try:
        import json as _json
        from skyrl_agent.tasks.bfcl_eval_task import (
            _normalise_function_list,
            _parse_json_field,
        )

        _get = inst.get if hasattr(inst, "get") else lambda k, d=None: getattr(inst, k, d)
        raw = _get("function", [])
        fns = _normalise_function_list(_parse_json_field(raw))
        gt_names = {fn.get("name", "") for fn in fns if fn.get("name")}
    except Exception:
        return _bm25_retrieve(pool, "", 3)

    if not gt_names:
        # No functions needed for this task (irrelevance-style)
        return []

    # Return all pool tools whose name is in the GT set
    retrieved = [t for t in pool if t.get("function", {}).get("name", "") in gt_names]
    return retrieved


# ── LLM-based retrieval ───────────────────────────────────────────

def _llm_retrieve(
    pool: List[Dict],
    query: str,
    k: int,
    model: str = "gpt-5-nano",
    base_url: str = "https://api.openai.com/v1",
) -> List[Dict]:
    """Use an LLM to pick all relevant tools from the pool for the given query.

    The LLM decides how many functions to return — it returns ALL functions
    that are relevant to the query, not a fixed number k.  k is kept as a
    parameter for the BM25 fallback only.
    Falls back to BM25 on any error.
    """
    if not pool:
        return []

    # Build a compact catalogue for the LLM
    catalogue_lines = []
    name_to_tool: Dict[str, Dict] = {}
    for i, tool in enumerate(pool, 1):
        fn = tool.get("function", tool)
        name = fn.get("name", f"tool_{i}")
        desc = fn.get("description", "").split(".")[0]   # first sentence only
        catalogue_lines.append(f"{i}. {name}: {desc}")
        name_to_tool[name] = tool

    catalogue = "\n".join(catalogue_lines)
    prompt = (
        f"Query: {query}\n\n"
        f"Available functions:\n{catalogue}\n\n"
        f"List ALL function names that are relevant to this query. "
        f"If none are relevant, output 'NONE'. "
        f"Output only the function names, one per line, no extra text."
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            import openai as _openai
            api_key = os.getenv("OPENAI_API_KEY", "")
            client = _openai.OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=32768,
            )
            raw = resp.choices[0].message.content or ""
            # LLM may output "NONE" when nothing is relevant
            if raw.strip().upper() == "NONE":
                return []
            # Parse function names from the response (one per line).
            # Handle "1. function_name" or bare "function_name" formats.
            chosen_names = []
            for line in raw.strip().splitlines():
                line = line.strip().lstrip("0123456789.-) ").rstrip(".,;:")
                if line and line.upper() != "NONE":
                    chosen_names.append(line)
            # No [:k] limit — LLM decides how many are relevant
            retrieved = [name_to_tool[n] for n in chosen_names if n in name_to_tool]
            if retrieved:
                return retrieved
            # Empty or unparseable response — retry
            if attempt < max_retries - 1:
                continue
            print(f"[BFCLToolSearch/llm] Could not parse response after {max_retries} attempts: "
                  f"{raw!r}, falling back to BM25")
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"[BFCLToolSearch/llm] LLM retrieval error after {max_retries} attempts: "
                  f"{e}, falling back to BM25")

    return _bm25_retrieve(pool, query, k)


# ── BM25-style keyword retrieval ──────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.split(r"\W+", text.lower())


def _bm25_retrieve(
    pool: List[Dict],
    query: str,
    k: int,
    b: float = 0.75,
    k1: float = 1.5,
) -> List[Dict]:
    """Simple BM25-inspired scoring over tool name + description + param names."""
    query_tokens = set(t for t in _tokenize(query) if t)
    if not query_tokens:
        return pool[:k]

    scores: List[tuple] = []
    for tool in pool:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        param_text = " ".join(
            pname + " " + pinfo.get("description", "")
            for pname, pinfo in params.get("properties", {}).items()
            if isinstance(pinfo, dict)
        )
        text = f"{name} {desc} {param_text}"
        doc_tokens = [t for t in _tokenize(text) if t]
        doc_set = set(doc_tokens)
        dl = len(doc_tokens)

        # BM25 term frequency saturation (avgdl=50 assumed)
        score = sum(
            (freq := doc_tokens.count(qt)) * (k1 + 1)
            / (freq + k1 * (1 - b + b * dl / 50))
            for qt in query_tokens
            if qt in doc_set
        )
        # Bonus for exact name prefix match
        if name.lower().startswith(query.lower()[:8]) or query.lower()[:8] in name.lower():
            score += 5.0

        scores.append((score, tool))

    scores.sort(key=lambda x: -x[0])
    return [t for _, t in scores[:k]]

import json
import os
import re
from typing import Dict, List, Optional, Union

from openai import OpenAI

from .base import BaseTool, register_tool


@register_tool("load")
class LoadTool(BaseTool):
    name = "load"
    description = (
        "Use this tool first. It retrieves relevant functions from an external tool document based on a query. "
        "All functions returned by this tool can be called directly in the next step. "
        "You must use functions from the tool document to help answer the question. "
        "Do not answer directly without using retrieved functions. "
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short description of the function you need."},
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        base_dir = os.path.dirname(__file__)
        # Retrieve over the full BFCL function corpus by default.
        self.tool_document_path = self.cfg.get(
            "tool_document_path",
            os.path.join(base_dir, "BFCL_all_unique_functions.json"),
        )
        self.max_tools = int(self.cfg.get("max_tools", 3))
        self.llm_model = self.cfg.get("llm_model", "gpt-5")
        self.llm_temperature = self.cfg.get("llm_temperature", 0)
        self.candidate_pool_size = int(self.cfg.get("candidate_pool_size", 120))
        self.openai_api_key = self.cfg.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
        self.openai_base_url = self.cfg.get("openai_base_url", os.getenv("OPENAI_API_BASE"))
        self._tool_docs_cache: Optional[List[dict]] = None

    def get_system_prompt_prefix(self) -> Optional[str]:
        """Return a system prompt prefix describing how to use this tool."""
        return None

    def _load_tool_document(self) -> List[dict]:
        if self._tool_docs_cache is not None:
            return self._tool_docs_cache
        if not os.path.exists(self.tool_document_path):
            return []
        try:
            with open(self.tool_document_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            docs: List[dict] = []
            if isinstance(data, dict) and isinstance(data.get("tools"), list):
                # {"tools":[{"name","description","parameters","response"}]}
                for item in data["tools"]:
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        docs.append(item)
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    # OpenAI function-tool schema entry
                    if item.get("type") == "function" and isinstance(item.get("function"), dict):
                        fn = item["function"]
                        name = fn.get("name")
                        if isinstance(name, str):
                            docs.append(
                                {
                                    "name": name,
                                    "description": fn.get("description", ""),
                                    "parameters": fn.get("parameters", {}),
                                    "response": {},
                                }
                            )
                    # Already flat function doc
                    elif isinstance(item.get("name"), str):
                        docs.append(item)

            self._tool_docs_cache = docs
            return docs
        except (OSError, json.JSONDecodeError):
            return []

    def _load_name_description_map(self) -> Dict[str, str]:
        name_desc: Dict[str, str] = {}
        for item in self._load_tool_document():
            name = item.get("name")
            if isinstance(name, str) and name and name not in name_desc:
                name_desc[name] = str(item.get("description", ""))
        return name_desc

    def _select_candidate_names(self, query: str, candidate_pool_size: int) -> List[str]:
        name_desc_map = self._load_name_description_map()
        if not name_desc_map:
            return []
        query_lower = query.lower()
        terms = [t for t in re.split(r"\W+", query_lower) if t]
        scored = []
        for name, desc in name_desc_map.items():
            name_lower = name.lower()
            desc_lower = desc.lower()
            score = 0
            if query_lower in name_lower:
                score += 8
            if query_lower and query_lower in desc_lower:
                score += 4
            for term in terms:
                if term in name_lower:
                    score += 3
                if term in desc_lower:
                    score += 1
            scored.append((score, name))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [name for _, name in scored[: max(candidate_pool_size, self.max_tools)]]
        if not top:
            top = list(name_desc_map.keys())[: max(candidate_pool_size, self.max_tools)]
        return top

    def _parse_name_list(self, raw_content: str) -> List[str]:
        if not raw_content:
            return []
        content = raw_content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, str)]
        except Exception:
            pass
        return []

    def _select_names_with_llm(self, query: str, candidate_names: List[str], k: int) -> List[str]:
        name_desc_map = self._load_name_description_map()
        if not candidate_names:
            return []
        options = "\n".join([f"- {name}: {name_desc_map.get(name, '')}" for name in candidate_names])
        prompt = (
            "You are selecting function names for tool-use.\n"
            f"User query: {query}\n"
            f"Select up to {k} names from the candidate list that best match the query intent.\n"
            "Rules:\n"
            "1) Return ONLY a JSON array of function names.\n"
            "2) Every name must come from the candidates.\n"
            "3) Prefer precise and directly relevant functions.\n\n"
            f"Candidates:\n{options}\n"
        )

        try:
            client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_temperature,
            )
            raw_content = response.choices[0].message.content or ""
            selected_names = self._parse_name_list(raw_content)
        except Exception:
            selected_names = []

        candidate_set = set(candidate_names)
        selected = []
        for name in selected_names:
            if name in candidate_set and name not in selected:
                selected.append(name)
            if len(selected) >= k:
                break
        return selected

    def _lookup_full_docs(self, selected_names: List[str], k: int) -> List[dict]:
        if not selected_names or k <= 0:
            return []
        full_docs = self._load_tool_document()
        if not full_docs:
            return []
        name_desc_map = self._load_name_description_map()
        docs_by_name: Dict[str, List[dict]] = {}
        for item in full_docs:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str):
                continue
            docs_by_name.setdefault(name, []).append(item)

        results: List[dict] = []
        for name in selected_names:
            candidates = docs_by_name.get(name, [])
            if not candidates:
                continue
            target_desc = name_desc_map.get(name, "")
            chosen = None
            if target_desc:
                for doc in candidates:
                    if doc.get("description", "") == target_desc:
                        chosen = doc
                        break
            if chosen is None:
                chosen = candidates[0]
            results.append(chosen)
            if len(results) >= k:
                break
        return results

    def retrieve(self, query: str, k: int) -> List[dict]:
        """
        Retrieve up to k relevant functions for query.
        Step 1: use GPT-5 to choose function names from the full BFCL function document.
        Step 2: fetch full function docs from the same source.
        """
        if k <= 0:
            return []
        candidates = self._select_candidate_names(query, self.candidate_pool_size)
        selected_names = self._select_names_with_llm(query, candidates, k)
        if not selected_names:
            selected_names = candidates[:k]
        return self._lookup_full_docs(selected_names, k)

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}

        query = params.get("query", "")
        k = self.max_tools if isinstance(self.max_tools, int) else 1
        if k < 0:
            return {"error": "max_tools must be a non-negative integer"}
        tools = []
        for tool in self.retrieve(query, k):
            tools.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                    "response": tool.get("response", {}),
                }
            )
        return {
            "tools": tools,
            "instruction": "Use the returned functions to solve the task; do not answer without using them first.",
        }

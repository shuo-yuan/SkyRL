import json
import os
from typing import List, Optional, Union

from openai import OpenAI

from .base import BaseTool, register_tool


@register_tool("load")
class LoadTool(BaseTool):
    name = "load"
    description = (
        "Loads tools from an external tool document based on provided queries. "
        "Each query is mapped to a single tool via the retrieve() function."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Tool query."},
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.tool_document_path = self.cfg.get(
            "tool_document_path",
            os.path.join(os.path.dirname(__file__), "tool_document.json"),
        )
        self.max_tools = 5
        self.llm_model = self.cfg.get("llm_model", "gpt-4o-mini")
        self.llm_temperature = self.cfg.get("llm_temperature", 0)
        self.openai_api_key = self.cfg.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
        self.openai_base_url = self.cfg.get("openai_base_url", os.getenv("OPENAI_API_BASE"))

    def _load_tool_document(self) -> List[dict]:
        if not os.path.exists(self.tool_document_path):
            return []
        try:
            with open(self.tool_document_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("tools", []) if isinstance(data, dict) else []
        except (OSError, json.JSONDecodeError):
            return []

    def retrieve(self, query: str, k: int) -> List[dict]:
        """
        Retrieve up to k tools from the tool document for the given query.

        TODO: Replace this stub with embedding-based retrieval or external LLM selection.
        """
        tools = self._load_tool_document()
        if not tools or k <= 0:
            return []
        tool_descriptions = "\n".join(
            [f"- {tool.get('name', '')}: {tool.get('description', '')}" for tool in tools]
        )
        prompt = (
            "You are selecting tools for a query.\n"
            f"Query: {query}\n"
            f"Select up to {k} tool names from the list below.\n"
            "Return only a JSON array of tool names.\n\n"
            f"Tools:\n{tool_descriptions}\n"
        )

        try:
            client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_temperature,
            )
            raw_content = response.choices[0].message.content
            selected_names = json.loads(raw_content)
        except Exception:
            selected_names = []

        if not isinstance(selected_names, list):
            selected_names = []

        selected_set = {name for name in selected_names if isinstance(name, str)}
        if not selected_set:
            return tools[:k]

        selected_tools = [tool for tool in tools if tool.get("name") in selected_set]
        return selected_tools[:k]

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
        return {"tools": tools}

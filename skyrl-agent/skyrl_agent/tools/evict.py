from typing import Union

from .base import BaseTool, register_tool


@register_tool("evict")
class EvictTool(BaseTool):
    name = "evict"
    description = "Accepts tool names and returns them unchanged for downstream eviction handling."
    parameters = {
        "type": "object",
        "properties": {
            "tool_names": {"type": "array", "items": {"type": "string"}, "description": "Tool names to evict."}
        },
        "required": ["tool_names"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}

        tool_names = params.get("tool_names", [])
        return {"tool_names": tool_names}

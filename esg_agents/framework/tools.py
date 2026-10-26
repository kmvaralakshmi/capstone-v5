"""Tool definition shared by every agent.

A Tool is a named, documented, schema-described Python callable. The agent's
planner (an LLM or the agent's own deterministic policy) selects tools by
name; the framework validates the arguments and executes the callable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    # JSON-schema "properties" dict, e.g. {"company_code": {"type": "string"}}
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def spec(self) -> Dict[str, Any]:
        """OpenAI Responses-API function specification."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.properties,
                "required": self.required,
                "additionalProperties": False,
            },
        }

    def validate(self, args: Dict[str, Any]) -> str | None:
        """Return an error string if args are unusable, else None."""
        if not isinstance(args, dict):
            return "arguments must be a JSON object"
        missing = [r for r in self.required if r not in args]
        if missing:
            return f"missing required argument(s): {missing}"
        unknown = [k for k in args if k not in self.properties]
        if unknown:
            return f"unknown argument(s): {unknown}"
        for key, schema in self.properties.items():
            if key in args and "enum" in schema and args[key] not in schema["enum"]:
                return f"argument '{key}' must be one of {schema['enum']}"
        return None

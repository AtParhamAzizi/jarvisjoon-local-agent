from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any]
    error_type: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "ok": self.ok,
            **self.data,
        }

        if self.error_type:
            result["error_type"] = self.error_type

        if self.message:
            result["message"] = self.message

        return result


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]

    def execute(self, **kwargs) -> ToolResult:
        return self.handler(**kwargs)
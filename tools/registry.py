from tool import Tool, ToolResult


class ToolRegistry:
    """
    A simple registry that stores tools by name, allows lookup,
    execution, and exposes tool definitions.
    """

    def __init__(self) -> None:
        # Internal mapping: tool name -> Tool instance
        self._tools = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """
        Look up a tool by name.

        Raises:
            KeyError: If no tool is registered under the given name.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")

        return self._tools[name]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """
        Execute a registered tool with the given keyword arguments.

        Any exception raised by the tool is caught and converted into a
        failed ToolResult, so callers get a consistent return shape.
        """
        tool = self.get(name)

        try:
            return tool.execute(**arguments)
        except Exception as e:
            return ToolResult(
                ok=False,
                data={},
                error_type=type(e).__name__,
                message=str(e),
            )

    def definitions(self) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]
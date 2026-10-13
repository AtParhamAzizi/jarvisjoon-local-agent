from tool import ToolResult
from workspace import Workspace

import subprocess

def execute_command(
    workspace: Workspace,
    command: list[str],
    timeout: int = 30,
) -> ToolResult:

    if not isinstance(command, list) or not command or not all(
        isinstance(argument, str) for argument in command
    ):
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_command",
            message="command must be a nonempty list of strings.",
        )

    result = subprocess.run(
        command,
        cwd=workspace.root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )

    return ToolResult(
        ok=result.returncode == 0,
        data={
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )

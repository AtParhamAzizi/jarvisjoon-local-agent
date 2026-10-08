from tool import ToolResult
from workspace import Workspace

import subprocess

def execute_command(
    workspace: Workspace,
    command: list[str],
    timeout: int = 30,
) -> ToolResult:

    result = subprocess.run(
        command,
        cwd=workspace.root,
        capture_output=True,
        text=True,
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
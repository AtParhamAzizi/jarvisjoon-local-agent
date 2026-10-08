from tool import ToolResult
from workspace import Workspace

def list_directory(
    workspace: Workspace,
    path: str = "."
) -> ToolResult:

    directory = workspace.resolve(path)

    if not directory.exists():
        return ToolResult(
            ok=False,
            data={},
            error_type="not_found",
            message=f"Directory does not exist: {path}",
        )

    if not directory.is_dir():
        return ToolResult(
            ok=False,
            data={},
            error_type="not_directory",
            message=f"Not a directory: {path}",
        )

    entries = []

    for item in sorted(directory.iterdir()):
        entries.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
        })

    return ToolResult(
        ok=True,
        data={
            "path": path,
            "entries": entries,
        }
    )
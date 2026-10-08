from tool import ToolResult
from workspace import Workspace

def read_file(
    workspace: Workspace,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> ToolResult:


    
    """
    Read a text file from the workspace.

    Args:
        workspace: The workspace used to resolve the path.
        path: Relative path to the file.
        start_line: 1-based line number to start from (default: 1).
        end_line: 1-based line number to end at (inclusive), or None for EOF.

    Returns:
        ToolResult: On success, data contains path, start_line, end_line, content.
                    On failure, error_type and message describe the problem.
    """



    file_path = workspace.resolve(path)

    if not file_path.exists():
        return ToolResult(
            ok=False,
            data={},
            error_type="not_found",
            message=f"File does not exist: {path}",
        )

    if not file_path.is_file():
        return ToolResult(
            ok=False,
            data={},
            error_type="not_file",
            message=f"Not a file: {path}",
        )


    # read as txt UTF-8 file
    text = file_path.read_text(encoding="utf-8", errors="replace")

    lines = text.splitlines()

    start = max(start_line - 1, 0)

    if end_line is None:
        selected = lines[start:]
    else:
        selected = lines[start:end_line]

    return ToolResult(
        ok=True,
        data={
            "path": path,
            "start_line": start + 1,
            "end_line": start + len(selected),
            "content": "\n".join(selected),
        },
    )
from tool import ToolResult
from workspace import Workspace

def write_file(
    workspace: Workspace,
    path: str,
    content: str,
) -> ToolResult:
    file_path = workspace.resolve(path)

    # Make sure the parent directory exists (create it if needed).
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the content as UTF-8, overwriting any existing file.
    file_path.write_text(content, encoding="utf-8")

    return ToolResult(
        ok=True,
        data={
            "path": path,
            # Count bytes, not characters, so multibyte UTF-8 is accurate.
            "bytes_written": len(content.encode("utf-8")),
        },
    )
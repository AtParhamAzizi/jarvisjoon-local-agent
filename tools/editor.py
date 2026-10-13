from tool import ToolResult
from workspace import Workspace

def edit_file(
    workspace: Workspace,
    path: str,
    old_text: str,
    new_text: str,
) -> ToolResult:
    file_path = workspace.resolve(path)

    if not file_path.exists():
        return ToolResult(
            ok=False,
            data={},
            error_type="not_found",
            message=f"File does not exist: {path}",
        )

    if not old_text:
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_old_text",
            message="The text to replace cannot be empty.",
        )

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        content = handle.read()

    count = content.count(old_text)

    if count == 0:
        return ToolResult(
            ok=False,
            data={},
            error_type="text_not_found",
            message="The requested text was not found.",
        )


    if count > 1:
        return ToolResult(
            ok=False,
            data={"matches": count},
            error_type="ambiguous_edit",
            message="The requested text occurs multiple times.",
        )

    new_content = content.replace(old_text, new_text, 1)

    with file_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(new_content)

    return ToolResult(
        ok=True,
        data={
            "path": path,
            "replacements": 1,
        },
    )

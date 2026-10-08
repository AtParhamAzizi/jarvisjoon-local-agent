"""File content search and filename search tools."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from tool import ToolResult
from workspace import Workspace


# Directories that should normally be ignored by a coding agent.
# These usually hold dependencies, build artifacts, caches, or VCS data
# that would only add noise to search results.
IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    "dist",
    "build",
}


def _should_ignore(path: Path) -> bool:
    """Return True if any path component belongs to an ignored directory."""
    return any(part in IGNORED_DIRECTORIES for part in path.parts)


def _is_binary_file(path: Path) -> bool:
    """
    Basic binary-file detection.

    Reads the first 8 KiB and looks for a NUL byte, which is a common
    marker for binary content. On read errors we conservatively treat
    the file as binary so it gets skipped.
    """
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)

        return b"\x00" in chunk

    except (OSError, PermissionError):
        # If we can't even read it, treat it as binary and move on.
        return True


def search_files(
    workspace: Workspace,
    query: str,
    path: str = ".",
    case_sensitive: bool = True,
    max_results: int = 100,
) -> ToolResult:
    """
    Search for text inside files.

    Args:
        workspace:
            Workspace instance.

        query:
            Text to search for.

        path:
            Directory or file to search.

        case_sensitive:
            Whether the search should be case-sensitive.

        max_results:
            Maximum number of matches to return.
    """
    # --- Validate inputs ---------------------------------------------------

    if not query:
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_query",
            message="Search query cannot be empty.",
        )

    if max_results <= 0:
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_max_results",
            message="max_results must be greater than zero.",
        )

    # --- Resolve the search target ----------------------------------------

    try:
        target = workspace.resolve(path)
    except PermissionError as exc:
        return ToolResult(
            ok=False,
            data={},
            error_type="permission_denied",
            message=str(exc),
        )

    if not target.exists():
        return ToolResult(
            ok=False,
            data={},
            error_type="not_found",
            message=f"Path does not exist: {path}",
        )

    # Build the list of candidate files to search.
    if target.is_file():
        files = [target]

    elif target.is_dir():
        files = [
            p
            for p in target.rglob("*")
            if p.is_file()
            and not _should_ignore(p.relative_to(workspace.root))
        ]

    else:
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_path",
            message=f"Invalid path: {path}",
        )

    # --- Prepare the query for matching -----------------------------------

    if case_sensitive:
        search_query = query
    else:
        search_query = query.lower()

    matches = []

    # --- Scan each file line by line --------------------------------------

    for file_path in files:

        if len(matches) >= max_results:
            break

        # Resolve every candidate too: a symlink inside the workspace could
        # otherwise make a search read a file outside the workspace.
        try:
            relative_path = file_path.resolve().relative_to(workspace.root)
        except (OSError, ValueError):
            continue

        # Skip files inside ignored directories.
        if _should_ignore(relative_path):
            continue

        # Skip binary files entirely.
        if _is_binary_file(file_path):
            continue

        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, PermissionError):
            # Unreadable file: skip it and keep going.
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            # Normalize the line if the search is case-insensitive.
            searchable_line = line if case_sensitive else line.lower()

            if search_query in searchable_line:
                matches.append(
                    {
                        "file": str(relative_path),
                        "line": line_number,
                        "text": line.strip(),
                    }
                )

                if len(matches) >= max_results:
                    break

    return ToolResult(
        ok=True,
        data={
            "query": query,
            "path": path,
            "case_sensitive": case_sensitive,
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) >= max_results,
        },
    )


def find_files(
    workspace: Workspace,
    pattern: str,
    path: str = ".",
    max_results: int = 500,
) -> ToolResult:
    """
    Find files by filename pattern.

    Examples:

        *.py
        *.json
        test_*.py
        interface.py
    """
    # --- Validate inputs ---------------------------------------------------

    if not pattern:
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_pattern",
            message="File pattern cannot be empty.",
        )

    if max_results <= 0:
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_max_results",
            message="max_results must be greater than zero.",
        )

    # --- Resolve the search target ----------------------------------------

    try:
        target = workspace.resolve(path)
    except PermissionError as exc:
        return ToolResult(
            ok=False,
            data={},
            error_type="permission_denied",
            message=str(exc),
        )

    if not target.exists():
        return ToolResult(
            ok=False,
            data={},
            error_type="not_found",
            message=f"Path does not exist: {path}",
        )

    # Build the list of candidate paths to check.
    if target.is_file():
        candidates = [target]

    elif target.is_dir():
        candidates = target.rglob("*")

    else:
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_path",
            message=f"Invalid path: {path}",
        )

    results = []

    # --- Match filenames against the pattern ------------------------------

    for file_path in candidates:

        if len(results) >= max_results:
            break

        if not file_path.is_file():
            continue

        try:
            relative_path = file_path.resolve().relative_to(workspace.root)
        except ValueError:
            # Path isn't under the workspace root; skip it.
            continue

        if _should_ignore(relative_path):
            continue

        if fnmatch.fnmatch(file_path.name, pattern):
            results.append(str(relative_path))

    # Return results in a stable, sorted order.
    results.sort()

    return ToolResult(
        ok=True,
        data={
            "pattern": pattern,
            "path": path,
            "files": results,
            "count": len(results),
            "truncated": len(results) >= max_results,
        },
    )

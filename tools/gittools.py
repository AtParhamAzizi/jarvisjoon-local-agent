"""Read-only Git tools for the local coding agent."""

from __future__ import annotations

import subprocess
from typing import Any

from tool import ToolResult
from workspace import Workspace


def _run_git(
    workspace: Workspace,
    args: list[str],
    timeout: int = 30,
) -> ToolResult:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )

    except FileNotFoundError:
        # The `git` executable isn't on PATH.
        return ToolResult(
            ok=False,
            data={},
            error_type="git_not_installed",
            message="Git executable was not found on the system.",
        )

    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            data={},
            error_type="timeout",
            message=f"Git command timed out after {timeout} seconds.",
        )

    except OSError as exc:
        # Any other OS-level failure (permissions, cwd missing, etc.).
        return ToolResult(
            ok=False,
            data={},
            error_type="execution_error",
            message=str(exc),
        )

    # Git ran: success is determined by its exit code.
    return ToolResult(
        ok=result.returncode == 0,
        data={
            "command": ["git", *args],
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        error_type=(None if result.returncode == 0 else "git_error"),
        message=(
            None
            if result.returncode == 0
            else result.stderr.strip()
        ),
    )


def git_status(
    workspace: Workspace,
) -> ToolResult:
    """
    Show the current Git working tree status.

    Equivalent to:

        git status --short
    """
    return _run_git(
        workspace,
        [
            "status",
            "--short",
        ],
    )


def git_diff(
    workspace: Workspace,
    staged: bool = False,
) -> ToolResult:
    """
    Show Git differences.

    staged=False:
        Show unstaged changes.

        git diff

    staged=True:
        Show staged changes.

        git diff --cached
    """
    if staged:
        args = [
            "diff",
            "--cached",
        ]
    else:
        args = [
            "diff",
        ]

    return _run_git(
        workspace,
        args,
    )


def git_log(
    workspace: Workspace,
    limit: int = 20,
) -> ToolResult:
    """
    Show recent Git commits.

    Equivalent to:

        git log --oneline -20

    The requested limit is clamped to a maximum of 100 to avoid
    accidentally requesting an enormous log.
    """
    if limit <= 0:
        return ToolResult(
            ok=False,
            data={},
            error_type="invalid_limit",
            message="limit must be greater than zero.",
        )

    # Protect against accidentally requesting an enormous log.
    limit = min(limit, 100)

    return _run_git(
        workspace,
        [
            "log",
            "--oneline",
            f"-{limit}",
        ],
    )
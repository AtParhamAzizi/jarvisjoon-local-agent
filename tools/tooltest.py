from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import traceback

from pathlib import Path
from typing import Any


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

# Add the project root to sys.path so the tools package is importable
# regardless of where this script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "tools"))


# ---------------------------------------------------------
# Import tools
# ---------------------------------------------------------


# Import a tool without stopping the entire test suite.
def import_tool(module_name: str, function_name: str):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        print(f"[IMPORT ERROR] {module_name}.{function_name}: {exc}")
        return None


# Resolve every tool used by the test suite. Each may be None if the
# corresponding module or function is missing.
Workspace = import_tool("workspace", "Workspace")

list_directory = import_tool("tools.lister", "list_directory")
read_file = import_tool("tools.reader", "read_file")
write_file = import_tool("tools.writer", "write_file")
edit_file = import_tool("tools.editor", "edit_file")

search_files = import_tool("tools.search", "search_files")
find_files = import_tool("tools.search", "find_files")

execute_command = import_tool("tools.commander", "execute_command")

git_status = import_tool("tools.gittools", "git_status")
git_diff = import_tool("tools.gittools", "git_diff")
git_log = import_tool("tools.gittools", "git_log")


# ---------------------------------------------------------
# Test reporting
# ---------------------------------------------------------

# Collected results for the final summary.
# Each entry: {"name": str, "passed": bool | None, "details": str}
RESULTS: list[dict[str, Any]] = []


def report(name: str, passed: bool, details: str = ""):
    """Record and print a single test result (pass or fail)."""
    status = "PASS" if passed else "FAIL"

    RESULTS.append({
        "name": name,
        "passed": passed,
        "details": details,
    })

    print(f"[{status}] {name}")

    if details:
        print(f"       {details}")


def skip(name: str, reason: str):
    """Record and print a skipped test (e.g. due to an import failure)."""
    RESULTS.append({
        "name": name,
        "passed": None,
        "details": reason,
    })

    print(f"[SKIP] {name}")
    print(f"       {reason}")


def get_data(result: Any) -> dict[str, Any]:
    """Normalize a ToolResult or dictionary to a dictionary."""
    if hasattr(result, "to_dict"):
        result = result.to_dict()

    if isinstance(result, dict):
        return result

    raise TypeError(
        f"Expected dict or ToolResult, got {type(result).__name__}"
    )


def assert_tool_success(result: Any) -> dict[str, Any]:
    """
    Normalize a tool result and raise AssertionError if it reports failure.

    Returns the normalized dict so callers can inspect its contents.
    """
    data = get_data(result)

    if data.get("ok") is not True:
        raise AssertionError(data.get("message", str(data)))

    return data


def call_tool(function, workspace, **kwargs):
    """
    Call a tool using only keyword arguments supported by its Python signature.

    The workspace is passed if the tool declares a `workspace` parameter
    (or, as a fallback, a `ws` parameter). Any extra keyword arguments
    that the function doesn't accept are dropped, unless the function
    declares **kwargs.
    """
    if function is None:
        raise RuntimeError("Tool could not be imported.")

    signature = inspect.signature(function)
    params = signature.parameters

    call_args = dict(kwargs)

    # Inject the workspace under whichever name the tool expects.
    if "workspace" in params:
        call_args["workspace"] = workspace
    elif "ws" in params:
        call_args["ws"] = workspace

    # Filter out unsupported kwargs unless the function accepts **kwargs.
    if not any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    ):
        call_args = {
            key: value
            for key, value in call_args.items()
            if key in params
        }

    return function(**call_args)


def run_test(name: str, function):
    """Run one test and record any raised exception as a failure."""
    try:
        function()
    except Exception as exc:
        report(
            name,
            False,
            f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def run_process(
    command: list[str],
    cwd: Path,
    timeout: int = 30,
):
    """Run a subprocess without invoking a shell."""
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        shell=False,
    )


def create_hello_world(workspace: Path):
    """Create the standard test program directly."""
    script = workspace / "hello.py"

    script.write_text(
        'print("Hello, World!")\n',
        encoding="utf-8",
    )

    return script


def initialize_git(workspace: Path):
    """
    Initialize a temporary Git repository and make an initial commit.

    Raises RuntimeError if any of the setup commands fail.
    """
    commands = [
        ["git", "init"],
        ["git", "config", "user.name", "Agent Tool Tester"],
        ["git", "config", "user.email", "agent-test@example.invalid"],
        ["git", "add", "hello.py"],
        [
            "git",
            "commit",
            "-m",
            "Initial test commit",
        ],
    ]

    for command in commands:
        result = run_process(command, workspace)

        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {command}\n"
                f"{result.stderr}"
            )


# ---------------------------------------------------------
# Individual tests
# ---------------------------------------------------------

def test_list_directory(workspace):
    """list_directory should list the seeded hello.py file."""
    result = call_tool(
        list_directory,
        workspace,
        path=".",
    )

    data = assert_tool_success(result)

    # Accommodate common result field names.
    entries = (
        data.get("entries")
        or data.get("files")
        or data.get("items")
        or []
    )

    names = [
        entry if isinstance(entry, str)
        else entry.get("name", "")
        for entry in entries
    ]

    assert "hello.py" in names, (
        f"hello.py not found in directory listing: {names}"
    )

    report(
        "list_directory",
        True,
        "Found hello.py in the workspace.",
    )


def test_write_file(workspace):
    """write_file should create a new file with the exact given content."""
    result = call_tool(
        write_file,
        workspace,
        path="generated.py",
        content='print("Hello, World!")\n',
    )

    assert_tool_success(result)

    generated = workspace.root / "generated.py"

    assert generated.exists(), "generated.py was not created."

    assert generated.read_text(encoding="utf-8") == (
        'print("Hello, World!")\n'
    )
    assert generated.read_bytes() == b'print("Hello, World!")\n'

    report(
        "write_file",
        True,
        "Created generated.py with Hello, World! code.",
    )


def test_read_file(workspace):
    """read_file should return the contents of hello.py."""
    result = call_tool(
        read_file,
        workspace,
        path="hello.py",
        start_line=1,
        end_line=10,
    )

    data = assert_tool_success(result)

    content = data.get("content")

    if content is None:
        content = data.get("text", "")

    assert "Hello, World!" in content
    assert content == (workspace.root / "hello.py").read_bytes().decode("utf-8")
    assert not get_data(call_tool(
        read_file, workspace, path="hello.py", start_line=1, end_line=-1,
    ))["ok"]

    report(
        "read_file",
        True,
        "Read the expected Python source.",
    )


def test_edit_file(workspace):
    """edit_file should replace the targeted text in hello.py."""
    original = (workspace.root / "hello.py").read_bytes()
    assert not get_data(call_tool(
        edit_file, workspace, path="hello.py", old_text="", new_text="unexpected",
    ))["ok"]
    result = call_tool(
        edit_file,
        workspace,
        path="hello.py",
        old_text="Hello, World!",
        new_text="Hello from the local agent!",
    )

    assert_tool_success(result)

    content = (workspace.root / "hello.py").read_text(
        encoding="utf-8"
    )

    assert "Hello from the local agent!" in content
    assert "Hello, World!" not in content
    assert (workspace.root / "hello.py").read_bytes() == original.replace(
        b"Hello, World!", b"Hello from the local agent!",
    )

    report(
        "edit_file",
        True,
        "Replaced the requested text.",
    )


def test_search_files(workspace):
    """search_files should find the edited text inside hello.py."""
    result = call_tool(
        search_files,
        workspace,
        query="Hello from the local agent!",
        path=".",
        case_sensitive=True,
        max_results=100,
    )

    data = assert_tool_success(result)

    matches = data.get("matches", [])

    if not matches:
        raise AssertionError(
            "No matches returned for text present in hello.py."
        )
    limited = assert_tool_success(call_tool(
        search_files, workspace, query="Hello from the local agent!", max_results=1,
    ))
    assert limited["count"] == 1 and not limited["truncated"]
    limited = assert_tool_success(call_tool(
        search_files, workspace, query="print", max_results=1,
    ))
    assert limited["count"] == 1 and limited["truncated"]

    report(
        "search_files",
        True,
        f"Found {len(matches)} matching result(s).",
    )


def test_find_files(workspace):
    """find_files should locate hello.py via the *.py pattern."""
    result = call_tool(
        find_files,
        workspace,
        pattern="*.py",
        path=".",
        max_results=100,
    )

    data = assert_tool_success(result)

    files = data.get("files", [])

    if not files:
        raise AssertionError("No Python files were returned.")

    normalized = [
        str(file).replace("\\", "/")
        for file in files
    ]

    assert any(
        file.endswith("hello.py")
        for file in normalized
    ), f"hello.py not found: {normalized}"
    limited = assert_tool_success(call_tool(
        find_files, workspace, pattern="hello.py", max_results=1,
    ))
    assert limited["count"] == 1 and not limited["truncated"]
    limited = assert_tool_success(call_tool(
        find_files, workspace, pattern="*.py", max_results=1,
    ))
    assert limited["count"] == 1 and limited["truncated"]

    report(
        "find_files",
        True,
        f"Found {len(files)} Python file(s).",
    )


def test_execute_command(workspace):
    """execute_command should run hello.py and produce the expected output."""
    assert not get_data(call_tool(
        execute_command, workspace, command="python hello.py",
    ))["ok"]
    result = call_tool(
        execute_command,
        workspace,
        command=[
            sys.executable,
            "hello.py",
        ],
        timeout=30,
    )

    data = assert_tool_success(result)

    stdout = data.get("stdout", "")

    assert "Hello from the local agent!" in stdout, (
        f"Unexpected program output: {stdout!r}"
    )

    report(
        "execute_command",
        True,
        "Executed hello.py and verified its output.",
    )


def test_git_status(workspace):
    """git_status should report the modified hello.py as a change."""
    result = call_tool(
        git_status,
        workspace,
    )

    data = assert_tool_success(result)

    stdout = data.get("stdout", "")

    # hello.py was committed before the edit, so its current
    # changes should appear in Git status.
    if not stdout.strip():
        raise AssertionError(
            "Expected a modified hello.py in Git status."
        )

    report(
        "git_status",
        True,
        "Git reports working-tree changes.",
    )


def test_git_diff(workspace):
    """git_diff should show the edited text as an unstaged change."""
    result = call_tool(
        git_diff,
        workspace,
        staged=False,
    )

    data = assert_tool_success(result)

    stdout = data.get("stdout", "")

    assert "Hello from the local agent!" in stdout, (
        "Expected edited content in Git diff."
    )

    report(
        "git_diff",
        True,
        "Git diff contains the source modification.",
    )


def test_git_log(workspace):
    """git_log should include the initial commit message."""
    result = call_tool(
        git_log,
        workspace,
        limit=10,
    )

    data = assert_tool_success(result)

    stdout = data.get("stdout", "")

    assert "Initial test commit" in stdout, (
        "Expected initial commit in Git log."
    )

    report(
        "git_log",
        True,
        "Retrieved the test repository's commit history.",
    )


# ---------------------------------------------------------
# Test runner
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("LOCAL CODING AGENT - TOOL TEST SUITE")
    print("=" * 60)

    if Workspace is None:
        print(
            "\nCannot run tests: "
            "tools.workspace.Workspace is unavailable."
        )
        sys.exit(1)

    try:
        with tempfile.TemporaryDirectory(
            prefix="local_agent_test_"
        ) as temp_dir:

            root = Path(temp_dir).resolve()

            # Build the workspace object the tools expect.
            workspace = Workspace(str(root))

            # Seed the workspace with a simple Python program.
            create_hello_world(root)

            # Initialize Git before editing the file, so we can
            # later observe the edit as a working-tree change.
            initialize_git(root)

            # (display name, test function) pairs in run order.
            tests = [
                ("list_directory", test_list_directory),
                ("read_file", test_read_file),
                ("write_file", test_write_file),
                ("edit_file", test_edit_file),
                ("search_files", test_search_files),
                ("find_files", test_find_files),
                ("execute_command", test_execute_command),
                ("git_status", test_git_status),
                ("git_diff", test_git_diff),
                ("git_log", test_git_log),
            ]

            # Map display names to the imported tool callables so we
            # can skip tests whose tool couldn't be imported.
            tool_map = {
                "list_directory": list_directory,
                "read_file": read_file,
                "write_file": write_file,
                "edit_file": edit_file,
                "search_files": search_files,
                "find_files": find_files,
                "execute_command": execute_command,
                "git_status": git_status,
                "git_diff": git_diff,
                "git_log": git_log,
            }

            for name, test_function in tests:
                function = tool_map[name]

                if function is None:
                    skip(name, "Tool import failed.")
                    continue

                # Bind the workspace as a default argument to avoid
                # late-binding issues inside the loop.
                run_test(
                    name,
                    lambda fn=test_function: fn(workspace),
                )

    except Exception as exc:
        print(f"\n[FATAL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    passed = sum(
        result["passed"] is True
        for result in RESULTS
    )

    failed = sum(
        result["passed"] is False
        for result in RESULTS
    )

    skipped = sum(
        result["passed"] is None
        for result in RESULTS
    )

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed : {passed}")
    print(f"Failed : {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total  : {len(RESULTS)}")

    print("=" * 60)

    if failed:
        sys.exit(1)

    if skipped:
        print(
            "Some tools were skipped. Check module imports "
            "and rerun the suite."
        )
        sys.exit(2)

    print("All tools passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Create the tools exposed to the local agent."""

from commander import execute_command
from editor import edit_file
from gittools import git_diff, git_log, git_status
from lister import list_directory
from reader import read_file
from registry import ToolRegistry
from search import find_files, search_files
from tool import Tool
from workspace import Workspace
from writer import write_file


def build_tools(workspace: Workspace) -> ToolRegistry:
    """Build a registry of tools, each restricted to ``workspace``."""
    registry = ToolRegistry()

    def add(name, description, properties, required, handler):
        registry.register(Tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            handler=handler,
        ))

    path = {"type": "string", "description": "Path relative to the workspace."}
    add("list_directory", "List files and directories.", {"path": path}, [],
        lambda path=".": list_directory(workspace, path))
    add("read_file", "Read a UTF-8 text file, optionally selecting a line range.", {
        "path": path, "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
    }, ["path"], lambda **args: read_file(workspace, **args))
    add("write_file", "Write UTF-8 text, creating parent folders and replacing any existing file.", {
        "path": path, "content": {"type": "string"},
    }, ["path", "content"], lambda **args: write_file(workspace, **args))
    add("edit_file", "Replace one unique occurrence of text in a UTF-8 file.", {
        "path": path, "old_text": {"type": "string"},
        "new_text": {"type": "string"},
    }, ["path", "old_text", "new_text"], lambda **args: edit_file(workspace, **args))
    add("search_files", "Search text in workspace files; binary files and common generated folders are skipped.", {
        "query": {"type": "string"}, "path": path,
        "case_sensitive": {"type": "boolean"}, "max_results": {"type": "integer"},
    }, ["query"], lambda **args: search_files(workspace, **args))
    add("find_files", "Find files by filename glob, such as *.py or test_*.py.", {
        "pattern": {"type": "string"}, "path": path, "max_results": {"type": "integer"},
    }, ["pattern"], lambda **args: find_files(workspace, **args))
    add("execute_command", "Run a program with argument-list syntax from the workspace root.", {
        "command": {"type": "array", "items": {"type": "string"}},
        "timeout": {"type": "integer"},
    }, ["command"], lambda **args: execute_command(workspace, **args))
    add("git_status", "Show the short Git working-tree status.", {}, [],
        lambda: git_status(workspace))
    add("git_diff", "Show unstaged changes, or staged changes when staged is true.", {
        "staged": {"type": "boolean"},
    }, [], lambda staged=False: git_diff(workspace, staged))
    add("git_log", "Show recent Git commits (up to 100).", {
        "limit": {"type": "integer"},
    }, [], lambda limit=20: git_log(workspace, limit))

    return registry

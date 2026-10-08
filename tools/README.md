# Local agent tools

These Python tools give a local agent a small set of ways to inspect and change a project. File tools use a `Workspace` to keep paths inside the configured workspace folder. `build_tools()` collects the tools in a registry, and the registry can expose their descriptions and run them by name.

## Available tools

| Tool | What it does |
| --- | --- |
| `list_directory` | Lists the names and types of items in a directory. Defaults to the workspace root. |
| `read_file` | Reads a UTF-8 text file. `start_line` and `end_line` select an optional, one-based inclusive line range. |
| `write_file` | Writes UTF-8 text, creating parent directories and replacing an existing file if present. |
| `edit_file` | Replaces a piece of text only when it appears exactly once in the file. It reports when the text is missing or ambiguous. |
| `search_files` | Finds a text string in files, with optional case sensitivity and a result limit. It skips binary files and common dependency, version-control, and build folders. |
| `find_files` | Finds files by filename glob, for example `*.py` or `test_*.py`. |
| `execute_command` | Runs a program from the workspace root. Pass an argument list, such as `["python", "--version"]`; it does not invoke a shell. |
| `git_status` | Returns Git's short working-tree status. |
| `git_diff` | Shows unstaged changes by default; set `staged` to `true` for staged changes. |
| `git_log` | Shows recent one-line commits. The default is 20 and the maximum is 100. |

Paths are relative to the workspace root unless absolute paths are passed; resolved paths outside the root are rejected by `Workspace`. Search and Git helpers return structured results through `ToolResult`. Git tools require Git to be installed and the workspace to be in a repository. Command execution can run any available executable, so the caller should only expose it to an agent trusted to run commands in that workspace.

## Using the registry

The modules currently import one another by their short module names. Add the `tools` directory to Python's module search path before importing `build_tools`:

```python
import sys
from pathlib import Path

tools_dir = Path("tools").resolve()
sys.path.insert(0, str(tools_dir))

from build_tools import build_tools
from workspace import Workspace

registry = build_tools(Workspace("."))
print(registry.definitions())
result = registry.execute("list_directory", {"path": "."})
print(result.to_dict())
```

Each tool definition includes a JSON-style parameter schema for an agent or caller. `registry.execute(name, arguments)` returns a `ToolResult`; call `to_dict()` to get a plain dictionary. Handler exceptions are converted to failed results by the registry.

## Result shape

Successful results have `ok: true` and include operation-specific fields. For example, a file read returns its path, selected line numbers, and content. Failed results have `ok: false` and may include `error_type` and a human-readable `message`. The exact fields depend on the tool.

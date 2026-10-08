# JarvisJoon Local Agent

This project is being built one piece at a time. The first piece is the set of tools the agent can use to work with a project folder: look around, read and change files, search for things, run a command, and check Git changes.

## Step 1: Give the agent useful tools

An agent needs a way to interact with the files and code it is helping with. The tools in [`tools/`](tools/) provide that connection. Each tool does one job, such as reading a file or finding a matching filename. The registry gathers them in one place so the rest of the agent can ask for a tool by name and pass it arguments.

The tools currently available are:

- `list_directory` to see what is in a folder.
- `read_file` to read a text file, including a selected range of lines.
- `write_file` to create or replace a file.
- `edit_file` to replace text when the requested text appears exactly once.
- `search_files` to look for text across files.
- `find_files` to find files by name, such as `*.py`.
- `execute_command` to run a program from the project folder.
- `git_status`, `git_diff`, and `git_log` to inspect the repository's Git history and changes.

File tools are given a `Workspace`, which keeps resolved file paths inside the chosen project folder. The command tool runs a program with an argument list rather than through a shell. Since it can still launch programs available on the computer, it should only be made available to an agent that is trusted to run commands in that workspace.

## Trying the tools

The tool modules import one another by their short names, so add the `tools` folder to Python's import path before importing `build_tools`:

```python
import sys
from pathlib import Path

tools_dir = Path("tools").resolve()
sys.path.insert(0, str(tools_dir))

from build_tools import build_tools
from workspace import Workspace

tools = build_tools(Workspace("."))
result = tools.execute("list_directory", {"path": "."})
print(result.to_dict())
```

`tools.definitions()` returns the names, descriptions, and parameter schemas that an agent can use to decide which tool to call. `tools.execute()` runs a tool and returns a `ToolResult`; `to_dict()` turns that result into a regular dictionary.

For more detail on each tool and its arguments, see [`tools/README.md`](tools/README.md).

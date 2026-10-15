# JarvisJoon Local Agent

JarvisJoon is a local coding assistant using `models/gemma-4-E2B-it/`. It inspects project files, searches code, checks Git status, and optionally applies changes after you approve each preview. The model proposes actions; application code validates and authorizes them. It starts read-only, uses no cloud API, and cannot execute arbitrary commands.

## Run your agent

Open PowerShell in the repository root:

```powershell
# First-time setup: skip creation if .venv exists
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r CLI/requirements.txt

# Check setup, then start chat
.\.venv\Scripts\python.exe -m CLI doctor
.\.venv\Scripts\python.exe -m Terminal
```

Activation is unnecessary when using the virtual environment's executable. `python -m CLI chat` launches the same interface. The model loads on your first question and stays in memory. Try `Read README.md and summarize the project.` or `Find where write_file is defined.` Type `/exit` to leave.

Enable optional file changes with:

```powershell
.\.venv\Scripts\python.exe -m Terminal --allow-edits
```

Every edit shows a diff and requires typing `yes`; other responses decline it. There is no approve-everything mode, and piped input cannot approve edits. Resetting/interruption does not undo applied changes. This interface is chat, not a PowerShell replacement.

## Build a real project

Start the interactive project-building mode, which enables file writes with individual approvals:

```powershell
.\jarvisjoon.cmd build
```

Or enter `/build` during an existing chat. `/readonly` disables writes again. Both commands reset conversation context, but leave files and session statistics unchanged.

Example request:

```text
Create a runnable Python command-line todo app in projects/todo-app.
Include complete source code, tests, and a README with setup and run commands.
Inspect the workspace first, then create the files and read back the entry point.
```

The agent is instructed to use actual file tools, create nested parent folders, and work file by file rather than only display code. Type `yes` for each file preview you approve. You can also target another existing folder with `build --workspace "C:\path\to\project"`.

For larger files/tasks, explicitly raise budgets if your hardware permits:

```powershell
.\jarvisjoon.cmd build --context-tokens 8192 --max-new-tokens 1536 --max-tool-calls 24
```

Build mode keeps the normal budgets unless you override them. Large projects may need several staged requests and `/reset` when context fills. It does not enable shell execution, dependency installation, or running tests. Follow the generated README's commands yourself; saved files are not proof the real model produced a correct project.

## Requirements and checkpoint

- Python with `venv` and `pip`; the current local environment uses Python 3.13.14. No supported-version matrix is defined.
- `CLI/requirements.txt`: `torch>=2.6,<3`, `transformers==5.11.0`, `accelerate>=1.6,<2`.
- A complete local checkpoint. Model files are excluded from Git; cloning alone does not supply them. Verify provenance and licensing.
- Sufficient RAM/VRAM for weights and inference. Current Safetensors weights total approximately 10.25 GB on disk; this is not a memory requirement or guarantee of fit.
- Git installed and a repository in the workspace for Git inspection; an interactive terminal for chat/approvals.

Package installation needs package access. Inference is local-only and never downloads weights automatically. `doctor` checks `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` or `chat_template.jinja.txt`, and Safetensors weights. If a `model.safetensors.index.json` exists, every referenced shard must be present. Retain the complete original checkpoint, not just these diagnostic files.

A successful diagnostic does not prove weights load, dependencies are compatible, or inference fits in memory. Loading uses `Gemma4ForCausalLM` with the checkpoint's text configuration and weight-key mapping, `local_files_only=True`, and `trust_remote_code=False`; unused vision/audio components are not instantiated. Missing or mismatched text weights are rejected. Hugging Face online access and telemetry are disabled during loading. This interface is text-only.

## GPU setup

The detected NVIDIA card on this machine is a GTX 1050 with 4 GiB VRAM. The original environment had CPU-only `torch 2.14.1+cpu`, so auto mode could not use CUDA. The checkpoint's approximately 10.25 GB of weights cannot fit entirely in that GPU's memory.

For this Windows/Python 3.13/Pascal configuration, install the pinned official CUDA 11.8 build:

```powershell
.\.venv\Scripts\python.exe -m pip install -r CLI/requirements-gpu.txt
.\.venv\Scripts\python.exe -m CLI doctor
.\jarvisjoon.cmd build --device auto
```

GPU requirements pin `torch==2.7.1+cu118`; the standard requirements remain valid for CPU setups. Stop running agent sessions before replacing PyTorch. This older CUDA build is intentional for the detected GPU/driver; it is not a recommendation for all modern GPUs.

With working CUDA, `auto` reserves 1 GiB per GPU for generation overhead. If the per-layer embedding exceeds the GPU weight budget (approximately 4.38 GiB in BF16 for this checkpoint), it stays resident in CPU RAM. Fitting decoder layers and a separate copy of the output head are placed on GPU with Accelerate alignment hooks; the tied input embedding remains on CPU. This avoids streaming an embedding larger than VRAM. Older GPUs use resident CPU BF16 weights and GPU FP32 computation with eager attention and explicit dtype/device alignment, because this checkpoint produced all-zero logits when tested in FP16 on the GTX 1050. GPUs with native BF16 support can use BF16; other configurations use Accelerate's automatic placement. `--device cuda` requests a full GPU load without CPU offload; do not use it for this checkpoint on a 4 GiB card. `--device cpu` explicitly disables GPU loading.

`doctor` reports PyTorch version, CUDA build, availability, and GPU memory without loading weights. After loading, `/status` and request progress report actual device placement and precision, not just the requested policy. GPU/CPU offloading still uses the CPU and is not a guarantee of faster inference or sufficient memory. If CUDA operations fail, inspect the error and driver compatibility before changing settings.

## Commands

Run from the repository root. Examples using `python` assume the virtual environment is active; otherwise use `.\.venv\Scripts\python.exe`.

| Command | Purpose |
| --- | --- |
| `python -m Terminal` or `python -m CLI chat` | Interactive conversation. |
| `python -m CLI build` or `.\jarvisjoon.cmd build` | Project-building chat with approved file writes. |
| `python -m CLI ask "your question"` | One request, then exit. |
| `python -m CLI ask "your question" --json` | Structured answer and tool events. |
| `python -m CLI doctor` | Check files/dependencies without loading weights. |
| `python -m CLI doctor --json` | JSON diagnostics. |
| `python -m CLI tools` | List enabled tools without loading weights. |
| `python -m CLI --help` | Command help. |
| `python -m CLI chat --help` | Chat options. |

```powershell
.\.venv\Scripts\python.exe -m CLI ask "Read README.md and summarize the project."
.\.venv\Scripts\python.exe -m Terminal --workspace "C:\path\to\project"
.\.venv\Scripts\python.exe -m Terminal --device cpu
.\.venv\Scripts\python.exe -m Terminal --model "C:\path\to\checkpoint"
```

### Options

For CLI, options go after the subcommand; for Terminal, after the module name.

| Option | Default | Meaning |
| --- | --- | --- |
| `--workspace PATH` | Current directory | Existing project folder. |
| `--model PATH` | Repository's `models/gemma-4-E2B-it/` | Local checkpoint, independent of workspace. |
| `--device DEVICE` | `auto` | `auto`, `cpu`, or `cuda`; auto uses CPU when CUDA is unavailable. |
| `--context-tokens N` | `4096` | Rendered conversation budget including reserved output. |
| `--max-new-tokens N` | `768` | Output limit per response. |
| `--max-tool-calls N` | `12` | Tool budget per user request. |
| `--generation-timeout N` | `120` | Generation deadline in seconds, checked between tokens; not a loading timeout. |
| `--allow-edits` | Disabled | Per-change approved file writes. |
| `--plain` | Disabled | Disable UI colors; also respects `NO_COLOR`. |

Numbers must be positive. Context must exceed output budget and leave space for prompts, tools, history, and results. Larger budgets can increase memory usage.

One-shot JSON contains `answer`, `completed`, and `events`. Progress goes to stderr; answers/JSON go to stdout. `ask` returns `0` for a completed response, `1` for an incomplete request; startup errors return `2`, CLI-level interrupts `130`. Inspect individual events too.

## Chat controls

| Command | Action |
| --- | --- |
| `/help` | Help and examples. |
| `/tools` | Enabled tools. |
| `/status` | Workspace, model, loading state, edit mode. |
| `/build` | Enable approved writes and start fresh context. |
| `/readonly` | Disable writes and start fresh context; applied edits remain. |
| `/clear` | Clear supported terminal displays without clearing conversation history. |
| `/reset` | Clear history, not edits. |
| `/paste` | Multiline input; finish with `/end` on its own line. |
| `/exit` or `/quit` | Exit. |

Ctrl+C interrupts and clears chat context. EOF exits; on Windows use Ctrl+Z then Enter. Sessions are in memory only, not automatically saved.

The terminal includes a workspace dashboard, styled reply/review panels, colored code fences and diff lines, tool outcome labels, and request timing. `/status` includes budgets and session statistics. Colors fall back safely for redirected output, unsupported consoles, `NO_COLOR`, or `--plain`. No additional UI dependencies are required.

## Project structure and architecture

```text
CLI/
  __main__.py       Entry point
  main.py           Arguments, diagnostics, dispatch
  runtime.py        Loading, generation, response parsing
  agent.py          History and bounded tool-use loop
  gateway.py        Validation, restricted paths, approvals
  requirements.txt  Model dependencies
  tests/            Agent/gateway/UI/CLI and fake-runtime tests
Terminal/
  __main__.py       Chat-first entry point
  interface.py      Plain-text UI and approval prompts
  README.md         Interface notes
tools/
  build_tools.py    Registry construction
  registry.py       Definitions and dispatch
  tool.py           Tool/result types
  workspace.py      Workspace path resolution
  lister.py         Directory listing
  reader.py         Reading
  writer.py         Creation/replacement
  editor.py         Exact-text edits
  search.py         Text/filename search
  commander.py      Low-level program execution
  gittools.py       Git status/diff/log
  tooltest.py       Standalone tool checks
  README.md         Low-level reference
models/gemma-4-E2B-it/  Local checkpoint, excluded from Git
local_agent_plan.md    Design direction and future work
README.md              Setup and overview
```

The CLI creates the gateway, runtime, and agent. Chat or `ask` supplies a prompt. The runtime lazily loads weights, renders native chat/tool definitions, generates, and parses text or calls. The agent sends calls through the gateway, which validates paths, arguments, limits, and approvals. Results enter conversation history for the next response. The loop ends with an answer, limit, declined edit, repeated failure, or inference error; the UI reports outcomes.

## Tools and permissions

| Tool | Purpose | Agent access |
| --- | --- | --- |
| `list_directory` | List entries. | Both modes. |
| `read_file` | UTF-8 reading, optional one-based inclusive line range. | Both modes. |
| `search_files` | Text search, case/result options. | Both modes. |
| `find_files` | Filename glob search. | Both modes. |
| `git_status` | Short working-tree status. | Both modes. |
| `write_file` | Create/replace contents. | Edit mode with approval. |
| `edit_file` | Replace text occurring exactly once. | Edit mode with approval. |
| `execute_command` | Run a program by argument list, without shell. | Not enabled. |
| `git_diff` | Unstaged/staged changes. | Not enabled. |
| `git_log` | Recent commits. | Not enabled. |

### Direct registry use

Tool modules use short-name imports:

```python
import sys
from pathlib import Path

tools_directory = Path("tools").resolve()
sys.path.insert(0, str(tools_directory))

from build_tools import build_tools
from workspace import Workspace

registry = build_tools(Workspace("."))
print(registry.definitions())
result = registry.execute("list_directory", {"path": "."})
print(result.to_dict())
```

Definitions expose names, descriptions, and schemas. Execution returns `ToolResult`; `to_dict()` exposes `ok` and operation data/error details. See `tools/README.md` for arguments. Direct registry use bypasses gateway approvals/restrictions: writes can overwrite files, and programs are not confined to the workspace by an OS sandbox.

## Safety and limits

- Read-only by default. Edit mode does not enable commands, network access, Git diff/log, or test execution.
- Gateway validation checks names, required arguments, supported types, and unexpected properties.
- Access stays within the workspace. Parent traversal, links/junctions, protected paths, and selected unsafe filenames are rejected.
- Protected paths include Git metadata, environment files, SSH/cloud credentials, `.agents`, `.gapcode`, models, virtual environments, dependencies, and selected credential/key filenames. This is not comprehensive secret detection; avoid sensitive workspaces.
- Files/string arguments are limited to 1 MiB. Searches scan at most 10,000 candidate files, accept result limits of 1-200, and return bounded output.
- Previews above 30,000 characters are rejected. Approved edits check original contents have not changed and replace files atomically.
- A declined edit or the same tool failure repeated twice stops the request. Tool calls are bounded.
- Oversized conversations are rejected, not silently truncated. Responses hitting generation time/output limits are not executed as partial actions.
- Display text is sanitized against terminal control sequences. These safeguards are not a complete OS security boundary.
- Review answers and diffs: the model can make mistakes and cannot run tests to verify edits in this version.

## Tests

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s CLI/tests -v
.\.venv\Scripts\python.exe -B tools/tooltest.py
```

Unit tests use fake runtimes/model doubles, not real weights. The standalone harness uses a temporary workspace and invokes programs/Git. Passing tests does not prove end-to-end inference works. For a real-model smoke check, run `doctor`, then a small read-only `ask` request; it loads weights and can take significant time/memory.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Module not found | Run from repository root with virtual-environment Python. |
| Libraries missing | Install requirements with the Python used to launch. |
| Checkpoint incomplete | Restore complete files or choose a local folder with `--model`. |
| Missing Gemma class/parser | Check the pinned Transformers version. |
| CUDA unavailable | Use CPU or configure compatible CUDA-enabled PyTorch. |
| Load failure/out of memory | Check error and RAM/VRAM. Smaller budgets reduce inference overhead, not weights; CPU still needs enough RAM. |
| Slow first reply | First request loads weights; later ones reuse them. CPU may remain slow. |
| Context exceeded | Reset, narrow the task, or increase context if memory permits. |
| Output/deadline limit | Narrow task or increase corresponding limit with sufficient context/memory. |
| Piped chat rejected | Use `CLI ask`; chat needs an interactive terminal. |
| Edit denied | Enable edits, approve interactively, choose a small unprotected file change. |
| Output shortened | Narrow search or read a smaller line range. |
| Git failure | Check Git installation and repository/workspace selection. |

## Current scope and future work

Implemented: local text inference, interactive/one-shot requests, bounded inspection, per-change approvals, diagnostics.

Not implemented: agent-driven commands/tests, cloud APIs, browsing, persistent sessions, voice, desktop control, multimodal input. `local_agent_plan.md` is design/future work, not a feature guarantee.

See `Terminal/README.md` and `tools/README.md` for details. Setup is documented here; there is currently no `CLI/README.md`.

# JarvisJoon terminal workspace

Start from the repository root with `./jarvisjoon.cmd` in PowerShell, or `python -m Terminal` using your virtual environment. Install dependencies using the root `README.md` first. `python -m CLI chat` opens the same interface.

## Workspace UI

- A startup dashboard shows workspace, local model, and read-only or approval-required access.
- Cyan panels and violet headings organize the conversation. Replies preserve text, indentation, and code fences; Markdown headings and fenced code receive visual emphasis, not full Markdown/syntax rendering.
- Tool outcomes use explicit OK/FAIL labels, not color alone. Each reply shows measured request duration and tool-call count.
- `/status` shows model-loading state, device policy, context/output/tool budgets, request count, and session tool calls. It does not load the checkpoint.
- After loading, `/status` also shows actual weight placement and precision. `--device auto` uses GPU/CPU offloading when CUDA is available; `cuda` requires a full GPU load. See the root README's GPU setup for the detected 4 GiB GTX 1050.
- File approvals appear in a separate review panel, with colored additions/deletions and the entire preview. Type `yes` to approve that exact change.
- Panels wrap to terminal width. ASCII borders avoid requiring special fonts. No UI packages are needed.

Colors activate only on supported interactive outputs, including Windows consoles with virtual-terminal support. Use `./jarvisjoon.cmd --plain` or set `NO_COLOR` to disable colors. Redirected output and dumb terminals automatically stay uncolored. One-shot JSON remains machine-readable; progress and approvals go to stderr.

## Controls

| Command | Action |
| --- | --- |
| `/help` | Guide and example prompts. |
| `/tools` | Enabled tool descriptions. |
| `/status` | Configuration and session statistics. |
| `/build` | Enable approved file creation/editing and reset conversation context. |
| `/readonly` | Disable writes and reset context; existing edits remain. |
| `/clear` | Clear supported terminal displays and redraw dashboard; preserve history. In plain mode, redraw without terminal control sequences. |
| `/reset` | Clear conversation context; preserve applied edits and session statistics. |
| `/paste` | Multiline request; finish with `/end` on its own line. |
| `/exit` or `/quit` | Exit without saving the session. |

Try `Read README.md and summarize the project.` or `Find where write_file is defined.` The checkpoint loads only on the first question and remains in memory for later requests.

This is chat, not a shell. Arbitrary commands are never executed. Enable proposed file changes with `./jarvisjoon.cmd --allow-edits`; each requires an interactive preview and explicit approval. Piped input cannot approve edits.

For project creation, start `./jarvisjoon.cmd build` or enter `/build`. Describe the app, features, and target subfolder. The agent is instructed to inspect existing files, create complete sources/configuration/tests/run instructions, and read back important files. Parent folders are created by approved writes. Each file needs approval; no commands or tests run automatically. Larger projects should be built in stages, with explicit budget increases only if memory allows.

Ctrl+C interrupts and resets conversation context. Applied edits remain. EOF exits (Ctrl+Z then Enter on Windows). Sessions are held in memory only. Model/file text is sanitized before styling to prevent untrusted terminal control sequences.

If the model cannot load, the agent reports the setup error. If context fills, use `/reset`, narrow the request, or increase `--context-tokens` if memory permits. Responses are displayed after generation, not streamed token by token; progress messages indicate activity.

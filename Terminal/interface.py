"""A styled, dependency-free chat interface with conservative approvals."""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import time
from typing import TextIO


HELP = """Try asking:
  Where is file writing implemented?
  Read README.md and summarize the project.
  Find Python files that mention Workspace.

Commands:
  /help     Show this guide
  /tools    Show enabled tools
  /status   Show model, workspace, budgets, and session stats
  /build    Enable approved file creation and editing; start fresh context
  /readonly Disable edits; start fresh context
  /clear    Clear display without resetting conversation
  /reset    Clear conversation without undoing edits
  /paste    Multiline input; finish with /end
  /exit     Leave the session

Ctrl+C interrupts work and clears conversation context.
EOF exits (Ctrl+Z then Enter on Windows).
This is chat, not a shell. Sessions are not saved automatically.
"""

PALETTE = {
    "accent": "\033[96m",
    "muted": "\033[90m",
    "success": "\033[92m",
    "warning": "\033[93m",
    "error": "\033[91m",
    "heading": "\033[1;95m",
}


def safe_text(value: str) -> str:
    value = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", value)
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    return "".join(character for character in value
                   if character in "\n\t" or (ord(character) >= 32 and not 127 <= ord(character) <= 159))


def supports_color(stream: TextIO) -> bool:
    if not stream.isatty() or "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        mode = wintypes.DWORD()
        return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode))
                    and kernel.SetConsoleMode(handle, mode.value | 0x0004))
    except (OSError, ValueError, AttributeError):
        return False


class TerminalUI:
    def __init__(self, output: TextIO | None = None, errors: TextIO | None = None,
                 plain: bool = False):
        self.output = output or sys.stdout
        self.errors = errors or sys.stderr
        self.color = not plain and supports_color(self.output)
        self.error_color = not plain and supports_color(self.errors)
        self.requests = 0
        self.tool_calls = 0
        self.last_elapsed = None

    def styled(self, text: str, style: str, errors: bool = False) -> str:
        clean = safe_text(text)
        enabled = self.error_color if errors else self.color
        return PALETTE[style] + clean + "\033[0m" if enabled else clean

    def say(self, text: str) -> None:
        print(safe_text(text), file=self.output, flush=True)

    def progress(self, text: str) -> None:
        print(self.styled("  > " + text, "muted", errors=True), file=self.errors, flush=True)

    def panel(self, title: str, body: str, style: str = "accent", errors: bool = False,
              markdown: bool = False) -> None:
        stream = self.errors if errors else self.output
        width = max(12, min(100, shutil.get_terminal_size((88, 24)).columns - 2))
        inner_width = width - 4
        label = safe_text(title).replace("\n", " ").replace("\t", " ")[:width - 6]
        top = "+- " + label + " " + "-" * (width - len(label) - 5) + "+"
        print(self.styled(top, style, errors), file=stream)
        in_code = False
        for source in safe_text(body).expandtabs(4).splitlines():
            line_style = None
            if markdown:
                if source.lstrip().startswith("```"):
                    in_code = not in_code
                    line_style = "muted"
                elif in_code:
                    line_style = "accent"
                elif source.startswith("#"):
                    line_style = "heading"
            elif errors:
                if source.startswith("+") and not source.startswith("+++"):
                    line_style = "success"
                elif source.startswith("-") and not source.startswith("---"):
                    line_style = "error"
            lines = textwrap.wrap(source, width=inner_width, replace_whitespace=False,
                                  drop_whitespace=False, break_on_hyphens=False) or [""]
            for line in lines:
                content = self.styled(line, line_style, errors) if line_style else line
                border = self.styled("|", style, errors)
                print(f"{border} {content}{' ' * (inner_width - len(line))} {border}", file=stream)
        print(self.styled("+" + "-" * (width - 2) + "+", style, errors), file=stream, flush=True)

    def dashboard(self, agent) -> None:
        mode = "APPROVAL REQUIRED" if agent.gateway.allow_edits else "READ ONLY"
        self.say("")
        self.panel("JARVISJOON / LOCAL WORKSPACE", "Your project. Your model. Your control.\n\n"
                   f"Workspace   {agent.gateway.workspace.root}\n"
                   f"Model       {agent.runtime.model_path.name}\n"
                   f"Access      {mode}  |  No shell access\n"
                   "\nAsk a question to load the local model. Nothing is downloaded.\n"
                   "Use /build to create projects, with approval for each file.")
        print(self.styled("  /help  guide   /tools  capabilities   /status  session   /exit  leave", "muted"),
              file=self.output, flush=True)
        self.say("")

    def status(self, agent) -> None:
        runtime = agent.runtime
        elapsed = f"{self.last_elapsed:.1f}s" if self.last_elapsed is not None else "none"
        self.panel("SESSION / STATUS", f"Workspace       {agent.gateway.workspace.root}\n"
                   f"Model           {runtime.model_path}\n"
                   f"Loaded          {'yes' if runtime.loaded else 'not yet'}\n"
                   f"Device policy   {runtime.device}\n"
                   f"Actual devices  {getattr(runtime, 'execution_summary', 'not reported')}\n"
                   f"Precision       {getattr(runtime, 'precision', 'auto')}\n"
                   f"Context budget  {runtime.max_context} tokens\n"
                   f"Reply budget    {runtime.max_new_tokens} tokens\n"
                   f"Tool budget     {agent.max_tool_calls} calls per request\n"
                   f"Edit approvals  {'enabled' if agent.gateway.allow_edits else 'disabled'}\n"
                   f"Requests        {self.requests}\n"
                   f"Tool calls      {self.tool_calls}\n"
                   f"Last request    {elapsed}\n"
                   "\nHistory is in memory only. /reset does not undo edits.")

    def approve(self, preview: str) -> bool:
        if not sys.stdin.isatty() or not self.errors.isatty():
            self.progress("I need an interactive terminal to approve edits. This change is denied.")
            return False
        self.panel("REVIEW / FILE CHANGE", preview, "warning", errors=True)
        print(self.styled("  Apply this exact change? Type yes; anything else declines.",
                          "warning", errors=True), file=self.errors, flush=True)
        try:
            return input().strip().lower() == "yes"
        except EOFError:
            return False

    def show_result(self, result) -> None:
        for event in result.events:
            outcome = "OK" if event["ok"] else "FAIL"
            detail = event["result"].get("message", "")
            print(self.styled(f"  [{outcome}] {event['tool']} {detail}".rstrip(),
                              "success" if event["ok"] else "error", errors=True),
                  file=self.errors, flush=True)
        self.say("")
        self.panel("JARVISJOON / REPLY" if result.completed else "JARVISJOON / INCOMPLETE",
                   result.answer, "accent" if result.completed else "warning", markdown=True)
        if self.last_elapsed is not None:
            print(self.styled(f"  {self.last_elapsed:.1f}s  |  {len(result.events)} tool calls  |  "
                              f"request {self.requests}", "muted"), file=self.output, flush=True)
        self.say("")

    def request(self, agent, prompt: str) -> None:
        self.requests += 1
        started = time.monotonic()
        result = agent.ask(prompt)
        if getattr(agent.runtime, "loaded", False):
            self.progress(f"Execution: {getattr(agent.runtime, 'execution_summary', 'not reported')} "
                          f"| Precision: {getattr(agent.runtime, 'precision', 'auto')}")
        self.last_elapsed = time.monotonic() - started
        self.tool_calls += len(result.events)
        self.show_result(result)

    def chat(self, agent) -> int:
        self.dashboard(agent)
        while True:
            try:
                print(self.styled("YOU / REQUEST", "heading"), file=self.output, flush=True)
                prompt = input(self.styled("  > ", "accent")).strip()
                if not prompt:
                    continue
                if prompt in {"/exit", "/quit"}:
                    self.say("Take care. Your session wasn't saved automatically.")
                    return 0
                if prompt == "/help":
                    self.panel("WORKSPACE / GUIDE", HELP)
                elif prompt == "/reset":
                    agent.reset()
                    self.say("Fresh start. Any edits already applied are still in place.")
                elif prompt in {"/build", "/readonly"}:
                    agent.gateway.set_edit_mode(prompt == "/build")
                    agent.reset()
                    self.dashboard(agent)
                    self.say("Conversation reset. Existing files and applied edits are unchanged.")
                    if agent.gateway.allow_edits:
                        self.say("Describe the project you want. I'll request approval for each file change.")
                elif prompt == "/tools":
                    self.panel("WORKSPACE / TOOLS", "\n\n".join(
                        f"{tool['name']}\n  {tool['description']}" for tool in agent.gateway.definitions()))
                elif prompt == "/status":
                    self.status(agent)
                elif prompt == "/clear":
                    if self.color:
                        print("\033[2J\033[H", end="", file=self.output, flush=True)
                    else:
                        self.say("Display clearing is unavailable in plain mode; history is unchanged.")
                    self.dashboard(agent)
                elif prompt == "/paste":
                    self.say("Paste your request. Type /end on its own line when you're ready.")
                    lines = []
                    while True:
                        line = input("... ")
                        if line == "/end":
                            break
                        lines.append(line)
                    self.request(agent, "\n".join(lines))
                elif prompt.startswith("/"):
                    self.say("Unknown command. Type /help for available commands.")
                else:
                    self.request(agent, prompt)
            except KeyboardInterrupt:
                agent.reset()
                self.say("\nStopped. Conversation cleared; completed edits remain. Ask again when ready.")
            except EOFError:
                self.say("\nSee you next time. Your session wasn't saved automatically.")
                return 0

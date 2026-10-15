"""Command-line setup, diagnostics, and one-shot requests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from CLI.agent import LocalAgent
from CLI.gateway import ToolGateway
from CLI.runtime import DEFAULT_MODEL, GemmaRuntime, inspect_hardware, inspect_model
from Terminal.interface import TerminalUI, safe_text


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Please choose a positive number.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JarvisJoon: a careful teammate powered by your local Gemma model.")
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project folder I may inspect (default: current folder).")
    options.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Local checkpoint folder; never downloaded automatically.")
    options.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    options.add_argument("--context-tokens", type=positive_integer, default=4096)
    options.add_argument("--max-new-tokens", type=positive_integer, default=768)
    options.add_argument("--max-tool-calls", type=positive_integer, default=12)
    options.add_argument("--generation-timeout", type=positive_integer, default=120, help="Generation deadline in seconds, checked between tokens.")
    options.add_argument("--allow-edits", action="store_true", help="Offer file edits, each with a preview and explicit approval.")
    options.add_argument("--plain", action="store_true", help="Disable terminal colors (also respects NO_COLOR).")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("chat", parents=[options], help="Have an interactive conversation.")
    commands.add_parser("build", parents=[options], help="Build projects interactively with approved file writes.")
    ask = commands.add_parser("ask", parents=[options], help="Ask one question and exit.")
    ask.add_argument("prompt", help="Your request, enclosed in quotes.")
    ask.add_argument("--json", action="store_true", help="Write a structured result to stdout.")
    doctor = commands.add_parser("doctor", parents=[options], help="Check model files and dependencies without loading weights.")
    doctor.add_argument("--json", action="store_true")
    commands.add_parser("tools", parents=[options], help="List enabled tools without loading the model.")
    return parser


def main(argv: list[str] | None = None, default_command: str | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if default_command and (not arguments or arguments[0] not in {"chat", "build", "ask", "doctor", "tools", "-h", "--help"}):
        arguments.insert(0, default_command)
    parser = build_parser()
    options = parser.parse_args(arguments)
    if options.command is None:
        parser.print_help()
        return 0
    ui = TerminalUI(plain=options.plain)
    try:
        if options.command == "doctor":
            report = inspect_model(options.model)
            report["hardware"] = inspect_hardware()
            ready = not report["missing"] and all(report["dependencies"].values())
            if options.device == "cuda":
                ready = ready and report["hardware"]["cuda_available"]
            if options.json:
                print(json.dumps(report, indent=2))
            else:
                ui.say(f"Let's check the setup.\nModel: {report['model']}\nWeights on disk: {report['weight_bytes'] / 1e9:.2f} GB")
                ui.say("Missing files: " + (", ".join(report["missing"]) or "none detected"))
                for dependency, installed in report["dependencies"].items():
                    ui.say(f"  {dependency}: {'installed' if installed else 'not installed'}")
                hardware = report["hardware"]
                ui.say(f"PyTorch: {hardware.get('torch_version', 'unavailable')} | CUDA build: {hardware.get('cuda_build') or 'none'}")
                ui.say(f"CUDA available: {hardware['cuda_available']}")
                for gpu in hardware["gpus"]:
                    ui.say(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gib']:.2f} GiB VRAM)")
                if "error" in hardware:
                    ui.say(f"Hardware check: {hardware['error']}")
                if not hardware["cuda_available"]:
                    ui.say("Auto mode uses CPU because CUDA is unavailable. Check your PyTorch build and NVIDIA driver.")
                ui.say(report["note"])
                if not ready:
                    ui.say("Next step: python -m pip install -r CLI/requirements.txt")
            return 0 if ready else 1
        if options.max_new_tokens >= options.context_tokens:
            parser.error("--context-tokens must leave room beyond --max-new-tokens.")
        gateway = ToolGateway(options.workspace, options.allow_edits or options.command == "build", ui.approve)
        if options.command == "tools":
            for definition in gateway.definitions():
                ui.say(f"{definition['name']}: {definition['description']}")
            return 0
        runtime = GemmaRuntime(options.model, options.device, options.context_tokens,
                               options.max_new_tokens, options.generation_timeout)
        agent = LocalAgent(runtime, gateway, options.max_tool_calls, ui.progress)
        if options.command in {"chat", "build"}:
            if not sys.stdin.isatty():
                ui.progress('Chat needs an interactive terminal. For scripts, use: python -m CLI ask "your question"')
                return 2
            return ui.chat(agent)
        result = agent.ask(options.prompt)
        if options.json:
            print(json.dumps(asdict(result), ensure_ascii=True, indent=2))
        else:
            for event in result.events:
                ui.progress(f"{'Done' if event['ok'] else 'Not completed'}: {event['tool']}")
            ui.say(result.answer)
        return 0 if result.completed else 1
    except KeyboardInterrupt:
        ui.progress("Stopped. Any already-approved edits remain in place.")
        return 130
    except (OSError, ValueError) as error:
        ui.progress(f"I couldn't start this request: {safe_text(str(error))}")
        return 2

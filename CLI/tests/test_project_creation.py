from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from CLI.agent import LocalAgent
from CLI.gateway import ToolGateway
from CLI.main import main
from CLI.runtime import AssistantTurn
from Terminal.interface import TerminalUI


def call(name, **arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


class ProjectRuntime:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.model_path = Path("checkpoint")

    def generate(self, messages, definitions):
        return next(self.turns)


class ProjectCreationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_agent_creates_nested_runnable_project_after_each_approval(self):
        files = {
            "hello-project/src/main.py": "def greet(name):\n    return f'Hello, {name}!'\n\nif __name__ == '__main__':\n    print(greet('world'))\n",
            "hello-project/README.md": "# Hello project\n\nRun: python src/main.py\n",
            "hello-project/tests/test_main.py": "import unittest\nfrom src.main import greet\n\nclass GreetingTests(unittest.TestCase):\n    def test_greeting(self):\n        self.assertEqual(greet('world'), 'Hello, world!')\n",
        }
        previews = []
        gateway = ToolGateway(self.root, True, lambda preview: previews.append(preview) or True)
        turns = [AssistantTurn(tool_calls=[call("write_file", path=path, content=content)])
                 for path, content in files.items()]
        turns.append(AssistantTurn(tool_calls=[call("read_file", path="hello-project/src/main.py")]))
        turns.append(AssistantTurn(content="Created the files. Run python src/main.py; I have not run it."))
        result = LocalAgent(ProjectRuntime(turns), gateway).ask("Build a greeting project.")
        self.assertTrue(result.completed)
        self.assertEqual(len(previews), 3)
        self.assertEqual(len(result.events), 4)
        self.assertTrue(all(event["ok"] for event in result.events))
        for path, content in files.items():
            self.assertEqual((self.root / path).read_text(encoding="utf-8"), content)
        run = subprocess.run([sys.executable, "src/main.py"], cwd=self.root / "hello-project",
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.strip(), "Hello, world!")

    def test_declined_creation_does_not_create_parent_directories(self):
        gateway = ToolGateway(self.root, True, lambda preview: False)
        result = gateway.execute("write_file", {"path": "new-project/src/main.py", "content": "print('hi')"})
        self.assertEqual(result["error_type"], "approval_denied")
        self.assertFalse((self.root / "new-project").exists())

    def test_readonly_and_build_switch_tool_permissions_and_prompt(self):
        gateway = ToolGateway(self.root)
        agent = LocalAgent(ProjectRuntime([]), gateway)
        self.assertIn("read-only", agent.messages[0]["content"])
        gateway.set_edit_mode(True)
        agent.reset()
        self.assertIn("implement it using write_file", agent.messages[0]["content"])
        self.assertIn("write_file", {tool["name"] for tool in gateway.definitions()})
        gateway.set_edit_mode(False)
        agent.reset()
        self.assertNotIn("write_file", {tool["name"] for tool in gateway.definitions()})
        result = gateway.execute("write_file", {"path": "new.txt", "content": "no"})
        self.assertFalse(result["ok"])
        self.assertIn("/build", result["message"])
        self.assertFalse((self.root / "new.txt").exists())

    def test_chat_build_and_readonly_commands_do_not_write_without_request(self):
        gateway = ToolGateway(self.root)
        agent = LocalAgent(ProjectRuntime([]), gateway)
        ui = TerminalUI(io.StringIO(), io.StringIO(), plain=True)
        with patch("builtins.input", side_effect=["/build", "/readonly", "/exit"]):
            self.assertEqual(ui.chat(agent), 0)
        self.assertFalse(gateway.allow_edits)
        self.assertIn("APPROVAL REQUIRED", ui.output.getvalue())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_launcher_build_command_enables_edits_without_changing_chat_default(self):
        for command, writable in [("build", True), ("chat", False)]:
            with self.subTest(command=command):
                with patch("sys.stdin.isatty", return_value=True), patch("Terminal.interface.TerminalUI.chat", return_value=0) as chat:
                    self.assertEqual(main([command, "--workspace", str(self.root)], default_command="chat"), 0)
                self.assertEqual(chat.call_args.args[0].gateway.allow_edits, writable)

    def test_one_shot_approved_creation_keeps_json_output(self):
        runtime = ProjectRuntime([
            AssistantTurn(tool_calls=[call("write_file", path="app/main.py", content="print('hello')\n")]),
            AssistantTurn(content="Created app/main.py."),
        ])
        with patch("CLI.main.GemmaRuntime", return_value=runtime), patch("Terminal.interface.TerminalUI.approve", return_value=True):
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                status = main(["ask", "Create an app", "--workspace", str(self.root), "--allow-edits", "--json"])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())["events"][0]["ok"])
        self.assertTrue((self.root / "app/main.py").exists())


if __name__ == "__main__":
    unittest.main()

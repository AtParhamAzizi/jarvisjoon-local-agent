from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from CLI.agent import LocalAgent
from CLI.gateway import MAX_FILE_BYTES, ToolGateway, bounded_result
from CLI.main import main
from CLI.runtime import AssistantTurn, DEFAULT_MODEL, RuntimeProblem, inspect_model
from Terminal.interface import TerminalUI, safe_text


def tool_call(name: str, arguments: dict) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


class FakeRuntime:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []

    def generate(self, messages, definitions):
        self.requests.append(json.loads(json.dumps(messages)))
        turn = next(self.turns)
        if isinstance(turn, Exception):
            raise turn
        return turn


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "hello.txt").write_bytes(b"hello\nsecond line\n")
        self.gateway = ToolGateway(self.root)

    def test_read_file(self):
        result = self.gateway.execute("read_file", {"path": "hello.txt", "start_line": 2})
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "second line\n")

    def test_default_denies_edits_and_commands(self):
        for name, arguments in [("write_file", {"path": "new.txt", "content": "no"}),
                                ("execute_command", {"command": ["python", "--version"]})]:
            self.assertFalse(self.gateway.execute(name, arguments)["ok"])
        self.assertFalse((self.root / "new.txt").exists())

    def test_schema_validation(self):
        for arguments in [{}, {"path": 9}, {"path": "hello.txt", "extra": True},
                          {"path": "hello.txt", "start_line": True}]:
            self.assertFalse(self.gateway.execute("read_file", arguments)["ok"])

    def test_path_escape_and_protected_files(self):
        (self.root / ".env").write_text("secret=123", encoding="utf-8")
        for path in ["../outside.txt", ".env", ".env.local", "models/config.json", "private.key", "hello.txt:secret"]:
            self.assertFalse(self.gateway.execute("read_file", {"path": path})["ok"])

    def test_search_does_not_read_protected_files(self):
        (self.root / ".env").write_text("hello secret", encoding="utf-8")
        (self.root / "models").mkdir()
        (self.root / "models" / "note.txt").write_text("hello weights", encoding="utf-8")
        result = self.gateway.execute("search_files", {"query": "hello"})
        self.assertEqual([match["file"] for match in result["matches"]], ["hello.txt"])
        self.assertEqual(self.gateway.execute("find_files", {"pattern": "*"})["files"], ["hello.txt"])

    def test_search_limits(self):
        result = self.gateway.execute("search_files", {"query": "line", "max_results": 1})
        self.assertEqual(result["count"], 1)
        self.assertFalse(self.gateway.execute("find_files", {"pattern": "*", "max_results": 1000})["ok"])

    def test_oversized_file_is_rejected(self):
        (self.root / "large.txt").write_bytes(b"a" * (MAX_FILE_BYTES + 1))
        self.assertFalse(self.gateway.execute("read_file", {"path": "large.txt"})["ok"])

    def test_approval_default_is_deny(self):
        gateway = ToolGateway(self.root, allow_edits=True)
        result = gateway.execute("write_file", {"path": "new.txt", "content": "no"})
        self.assertEqual(result["error_type"], "approval_denied")

    def test_approved_atomic_edit_preserves_unicode_and_newlines(self):
        previews = []
        gateway = ToolGateway(self.root, True, lambda preview: previews.append(preview) or True)
        result = gateway.execute("edit_file", {"path": "hello.txt", "old_text": "hello", "new_text": "héllo"})
        self.assertTrue(result["ok"])
        self.assertIn("héllo", previews[0])
        self.assertEqual((self.root / "hello.txt").read_bytes(), "héllo\nsecond line\n".encode())
        self.assertEqual(list(self.root.glob(".jarvisjoon-*")), [])

    def test_stale_file_is_not_overwritten(self):
        def approve(preview):
            (self.root / "hello.txt").write_text("a user's edit", encoding="utf-8")
            return True

        gateway = ToolGateway(self.root, True, approve)
        result = gateway.execute("write_file", {"path": "hello.txt", "content": "model edit"})
        self.assertEqual(result["error_type"], "stale_file")
        self.assertEqual((self.root / "hello.txt").read_text(), "a user's edit")

    def test_crlf_is_preserved_during_edit(self):
        (self.root / "hello.txt").write_bytes(b"hello\r\nsecond line\r\n")
        gateway = ToolGateway(self.root, True, lambda preview: True)
        result = gateway.execute("edit_file", {"path": "hello.txt", "old_text": "hello", "new_text": "updated"})
        self.assertTrue(result["ok"])
        self.assertEqual((self.root / "hello.txt").read_bytes(), b"updated\r\nsecond line\r\n")

    def test_ambiguous_edit_never_requests_approval(self):
        approval = Mock(return_value=True)
        gateway = ToolGateway(self.root, True, approval)
        result = gateway.execute("edit_file", {"path": "hello.txt", "old_text": "missing", "new_text": "no"})
        self.assertFalse(result["ok"])
        approval.assert_not_called()

    def test_symlink_is_rejected(self):
        try:
            (self.root / "link.txt").symlink_to(self.root / "hello.txt")
        except OSError:
            self.skipTest("Creating symlinks needs platform permissions.")
        self.assertFalse(self.gateway.execute("read_file", {"path": "link.txt"})["ok"])

    def test_results_are_bounded(self):
        result = bounded_result({"ok": True, "content": "text" * 10_000})
        self.assertTrue(result["truncated"])
        self.assertLess(len(json.dumps(result)), 13_000)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "hello.txt").write_text("hello", encoding="utf-8")

    def test_tool_round_trip_uses_native_result_history(self):
        runtime = FakeRuntime([
            AssistantTurn(tool_calls=[tool_call("read_file", {"path": "hello.txt"})]),
            AssistantTurn(content="The file says hello."),
        ])
        result = LocalAgent(runtime, ToolGateway(self.root)).ask("Read hello.txt")
        self.assertTrue(result.completed)
        self.assertEqual(result.events[0]["result"]["content"], "hello")
        self.assertEqual(runtime.requests[1][-1]["tool_responses"][0]["name"], "read_file")

    def test_failure_and_reset(self):
        agent = LocalAgent(FakeRuntime([RuntimeProblem("No dependencies.")]), ToolGateway(self.root))
        self.assertFalse(agent.ask("Hello").completed)
        agent.reset()
        self.assertEqual(len(agent.messages), 1)

    def test_malformed_call_fails_closed(self):
        for call in [{"function": None}, {"function": {"name": "write_file", "arguments": "{}"}}]:
            runtime = FakeRuntime([AssistantTurn(tool_calls=[call])])
            result = LocalAgent(runtime, ToolGateway(self.root)).ask("Do something")
            self.assertFalse(result.completed)
            self.assertEqual(result.events, [])

    def test_call_limit(self):
        runtime = FakeRuntime([AssistantTurn(tool_calls=[tool_call("read_file", {"path": "hello.txt"})])] * 3)
        result = LocalAgent(runtime, ToolGateway(self.root), max_tool_calls=1).ask("Read")
        self.assertFalse(result.completed)
        self.assertEqual(len(result.events), 1)

    def test_denial_stops_without_retry(self):
        runtime = FakeRuntime([AssistantTurn(tool_calls=[tool_call("write_file", {"path": "new.txt", "content": "no"})])])
        result = LocalAgent(runtime, ToolGateway(self.root, True, lambda preview: False)).ask("Write")
        self.assertFalse(result.completed)
        self.assertFalse((self.root / "new.txt").exists())
        self.assertEqual(len(runtime.requests), 1)

    def test_repeated_failure_stops(self):
        turn = AssistantTurn(tool_calls=[tool_call("read_file", {"path": "missing.txt"})])
        runtime = FakeRuntime([turn, turn])
        result = LocalAgent(runtime, ToolGateway(self.root)).ask("Read")
        self.assertFalse(result.completed)
        self.assertEqual(len(runtime.requests), 2)


class InterfaceTests(unittest.TestCase):
    def test_terminal_escape_sequences_are_removed(self):
        self.assertEqual(safe_text("\x1b[31mHello\x1b[0m\x00"), "Hello")
        self.assertEqual(safe_text("\x1b]0;fake title\x07Hello"), "Hello")

    def test_noninteractive_approval_is_denied(self):
        ui = TerminalUI(io.StringIO(), io.StringIO())
        with patch("sys.stdin.isatty", return_value=False):
            self.assertFalse(ui.approve("Preview"))

    def test_doctor_does_not_load_model(self):
        with patch("CLI.runtime.GemmaRuntime.load", side_effect=AssertionError("must not load")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(["doctor", "--json"])
        self.assertIn("dependencies", json.loads(output.getvalue()))

    def test_model_path_does_not_depend_on_current_directory(self):
        self.assertEqual(DEFAULT_MODEL.parent.name, "models")
        self.assertIn("missing", inspect_model(DEFAULT_MODEL))

    def test_one_shot_json_keeps_stdout_machine_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("CLI.main.GemmaRuntime", return_value=FakeRuntime([AssistantTurn(content="Hello.")])):
                with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                    status = main(["ask", "Hello", "--workspace", directory, "--json"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["answer"], "Hello.")


if __name__ == "__main__":
    unittest.main()

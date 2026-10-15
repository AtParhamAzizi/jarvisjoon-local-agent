from __future__ import annotations

import io
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from CLI.agent import AgentResult
from Terminal.interface import TerminalUI, supports_color


class InteractiveStream(io.StringIO):
    def isatty(self):
        return True


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.errors = io.StringIO()
        self.ui = TerminalUI(self.output, self.errors, plain=True)
        self.agent = SimpleNamespace(
            gateway=SimpleNamespace(
                workspace=SimpleNamespace(root=Path("project")), allow_edits=False,
                definitions=lambda: [{"name": "read_file", "description": "Read a file."}],
            ),
            runtime=SimpleNamespace(model_path=Path("models/checkpoint"), loaded=False,
                                    device="auto", max_context=4096, max_new_tokens=768),
            max_tool_calls=12, reset=Mock(), ask=Mock(),
        )

    def test_plain_disables_color_even_on_a_terminal(self):
        with patch("Terminal.interface.supports_color") as detection:
            ui = TerminalUI(InteractiveStream(), InteractiveStream(), plain=True)
        detection.assert_not_called()
        self.assertFalse(ui.color)
        self.assertFalse(ui.error_color)

    def test_redirected_output_and_no_color_disable_ansi(self):
        self.assertFalse(supports_color(io.StringIO()))
        with patch.dict(os.environ, {"NO_COLOR": ""}):
            self.assertFalse(supports_color(InteractiveStream()))
        with patch.dict(os.environ, {"TERM": "dumb"}):
            self.assertFalse(supports_color(InteractiveStream()))

    def test_panels_wrap_without_losing_text(self):
        with patch("Terminal.interface.shutil.get_terminal_size", return_value=os.terminal_size((32, 24))):
            self.ui.panel("Title", "a" * 80 + "\n\n    indented")
        lines = self.output.getvalue().splitlines()
        self.assertTrue(all(len(line) == 30 for line in lines))
        content = "".join(line[2:-2].rstrip() for line in lines[1:-1])
        self.assertEqual(content.count("a"), 80)
        self.assertIn("    indented", content)

    def test_untrusted_panel_content_cannot_inject_controls(self):
        self.ui.panel("title\033[2J", "\033]0;unsafe\007hello\033[31m\x00")
        self.assertNotIn("\033", self.output.getvalue())
        self.assertNotIn("unsafe", self.output.getvalue())
        self.assertIn("hello", self.output.getvalue())

    def test_color_is_generated_only_by_renderer(self):
        self.ui.color = True
        self.ui.panel("Reply", "# Heading\n```python\nprint('hi')\n```", markdown=True)
        self.assertIn("\033[1;95m# Heading", self.output.getvalue())
        self.assertIn("\033[96mprint('hi')", self.output.getvalue())

    def test_approval_needs_interactive_input_and_errors(self):
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input") as read:
            self.assertFalse(self.ui.approve("preview"))
        read.assert_not_called()

    def test_approval_requires_explicit_yes_and_renders_full_diff(self):
        ui = TerminalUI(self.output, InteractiveStream(), plain=True)
        with patch("sys.stdin.isatty", return_value=True):
            for response, expected in [("yes", True), (" YES ", True), ("y", False), ("", False)]:
                with patch("builtins.input", return_value=response):
                    self.assertEqual(ui.approve("--- before\n+++ after\n-old\n+new"), expected)
        self.assertIn("-old", ui.errors.getvalue())
        self.assertIn("+new", ui.errors.getvalue())
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
            self.assertFalse(ui.approve("preview"))

    def test_request_tracks_actual_results_and_elapsed_time(self):
        self.agent.ask.return_value = AgentResult("Answer", True, [
            {"tool": "read_file", "ok": True, "result": {}},
            {"tool": "find_files", "ok": False, "result": {"message": "Missing"}},
        ])
        with patch("Terminal.interface.time.monotonic", side_effect=[10, 12.5]):
            self.ui.request(self.agent, "Read")
        self.agent.ask.assert_called_once_with("Read")
        self.assertEqual(self.ui.requests, 1)
        self.assertEqual(self.ui.tool_calls, 2)
        self.assertEqual(self.ui.last_elapsed, 2.5)
        self.assertIn("2.5s", self.output.getvalue())
        self.assertIn("[OK] read_file", self.errors.getvalue())
        self.assertIn("[FAIL] find_files Missing", self.errors.getvalue())

    def test_incomplete_result_is_not_labelled_success(self):
        self.ui.show_result(AgentResult("Model unavailable", False))
        self.assertIn("INCOMPLETE", self.output.getvalue())

    def test_chat_navigation_does_not_load_model(self):
        with patch("builtins.input", side_effect=["/help", "/tools", "/status", "/clear", "/exit"]):
            self.assertEqual(self.ui.chat(self.agent), 0)
        self.agent.ask.assert_not_called()
        self.agent.reset.assert_not_called()
        text = self.output.getvalue()
        for label in ["JARVISJOON", "READ ONLY", "WORKSPACE / TOOLS", "SESSION / STATUS", "4096"]:
            self.assertIn(label, text)
        self.assertNotIn("\033", text)

    def test_clear_keeps_history_but_reset_clears_it(self):
        self.ui.color = True
        with patch("builtins.input", side_effect=["/clear", "/reset", "/exit"]):
            self.ui.chat(self.agent)
        self.assertIn("\033[2J\033[H", self.output.getvalue())
        self.agent.reset.assert_called_once()

    def test_multiline_input_is_forwarded_intact(self):
        self.agent.ask.return_value = AgentResult("Done", True)
        with patch("builtins.input", side_effect=["/paste", "first", "second", "/end", "/exit"]):
            self.ui.chat(self.agent)
        self.agent.ask.assert_called_once_with("first\nsecond")

    def test_interrupt_resets_and_eof_exits(self):
        with patch("builtins.input", side_effect=[KeyboardInterrupt, EOFError]):
            self.assertEqual(self.ui.chat(self.agent), 0)
        self.agent.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()

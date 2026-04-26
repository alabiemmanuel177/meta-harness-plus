"""Tests for LLM-backed and rule-based agent policies."""

from __future__ import annotations

import unittest

from meta_harness_plus.agent import Turn
from meta_harness_plus.agent_policy import (
    RuleBasedPolicy,
    _build_prompt,
    _extract_command,
    llm_agent_policy,
)


class TestExtractCommand(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(_extract_command("ls -la"), "ls -la")

    def test_strips_dollar_prompt(self):
        self.assertEqual(_extract_command("$ ls -la"), "ls -la")

    def test_done_token(self):
        self.assertEqual(_extract_command("<DONE>"), "<DONE>")

    def test_done_token_with_extra_text(self):
        # Conversational lead-in but DONE on its own line — DONE wins
        # (we'd rather over-stop than execute chatter).
        self.assertEqual(_extract_command("Sure thing.\n<DONE>"), "<DONE>")

    def test_code_fence_bash(self):
        self.assertEqual(_extract_command("```bash\necho hi\n```"), "echo hi")

    def test_code_fence_plain(self):
        self.assertEqual(_extract_command("```\nls\n```"), "ls")

    def test_empty_yields_done(self):
        self.assertEqual(_extract_command(""), "<DONE>")

    def test_first_nonempty_line_wins(self):
        self.assertEqual(_extract_command("\n\necho x\necho y"), "echo x")


class TestBuildPrompt(unittest.TestCase):
    def test_no_history(self):
        p = _build_prompt("write hello", [], "")
        self.assertIn("Goal: write hello", p)
        self.assertIn("(no turns yet)", p)
        self.assertIn("Next shell command", p)

    def test_with_history(self):
        history = [
            Turn(turn_idx=0, observation="", action="ls",
                 stdout="a\nb\n", stderr="", exit_code=0, elapsed_s=0.01),
        ]
        p = _build_prompt("look around", history, "a\nb\n")
        self.assertIn("$ ls", p)
        self.assertIn("a\nb", p)


class TestLLMAgentPolicy(unittest.TestCase):
    def test_chat_fn_drives_actions(self):
        seen = {"sys": None, "user": None}

        def fake_chat(system, user):
            seen["sys"] = system
            seen["user"] = user
            return ("echo hi\n", 17)

        policy = llm_agent_policy(fake_chat)
        action, tokens = policy("write hello", [], "")
        self.assertEqual(action, "echo hi")
        self.assertEqual(tokens, 17)
        self.assertIn("careful shell-using agent", seen["sys"])
        self.assertIn("write hello", seen["user"])

    def test_chat_returning_done_is_propagated(self):
        def chat(system, user):
            return ("Looks done.\n<DONE>", 5)

        policy = llm_agent_policy(chat)
        action, tokens = policy("x", [], "")
        # When the model emits <DONE> on its own line, we stop — even if
        # there's chatter on a previous line. Conservative on purpose.
        self.assertEqual(action, "<DONE>")
        self.assertEqual(tokens, 5)


class TestRuleBasedPolicy(unittest.TestCase):
    def test_first_match_wins(self):
        p = RuleBasedPolicy([
            ("greet", ["echo hello", "<DONE>"]),
            ("count", ["wc -l file"]),
        ])
        a, _ = p("please greet me", [], "")
        self.assertEqual(a, "echo hello")
        a, _ = p("please greet me", [Turn(0, "", "echo hello", "hello", "", 0, 0.0)], "hello")
        self.assertEqual(a, "<DONE>")

    def test_no_rule_match_returns_done(self):
        p = RuleBasedPolicy([("a", ["x"])])
        self.assertEqual(p("b", [], "")[0], "<DONE>")


if __name__ == "__main__":
    unittest.main()

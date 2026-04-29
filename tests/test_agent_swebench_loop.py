"""Tests for the multi-turn agent loop. Uses fake chat_fn + mocked
DockerShellExecutor so the test suite stays Docker-free."""

from __future__ import annotations

import unittest
from unittest import mock

from meta_harness_plus.agent_docker import DockerShellExecutor, ExecResult
from meta_harness_plus.agent_swebench_loop import (
    BudgetGuard,
    BudgetTracker,
    RecoveryPolicy,
    TOOL_NAMES,
    TOOLS,
    Trajectory,
    _execute_tool,
    run_agent_loop,
    select_tools,
)


# ----- helpers -----

def _fake_shell(diff_text="diff --git a/x.py b/x.py\n+pass\n"):
    sh = mock.MagicMock(spec=DockerShellExecutor)
    sh.get_diff.return_value = diff_text
    sh.read_file.return_value = ExecResult("def foo(): pass", "", 0, 0.01)
    sh.read_file_range.return_value = ExecResult("    1\tdef foo(): pass", "", 0, 0.01)
    sh.search_text.return_value = ExecResult("foo.py:1:def foo(): pass", "", 0, 0.01)
    sh.go_to_definition.return_value = ExecResult("foo.py:1:1: function def foo(): pass", "", 0, 0.01)
    sh.find_references.return_value = ExecResult("foo.py:1:5: def foo(): pass", "", 0, 0.01)
    sh.replace_text.return_value = ExecResult("replaced 1 occurrence(s)", "", 0, 0.01)
    sh.write_file.return_value = ExecResult("", "", 0, 0.01)
    sh.run_reproduction.return_value = ExecResult("ok", "", 0, 0.01)
    sh.run_python.return_value = ExecResult("ok", "", 0, 0.01)
    sh.run_tests.return_value = ExecResult("PASSED", "", 0, 0.5)
    sh.list_files.return_value = ExecResult("file1.py\nfile2.py", "", 0, 0.01)
    sh.run.return_value = ExecResult("", "", 0, 0.01)
    sh.create_checkpoint.return_value = ExecResult("/tmp/mh_checkpoint_start.patch", "", 0, 0.01)
    sh.restore_checkpoint.return_value = ExecResult("", "", 0, 0.01)
    sh.reset_to_clean.return_value = ExecResult("", "", 0, 0.01)
    return sh


def _scripted_chat(scripted_responses):
    """Build a chat_fn that returns each pre-canned response in turn."""
    state = {"i": 0}

    def chat(**kwargs):
        i = state["i"]
        state["i"] += 1
        if i >= len(scripted_responses):
            return {"content": [], "usage": {"input_tokens": 0, "output_tokens": 0},
                    "stop_reason": "end_turn"}
        return scripted_responses[i]

    return chat


# ----- tool-set selection -----

class TestSelectTools(unittest.TestCase):
    def test_minimal_excludes_bash(self):
        names = {t["name"] for t in select_tools("minimal")}
        self.assertIn("read_file", names)
        self.assertIn("read_file_range", names)
        self.assertIn("search_text", names)
        self.assertIn("go_to_definition", names)
        self.assertIn("find_references", names)
        self.assertIn("replace_text", names)
        self.assertIn("write_file", names)
        self.assertIn("run_reproduction", names)
        self.assertIn("run_python", names)
        self.assertIn("run_tests", names)
        self.assertIn("propose_test_fix", names)
        self.assertIn("list_files", names)
        self.assertIn("done", names)
        self.assertNotIn("bash", names)

    def test_full_includes_all(self):
        names = {t["name"] for t in select_tools("full")}
        self.assertEqual(names, TOOL_NAMES)


# ----- BudgetGuard / Tracker -----

class TestBudgetTracker(unittest.TestCase):
    def test_max_turns(self):
        t = BudgetTracker(guard=BudgetGuard(max_turns=3))
        for _ in range(2):
            t.add_turn(100, 50, 1.0, 0.001)
        self.assertEqual(t.should_stop(), (False, ""))
        t.add_turn(100, 50, 1.0, 0.001)
        stop, reason = t.should_stop()
        self.assertTrue(stop)
        self.assertIn("max_turns", reason)

    def test_usd_cap(self):
        t = BudgetTracker(guard=BudgetGuard(max_usd=0.01))
        t.add_turn(0, 0, 0.0, 0.005)
        self.assertEqual(t.should_stop(), (False, ""))
        t.add_turn(0, 0, 0.0, 0.010)
        stop, reason = t.should_stop()
        self.assertTrue(stop)
        self.assertIn("usd", reason)

    def test_wall_cap(self):
        t = BudgetTracker(guard=BudgetGuard(max_wall_s=1.0))
        t.add_turn(0, 0, 1.5, 0)
        stop, reason = t.should_stop()
        self.assertTrue(stop)
        self.assertIn("wall", reason)


# ----- _execute_tool -----

class TestExecuteTool(unittest.TestCase):
    def test_read_file(self):
        sh = _fake_shell()
        out = _execute_tool(sh, "read_file", {"path": "foo.py"},
                            fail_to_pass=[])
        self.assertIn("def foo()", out)

    def test_write_file(self):
        sh = _fake_shell()
        out = _execute_tool(sh, "write_file",
                            {"path": "bar.py", "content": "x = 1"},
                            fail_to_pass=[])
        self.assertIn("Wrote", out)
        sh.write_file.assert_called_with("bar.py", "x = 1")

    def test_write_file_blocks_tests(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "write_file",
            {"path": "pkg/tests/test_bug.py", "content": "x = 1"},
            fail_to_pass=[],
        )
        self.assertIn("editing test files is disabled", out)
        sh.write_file.assert_not_called()

    def test_read_file_range(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "read_file_range",
            {"path": "foo.py", "start_line": 1, "end_line": 3},
            fail_to_pass=[],
        )
        self.assertIn("def foo()", out)
        sh.read_file_range.assert_called_with("foo.py", 1, 3)

    def test_search_text(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "search_text",
            {"query": "def foo", "path": "pkg", "max_matches": 5},
            fail_to_pass=[],
        )
        self.assertIn("foo.py:1", out)
        sh.search_text.assert_called_with("def foo", "pkg", max_matches=5)

    def test_go_to_definition(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "go_to_definition",
            {"symbol": "foo", "path": "pkg", "max_results": 3},
            fail_to_pass=[],
        )
        self.assertIn("function", out)
        sh.go_to_definition.assert_called_with("foo", "pkg", max_results=3)

    def test_find_references(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "find_references",
            {"symbol": "foo", "path": "pkg", "max_matches": 4},
            fail_to_pass=[],
        )
        self.assertIn("foo.py", out)
        sh.find_references.assert_called_with("foo", "pkg", max_matches=4)

    def test_replace_text(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "replace_text",
            {"path": "foo.py", "old": "pass", "new": "return 1"},
            fail_to_pass=[],
        )
        self.assertIn("replaced", out)
        sh.replace_text.assert_called_with(
            "foo.py",
            "pass",
            "return 1",
            expected_count=1,
        )

    def test_replace_text_blocks_tests(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "replace_text",
            {"path": "pkg/tests/test_bug.py", "old": "a", "new": "b"},
            fail_to_pass=[],
        )
        self.assertIn("editing test files is disabled", out)
        sh.replace_text.assert_not_called()

    def test_run_reproduction(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "run_reproduction",
            {
                "code": "print('ok')",
                "expected_behavior": "prints ok",
                "timeout_s": 5,
            },
            fail_to_pass=[],
        )
        self.assertIn("Reproduction exit_code=0", out)
        self.assertIn("prints ok", out)
        sh.run_reproduction.assert_called_with("print('ok')", timeout_s=5.0)

    def test_run_python(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "run_python",
            {"code": "print('ok')", "timeout_s": 5},
            fail_to_pass=[],
        )
        self.assertIn("Python exit_code=0", out)
        self.assertIn("ok", out)
        sh.run_python.assert_called_with("print('ok')", timeout_s=5.0)

    def test_run_tests_with_default_uses_fail_to_pass(self):
        sh = _fake_shell()
        out = _execute_tool(sh, "run_tests", {"test_targets": []},
                            fail_to_pass=["tests/test_a.py::test_foo"])
        sh.run_tests.assert_called()
        # Targets passed in were FAIL_TO_PASS.
        called_targets = sh.run_tests.call_args.args[0]
        self.assertEqual(called_targets, ["tests/test_a.py::test_foo"])

    def test_done_returns_summary(self):
        sh = _fake_shell()
        out = _execute_tool(sh, "done", {"summary": "fixed it"},
                            fail_to_pass=[])
        self.assertIn("fixed it", out)

    def test_propose_test_fix(self):
        sh = _fake_shell()
        out = _execute_tool(
            sh,
            "propose_test_fix",
            {
                "test_path": "tests/test_bug.py",
                "reason": "asserts old behavior",
                "patch": "diff --git a/tests/test_bug.py b/tests/test_bug.py\n",
            },
            fail_to_pass=[],
        )
        self.assertIn("Test-fix proposal recorded", out)

    def test_unknown_tool(self):
        sh = _fake_shell()
        out = _execute_tool(sh, "delete_universe", {},
                            fail_to_pass=[])
        self.assertIn("Unknown tool", out)


# ----- full agent loop -----

class TestRunAgentLoop(unittest.TestCase):
    def test_done_after_two_turns(self):
        # Turn 0: read_file.  Turn 1: done.
        responses = [
            {
                "content": [
                    {"type": "text", "text": "Let me look."},
                    {"type": "tool_use", "id": "tu1", "name": "read_file",
                     "input": {"path": "x.py"}},
                ],
                "usage": {"input_tokens": 100, "output_tokens": 30},
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu2", "name": "done",
                     "input": {"summary": "ok"}},
                ],
                "usage": {"input_tokens": 200, "output_tokens": 20},
                "stop_reason": "tool_use",
            },
        ]
        # No FAIL_TO_PASS targets → done verification gate is bypassed.
        traj = run_agent_loop(
            instance_id="x__x-1",
            problem_statement="bug",
            fail_to_pass=[],
            sh=_fake_shell(),
            chat_fn=_scripted_chat(responses),
            system_prompt="sys",
        )
        self.assertTrue(traj.done_emitted)
        self.assertEqual(traj.stop_reason, "done")
        self.assertEqual(len(traj.turns), 2)
        self.assertGreater(len(traj.final_patch), 0)
        self.assertEqual(traj.total_in_tokens, 300)
        self.assertEqual(traj.total_out_tokens, 50)

    def test_done_rejected_when_fail_to_pass_not_verified(self):
        # Done called before any successful FAIL_TO_PASS run → first done
        # is redirected; second done is accepted (one redirect only).
        responses = [
            {
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "done",
                     "input": {"summary": "first done"}},
                ],
                "usage": {"input_tokens": 50, "output_tokens": 10},
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu2", "name": "done",
                     "input": {"summary": "second done"}},
                ],
                "usage": {"input_tokens": 60, "output_tokens": 10},
                "stop_reason": "tool_use",
            },
        ]
        traj = run_agent_loop(
            instance_id="x__x-1",
            problem_statement="bug",
            fail_to_pass=["tests/test_a.py::test_foo"],
            sh=_fake_shell(),
            chat_fn=_scripted_chat(responses),
            system_prompt="sys",
        )
        # First done was redirected, second was accepted.
        self.assertTrue(traj.done_emitted)
        self.assertEqual(traj.stop_reason, "done")
        self.assertEqual(len(traj.turns), 2)

    def test_successful_tests_autostop_with_production_patch(self):
        responses = [
            {
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "replace_text",
                     "input": {"path": "x.py", "old": "a", "new": "b"}},
                ],
                "usage": {"input_tokens": 100, "output_tokens": 30},
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu2", "name": "run_tests",
                     "input": {"test_targets": ["tests/test_a.py"]}},
                ],
                "usage": {"input_tokens": 200, "output_tokens": 20},
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu3", "name": "read_file",
                     "input": {"path": "should_not_run.py"}},
                ],
                "usage": {"input_tokens": 300, "output_tokens": 20},
                "stop_reason": "tool_use",
            },
        ]
        sh = _fake_shell(diff_text="diff --git a/x.py b/x.py\n+fixed\n")
        traj = run_agent_loop(
            instance_id="x__x-1",
            problem_statement="bug",
            fail_to_pass=["tests/test_a.py"],
            sh=sh,
            chat_fn=_scripted_chat(responses),
            system_prompt="sys",
        )
        self.assertTrue(traj.done_emitted)
        self.assertEqual(traj.stop_reason, "tests_passed")
        self.assertEqual(len(traj.turns), 2)
        sh.get_diff.assert_called_with(exclude_tests=True)

    def test_max_turns_cap(self):
        # All turns: write_file + no done => loop hits max_turns.
        endless = {
            "content": [
                {"type": "tool_use", "id": "tu", "name": "write_file",
                 "input": {"path": "x.py", "content": "x"}},
            ],
            "usage": {"input_tokens": 50, "output_tokens": 5},
            "stop_reason": "tool_use",
        }
        traj = run_agent_loop(
            instance_id="x", problem_statement="x", fail_to_pass=[],
            sh=_fake_shell(), chat_fn=_scripted_chat([endless] * 10),
            system_prompt="sys",
            budget_guard=BudgetGuard(max_turns=3, max_usd=999.0),
        )
        self.assertFalse(traj.done_emitted)
        self.assertIn("max_turns", traj.stop_reason)
        self.assertEqual(len(traj.turns), 3)

    def test_no_tool_call_exits(self):
        only_text = {
            "content": [{"type": "text", "text": "I give up."}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
        traj = run_agent_loop(
            instance_id="x", problem_statement="x", fail_to_pass=[],
            sh=_fake_shell(), chat_fn=_scripted_chat([only_text]),
            system_prompt="sys",
        )
        self.assertEqual(traj.stop_reason, "no_tool_call")
        self.assertEqual(len(traj.turns), 1)

    def test_max_tokens_without_tool_gets_recovery_turn(self):
        responses = [
            {
                "content": [{"type": "text", "text": "Long unfinished thought"}],
                "usage": {"input_tokens": 10, "output_tokens": 100},
                "stop_reason": "max_tokens",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu2", "name": "done",
                     "input": {"summary": "ok"}},
                ],
                "usage": {"input_tokens": 20, "output_tokens": 5},
                "stop_reason": "tool_use",
            },
        ]
        traj = run_agent_loop(
            instance_id="x", problem_statement="x", fail_to_pass=[],
            sh=_fake_shell(), chat_fn=_scripted_chat(responses),
            system_prompt="sys",
            budget_guard=BudgetGuard(max_turns=3, max_usd=999.0),
        )
        self.assertTrue(traj.done_emitted)
        self.assertEqual(traj.stop_reason, "done")
        self.assertEqual(len(traj.turns), 2)

    def test_reproduction_gate_blocks_first_edit(self):
        responses = [
            {
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "replace_text",
                     "input": {"path": "x.py", "old": "a", "new": "b"}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu2", "name": "run_reproduction",
                     "input": {"code": "print('ok')", "expected_behavior": "ok"}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu3", "name": "replace_text",
                     "input": {"path": "x.py", "old": "a", "new": "b"}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu4", "name": "done",
                     "input": {"summary": "ok"}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "tool_use",
            },
        ]
        sh = _fake_shell()
        traj = run_agent_loop(
            instance_id="x", problem_statement="x", fail_to_pass=[],
            sh=sh, chat_fn=_scripted_chat(responses), system_prompt="sys",
            require_reproduction_before_edit=True,
        )
        self.assertTrue(traj.reproduction_attempted)
        self.assertTrue(traj.reproduction_passed)
        self.assertEqual(sh.replace_text.call_count, 1)
        first_obs = traj.turns[0].tool_results[0]["content"]
        self.assertIn("Speculative test gate", first_obs)

    def test_state_recovery_restores_after_repeated_same_file_edits(self):
        edit = {
            "content": [
                {"type": "tool_use", "id": "tu", "name": "replace_text",
                 "input": {"path": "x.py", "old": "a", "new": "b"}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "tool_use",
        }
        done = {
            "content": [
                {"type": "tool_use", "id": "done", "name": "done",
                 "input": {"summary": "ok"}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "tool_use",
        }
        sh = _fake_shell()
        traj = run_agent_loop(
            instance_id="x", problem_statement="x", fail_to_pass=[],
            sh=sh, chat_fn=_scripted_chat([edit, edit, edit, done]),
            system_prompt="sys",
            recovery_policy=RecoveryPolicy(
                enabled=True,
                same_file_edit_threshold=3,
                max_recoveries=1,
            ),
        )
        self.assertEqual(len(traj.recovery_events), 1)
        self.assertEqual(traj.recovery_events[0].edited_path, "x.py")
        sh.restore_checkpoint.assert_called()

    def test_test_fix_proposal_is_recorded(self):
        responses = [
            {
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "propose_test_fix",
                     "input": {
                         "test_path": "tests/test_bug.py",
                         "reason": "hidden test asserts stale behavior",
                         "patch": "diff --git a/tests/test_bug.py b/tests/test_bug.py\n",
                     }},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {"type": "tool_use", "id": "tu2", "name": "done",
                     "input": {"summary": "audit only"}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "tool_use",
            },
        ]
        traj = run_agent_loop(
            instance_id="x", problem_statement="x", fail_to_pass=[],
            sh=_fake_shell(), chat_fn=_scripted_chat(responses),
            system_prompt="sys",
        )
        self.assertEqual(len(traj.test_fix_proposals), 1)
        self.assertIn("stale behavior", traj.test_fix_proposals[0]["reason"])


if __name__ == "__main__":
    unittest.main()

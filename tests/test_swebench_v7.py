"""Deterministic tests for v7 reviewer/audit helpers."""

from __future__ import annotations

import json
import unittest

from meta_harness_plus.agent_swebench_loop import Trajectory, TurnRecord
from meta_harness_plus.swebench_adapter import SWEBenchInstance
from meta_harness_plus.swebench_v7 import (
    build_revision_problem_statement,
    run_patch_reviewer,
    trajectory_summary,
)


def _inst() -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id="sympy__sympy-1",
        repo="sympy/sympy",
        base_commit="abc123",
        problem_statement="sqrt simplification is wrong",
        hints_text="",
        test_patch="",
        fail_to_pass=["tests/test_bug.py::test_case"],
        pass_to_pass=[],
        version="1.0",
        gold_patch="",
    )


class TestV7Reviewer(unittest.TestCase):
    def test_run_patch_reviewer_parses_json(self):
        seen = {}

        def chat(**kwargs):
            seen.update(kwargs)
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "approved": False,
                    "needs_revision": True,
                    "risk_summary": "misses None input",
                    "edge_cases": ["None input"],
                    "reproduction_code": "assert f(None) is None",
                    "test_commands": ["python -m pytest tests/test_bug.py"],
                    "suspect_benchmark_test": True,
                    "test_fix_reason": "selector asserts old behavior",
                    "proposed_test_patch": "diff --git a/tests/test_bug.py b/tests/test_bug.py\n",
                })}],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

        verdict = run_patch_reviewer(
            inst=_inst(),
            patch="diff --git a/x.py b/x.py\n",
            actor_trajectory=None,
            reviewer_chat_fn=chat,
            model="reviewer",
        )
        self.assertFalse(verdict.approved)
        self.assertTrue(verdict.needs_revision)
        self.assertEqual(verdict.edge_cases, ["None input"])
        self.assertTrue(verdict.suspect_benchmark_test)
        self.assertGreater(verdict.usd, 0)
        self.assertEqual(seen["tools"], [])
        self.assertIn("Proposed production patch", seen["messages"][0]["content"])

    def test_trajectory_summary_keeps_recent_tool_context(self):
        traj = Trajectory(instance_id="x")
        traj.turns.append(TurnRecord(
            turn_idx=0,
            text="",
            tool_calls=[{"name": "run_reproduction"}],
            tool_results=[{"content": "Reproduction exit_code=1\nboom"}],
            in_tokens=1,
            out_tokens=1,
            wall_ms=1,
        ))
        out = trajectory_summary(traj)
        self.assertIn("run_reproduction", out)
        self.assertIn("boom", out)

    def test_revision_problem_statement_includes_reviewer_feedback(self):
        verdict = run_patch_reviewer(
            inst=_inst(),
            patch="diff --git a/x.py b/x.py\n",
            actor_trajectory=None,
            reviewer_chat_fn=lambda **_: {
                "content": [{"type": "text", "text": json.dumps({
                    "approved": False,
                    "needs_revision": True,
                    "risk_summary": "edge risk",
                    "edge_cases": ["empty list"],
                    "reproduction_code": "",
                    "test_commands": ["python -m pytest tests/test_bug.py"],
                    "suspect_benchmark_test": False,
                    "test_fix_reason": "",
                    "proposed_test_patch": "",
                })}],
                "usage": {},
            },
            model="reviewer",
        )
        prompt = build_revision_problem_statement(_inst(), verdict)
        self.assertIn("edge risk", prompt)
        self.assertIn("empty list", prompt)
        self.assertIn("pytest", prompt)


if __name__ == "__main__":
    unittest.main()

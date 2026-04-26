"""Tests for the agent-task abstraction (multi-turn observation/action loop)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_harness_plus.agent import (
    AgentEnv,
    AgentExample,
    AgentHarness,
    AgentScorer,
    AgentTask,
    LocalSandboxShell,
    MockShell,
    Trajectory,
    Turn,
    trajectory_to_dict,
    write_trajectories,
)
from meta_harness_plus.tasks.terminalbench_fixture import build_terminalbench_fixture


# ----- helpers -----

def _scripted_policy(actions):
    """Build a deterministic policy that emits the given list of actions
    in order, then ``<DONE>`` forever after."""
    state = {"i": 0}

    def policy(goal, history, last_obs):
        i = state["i"]
        state["i"] += 1
        if i >= len(actions):
            return ("<DONE>", 1)
        return (actions[i], 5)

    return policy


def _always_done_policy(goal, history, last_obs):
    return ("<DONE>", 0)


def _make_sandbox():
    return Path(tempfile.mkdtemp(prefix="mh_agent_test_"))


# ----- LocalSandboxShell -----

class TestLocalSandboxShell(unittest.TestCase):
    def setUp(self):
        self.sandbox = _make_sandbox()

    def tearDown(self):
        shutil.rmtree(self.sandbox, ignore_errors=True)

    def test_run_simple(self):
        sh = LocalSandboxShell()
        out, err, code = sh.run("echo hi", cwd=self.sandbox, timeout_s=5.0)
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "hi")

    def test_timeout(self):
        sh = LocalSandboxShell()
        out, err, code = sh.run("sleep 2", cwd=self.sandbox, timeout_s=0.3)
        self.assertEqual(code, 124)
        self.assertIn("timeout", err.lower())

    def test_denylist_blocks_obvious_destruction(self):
        sh = LocalSandboxShell()
        for cmd in ("rm -rf /", "shutdown now", "mkfs.ext4 /dev/sda", "REBOOT"):
            out, err, code = sh.run(cmd, cwd=self.sandbox, timeout_s=1.0)
            self.assertEqual(code, 126, f"command should be denied: {cmd!r}")
            self.assertIn("denied", err.lower())

    def test_cwd_is_sandbox(self):
        sh = LocalSandboxShell()
        out, _, _ = sh.run("pwd", cwd=self.sandbox, timeout_s=1.0)
        self.assertEqual(out.strip(), str(self.sandbox.resolve()).rstrip("/"))


# ----- AgentHarness loop semantics -----

class TestAgentHarness(unittest.TestCase):
    def setUp(self):
        self.sandbox = _make_sandbox()

    def tearDown(self):
        shutil.rmtree(self.sandbox, ignore_errors=True)

    def test_runs_until_done(self):
        # Policy: write a file, then say done.
        policy = _scripted_policy(["echo hello > out.txt", "<DONE>"])

        def check(env: AgentEnv) -> bool:
            return env.read_file("out.txt").strip() == "hello"

        ex = AgentExample(
            task_id="t1", goal="write hello", setup=lambda p: None,
            check_fn=check, max_turns=4, timeout_s=5.0,
        )
        harness = AgentHarness(policy=policy)
        traj = harness.run(ex, executor=LocalSandboxShell(), sandbox=self.sandbox)
        self.assertTrue(traj.success)
        # 1 shell turn + 1 done turn = 2 turns
        self.assertEqual(traj.n_turns, 2)
        self.assertGreater(traj.total_tokens, 0)

    def test_hits_max_turns(self):
        policy = _scripted_policy(["true", "true", "true", "true", "true"])

        def check(env: AgentEnv) -> bool:
            return False

        ex = AgentExample(
            task_id="t2", goal="loop forever", setup=lambda p: None,
            check_fn=check, max_turns=3, timeout_s=5.0,
        )
        harness = AgentHarness(policy=policy)
        traj = harness.run(ex, executor=LocalSandboxShell(), sandbox=self.sandbox)
        self.assertEqual(traj.n_turns, 3)
        self.assertTrue(traj.hit_max_turns)
        self.assertFalse(traj.success)

    def test_records_per_turn_cost(self):
        policy = _scripted_policy(["echo a", "echo b", "<DONE>"])

        def check(env: AgentEnv) -> bool:
            return True

        ex = AgentExample(
            task_id="t3", goal="echo twice", setup=lambda p: None,
            check_fn=check, max_turns=4, timeout_s=5.0,
        )
        harness = AgentHarness(policy=policy)
        traj = harness.run(ex, executor=LocalSandboxShell(), sandbox=self.sandbox)
        self.assertEqual(traj.n_turns, 3)
        # Two real shell calls + one DONE turn — all carry est_tokens=5 from policy.
        self.assertEqual(traj.total_tokens, 5 * 3)
        self.assertGreater(traj.total_latency_ms, 0.0)

    def test_observation_truncation(self):
        # Long stdout — verify truncation kicks in.
        long_cmd = 'python3 -c "print(\\"x\\" * 5000)"'
        policy = _scripted_policy([long_cmd, "<DONE>"])

        def check(env: AgentEnv) -> bool:
            return True

        ex = AgentExample(
            task_id="t_trunc", goal="long output", setup=lambda p: None,
            check_fn=check, max_turns=3, timeout_s=5.0,
        )
        harness = AgentHarness(policy=policy, max_observation_chars=200)
        traj = harness.run(ex, executor=LocalSandboxShell(), sandbox=self.sandbox)
        self.assertTrue(traj.turns[0].truncated)
        self.assertLessEqual(len(traj.turns[0].stdout), 220)

    def test_setup_failure_recorded(self):
        def bad_setup(p):
            raise RuntimeError("boom")

        ex = AgentExample(
            task_id="t_bad", goal="x", setup=bad_setup,
            check_fn=lambda env: False, max_turns=2, timeout_s=2.0,
        )
        harness = AgentHarness(policy=_always_done_policy)
        traj = harness.run(ex, executor=LocalSandboxShell(), sandbox=self.sandbox)
        self.assertIsNotNone(traj.error)
        self.assertIn("setup failed", traj.error)
        self.assertFalse(traj.success)
        self.assertEqual(traj.n_turns, 0)

    def test_policy_failure_recorded(self):
        def bad_policy(goal, history, last_obs):
            raise RuntimeError("model errored")

        ex = AgentExample(
            task_id="t_polerr", goal="x", setup=lambda p: None,
            check_fn=lambda env: True, max_turns=2, timeout_s=2.0,
        )
        harness = AgentHarness(policy=bad_policy)
        traj = harness.run(ex, executor=LocalSandboxShell(), sandbox=self.sandbox)
        self.assertIsNotNone(traj.error)
        self.assertIn("policy failed", traj.error)

    def test_command_timeout_recorded(self):
        policy = _scripted_policy(["sleep 5", "<DONE>"])

        ex = AgentExample(
            task_id="t_to", goal="hang", setup=lambda p: None,
            check_fn=lambda env: False, max_turns=2, timeout_s=0.3,
        )
        harness = AgentHarness(policy=policy)
        traj = harness.run(ex, executor=LocalSandboxShell(), sandbox=self.sandbox)
        self.assertTrue(traj.timed_out)
        self.assertEqual(traj.turns[0].exit_code, 124)


# ----- MockShell -----

class TestMockShell(unittest.TestCase):
    def test_pattern_match(self):
        sh = MockShell([
            ("echo hi", ("hi\n", "", 0)),
            ("ls", ("a\nb\n", "", 0)),
        ])
        self.assertEqual(sh.run("echo hi", Path("/"), 1.0), ("hi\n", "", 0))
        self.assertEqual(sh.run("ls -la", Path("/"), 1.0), ("a\nb\n", "", 0))

    def test_unmatched_returns_127(self):
        sh = MockShell([])
        out, err, code = sh.run("nonsense", Path("/"), 1.0)
        self.assertEqual(code, 127)


# ----- AgentScorer -----

class TestAgentScorer(unittest.TestCase):
    def test_score_two_examples(self):
        # Two tasks: one that succeeds, one that fails (always).
        ex_pass = AgentExample(
            task_id="pass",
            goal="echo to file",
            setup=lambda p: None,
            check_fn=lambda env: env.read_file("done.txt").strip() == "ok",
            max_turns=3, timeout_s=2.0,
        )
        ex_fail = AgentExample(
            task_id="fail", goal="impossible",
            setup=lambda p: None,
            check_fn=lambda env: False,
            max_turns=2, timeout_s=2.0,
        )
        task = AgentTask("mini", [ex_pass, ex_fail])

        # Policy is per-example aware via dispatch on goal.
        def policy(goal, history, last_obs):
            if "echo to file" in goal:
                if not history:
                    return ("echo ok > done.txt", 5)
                return ("<DONE>", 1)
            # impossible: just <DONE> immediately so we don't hit max turns
            return ("<DONE>", 1)

        harness = AgentHarness(policy=policy)
        scorer = AgentScorer(task=task)
        sv = scorer.score(harness)
        self.assertEqual(sv.n_evaluated, 2)
        self.assertAlmostEqual(sv.accuracy, 0.5)

    def test_n_repeats_returns_spread(self):
        ex = AgentExample(
            task_id="ok", goal="x", setup=lambda p: None,
            check_fn=lambda env: True, max_turns=1, timeout_s=2.0,
        )
        task = AgentTask("solo", [ex])
        harness = AgentHarness(policy=_always_done_policy)
        scorer = AgentScorer(task=task)
        sv = scorer.score(harness, n_repeats=3)
        self.assertEqual(sv.n_repeats, 3)
        # Deterministic policy → spread is 0.
        self.assertEqual(sv.accuracy_spread, 0.0)

    def test_keep_sandboxes(self):
        ex = AgentExample(
            task_id="keep", goal="x", setup=lambda p: (p / "foo.txt").write_text("bar"),
            check_fn=lambda env: env.read_file("foo.txt") == "bar",
            max_turns=1, timeout_s=2.0,
        )
        task = AgentTask("keep_solo", [ex])
        with tempfile.TemporaryDirectory() as root:
            scorer = AgentScorer(task=task, sandbox_root=root, keep_sandboxes=True)
            scorer.score(AgentHarness(policy=_always_done_policy))
            # Some agent_keep_*.* dir should still exist under root.
            kept = list(Path(root).iterdir())
            self.assertTrue(any("foo.txt" in str(list(d.iterdir())) for d in kept))


# ----- Local TerminalBench-style fixture end-to-end (no LLM) -----

class TestTerminalBenchFixture(unittest.TestCase):
    """Run a known-good shell-script policy over the local fixture and
    verify each fixture task is actually solvable end-to-end. This is
    the deterministic smoke test the agent infrastructure needs."""

    def test_solvable_with_oracle_policy(self):
        task = build_terminalbench_fixture()
        # Hand-author an oracle policy keyed on task_id (extracted from
        # goal text). Each branch issues the exact shell commands that
        # solve the task. Use single quotes outside, single quotes inside
        # only via printf-friendly escapes — bash -c on the executor
        # interprets these correctly.
        scripts = {
            "Count how many times the word 'alpha'": [
                "grep -o 'alpha' doc.txt | wc -l | tr -d ' ' > answer.txt",
                "<DONE>",
            ],
            "Read users.json": [
                # Single-quoted python -c body so $ won't be interpolated
                # and double quotes inside python are safe.
                ("python3 -c 'import json;"
                 "d=json.load(open(\"users.json\"));"
                 "print(max(d[\"users\"],key=lambda u:u[\"age\"])[\"name\"])'"
                 " > oldest.txt"),
                "<DONE>",
            ],
            "Rename every *.log file": [
                "for f in *.log; do mv \"$f\" \"${f%.log}.txt\"; done",
                "<DONE>",
            ],
            "Read spec.txt and produce solution.py": [
                "printf 'def add(a, b):\\n    return a + b\\n' > solution.py",
                "<DONE>",
            ],
            "Extract every line containing 'ERROR'": [
                "grep ERROR log.txt > errors.txt",
                "<DONE>",
            ],
            "Compute the sum of all integers": [
                # Single-quoted awk so the $1 stays literal.
                "awk '{s+=$1} END {print s}' nums.txt > sum.txt",
                "<DONE>",
            ],
        }

        def oracle(goal, history, last_obs):
            for needle, steps in scripts.items():
                if needle in goal:
                    i = len(history)
                    if i < len(steps):
                        return (steps[i], 10)
                    return ("<DONE>", 1)
            return ("<DONE>", 1)

        scorer = AgentScorer(task=task)
        sv = scorer.score(AgentHarness(policy=oracle))
        self.assertAlmostEqual(sv.accuracy, 1.0, places=3,
                               msg="oracle policy should solve every fixture task")
        self.assertEqual(sv.n_evaluated, len(task.examples))

    def test_failing_policy_scores_zero(self):
        task = build_terminalbench_fixture()
        sv = AgentScorer(task=task).score(AgentHarness(policy=_always_done_policy))
        self.assertEqual(sv.accuracy, 0.0)


# ----- trajectory IO -----

class TestTrajectoryIO(unittest.TestCase):
    def test_to_dict_and_jsonl(self):
        traj = Trajectory(task_id="x", success=True, n_turns=1)
        traj.turns.append(Turn(
            turn_idx=0, observation="", action="echo hi",
            stdout="hi\n", stderr="", exit_code=0, elapsed_s=0.01, tokens=3,
        ))
        d = trajectory_to_dict(traj)
        self.assertEqual(d["task_id"], "x")
        self.assertEqual(d["turns"][0]["action"], "echo hi")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trajs.jsonl"
            write_trajectories([traj, traj], p)
            content = p.read_text().splitlines()
            self.assertEqual(len(content), 2)


if __name__ == "__main__":
    unittest.main()

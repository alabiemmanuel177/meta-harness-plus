"""Agent-task abstraction: multi-turn observation/action loops.

The ``Task`` / ``Harness`` / ``Scorer`` triple in the rest of this
package is built for **classification**: one input → one prediction →
binary correctness. Agent tasks (TerminalBench-style) need:

- multi-turn observation/action loops,
- a sandboxed shell or pluggable executor,
- per-turn token / latency / cost accounting,
- a success criterion that can be a pluggable ``check_fn`` instead of
  a single label match.

This module defines the agent-side abstractions that mirror the
classification side without forking the framework. The same
``MetaSearch`` proposer/voter machinery can search over agent
``AgentHarness`` shapes once an ``AgentScorer`` is in place — the
shapes just happen to compose ``AgentPolicy`` components instead of
classification ``Component`` s.

Design:

- ``ShellExecutor`` — pluggable shell-command runner. Default impl
  ``LocalSandboxShell`` runs commands in a per-task temp directory
  with a hard timeout; safer than letting the agent ``rm -rf /``.
- ``AgentEnv`` — owns the executor, holds state across turns.
- ``AgentHarness`` — wraps a ``policy`` callable; runs the
  observation/action loop until ``done`` or ``max_turns``.
- ``AgentTask`` — a sequence of ``AgentExample`` with a per-example
  ``check_fn(env_state) -> bool``.
- ``AgentScorer`` — runs the ``AgentHarness`` over each example,
  records per-example correctness + per-turn cost, returns a
  ``ScoreVector`` in the same shape the classification ``Scorer``
  uses (so existing Pareto/aggregation machinery works unchanged).

The local TerminalBench-style fixture lives in
``meta_harness_plus.tasks.terminalbench_fixture``; a real-TerminalBench
adapter stub lives in ``meta_harness_plus.tasks.terminalbench_adapter``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .scorer import ScoreVector


# ----------------- example & trajectory types -----------------

@dataclass
class AgentExample:
    """One agent task instance.

    ``setup`` is a callable taking the sandbox path and seeding files
    or initial state. ``check_fn`` decides success after the loop ends.
    ``goal`` is the natural-language instruction shown to the agent.
    """
    task_id: str
    goal: str
    setup: Callable[[Path], None] = field(default=lambda p: None)
    check_fn: Callable[["AgentEnv"], bool] = field(default=lambda env: False)
    max_turns: int = 6
    timeout_s: float = 30.0
    meta: dict = field(default_factory=dict)


@dataclass
class Turn:
    """One observation/action exchange."""
    turn_idx: int
    observation: str
    action: str
    stdout: str
    stderr: str
    exit_code: int
    elapsed_s: float
    tokens: int = 0
    truncated: bool = False


@dataclass
class Trajectory:
    """The full record of an agent's run on a single example."""
    task_id: str
    turns: list[Turn] = field(default_factory=list)
    success: bool = False
    final_observation: str = ""
    error: str | None = None
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    n_turns: int = 0
    timed_out: bool = False
    hit_max_turns: bool = False


# ----------------- shell executor -----------------

class ShellExecutor:
    """Pluggable shell-command runner. Subclass + override ``run``."""

    def run(self, command: str, cwd: Path, timeout_s: float) -> tuple[str, str, int]:
        """Return (stdout, stderr, exit_code). Must obey ``timeout_s``."""
        raise NotImplementedError


class LocalSandboxShell(ShellExecutor):
    """Run shell commands in a constrained subprocess.

    Safety guards (best-effort, not a security boundary):

    - hard timeout via ``subprocess.run(..., timeout=...)``
    - command runs with cwd inside the agent's sandbox dir
    - blocks a small denylist of obviously-destructive patterns
      (``rm -rf /``, ``:(){:|:&};:``, ``mkfs``, ``shutdown``, ``reboot``)
    - inherits environment minus ``HOME`` and ``PATH`` overrides

    This is **not** a security sandbox — agents could still escape via
    ``cd /`` and read files. But it stops the obvious foot-guns when
    running untrusted-LLM-generated commands locally during research.
    For real adversarial settings, plug in a Docker- or
    firecracker-backed ``ShellExecutor``.
    """

    DENY_PATTERNS = (
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        ":(){:|:&};:",
        "mkfs",
        "shutdown",
        "reboot",
        "dd if=/dev/zero of=/dev/",
        "> /dev/sda",
    )

    def __init__(self, env: dict[str, str] | None = None):
        # Minimal, safe environment. Inherit PATH so common tools work.
        self.env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C",
            "LC_ALL": "C",
        }
        if env:
            self.env.update(env)

    def _is_denied(self, command: str) -> bool:
        cmd_lower = command.strip().lower()
        return any(p in cmd_lower for p in self.DENY_PATTERNS)

    def run(self, command: str, cwd: Path, timeout_s: float) -> tuple[str, str, int]:
        if self._is_denied(command):
            return ("", "denied: command matched safety denylist", 126)
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=str(cwd),
                env=self.env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return (proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired as e:
            so = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", errors="replace")
            se = (e.stderr or "") if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", errors="replace")
            return (so, f"timeout after {timeout_s}s\n{se}", 124)
        except Exception as e:  # pragma: no cover — defensive
            return ("", f"executor error: {type(e).__name__}: {e}", 1)


class MockShell(ShellExecutor):
    """Deterministic in-memory shell for tests.

    ``script`` maps regex-or-prefix patterns to ``(stdout, stderr, code)``
    tuples. Falls back to ``("", "command not in script", 127)``.
    """

    def __init__(self, script: list[tuple[str, tuple[str, str, int]]]):
        self.script = script
        self.calls: list[str] = []

    def run(self, command: str, cwd: Path, timeout_s: float) -> tuple[str, str, int]:
        self.calls.append(command)
        for pattern, response in self.script:
            if pattern in command:
                return response
        return ("", f"command not in mock script: {command!r}", 127)


# ----------------- env -----------------

@dataclass
class AgentEnv:
    """Mutable per-example environment. Owned by AgentHarness during a run."""
    sandbox: Path
    executor: ShellExecutor
    history: list[Turn] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)
    last_stdout: str = ""
    last_stderr: str = ""
    last_exit_code: int = 0
    done: bool = False

    def read_file(self, rel: str) -> str:
        p = self.sandbox / rel
        if not p.exists():
            return ""
        try:
            return p.read_text(errors="replace")
        except Exception:  # pragma: no cover — defensive
            return ""

    def file_exists(self, rel: str) -> bool:
        return (self.sandbox / rel).exists()


# ----------------- policy types -----------------

# A policy maps (goal, history, last_observation) -> (action_str, est_tokens).
# The action is a shell command string. Returning the special string ``"<DONE>"``
# signals the loop to stop early.
AgentPolicy = Callable[[str, list[Turn], str], tuple[str, int]]


# ----------------- harness -----------------

@dataclass
class AgentHarness:
    """A simple agent harness: policy + executor + loop.

    The shape that ``MetaSearch`` would optimize over is the (policy,
    max_observation_chars, max_turns_override) triple. We keep it as
    a dataclass so the same logging/JSON-serialization machinery used
    elsewhere works.
    """
    policy: AgentPolicy
    name: str = "agent_harness"
    max_observation_chars: int = 2000
    truncation_suffix: str = "\n…[truncated]"
    # When >0, overrides the per-example ``max_turns``. Useful for
    # search-budget control across a candidate.
    max_turns_override: int = 0

    def run(
        self,
        example: AgentExample,
        executor: ShellExecutor,
        sandbox: Path,
    ) -> Trajectory:
        traj = Trajectory(task_id=example.task_id)
        env = AgentEnv(sandbox=sandbox, executor=executor)
        try:
            example.setup(sandbox)
        except Exception as e:
            traj.error = f"setup failed: {type(e).__name__}: {e}"
            return traj

        max_turns = (
            self.max_turns_override if self.max_turns_override > 0
            else example.max_turns
        )

        observation = ""
        for i in range(max_turns):
            try:
                action, est_tokens = self.policy(example.goal, env.history, observation)
            except Exception as e:
                traj.error = f"policy failed at turn {i}: {type(e).__name__}: {e}"
                break

            if action.strip() == "<DONE>":
                env.done = True
                # Account for the policy's "thinking" tokens but no shell call.
                t0 = time.perf_counter()
                turn = Turn(
                    turn_idx=i, observation=observation, action=action,
                    stdout="", stderr="", exit_code=0,
                    elapsed_s=time.perf_counter() - t0,
                    tokens=est_tokens,
                )
                traj.turns.append(turn)
                env.history.append(turn)
                break

            t0 = time.perf_counter()
            stdout, stderr, code = executor.run(action, cwd=sandbox, timeout_s=example.timeout_s)
            elapsed = time.perf_counter() - t0

            truncated = False
            if len(stdout) > self.max_observation_chars:
                stdout = stdout[: self.max_observation_chars] + self.truncation_suffix
                truncated = True

            turn = Turn(
                turn_idx=i, observation=observation, action=action,
                stdout=stdout, stderr=stderr, exit_code=code,
                elapsed_s=elapsed, tokens=est_tokens, truncated=truncated,
            )
            traj.turns.append(turn)
            env.history.append(turn)
            env.last_stdout = stdout
            env.last_stderr = stderr
            env.last_exit_code = code

            if code == 124:
                traj.timed_out = True
                # Don't break — let the policy decide what to do next.

            observation = stdout if stdout else stderr

        if len(traj.turns) >= max_turns and not env.done:
            traj.hit_max_turns = True

        try:
            traj.success = bool(example.check_fn(env))
        except Exception as e:
            traj.error = (traj.error or "") + f"; check_fn failed: {type(e).__name__}: {e}"
            traj.success = False

        traj.final_observation = observation
        traj.n_turns = len(traj.turns)
        traj.total_tokens = sum(t.tokens for t in traj.turns)
        traj.total_latency_ms = 1000.0 * sum(t.elapsed_s for t in traj.turns)
        return traj


# ----------------- task & scorer -----------------

class AgentTask:
    """Sequence of AgentExamples + a stable name."""

    def __init__(self, name: str, examples: Sequence[AgentExample]):
        self.name = name
        self.examples = list(examples)

    def screen_subset(self, size: int, seed: int = 0) -> list[AgentExample]:
        import random
        rng = random.Random(seed)
        idx = list(range(len(self.examples)))
        rng.shuffle(idx)
        return [self.examples[i] for i in idx[:size]]


class AgentScorer:
    """Run an AgentHarness over an AgentTask, return a ScoreVector.

    The score's ``accuracy`` is the success rate (mean of per-example
    ``check_fn`` results). ``tokens`` and ``latency_ms`` are means of
    the per-trajectory totals. ``per_class_accuracy`` is empty since
    agent tasks aren't class-stratified.

    Sandboxing: each example gets its own temp dir under
    ``sandbox_root`` (or ``/tmp`` by default). After the run the dir
    is removed unless ``keep_sandboxes=True``.
    """

    def __init__(
        self,
        task: AgentTask,
        executor_factory: Callable[[], ShellExecutor] | None = None,
        sandbox_root: str | Path | None = None,
        keep_sandboxes: bool = False,
    ):
        self.task = task
        self.executor_factory = executor_factory or LocalSandboxShell
        self.sandbox_root = Path(sandbox_root) if sandbox_root else None
        self.keep_sandboxes = keep_sandboxes
        self.last_trajectories: list[Trajectory] = []

    def score(
        self,
        harness: AgentHarness,
        examples: Sequence[AgentExample] | None = None,
        *,
        n_repeats: int = 1,
        max_workers: int = 1,
    ) -> ScoreVector:
        ex_list = list(examples) if examples is not None else list(self.task.examples)
        if not ex_list:
            return ScoreVector(0.0, 0.0, 0.0, 0, n_repeats=max(1, n_repeats))
        if n_repeats < 1 or max_workers < 1:
            raise ValueError("n_repeats and max_workers must be >= 1")

        def run_one(example: AgentExample) -> Trajectory:
            executor = self.executor_factory()
            if self.sandbox_root is not None:
                self.sandbox_root.mkdir(parents=True, exist_ok=True)
                sandbox = Path(tempfile.mkdtemp(prefix=f"agent_{example.task_id}_", dir=str(self.sandbox_root)))
            else:
                sandbox = Path(tempfile.mkdtemp(prefix=f"agent_{example.task_id}_"))
            try:
                return harness.run(example, executor=executor, sandbox=sandbox)
            finally:
                if not self.keep_sandboxes:
                    shutil.rmtree(sandbox, ignore_errors=True)

        def sweep() -> list[Trajectory]:
            if max_workers == 1:
                return [run_one(ex) for ex in ex_list]
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(run_one, ex_list))

        accs: list[float] = []
        token_sums: list[float] = []
        latency_sums: list[float] = []
        last_trajs: list[Trajectory] = []
        for _ in range(n_repeats):
            trajs = sweep()
            last_trajs = trajs
            n = len(trajs)
            successes = sum(1 for t in trajs if t.success)
            accs.append(successes / n)
            token_sums.append(sum(t.total_tokens for t in trajs) / n)
            latency_sums.append(sum(t.total_latency_ms for t in trajs) / n)

        self.last_trajectories = last_trajs
        if n_repeats == 1:
            return ScoreVector(
                accuracy=accs[0],
                tokens=token_sums[0],
                latency_ms=latency_sums[0],
                n_evaluated=len(ex_list),
            )
        # Median acc, mean costs, spread for reproducibility tracking.
        s = sorted(accs)
        median_acc = s[len(s) // 2] if len(s) % 2 == 1 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
        return ScoreVector(
            accuracy=median_acc,
            tokens=sum(token_sums) / n_repeats,
            latency_ms=sum(latency_sums) / n_repeats,
            n_evaluated=len(ex_list),
            n_repeats=n_repeats,
            accuracy_spread=max(accs) - min(accs),
        )


# ----------------- helpers -----------------

def trajectory_to_dict(t: Trajectory) -> dict:
    """JSON-serializable view of a Trajectory (for logging)."""
    return {
        "task_id": t.task_id,
        "success": t.success,
        "n_turns": t.n_turns,
        "total_tokens": t.total_tokens,
        "total_latency_ms": t.total_latency_ms,
        "timed_out": t.timed_out,
        "hit_max_turns": t.hit_max_turns,
        "error": t.error,
        "final_observation": t.final_observation,
        "turns": [
            {
                "turn_idx": tn.turn_idx,
                "observation": tn.observation,
                "action": tn.action,
                "stdout": tn.stdout,
                "stderr": tn.stderr,
                "exit_code": tn.exit_code,
                "elapsed_s": tn.elapsed_s,
                "tokens": tn.tokens,
                "truncated": tn.truncated,
            }
            for tn in t.turns
        ],
    }


def write_trajectories(trajectories: Iterable[Trajectory], path: str | Path) -> None:
    """Append-write a JSONL file of trajectories. Useful for replay/debug."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for tr in trajectories:
            f.write(json.dumps(trajectory_to_dict(tr)) + "\n")

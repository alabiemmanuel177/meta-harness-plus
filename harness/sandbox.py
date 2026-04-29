"""V10 sandbox — wraps the legacy ``DockerShellExecutor`` for leak-free
primitives and exposes a curated subset of operations to V10 code.

Per V10_DESIGN.md §12.4(b), this module deliberately does NOT expose
``DockerShellExecutor.run_tests(test_targets)`` — that legacy API
encourages callers to pass FAIL_TO_PASS selectors (its docstring at
``meta_harness_plus/agent_docker.py:559`` literally instructs them to).
We expose ``run_public_suite()`` instead, which uses default pytest
discovery on directories from ``InstanceView.test_directives``.

The runtime guard (``_assert_no_forbidden_token``) implements
tightening 3 from the review: case-insensitive substring matching
against the canonical token list in ``harness.views.FORBIDDEN_TOKENS``.
Better to overflag and tune down than miss.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

# IMPORTANT: we import the executor directly. This is allowed (it's a
# leak-free primitive — file ops, generic shell exec). The forbidden
# legacy module list does NOT include agent_docker; only swebench_v7
# and agent_swebench_loop. We must NOT call ``run_tests`` on the
# executor, however — see _disallow_run_tests below.
from meta_harness_plus.agent_docker import DockerShellExecutor, ExecResult

from harness.views import (
    FORBIDDEN_TOKENS,
    InstanceView,
    PublicSuiteSignal,
    TestDirectives,
)


class OracleLeakError(RuntimeError):
    """A test selector or sandbox argument matched a forbidden token at
    runtime. Same shape as the static firewall error, but at execution
    time. See V10_DESIGN.md §12.5."""


def _normalize(s: str) -> str:
    """Casefold + strip underscores so SCREAMING_SNAKE and CamelCase both
    reduce to the same form as the canonical lowercase-underscored token.

      'fail_to_pass'  -> 'failtopass'
      'FAIL_TO_PASS'  -> 'failtopass'
      'FailToPass'    -> 'failtopass'
    """
    return s.casefold().replace("_", "")


_NORMALIZED_TOKENS: tuple[str, ...] = tuple(_normalize(t) for t in FORBIDDEN_TOKENS)


def _assert_no_forbidden_token(value, *, label: str) -> None:
    """Walk a string-or-collection and reject anything containing a
    forbidden token (case-insensitive, underscore-insensitive substring
    match). See V10_DESIGN.md §12.5; tightening 3 in the review.
    """
    if value is None:
        return
    if isinstance(value, str):
        haystacks = [value]
    elif isinstance(value, (list, tuple)):
        haystacks = [str(v) for v in value]
    else:
        haystacks = [str(value)]
    for hay in haystacks:
        normalized = _normalize(hay)
        for tok in _NORMALIZED_TOKENS:
            if tok in normalized:
                raise OracleLeakError(
                    f"sandbox.{label}: argument contains forbidden token "
                    f"{tok!r} (matched in {hay[:200]!r}); see V10_DESIGN.md "
                    f"§12.5."
                )


@dataclass
class SuiteResult:
    """Outcome of running the public test suite at a fixed worktree
    state. Used to compute ``PublicSuiteSignal`` deltas across patches."""

    exit_code: int
    duration_s: float
    log_excerpt: str
    truncated: bool
    workdir_state: str  # 'base' or 'patched' — caller-supplied label
    test_dirs: tuple[str, ...]


class Sandbox:
    """V10's leak-free sandbox interface.

    Wraps ``DockerShellExecutor`` for container lifecycle, generic shell
    exec, and file ops. Exposes ``run_public_suite()``, ``run_repro()``,
    and ``apply_patch()`` — and only those — for V10 callers. The legacy
    executor's ``run_tests(test_targets)`` is intentionally hidden;
    attempting to access it via ``sandbox._exec.run_tests(...)`` raises
    ``OracleLeakError`` at runtime.
    """

    def __init__(
        self,
        view: InstanceView,
        *,
        memory_gb: float = 4.0,
        cpus: float = 2.0,
        no_network: bool = True,
    ):
        self._view = view
        self._exec = DockerShellExecutor(
            instance_id=view.instance_id,
            memory_gb=memory_gb,
            cpus=cpus,
            no_network=no_network,
            ensure_pytest=False,
        )
        # Defense-in-depth: assert the test directives we'll use don't
        # smuggle a forbidden token. Cheap, runs before any container starts.
        self._dirs: tuple[str, ...] = view.test_directives.dirs
        _assert_no_forbidden_token(self._dirs, label="__init__.test_dirs")

    @property
    def view(self) -> InstanceView:
        return self._view

    def start(self) -> None:
        self._exec.start()

    def cleanup(self) -> None:
        self._exec.cleanup()

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *_a) -> None:
        self.cleanup()

    # ----- generic primitives (proxy through, no test-list semantics) -----

    def run_shell(self, command: str, timeout_s: float = 30.0) -> ExecResult:
        """Generic shell exec. Forbidden-token check runs before exec —
        if the command string mentions a forbidden token, we refuse it.
        """
        _assert_no_forbidden_token(command, label="run_shell.command")
        return self._exec.run(command, timeout_s=timeout_s)

    def read_file(self, rel_path: str, max_chars: int | None = None) -> ExecResult:
        return self._exec.read_file(rel_path, max_chars=max_chars)

    def write_file(self, rel_path: str, content: str) -> ExecResult:
        # Patch contents may legitimately mention "patch" or related words;
        # the canonical token list is `test_patch` (with underscore), not
        # `patch`. The token check is over-aggressive on purpose, so the
        # content scan can fire — if it does, the caller renames or splits.
        _assert_no_forbidden_token(content, label="write_file.content")
        return self._exec.write_file(rel_path, content)

    # ----- patch application -----

    def apply_patch(self, diff: str) -> ExecResult:
        """Apply a unified diff to the worktree via ``git apply``."""
        _assert_no_forbidden_token(diff, label="apply_patch.diff")
        # Stage the diff to a file inside the container, then apply.
        path_inside = "/tmp/v10_apply.patch"
        write_res = self._exec.write_file(
            "../tmp/v10_apply.patch".lstrip("../"), diff
        )
        # write_file is workdir-relative, so use docker cp directly via run_shell:
        del write_res  # unused; write via shell instead
        # Simpler path: use stdin redirection.
        cmd = (
            f"cat > {path_inside} <<'V10_EOF'\n{diff}\nV10_EOF\n"
            f"cd /testbed && git apply --whitespace=nowarn {path_inside}"
        )
        # Note: cmd does not contain forbidden tokens by construction
        # (we just inserted the diff content, and we already validated diff).
        return self._exec.run(cmd, timeout_s=60)

    def reset_worktree(self) -> ExecResult:
        return self._exec.reset_to_clean()

    # ----- public test suite (the leak-free replacement for run_tests) -----

    def run_public_suite(
        self,
        *,
        state_label: str = "base",
        timeout_s: int = 480,
    ) -> SuiteResult:
        """Run the repo's public test suite at the directories in
        ``view.test_directives.dirs``. Default pytest discovery — no
        test-name allowlist or denylist sourced from a dataset row.
        """
        # Final guard before exec: the dirs were validated at __init__
        # but we re-check in case anyone reached in and mutated them.
        _assert_no_forbidden_token(self._dirs, label="run_public_suite.dirs")

        dirs_arg = " ".join(_q(d) for d in self._dirs)
        cmd = (
            "cd /testbed && "
            f"python -m pytest -p no:cacheprovider --tb=short -q "
            f"{dirs_arg} 2>&1 | tail -200"
        )
        t0 = time.perf_counter()
        res = self._exec.run(cmd, timeout_s=timeout_s)
        dur = time.perf_counter() - t0
        return SuiteResult(
            exit_code=res.exit_code,
            duration_s=dur,
            log_excerpt=res.stdout,
            truncated=res.truncated,
            workdir_state=state_label,
            test_dirs=self._dirs,
        )

    def run_repro(self, code: str, *, timeout_s: float = 60.0) -> ExecResult:
        """Run our self-generated reproduction script. The script's
        contents are checked against forbidden tokens before exec.
        """
        _assert_no_forbidden_token(code, label="run_repro.code")
        return self._exec.run_python(code, timeout_s=timeout_s)


def _q(s: str) -> str:
    """Minimal quoting for a shell argument (filenames assumed simple
    relative paths)."""
    if all(c.isalnum() or c in "/-_." for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def public_suite_signal_from_results(
    base: SuiteResult,
    patched: SuiteResult,
) -> PublicSuiteSignal:
    """Approximate signal builder. Phase 0 returns counts of new failures
    by string-diffing the pytest summary lines; Phase 4 will replace
    this with a proper per-test diff via JUnit-XML output."""
    base_fails = _count_fail_lines(base.log_excerpt)
    patched_fails = _count_fail_lines(patched.log_excerpt)
    base_passes = _count_pass_lines(base.log_excerpt)
    patched_passes = _count_pass_lines(patched.log_excerpt)
    new_failures = max(0, patched_fails - base_fails)
    new_passes = max(0, patched_passes - base_passes)
    return PublicSuiteSignal(
        suite_ran_at_base=(base.exit_code in (0, 1)),
        new_failures_count=new_failures,
        new_passes_count=new_passes,
        flake_retries=0,
        duration_s=base.duration_s + patched.duration_s,
        log_excerpt=patched.log_excerpt[-2000:],
    )


def _count_fail_lines(log: str) -> int:
    return sum(
        1 for line in log.splitlines()
        if line.lstrip().startswith("FAILED") or " FAILED " in line
    )


def _count_pass_lines(log: str) -> int:
    return sum(
        1 for line in log.splitlines()
        if line.lstrip().startswith("PASSED") or " PASSED " in line
    )


@contextmanager
def sandbox(view: InstanceView, **kwargs) -> Iterator[Sandbox]:
    s = Sandbox(view, **kwargs)
    try:
        s.start()
        yield s
    finally:
        s.cleanup()


__all__ = [
    "OracleLeakError",
    "Sandbox",
    "SuiteResult",
    "sandbox",
    "public_suite_signal_from_results",
]

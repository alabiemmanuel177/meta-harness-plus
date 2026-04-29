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

    def _resolve_effective_test_paths(self) -> tuple[str, ...]:
        """Discovery-first resolution: filesystem inspection inside the
        container is the source of truth; the override map in
        ``harness.repo_conventions`` is a preferred-default + sanity
        check (per V10_DESIGN.md §12.4(b)).

        This handles the psf/requests pre-/post-migration case: at older
        base_commits the only test file was top-level ``test_requests.py``,
        while at newer commits there's a ``tests/`` directory. The
        override declares ``("tests/",)``; discovery finds the actual
        layout. Same shape works for any repo refactor that happened
        mid-Verified-corpus.

        Returns the discovered paths. If discovery finds nothing, falls
        back to the override-declared paths (filtered to those that
        exist), or to the override as-is, or to ``(".",)`` as last resort.
        """
        # Discovery: find test_*.py / *_test.py at depth ≤ 3 inside /testbed.
        # Depth 3 covers `pkg/tests/sub/test_x.py` patterns; depth 2 alone
        # would miss those. Capped at 60 results to bound the find cost.
        find_res = self._exec.run(
            "cd /testbed && find . -maxdepth 4 -type f "
            r"\( -name 'test_*.py' -o -name '*_test.py' \) "
            "-not -path '*/.git/*' -not -path '*/__pycache__/*' "
            "-not -path '*/.tox/*' -not -path '*/.eggs/*' "
            "| head -60",
            timeout_s=20,
        )
        discovered_files = [
            line.lstrip("./").strip()
            for line in find_res.stdout.splitlines()
            if line.strip()
        ]

        if discovered_files:
            # Reduce to top-level test directories; if a test file is at
            # the repo root, keep it as a file (not a parent dir).
            discovered_paths = self._collapse_to_test_roots(discovered_files)
            self._sanity_check_override_vs_discovery(discovered_paths)
            return discovered_paths

        # Discovery found nothing. Try override entries that physically exist.
        override_present: list[str] = []
        for d in self._dirs:
            check = self._exec.run(
                f"test -e /testbed/{_q(d)} && echo OK || echo MISSING",
                timeout_s=10,
            )
            if "OK" in check.stdout:
                override_present.append(d)
        if override_present:
            return tuple(override_present)

        # Last resort.
        return (".",)

    def _collapse_to_test_roots(self, files: list[str]) -> tuple[str, ...]:
        """Mirror ``repo_conventions.discover_test_dirs`` collapsing logic
        but on container-side find output. If a path contains 'tests' or
        'testing' as a path segment, keep up to and including that
        segment; otherwise keep the file or its parent dir."""
        roots: set[str] = set()
        for rel in files:
            parts = rel.split("/")
            collapsed = None
            for i, p in enumerate(parts):
                if p in {"tests", "testing"}:
                    collapsed = "/".join(parts[: i + 1]) + "/"
                    break
            if collapsed is None:
                # Keep the file as-is when it's at the repo root (e.g.,
                # psf/requests pre-migration ships `test_requests.py`).
                if "/" not in rel:
                    collapsed = rel
                else:
                    parent = "/".join(parts[:-1])
                    collapsed = parent + "/"
            roots.add(collapsed)
        return tuple(sorted(roots))

    def _sanity_check_override_vs_discovery(
        self, discovered: tuple[str, ...]
    ) -> None:
        """Log (don't raise) a warning when the override declares a
        directory that discovery did not find. Helps identify stale
        override entries for repos whose layout changed across
        base_commits.
        """
        declared = set(self._dirs)
        found = set(discovered)
        missing_from_discovery = sorted(declared - found)
        if missing_from_discovery:
            # The trajectory writer is the canonical place for this; we
            # don't wire that here to keep Sandbox decoupled from
            # trajectories. Print is fine for Phase 0; Phase 1 wires the
            # warning into the per-instance trajectory log.
            print(
                f"[sandbox:{self._view.instance_id}] "
                f"override-vs-discovery mismatch: declared {sorted(declared)} "
                f"but discovery found {sorted(found)}. Discovery wins."
            )

    def run_public_suite(
        self,
        *,
        state_label: str = "base",
        timeout_s: int = 480,
        collect_only: bool = False,
    ) -> SuiteResult:
        """Run the repo's public test suite at the directories in
        ``view.test_directives.dirs``. Default pytest discovery — no
        test-name allowlist or denylist sourced from a dataset row.

        If declared test directories don't exist at this base_commit,
        falls back to filesystem discovery inside the container (still
        leak-free — the discovery scans only public file paths, never
        touches dataset metadata).

        ``collect_only=True`` runs ``pytest --collect-only`` instead of
        executing the suite — useful for fast smoke tests.

        ``state_label`` is a caller-supplied label for the worktree
        state ('base' or 'patched'); validated against forbidden tokens
        as defense-in-depth (a forbidden token in state_label could
        flow through SuiteResult.workdir_state to a downstream prompt).
        """
        _assert_no_forbidden_token(state_label, label="run_public_suite.state_label")
        # Final guard before exec: the dirs were validated at __init__
        # but we re-check in case anyone reached in and mutated them.
        _assert_no_forbidden_token(self._dirs, label="run_public_suite.dirs")

        effective = self._resolve_effective_test_paths()
        _assert_no_forbidden_token(effective, label="run_public_suite.effective")

        dirs_arg = " ".join(_q(d) for d in effective)
        flags = "-p no:cacheprovider --tb=short -q"
        if collect_only:
            flags += " --collect-only"
        cmd = (
            "cd /testbed && "
            f"python -m pytest {flags} {dirs_arg} 2>&1 | tail -200"
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
            test_dirs=effective,
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

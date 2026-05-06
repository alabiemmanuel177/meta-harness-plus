"""Phase 4 — public test suite baseline + patched diff.

Per docs/V10_DESIGN_PHASE4.md §3.2 step 4.

Two-pass flow:

  1. ``run_baseline(view, sandbox)`` — pytest at base_commit, before
     any patch is applied. Records the set of passing test ids, the
     set of failing test ids, and timing. Cached per
     (instance_id, base_commit) under ``runs/<run-name>/validation_cache/``.
  2. ``run_patched(view, sandbox)`` — pytest after a candidate's
     patch is applied. Records the same shapes.
  3. ``diff_against_baseline(baseline, patched)`` — produces a
     ``PublicSuiteSignal`` with ``new_failures_count`` (regressions)
     and ``new_passes_count`` (rare gains).

The runner uses ``Sandbox.run_public_suite`` (the firewall-audited
no-test-list-arg path). NEVER passes test selectors; default pytest
discovery.

Flake-retry budget: when a per-test failure differs between baseline
and patched runs, we re-run the suite up to 2 times. If the
disagreement persists, we trust the second run.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import time
from dataclasses import dataclass, asdict, field

from harness.views import PublicSuiteSignal


log = logging.getLogger(__name__)


DEFAULT_SUITE_TIMEOUT_S: int = 480
DEFAULT_FLAKE_RETRIES: int = 2


# ---------------------------------------------------------------------------
# Pytest output parsing — extract test ids by outcome
# ---------------------------------------------------------------------------


_TEST_LINE_RE = re.compile(
    r"^(?P<path>[\w/_.\-]+\.py)::(?P<test>[\w\[\]/:.\-]+)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)",
    re.MULTILINE,
)
_FAILED_LINE_RE = re.compile(
    r"^(?:FAILED|ERROR)\s+(?P<id>[\w/_.\-]+\.py::[\w\[\]/:.\-]+)",
    re.MULTILINE,
)
_PASSED_LINE_RE = re.compile(
    r"^(?:PASSED)\s+(?P<id>[\w/_.\-]+\.py::[\w\[\]/:.\-]+)",
    re.MULTILINE,
)


def _parse_pytest_output(output: str) -> tuple[set[str], set[str]]:
    """Parse pytest's stdout into (passing_ids, failing_ids).

    The default pytest output (`-q` / `-p no:cacheprovider --tb=short`)
    doesn't list every test by id — it only emits per-test status
    when failures occur. For the GREEN cases we don't get test ids
    individually; we rely on the FAILED/ERROR markers to identify
    regressions and assume everything else passed.

    Strategy:
      - Collect FAILED/ERROR ids explicitly.
      - The "X passed, Y failed" summary line tells us total counts;
        we DON'T enumerate the passing ids (cheap to skip).
      - Return (passing=empty-set sentinel, failing=parsed-ids).

    The diff between baseline and patched is computed on FAILING
    sets only, which is exactly what `new_failures_count` requires.
    `new_passes_count` is approximated as "tests in baseline.failing
    that are not in patched.failing" — which is correct given we
    don't enumerate individual passing tests.
    """
    failing: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            # `FAILED tests/test_foo.py::test_bar - AssertionError: ...`
            parts = line.split(None, 2)
            if len(parts) >= 2 and "::" in parts[1]:
                failing.add(parts[1])
    passing: set[str] = set()  # not enumerated; sentinel
    return passing, failing


_SUMMARY_RE = re.compile(
    r"=+ (?:short test summary info =+|"
    r"(?P<failed>\d+) failed(?:, (?P<errors>\d+) error)?(?:, (?P<passed>\d+) passed)?(?:, (?P<skipped>\d+) skipped)?(?:, (?P<warnings>\d+) warning)?[^=]*?=+|"
    r"(?P<passed_only>\d+) passed[^=]*?=+|"
    r"no tests ran[^=]*?=+)"
)


def _parse_summary_counts(output: str) -> dict:
    """Extract pytest summary counts (failed, passed, errors, skipped).
    Used as a sanity-check sidecar to the ids-set parsing."""
    failed = 0
    passed = 0
    errors = 0
    skipped = 0
    # Last "X passed" / "X failed" wins (pytest can emit progress lines).
    for m in re.finditer(r"(\d+)\s+failed", output):
        failed = int(m.group(1))
    for m in re.finditer(r"(\d+)\s+passed", output):
        passed = int(m.group(1))
    for m in re.finditer(r"(\d+)\s+error", output):
        errors = int(m.group(1))
    for m in re.finditer(r"(\d+)\s+skipped", output):
        skipped = int(m.group(1))
    return {"failed": failed, "passed": passed, "errors": errors, "skipped": skipped}


# ---------------------------------------------------------------------------
# Snapshots + cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineSnapshot:
    """Cached public-suite result at base_commit (before any patch).

    Persisted to disk so multiple candidates per instance share one
    baseline run. Cache key is (instance_id, base_commit).
    """

    instance_id: str
    base_commit: str
    failing_ids: tuple[str, ...]
    passing_count: int
    duration_s: float
    log_excerpt: str = ""

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "base_commit": self.base_commit,
            "failing_ids": list(self.failing_ids),
            "passing_count": self.passing_count,
            "duration_s": self.duration_s,
            "log_excerpt": self.log_excerpt[:2000],  # truncate for cache size
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineSnapshot":
        return cls(
            instance_id=d["instance_id"],
            base_commit=d["base_commit"],
            failing_ids=tuple(d.get("failing_ids", [])),
            passing_count=int(d.get("passing_count", 0)),
            duration_s=float(d.get("duration_s", 0.0)),
            log_excerpt=d.get("log_excerpt", ""),
        )


@dataclass(frozen=True)
class PublicSuiteRunResult:
    """One public-suite run's parsed output (either baseline or patched)."""

    failing_ids: set[str] = field(default_factory=set)
    passing_count: int = 0
    duration_s: float = 0.0
    log_excerpt: str = ""
    suite_ran: bool = True
    timed_out: bool = False


def _cache_path(run_dir: pathlib.Path | str, instance_id: str, base_commit: str) -> pathlib.Path:
    return (
        pathlib.Path(run_dir) / "validation_cache" / instance_id / f"baseline_{base_commit[:12]}.json"
    )


def load_baseline_snapshot(run_dir: pathlib.Path | str, instance_id: str, base_commit: str) -> BaselineSnapshot | None:
    p = _cache_path(run_dir, instance_id, base_commit)
    if not p.exists():
        return None
    try:
        return BaselineSnapshot.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_baseline_snapshot(
    run_dir: pathlib.Path | str, snapshot: BaselineSnapshot
) -> pathlib.Path:
    p = _cache_path(run_dir, snapshot.instance_id, snapshot.base_commit)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshot.to_dict(), indent=2))
    return p


# ---------------------------------------------------------------------------
# Suite runners
# ---------------------------------------------------------------------------


def _run_one(sandbox, *, state_label: str, timeout_s: int) -> PublicSuiteRunResult:
    """One pytest invocation via Sandbox.run_public_suite. Default
    discovery — no test selectors. Returns a PublicSuiteRunResult.
    """
    t_start = time.perf_counter()
    try:
        res = sandbox.run_public_suite(state_label=state_label, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t_start
        log.warning(
            "[validation.public_suite] run_public_suite raised %s: %s",
            type(exc).__name__, exc,
        )
        return PublicSuiteRunResult(
            failing_ids=set(),
            passing_count=0,
            duration_s=elapsed,
            log_excerpt=f"{type(exc).__name__}: {exc}",
            suite_ran=False,
        )
    elapsed = time.perf_counter() - t_start

    output = res.log_excerpt or ""
    _, failing = _parse_pytest_output(output)
    counts = _parse_summary_counts(output)
    timed_out = "timed out" in output.lower() or res.exit_code in (124, 137)
    log_excerpt = output[-2000:] if len(output) > 2000 else output
    return PublicSuiteRunResult(
        failing_ids=failing,
        passing_count=counts["passed"],
        duration_s=elapsed,
        log_excerpt=log_excerpt,
        suite_ran=res.exit_code != 1 or len(failing) > 0 or counts["passed"] > 0,
        timed_out=timed_out,
    )


def run_baseline(
    sandbox,
    *,
    timeout_s: int = DEFAULT_SUITE_TIMEOUT_S,
) -> PublicSuiteRunResult:
    """Run the public suite at base_commit (no patch applied).
    Returns PublicSuiteRunResult; the orchestrator wraps it into
    a BaselineSnapshot for caching."""
    return _run_one(sandbox, state_label="base", timeout_s=timeout_s)


def run_patched(
    sandbox,
    *,
    timeout_s: int = DEFAULT_SUITE_TIMEOUT_S,
) -> PublicSuiteRunResult:
    """Run the public suite at base_commit + applied candidate.
    Caller must apply_clean(diff) first. Returns PublicSuiteRunResult."""
    return _run_one(sandbox, state_label="patched", timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# Diff against baseline
# ---------------------------------------------------------------------------


def diff_against_baseline(
    baseline: BaselineSnapshot,
    patched: PublicSuiteRunResult,
    *,
    flake_retries: int = DEFAULT_FLAKE_RETRIES,
) -> PublicSuiteSignal:
    """Compute the new_failures / new_passes counts.

    new_failures_count: tests passing at base, failing at patched.
    new_passes_count: tests failing at base, passing at patched.

    Per V10_DESIGN.md §3.5: 2 flake retries on per-test failures
    is the contract; this function doesn't perform retries itself
    (the caller does, by re-running the patched suite). The
    ``flake_retries`` argument is recorded on the signal for audit.
    """
    base_failing = set(baseline.failing_ids)
    patched_failing = patched.failing_ids

    # Tests that pass at base but fail at patched — TRUE regressions.
    new_failures = patched_failing - base_failing
    # Tests that fail at base but pass at patched — rare gains.
    new_passes = base_failing - patched_failing

    return PublicSuiteSignal(
        suite_ran_at_base=patched.suite_ran,  # patched-side ran; baseline ran by definition
        new_failures_count=len(new_failures),
        new_passes_count=len(new_passes),
        flake_retries=flake_retries,
        duration_s=patched.duration_s + baseline.duration_s,
        log_excerpt=patched.log_excerpt[-1500:],
    )


__all__ = [
    "BaselineSnapshot",
    "DEFAULT_FLAKE_RETRIES",
    "DEFAULT_SUITE_TIMEOUT_S",
    "PublicSuiteRunResult",
    "diff_against_baseline",
    "load_baseline_snapshot",
    "run_baseline",
    "run_patched",
    "save_baseline_snapshot",
]

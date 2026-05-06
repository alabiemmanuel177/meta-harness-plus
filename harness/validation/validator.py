"""Phase 4 — per-candidate validator orchestrator.

Per docs/V10_DESIGN_PHASE4.md §3.2:

  validate_candidate(view, sandbox, candidate, ...) -> ValidationResult

  1. apply_clean. If False -> emit a CandidateView with apply_failed
     marker; skip steps 2-5.
  2. compute_diff_stats (incl AST-normalized hash).
  3. forward repro_signal from caller.
  4. public-suite delta (with cache).
  5. static (ruff + mypy).
  6. assemble CandidateView.

The orchestrator handles worktree resets between candidates (the
caller may pass multiple candidates per instance).
"""

from __future__ import annotations

import logging
import pathlib
import time
from dataclasses import dataclass

from harness.patch_gen.views import PatchCandidate
from harness.validation.apply import ApplyResult, apply_clean
from harness.validation.diff_stats import compute_diff_stats
from harness.validation.public_suite import (
    BaselineSnapshot,
    DEFAULT_FLAKE_RETRIES,
    DEFAULT_SUITE_TIMEOUT_S,
    diff_against_baseline,
    load_baseline_snapshot,
    run_baseline,
    run_patched,
    save_baseline_snapshot,
)
from harness.validation.static import (
    DEFAULT_STATIC_TIMEOUT_S,
    run_static,
)
from harness.views import (
    CandidateView,
    DiffStats,
    InstanceView,
    PublicSuiteSignal,
    ReproSignal,
    StaticSignal,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one candidate. Wraps CandidateView with
    audit-only sidecars (timings, apply detail, baseline cache hit)."""

    candidate_view: CandidateView
    apply_result: ApplyResult
    baseline_cache_hit: bool
    duration_s: float


def _empty_public_suite_signal() -> PublicSuiteSignal:
    return PublicSuiteSignal(
        suite_ran_at_base=False,
        new_failures_count=0,
        new_passes_count=0,
        flake_retries=0,
        duration_s=0.0,
        log_excerpt="",
    )


def _empty_static_signal() -> StaticSignal:
    return StaticSignal(ruff_clean=None, mypy_clean=None, duration_s=0.0)


def validate_candidate(
    *,
    view: InstanceView,
    sandbox,
    candidate: PatchCandidate,
    repro_signal: ReproSignal | None = None,
    run_dir: str | pathlib.Path | None = None,
    suite_timeout_s: int = DEFAULT_SUITE_TIMEOUT_S,
    static_timeout_s: int = DEFAULT_STATIC_TIMEOUT_S,
    flake_retries: int = DEFAULT_FLAKE_RETRIES,
    skip_static: bool = False,
) -> ValidationResult:
    """Validate one candidate against the public suite + static.

    Args:
      view: the firewall-clean InstanceView.
      sandbox: a started Sandbox at base_commit. Worktree must be
        clean on entry; the orchestrator resets it on exit.
      candidate: the PatchCandidate from Phase 3.
      repro_signal: forwarded from Phase 2 (None when no repro was
        produced for this instance).
      run_dir: where to read/write the baseline cache. None disables
        caching.
      skip_static: if True, return StaticSignal(None, None) without
        running ruff/mypy. Useful for quick smoke runs.

    Returns ``ValidationResult``. Even on apply failure, a
    ``CandidateView`` is emitted with the `apply_failed` marker in
    `notes`.
    """
    t_start = time.perf_counter()

    # 1) diff stats — pure function on the diff text alone.
    try:
        diff_stats = compute_diff_stats(candidate.diff)
    except Exception as exc:  # noqa: BLE001
        log.warning("[validator] compute_diff_stats failed for %s: %s",
                    candidate.candidate_id, exc)
        diff_stats = DiffStats(
            files_touched=tuple(),
            additions=0,
            deletions=0,
            ast_normalized_hash="parse_error",
        )

    # 2) Reset worktree before apply (defense — caller is supposed to
    # leave it clean, but this is a nail-down).
    try:
        sandbox.reset_worktree()
    except Exception as exc:  # noqa: BLE001
        log.info("[validator] reset_worktree raised %s; continuing",
                 type(exc).__name__)

    # 3) apply_clean
    apply_res = apply_clean(sandbox, candidate.diff)

    if not apply_res.applied:
        # Skip the test-running steps; emit a "broken" CandidateView.
        cv = CandidateView(
            candidate_id=candidate.candidate_id,
            diff=candidate.diff,
            diff_stats=diff_stats,
            repro_signal=repro_signal,
            public_suite_signal=_empty_public_suite_signal(),
            static_signal=_empty_static_signal(),
            cluster_id="",
            notes=f"apply_failed: {apply_res.stderr_excerpt[:300]}",
        )
        return ValidationResult(
            candidate_view=cv,
            apply_result=apply_res,
            baseline_cache_hit=False,
            duration_s=time.perf_counter() - t_start,
        )

    # 4) baseline (cached)
    baseline: BaselineSnapshot | None = None
    cache_hit = False
    if run_dir is not None:
        baseline = load_baseline_snapshot(run_dir, view.instance_id, view.base_commit)
    if baseline is not None:
        cache_hit = True
    else:
        # Reset worktree to ensure baseline runs without the patch.
        try:
            sandbox.reset_worktree()
        except Exception:  # noqa: BLE001
            pass
        baseline_run = run_baseline(sandbox, timeout_s=suite_timeout_s)
        baseline = BaselineSnapshot(
            instance_id=view.instance_id,
            base_commit=view.base_commit,
            failing_ids=tuple(sorted(baseline_run.failing_ids)),
            passing_count=baseline_run.passing_count,
            duration_s=baseline_run.duration_s,
            log_excerpt=baseline_run.log_excerpt,
        )
        if run_dir is not None:
            save_baseline_snapshot(run_dir, baseline)

    # 5) re-apply the patch (we just reset for baseline if no cache)
    if not cache_hit:
        sandbox.reset_worktree()
        re_apply = apply_clean(sandbox, candidate.diff)
        if not re_apply.applied:
            log.warning(
                "[validator] %s re-apply after baseline failed: %s",
                candidate.candidate_id, re_apply.stderr_excerpt[:200],
            )
            cv = CandidateView(
                candidate_id=candidate.candidate_id,
                diff=candidate.diff,
                diff_stats=diff_stats,
                repro_signal=repro_signal,
                public_suite_signal=_empty_public_suite_signal(),
                static_signal=_empty_static_signal(),
                cluster_id="",
                notes=f"reapply_failed: {re_apply.stderr_excerpt[:300]}",
            )
            return ValidationResult(
                candidate_view=cv,
                apply_result=re_apply,
                baseline_cache_hit=False,
                duration_s=time.perf_counter() - t_start,
            )

    # 6) patched suite
    patched_run = run_patched(sandbox, timeout_s=suite_timeout_s)
    public_signal = diff_against_baseline(
        baseline, patched_run, flake_retries=flake_retries,
    )

    # 7) static
    if skip_static:
        static_signal = _empty_static_signal()
    else:
        static_signal = run_static(sandbox, timeout_s=static_timeout_s)

    # 8) assemble CandidateView
    cv = CandidateView(
        candidate_id=candidate.candidate_id,
        diff=candidate.diff,
        diff_stats=diff_stats,
        repro_signal=repro_signal,
        public_suite_signal=public_signal,
        static_signal=static_signal,
        cluster_id=diff_stats.ast_normalized_hash[:12],  # shorter id for grouping
        notes="",
    )

    elapsed = time.perf_counter() - t_start
    return ValidationResult(
        candidate_view=cv,
        apply_result=apply_res,
        baseline_cache_hit=cache_hit,
        duration_s=elapsed,
    )


__all__ = [
    "ValidationResult",
    "validate_candidate",
]

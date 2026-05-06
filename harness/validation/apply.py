"""Phase 4 — git apply --check + git apply on the candidate diff.

Per docs/V10_DESIGN_PHASE4.md §3.2 step 1: the apply gate. If the
patch fails to apply, the candidate is the worst possible — Phase 5
will treat the resulting CandidateView as broken without further
checks.

This module is leak-free by construction: it only touches
``Sandbox.apply_patch`` and ``Sandbox.reset_worktree`` (both already
firewall-audited in V10_DESIGN.md §12).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of one ``git apply`` invocation against the worktree."""

    applied: bool                    # True iff the patch landed cleanly
    stderr_excerpt: str = ""         # for audit only; truncated
    duration_s: float = 0.0


def apply_clean(sandbox, diff: str, *, timeout_s: float = 60.0) -> ApplyResult:
    """Apply ``diff`` to the sandbox's worktree via ``git apply``.

    Returns ``ApplyResult.applied=True`` iff git exits 0. On failure,
    the worktree is left unchanged (git apply rejects atomically).

    The caller is responsible for resetting the worktree BEFORE
    calling apply_clean if a previous candidate's diff is in flight.
    """
    import time

    if not diff or not diff.strip():
        return ApplyResult(applied=False, stderr_excerpt="(empty diff)")

    t_start = time.perf_counter()
    try:
        res = sandbox.apply_patch(diff)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t_start
        return ApplyResult(
            applied=False,
            stderr_excerpt=f"{type(exc).__name__}: {exc}",
            duration_s=elapsed,
        )
    elapsed = time.perf_counter() - t_start

    if res.exit_code == 0:
        return ApplyResult(applied=True, duration_s=elapsed)

    err = (res.stderr or res.stdout or "").strip()
    if len(err) > 1000:
        err = err[:1000] + "\n…(truncated)"
    return ApplyResult(applied=False, stderr_excerpt=err, duration_s=elapsed)


__all__ = ["ApplyResult", "apply_clean"]

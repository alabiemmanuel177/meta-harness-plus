"""Phase 4 — static analysis (ruff + mypy).

Per docs/V10_DESIGN_PHASE4.md §3.2 step 5. Runs ruff if
``pyproject.toml`` declares it, mypy if ``mypy.ini`` exists. Per-tool
timeout 60s.

The validator returns ``StaticSignal.ruff_clean=None`` /
``mypy_clean=None`` when the tool is not configured for the repo —
the selector treats None as a missing signal, not as a positive or
negative.
"""

from __future__ import annotations

import logging
import time

from harness.views import StaticSignal


log = logging.getLogger(__name__)


DEFAULT_STATIC_TIMEOUT_S: int = 60


def _ruff_configured(sandbox) -> bool:
    """Return True iff `ruff` configuration is present in the repo
    (either pyproject.toml [tool.ruff] section or ruff.toml)."""
    try:
        res = sandbox.run_shell(
            "cd /testbed && ("
            "  test -f ruff.toml && echo R; "
            "  grep -q 'tool.ruff' pyproject.toml 2>/dev/null && echo P; "
            "  exit 0)",
            timeout_s=10.0,
        )
        return "R" in (res.stdout or "") or "P" in (res.stdout or "")
    except Exception:  # noqa: BLE001
        return False


def _mypy_configured(sandbox) -> bool:
    try:
        res = sandbox.run_shell(
            "cd /testbed && ("
            "  test -f mypy.ini && echo M; "
            "  grep -q 'tool.mypy' pyproject.toml 2>/dev/null && echo P; "
            "  exit 0)",
            timeout_s=10.0,
        )
        return "M" in (res.stdout or "") or "P" in (res.stdout or "")
    except Exception:  # noqa: BLE001
        return False


def _run_ruff(sandbox, *, timeout_s: int) -> bool | None:
    """Run ruff. Return True if clean, False if reports any issue,
    None if not installed in the container."""
    try:
        res = sandbox.run_shell(
            "cd /testbed && python -m ruff check --no-cache . 2>&1 | tail -30",
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.info("[validation.static] ruff failed to run: %s", exc)
        return None
    out = (res.stdout or "")
    if "No module named" in out and "ruff" in out:
        return None
    return res.exit_code == 0


def _run_mypy(sandbox, *, timeout_s: int) -> bool | None:
    try:
        res = sandbox.run_shell(
            "cd /testbed && python -m mypy --no-color-output --no-error-summary . 2>&1 | tail -30",
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.info("[validation.static] mypy failed to run: %s", exc)
        return None
    out = (res.stdout or "")
    if "No module named" in out and "mypy" in out:
        return None
    return res.exit_code == 0


def run_static(sandbox, *, timeout_s: int = DEFAULT_STATIC_TIMEOUT_S) -> StaticSignal:
    """Run ruff + mypy on the patched worktree. Returns StaticSignal
    with None for tools not configured."""
    t_start = time.perf_counter()

    ruff_clean: bool | None = None
    if _ruff_configured(sandbox):
        ruff_clean = _run_ruff(sandbox, timeout_s=timeout_s)

    mypy_clean: bool | None = None
    if _mypy_configured(sandbox):
        mypy_clean = _run_mypy(sandbox, timeout_s=timeout_s)

    elapsed = time.perf_counter() - t_start
    return StaticSignal(
        ruff_clean=ruff_clean,
        mypy_clean=mypy_clean,
        duration_s=elapsed,
    )


__all__ = ["DEFAULT_STATIC_TIMEOUT_S", "run_static"]

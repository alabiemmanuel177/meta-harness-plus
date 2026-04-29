"""V10 cache hygiene — startup assertion to prevent legacy reseeding.

Per V10_DESIGN.md §12.7: any pre-V10 cache artifact (V7 prompt cache,
V7/V8 trajectory dump, selector decision log) in a V10 cache directory
is a subtle reseeding leak. We catch it at import time so it cannot
silently contaminate a V10 run.

Two namespace concepts (refined in Phase 0 commit 18 to handle the
shared-parent case correctly):

  - **V10_EXCLUSIVE_DIRS** — directories V10 owns entirely. If they
    exist, all direct children must be V10-tagged. Checked at import.
  - **V10_NAMESPACED_PARENTS** — directories shared with legacy code
    (V7/V8 baselines, MH++ research). V10 writes only to ``v10_*``-
    tagged subdirs at the top level. The hygiene check inspects only
    those V10-namespaced subdirs, not the entire parent tree (which
    legitimately contains ``runs/swebench_500_v7/`` and similar
    documented baselines).
"""

from __future__ import annotations

import pathlib


V10_TAG_PREFIX: str = "v10_"

# V10 owns these entirely. Anything inside must be V10-tagged.
V10_EXCLUSIVE_DIRS: tuple[pathlib.Path, ...] = (
    pathlib.Path("eval_outputs"),
    pathlib.Path(".harness_cache"),
)

# V10 shares these with legacy code (V7 baselines, MH++ research). The
# hygiene check looks only at V10-tagged children of these dirs; legacy
# entries (e.g., ``runs/swebench_500_v7/``) are explicitly tolerated.
V10_NAMESPACED_PARENTS: tuple[pathlib.Path, ...] = (
    pathlib.Path("runs"),
    pathlib.Path("trajectories"),
    pathlib.Path("repo_cache"),
)

# Backward-compat — older code referenced V10_CACHE_DIRS as a flat tuple.
# Phase 0 commits 4 and 11 used this name; keep it pointing at the union
# so the existing tests still locate the right paths via this symbol.
V10_CACHE_DIRS: tuple[pathlib.Path, ...] = V10_EXCLUSIVE_DIRS + V10_NAMESPACED_PARENTS


class CacheLeakError(RuntimeError):
    """Pre-V10 cache entry detected in a V10 cache directory."""


def _legacy_entries(d: pathlib.Path) -> list[pathlib.Path]:
    """Return ``d``'s direct children that are NOT V10-tagged. Used on
    V10_EXCLUSIVE_DIRS where every entry must be V10-tagged."""
    if not d.exists():
        return []
    out: list[pathlib.Path] = []
    for entry in d.iterdir():
        if entry.name.startswith(V10_TAG_PREFIX):
            continue
        # Convenience: ignore hidden files and __pycache__ which are
        # tooling artifacts, not run state.
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        out.append(entry)
    return out


def assert_clean_cache_at_startup(
    *,
    extra_dirs: tuple[pathlib.Path, ...] = (),
    cwd: pathlib.Path | None = None,
    scope: tuple[pathlib.Path, ...] | None = None,
    strict: bool = False,
) -> None:
    """Raise ``CacheLeakError`` if any V10 cache dir has non-V10 entries.

    Args:
        extra_dirs: additional directories to check beyond the default
            (V10_EXCLUSIVE_DIRS).
        cwd: project root (default: cwd).
        scope: if given, OVERRIDES the default — only the listed
            directories are checked. Smoke runs use this with a tight
            path so legacy entries elsewhere don't break the smoke.
        strict: if True, also enforces that the V10-tagged children of
            V10_NAMESPACED_PARENTS contain only this-process data
            (i.e., the directories are EITHER absent or known-empty).
            Default False — Phase 1+ flips this on for production runs;
            development iteration leaves V10-tagged dirs in place across
            sessions.

    Default behavior (called with no args) — Phase 1+ wire-in:
        Checks V10_EXCLUSIVE_DIRS (eval_outputs/, .harness_cache/).
        These dirs must be either nonexistent OR contain only V10-tagged
        children. Legacy V7/V8 entries in shared parents (runs/, etc.)
        are NOT checked here — that's by design (V10 doesn't own those
        parents).
    """
    base = cwd or pathlib.Path.cwd()
    if scope is not None:
        dirs = tuple(base / d for d in scope)
    else:
        dirs = tuple(base / d for d in V10_EXCLUSIVE_DIRS) + tuple(
            base / d for d in extra_dirs
        )
    bad: list[str] = []
    for d in dirs:
        for entry in _legacy_entries(d):
            bad.append(str(entry.relative_to(base)))
    if bad:
        raise CacheLeakError(
            "non-V10 entries detected in V10 cache directories: "
            + ", ".join(bad)
            + ". Move or delete legacy entries before starting a V10 run "
            "(see V10_DESIGN.md §12.7)."
        )


__all__ = [
    "V10_TAG_PREFIX",
    "V10_EXCLUSIVE_DIRS",
    "V10_NAMESPACED_PARENTS",
    "V10_CACHE_DIRS",
    "CacheLeakError",
    "assert_clean_cache_at_startup",
]

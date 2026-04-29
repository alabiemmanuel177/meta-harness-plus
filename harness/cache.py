"""V10 cache hygiene — startup assertion to prevent legacy reseeding.

Per V10_DESIGN.md §12.7: any pre-V10 cache artifact (V7 prompt cache,
V7/V8 trajectory dump, selector decision log) in a V10 cache directory
is a subtle reseeding leak. We catch it at import time so it cannot
silently contaminate a V10 run.

Caches must be either nonexistent, empty, or contain only entries whose
names start with ``V10_TAG_PREFIX``.
"""

from __future__ import annotations

import pathlib


V10_TAG_PREFIX: str = "v10_"

V10_CACHE_DIRS: tuple[pathlib.Path, ...] = (
    pathlib.Path("runs"),
    pathlib.Path("trajectories"),
    pathlib.Path("repo_cache"),
    pathlib.Path(".harness_cache"),
)


class CacheLeakError(RuntimeError):
    """Pre-V10 cache entry detected in a V10 cache directory."""


def _legacy_entries(d: pathlib.Path) -> list[pathlib.Path]:
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
) -> None:
    """Raise ``CacheLeakError`` if any V10 cache dir has non-V10 entries.

    Args:
        extra_dirs: additional directories to check beyond V10_CACHE_DIRS.
        cwd: project root (default: cwd).
        scope: if given, OVERRIDES V10_CACHE_DIRS — only the listed
            directories are checked. Phase 0 smoke uses a scoped check
            (e.g., trajectories/v10_smoke/) so legacy V7/V8 caches in
            runs/ and trajectories/ don't break the smoke; Phase 1+
            production runs use the global default to enforce strict
            hygiene before any LLM call.

    Phase 0 callers: invoke from ``scripts/smoke_phase0.py`` once at
    startup with ``scope`` set to the smoke's own write paths. Phase 3+
    wires this into ``harness/__init__.py`` (no scope) so the global
    check fires before any V10 code runs.
    """
    base = cwd or pathlib.Path.cwd()
    if scope is not None:
        dirs = tuple(base / d for d in scope)
    else:
        dirs = tuple(base / d for d in V10_CACHE_DIRS) + tuple(base / d for d in extra_dirs)
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
    "V10_CACHE_DIRS",
    "CacheLeakError",
    "assert_clean_cache_at_startup",
]

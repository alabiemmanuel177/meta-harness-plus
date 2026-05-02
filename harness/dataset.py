"""SWE-bench loader — projects raw rows to ``InstanceView``.

This module contains the SINGLE asserted boundary at which oracle-
derived dataset fields are dropped. ``_project_to_view`` is whitelist-
based: it reads only the keys it needs (``_PROJECTED_KEYS``) and
stores them in ``InstanceView``. No oracle-derived field name is read
by subscript or by ``getattr`` — the firewall test enforces this at
AST level (see V10_DESIGN.md §12.4 for the full list of fields kept
out of view).

Two splits are supported:

  - ``verified`` (default): meta_harness_plus/tasks/data/swebench_verified.jsonl
    (500 rows, the SWE-bench Verified set).
  - ``pro``: meta_harness_plus/tasks/data/swebench_pro.jsonl (731 rows,
    the SWE-bench Pro public set).

The active split is selected by the ``V10_SPLIT`` env var, or by the
``split_name=`` parameter on the public loaders. Same projection
function for both splits — Pro carries an additional ``dockerhub_tag``
field that maps to ``jefzda/sweap-images:{tag}`` for the sandbox.

Boundary contract (asserted at runtime in ``_assert_projection_boundary``):

  - The output is an ``InstanceView``. The view's class is in
    ``harness.views.VIEW_CLASSES``, which guarantees no field name
    matches an oracle-derived token (the views' ``__post_init__``
    rejects any such addition).
  - The set of source-row keys we read is exactly ``_PROJECTED_KEYS``.
    Any drift (e.g., someone adds a new dataset field that happens to
    contain oracle data) is caught by the firewall AST scan at the
    next test run.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Iterable, Iterator

from harness.views import (
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)
from harness.repo_conventions import resolve_test_directives_static


# ---------------------------------------------------------------------------
# The whitelist — exactly what we read from each row. Keep this list short.
# ``dockerhub_tag`` is Pro-only infrastructure (image lookup), not
# oracle data — see V10_DESIGN.md §13.2 (Pro delta).
# ---------------------------------------------------------------------------

_PROJECTED_KEYS: frozenset[str] = frozenset({
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "dockerhub_tag",  # Pro-only; absent on Verified rows (handled with .get())
})


# ---------------------------------------------------------------------------
# Split registry — extend this dict to wire a new dataset split.
# ---------------------------------------------------------------------------

_DATA_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "meta_harness_plus" / "tasks" / "data"
)

_SPLITS: dict[str, dict] = {
    "verified": {
        "path": _DATA_DIR / "swebench_verified.jsonl",
        "expected_rows": 500,
        "min_size_bytes": 1_000_000,
    },
    "pro": {
        "path": _DATA_DIR / "swebench_pro.jsonl",
        "expected_rows": 731,
        "min_size_bytes": 5_000_000,
    },
}

_DEFAULT_SPLIT_ENV = "V10_SPLIT"


def _resolve_split_name(split_name: str | None) -> str:
    """Pick the active split. Explicit arg wins; else env var; else default."""
    if split_name is None:
        split_name = os.environ.get(_DEFAULT_SPLIT_ENV, "verified")
    if split_name not in _SPLITS:
        raise ValueError(
            f"unknown split {split_name!r}; must be one of {sorted(_SPLITS)}"
        )
    return split_name


def _split_config(split_name: str | None = None) -> dict:
    return _SPLITS[_resolve_split_name(split_name)]


def _default_dataset_path(split_name: str | None = None) -> pathlib.Path:
    """Single source of truth for the active split.

    Per V10_DESIGN.md §13.2: these files are canonical. We do NOT call
    ``datasets.load_dataset(...)`` at runtime; HF cache, arrow shards,
    and any other network-derived path are explicitly ruled out. The
    files are committed once and frozen; treat them as build artifacts,
    not refreshable inputs.
    """
    return _split_config(split_name)["path"]


def _assert_dataset_source_of_truth(path: pathlib.Path, split_name: str | None = None) -> None:
    cfg = _split_config(split_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. V10 reads only the local jsonl; "
            f"no HuggingFace fallback. See V10_DESIGN.md §13.2."
        )
    # Sanity-check size: a 0-byte or wildly-shrunken file means the cache
    # is corrupted.
    if path.stat().st_size < cfg["min_size_bytes"]:
        raise ValueError(
            f"Dataset at {path} is suspiciously small "
            f"({path.stat().st_size} bytes). Expected ~{cfg['min_size_bytes']} bytes / "
            f"{cfg['expected_rows']} rows."
        )


def _iter_raw_rows(path: pathlib.Path, split_name: str | None = None) -> Iterator[dict]:
    _assert_dataset_source_of_truth(path, split_name=split_name)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_skeleton_for_view(repo: str, base_commit: str) -> RepoSkeleton:
    """Try to load a cached skeleton from ``harness.skeleton``'s cache;
    return an empty skeleton if not cached.

    Phase 1 callers fill in the skeleton on-demand via
    ``harness.skeleton.load_or_build_skeleton(view, sandbox=...)``.
    Cached skeletons are loaded transparently; empty is the documented
    default for instances whose skeleton hasn't been built yet.

    This replaces the Phase 0 placeholder (``_build_skeleton_phase0``)
    that always returned empty.
    """
    # Lazy import to keep dataset.py free of skeleton's heavier deps
    # (and to avoid an import cycle should skeleton.py ever import
    # dataset.py for any reason).
    from harness.skeleton import _cache_path, _deserialize_skeleton
    path = _cache_path(repo, base_commit)
    if path.exists():
        try:
            return _deserialize_skeleton(path)
        except Exception:
            # Malformed cache file → fall through to empty rather than
            # crashing the loader. Phase 1's load_or_build_skeleton
            # will rebuild on next access.
            pass
    return RepoSkeleton(repo=repo, base_commit=base_commit, files=())


def _project_to_view(row: dict) -> InstanceView:
    """The asserted boundary.

    This is the only place a raw dataset row touches V10 code. The
    projection is whitelist-based: each field below is read by name from
    the row dict. Oracle-derived fields are not in the whitelist and
    therefore never reach an InstanceView.

    ``dockerhub_tag`` is read with ``.get()`` because it is present
    only on Pro rows. Verified rows leave it empty; Sandbox falls back
    to the legacy ``swebench/sweb.eval.*`` image-name derivation.
    """
    instance_id = row["instance_id"]
    repo = row["repo"]
    base_commit = row["base_commit"]
    problem_statement = row["problem_statement"]
    dockerhub_tag = row.get("dockerhub_tag", "") or ""

    override = resolve_test_directives_static(repo)
    if override is None:
        # No override for this repo — Phase 0 records the gap so we can
        # add an entry to repo_conventions.py before we run the smoke.
        # Not a hard error: discovery happens later when the sandbox
        # mounts the repo at base_commit.
        dirs = ("tests/",)
        source = "fallback:default(unknown_repo)"
    else:
        dirs = override
        source = "override:repo_conventions"

    return InstanceView(
        instance_id=instance_id,
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem_statement,
        repo_skeleton=_build_skeleton_for_view(repo, base_commit),
        test_directives=TestDirectives(dirs=dirs, source=source),
        dockerhub_tag=dockerhub_tag,
    )


def _assert_projection_boundary(row: dict, view: InstanceView) -> None:
    """Defense in depth: confirm projection identity for the keys we
    explicitly read. Runs once per row at load time; cheap.

    Note: oracle-derived tokens may legitimately appear inside a
    problem_statement (an issue text can quote an error message that
    mentions one of the tokens by name) — this is the over-flag case
    noted in V10_DESIGN.md §12.5. We do NOT scan problem_statement here
    on the fundamental ground that runtime token-walking belongs at LLM
    call sites, not at the dataset projection boundary.
    """
    if view.instance_id != row["instance_id"]:
        raise AssertionError(
            f"projection mismatch: view.instance_id={view.instance_id!r}, "
            f"row.instance_id={row['instance_id']!r}"
        )
    if view.repo != row["repo"]:
        raise AssertionError(
            f"projection mismatch: view.repo={view.repo!r}, row.repo={row['repo']!r}"
        )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_verified_views(
    *,
    n: int | None = None,
    instance_ids: Iterable[str] | None = None,
    cached_path: str | pathlib.Path | None = None,
    split_name: str | None = None,
) -> list[InstanceView]:
    """Load SWE-bench rows from the local cache and project to
    InstanceView. Whitelisted-field-only; never reads forbidden fields.

    ``split_name`` selects between the registered splits (default reads
    the ``V10_SPLIT`` env var, then falls back to ``"verified"``).
    The function name preserves the legacy public API; use
    ``load_pro_views`` for explicit Pro loading.
    """
    resolved_split = _resolve_split_name(split_name)
    path = pathlib.Path(cached_path) if cached_path else _default_dataset_path(resolved_split)

    wanted: set[str] | None = set(instance_ids) if instance_ids is not None else None
    out: list[InstanceView] = []
    for row in _iter_raw_rows(path, split_name=resolved_split):
        if wanted is not None and row["instance_id"] not in wanted:
            continue
        view = _project_to_view(row)
        _assert_projection_boundary(row, view)
        out.append(view)
        if n is not None and len(out) >= n:
            break
    return out


def load_verified_view(instance_id: str, *, split_name: str | None = None) -> InstanceView:
    """Convenience: project exactly one instance from the active split."""
    views = load_verified_views(instance_ids=[instance_id], split_name=split_name)
    if not views:
        raise KeyError(f"instance_id not in dataset: {instance_id!r}")
    return views[0]


def load_pro_views(
    *,
    n: int | None = None,
    instance_ids: Iterable[str] | None = None,
) -> list[InstanceView]:
    """Explicit Pro loader — equivalent to ``load_verified_views(split_name='pro')``.

    Use this when callers want to be unambiguous about which split they
    want, regardless of the ``V10_SPLIT`` env var.
    """
    return load_verified_views(n=n, instance_ids=instance_ids, split_name="pro")


def load_pro_view(instance_id: str) -> InstanceView:
    """Project exactly one instance from the SWE-bench Pro split."""
    return load_verified_view(instance_id, split_name="pro")


__all__ = [
    "load_verified_views",
    "load_verified_view",
    "load_pro_views",
    "load_pro_view",
]

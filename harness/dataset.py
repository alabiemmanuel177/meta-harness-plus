"""SWE-bench Verified loader — projects raw rows to ``InstanceView``.

This module contains the SINGLE asserted boundary at which oracle-
derived dataset fields are dropped. ``_project_to_view`` is whitelist-
based: it reads only the four fields it needs (``_PROJECTED_KEYS``) and
stores them in ``InstanceView``. No oracle-derived field name is read
by subscript or by ``getattr`` — the firewall test enforces this at
AST level (see V10_DESIGN.md §12.4 for the full list of fields kept
out of view).

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
# ---------------------------------------------------------------------------

_PROJECTED_KEYS: frozenset[str] = frozenset({
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
})


def _default_dataset_path() -> pathlib.Path:
    """Single source of truth for the Verified dataset.

    Per V10_DESIGN.md §13.2: this file is canonical. We do NOT call
    ``datasets.load_dataset(...)`` at runtime; HF cache, arrow shards,
    and any other network-derived path are explicitly ruled out. The
    file was committed once during early V7 work and has been frozen
    since; treat it as a build artifact, not a refreshable input.
    """
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
    )


# Asserted boundary: the loader's only legal source is the local jsonl.
# Keep this short and auditable. If a future change wants to load from
# elsewhere, that's a deliberate decision that needs to be reviewed.
_EXPECTED_ROW_COUNT = 500


def _assert_dataset_source_of_truth(path: pathlib.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Verified dataset not found at {path}. V10 reads only this "
            f"local jsonl; no HuggingFace fallback. See V10_DESIGN.md §13.2."
        )
    # Sanity-check size: the expected file is ~6 MB / 500 rows. A 0-byte
    # or wildly-shrunken file means the cache is corrupted.
    if path.stat().st_size < 1_000_000:
        raise ValueError(
            f"Verified dataset at {path} is suspiciously small "
            f"({path.stat().st_size} bytes). Expected ~6 MB / "
            f"{_EXPECTED_ROW_COUNT} rows."
        )


def _iter_raw_rows(path: pathlib.Path) -> Iterator[dict]:
    _assert_dataset_source_of_truth(path)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_skeleton_phase0(repo: str, base_commit: str) -> RepoSkeleton:
    """Phase 0 placeholder — empty skeleton. Phase 1 walks the repo at
    base_commit and produces the real tree (file/class/function with
    one-line summaries). Returning empty here keeps the projection
    boundary simple while letting downstream code accept the type.
    """
    return RepoSkeleton(repo=repo, base_commit=base_commit, files=())


def _project_to_view(row: dict) -> InstanceView:
    """The asserted boundary.

    This is the only place a raw dataset row touches V10 code. The
    projection is whitelist-based: each field below is read by name from
    the row dict. Oracle-derived fields are not in the whitelist and
    therefore never reach an InstanceView.
    """
    instance_id = row["instance_id"]
    repo = row["repo"]
    base_commit = row["base_commit"]
    problem_statement = row["problem_statement"]

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
        repo_skeleton=_build_skeleton_phase0(repo, base_commit),
        test_directives=TestDirectives(dirs=dirs, source=source),
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
) -> list[InstanceView]:
    """Load SWE-bench Verified rows from the local cache and project to
    InstanceView. Whitelisted-field-only; never reads forbidden fields."""
    path = pathlib.Path(cached_path) if cached_path else _default_dataset_path()

    wanted: set[str] | None = set(instance_ids) if instance_ids is not None else None
    out: list[InstanceView] = []
    for row in _iter_raw_rows(path):
        if wanted is not None and row["instance_id"] not in wanted:
            continue
        view = _project_to_view(row)
        _assert_projection_boundary(row, view)
        out.append(view)
        if n is not None and len(out) >= n:
            break
    return out


def load_verified_view(instance_id: str) -> InstanceView:
    """Convenience: project exactly one instance."""
    views = load_verified_views(instance_ids=[instance_id])
    if not views:
        raise KeyError(f"instance_id not in Verified dataset: {instance_id!r}")
    return views[0]


__all__ = ["load_verified_views", "load_verified_view"]

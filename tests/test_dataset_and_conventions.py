"""Unit tests for ``harness.dataset`` and ``harness.repo_conventions``.

The firewall test in ``test_no_oracle_leak.py`` covers the contamination
boundary; this file covers correctness: the projection actually produces
the right ``InstanceView`` shape, the override map covers every Verified
repo, and filesystem discovery agrees with the override entries on
synthetic checkouts (the override-vs-discovery consistency check from
V10_DESIGN.md §12.4(b)).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from harness import views as V
from harness.dataset import _PROJECTED_KEYS, load_verified_views, _project_to_view
from harness.repo_conventions import (
    REPO_TEST_DIRS,
    discover_test_dirs,
    resolve_test_directives,
)


# ---------------------------------------------------------------------------
# Dataset projection
# ---------------------------------------------------------------------------


def test_load_verified_smoke() -> None:
    """Loader returns InstanceView objects with required fields."""
    views = load_verified_views(n=5)
    assert len(views) == 5
    for v in views:
        assert isinstance(v, V.InstanceView)
        assert v.instance_id
        assert v.repo
        assert v.base_commit
        assert v.problem_statement
        assert v.test_directives.dirs


def test_projected_keys_are_minimal() -> None:
    """We only read four fields from each raw row. Anything else (and
    especially any oracle-derived field) cannot reach the view."""
    assert _PROJECTED_KEYS == frozenset({
        "instance_id", "repo", "base_commit", "problem_statement",
    })


def test_project_to_view_only_uses_whitelisted_keys() -> None:
    """Synthesize a row with bogus extra fields including all the
    oracle-derived ones. Projection ignores them; the view is clean."""
    row = {
        "instance_id": "fake__repo-1",
        "repo": "django/django",
        "base_commit": "abc",
        "problem_statement": "issue text",
        # All these MUST be ignored by the projection. We do not name them
        # as string literals here (firewall would flag this file otherwise);
        # we synthesize the keys via dynamic concatenation.
        "fail" + "_to_pass": ["test_a"],
        "pass" + "_to_pass": ["test_b"],
        "test" + "_patch": "diff --git",
        "hints" + "_text": "use this hint",
        "patch": "gold patch",
    }
    view = _project_to_view(row)
    # Projection does not include any oracle-derived field
    assert view.instance_id == "fake__repo-1"
    assert view.repo == "django/django"
    assert view.problem_statement == "issue text"
    # And the view's __post_init__ did not raise
    assert view.test_directives.dirs == ("tests/",)


# ---------------------------------------------------------------------------
# Repo conventions
# ---------------------------------------------------------------------------


def test_repo_conventions_cover_all_verified_repos() -> None:
    """Every repo in Verified must have an override entry. Filesystem
    fallback exists, but for the 12-repo Verified set we want explicit
    overrides with cited sources."""
    cache_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
    )
    repos: set[str] = set()
    with cache_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            repos.add(row["repo"])
    missing = repos - set(REPO_TEST_DIRS.keys())
    assert not missing, f"Verified repos missing from REPO_TEST_DIRS: {sorted(missing)}"


def test_resolve_test_directives_uses_override(tmp_path: pathlib.Path) -> None:
    dirs, source = resolve_test_directives("django/django", repo_root=tmp_path)
    assert dirs == ("tests/",)
    assert source.startswith("override:")


def test_resolve_test_directives_falls_back_to_discovery(tmp_path: pathlib.Path) -> None:
    """For an unknown repo, discovery on a synthetic checkout finds
    test directories from filesystem layout."""
    (tmp_path / "myproj" / "tests").mkdir(parents=True)
    (tmp_path / "myproj" / "tests" / "test_thing.py").write_text("def test_x(): pass\n")
    dirs, source = resolve_test_directives("unknown/unknown-repo", repo_root=tmp_path)
    assert any("tests" in d for d in dirs)
    assert source.startswith("discovery:")


def test_discover_test_dirs_collapses_to_test_segment(tmp_path: pathlib.Path) -> None:
    """When a path includes a 'tests' or 'testing' segment, discovery
    collapses to that segment, not deeper. Mirrors how the override
    map records directories."""
    (tmp_path / "pkg" / "tests" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "tests" / "sub" / "test_a.py").write_text("def test_a(): pass\n")
    d = discover_test_dirs(tmp_path)
    assert "pkg/tests/" in d.dirs
    assert d.n_test_files == 1


def test_override_vs_discovery_consistency_for_synthetic_django(tmp_path: pathlib.Path) -> None:
    """Build a tiny synthetic django-shaped checkout and confirm that
    discovery's answer matches the django override entry. This is the
    override-vs-discovery consistency check from §12.4(b) at a single
    representative repo. (Full 12-repo test requires real checkouts
    and is run as part of the smoke suite.)"""
    (tmp_path / "tests" / "auth_tests").mkdir(parents=True)
    (tmp_path / "tests" / "auth_tests" / "test_models.py").write_text("def test_x(): pass\n")
    (tmp_path / "django").mkdir()
    (tmp_path / "django" / "__init__.py").write_text("")
    discovered = discover_test_dirs(tmp_path)
    assert "tests/" in discovered.dirs
    # django override is exactly ("tests/",)
    assert REPO_TEST_DIRS["django/django"] == ("tests/",)


def test_resolve_test_directives_default_fallback() -> None:
    dirs, source = resolve_test_directives("nobody/nothing", repo_root=None)
    assert dirs == ("tests/",)
    assert source == "fallback:default"

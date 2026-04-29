"""Tests for harness.skeleton — Stage 1a builder + (repo, base_commit)
cache.

Docker-dependent build is exercised separately by an integration smoke
in scripts/. Here we test the pure parts: serialization round-trip,
cache path conventions, dataset integration with cached skeletons.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from harness import skeleton as sk
from harness.dataset import _project_to_view
from harness.skeleton import (
    CACHE_ROOT,
    _cache_path,
    _deserialize_skeleton,
    _file_summary_dict,
    _file_summary_from_dict,
    _serialize_skeleton,
    cache_path_for,
    load_or_build_skeleton,
)
from harness.views import (
    ClassSummary,
    FileSummary,
    FunctionSummary,
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)


# ---------------------------------------------------------------------------
# Cache path conventions
# ---------------------------------------------------------------------------


def test_cache_path_uses_v10_namespace_and_safe_slug() -> None:
    p = _cache_path("django/django", "abcdef0123456789")
    assert p.parts[0] == "repo_cache"
    assert p.parts[1] == "v10_skeletons"
    assert "django_django" in p.parts          # slash → underscore
    assert p.parts[3] == "abcdef012345"        # truncated to 12 chars
    assert p.name == "skeleton.json"


def test_cache_path_for_view_resolves_consistently() -> None:
    view = _make_view()
    p1 = cache_path_for(view)
    p2 = _cache_path(view.repo, view.base_commit)
    assert p1 == p2


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_file_summary_dict_round_trip() -> None:
    f = FileSummary(
        path="django/contrib/auth/forms.py",
        classes=(
            ClassSummary(
                name="UserCreationForm",
                line_start=10,
                line_end=80,
                methods=(
                    FunctionSummary(name="__init__", line_start=15, line_end=25),
                    FunctionSummary(name="clean", line_start=27, line_end=40),
                ),
            ),
        ),
        functions=(FunctionSummary(name="_helper", line_start=82, line_end=90),),
    )
    d = _file_summary_dict(f)
    f2 = _file_summary_from_dict(d)
    assert f2.path == f.path
    assert len(f2.classes) == 1
    assert f2.classes[0].name == "UserCreationForm"
    assert len(f2.classes[0].methods) == 2
    assert f2.classes[0].methods[1].name == "clean"
    assert len(f2.functions) == 1


def test_skeleton_serialize_deserialize_round_trip(tmp_path: pathlib.Path) -> None:
    skel = RepoSkeleton(
        repo="django/django",
        base_commit="abc1234567890",
        files=(
            FileSummary(
                path="x/y.py",
                classes=(ClassSummary(name="Foo", line_start=1, line_end=10),),
                functions=(FunctionSummary(name="bar", line_start=12, line_end=20),),
            ),
        ),
    )
    path = tmp_path / "skel.json"
    _serialize_skeleton(skel, path)
    rt = _deserialize_skeleton(path)
    assert rt.repo == skel.repo
    assert rt.base_commit == skel.base_commit
    assert len(rt.files) == 1
    assert rt.files[0].path == "x/y.py"
    assert rt.files[0].classes[0].name == "Foo"


def test_skeleton_serialization_includes_count_metadata(tmp_path: pathlib.Path) -> None:
    skel = RepoSkeleton(
        repo="psf/requests",
        base_commit="abc",
        files=(
            FileSummary(
                path="a.py",
                classes=(ClassSummary(name="A", line_start=1, line_end=2),),
                functions=(),
            ),
            FileSummary(
                path="b.py",
                classes=(),
                functions=(FunctionSummary(name="f", line_start=1, line_end=2),),
            ),
        ),
    )
    path = tmp_path / "s.json"
    _serialize_skeleton(skel, path)
    payload = json.loads(path.read_text())
    assert payload["n_files"] == 2
    assert payload["n_classes"] == 1
    assert payload["n_functions"] == 1
    assert payload["_v10_skeleton_version"] == 1


def test_deserialize_rejects_unknown_version(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "_v10_skeleton_version": 999,
        "repo": "x",
        "base_commit": "y",
        "files": [],
    }))
    with pytest.raises(ValueError):
        _deserialize_skeleton(path)


# ---------------------------------------------------------------------------
# load_or_build_skeleton — cache lookup branch (no docker)
# ---------------------------------------------------------------------------


def test_load_or_build_returns_cached_when_available(tmp_path: pathlib.Path) -> None:
    view = _make_view()
    skel = RepoSkeleton(repo=view.repo, base_commit=view.base_commit, files=())
    # Place a cached skeleton at the override location.
    rel = _cache_path(view.repo, view.base_commit).relative_to(CACHE_ROOT)
    cache_path = tmp_path / rel
    _serialize_skeleton(skel, cache_path)
    out = load_or_build_skeleton(view, sandbox=None, cache_root=tmp_path)
    assert out.repo == view.repo


def test_load_or_build_raises_when_no_cache_and_no_sandbox(tmp_path: pathlib.Path) -> None:
    view = _make_view()
    with pytest.raises(RuntimeError, match="not cached"):
        load_or_build_skeleton(view, sandbox=None, cache_root=tmp_path)


# ---------------------------------------------------------------------------
# Dataset integration — projection picks up cached skeleton
# ---------------------------------------------------------------------------


def test_dataset_projection_loads_cached_skeleton(monkeypatch, tmp_path: pathlib.Path) -> None:
    """If a skeleton is cached at the canonical location, the dataset
    projection picks it up automatically. Otherwise the projection
    returns an empty skeleton (Phase 0 default)."""
    # Patch the production CACHE_ROOT to tmp_path so we don't pollute
    # the real cache.
    monkeypatch.setattr(sk, "CACHE_ROOT", tmp_path)
    # Re-importing _cache_path to rebind the patched CACHE_ROOT:
    def patched_cache_path(repo: str, base_commit: str) -> pathlib.Path:
        return tmp_path / sk._slug(repo) / base_commit[:12] / "skeleton.json"
    monkeypatch.setattr(sk, "_cache_path", patched_cache_path)

    row = {
        "instance_id": "djx__djx-1",
        "repo": "django/django",
        "base_commit": "abc1234567890def",
        "problem_statement": "broken",
    }
    # First: no cached skeleton, projection returns empty.
    view_empty = _project_to_view(row)
    assert len(view_empty.repo_skeleton.files) == 0

    # Place a cached skeleton at the patched path.
    skel = RepoSkeleton(
        repo="django/django",
        base_commit="abc1234567890def",
        files=(
            FileSummary(
                path="cached.py",
                classes=(ClassSummary(name="C", line_start=1, line_end=5),),
                functions=(),
            ),
        ),
    )
    cache_path = patched_cache_path("django/django", "abc1234567890def")
    _serialize_skeleton(skel, cache_path)

    view_cached = _project_to_view(row)
    assert len(view_cached.repo_skeleton.files) == 1
    assert view_cached.repo_skeleton.files[0].path == "cached.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_view() -> InstanceView:
    return InstanceView(
        instance_id="ex__ex-1",
        repo="example/example",
        base_commit="abc1234567890",
        problem_statement="something",
        repo_skeleton=RepoSkeleton(repo="example/example", base_commit="abc1234567890", files=()),
        test_directives=TestDirectives(dirs=("tests/",), source="test:smoke"),
    )

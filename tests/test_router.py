"""Phase 3 P3d-fix — two-way router tests + firewall.

Covers:

  - Feature extraction shape (issue_word_count, traceback regex,
    candidate_file_count, repo_id passthrough).
  - Each routing branch fires at the documented thresholds.
  - Determinism: same RouterFeatures always returns same strategy.
  - Defense in depth: importing router.py does NOT pull in
    harness.repro / harness.eval / harness.memory.
  - PIPELINE_ONE_SHOT was dropped in P3d-fix; this is asserted
    explicitly so a future revival is a deliberate choice not a
    silent re-add.

No LLM calls; no Docker.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from harness.localization_signals import RankedFile
from harness.patch_gen.router import (
    AGENT_LARGE_TOP1_LOC_MIN,
    AGENT_LONG_ISSUE_WORDS_MIN,
    PatchGenStrategy,
    RouterFeatures,
    _has_traceback,
    extract_features,
    route,
)
from harness.views import (
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_view(
    *,
    instance_id: str = "example__example-1",
    repo: str = "example/example",
    problem_statement: str = "Bug: foo() returns None",
) -> InstanceView:
    return InstanceView(
        instance_id=instance_id,
        repo=repo,
        base_commit="abcdef0123",
        problem_statement=problem_statement,
        repo_skeleton=RepoSkeleton(repo=repo, base_commit="abcdef0123"),
        test_directives=TestDirectives(dirs=("tests/",), source="discovery:1"),
    )


def _make_ranked_files(n: int) -> list[RankedFile]:
    return [
        RankedFile(
            file_path=f"src/f{i}.py",
            final_score=1.0 - i * 0.05,
            rationale=f"r{i}",
            upstream_signals=("bm25",),
            upstream_best_rank={"bm25": i + 1},
        )
        for i in range(n)
    ]


def _features(
    *,
    words: int = 200,
    traceback: bool = False,
    candidate_files: int = 10,
    top1_loc: int = 800,
    repo: str = "example/example",
    skel_size: int = 0,
) -> RouterFeatures:
    return RouterFeatures(
        issue_word_count=words,
        has_traceback_in_issue=traceback,
        candidate_file_count=candidate_files,
        top1_file_loc=top1_loc,
        repo_id=repo,
        skeleton_size_chars=skel_size,
    )


# ---------------------------------------------------------------------------
# 1. Feature extraction (unchanged from P3d)
# ---------------------------------------------------------------------------


def test_extract_features_basic():
    view = _make_view(problem_statement="Bug: foo returns None when bar")
    rfs = _make_ranked_files(5)
    feats = extract_features(view, rfs)
    assert feats.issue_word_count == 6
    assert feats.has_traceback_in_issue is False
    assert feats.candidate_file_count == 5
    assert feats.repo_id == "example/example"
    assert feats.top1_file_loc == 0
    assert feats.skeleton_size_chars == 0


def test_extract_features_passthrough_top1_loc():
    view = _make_view()
    rfs = _make_ranked_files(3)
    feats = extract_features(view, rfs, top1_file_loc=2_500, skeleton_size_chars=12_345)
    assert feats.top1_file_loc == 2_500
    assert feats.skeleton_size_chars == 12_345


def test_extract_features_traceback_detection_python():
    issue = (
        "Calling User.objects.create() crashes:\n\n"
        "Traceback (most recent call last):\n"
        '  File "/path/manage.py", line 22, in <module>\n'
        "    main()\n"
        "ValueError: foo\n"
    )
    view = _make_view(problem_statement=issue)
    feats = extract_features(view, _make_ranked_files(2))
    assert feats.has_traceback_in_issue is True


def test_has_traceback_helper_negative():
    """English text containing the words 'traceback' or 'file' should
    NOT match — the regex requires the structural traceback header
    or `File "..."` form."""
    assert _has_traceback("There's no traceback here.") is False
    assert _has_traceback("Just a file name like foo.py.") is False
    assert _has_traceback("") is False
    assert _has_traceback("normal sentence") is False


def test_router_features_post_init_validation():
    with pytest.raises(ValueError, match="issue_word_count"):
        RouterFeatures(
            issue_word_count=-1, has_traceback_in_issue=False,
            candidate_file_count=0, top1_file_loc=0,
            repo_id="x/y", skeleton_size_chars=0,
        )
    with pytest.raises(ValueError, match="candidate_file_count"):
        RouterFeatures(
            issue_word_count=0, has_traceback_in_issue=False,
            candidate_file_count=-1, top1_file_loc=0,
            repo_id="x/y", skeleton_size_chars=0,
        )


# ---------------------------------------------------------------------------
# 2. Routing branches (P3d-fix two-way)
# ---------------------------------------------------------------------------


def test_route_agent_when_long_issue_no_traceback_large_file():
    """Rule 1 fires: long issue (>=300 words) + no traceback + large
    top-1 file (>=800 LOC) → AGENT."""
    feats = _features(
        words=AGENT_LONG_ISSUE_WORDS_MIN + 50,
        traceback=False,
        top1_loc=AGENT_LARGE_TOP1_LOC_MIN + 100,
    )
    assert route(feats) == PatchGenStrategy.AGENT


def test_route_agent_at_exact_thresholds():
    """Boundary check: each AGENT condition exactly at its min →
    still fires (>= comparisons)."""
    feats = _features(
        words=AGENT_LONG_ISSUE_WORDS_MIN,
        traceback=False,
        top1_loc=AGENT_LARGE_TOP1_LOC_MIN,
    )
    assert route(feats) == PatchGenStrategy.AGENT


def test_route_falls_through_when_traceback_present():
    """Tracebacks anchor the localizer; even a long issue with a
    large file won't go to bare AGENT — fall through to default."""
    feats = _features(
        words=AGENT_LONG_ISSUE_WORDS_MIN + 100,
        traceback=True,
        top1_loc=AGENT_LARGE_TOP1_LOC_MIN + 500,
    )
    assert route(feats) == PatchGenStrategy.BOOTSTRAPPED_AGENT


def test_route_falls_through_when_short_issue():
    """Below word threshold → default."""
    feats = _features(
        words=AGENT_LONG_ISSUE_WORDS_MIN - 1,
        traceback=False,
        top1_loc=AGENT_LARGE_TOP1_LOC_MIN + 1_000,
    )
    assert route(feats) == PatchGenStrategy.BOOTSTRAPPED_AGENT


def test_route_falls_through_when_small_top1_file():
    """Top-1 file below the LOC threshold → default. Small files
    are well-suited to the seeded agent."""
    feats = _features(
        words=AGENT_LONG_ISSUE_WORDS_MIN + 100,
        traceback=False,
        top1_loc=AGENT_LARGE_TOP1_LOC_MIN - 1,
    )
    assert route(feats) == PatchGenStrategy.BOOTSTRAPPED_AGENT


def test_route_typical_dev50_instance_to_bootstrapped_agent():
    """The dev_50 median instance — short issue, no traceback,
    medium file — should land on BOOTSTRAPPED_AGENT."""
    feats = _features(
        words=200, traceback=False, top1_loc=600,
    )
    assert route(feats) == PatchGenStrategy.BOOTSTRAPPED_AGENT


def test_route_default_with_traceback_and_short_issue():
    """Tight pipeline-shape from old Rule 1 — now goes to default
    (BOOTSTRAPPED_AGENT) since PIPELINE_ONE_SHOT is dropped."""
    feats = _features(
        words=150, traceback=True, top1_loc=400,
    )
    assert route(feats) == PatchGenStrategy.BOOTSTRAPPED_AGENT


# ---------------------------------------------------------------------------
# 3. PIPELINE_ONE_SHOT explicitly removed (regression guard)
# ---------------------------------------------------------------------------


def test_pipeline_one_shot_strategy_does_not_exist():
    """P3d-fix removed PIPELINE_ONE_SHOT from the strategy enum.
    A future revival should be a deliberate choice, not a silent
    re-add. This test fails if anyone accidentally adds it back."""
    members = {s.name for s in PatchGenStrategy}
    assert "PIPELINE_ONE_SHOT" not in members
    assert members == {"AGENT", "BOOTSTRAPPED_AGENT"}


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_route_is_deterministic():
    """route(features) called twice with the same features returns
    the same strategy."""
    feats = _features(words=400, traceback=False, top1_loc=1_500)
    a = route(feats)
    b = route(feats)
    assert a is b


def test_route_strategy_enum_membership():
    """Every routing call returns a PatchGenStrategy member."""
    test_cases = [
        _features(words=100, traceback=True, top1_loc=500),
        _features(words=800, traceback=False, top1_loc=4_000),
        _features(words=300, traceback=False, top1_loc=2_000),
        _features(words=50, traceback=True, top1_loc=100),
    ]
    for feats in test_cases:
        result = route(feats)
        assert isinstance(result, PatchGenStrategy)


# ---------------------------------------------------------------------------
# 5. Firewall — router.py must not import repro / eval / memory modules
# ---------------------------------------------------------------------------


_ROUTER_PATH = PROJECT_ROOT / "harness" / "patch_gen" / "router.py"

_FORBIDDEN_IMPORT_MODULES = (
    "harness.repro",
    "harness.eval",
    "harness.memory",
)


def _imported_modules(py_path: pathlib.Path) -> set[str]:
    tree = ast.parse(py_path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def test_router_does_not_import_repro_eval_memory():
    """router.py is a pure dispatcher — no repro/eval/memory module
    imports allowed. AST scan, mirrors tests/test_repro_firewall.py
    pattern."""
    imported = _imported_modules(_ROUTER_PATH)
    bad: list[str] = []
    for mod in imported:
        for forbidden in _FORBIDDEN_IMPORT_MODULES:
            if mod == forbidden or mod.startswith(forbidden + "."):
                bad.append(mod)
    assert not bad, (
        f"router.py imports forbidden modules: {bad}. The router must "
        f"not depend on repro / eval / memory infrastructure."
    )


def test_router_imports_are_minimal():
    """Sanity check: the router only imports a handful of typed
    helpers + stdlib. If this set grows materially, audit the new
    deps for firewall and cost-creep concerns."""
    imported = _imported_modules(_ROUTER_PATH)
    expected_safe = {
        "__future__", "re", "dataclasses", "enum",
        "harness.localization_signals", "harness.views",
    }
    surprise = imported - expected_safe
    assert not surprise, (
        f"router.py picked up unexpected imports: {surprise}. If "
        f"intentional, update _expected_safe and audit for firewall."
    )

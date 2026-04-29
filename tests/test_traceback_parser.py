"""Tests for harness.traceback_parser — Stage 1c traceback signal source.

Covers:
  - regex parses the standard Python traceback format
  - regex parses GitHub-issue-rendered fragments (no header)
  - suffix matching emits ALL repo files with matching basename
  - frame priority: last frame gets rank 1
  - end-to-end emit_traceback_frames produces TracebackFrame signals
"""

from __future__ import annotations

import pytest

from harness.localization_signals import STAGE_1C_TRACEBACK
from harness.traceback_parser import (
    emit_traceback_frames,
    parse_tracebacks,
    rank_for_frame,
    suffix_match,
)
from harness.views import (
    ClassSummary,
    FileSummary,
    FunctionSummary,
    RepoSkeleton,
)


# ---------------------------------------------------------------------------
# Frame regex — standard Python traceback
# ---------------------------------------------------------------------------


def test_parse_standard_python_traceback() -> None:
    issue = """
The following crashes:

Traceback (most recent call last):
  File "/home/user/project/foo.py", line 42, in main
    bar()
  File "/home/user/project/lib/bar.py", line 100, in bar
    raise ValueError('boom')
ValueError: boom

Reproduce by running ./script.sh.
"""
    frames = parse_tracebacks(issue)
    assert len(frames) == 2
    assert frames[0].file_path_quoted == "/home/user/project/foo.py"
    assert frames[0].line_no == 42
    assert frames[0].function_name == "main"
    assert frames[0].frame_index == 0
    assert frames[1].file_path_quoted == "/home/user/project/lib/bar.py"
    assert frames[1].line_no == 100
    assert frames[1].function_name == "bar"
    assert frames[1].frame_index == 1


def test_parse_github_rendered_fragment_without_header() -> None:
    """GitHub issues often paste a frame without the leading
    'Traceback (most recent call last):' line."""
    issue = """When I call the function, I see:

  File "sympy/core/expr.py", line 234, in evalf
    return self._eval_evalf(prec)

This is unexpected.
"""
    frames = parse_tracebacks(issue)
    assert len(frames) == 1
    assert frames[0].file_path_quoted == "sympy/core/expr.py"
    assert frames[0].line_no == 234
    assert frames[0].function_name == "evalf"


def test_parse_truncated_path() -> None:
    """A frame with an ellipsis-truncated path should still parse;
    the parsed quoted path keeps the truncation marker for the
    matcher to handle."""
    issue = 'File ".../sympy/core/expr.py", line 234, in evalf'
    frames = parse_tracebacks(issue)
    assert len(frames) == 1
    assert frames[0].file_path_quoted == ".../sympy/core/expr.py"
    assert frames[0].normalized_quoted() == "sympy/core/expr.py"


def test_parse_frame_without_function_name() -> None:
    """Some traceback variants drop the ', in func' tail."""
    issue = 'File "foo.py", line 42'
    frames = parse_tracebacks(issue)
    assert len(frames) == 1
    assert frames[0].function_name == ""


def test_parse_no_traceback_returns_empty() -> None:
    issue = "Feature request: add a method that returns the current user."
    assert parse_tracebacks(issue) == []


def test_parse_multiple_tracebacks_in_one_issue() -> None:
    issue = """First example:
File "a.py", line 1, in f
  pass

Second example:
File "b.py", line 2, in g
  pass
"""
    frames = parse_tracebacks(issue)
    assert len(frames) == 2
    # frame_index continues across multiple tracebacks; frame_index=1
    # is the deepest signal we have.
    assert frames[1].file_path_quoted == "b.py"
    assert frames[1].frame_index == 1


def test_parse_excerpt_includes_following_indented_code_line() -> None:
    """The excerpt should include the next indented code line that
    typically follows the File "..." line in a Python traceback."""
    issue = '''  File "foo.py", line 42, in bar
    return self.compute()
'''
    frames = parse_tracebacks(issue)
    assert "return self.compute()" in frames[0].source_excerpt


# ---------------------------------------------------------------------------
# Suffix matching
# ---------------------------------------------------------------------------


def test_suffix_match_exact() -> None:
    candidates = ["sympy/core/expr.py", "sympy/utilities/iterables.py"]
    assert suffix_match("sympy/core/expr.py", candidates) == ["sympy/core/expr.py"]


def test_suffix_match_basename_emits_all() -> None:
    """Quoted = bare filename → match ALL candidates with that
    basename. Reranker disambiguates."""
    candidates = [
        "sympy/core/expr.py",
        "sympy/printing/expr.py",   # synthetic — illustrate multi-match
        "sympy/parsing/expr.py",
        "sympy/utilities/iterables.py",
    ]
    matches = suffix_match("expr.py", candidates)
    assert len(matches) == 3
    assert all("expr.py" in m for m in matches)


def test_suffix_match_partial_path() -> None:
    """Quoted = 'core/expr.py' → matches candidates ending with that."""
    candidates = ["sympy/core/expr.py", "sympy/printing/expr.py"]
    assert suffix_match("core/expr.py", candidates) == ["sympy/core/expr.py"]


def test_suffix_match_truncated_with_ellipsis() -> None:
    candidates = ["sympy/core/expr.py", "other/expr.py"]
    matches = suffix_match(".../sympy/core/expr.py", candidates)
    # After normalization "sympy/core/expr.py" → exact match
    assert "sympy/core/expr.py" in matches


def test_suffix_match_returns_empty_on_no_match() -> None:
    assert suffix_match("nonexistent.py", ["foo.py", "bar.py"]) == []


def test_suffix_match_dedupes() -> None:
    candidates = ["foo.py", "foo.py"]
    assert suffix_match("foo.py", candidates) == ["foo.py"]


# ---------------------------------------------------------------------------
# Frame priority / rank
# ---------------------------------------------------------------------------


def test_rank_for_frame_last_frame_gets_rank_1() -> None:
    from harness.localization_signals import TracebackFrame

    f_last = TracebackFrame(
        stage=STAGE_1C_TRACEBACK, file_path="x.py",
        line_no=10, function_name="foo", frame_index=2,
    )
    rank = rank_for_frame(
        f_last, n_frames_total=3, match_index_within_frame=0,
    )
    assert rank == 1


def test_rank_for_frame_outermost_gets_high_rank() -> None:
    from harness.localization_signals import TracebackFrame

    f_outer = TracebackFrame(
        stage=STAGE_1C_TRACEBACK, file_path="x.py",
        line_no=10, function_name="foo", frame_index=0,
    )
    rank = rank_for_frame(
        f_outer, n_frames_total=3,
        match_index_within_frame=0, max_matches_per_frame=30,
    )
    # distance_from_last = 2; rank = 2 * 30 + 0 + 1 = 61
    assert rank == 61


def test_rank_for_frame_multimatch_at_same_frame() -> None:
    from harness.localization_signals import TracebackFrame

    f = TracebackFrame(
        stage=STAGE_1C_TRACEBACK, file_path="x.py",
        line_no=10, function_name="foo", frame_index=2,
    )
    r0 = rank_for_frame(f, n_frames_total=3, match_index_within_frame=0)
    r1 = rank_for_frame(f, n_frames_total=3, match_index_within_frame=1)
    assert r1 == r0 + 1


# ---------------------------------------------------------------------------
# End-to-end: emit_traceback_frames with skeleton
# ---------------------------------------------------------------------------


def _skeleton_with(paths: list[str]) -> RepoSkeleton:
    return RepoSkeleton(
        repo="syn/syn",
        base_commit="0000",
        files=tuple(
            FileSummary(path=p, classes=(), functions=()) for p in paths
        ),
    )


def test_emit_traceback_frames_end_to_end() -> None:
    issue = '''Traceback (most recent call last):
  File "sympy/core/expr.py", line 234, in evalf
    return self._eval_evalf(prec)
  File "sympy/printing/lambdarepr.py", line 50, in _print_Symbol
    raise NotImplementedError
NotImplementedError
'''
    skeleton = _skeleton_with([
        "sympy/core/expr.py",
        "sympy/printing/lambdarepr.py",
        "sympy/utilities/iterables.py",
    ])
    signals = emit_traceback_frames(issue, skeleton)
    # Two frames, each matching exactly one repo file → 2 signals.
    assert len(signals) == 2
    assert {s.file_path for s in signals} == {
        "sympy/core/expr.py",
        "sympy/printing/lambdarepr.py",
    }


def test_emit_traceback_frames_multimatch_emits_all() -> None:
    """Issue quotes 'expr.py' (bare filename); skeleton has 3 files
    named expr.py — all three become signals."""
    issue = 'File "expr.py", line 234, in evalf'
    skeleton = _skeleton_with([
        "sympy/core/expr.py",
        "sympy/printing/expr.py",
        "sympy/parsing/expr.py",
        "sympy/utilities/iterables.py",
    ])
    signals = emit_traceback_frames(issue, skeleton)
    assert len(signals) == 3
    expr_paths = {s.file_path for s in signals}
    assert expr_paths == {
        "sympy/core/expr.py",
        "sympy/printing/expr.py",
        "sympy/parsing/expr.py",
    }


def test_emit_traceback_frames_preserves_excerpt_for_attribution() -> None:
    """Per-instance attribution: each emitted signal must carry the
    issue text where the frame was extracted."""
    issue = 'File "foo.py", line 42, in bar\n    code_line_here'
    skeleton = _skeleton_with(["pkg/foo.py"])
    signals = emit_traceback_frames(issue, skeleton)
    assert len(signals) == 1
    assert "File" in signals[0].excerpt
    assert "foo.py" in signals[0].excerpt
    assert signals[0].file_path == "pkg/foo.py"


def test_emit_traceback_frames_no_traceback_in_issue() -> None:
    issue = "feature: add a foo() method"
    skeleton = _skeleton_with(["x.py"])
    assert emit_traceback_frames(issue, skeleton) == []


def test_emit_traceback_frames_empty_skeleton() -> None:
    issue = 'File "foo.py", line 42, in bar'
    skeleton = _skeleton_with([])
    assert emit_traceback_frames(issue, skeleton) == []

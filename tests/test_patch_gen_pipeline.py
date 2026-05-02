"""Phase 3 P3b — pipeline-path unit tests.

Covers:

  - PatchCandidate / PatchGenContext / FileSnippet schema validation
    (frozen dataclasses + FORBIDDEN_TOKENS via _assert_no_forbidden_field_names).
  - build_patch_gen_context_with_superset_check:
      * happy path (top-K files read into FileSnippet[])
      * truncation when file > per_file_char_cap
      * ContextOversizeError when projected_token_count > limit
      * ContextSupersetError when repro_context_files ⊄ patch files
      * superset assertion bypass when repro_context_files=None
  - generate_pipeline_one_shot with mocked complete_chat:
      * parses {"rationale": ..., "diff": ...} → PatchCandidate
      * tolerates accidental markdown fences
      * raises PatchGenError on missing/empty diff
      * candidate_id correctness across temperatures
  - generate_pipeline (K-orchestrator) with mocked complete_chat:
      * runs K candidates at K temperatures
      * cost cap stops the loop and reports cost_cap_hit=True
      * one parse failure does not abort the K-loop

No real LLM calls; no real Docker. ``_FakeSandbox`` provides
``read_file`` only.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from harness.localization_signals import RankedFile
from harness.patch_gen import (
    ContextOversizeError,
    DEFAULT_PER_FILE_CHAR_CAP,
    DEFAULT_PROJECTED_TOKEN_LIMIT,
    FileSnippet,
    PatchCandidate,
    PatchGenContext,
    PatchGenError,
    build_patch_gen_context_with_superset_check,
    generate_pipeline,
    generate_pipeline_one_shot,
)
from harness.patch_gen.pipeline import _candidate_id_for_temperature
from harness.patch_gen.views import ContextSupersetError
from harness.views import (
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeExecResult:
    exit_code: int
    stdout: str
    stderr: str = ""


class _FakeSandbox:
    """Just enough Sandbox surface for context.py — only ``read_file``
    is exercised. The wrapper holds an in-memory map ``path -> content``."""

    def __init__(self, files: dict[str, str]):
        self._files = files

    def read_file(self, path: str, max_chars: int | None = None) -> _FakeExecResult:
        if path not in self._files:
            return _FakeExecResult(exit_code=1, stdout="")
        content = self._files[path]
        if max_chars is not None and len(content) > max_chars:
            content = content[:max_chars]
        return _FakeExecResult(exit_code=0, stdout=content)


def _make_view(instance_id: str = "fake__example-1") -> InstanceView:
    return InstanceView(
        instance_id=instance_id,
        repo="example/example",
        base_commit="abcdef0123",
        problem_statement="Bug: foo() returns None instead of raising",
        repo_skeleton=RepoSkeleton(
            repo="example/example",
            base_commit="abcdef0123",
        ),
        test_directives=TestDirectives(dirs=("tests/",), source="discovery:1"),
    )


def _make_ranked_files(paths: list[str]) -> list[RankedFile]:
    return [
        RankedFile(
            file_path=p,
            final_score=1.0 - i * 0.05,
            rationale=f"r{i}",
            upstream_signals=("bm25",),
            upstream_best_rank={"bm25": i + 1},
        )
        for i, p in enumerate(paths)
    ]


# ---------------------------------------------------------------------------
# 1. View-shape validation
# ---------------------------------------------------------------------------


def test_patch_candidate_required_fields():
    pc = PatchCandidate(
        instance_id="x__y-1",
        candidate_id="pipe_t0",
        diff="--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n",
        source_route="pipeline",
        source_temperature=0.0,
        source_attempt_index=0,
        generator_model="deepseek-chat",
        generator_input_tokens=10,
        generator_output_tokens=5,
        generation_cost_usd=0.001,
        duration_s=1.0,
    )
    assert pc.candidate_id == "pipe_t0"
    assert pc.source_route == "pipeline"


def test_patch_candidate_rejects_empty_diff():
    with pytest.raises(ValueError, match="diff required"):
        PatchCandidate(
            instance_id="x__y-1",
            candidate_id="pipe_t0",
            diff="",
            source_route="pipeline",
            source_temperature=0.0,
            source_attempt_index=0,
            generator_model="m", generator_input_tokens=0,
            generator_output_tokens=0, generation_cost_usd=0.0, duration_s=0.0,
        )


def test_patch_candidate_rejects_unknown_route():
    with pytest.raises(ValueError, match="source_route"):
        PatchCandidate(
            instance_id="x__y-1", candidate_id="x", diff="d",
            source_route="WAT", source_temperature=0.0,
            source_attempt_index=0, generator_model="m",
            generator_input_tokens=0, generator_output_tokens=0,
            generation_cost_usd=0.0, duration_s=0.0,
        )


def test_patch_candidate_pipeline_requires_temperature():
    with pytest.raises(ValueError, match="temperature"):
        PatchCandidate(
            instance_id="x__y-1", candidate_id="x", diff="d",
            source_route="pipeline", source_temperature=None,
            source_attempt_index=0, generator_model="m",
            generator_input_tokens=0, generator_output_tokens=0,
            generation_cost_usd=0.0, duration_s=0.0,
        )


def test_patch_gen_context_validates_tuple_types():
    with pytest.raises(TypeError, match="test_directives"):
        PatchGenContext(
            instance_id="x", problem_statement="p",
            test_directives=["tests/"],   # type: ignore[arg-type]
            files=tuple(), projected_token_count=0,
        )


def test_patch_gen_context_file_paths_property():
    snip = FileSnippet(path="a.py", content="x", truncated=False, final_score=1.0, rationale="")
    ctx = PatchGenContext(
        instance_id="x", problem_statement="p",
        test_directives=("tests/",),
        files=(snip,), projected_token_count=10,
    )
    assert ctx.file_paths == ("a.py",)


# ---------------------------------------------------------------------------
# 2. Context builder — happy path + edge cases
# ---------------------------------------------------------------------------


def test_build_context_happy_path():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py", "src/bar.py"])
    sb = _FakeSandbox({
        "src/foo.py": "def foo():\n    return None\n",
        "src/bar.py": "def bar():\n    return 1\n",
    })
    ctx = build_patch_gen_context_with_superset_check(
        view=view, ranked_files=rfs, sandbox=sb,
    )
    assert ctx.instance_id == view.instance_id
    assert ctx.problem_statement == view.problem_statement
    assert ctx.test_directives == ("tests/",)
    assert len(ctx.files) == 2
    assert ctx.files[0].path == "src/foo.py"
    assert ctx.files[0].truncated is False
    assert ctx.projected_token_count > 0


def test_build_context_truncates_long_file():
    view = _make_view()
    rfs = _make_ranked_files(["src/big.py"])
    big = "x" * (DEFAULT_PER_FILE_CHAR_CAP * 2)
    sb = _FakeSandbox({"src/big.py": big})
    ctx = build_patch_gen_context_with_superset_check(
        view=view, ranked_files=rfs, sandbox=sb,
    )
    assert ctx.files[0].truncated is True
    assert len(ctx.files[0].content) == DEFAULT_PER_FILE_CHAR_CAP


def test_build_context_drops_unreadable_file_but_keeps_others():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py", "src/missing.py", "src/bar.py"])
    sb = _FakeSandbox({
        "src/foo.py": "def foo(): pass\n",
        "src/bar.py": "def bar(): pass\n",
        # src/missing.py is intentionally absent -> exit_code=1, dropped
    })
    ctx = build_patch_gen_context_with_superset_check(
        view=view, ranked_files=rfs, sandbox=sb,
    )
    paths = [f.path for f in ctx.files]
    assert paths == ["src/foo.py", "src/bar.py"]


def test_build_context_oversize_raises():
    view = _make_view()
    # Make 10 files each ~50K chars (after truncation) → ~500K chars
    # ≈ ~125K tokens. Set a tiny limit to trigger the gate.
    rfs = _make_ranked_files([f"src/f{i}.py" for i in range(10)])
    big = "x" * 50_000
    sb = _FakeSandbox({f"src/f{i}.py": big for i in range(10)})
    with pytest.raises(ContextOversizeError):
        build_patch_gen_context_with_superset_check(
            view=view, ranked_files=rfs, sandbox=sb,
            projected_token_limit=1_000,
        )


def test_build_context_superset_assertion_passes_when_subset():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py", "src/bar.py", "src/baz.py"])
    sb = _FakeSandbox({
        "src/foo.py": "x", "src/bar.py": "y", "src/baz.py": "z",
    })
    ctx = build_patch_gen_context_with_superset_check(
        view=view, ranked_files=rfs, sandbox=sb,
        repro_context_files=("src/foo.py",),  # subset
    )
    assert "src/foo.py" in ctx.file_paths


def test_build_context_superset_assertion_fires_when_repro_has_extra():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox({"src/foo.py": "x"})
    with pytest.raises(ContextSupersetError, match="missing files"):
        build_patch_gen_context_with_superset_check(
            view=view, ranked_files=rfs, sandbox=sb,
            repro_context_files=("src/foo.py", "src/missing_for_repro.py"),
        )


def test_build_context_none_repro_files_skips_assertion():
    """repro_context_files=None bypasses the assertion (Phase 2 wasn't
    run for this instance, or the operator chose to bypass)."""
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox({"src/foo.py": "x"})
    ctx = build_patch_gen_context_with_superset_check(
        view=view, ranked_files=rfs, sandbox=sb,
        repro_context_files=None,
    )
    assert ctx.repro_context_files is None


def test_build_context_empty_ranked_files_raises():
    view = _make_view()
    sb = _FakeSandbox({})
    with pytest.raises(ValueError, match="ranked_files empty"):
        build_patch_gen_context_with_superset_check(
            view=view, ranked_files=[], sandbox=sb,
        )


def test_build_context_no_readable_files_raises():
    view = _make_view()
    rfs = _make_ranked_files(["src/missing.py"])
    sb = _FakeSandbox({})
    with pytest.raises(ValueError, match="no readable files"):
        build_patch_gen_context_with_superset_check(
            view=view, ranked_files=rfs, sandbox=sb,
        )


# ---------------------------------------------------------------------------
# 3. candidate_id construction
# ---------------------------------------------------------------------------


def test_candidate_id_temperature_zero():
    assert _candidate_id_for_temperature(0.0) == "pipe_t0"


def test_candidate_id_temperature_half():
    assert _candidate_id_for_temperature(0.5) == "pipe_t05"


def test_candidate_id_temperature_seven():
    assert _candidate_id_for_temperature(0.7) == "pipe_t07"


def test_candidate_id_temperature_one():
    assert _candidate_id_for_temperature(1.0) == "pipe_t1"


def test_candidate_id_distinct_across_typical_K4():
    ids = {_candidate_id_for_temperature(t) for t in (0.0, 0.3, 0.7, 1.0)}
    assert len(ids) == 4


# ---------------------------------------------------------------------------
# 4. Single-shot generator with mocked LLM
# ---------------------------------------------------------------------------


@dataclass
class _FakeChat:
    text: str
    model: str = "deepseek-chat"
    input_tokens: int = 100
    output_tokens: int = 50


def _ctx_for_single_shot() -> PatchGenContext:
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox({"src/foo.py": "def foo(): return None\n"})
    return build_patch_gen_context_with_superset_check(
        view=view, ranked_files=rfs, sandbox=sb,
    )


def test_generate_pipeline_one_shot_parses_response():
    ctx = _ctx_for_single_shot()
    chat_text = '{"rationale": "Replace return None with raise.", "diff": "--- a\\n+++ b\\n@@ -1 +1 @@\\n-return None\\n+raise ValueError\\n"}'
    with patch("harness.llm.clients.complete_chat", return_value=_FakeChat(text=chat_text)):
        cand = generate_pipeline_one_shot(ctx=ctx, temperature=0.0, attempt_index=0)
    assert cand.candidate_id == "pipe_t0"
    assert cand.source_route == "pipeline"
    assert cand.source_temperature == 0.0
    assert cand.source_attempt_index == 0
    assert "return None" in cand.diff


def test_generate_pipeline_one_shot_tolerates_markdown_fence():
    ctx = _ctx_for_single_shot()
    chat_text = '```json\n{"rationale": "x", "diff": "diff data"}\n```'
    with patch("harness.llm.clients.complete_chat", return_value=_FakeChat(text=chat_text)):
        cand = generate_pipeline_one_shot(ctx=ctx, temperature=0.5, attempt_index=1)
    assert cand.candidate_id == "pipe_t05"


def test_generate_pipeline_one_shot_raises_on_missing_diff():
    ctx = _ctx_for_single_shot()
    chat_text = '{"rationale": "x"}'
    with patch("harness.llm.clients.complete_chat", return_value=_FakeChat(text=chat_text)):
        with pytest.raises(PatchGenError, match="missing key"):
            generate_pipeline_one_shot(ctx=ctx, temperature=0.0, attempt_index=0)


def test_generate_pipeline_one_shot_raises_on_empty_diff():
    ctx = _ctx_for_single_shot()
    chat_text = '{"rationale": "x", "diff": "   "}'
    with patch("harness.llm.clients.complete_chat", return_value=_FakeChat(text=chat_text)):
        with pytest.raises(PatchGenError, match="empty"):
            generate_pipeline_one_shot(ctx=ctx, temperature=0.0, attempt_index=0)


def test_generate_pipeline_one_shot_raises_on_invalid_json():
    ctx = _ctx_for_single_shot()
    chat_text = "this is not JSON"
    with patch("harness.llm.clients.complete_chat", return_value=_FakeChat(text=chat_text)):
        with pytest.raises(PatchGenError, match="not valid JSON"):
            generate_pipeline_one_shot(ctx=ctx, temperature=0.0, attempt_index=0)


# ---------------------------------------------------------------------------
# 5. K-orchestrator
# ---------------------------------------------------------------------------


def _patch_gen_setup():
    view = _make_view()
    rfs = _make_ranked_files(["src/foo.py"])
    sb = _FakeSandbox({"src/foo.py": "def foo(): return None\n"})
    return view, rfs, sb


def test_generate_pipeline_k2_runs_two_candidates():
    view, rfs, sb = _patch_gen_setup()
    chat_text = '{"rationale": "r", "diff": "diff data"}'
    with patch("harness.llm.clients.complete_chat", return_value=_FakeChat(text=chat_text)):
        result = generate_pipeline(
            view=view, ranked_files=rfs, sandbox=sb,
            temperatures=(0.0, 0.5),
        )
    assert len(result.candidates) == 2
    assert result.candidates[0].candidate_id == "pipe_t0"
    assert result.candidates[1].candidate_id == "pipe_t05"
    assert result.cost_cap_hit is False
    assert result.parse_failures == ()


def test_generate_pipeline_cost_cap_stops_loop():
    """With expensive tokens in the fake chat result, cost_cap_usd
    triggers between attempts and the loop stops."""
    view, rfs, sb = _patch_gen_setup()
    # 1M output tokens at deepseek-chat output rate ($1.10/1M) = $1.10
    expensive = _FakeChat(text='{"rationale": "r", "diff": "d"}', input_tokens=0, output_tokens=1_000_000)
    with patch("harness.llm.clients.complete_chat", return_value=expensive):
        result = generate_pipeline(
            view=view, ranked_files=rfs, sandbox=sb,
            temperatures=(0.0, 0.5, 0.7, 1.0),
            cost_cap_usd=1.0,
        )
    # First attempt exceeds cap; second iteration sees hard_cap_reached.
    assert len(result.candidates) == 1
    assert result.cost_cap_hit is True


def test_generate_pipeline_parse_failure_does_not_abort_k_loop():
    """One bad response shouldn't kill the whole K-loop. Other
    temperatures still produce candidates."""
    view, rfs, sb = _patch_gen_setup()
    bad = _FakeChat(text="not json")
    good = _FakeChat(text='{"rationale": "r", "diff": "diff data"}')
    call_n = {"i": 0}
    def _alternate(*args, **kwargs):
        call_n["i"] += 1
        return bad if call_n["i"] == 1 else good
    with patch("harness.llm.clients.complete_chat", side_effect=_alternate):
        result = generate_pipeline(
            view=view, ranked_files=rfs, sandbox=sb,
            temperatures=(0.0, 0.5),
        )
    assert len(result.candidates) == 1
    assert result.candidates[0].candidate_id == "pipe_t05"
    assert len(result.parse_failures) == 1


# ---------------------------------------------------------------------------
# 6. Defense in depth — view dataclasses use the FORBIDDEN_TOKENS firewall
# ---------------------------------------------------------------------------


def test_patch_candidate_field_names_pass_firewall():
    """If a future field rename matches a forbidden token, this fires
    via _assert_no_forbidden_field_names called from __post_init__."""
    pc = PatchCandidate(
        instance_id="x__y-1", candidate_id="x", diff="d",
        source_route="pipeline", source_temperature=0.0,
        source_attempt_index=0, generator_model="m",
        generator_input_tokens=0, generator_output_tokens=0,
        generation_cost_usd=0.0, duration_s=0.0,
    )
    # If __post_init__ raised, we wouldn't be here.
    assert pc is not None

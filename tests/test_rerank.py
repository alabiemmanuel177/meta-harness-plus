"""Tests for harness.rerank — Stage 1g LLM rerank.

Covers prompt construction (per-strategy ranks visible), token-budget
truncation, JSON output parsing (including malformed/fenced output),
and trajectory wiring via a mocked LLM client.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from harness.localization_signals import (
    RankedFile,
    RerankerOutput,
    SIGNAL_CLASSES,
    UPSTREAM_SIGNAL_KINDS,
)
from harness.rerank import (
    CandidateForRerank,
    DEFAULT_PROMPT_TOKEN_CAP,
    RerankerError,
    RerankerInput,
    SYSTEM_PROMPT,
    _truncate_candidates_to_budget,
    build_user_prompt,
    parse_reranker_output,
    rerank,
)
from harness.views import (
    InstanceView,
    RepoSkeleton,
    TestDirectives,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_view(issue: str = "Some bug.") -> InstanceView:
    return InstanceView(
        instance_id="t__t-1",
        repo="t/t",
        base_commit="abc1234567890",
        problem_statement=issue,
        repo_skeleton=RepoSkeleton(repo="t/t", base_commit="abc1234567890"),
        test_directives=TestDirectives(dirs=("tests/",), source="test:smoke"),
    )


def _make_candidate(
    path: str,
    *,
    bm25_para_rank: int | None = None,
    embedding_rank: int | None = None,
    traceback_rank: int | None = None,
    summary: str = "(no summary)",
    n_lines: int = 100,
) -> CandidateForRerank:
    ranks: dict = {}
    scores: dict = {}
    if bm25_para_rank is not None:
        ranks["bm25_first_paragraph"] = bm25_para_rank
        scores["bm25_first_paragraph"] = 1.0 / bm25_para_rank
    if embedding_rank is not None:
        ranks["embedding"] = embedding_rank
        scores["embedding"] = 0.9 / embedding_rank
    if traceback_rank is not None:
        ranks["traceback"] = traceback_rank
        scores["traceback"] = 1.0 / traceback_rank
    return CandidateForRerank(
        file_path=path,
        per_strategy_rank=ranks,
        per_strategy_score=scores,
        file_summary=summary,
        n_lines=n_lines,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_user_prompt_includes_per_strategy_ranks() -> None:
    """Each candidate's prompt row must show per-strategy ranks
    explicitly. This is the load-bearing change for Stage 1g."""
    view = _make_view("Test issue with broken thing.")
    cands = (
        _make_candidate("a/forms.py", bm25_para_rank=1, embedding_rank=7,
                        summary="classes: UserCreationForm"),
        _make_candidate("a/models.py", bm25_para_rank=12, embedding_rank=1,
                        summary="classes: User"),
    )
    inp = RerankerInput(view=view, candidates=cands, top_k=2)
    prompt = build_user_prompt(inp)
    # Per-strategy rank table must be visible.
    assert "bm25_first_paragraph=1" in prompt
    assert "embedding=7" in prompt
    assert "bm25_first_paragraph=12" in prompt
    assert "embedding=1" in prompt
    # File paths and summaries present.
    assert "a/forms.py" in prompt
    assert "a/models.py" in prompt
    assert "UserCreationForm" in prompt


def test_system_prompt_describes_strategy_kinds() -> None:
    """The system prompt must list the strategy kinds + collapse rules
    so the model emits valid upstream_signals subsets."""
    assert "bm25_first_paragraph" in SYSTEM_PROMPT
    assert "bm25_full_issue" in SYSTEM_PROMPT
    assert "embedding" in SYSTEM_PROMPT
    assert "traceback" in SYSTEM_PROMPT
    # Collapse rule.
    assert "collapse" in SYSTEM_PROMPT.lower() or "drop the strategy suffix" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Token budget truncation
# ---------------------------------------------------------------------------


def test_truncation_keeps_priority_candidates() -> None:
    """When the candidate count exceeds budget, top-priority (lowest
    best per-strategy rank) candidates are kept."""
    issue = "x" * 200_000  # forces aggressive truncation
    cands = tuple(
        _make_candidate(f"f{i}.py", bm25_para_rank=i + 1, embedding_rank=50)
        for i in range(200)
    )
    truncated = _truncate_candidates_to_budget(cands, issue, cap_tokens=60_000)
    assert len(truncated) < len(cands)
    # The candidate with rank=1 must survive.
    survivors = [c.file_path for c in truncated]
    assert "f0.py" in survivors


def test_truncation_no_op_when_under_budget() -> None:
    cands = tuple(
        _make_candidate(f"f{i}.py", bm25_para_rank=i + 1) for i in range(10)
    )
    truncated = _truncate_candidates_to_budget(
        cands, "small issue", cap_tokens=DEFAULT_PROMPT_TOKEN_CAP,
    )
    assert truncated == cands


# ---------------------------------------------------------------------------
# JSON output parsing
# ---------------------------------------------------------------------------


def test_parse_clean_json() -> None:
    text = json.dumps({
        "ranked": [
            {
                "file_path": "a.py", "final_score": 0.9,
                "rationale": "issue mentions Foo",
                "upstream_signals": ["bm25", "embedding"],
                "upstream_best_rank": {"bm25": 1, "embedding": 3},
            },
        ],
    })
    out = parse_reranker_output(text, top_k=5)
    assert len(out) == 1
    assert isinstance(out[0], RankedFile)
    assert out[0].file_path == "a.py"
    assert out[0].upstream_best_rank == {"bm25": 1, "embedding": 3}


def test_parse_handles_json_fence() -> None:
    text = '```json\n{"ranked": [{"file_path": "a.py", "final_score": 0.5, "rationale": "r", "upstream_signals": ["bm25"]}]}\n```'
    out = parse_reranker_output(text, top_k=5)
    assert len(out) == 1
    assert out[0].file_path == "a.py"


def test_parse_handles_trailing_prose() -> None:
    text = '{"ranked": [{"file_path": "a.py", "final_score": 0.5, "rationale": "r", "upstream_signals": ["bm25"]}]}\n\nNote: I picked this file.'
    out = parse_reranker_output(text, top_k=5)
    assert len(out) == 1


def test_parse_filters_unknown_signal_kinds() -> None:
    """Model hallucinations of unknown upstream signals are filtered
    rather than blowing up the whole rerank result."""
    text = json.dumps({
        "ranked": [
            {
                "file_path": "a.py", "final_score": 0.9,
                "rationale": "r",
                "upstream_signals": ["bm25", "made_up_signal", "embedding"],
                "upstream_best_rank": {"bm25": 1, "embedding": 5, "fictional": 99},
            },
        ],
    })
    out = parse_reranker_output(text, top_k=5)
    assert "made_up_signal" not in out[0].upstream_signals
    assert "fictional" not in (out[0].upstream_best_rank or {})
    assert "bm25" in out[0].upstream_signals
    assert "embedding" in out[0].upstream_signals


def test_parse_raises_on_malformed_json() -> None:
    with pytest.raises(ValueError):
        parse_reranker_output("not JSON at all", top_k=5)


def test_parse_raises_on_missing_ranked_key() -> None:
    with pytest.raises(ValueError, match="missing 'ranked'"):
        parse_reranker_output('{"something_else": []}', top_k=5)


def test_parse_truncates_to_top_k() -> None:
    text = json.dumps({
        "ranked": [
            {"file_path": f"f{i}.py", "final_score": 0.5, "rationale": "r",
             "upstream_signals": ["bm25"]}
            for i in range(10)
        ],
    })
    out = parse_reranker_output(text, top_k=3)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Mocked end-to-end rerank: trajectory wiring + retry-on-malformed
# ---------------------------------------------------------------------------


def test_rerank_emits_typed_signal_to_trajectory(tmp_path) -> None:
    """rerank() with a mocked LLM call returns a RerankResult and
    writes the RerankerOutput signal via trajectory_writer.write_signal."""
    view = _make_view()
    cands = (
        _make_candidate("a.py", bm25_para_rank=1, embedding_rank=2),
        _make_candidate("b.py", bm25_para_rank=5),
    )
    inp = RerankerInput(view=view, candidates=cands, top_k=2)

    canned_response = json.dumps({
        "ranked": [
            {"file_path": "a.py", "final_score": 0.95, "rationale": "rank 1 in para",
             "upstream_signals": ["bm25", "embedding"],
             "upstream_best_rank": {"bm25": 1, "embedding": 2}},
            {"file_path": "b.py", "final_score": 0.40, "rationale": "rank 5",
             "upstream_signals": ["bm25"],
             "upstream_best_rank": {"bm25": 5}},
        ],
    })

    fake_chat = MagicMock(text=canned_response, model="deepseek-chat",
                           input_tokens=1000, output_tokens=200)

    captured_signals: list = []
    class FakeWriter:
        def write_signal(self, sig):
            captured_signals.append(sig)
        def write(self, *a, **kw):
            pass

    with patch("harness.llm.clients.complete_chat", return_value=fake_chat), \
         patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"):
        result = rerank(inp, trajectory_writer=FakeWriter())

    assert len(result.ranked_files) == 2
    assert result.ranked_files[0].file_path == "a.py"
    assert result.ranked_files[0].final_score == 0.95
    # Trajectory got a typed RerankerOutput.
    assert len(captured_signals) == 1
    assert isinstance(captured_signals[0], RerankerOutput)
    assert captured_signals[0].model == "deepseek-chat"
    assert captured_signals[0].input_tokens == 1000


def test_rerank_retries_on_malformed_then_raises() -> None:
    """Malformed first response → one retry; second malformed → RerankerError."""
    view = _make_view()
    cands = (_make_candidate("a.py", bm25_para_rank=1),)
    inp = RerankerInput(view=view, candidates=cands, top_k=1)

    bad_chat = MagicMock(text="not JSON", model="deepseek-chat",
                          input_tokens=10, output_tokens=10)

    with patch("harness.llm.clients.complete_chat", return_value=bad_chat), \
         patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"):
        with pytest.raises(RerankerError):
            rerank(inp, n_retries=1)


def test_rerank_recovers_after_one_bad_then_good() -> None:
    """First response malformed, second valid → success."""
    view = _make_view()
    cands = (_make_candidate("a.py", bm25_para_rank=1),)
    inp = RerankerInput(view=view, candidates=cands, top_k=1)

    good_text = json.dumps({
        "ranked": [
            {"file_path": "a.py", "final_score": 0.9, "rationale": "ok",
             "upstream_signals": ["bm25"]},
        ],
    })

    bad_chat = MagicMock(text="bad", model="deepseek-chat",
                          input_tokens=10, output_tokens=10)
    good_chat = MagicMock(text=good_text, model="deepseek-chat",
                           input_tokens=20, output_tokens=20)

    with patch("harness.llm.clients.complete_chat",
               side_effect=[bad_chat, good_chat]), \
         patch("harness.llm.clients.model_for_role", return_value="deepseek-chat"):
        result = rerank(inp, n_retries=1)
    assert len(result.ranked_files) == 1
    assert result.ranked_files[0].file_path == "a.py"

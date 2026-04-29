"""Tests for harness.localization_signals — the structured schema for
Phase 1 localization signals.

The signals are typed frozen dataclasses; this test confirms:

  - Each signal class has no forbidden field names.
  - Each constructs cleanly with a valid example.
  - ``as_dict()`` returns JSON-round-trippable dicts with the
    ``_signal`` discriminator.
  - The trajectory writer's ``write_signal`` method records each
    type and rejects non-signal arguments.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from harness import localization_signals as ls
from harness.localization_signals import (
    BM25Hit,
    DepGraphHop,
    EmbeddingHit,
    GitArchaeologyHit,
    RerankerOutput,
    SIGNAL_CLASSES,
    SymbolGrepHit,
    TracebackFrame,
    STAGE_1B_FILE_CANDIDATES,
    STAGE_1C_TRACEBACK,
    STAGE_1D_SYMBOL_GREP,
    STAGE_1E_GIT_ARCHEOLOGY,
    STAGE_1F_DEP_GRAPH,
    STAGE_1G_RERANK,
)
from harness.trajectory import TrajectoryWriter, trajectory


# ---------------------------------------------------------------------------
# Field-name validation
# ---------------------------------------------------------------------------


def test_all_signal_classes_have_clean_field_names() -> None:
    """No signal class has a field whose name matches a forbidden token.
    The dataclass __post_init__ runs the same validator the views use."""
    from dataclasses import fields
    from harness.views import FORBIDDEN_TOKENS

    bad: list[str] = []
    for cls in SIGNAL_CLASSES:
        for f in fields(cls):
            for tok in FORBIDDEN_TOKENS:
                if tok in f.name.casefold():
                    bad.append(f"{cls.__name__}.{f.name}")
    assert not bad, f"signal classes with forbidden field names: {bad}"


# ---------------------------------------------------------------------------
# Per-class construction smoke
# ---------------------------------------------------------------------------


def test_bm25_hit_constructs_and_serializes() -> None:
    h = BM25Hit(
        stage=STAGE_1B_FILE_CANDIDATES,
        file_path="django/contrib/auth/forms.py",
        score=12.4,
        rank=1,
        query_basis="issue_text",
    )
    d = h.as_dict()
    assert d["_signal"] == "BM25Hit"
    assert d["file_path"].endswith("forms.py")
    json.dumps(d)  # must be JSON-serializable


def test_embedding_hit_constructs_and_serializes() -> None:
    h = EmbeddingHit(
        stage=STAGE_1B_FILE_CANDIDATES,
        file_path="django/contrib/auth/forms.py",
        cosine_similarity=0.81,
        rank=2,
        model="text-embedding-3-large",
        chunk_basis="whole_file",
    )
    json.dumps(h.as_dict())


def test_traceback_frame_constructs() -> None:
    f = TracebackFrame(
        stage=STAGE_1C_TRACEBACK,
        file_path="django/db/models/manager.py",
        line_no=120,
        function_name="manager_from_queryset",
        frame_index=2,
        excerpt="    obj = queryset.get(pk=pk)",
    )
    json.dumps(f.as_dict())


def test_symbol_grep_hit_constructs() -> None:
    h = SymbolGrepHit(
        stage=STAGE_1D_SYMBOL_GREP,
        needle="ImproperlyConfigured",
        file_path="django/core/exceptions.py",
        line_no=44,
        line_text="class ImproperlyConfigured(Exception):",
        rarity_score=0.93,
    )
    json.dumps(h.as_dict())


def test_git_archaeology_hit_constructs() -> None:
    h = GitArchaeologyHit(
        stage=STAGE_1E_GIT_ARCHEOLOGY,
        file_path="django/contrib/auth/forms.py",
        sha="aabbccddeeff",
        parent_sha="0011223344",
        commit_message_summary="fix: clear non-empty-line whitespace on UserCreationForm",
        fix_signal=True,
        age_days=400,
        files_changed=2,
    )
    json.dumps(h.as_dict())


def test_dep_graph_hop_constructs() -> None:
    h = DepGraphHop(
        stage=STAGE_1F_DEP_GRAPH,
        source_file="django/contrib/auth/forms.py",
        target_file="django/contrib/auth/models.py",
        edge_type="imports",
        hop_distance=1,
        confidence=0.95,
    )
    json.dumps(h.as_dict())


def test_reranker_output_constructs() -> None:
    r = RerankerOutput(
        stage=STAGE_1G_RERANK,
        model="claude-sonnet-4-6",
        input_tokens=42_000,
        output_tokens=1_200,
        prompt_token_count=42_000,
        ranked_files=(
            ("django/contrib/auth/forms.py", 0.92, "issue mentions UserCreationForm directly"),
            ("django/contrib/auth/models.py", 0.71, "imported by forms.py; defines User"),
        ),
        skeleton_compression="symbol_skeleton",
    )
    d = r.as_dict()
    json.dumps(d, default=str)
    assert d["_signal"] == "RerankerOutput"
    assert len(d["ranked_files"]) == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_bm25_hit_rejects_zero_rank() -> None:
    with pytest.raises(ValueError):
        BM25Hit(
            stage=STAGE_1B_FILE_CANDIDATES,
            file_path="x.py", score=1.0, rank=0, query_basis="issue_text",
        )


def test_dep_graph_hop_rejects_zero_hop() -> None:
    with pytest.raises(ValueError):
        DepGraphHop(
            stage=STAGE_1F_DEP_GRAPH,
            source_file="a.py", target_file="b.py",
            edge_type="imports", hop_distance=0, confidence=0.5,
        )


def test_reranker_output_rejects_malformed_ranked_files() -> None:
    with pytest.raises(ValueError):
        RerankerOutput(
            stage=STAGE_1G_RERANK,
            model="claude-sonnet-4-6",
            input_tokens=10, output_tokens=10, prompt_token_count=10,
            ranked_files=(("x.py", 0.5),),  # 2-tuple, must be 3
            skeleton_compression="full",
        )


# ---------------------------------------------------------------------------
# Trajectory wiring
# ---------------------------------------------------------------------------


def test_trajectory_write_signal_records_typed_record(tmp_path: pathlib.Path) -> None:
    with trajectory(tmp_path, "test_inst", candidate_id="phase1") as tw:
        tw.write_signal(BM25Hit(
            stage=STAGE_1B_FILE_CANDIDATES,
            file_path="x.py", score=1.5, rank=1, query_basis="issue_text",
        ))
        tw.write_signal(SymbolGrepHit(
            stage=STAGE_1D_SYMBOL_GREP,
            needle="Foo", file_path="x.py", line_no=10,
            line_text="class Foo:", rarity_score=0.5,
        ))

    files = list((tmp_path / "test_inst" / "phase1").glob("turn_*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(l) for l in files[0].read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["event"] == "signal_BM25Hit"
    assert lines[0]["payload"]["_signal"] == "BM25Hit"
    assert lines[0]["payload"]["file_path"] == "x.py"
    assert lines[1]["event"] == "signal_SymbolGrepHit"


def test_trajectory_write_signal_rejects_non_signal(tmp_path: pathlib.Path) -> None:
    with trajectory(tmp_path, "test_inst", candidate_id="phase1") as tw:
        with pytest.raises(TypeError):
            tw.write_signal({"not": "a signal"})


def test_all_signal_classes_round_trip_through_json(tmp_path: pathlib.Path) -> None:
    """Every signal class must serialize via as_dict() and round-trip
    through json.dumps + json.loads with no information loss for
    primitive fields. (Tuples become lists on round-trip; that's
    expected JSON behavior.)
    """
    samples = {
        "BM25Hit": BM25Hit(stage="1b", file_path="a", score=1.0, rank=1, query_basis="issue_text"),
        "EmbeddingHit": EmbeddingHit(stage="1b", file_path="a", cosine_similarity=0.5,
                                      rank=1, model="x", chunk_basis="whole_file"),
        "TracebackFrame": TracebackFrame(stage="1c", file_path="a", line_no=1,
                                          function_name="f", frame_index=0),
        "SymbolGrepHit": SymbolGrepHit(stage="1d", needle="x", file_path="a",
                                        line_no=1, line_text="x = 1", rarity_score=0.5),
        "GitArchaeologyHit": GitArchaeologyHit(
            stage="1e", file_path="a", sha="abc", parent_sha="def",
            commit_message_summary="fix: x", fix_signal=True, age_days=1, files_changed=1,
        ),
        "DepGraphHop": DepGraphHop(stage="1f", source_file="a", target_file="b",
                                    edge_type="imports", hop_distance=1, confidence=0.5),
        "RerankerOutput": RerankerOutput(
            stage="1g", model="x",
            input_tokens=1, output_tokens=1, prompt_token_count=1,
            ranked_files=(("a", 0.5, "rationale"),),
            skeleton_compression="full",
        ),
    }
    for name, sample in samples.items():
        d = sample.as_dict()
        round_tripped = json.loads(json.dumps(d, default=str))
        assert round_tripped["_signal"] == name
        assert round_tripped["stage"] == sample.stage

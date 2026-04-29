"""Post-submission grader — the ONLY V10 module that touches the
official SWE-bench evaluation harness, the ONLY V10 module that writes
to ``eval_outputs/``, and the ONLY V10 module that reads the gold
``patch`` field from the Verified dataset.

Per V10_DESIGN.md §12.6, the grader is a one-way pipe:

    V10 selection → predictions.jsonl → harness.eval.grade →
    eval_outputs/{run_id}/report.json

Nothing under ``harness/`` reads from ``eval_outputs/``. The firewall
test in ``tests/test_no_oracle_leak.py`` enforces this with an AST scan
that fails any other module referencing the path.

Phase 1 stage 1b adds a SECOND eval-only path in this module:
``evaluate_retrieval_recall``. It reads the gold patch from the
Verified dataset to compute the touched-file set, then compares to the
retrieval pipeline's top-K. The retrieval recall eval is run *offline*
during component sweeps (dev-50 / dev-100) — never in the inference
path. The firewall test asserts no other harness/ module reads the
``patch`` field by AST scan: subscripts ``row["patch"]``, attribute
access ``row.patch``, and ``getattr(row, "patch")``.

This module is FIREWALL_INFRA_FILES. It is allowed to mention
``eval_outputs`` in source. It is allowed to import the SWE-bench
official harness. It is allowed to handle records that contain
``resolved`` and other oracle fields — those records leave V10 via
``write_predictions`` and return only as files on disk that no other
V10 module can read.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Iterable


EVAL_OUTPUTS_ROOT = pathlib.Path("eval_outputs")
PREDICTIONS_SCHEMA_KEYS = ("instance_id", "model_name_or_path", "model_patch")


@dataclass
class Prediction:
    """The shape we hand to the official harness. No oracle fields here —
    we only submit instance_id + model name + patch."""

    instance_id: str
    model_name_or_path: str
    model_patch: str

    def as_jsonl(self) -> str:
        return json.dumps({
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        })


def write_predictions(
    predictions: Iterable[Prediction],
    *,
    run_id: str,
    out_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Write a JSONL predictions file at ``runs/v10_<run_id>/predictions.jsonl``.

    ``runs/`` belongs to V10 selection (writers, not readers). The grader
    reads this file, then writes its own outputs to ``eval_outputs/``.
    """
    base = pathlib.Path(out_dir) if out_dir else pathlib.Path("runs") / f"v10_{run_id}"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "predictions.jsonl"
    with path.open("w") as fh:
        for p in predictions:
            fh.write(p.as_jsonl() + "\n")
    return path


def grade(
    *,
    predictions_path: pathlib.Path | str,
    run_id: str,
    instance_ids: Iterable[str] | None = None,
    max_workers: int = 4,
) -> pathlib.Path:
    """Invoke the official SWE-bench harness on the given predictions.

    Output is written to ``eval_outputs/v10_<run_id>/`` and is
    intentionally NOT returned in any form that V10 selection code can
    consume. The path is logged to stdout/stderr for human review and
    for the trajectory dump.

    Returns the eval-outputs directory path (callers may share that
    path with reviewers but must not import / read its contents inside
    the harness package).
    """
    from swebench.harness.run_evaluation import main as swebench_run

    output_dir = EVAL_OUTPUTS_ROOT / f"v10_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    swebench_run(
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        instance_ids=list(instance_ids) if instance_ids else None,
        predictions_path=str(predictions_path),
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="instance",
        clean=False,
        open_file_limit=4096,
        run_id=f"v10_{run_id}",
        timeout=1800,
        namespace="swebench",
        rewrite_reports=False,
        modal=False,
        instance_image_tag="latest",
        env_image_tag="latest",
        report_dir=str(output_dir),
    )

    return output_dir


# ---------------------------------------------------------------------------
# Retrieval recall eval — Phase 1 stage 1b
# ---------------------------------------------------------------------------


_DIFF_GIT_RE = __import__("re").compile(r"^diff --git a/(\S+) b/(\S+)$", __import__("re").MULTILINE)


def _gold_touched_files_from_patch(patch_text: str) -> set[str]:
    """Parse the ``a/<path>`` references from a unified-diff patch.
    Returns the set of file paths the gold patch modifies.

    This function is the ONLY place in V10 that parses a gold patch.
    It lives in eval.py per the firewall convention.
    """
    if not patch_text:
        return set()
    out: set[str] = set()
    for m in _DIFF_GIT_RE.finditer(patch_text):
        out.add(m.group(1))
    return out


def _load_gold_touched_files(instance_ids: Iterable[str]) -> dict[str, set[str]]:
    """Read the gold patch for each instance from the local Verified
    dataset and parse its touched files. THIS IS THE ONLY V10 PATH
    THAT READS THE PATCH FIELD.
    """
    cache_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
    )
    wanted = set(instance_ids)
    out: dict[str, set[str]] = {}
    with cache_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["instance_id"] not in wanted:
                continue
            # The eval-only patch read. The firewall test asserts no
            # other harness/ module accesses this field.
            gold_patch = row.get("patch", "") or ""
            out[row["instance_id"]] = _gold_touched_files_from_patch(gold_patch)
    return out


@dataclass
class RetrievalRecallEntry:
    instance_id: str
    repo: str
    gold_files: tuple[str, ...]
    retrieved_top10: tuple[str, ...]
    top1_hit: bool
    top5_hit: bool
    top10_hit: bool
    n_gold: int
    n_retrieved: int


@dataclass
class RetrievalRecallReport:
    n_instances: int
    n_top1: int
    n_top5: int
    n_top10: int
    per_instance: list


def evaluate_retrieval_recall(
    *,
    retrieved_files_per_instance: dict,
) -> RetrievalRecallReport:
    """Compute top-1/top-5/top-10 file recall.

    Args:
        retrieved_files_per_instance: ``{instance_id: list_of_file_paths}``.
            The list is the retrieval pipeline's ranked candidates;
            top-K means the first K entries.

    "Hit" semantics: top-K hits if at least one of the K retrieved
    files is in the gold touched-file set. (Lenient — any-of-gold;
    Phase 1 follow-up may switch to exact-match-of-all-gold.)
    """
    instance_ids = list(retrieved_files_per_instance.keys())
    gold = _load_gold_touched_files(instance_ids)

    repo_lookup = _load_repo_per_instance(instance_ids)

    per_instance: list[RetrievalRecallEntry] = []
    n_top1 = n_top5 = n_top10 = 0
    for iid in instance_ids:
        retrieved = list(retrieved_files_per_instance[iid] or [])
        gold_files = gold.get(iid, set())
        gold_files_norm = {_normalize_path(p) for p in gold_files}
        retrieved_norm = [_normalize_path(p) for p in retrieved]

        def _any_in_gold(prefix: list[str]) -> bool:
            return any(p in gold_files_norm for p in prefix)

        top1_hit = _any_in_gold(retrieved_norm[:1])
        top5_hit = _any_in_gold(retrieved_norm[:5])
        top10_hit = _any_in_gold(retrieved_norm[:10])
        n_top1 += int(top1_hit)
        n_top5 += int(top5_hit)
        n_top10 += int(top10_hit)
        per_instance.append(RetrievalRecallEntry(
            instance_id=iid,
            repo=repo_lookup.get(iid, "?"),
            gold_files=tuple(sorted(gold_files)),
            retrieved_top10=tuple(retrieved[:10]),
            top1_hit=top1_hit,
            top5_hit=top5_hit,
            top10_hit=top10_hit,
            n_gold=len(gold_files),
            n_retrieved=len(retrieved),
        ))
    return RetrievalRecallReport(
        n_instances=len(instance_ids),
        n_top1=n_top1, n_top5=n_top5, n_top10=n_top10,
        per_instance=per_instance,
    )


def _load_repo_per_instance(instance_ids: Iterable[str]) -> dict[str, str]:
    """Load only the repo field per instance — non-oracle, used for
    grouping the retrieval-eval results by repo in the report."""
    cache_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
    )
    wanted = set(instance_ids)
    out: dict[str, str] = {}
    with cache_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["instance_id"] in wanted:
                out[row["instance_id"]] = row["repo"]
    return out


def _normalize_path(path: str) -> str:
    """Normalize file paths for comparison: strip leading slashes,
    collapse common prefix-stripping artifacts."""
    return path.strip().lstrip("./").lstrip("/")


__all__ = [
    "Prediction",
    "write_predictions",
    "grade",
    "EVAL_OUTPUTS_ROOT",
    "evaluate_retrieval_recall",
    "RetrievalRecallReport",
    "RetrievalRecallEntry",
]

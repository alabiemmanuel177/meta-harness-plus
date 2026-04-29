"""Post-submission grader — the ONLY V10 module that touches the
official SWE-bench evaluation harness, and the ONLY V10 module that
writes to ``eval_outputs/``.

Per V10_DESIGN.md §12.6, the grader is a one-way pipe:

    V10 selection → predictions.jsonl → harness.eval.grade →
    eval_outputs/{run_id}/report.json

Nothing under ``harness/`` reads from ``eval_outputs/``. The firewall
test in ``tests/test_no_oracle_leak.py`` enforces this with an AST scan
that fails any other module referencing the path.

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


__all__ = ["Prediction", "write_predictions", "grade", "EVAL_OUTPUTS_ROOT"]

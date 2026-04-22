"""Filesystem-laid-out run log.

Faithful to the original Meta-Harness design ethos: *everything on disk*.
An LLM proposer can ``grep`` / ``cat`` through this tree without a special
protocol.

Layout:
  run_dir/
    candidates/
      cand_0001/
        harness.json          # Harness.describe() output
        score.json            # final ScoreVector
        attribution.json      # list[AttributionSnapshot] for this candidate
        traces/eval_XX.json   # per-example traces from screen + full eval
    frontier.json             # current Pareto frontier
    attribution_stats.json    # running AttributionStats per kind
    history.jsonl             # one line per (iteration, event)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback — stringify.
    return str(obj)


class RunLogger:
    def __init__(self, run_dir: str | os.PathLike):
        self.run_dir = Path(run_dir)
        (self.run_dir / "candidates").mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "history.jsonl"

    def candidate_dir(self, candidate_id: str) -> Path:
        d = self.run_dir / "candidates" / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, path: str | os.PathLike, payload: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)

    def record_candidate(self, candidate_id: str, harness_desc: list[dict]) -> None:
        self.write_json(self.candidate_dir(candidate_id) / "harness.json", harness_desc)

    def record_score(self, candidate_id: str, score) -> None:
        self.write_json(self.candidate_dir(candidate_id) / "score.json", score)

    def record_attribution(self, candidate_id: str, snapshots) -> None:
        self.write_json(self.candidate_dir(candidate_id) / "attribution.json", snapshots)

    def record_frontier(self, frontier_entries) -> None:
        self.write_json(self.run_dir / "frontier.json", frontier_entries)

    def record_attribution_stats(self, stats) -> None:
        self.write_json(self.run_dir / "attribution_stats.json", stats)

    def event(self, **kwargs) -> None:
        with self.history_path.open("a") as f:
            f.write(json.dumps(_to_jsonable(kwargs), sort_keys=True) + "\n")

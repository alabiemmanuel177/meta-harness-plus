"""Trajectory logger — write-through JSONL for every LLM call, tool
call, and selector decision in a V10 run.

Per V10_DESIGN.md §3.1, every per-instance per-candidate trajectory
lands at ``trajectories/v10_{run_id}/{instance_id}/{candidate_id}/turn_NNN.jsonl``.
Logs are append-only, gzip on close, and reviewer-rollout-shaped: a
reviewer should be able to re-execute the run from the trajectory.

The logger is deliberately dumb — it serializes whatever it's given.
Care is taken upstream to ensure no oracle-derived field reaches a log.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


_LOCK = threading.Lock()


@dataclass
class TrajectoryWriter:
    """Append-only writer for one instance/candidate combo.

    Each ``write(...)`` call appends a JSON line with the event type,
    timestamp, and payload. No buffering across calls — the caller can
    crash mid-run and we've still recorded everything up to that point.
    """

    base_dir: pathlib.Path
    instance_id: str
    candidate_id: str
    turn_index: int = 0
    file_handle: Any = None  # opened lazily

    def __post_init__(self) -> None:
        self._dir = self.base_dir / self.instance_id / self.candidate_id
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"turn_{self.turn_index:04d}.jsonl"
        self.file_handle = path.open("a")
        self._closed = False

    def write(self, event_type: str, payload: dict | None = None) -> None:
        if self._closed:
            raise RuntimeError("trajectory writer closed")
        record = {
            "ts": time.time(),
            "event": event_type,
            "instance_id": self.instance_id,
            "candidate_id": self.candidate_id,
            "turn": self.turn_index,
            "payload": payload or {},
        }
        with _LOCK:
            self.file_handle.write(json.dumps(record, default=str) + "\n")
            self.file_handle.flush()

    def advance_turn(self) -> None:
        """Close the current turn file, open the next."""
        self.close()
        self.turn_index += 1
        path = self._dir / f"turn_{self.turn_index:04d}.jsonl"
        self.file_handle = path.open("a")
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.file_handle.flush()
            self.file_handle.close()
        finally:
            self._closed = True

    def gzip_and_finalize(self) -> None:
        """Compress all turn_*.jsonl files in this candidate dir.
        Called at end of candidate processing."""
        self.close()
        for path in sorted(self._dir.glob("turn_*.jsonl")):
            with path.open("rb") as fin, gzip.open(str(path) + ".gz", "wb") as fout:
                fout.writelines(fin)
            path.unlink()


@contextmanager
def trajectory(
    base_dir: pathlib.Path,
    instance_id: str,
    candidate_id: str = "smoke",
) -> Iterator[TrajectoryWriter]:
    """Context manager wrapping a TrajectoryWriter."""
    w = TrajectoryWriter(
        base_dir=base_dir,
        instance_id=instance_id,
        candidate_id=candidate_id,
    )
    try:
        yield w
    finally:
        w.close()


__all__ = ["TrajectoryWriter", "trajectory"]

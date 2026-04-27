"""AIME (American Invitational Math Exam) task loader.

AIME problems have a unique answer format: an integer between 0 and 999
inclusive. This makes scoring exact and unambiguous (vs MATH which has
boxed expressions).

Default source: ``math-ai/aime25`` on HuggingFace (30 problems = AIME I +
AIME II, 2025). Each record has fields:

    {"problem": "<latex problem>", "answer": "<int 0-999 as string>", "id": "<idx>"}

We frame this as a generation task (not classification), reusing
``MathLLMPredictor`` and an AIME-specific answer parser that's strict
about integer 0-999.

Falls back to a small synthetic 5-problem fixture for unit tests when
no network/dataset is available.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from ..task import Task, TaskExample
from .jsonl_loader import build_task_from_jsonl


_DATA_DIR = Path(__file__).parent / "data" / "aime"


# AIME answer canonicalization: strip whitespace, leading zeros (but
# keep "0"), reject non-integers and out-of-range values.
def parse_aime_answer(text: str) -> int | None:
    """Parse a model output into an AIME integer 0-999, or None if invalid.

    Looks for the AIME convention: the final integer in the response,
    optionally inside ``\\boxed{...}`` or ``#### N`` markers. Falls back
    to the last bare integer in the text.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    # Strongest signal: \boxed{N}
    m = re.search(r"\\boxed\{\s*(-?\d+)\s*\}", s)
    if m:
        return _coerce_aime(m.group(1))

    # GSM8K-style #### N marker (some prompts inherit it).
    m = re.search(r"####\s*(-?\d+)", s)
    if m:
        return _coerce_aime(m.group(1))

    # The literal phrase "answer is N" / "answer: N" — common pattern.
    m = re.search(r"answer\s*(?:is|:)\s*\\?\$?(-?\d+)", s, re.IGNORECASE)
    if m:
        return _coerce_aime(m.group(1))

    # Fallback: take the LAST integer in the text. Helpful for
    # responses that end with the bare number.
    matches = re.findall(r"-?\d+", s)
    if not matches:
        return None
    return _coerce_aime(matches[-1])


def _coerce_aime(raw: str) -> int | None:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    if 0 <= v <= 999:
        return v
    return None


def _is_correct(predicted_text: str, label_text: str) -> bool:
    """AIME equality: parse both sides as ints, compare."""
    p = parse_aime_answer(predicted_text)
    if p is None:
        return False
    try:
        gold = int(str(label_text).strip())
    except (TypeError, ValueError):
        return False
    return p == gold


# ---------- loader ----------

def build_aime25_task(*, n_train: int = 0, n_eval: int | None = None,
                     verify: bool = True) -> Task:
    """Load AIME-25 (30 problems) as a Task.

    Reads the dataset from HuggingFace ``math-ai/aime25`` if cached or
    network-available; otherwise looks for a JSONL fallback at
    ``meta_harness_plus/tasks/data/aime/aime25.jsonl``.

    Args:
      n_train: If >0, take this many problems from the test split as a
        train subset (since AIME-25 has no separate train split, this
        is just a reserved slice for few-shot retrieval).
      n_eval: Cap eval set; default = all remaining.
      verify: If True (default), assert every answer is an integer
        0-999. Raises ValueError on a violation — bad data poisons
        downstream metrics.
    """
    rows = _load_aime25_rows()
    if verify:
        for r in rows:
            v = _coerce_aime(r["answer"])
            if v is None:
                raise ValueError(
                    f"AIME-25 row id={r.get('id')!r} has non-integer-or-out-of-range answer "
                    f"{r['answer']!r} — refusing to load. Re-download or fix the source."
                )

    train: list[TaskExample] = []
    eval_set: list[TaskExample] = []
    for i, r in enumerate(rows):
        ex = TaskExample(input=str(r["problem"]), label=str(r["answer"]),
                         meta=(("id", str(r.get("id", i))),))
        if i < n_train:
            train.append(ex)
        else:
            eval_set.append(ex)
    if n_eval is not None:
        eval_set = eval_set[:n_eval]

    # AIME has 1000 possible answers (0..999); we don't enumerate them
    # in ``classes`` because the predictor is open-ended generation.
    return Task(name="aime25", train=train, eval_set=eval_set, classes=[])


def _load_aime25_rows() -> list[dict]:
    """Try HuggingFace, fall back to bundled JSONL, fall back to fixture."""
    cached = _DATA_DIR / "aime25.jsonl"
    if cached.exists():
        rows = []
        for line in cached.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if len(rows) >= 30:
            return rows

    # HuggingFace path
    try:
        from datasets import load_dataset
        ds = load_dataset("math-ai/aime25", split="test")
        rows = [{"problem": r["problem"], "answer": r["answer"],
                 "id": r["id"]} for r in ds]
        # Cache locally for reproducibility.
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with cached.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return rows
    except Exception:
        pass

    # Fixture fallback (5 hand-authored AIME-style problems for tests).
    return _fixture_rows()


def _fixture_rows() -> list[dict]:
    return [
        {"problem": "What is 2+2? Provide the integer.", "answer": "4", "id": "fixture0"},
        {"problem": "What is 7*9? Provide the integer.", "answer": "63", "id": "fixture1"},
        {"problem": "What is 100-25? Provide the integer.", "answer": "75", "id": "fixture2"},
        {"problem": "What is the sum of integers from 1 to 10? Provide the integer.", "answer": "55", "id": "fixture3"},
        {"problem": "What is 12 squared? Provide the integer.", "answer": "144", "id": "fixture4"},
    ]


def build_aime25_fixture_task() -> Task:
    """Tiny 5-problem AIME-shaped fixture for unit tests; no network."""
    return Task(
        name="aime25_fixture",
        train=[],
        eval_set=[
            TaskExample(input=r["problem"], label=r["answer"],
                        meta=(("id", r["id"]),))
            for r in _fixture_rows()
        ],
        classes=[],
    )

"""HMMT (Harvard-MIT Math Tournament) Feb-2025 task loader.

Source: ``MathArena/hmmt_feb_2025`` on HuggingFace.
Commit pinned in the loader docstring for reproducibility:

    hf://datasets/MathArena/hmmt_feb_2025@a7ceac11a0b5b8c18a9b50d41d7ff96311b189c2/

Schema (per HuggingFace):
    {"problem_idx": int, "problem": str, "answer": str, "problem_type": [str]}

**14/30 integer-subset caveat (must be quoted in any writeup):**
HMMT-Feb-2025 has 30 problems; **14 have integer answers** and 16 have
non-integer answers (fractions, surds, latex expressions). Our
deterministic grader only handles integers cleanly. We default to the
14-problem integer subset so every baseline / search number is on a
strictly-comparable, exact-match grading surface. The trade-off: the
power on a 14-problem set is lower than on the full 30. If Phase 2
baseline CIs come back with too much overlap to distinguish methods,
the right move is to extend the integer pool with HMMT-Feb-2024 +
HMMT-Nov-2024 (≈ +25 integer-answer problems likely), not to flip on
a sympy-equivalence grader (which adds its own correctness story).

Loader policy: default ``integer_only=True`` filters to the 14 integer-
answer problems, matching the existing AIME-style grader. Set
``integer_only=False`` to load all 30 problems with the integer subset
labelled ``"<non_int>"`` so downstream code can either skip or wire a
sympy-based grader. We deliberately do NOT auto-pivot to sympy in
this loader — equivalence over latex is its own correctness story.

For unit tests we ship a 5-problem fixture (no network) covering both
the AIME-style format and HMMT-style problem text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..task import Task, TaskExample


_DATA_DIR = Path(__file__).parent / "data" / "hmmt"
_DATASET_ID = "MathArena/hmmt_feb_2025"
_PINNED_COMMIT = "a7ceac11a0b5b8c18a9b50d41d7ff96311b189c2"


def _is_integer_answer(s: str) -> bool:
    if s is None:
        return False
    try:
        int(str(s).strip())
        return True
    except (TypeError, ValueError):
        return False


def build_hmmt_feb2025_task(
    *,
    integer_only: bool = True,
    n_eval: int | None = None,
    verify: bool = True,
) -> Task:
    """Load HMMT-Feb-2025 as a Task.

    Args:
      integer_only: If True (default), filter to the 14 integer-answer
        problems. If False, load all 30; non-integer answers are kept
        verbatim and *will* fail the AIME-style grader unless callers
        plug in a richer equivalence test.
      n_eval: Cap eval set; default = all loaded.
      verify: If True, assert every loaded problem has a non-empty
        answer string. Raises ValueError on a violation.
    """
    rows = _load_hmmt_rows()
    if integer_only:
        rows = [r for r in rows if _is_integer_answer(r["answer"])]
    if verify:
        for r in rows:
            if not str(r["answer"]).strip():
                raise ValueError(
                    f"HMMT row idx={r.get('problem_idx')!r} has empty answer."
                )

    eval_set: list[TaskExample] = []
    for r in rows:
        meta = (
            ("idx", str(r["problem_idx"])),
            ("type", "/".join(r.get("problem_type") or [])),
            ("integer_answer", "true" if _is_integer_answer(r["answer"]) else "false"),
        )
        eval_set.append(TaskExample(
            input=str(r["problem"]),
            label=str(r["answer"]).strip(),
            meta=meta,
        ))
    if n_eval is not None:
        eval_set = eval_set[:n_eval]

    name = "hmmt_feb2025_intonly" if integer_only else "hmmt_feb2025_all"
    return Task(name=name, train=[], eval_set=eval_set, classes=[])


def _load_hmmt_rows() -> list[dict]:
    cached = _DATA_DIR / "hmmt_feb2025.jsonl"
    if cached.exists():
        rows = []
        for line in cached.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if len(rows) >= 30:
            return rows

    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET_ID, split="train")
        rows = []
        for r in ds:
            rows.append({
                "problem_idx": int(r["problem_idx"]),
                "problem": str(r["problem"]),
                "answer": str(r["answer"]),
                "problem_type": list(r.get("problem_type") or []),
            })
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with cached.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return rows
    except Exception:
        return _fixture_rows()


def _fixture_rows() -> list[dict]:
    """5-problem HMMT-shaped fixture (3 integer-answer, 2 non-integer).

    Used when no network/dataset is available so unit tests don't depend
    on HuggingFace downloads.
    """
    return [
        {"problem_idx": 1,
         "problem": "Compute the sum of positive divisors of 12.",
         "answer": "28", "problem_type": ["Number Theory"]},
        {"problem_idx": 2,
         "problem": "How many ways to arrange ABCD in a circle?",
         "answer": "6", "problem_type": ["Combinatorics"]},
        {"problem_idx": 3,
         "problem": "Find the smallest n with n^2 > 100.",
         "answer": "11", "problem_type": ["Algebra"]},
        # Non-integer answers (filtered out by default).
        {"problem_idx": 4,
         "problem": "Compute the area of an equilateral triangle with side 1.",
         "answer": r"\frac{\sqrt{3}}{4}", "problem_type": ["Geometry"]},
        {"problem_idx": 5,
         "problem": "Solve x^2 = 5 for positive x.",
         "answer": r"\sqrt{5}", "problem_type": ["Algebra"]},
    ]


def build_hmmt_fixture_task() -> Task:
    """Tiny offline fixture for unit tests (3 integer-answer rows)."""
    rows = [r for r in _fixture_rows() if _is_integer_answer(r["answer"])]
    return Task(
        name="hmmt_fixture",
        train=[],
        eval_set=[
            TaskExample(input=r["problem"], label=r["answer"],
                        meta=(("idx", str(r["problem_idx"])),))
            for r in rows
        ],
        classes=[],
    )


def parse_hmmt_int_answer(text: str) -> int | None:
    """Permissive integer parser for HMMT (allows negative + up to ~10⁵).

    Differs from ``parse_aime_answer`` only in range — HMMT integer
    answers can be 0..several thousand and occasionally negative. We
    cap at +/- 10⁶ to reject runaway garbage.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    m = re.search(r"\\boxed\{\s*(-?\d+)\s*\}", s)
    if m:
        return _coerce(m.group(1))
    m = re.search(r"answer\s*(?:is|:)\s*\\?\$?(-?\d+)", s, re.IGNORECASE)
    if m:
        return _coerce(m.group(1))
    matches = re.findall(r"-?\d+", s)
    if not matches:
        return None
    return _coerce(matches[-1])


def _coerce(raw: str) -> int | None:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    if -1_000_000 <= v <= 1_000_000:
        return v
    return None

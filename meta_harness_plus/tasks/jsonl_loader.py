"""JSONL task loader.

Read a classification task from JSONL files where each line is a record
``{"input": "...", "label": "..."}``. This is the zero-dep path for
plugging in any real classification dataset — point it at your own files.

No external ``datasets`` / HuggingFace dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..task import Task, TaskExample


def load_jsonl(path: str | Path) -> list[TaskExample]:
    """Parse a JSONL file of ``{"input": "...", "label": "..."}`` records."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset file not found: {p}")
    out: list[TaskExample] = []
    with p.open() as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON — {e}") from e
            if "input" not in rec or "label" not in rec:
                raise ValueError(
                    f"{p}:{lineno}: record missing 'input' and/or 'label' keys: {rec!r}"
                )
            out.append(TaskExample(input=str(rec["input"]), label=str(rec["label"])))
    return out


def build_task_from_jsonl(
    name: str,
    train_path: str | Path,
    eval_path: str | Path,
    classes: Iterable[str] | None = None,
) -> Task:
    """Build a ``Task`` from two JSONL files.

    If ``classes`` is None, the class set is inferred from the union of
    train + eval labels, sorted alphabetically — deterministic ordering is
    important because downstream prompts list classes in this order.
    """
    train = load_jsonl(train_path)
    eval_set = load_jsonl(eval_path)
    if classes is None:
        cls_set = {e.label for e in train} | {e.label for e in eval_set}
        classes = sorted(cls_set)
    else:
        classes = list(classes)
        # Validate no surprise labels in the data.
        cls_set = set(classes)
        for split_name, items in (("train", train), ("eval", eval_set)):
            extras = {e.label for e in items} - cls_set
            if extras:
                raise ValueError(
                    f"{split_name} set contains labels not in ``classes``: {sorted(extras)}"
                )
    return Task(name=name, train=train, eval_set=eval_set, classes=list(classes))



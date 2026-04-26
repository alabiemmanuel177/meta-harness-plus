"""Loaders for public classification datasets bundled under tasks/data/.

Two public-dataset cells used in the paper:

- ``agnews``: AG News, 4-class news topic classification (World, Sports,
  Business, Sci/Tech). ~7600 test rows in the original; we sample 48
  balanced eval items + 80 balanced train items.

- ``emotion``: dair-ai/emotion, 6-class emotion classification (sadness,
  joy, love, anger, fear, surprise). 2000 test rows in the original;
  we sample 48 balanced eval + 78 balanced train.

Both are downloaded by ``scripts/download_public_datasets.py`` and
written as JSONL ``{"input", "label"}`` records under
``tasks/data/{agnews, emotion}/``.

These are public benchmarks chosen specifically to break the
"all-our-benchmarks-are-hand-curated" reviewer concern. AG News in
particular is a near-direct analogue of our hand-curated news_hard_50.
"""
from __future__ import annotations

from pathlib import Path

from ..task import Task
from .jsonl_loader import build_task_from_jsonl


_DATA_DIR = Path(__file__).parent / "data"


def build_agnews_task() -> Task:
    d = _DATA_DIR / "agnews"
    return build_task_from_jsonl(
        name="agnews",
        train_path=d / "agnews_train.jsonl",
        eval_path=d / "agnews_test.jsonl",
    )


def build_emotion_task() -> Task:
    d = _DATA_DIR / "emotion"
    return build_task_from_jsonl(
        name="emotion",
        train_path=d / "emotion_train.jsonl",
        eval_path=d / "emotion_test.jsonl",
    )

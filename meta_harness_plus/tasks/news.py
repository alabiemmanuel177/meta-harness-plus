"""Bundled news-headline classification task.

AG-News-style: 4 classes (world, sports, business, tech), 40 train / 20
eval, hand-curated headlines in the style of real wire-service copy.
Different domain from ``symptom_classification`` — gives the RAG-vs-MH++
research question a cross-domain replication point.
"""

from __future__ import annotations

from pathlib import Path

from ..task import Task
from .jsonl_loader import build_task_from_jsonl


NEWS_CLASSES = ("business", "sports", "tech", "world")

_DATA_DIR = Path(__file__).parent / "data"


def build_news_task(seed: int = 0) -> Task:
    _ = seed
    return build_task_from_jsonl(
        name="news_classification",
        train_path=_DATA_DIR / "news_train.jsonl",
        eval_path=_DATA_DIR / "news_eval.jsonl",
        classes=NEWS_CLASSES,
    )

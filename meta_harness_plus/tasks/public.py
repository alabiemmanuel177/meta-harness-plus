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


def build_newsgroups20_task() -> Task:
    """20 Newsgroups, top-8 most-common classes (rec.sport.hockey,
    soc.religion.christian, rec.motorcycles, rec.sport.baseball, sci.crypt,
    rec.autos, sci.med, sci.space). Posts truncated to 600 chars for
    retrieval-friendliness."""
    d = _DATA_DIR / "newsgroups20"
    return build_task_from_jsonl(
        name="newsgroups20",
        train_path=d / "newsgroups20_train.jsonl",
        eval_path=d / "newsgroups20_test.jsonl",
    )


def build_symptom2disease_task() -> Task:
    """Symptom2Disease (gretelai/symptom_to_diagnosis), top-8 most-common
    disease classes from 22 (cervical spondylosis, impetigo, arthritis,
    dengue, drug reaction, malaria, allergy, bronchial asthma).
    Public-dataset analogue of our hand-curated symptom_hard."""
    d = _DATA_DIR / "symptom2disease"
    return build_task_from_jsonl(
        name="symptom2disease",
        train_path=d / "symptom2disease_train.jsonl",
        eval_path=d / "symptom2disease_test.jsonl",
    )


def build_patents_task() -> Task:
    """Patent classification (ccdv/patent-classification), top-8 of 9
    high-level CPC categories: Physics, Electricity, Human Necessities, etc.

    Public-dataset substitute for USPTO-50k. The original USPTO-50k uses
    hundreds of fine-grained CPC codes which exceed our 8-class budget;
    this benchmark uses the high-level CPC sections instead. Patent
    abstracts truncated to first 600 chars for retrieval-friendliness."""
    d = _DATA_DIR / "patents"
    return build_task_from_jsonl(
        name="patents",
        train_path=d / "patents_train.jsonl",
        eval_path=d / "patents_test.jsonl",
    )

"""Built-in tasks.

- ``toy_classification`` — deterministic offline task used for unit tests;
  no external LLM required.
- ``symptom_classification`` — hand-curated medical specialty classification
  (5 classes, 30 train / 20 eval). Realistic short patient complaints.
- ``jsonl_loader`` — generic loader for any JSONL ``{"input", "label"}`` dataset.
"""

from .toy_classification import build_toy_task, mock_llm
from .jsonl_loader import build_task_from_jsonl, load_jsonl
from .symptom import (
    SYMPTOM_CLASSES,
    build_symptom_task,
    build_symptom_hard_task,
    symptom_mock_llm,
)
from .news import NEWS_CLASSES, build_news_task, build_news_hard_task

__all__ = [
    "build_toy_task",
    "mock_llm",
    "build_symptom_task",
    "build_symptom_hard_task",
    "build_task_from_jsonl",
    "load_jsonl",
    "SYMPTOM_CLASSES",
    "symptom_mock_llm",
    "NEWS_CLASSES",
    "build_news_task",
    "build_news_hard_task",
]

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
from .news import (
    NEWS_CLASSES,
    build_news_task,
    build_news_hard_task,
    build_news_hard_50_task,
)
from .public import (
    build_agnews_task,
    build_emotion_task,
    build_newsgroups20_task,
    build_patents_task,
    build_symptom2disease_task,
)
from .math_task import (
    build_gsm8k_task,
    extract_math_answer,
)
from .terminalbench_fixture import build_terminalbench_fixture
from .uspto import (
    USPTO50K_CLASSES,
    build_uspto50k_task,
    build_uspto_fixture_task,
    build_uspto_patents_task,
)
from .massive import (
    MASSIVE_TOP8,
    build_massive_fixture_task,
    build_massive_task,
)
from .lawbench import (
    LAWBENCH_CLASSIFICATION_SUBTASKS,
    build_all_lawbench_classification_tasks,
    build_lawbench_fixture_task,
    build_lawbench_task,
    list_available_lawbench_subtasks,
)

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
    "build_news_hard_50_task",
    "build_agnews_task",
    "build_emotion_task",
    "build_newsgroups20_task",
    "build_patents_task",
    "build_symptom2disease_task",
    "build_gsm8k_task",
    "extract_math_answer",
    "build_terminalbench_fixture",
    "USPTO50K_CLASSES",
    "build_uspto50k_task",
    "build_uspto_fixture_task",
    "build_uspto_patents_task",
    "MASSIVE_TOP8",
    "build_massive_fixture_task",
    "build_massive_task",
    "LAWBENCH_CLASSIFICATION_SUBTASKS",
    "build_all_lawbench_classification_tasks",
    "build_lawbench_fixture_task",
    "build_lawbench_task",
    "list_available_lawbench_subtasks",
]

"""Named baseline harnesses — canonical reference points for head-to-head comparison.

Every bakeoff should seed the Pareto frontier with one or more of these so
MH++-discovered harnesses have an explicit point to beat. In particular
``rag_baseline`` implements the classic RAG shape (retriever + top-k few-shot
+ predictor); if the search can't Pareto-dominate it, that's a published
negative result worth reporting, not hand-waved.
"""

from __future__ import annotations

from typing import Any

from .components import (
    BagOfWordsRetriever,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
)
from .harness import Harness, Component
from .task import Task


def rag_baseline(
    task: Task,
    predictor: Component,
    *,
    retriever_k: int = 3,
    fewshot_k: int = 2,
    formatter: Component | None = None,
) -> Harness:
    """Canonical RAG harness: BoW-retrieve top-k, use top-k as few-shot, classify.

    No voting. Parameters ``retriever_k`` / ``fewshot_k`` default to
    commonly-reported RAG values. Caller supplies the Predictor component
    so the same baseline works with mock LLMs (``MockLLMPredictor``) or
    real ones (``LLMPredictor``).
    """
    return Harness(components=[
        BagOfWordsRetriever(corpus=task.train, k=retriever_k),
        TopKFewShot(k=fewshot_k),
        formatter or SimpleFormatter(),
        predictor,
        NullVoter(),
    ])


def bare_baseline(task: Task, predictor: Component) -> Harness:
    """No retrieval, no fewshot, no voting — just the base predictor.

    The cheapest possible harness. A floor for comparison: if RAG doesn't
    beat this on a task, RAG isn't helping on that task.
    """
    return Harness(components=[
        NullRetriever(),
        NullFewShot(),
        SimpleFormatter(),
        predictor,
        NullVoter(),
    ])


def describe_baseline(harness: Harness) -> dict[str, Any]:
    """Compact dict describing a baseline harness — for logging in RESULTS docs."""
    return {
        "components": [c.config() for c in harness.components],
        "kinds": [c.kind for c in harness.components],
    }

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
    CoTFormatter,
    DiversityReranker,
    MajorityVoter,
    NullFewShot,
    NullReranker,
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


def cot_baseline(
    task: Task,
    predictor: Component,
    *,
    fewshot_k: int = 2,
    retriever_k: int = 3,
) -> Harness:
    """Hand-tuned chain-of-thought + RAG baseline (Tier 1.3 strong baseline).

    A reasonable manual prompt-engineering effort: BoW retrieval with
    top-k few-shot + CoT formatter elicits step-by-step reasoning before
    the class label. Stronger than vanilla RAG on tasks where reasoning
    helps; weaker on tasks where the bare LLM already aces it.

    Reviewers will ask 'did MH++ beat hand-tuned CoT?' — having the
    baseline available as a one-liner makes the comparison cheap.
    """
    return Harness(components=[
        BagOfWordsRetriever(corpus=task.train, k=retriever_k),
        TopKFewShot(k=fewshot_k),
        CoTFormatter(),
        predictor,
        NullVoter(),
    ])


def voting_rag_baseline(
    task: Task,
    predictor: Component,
    *,
    retriever_k: int = 3,
    fewshot_k: int = 2,
) -> Harness:
    """RAG + self-consistency voting (Tier 1.3 strong baseline).

    Caller is expected to have constructed ``predictor`` with
    ``n_samples > 1`` and ``temperature > 0`` for voting to actually
    matter. We add MajorityVoter on the back end. Common production
    shape; another reasonable RAG variant to compare MH++ against.
    """
    return Harness(components=[
        BagOfWordsRetriever(corpus=task.train, k=retriever_k),
        TopKFewShot(k=fewshot_k),
        SimpleFormatter(),
        predictor,
        MajorityVoter(),
    ])


def diverse_rag_baseline(
    task: Task,
    predictor: Component,
    *,
    retriever_k: int = 5,
    fewshot_k: int = 3,
) -> Harness:
    """RAG with diversity reranker on the retrieved set (Tier 1.3).

    Picks more retrievals than fewshot uses, then diversity-reranks so
    each unique retrieved label is represented before duplicates fill
    in. The fewshot then gets a more class-balanced sample. Fully
    deterministic (no LLM call in the reranker), so the only added cost
    is the larger retrieval pool.
    """
    return Harness(components=[
        BagOfWordsRetriever(corpus=task.train, k=retriever_k),
        DiversityReranker(),
        TopKFewShot(k=fewshot_k),
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

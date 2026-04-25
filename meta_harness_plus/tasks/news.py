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


def build_news_hard_task(seed: int = 0) -> Task:
    """Adversarial news-headline variant.

    15 eval items where the class bridges 2 of the 4 categories (business
    <-> sports, world <-> business, tech <-> world, etc.) — the correct
    label comes from the *subject* of the headline, not its surface
    keywords:

    - "Tennis star invests in pickleball league ..." → sports (athlete's
      action), not business (despite the investment language)
    - "Quantum computing startup files IPO ..." → business (it's an IPO),
      not tech (despite the quantum framing)
    - "Olympic committee bans country over doping ..." → sports, not world

    50 train items (the easy 40 + 10 new disambiguating items) give
    retrieval something to latch onto. RAG should meaningfully beat the
    bare LLM here, creating the conditions for MH++ vs RAG comparison —
    same structural idea as symptom_hard for the medical domain.
    """
    _ = seed
    return build_task_from_jsonl(
        name="news_hard",
        train_path=_DATA_DIR / "news_hard_train.jsonl",
        eval_path=_DATA_DIR / "news_hard_eval.jsonl",
        classes=NEWS_CLASSES,
    )


def build_news_hard_50_task(seed: int = 0) -> Task:
    """Bigger-eval variant of news_hard — 50 eval items vs 15.

    Same training set (56 items including disambiguating patterns).
    50 hand-curated adversarial eval items balanced ~12-13 per class
    (13 sports / 12 business / 13 world / 12 tech). Mix of the original
    15 + 35 new items spanning class bridges (athlete-business deals,
    nation-state cyberattacks, league financial structure, etc.).

    Why bigger eval: with 15 items × 2 repeats, each correct/incorrect
    moves accuracy by 1/30 ≈ 3.3pt. Strict Pareto dominance over RAG
    (matching tokens, ≥1pt accuracy) requires beating RAG by ≥3.3pt —
    an entire example. With 50 items × 2 repeats = 100 forward passes,
    each example moves accuracy by 1pt, making fine-grained dominance
    targets actually meet-able.
    """
    _ = seed
    return build_task_from_jsonl(
        name="news_hard_50",
        train_path=_DATA_DIR / "news_hard_train.jsonl",
        eval_path=_DATA_DIR / "news_hard_50_eval.jsonl",
        classes=NEWS_CLASSES,
    )

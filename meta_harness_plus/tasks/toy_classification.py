"""Deterministic toy text-classification task + mock LLM.

Offline benchmark for the framework. The mock LLM rewards harnesses that:
  - retrieve relevant training examples (retriever helps)
  - include few-shot examples in the prompt (fewshot helps)
  - sample multiple candidates + vote (voter helps, at token cost)

There is NO external API call. Everything is deterministic given the seed.

The reward surface is designed so there's a real Pareto tradeoff: more samples
cost more tokens but give diminishing accuracy returns — a good search should
discover the 2-3 sample sweet spot, not push n_samples to infinity.
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence

from ..harness import Context, Harness
from ..task import Task, TaskExample


# Five tiny themed classes with disjoint keyword pools.
CLASS_KEYWORDS = {
    "sports":    ["goal", "team", "score", "match", "player", "league", "champion"],
    "cooking":   ["recipe", "flavor", "spice", "bake", "roast", "dish", "ingredient"],
    "finance":   ["invest", "stock", "bond", "dividend", "market", "portfolio", "yield"],
    "health":    ["doctor", "patient", "symptom", "medicine", "therapy", "dose", "clinic"],
    "tech":      ["server", "compile", "database", "algorithm", "cache", "deploy", "bug"],
}
CLASSES = tuple(CLASS_KEYWORDS.keys())


def _make_sentence(klass: str, rng: random.Random) -> str:
    """Produce a short sentence with just enough class signal to be noisy.

    2 own-class keywords + 2 distractors drawn from *different* other classes.
    This keeps the true class identifiable but weak enough that noise can flip
    single-sample predictions, creating real headroom for retrieval + voting.
    """
    own = CLASS_KEYWORDS[klass]
    words = rng.sample(own, k=2)
    others = [k for k in CLASSES if k != klass]
    rng.shuffle(others)
    for k in others[:2]:
        words.append(rng.choice(CLASS_KEYWORDS[k]))
    rng.shuffle(words)
    return " ".join(words)


def build_toy_task(
    n_train: int = 50,
    n_eval: int = 20,
    seed: int = 0,
) -> Task:
    rng = random.Random(seed)
    train: list[TaskExample] = []
    for _ in range(n_train):
        klass = rng.choice(CLASSES)
        train.append(TaskExample(input=_make_sentence(klass, rng), label=klass))
    eval_set: list[TaskExample] = []
    for _ in range(n_eval):
        klass = rng.choice(CLASSES)
        eval_set.append(TaskExample(input=_make_sentence(klass, rng), label=klass))
    return Task(name="toy_classification", train=train, eval_set=eval_set, classes=CLASSES)


# ---------------- Mock LLM ----------------

# Deterministic "perceived correctness" table. For a given (input, sample_idx)
# the mock LLM computes a score per class based on keyword overlap with the
# prompt. The trick: in the *absence* of few-shots the scores are noisier,
# so adding few-shots genuinely improves accuracy. Sampling noise decreases
# per-sample confidence, so majority voting wins out — but at token cost.

def _class_signal(text: str) -> dict[str, float]:
    """Keyword-overlap signal of ``text`` against each class."""
    tokens = set(text.lower().split())
    return {klass: sum(1 for kw in kws if kw in tokens) for klass, kws in CLASS_KEYWORDS.items()}


def _det_noise(prompt: str, sample_idx: int) -> float:
    """Deterministic 'random' noise in [-1, +1] from sha1 of (prompt, sample_idx)."""
    h = hashlib.sha1(f"{prompt}||{sample_idx}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF  # [0, 1]
    return 2.0 * v - 1.0


def mock_llm(
    *,
    has_fewshot_signal: bool = True,
    sample_noise: float = 2.5,
) -> callable:
    """Return an llm_fn suitable for ``MockLLMPredictor``.

    The returned fn is a closure over config — you can vary it to simulate
    different base models for held-out evaluation (future work).
    """
    def llm_fn(prompt: str, ctx: Context, harness: Harness, sample_idx: int):
        # Base signal comes from the QUERY only (simulates a model that reads
        # the query contents most heavily). Few-shots contribute via the
        # explicit bonus below, not by polluting base signal.
        base_signal = _class_signal(ctx.example.input)
        bonus = {k: 0.0 for k in CLASSES}
        if has_fewshot_signal:
            for fs in ctx.few_shots:
                # Few-shots labelled with the true class lift it; wrong-labelled
                # few-shots mislead. Magnitude is meaningful vs base (~2) + noise.
                if fs.label == ctx.example.label:
                    bonus[fs.label] += 1.2
                else:
                    bonus[fs.label] += 0.4
        noisy = {
            k: base_signal[k] + bonus[k] + sample_noise * _det_noise(prompt + k, sample_idx)
            for k in CLASSES
        }
        pred = max(noisy, key=lambda k: noisy[k])
        tokens = 5
        latency = 3.0
        return pred, tokens, latency
    return llm_fn

"""Bundled medical symptom classification task.

- 5 clinical specialties (cardiology, dermatology, gastroenterology,
  neurology, orthopedics)
- 30 train + 20 eval items, balanced at 6/4 per class
- Realistic patient-complaint phrasing (short single sentences)
- Class-specific terminology dominates, but some inter-class overlap
  ("pain", "worse", "after") so it's not trivially keyword-matchable

Also exposes ``symptom_mock_llm()`` — a deterministic offline "LLM" keyed
to medical vocabulary so tests and demos can exercise the search pipeline
without hitting a real backend.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from ..harness import Context, Harness
from ..task import Task
from .jsonl_loader import build_task_from_jsonl


SYMPTOM_CLASSES = (
    "cardiology",
    "dermatology",
    "gastroenterology",
    "neurology",
    "orthopedics",
)

_DATA_DIR = Path(__file__).parent / "data"


def build_symptom_task(seed: int = 0) -> Task:
    """Load the bundled medical symptom dataset as a ``Task``.

    ``seed`` is accepted for API symmetry with ``build_toy_task`` but
    currently unused — the JSONL file order is the canonical order.
    """
    _ = seed
    return build_task_from_jsonl(
        name="symptom_classification",
        train_path=_DATA_DIR / "symptom_train.jsonl",
        eval_path=_DATA_DIR / "symptom_eval.jsonl",
        classes=SYMPTOM_CLASSES,
    )


def build_symptom_hard_task(seed: int = 0) -> Task:
    """Harder adversarial variant of the symptom task.

    Eval queries are deliberately ambiguous — the class is rarely
    determinable from keywords alone. Two characteristic examples:

    - "Pain that wakes the patient at night and improves after eating"
      — sounds cardiac at first glance; the post-meal relief signals a
      peptic ulcer (GI). Only discoverable if the LLM retrieves the
      disambiguating training pattern.
    - "Sharp chest pain reproducible by pressing on the chest wall"
      — the knee-jerk class is cardiology; palpation-reproducibility is
      the musculoskeletal signature (ortho).

    Training set includes the original 30 items PLUS 20 new items that
    teach disambiguation patterns. RAG should meaningfully beat the bare
    LLM here — that's the point: a task where harness shape matters.
    """
    _ = seed
    return build_task_from_jsonl(
        name="symptom_hard",
        train_path=_DATA_DIR / "symptom_hard_train.jsonl",
        eval_path=_DATA_DIR / "symptom_hard_eval.jsonl",
        classes=SYMPTOM_CLASSES,
    )


# ---------------- mock LLM tuned to medical vocabulary ----------------

SYMPTOM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cardiology": (
        "chest", "heart", "heartbeat", "palpitations", "blood", "pressure",
        "cardiac", "arm", "breath", "exertion", "ankles", "fatigue",
        "palpitation", "systolic", "tachycardia", "racing", "pulse",
    ),
    "dermatology": (
        "rash", "skin", "acne", "mole", "itchy", "patches", "hives",
        "bumps", "scaly", "flaky", "redness", "cystic", "pigment", "torso",
        "shaving", "eczema", "dermatitis", "blisters",
    ),
    "gastroenterology": (
        "stomach", "bowel", "abdominal", "nausea", "vomiting", "heartburn",
        "reflux", "diarrhea", "constipation", "bloating", "stools", "meals",
        "digestive", "swallowing", "appetite", "tarry", "cramping",
    ),
    "neurology": (
        "headache", "numbness", "tingling", "speech", "vision", "muscle",
        "twitching", "memory", "confusion", "tremor", "weakness", "pins",
        "neuropathy", "migraine", "light", "brain", "neurological",
    ),
    "orthopedics": (
        "knee", "joint", "shoulder", "ankle", "hip", "wrist", "stiffness",
        "swollen", "tendon", "bone", "finger", "elbow", "achilles", "arm",
        "overhead", "kneeling", "swelling",
    ),
}


def _class_signal(text: str) -> dict[str, float]:
    tokens = {t.lower().strip(".,;:!?") for t in text.split()}
    return {klass: sum(1 for kw in kws if kw in tokens)
            for klass, kws in SYMPTOM_KEYWORDS.items()}


def _det_noise(prompt: str, sample_idx: int) -> float:
    h = hashlib.sha1(f"{prompt}||{sample_idx}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return 2.0 * v - 1.0


def symptom_mock_llm(
    *,
    has_fewshot_signal: bool = True,
    sample_noise: float = 2.8,
) -> Callable:
    """Offline mock LLM keyed to ``SYMPTOM_KEYWORDS``.

    Shape mirrors ``toy_classification.mock_llm``: the returned callable
    plugs into ``MockLLMPredictor.llm_fn``. Base signal comes from the
    query's medical-term overlap with each class; matching-labelled
    few-shots add a bonus; per-sample noise lets voting/self-consistency
    meaningfully raise accuracy.

    Tuned so a bare harness gets ~0.5-0.6 accuracy and a full harness
    (retrieval + few-shot + vote) climbs to ~0.80-0.90 — enough dynamic
    range for the search to have headroom on 20-item eval.
    """
    def llm_fn(prompt: str, ctx: Context, harness: Harness, sample_idx: int):
        base_signal = _class_signal(ctx.example.input)
        bonus = {k: 0.0 for k in SYMPTOM_CLASSES}
        if has_fewshot_signal:
            for fs in ctx.few_shots:
                if fs.label == ctx.example.label:
                    bonus[fs.label] += 1.0
                else:
                    bonus[fs.label] += 0.3
        noisy = {
            k: base_signal.get(k, 0.0) + bonus[k]
            + sample_noise * _det_noise(prompt + k, sample_idx)
            for k in SYMPTOM_CLASSES
        }
        pred = max(noisy, key=lambda k: noisy[k])
        return pred, 6, 3.0  # plausibly realistic token/latency proxies
    return llm_fn

"""Math task wrapper for open-ended generation benchmarks (GSM8K, MATH, AIME).

The standard Task abstraction is built around classification with a fixed
``classes: list[str]``. Math benchmarks are open-ended generation: the
LLM produces a numerical answer in arbitrary phrasing, and scoring
compares the *extracted* answer against the gold.

This module provides:
- ``build_gsm8k_task(n_train, n_eval)``: GSM8K loader producing a Task
  whose classes list is the *set of gold answers seen* in the eval+train
  splits (so the existing LLMPredictor's class-matching still works
  numerically — gold "18" matches predicted "18" by string equality).
- ``extract_math_answer(text)``: parser that pulls the last numeric
  expression from arbitrary text. Handles ``#### N`` (GSM8K format),
  ``$N``, ``boxed{N}`` (MATH format), and bare trailing numbers.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..task import Task, TaskExample


# Regex patterns ordered by specificity. Highest-confidence patterns first.
_GSM8K_FINAL = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")  # #### 42
_BOXED = re.compile(r"\\?boxed\{(-?\d+(?:\.\d+)?)\}")    # \boxed{42}
_DOLLAR_AMT = re.compile(r"\$\s*(-?\d+(?:\.\d+)?)")      # $42
_LAST_NUMBER = re.compile(r"(-?\d+(?:\.\d+)?)")           # any number


def extract_math_answer(text: str) -> str:
    """Extract a numerical answer from arbitrary text. Returns empty string
    if no number is found.

    Tries patterns in order of specificity. For ``LAST_NUMBER`` we use
    ``findall`` and take the last match (LLMs often put the final answer
    last after working through the problem)."""
    if not text:
        return ""
    text = text.strip()

    m = _GSM8K_FINAL.search(text)
    if m:
        return _normalize_num(m.group(1))
    m = _BOXED.search(text)
    if m:
        return _normalize_num(m.group(1))
    dollars = _DOLLAR_AMT.findall(text)
    if dollars:
        return _normalize_num(dollars[-1])
    matches = _LAST_NUMBER.findall(text)
    if matches:
        return _normalize_num(matches[-1])
    return ""


def _normalize_num(s: str) -> str:
    """Strip trailing zeros after decimal, return integer-form when possible."""
    s = s.strip().lstrip("+")
    try:
        n = float(s)
    except ValueError:
        return s
    if n == int(n):
        return str(int(n))
    return ("%g" % n)


def _parse_gsm8k_answer(answer: str) -> str:
    """Parse the gold answer from GSM8K's ``answer`` field.

    GSM8K format: rationale + ``#### N`` on the last line. Returns ``N``."""
    m = _GSM8K_FINAL.search(answer)
    return _normalize_num(m.group(1)) if m else ""


def build_gsm8k_task(n_train: int = 80, n_eval: int = 50, seed: int = 0) -> Task:
    """Load GSM8K (openai/gsm8k, 'main' subset) and build a Task.

    Returns a Task whose ``classes`` is the set of unique gold answers
    seen in train+eval. ``input`` is the question; ``label`` is the
    extracted numerical gold answer (string form).

    Cost-control: by default we sample 80 train + 50 eval for budget;
    the original GSM8K test split has 1319 rows.

    Note: This is an open-ended generation task. The standard
    LLMPredictor's case-insensitive class match works *if* the LLM
    outputs the exact number as its final token. For better extraction,
    pair this task with a CoT formatter and use ``extract_math_answer``
    in the scorer's prediction parsing (see MathLLMPredictor)."""
    import random
    from datasets import load_dataset
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)

    train_idx = list(range(len(ds_train)))
    rng.shuffle(train_idx)
    eval_idx = list(range(len(ds_test)))
    rng.shuffle(eval_idx)

    train_items: list[TaskExample] = []
    for i in train_idx[:n_train]:
        row = ds_train[i]
        gold = _parse_gsm8k_answer(row["answer"])
        if gold:
            train_items.append(TaskExample(
                input=row["question"], label=gold,
            ))
    eval_items: list[TaskExample] = []
    for i in eval_idx[:n_eval]:
        row = ds_test[i]
        gold = _parse_gsm8k_answer(row["answer"])
        if gold:
            eval_items.append(TaskExample(
                input=row["question"], label=gold,
            ))

    classes = sorted({e.label for e in train_items} | {e.label for e in eval_items})
    return Task(
        name="gsm8k",
        train=train_items,
        eval_set=eval_items,
        classes=classes,
    )

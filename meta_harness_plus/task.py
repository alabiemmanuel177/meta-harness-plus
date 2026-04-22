"""Task abstraction: a labelled dataset the harness is evaluated on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TaskExample:
    """One evaluation datum: an input and its reference label."""
    input: str
    label: str
    # Optional per-example context (e.g. retrieval corpus id). Kept free-form.
    meta: tuple = ()


class Task:
    """A task exposes a training set (usable by harness components) and an eval set."""

    def __init__(
        self,
        name: str,
        train: Sequence[TaskExample],
        eval_set: Sequence[TaskExample],
        classes: Sequence[str],
    ):
        self.name = name
        self.train = list(train)
        self.eval_set = list(eval_set)
        self.classes = list(classes)

    def screen_subset(self, size: int, seed: int = 0) -> list[TaskExample]:
        """Return a deterministic small subset for cheap screening evals.

        Deterministic given (size, seed) so the screen is reproducible across
        candidates — fair comparison during successive halving.
        """
        import random
        rng = random.Random(seed)
        idx = list(range(len(self.eval_set)))
        rng.shuffle(idx)
        return [self.eval_set[i] for i in idx[:size]]

"""Component registry: JSON specs → Component instances.

Lets an LLM propose harnesses as structured JSON like::

    {"kind": "retriever", "name": "bow_retriever", "config": {"k": 3}}

instead of as free-form Python. We lose a bit of expressiveness vs the
original paper's code-generation approach, but gain safety (no arbitrary
code exec), validation (unknown names are rejected rather than crashing at
runtime), and analyzability (every proposal is diff-able as data).

The ``default_registry()`` factory is task-aware — it takes the ``Task`` so
retrievers can point at ``task.train`` without the LLM having to know about
runtime objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..components import (
    BagOfWordsRetriever,
    MajorityVoter,
    MockLLMPredictor,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
)
from ..harness import Component, Harness
from ..task import Task


Factory = Callable[[dict[str, Any]], Component]


@dataclass
class Entry:
    kind: str
    name: str
    factory: Factory
    required_fields: tuple[str, ...] = ()
    allowed_fields: tuple[str, ...] = ()


class ComponentRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Entry] = {}

    def register(self, entry: Entry) -> None:
        self._entries[(entry.kind, entry.name)] = entry

    def available(self) -> list[tuple[str, str]]:
        return sorted(self._entries.keys())

    def available_for_kind(self, kind: str) -> list[str]:
        return sorted(name for (k, name) in self._entries if k == kind)

    def instantiate(self, spec: dict[str, Any]) -> Component:
        kind = spec.get("kind")
        name = spec.get("name")
        if kind is None or name is None:
            raise ValueError(f"spec missing kind/name: {spec}")
        key = (kind, name)
        if key not in self._entries:
            raise ValueError(
                f"unknown component kind={kind!r} name={name!r}. "
                f"Available: {self.available_for_kind(kind) or list(self._entries.keys())}"
            )
        entry = self._entries[key]
        config = spec.get("config", {}) or {}
        for f in entry.required_fields:
            if f not in config:
                raise ValueError(f"{kind}/{name} missing required config field {f!r}")
        if entry.allowed_fields:
            for f in config:
                if f not in entry.allowed_fields:
                    raise ValueError(
                        f"{kind}/{name} config contains disallowed field {f!r}; "
                        f"allowed: {entry.allowed_fields}"
                    )
        return entry.factory(config)

    def build_harness(self, specs: list[dict[str, Any]]) -> Harness:
        return Harness(components=[self.instantiate(s) for s in specs])


def default_registry(task: Task, llm_fn) -> ComponentRegistry:
    """Register the shipped toy-harness components.

    ``llm_fn`` is the callable used by ``MockLLMPredictor`` — in the
    llm-proposer branch, an ``LLMPredictor`` may also be registered separately
    so the LLM can propose real-API-backed harnesses.
    """
    reg = ComponentRegistry()

    # Retrievers
    reg.register(Entry(
        kind="retriever", name="null_retriever",
        factory=lambda cfg: NullRetriever(),
    ))
    reg.register(Entry(
        kind="retriever", name="bow_retriever",
        factory=lambda cfg: BagOfWordsRetriever(corpus=task.train, k=int(cfg.get("k", 3))),
        allowed_fields=("k",),
    ))

    # Few-shot
    reg.register(Entry(
        kind="fewshot", name="null_fewshot",
        factory=lambda cfg: NullFewShot(),
    ))
    reg.register(Entry(
        kind="fewshot", name="topk_fewshot",
        factory=lambda cfg: TopKFewShot(k=int(cfg.get("k", 2))),
        allowed_fields=("k",),
    ))

    # Formatter
    reg.register(Entry(
        kind="formatter", name="simple_formatter",
        factory=lambda cfg: SimpleFormatter(system_hint=cfg.get("system_hint", "Classify the input.")),
        allowed_fields=("system_hint",),
    ))

    # Predictor (mock)
    reg.register(Entry(
        kind="predictor", name="mock_llm_predictor",
        factory=lambda cfg: MockLLMPredictor(
            llm_fn=llm_fn, n_samples=int(cfg.get("n_samples", 1)),
        ),
        allowed_fields=("n_samples",),
    ))

    # Voter
    reg.register(Entry(
        kind="voter", name="null_voter",
        factory=lambda cfg: NullVoter(),
    ))
    reg.register(Entry(
        kind="voter", name="majority_voter",
        factory=lambda cfg: MajorityVoter(),
    ))

    return reg

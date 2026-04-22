"""Harness = ordered composition of Components with shared state.

A Component takes the evolving ``Context`` and mutates it. The final Context
carries the prediction, plus cost accounting (tokens, latency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .task import TaskExample


@dataclass
class Context:
    """Per-example execution context threaded through every Component."""
    example: TaskExample
    # Components read/write these:
    retrieved: list[TaskExample] = field(default_factory=list)
    few_shots: list[TaskExample] = field(default_factory=list)
    prompt: str = ""
    candidate_predictions: list[str] = field(default_factory=list)
    prediction: str | None = None
    # Cost accounting:
    tokens: int = 0
    latency_ms: float = 0.0
    # Free-form slot for components to stash intermediate state:
    scratch: dict[str, Any] = field(default_factory=dict)


class Component:
    """Base class. Subclasses override ``name`` and ``run``.

    A Component has a ``kind`` — the slot it fills in a harness (e.g. 'retriever',
    'voter'). Two components with the same kind are swappable, which is what
    lets attribution's drop-one ablation work cleanly.
    """
    kind: str = "component"
    name: str = "component"

    def run(self, ctx: Context, harness: "Harness") -> None:
        raise NotImplementedError

    def config(self) -> dict:
        """Serializable config — used for logging + equality."""
        return {"kind": self.kind, "name": self.name}


@dataclass
class Harness:
    """Ordered pipeline of Components. Deterministic given a fixed LLM."""
    components: list[Component]
    id: str = ""  # filled by logger

    def run(self, example: TaskExample) -> Context:
        ctx = Context(example=example)
        for comp in self.components:
            comp.run(ctx, self)
        if ctx.prediction is None:
            # No Voter / Formatter set a prediction — fall back to any candidate.
            ctx.prediction = ctx.candidate_predictions[0] if ctx.candidate_predictions else ""
        return ctx

    def kinds(self) -> list[str]:
        return [c.kind for c in self.components]

    def describe(self) -> list[dict]:
        return [c.config() for c in self.components]

    def swap(self, kind: str, replacement: Component) -> "Harness":
        """Return a new Harness with the (first) component of ``kind`` replaced."""
        new_components = list(self.components)
        for i, c in enumerate(new_components):
            if c.kind == kind:
                new_components[i] = replacement
                break
        return Harness(components=new_components)

    def drop(self, kind: str) -> "Harness":
        """Return a new Harness with components of ``kind`` removed."""
        return Harness(components=[c for c in self.components if c.kind != kind])

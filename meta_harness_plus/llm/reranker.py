"""LLM-backed reranker: uses an LLMClient to pick the most relevant
retrieved items for the query.

Lives in the llm/ subpackage because it depends on LLMClient. Registered
alongside the zero-dep NullReranker / DiversityReranker from components.py
in ``llm_search_registry``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..components import Reranker
from ..harness import Context, Harness
from .client import LLMClient


@dataclass
class LLMReranker(Reranker):
    """Ask the LLM to pick the top-m most query-relevant items from
    ``ctx.retrieved`` and reorder accordingly.

    One extra LLM call per eval item — real token + latency cost that the
    Pareto scorer accounts for. The claim the search gets to test: is the
    reranker's accuracy lift worth its per-item cost?

    Output-parsing is forgiving: we look for the last non-empty line in
    the LLM response and pull any integers in [0, len(retrieved)) as
    picked indices. If parsing fails, the retrieved list passes through
    unchanged (degrades to NullReranker).
    """
    client: LLMClient
    m: int = 3
    name: str = "llm_reranker"
    max_tokens: int = 256
    temperature: float = 0.0
    kind: str = field(default="reranker", init=False)

    def run(self, ctx: Context, harness: Harness) -> None:
        if not ctx.retrieved or self.m >= len(ctx.retrieved):
            return
        items_text = "\n".join(
            f"[{i}] ({item.label}) {item.input}"
            for i, item in enumerate(ctx.retrieved)
        )
        system = (
            f"Rerank retrieved examples by relevance to the query. Output "
            f"ONLY the top {self.m} indices as space-separated integers on "
            f"the final line. No other prose."
        )
        user = f"Query: {ctx.example.input}\n\nCandidates:\n{items_text}"
        resp = self.client.complete(
            system=system, user=user,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )
        ctx.tokens += resp.total_tokens
        ctx.latency_ms += resp.latency_ms

        # Parse indices from the last non-empty line.
        lines = [ln.strip() for ln in resp.text.strip().splitlines() if ln.strip()]
        indices: list[int] = []
        if lines:
            for tok in lines[-1].split():
                tok = tok.strip(",.:;[]()")
                if tok.lstrip("-").isdigit():
                    idx = int(tok)
                    if 0 <= idx < len(ctx.retrieved) and idx not in indices:
                        indices.append(idx)
        if not indices:
            return  # degrade to no-op; ctx.retrieved unchanged
        picked: list = [ctx.retrieved[i] for i in indices[: self.m]]
        rest: list = [ctx.retrieved[i] for i in range(len(ctx.retrieved))
                      if i not in set(indices[: self.m])]
        ctx.retrieved = picked + rest

    def config(self) -> dict:
        return {"kind": self.kind, "name": self.name,
                "m": self.m, "temperature": self.temperature}

"""Predictor component backed by a real LLMClient.

The LLM is prompted as a classifier: produce one of ``classes`` verbatim on
the last line. We parse the last non-empty line, case-insensitively match
against ``classes``, and fall back to the first class if parsing fails.

Real costs (input_tokens + output_tokens, latency_ms) are accounted into
``Context`` — so Pareto scoring reflects actual API consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..harness import Component, Context, Harness
from .client import LLMClient


def _parse_class(text: str, classes: Sequence[str]) -> str:
    """Case-insensitive match of the last non-empty line against ``classes``.

    If the LLM returns ``"Thinking...\\n\\nsports"``, the last line wins.
    If no line matches, fall back to the first class that appears anywhere
    in the text; failing that, return classes[0].
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    lower_classes = {c.lower(): c for c in classes}
    if lines:
        last = lines[-1].lower()
        if last in lower_classes:
            return lower_classes[last]
        # Try the first word of the last line.
        first_tok = last.split()[0] if last.split() else ""
        if first_tok in lower_classes:
            return lower_classes[first_tok]
    # Fallback: look for any class name anywhere in the text.
    low = text.lower()
    for c_low, c_orig in lower_classes.items():
        if c_low in low:
            return c_orig
    return classes[0]


@dataclass
class LLMPredictor(Component):
    """Predictor that calls a real LLM.

    ``n_samples`` drives self-consistency (each sample is an independent call).
    Temperature ``> 0`` is recommended when ``n_samples > 1``.
    """
    client: LLMClient
    classes: Sequence[str]
    name: str = "llm_predictor"
    n_samples: int = 1
    temperature: float = 0.0
    # Default large enough for reasoning models (gpt-oss, r1, etc.) to finish
    # their chain-of-thought before emitting the class label. For non-reasoning
    # models this is safe overhead, not wasted tokens — they stop early.
    max_tokens: int = 512
    system_prompt_suffix: str = (
        "Respond with exactly one class name on the final line, "
        "lowercase, no punctuation. No prose."
    )
    kind: str = field(default="predictor", init=False)

    def run(self, ctx: Context, harness: Harness) -> None:
        system = self.system_prompt_suffix + f"\nValid classes: {', '.join(self.classes)}."
        preds: list[str] = []
        for _ in range(self.n_samples):
            resp = self.client.complete(
                system=system,
                user=ctx.prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            preds.append(_parse_class(resp.text, self.classes))
            ctx.tokens += resp.total_tokens
            ctx.latency_ms += resp.latency_ms
        ctx.candidate_predictions = preds

    def config(self) -> dict:
        return {
            "kind": self.kind, "name": self.name,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
        }

"""Built-in Components for the reference classification harness.

Every concrete Component declares a ``kind`` so attribution's swap/drop can
reason about slots. Baselines (``NullRetriever``, ``NullVoter``, …) are what
attribution swaps in when ablating a component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .harness import Component, Context, Harness
from .task import TaskExample


# ---------- Retriever ----------

class Retriever(Component):
    kind = "retriever"


class NullRetriever(Retriever):
    """Baseline: no retrieval. Zero token cost, zero latency."""
    name = "null_retriever"

    def run(self, ctx: Context, harness: Harness) -> None:
        ctx.retrieved = []
        ctx.tokens += 0
        ctx.latency_ms += 0.0


@dataclass
class BagOfWordsRetriever(Retriever):
    """Token-overlap retrieval from the training set. k configurable."""
    name: str = "bow_retriever"
    k: int = 3
    corpus: Sequence[TaskExample] = field(default_factory=list)
    kind: str = field(default="retriever", init=False)

    def _score(self, q: str, d: str) -> int:
        q_tokens = set(q.lower().split())
        d_tokens = set(d.lower().split())
        return len(q_tokens & d_tokens)

    def run(self, ctx: Context, harness: Harness) -> None:
        scored = [(self._score(ctx.example.input, e.input), e) for e in self.corpus]
        scored.sort(key=lambda t: t[0], reverse=True)
        ctx.retrieved = [e for s, e in scored[: self.k] if s > 0]
        # Rough cost: ~10 tokens per retrieved example, 1ms of latency per retrieval.
        ctx.tokens += 10 * len(ctx.retrieved)
        ctx.latency_ms += 1.0 * len(ctx.retrieved)

    def config(self) -> dict:
        return {"kind": self.kind, "name": self.name, "k": self.k}


# ---------- FewShot selector ----------

class FewShotSelector(Component):
    kind = "fewshot"


class NullFewShot(FewShotSelector):
    name = "null_fewshot"

    def run(self, ctx: Context, harness: Harness) -> None:
        ctx.few_shots = []


@dataclass
class TopKFewShot(FewShotSelector):
    """Use the top-K retrieved examples as few-shots (with label leakage)."""
    name: str = "topk_fewshot"
    k: int = 2
    kind: str = field(default="fewshot", init=False)

    def run(self, ctx: Context, harness: Harness) -> None:
        ctx.few_shots = ctx.retrieved[: self.k]
        ctx.tokens += 15 * len(ctx.few_shots)
        ctx.latency_ms += 0.5 * len(ctx.few_shots)

    def config(self) -> dict:
        return {"kind": self.kind, "name": self.name, "k": self.k}


# ---------- Formatter ----------

class Formatter(Component):
    kind = "formatter"


@dataclass
class SimpleFormatter(Formatter):
    """Build a prompt string. Real cost is the input-token count."""
    name: str = "simple_formatter"
    system_hint: str = "Classify the input."
    kind: str = field(default="formatter", init=False)

    def run(self, ctx: Context, harness: Harness) -> None:
        lines = [self.system_hint]
        for fs in ctx.few_shots:
            lines.append(f"Example: {fs.input} -> {fs.label}")
        lines.append(f"Query: {ctx.example.input}")
        ctx.prompt = "\n".join(lines)
        # Token-accounting proxy: 1 token per word.
        ctx.tokens += len(ctx.prompt.split())
        ctx.latency_ms += 2.0

    def config(self) -> dict:
        return {"kind": self.kind, "name": self.name, "system_hint": self.system_hint}


@dataclass
class CoTFormatter(SimpleFormatter):
    """Chain-of-thought formatter: same prompt skeleton as SimpleFormatter but
    with a system hint that elicits brief analysis before the class label.

    Real cost (reasoning tokens) is counted by the Predictor, not here —
    the formatter only sets up the prompt. Pair with ``LLMPredictor`` whose
    ``max_tokens`` is large enough to accommodate the extra reasoning.
    """
    name: str = "cot_formatter"
    system_hint: str = (
        "Classify the input into one of the given classes. "
        "Briefly analyze which key terms suggest each possibility, then output "
        "ONLY the class name, lowercase, on the final line. No other prose."
    )
    kind: str = field(default="formatter", init=False)


# ---------- Predictor ----------

class Predictor(Component):
    """Turns the prompt into candidate_predictions. The 'LLM call' lives here."""
    kind = "predictor"


@dataclass
class MockLLMPredictor(Predictor):
    """Deterministic mock LLM.

    Calls a pluggable scoring fn ``llm_fn(prompt, ctx, harness)`` that returns
    (prediction, tokens, latency_ms). In tests we plug in a fn that rewards
    harnesses that do retrieval + few-shot + voting.

    ``n_samples`` controls self-consistency — higher → more robust, more cost.
    """
    name: str = "mock_llm_predictor"
    llm_fn: Callable[[str, Context, Harness, int], tuple[str, int, float]] | None = None
    n_samples: int = 1
    kind: str = field(default="predictor", init=False)

    def run(self, ctx: Context, harness: Harness) -> None:
        assert self.llm_fn is not None, "llm_fn must be set"
        preds: list[str] = []
        for i in range(self.n_samples):
            pred, toks, lat = self.llm_fn(ctx.prompt, ctx, harness, i)
            preds.append(pred)
            ctx.tokens += toks
            ctx.latency_ms += lat
        ctx.candidate_predictions = preds

    def config(self) -> dict:
        return {"kind": self.kind, "name": self.name, "n_samples": self.n_samples}


# ---------- Voter ----------

class Voter(Component):
    kind = "voter"


class NullVoter(Voter):
    """Baseline: take the first candidate."""
    name = "null_voter"

    def run(self, ctx: Context, harness: Harness) -> None:
        if ctx.candidate_predictions:
            ctx.prediction = ctx.candidate_predictions[0]


class MajorityVoter(Voter):
    name = "majority_voter"

    def run(self, ctx: Context, harness: Harness) -> None:
        from collections import Counter
        if not ctx.candidate_predictions:
            return
        counts = Counter(ctx.candidate_predictions)
        ctx.prediction = counts.most_common(1)[0][0]
        # Voting is nearly free but not quite.
        ctx.latency_ms += 0.1


# ---------- Factory / baseline registry ----------

def baseline_for(kind: str) -> Component | None:
    """What attribution substitutes in when ablating ``kind``."""
    return {
        "retriever": NullRetriever(),
        "fewshot": NullFewShot(),
        "voter": NullVoter(),
        # Formatter baseline is SimpleFormatter — ablates CoT (and any other
        # non-default formatter variants) back to the plain prompt. Dropping
        # formatter entirely breaks the pipeline; swapping to plain is the
        # meaningful counterfactual for "did this formatter help?".
        "formatter": SimpleFormatter(),
        # No baseline for 'predictor' — it's the LLM call, can't be dropped.
    }.get(kind)

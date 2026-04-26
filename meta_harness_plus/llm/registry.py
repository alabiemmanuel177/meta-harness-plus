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
    BM25Retriever,
    BootstrapFewShot,
    CompressedCoTFormatter,
    CoTFormatter,
    DiversityReranker,
    MajorityVoter,
    MockLLMPredictor,
    NullFewShot,
    NullReranker,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TFIDFRetriever,
    TopKFewShot,
)
from ..harness import Component, Harness
from ..task import Task
from .client import LLMClient
from .predictor import LLMPredictor
from .reranker import LLMReranker


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
        # Pre-bootstrapped correct demos for BootstrapFewShot. Set externally
        # via set_bootstrap_demos() at search startup (one-time cost).
        self._bootstrap_demos: list = []

    def register(self, entry: Entry) -> None:
        self._entries[(entry.kind, entry.name)] = entry

    def set_bootstrap_demos(self, demos: list) -> None:
        """Store pre-bootstrapped demos for BootstrapFewShot. Idempotent."""
        self._bootstrap_demos = list(demos)

    def available(self) -> list[tuple[str, str]]:
        return sorted(self._entries.keys())

    def available_for_kind(self, kind: str) -> list[str]:
        return sorted(name for (k, name) in self._entries if k == kind)

    def entry(self, kind: str, name: str) -> Entry | None:
        return self._entries.get((kind, name))

    def describe_available(self) -> list[dict]:
        """Structured listing including allowed config fields per component.

        Used by ``LLMProposer`` to help the LLM emit config dicts with the
        right field names — the registry is strict about unknown fields, and
        reasoning models otherwise guess common-sense names like ``top_k``
        that don't match the actual factory signature.
        """
        out = []
        for (kind, name), e in sorted(self._entries.items()):
            out.append({
                "kind": kind,
                "name": name,
                "required_config_fields": list(e.required_fields),
                "allowed_config_fields": list(e.allowed_fields),
            })
        return out

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
    reg.register(Entry(
        kind="retriever", name="tfidf_retriever",
        factory=lambda cfg: TFIDFRetriever(corpus=task.train, k=int(cfg.get("k", 3))),
        allowed_fields=("k",),
    ))
    reg.register(Entry(
        kind="retriever", name="bm25_retriever",
        factory=lambda cfg: BM25Retriever(
            corpus=task.train, k=int(cfg.get("k", 3)),
            k1=float(cfg.get("k1", 1.5)),
            b=float(cfg.get("b", 0.75)),
        ),
        allowed_fields=("k", "k1", "b"),
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

    # Formatters
    reg.register(Entry(
        kind="formatter", name="simple_formatter",
        factory=lambda cfg: SimpleFormatter(system_hint=cfg.get("system_hint", "Classify the input.")),
        allowed_fields=("system_hint",),
    ))
    reg.register(Entry(
        kind="formatter", name="cot_formatter",
        factory=lambda cfg: CoTFormatter(system_hint=cfg["system_hint"]) if "system_hint" in cfg else CoTFormatter(),
        allowed_fields=("system_hint",),
    ))
    reg.register(Entry(
        kind="formatter", name="compressed_cot_formatter",
        factory=lambda cfg: CompressedCoTFormatter(
            max_reasoning_words=int(cfg.get("max_reasoning_words", 15)),
        ),
        allowed_fields=("max_reasoning_words",),
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


def llm_search_registry(
    task: Task,
    client: LLMClient,
    *,
    include_mock_predictor: bool = False,
    mock_llm_fn=None,
    predictor_max_tokens: int = 512,
) -> ComponentRegistry:
    """Registry for real-LLM-backed searches.

    Registers the same non-predictor components as ``default_registry`` plus
    an ``llm_predictor`` entry that uses ``client`` and ``task.classes``.
    Mock predictor is deliberately omitted unless ``include_mock_predictor``
    is set — we want all harnesses on the frontier to use real costs.
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
    reg.register(Entry(
        kind="retriever", name="tfidf_retriever",
        factory=lambda cfg: TFIDFRetriever(corpus=task.train, k=int(cfg.get("k", 3))),
        allowed_fields=("k",),
    ))
    reg.register(Entry(
        kind="retriever", name="bm25_retriever",
        factory=lambda cfg: BM25Retriever(
            corpus=task.train, k=int(cfg.get("k", 3)),
            k1=float(cfg.get("k1", 1.5)),
            b=float(cfg.get("b", 0.75)),
        ),
        allowed_fields=("k", "k1", "b"),
    ))

    # Reranker (richer action space beyond RAG)
    reg.register(Entry(
        kind="reranker", name="null_reranker",
        factory=lambda cfg: NullReranker(),
    ))
    reg.register(Entry(
        kind="reranker", name="diversity_reranker",
        factory=lambda cfg: DiversityReranker(),
    ))
    reg.register(Entry(
        kind="reranker", name="llm_reranker",
        factory=lambda cfg: LLMReranker(
            client=client,
            m=int(cfg.get("m", 3)),
            temperature=float(cfg.get("temperature", 0.0)),
        ),
        allowed_fields=("m", "temperature"),
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
    # DSPy-style bootstrap fewshot — uses pre-bootstrapped correct demos
    # if available; falls back to top-k retrieval otherwise.
    reg.register(Entry(
        kind="fewshot", name="bootstrap_fewshot",
        factory=lambda cfg: BootstrapFewShot(
            k=int(cfg.get("k", 4)),
            demos_pool=tuple(reg._bootstrap_demos),
        ),
        allowed_fields=("k",),
    ))

    # Formatters — coerce system_hint to str defensively (LLM proposers
    # sometimes emit ``"system_hint": true`` in their JSON which would
    # otherwise crash the formatter mid-run).
    reg.register(Entry(
        kind="formatter", name="simple_formatter",
        factory=lambda cfg: SimpleFormatter(system_hint=str(cfg.get("system_hint", "Classify the input."))),
        allowed_fields=("system_hint",),
    ))
    reg.register(Entry(
        kind="formatter", name="cot_formatter",
        factory=lambda cfg: CoTFormatter(system_hint=str(cfg["system_hint"])) if "system_hint" in cfg else CoTFormatter(),
        allowed_fields=("system_hint",),
    ))
    reg.register(Entry(
        kind="formatter", name="compressed_cot_formatter",
        factory=lambda cfg: CompressedCoTFormatter(
            max_reasoning_words=int(cfg.get("max_reasoning_words", 15)),
        ),
        allowed_fields=("max_reasoning_words",),
    ))

    # Predictor — real LLM
    reg.register(Entry(
        kind="predictor", name="llm_predictor",
        factory=lambda cfg: LLMPredictor(
            client=client,
            classes=task.classes,
            n_samples=int(cfg.get("n_samples", 1)),
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=predictor_max_tokens,
        ),
        allowed_fields=("n_samples", "temperature"),
    ))

    if include_mock_predictor and mock_llm_fn is not None:
        reg.register(Entry(
            kind="predictor", name="mock_llm_predictor",
            factory=lambda cfg: MockLLMPredictor(
                llm_fn=mock_llm_fn, n_samples=int(cfg.get("n_samples", 1)),
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

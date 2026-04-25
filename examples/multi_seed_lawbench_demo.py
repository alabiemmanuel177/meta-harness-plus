"""Dogfood the tier-1 multi-seed runner on the bundled LawBench fixture.

This is a deterministic, no-LLM-needed demonstration that the
``MultiSeedRunner`` + bootstrap CI + held-out test infrastructure works
end-to-end. Uses a generic keyword mock LLM keyed to the legal-fixture's
class vocabulary.

Run::

    python3 examples/multi_seed_lawbench_demo.py

Expected output: 5 seeds, mean held-out accuracy with 95% CI, mean
held-out tokens. The CI width tells you how stable the search-best
harness is across seeds.
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.components import (
    BagOfWordsRetriever,
    BM25Retriever,
    DiversityReranker,
    MajorityVoter,
    MockLLMPredictor,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TFIDFRetriever,
    TopKFewShot,
)
from meta_harness_plus.harness import Context, Harness
from meta_harness_plus.multi_seed import MultiSeedRunner
from meta_harness_plus.runner import SearchConfig
from meta_harness_plus.search.mock_proposer import (
    AttributionGuidedMutationProposer,
)
from meta_harness_plus.tasks.lawbench import build_lawbench_fixture_task


# Class-keyword cues for the synthetic legal fixture. Lets a deterministic
# mock LLM produce realistic per-class scoring without an LLM call.
LEGAL_KEYWORDS = {
    "contract":  ("contract", "lease", "agreement", "warranty", "terms",
                  "license", "loan", "breach", "vendor", "buyer", "borrower"),
    "tort":      ("negligent", "negligence", "injured", "injury", "damage",
                  "fall", "infection", "doctor", "hospital", "patient"),
    "criminal":  ("felony", "robbery", "homicide", "charges", "evidence",
                  "embezzlement", "search", "arson", "interrogation"),
    "family":    ("divorce", "custody", "marital", "adoption", "child",
                  "parental", "estate", "trust", "support"),
    "ip":        ("patent", "trademark", "copyright", "trade", "infringement",
                  "trade-secret", "secret", "license", "publisher", "design"),
}


def _det_noise(prompt: str, sample_idx: int) -> float:
    h = hashlib.sha1(f"{prompt}||{sample_idx}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return 2.0 * v - 1.0


def legal_mock_llm(*, sample_noise: float = 1.5):
    classes = tuple(LEGAL_KEYWORDS.keys())

    def llm_fn(prompt: str, ctx: Context, harness: Harness, sample_idx: int):
        text = ctx.example.input.lower()
        signal = {c: sum(1 for kw in LEGAL_KEYWORDS[c] if kw in text)
                  for c in classes}
        bonus = {c: 0.0 for c in classes}
        for fs in ctx.few_shots:
            if fs.label in bonus:
                if fs.label == ctx.example.label:
                    bonus[fs.label] += 0.8
                else:
                    bonus[fs.label] += 0.25
        noisy = {c: signal[c] + bonus[c] + sample_noise * _det_noise(prompt + c, sample_idx)
                 for c in classes}
        pred = max(noisy, key=lambda k: noisy[k])
        return pred, 6, 2.0

    return llm_fn


def main():
    task = build_lawbench_fixture_task()
    llm = legal_mock_llm()

    mutators = {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=3),
            lambda: TFIDFRetriever(corpus=task.train, k=3),
            lambda: BM25Retriever(corpus=task.train, k=3),
        ],
        "reranker": [lambda: DiversityReranker()],  # always-on diversity
        "fewshot": [lambda: NullFewShot(), lambda: TopKFewShot(k=2)],
        "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
        "predictor": [
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=3),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=5),
        ],
    }

    seed_harness = Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm, n_samples=1), NullVoter(),
    ])

    runner = MultiSeedRunner(
        task=task,
        proposer_factory=lambda s: AttributionGuidedMutationProposer(
            mutators=mutators, seed=s, depth_weights={1: 0.3, 2: 0.5, 3: 0.2},
        ),
        config=SearchConfig(
            n_iterations=4, proposals_per_iter=4, screen_size=4,
            halving_k0=2, halving_eta=2, halving_final_keep=1,
        ),
        seed_harnesses=[seed_harness],
        holdout_frac=0.3,
        holdout_seed=0,
        n_eval_repeats_holdout=2,
    )

    seeds = [0, 1, 2, 3, 4]
    result = runner.run(seeds)
    print(f"multi-seed search complete: seeds={seeds}, task={task.name}, "
          f"holdout={result.holdout_size} of {len(task.eval_set)}")

    print("\n=== Per-seed held-out accuracy ===")
    for sr in result.per_seed:
        if sr.holdout_scores:
            cid, hs = sr.holdout_scores[0]
            print(f"  seed {sr.seed}: best harness = {cid}  "
                  f"holdout acc={hs.accuracy:.3f}  tokens={hs.tokens:.1f}")

    print("\n" + result.summary())

    print("\n=== Bootstrap confidence intervals ===")
    print(f"  held-out accuracy: mean={result.best_acc_ci.mean:.3f}  "
          f"95% CI=[{result.best_acc_ci.low:.3f}, {result.best_acc_ci.high:.3f}]  "
          f"width={result.best_acc_ci.high - result.best_acc_ci.low:.3f}")
    print(f"  held-out tokens:   mean={result.best_cost_at_top_acc_ci.mean:.1f}  "
          f"95% CI=[{result.best_cost_at_top_acc_ci.low:.1f}, "
          f"{result.best_cost_at_top_acc_ci.high:.1f}]")


if __name__ == "__main__":
    main()

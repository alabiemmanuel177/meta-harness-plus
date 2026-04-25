"""Ablation study: full MH++ vs C1-off vs C2-off vs C3-off.

Demonstrates the ablation framework end-to-end on the deterministic
toy task with the mock LLM. No network required. Outputs a table
showing how each ablation hurts (or doesn't) compared to full MH++.

Usage:
    python3 examples/run_ablation_study.py [--seeds 0 1 2 3 4]

Reviewer-protective: answers "did C1 help? did C2? did C3?" with
data, not vibes.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.ablations import (
    AblationConfig,
    AblationResult,
    make_frontier,
    summarize_frontier,
)
from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.components import (
    BagOfWordsRetriever,
    MajorityVoter,
    MockLLMPredictor,
    NullFewShot,
    NullRetriever,
    NullVoter,
    SimpleFormatter,
    TopKFewShot,
    baseline_for,
)
from meta_harness_plus.harness import Harness
from meta_harness_plus.runner import SearchConfig, SearchRunner
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.search.mock_proposer import (
    AttributionGuidedMutationProposer,
)
from meta_harness_plus.search.random_proposer import RandomProposer
from meta_harness_plus.tasks import build_toy_task, mock_llm


def _mutators(task, llm):
    return {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=2),
            lambda: BagOfWordsRetriever(corpus=task.train, k=3),
            lambda: BagOfWordsRetriever(corpus=task.train, k=5),
        ],
        "fewshot": [lambda: NullFewShot(),
                    lambda: TopKFewShot(k=1),
                    lambda: TopKFewShot(k=2),
                    lambda: TopKFewShot(k=3)],
        "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
        "predictor": [lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
                      lambda: MockLLMPredictor(llm_fn=llm, n_samples=3),
                      lambda: MockLLMPredictor(llm_fn=llm, n_samples=5)],
    }


def _seed_harness(task, llm) -> Harness:
    return Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm, n_samples=1), NullVoter(),
    ])


def run_one_ablation(
    config: AblationConfig,
    task,
    llm,
    seed: int,
    iterations: int,
    proposals: int,
    eval_size: int,
) -> AblationResult:
    """Execute one ablation condition with one seed. Returns the summary."""
    scorer = Scorer(task)
    attribution = AttributionTracker(scorer, baseline_for)
    mutators = _mutators(task, llm)

    # C3 ablation: random proposer instead of attribution-guided.
    if config.disable_attribution:
        proposer = RandomProposer(mutators=mutators, seed=seed)
    else:
        proposer = AttributionGuidedMutationProposer(
            mutators=mutators, seed=seed,
            depth_weights={1: 0.3, 2: 0.5, 3: 0.2},
        )

    # C2 ablation: full-eval every candidate (kill halving).
    # Achieved by setting halving_k0 = full eval_size and final_keep
    # large enough that no halving happens.
    if config.disable_halving:
        cfg = SearchConfig(
            n_iterations=iterations,
            proposals_per_iter=proposals,
            screen_size=eval_size,           # full eval each candidate
            halving_k0=eval_size,            # ditto
            halving_eta=2,
            halving_final_keep=proposals,    # keep everyone — no halving
            full_eval_size=eval_size,
            screen_seed=seed,
        )
    else:
        cfg = SearchConfig(
            n_iterations=iterations,
            proposals_per_iter=proposals,
            screen_size=4,
            halving_k0=2,
            halving_eta=2,
            halving_final_keep=2,
            full_eval_size=eval_size,
            screen_seed=seed,
        )

    # C1 ablation handled by frontier_factory injection.
    runner = SearchRunner(
        task=task, scorer=scorer, proposer=proposer, attribution=attribution,
        config=cfg,
        seed_harnesses=[_seed_harness(task, llm)],
        frontier_factory=lambda: make_frontier(config),
    )
    state = runner.run()
    return summarize_frontier(config.label, state.frontier)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--proposals", type=int, default=4)
    ap.add_argument("--eval-size", type=int, default=20)
    args = ap.parse_args()

    task = build_toy_task(seed=0)
    llm = mock_llm()

    conditions = [
        AblationConfig(),                                  # full MH++
        AblationConfig(disable_pareto=True),               # C1 off
        AblationConfig(disable_halving=True),              # C2 off
        AblationConfig(disable_attribution=True),          # C3 off
    ]

    print(f"\n{'='*84}\n  Ablation study on toy task (mock LLM, no network)\n{'='*84}")
    print(f"  seeds={args.seeds}  iterations={args.iterations} × {args.proposals} proposals")
    print(f"  eval_size={args.eval_size}\n")

    table: dict[str, list[AblationResult]] = {}
    for cond in conditions:
        results = []
        for seed in args.seeds:
            r = run_one_ablation(cond, task, llm, seed,
                                 args.iterations, args.proposals, args.eval_size)
            results.append(r)
        table[cond.label] = results

    # Print summary table.
    print(f"{'Condition':<22} {'mean acc':>10} {'mean tok':>10} {'mean lat':>10} {'frontier':>10}")
    print("-" * 70)
    for label, results in table.items():
        n = len(results)
        mean_acc = sum(r.best_acc for r in results) / n
        mean_tok = sum(r.best_acc_tokens for r in results) / n
        mean_lat = sum(r.best_acc_latency_ms for r in results) / n
        mean_size = sum(r.n_frontier_points for r in results) / n
        print(f"{label:<22} {mean_acc:>10.3f} {mean_tok:>10.1f} {mean_lat:>10.1f} {mean_size:>10.1f}")

    print()
    full = table.get("full-MH++", [])
    if full:
        full_acc = sum(r.best_acc for r in full) / len(full)
        print("Δ (ablation - full MH++) on best accuracy:")
        for label, results in table.items():
            if label == "full-MH++":
                continue
            ab_acc = sum(r.best_acc for r in results) / len(results)
            delta = ab_acc - full_acc
            print(f"  {label:<22}  {delta:+.3f}")


if __name__ == "__main__":
    main()

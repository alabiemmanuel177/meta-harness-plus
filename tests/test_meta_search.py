"""Self-improving search (Tier 8.3) — tests for MetaCandidate + MetaScorer
+ meta_pareto_search.

Uses the deterministic toy task with mock LLM so tests run in <1s and
the meta-search can pick between known-different configs without an
LLM in the loop.
"""

from __future__ import annotations

import unittest

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
from meta_harness_plus.meta_search import (
    MetaCandidate,
    MetaScorer,
    MetaScoreVector,
    _meta_dominates,
    best_meta_by_accuracy,
    meta_pareto_search,
)
from meta_harness_plus.runner import SearchConfig
from meta_harness_plus.search.mock_proposer import (
    AttributionGuidedMutationProposer,
)
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestMetaScoreVector(unittest.TestCase):
    def test_objective_signs(self):
        self.assertEqual(MetaScoreVector.objective_signs(), (+1, -1, -1))

    def test_dominates_strictly_better(self):
        a = MetaScoreVector(peak_accuracy=0.9, search_compute=10, wall_seconds=1.0)
        b = MetaScoreVector(peak_accuracy=0.7, search_compute=20, wall_seconds=2.0)
        self.assertTrue(_meta_dominates(a, b))
        self.assertFalse(_meta_dominates(b, a))

    def test_no_dominance_on_tradeoff(self):
        # a more accurate but more compute; b less accurate but cheaper.
        a = MetaScoreVector(peak_accuracy=0.9, search_compute=100, wall_seconds=10.0)
        b = MetaScoreVector(peak_accuracy=0.7, search_compute=10, wall_seconds=1.0)
        self.assertFalse(_meta_dominates(a, b))
        self.assertFalse(_meta_dominates(b, a))


class TestMetaCandidate(unittest.TestCase):
    def test_equality_by_name(self):
        cfg = SearchConfig(n_iterations=1)
        a = MetaCandidate(name="a", config=cfg,
                          proposer_factory=lambda s: None)
        b = MetaCandidate(name="a", config=cfg,
                          proposer_factory=lambda s: None)
        c = MetaCandidate(name="c", config=cfg,
                          proposer_factory=lambda s: None)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_hashable(self):
        cfg = SearchConfig(n_iterations=1)
        cand = MetaCandidate(name="a", config=cfg,
                             proposer_factory=lambda s: None)
        s = {cand}
        self.assertIn(cand, s)


def _local_split_holdout(task, frac=0.3):
    """Inline holdout split — tier-8 branch doesn't depend on tier-1's
    multi_seed module. Same logic as multi_seed.split_holdout but local."""
    import random
    from collections import defaultdict
    from meta_harness_plus.task import Task
    rng = random.Random(0)
    by_class = defaultdict(list)
    for ex in task.eval_set:
        by_class[ex.label].append(ex)
    search_eval, holdout = [], []
    for items in by_class.values():
        idx = list(range(len(items)))
        rng.shuffle(idx)
        n_holdout = max(1, int(round(frac * len(items))))
        holdout_idx = set(idx[:n_holdout])
        for i, item in enumerate(items):
            (holdout if i in holdout_idx else search_eval).append(item)
    new_task = Task(
        name=f"{task.name}_search",
        train=task.train, eval_set=search_eval, classes=task.classes,
    )
    return new_task, holdout


def _toy_setup():
    task = build_toy_task(seed=0)
    llm = mock_llm()
    search_task, holdout = _local_split_holdout(task)
    mutators = {
        "retriever": [
            lambda: NullRetriever(),
            lambda: BagOfWordsRetriever(corpus=task.train, k=3),
        ],
        "fewshot": [lambda: NullFewShot(), lambda: TopKFewShot(k=2)],
        "voter": [lambda: NullVoter(), lambda: MajorityVoter()],
        "predictor": [
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=1),
            lambda: MockLLMPredictor(llm_fn=llm, n_samples=5),
        ],
    }
    seed_harness = Harness(components=[
        NullRetriever(), NullFewShot(), SimpleFormatter(),
        MockLLMPredictor(llm_fn=llm, n_samples=1), NullVoter(),
    ])
    return search_task, holdout, mutators, seed_harness


class TestMetaScorer(unittest.TestCase):
    def test_scores_a_single_candidate(self):
        task, holdout, mutators, seed = _toy_setup()
        scorer = MetaScorer(task=task, holdout=holdout, seed=0)
        cand = MetaCandidate(
            name="2iter_4prop",
            config=SearchConfig(
                n_iterations=2, proposals_per_iter=4, screen_size=4,
                halving_k0=2, halving_eta=2, halving_final_keep=1,
            ),
            proposer_factory=lambda s: AttributionGuidedMutationProposer(
                mutators=mutators, seed=s,
                depth_weights={1: 0.3, 2: 0.5, 3: 0.2},
            ),
            seed_harnesses_factory=lambda: [seed],
        )
        score = scorer.score(cand)
        self.assertIsInstance(score, MetaScoreVector)
        self.assertGreaterEqual(score.peak_accuracy, 0.0)
        self.assertLessEqual(score.peak_accuracy, 1.0)
        self.assertGreater(score.search_compute, 0)


class TestMetaParetoSearch(unittest.TestCase):
    def test_returns_non_dominated_set(self):
        task, holdout, mutators, seed = _toy_setup()
        scorer = MetaScorer(task=task, holdout=holdout, seed=0)
        candidates = [
            MetaCandidate(
                name="cheap",
                config=SearchConfig(
                    n_iterations=1, proposals_per_iter=2, screen_size=3,
                    halving_k0=2, halving_eta=2, halving_final_keep=1,
                ),
                proposer_factory=lambda s: AttributionGuidedMutationProposer(
                    mutators=mutators, seed=s,
                    depth_weights={1: 0.5, 2: 0.5},
                ),
                seed_harnesses_factory=lambda: [seed],
            ),
            MetaCandidate(
                name="moderate",
                config=SearchConfig(
                    n_iterations=2, proposals_per_iter=4, screen_size=4,
                    halving_k0=2, halving_eta=2, halving_final_keep=2,
                ),
                proposer_factory=lambda s: AttributionGuidedMutationProposer(
                    mutators=mutators, seed=s,
                    depth_weights={1: 0.3, 2: 0.5, 3: 0.2},
                ),
                seed_harnesses_factory=lambda: [seed],
            ),
            MetaCandidate(
                name="expensive",
                config=SearchConfig(
                    n_iterations=3, proposals_per_iter=6, screen_size=5,
                    halving_k0=2, halving_eta=2, halving_final_keep=2,
                ),
                proposer_factory=lambda s: AttributionGuidedMutationProposer(
                    mutators=mutators, seed=s,
                    depth_weights={1: 0.2, 2: 0.5, 3: 0.3},
                ),
                seed_harnesses_factory=lambda: [seed],
            ),
        ]
        survivors = meta_pareto_search(candidates, scorer)
        self.assertGreaterEqual(len(survivors), 1)
        # No survivor dominates another.
        for i, ri in enumerate(survivors):
            for j, rj in enumerate(survivors):
                if i != j:
                    self.assertFalse(_meta_dominates(ri.score, rj.score),
                                     f"{ri.candidate.name} dominates {rj.candidate.name}")

    def test_best_by_accuracy_picks_highest(self):
        task, holdout, mutators, seed = _toy_setup()
        scorer = MetaScorer(task=task, holdout=holdout, seed=0)
        candidates = [
            MetaCandidate(
                name=f"cfg_{i}",
                config=SearchConfig(
                    n_iterations=i + 1, proposals_per_iter=3, screen_size=3,
                    halving_k0=2, halving_eta=2, halving_final_keep=1,
                ),
                proposer_factory=lambda s: AttributionGuidedMutationProposer(
                    mutators=mutators, seed=s,
                    depth_weights={1: 0.3, 2: 0.5, 3: 0.2},
                ),
                seed_harnesses_factory=lambda: [seed],
            )
            for i in range(2)
        ]
        survivors = meta_pareto_search(candidates, scorer)
        best = best_meta_by_accuracy(survivors)
        self.assertIsNotNone(best)
        for s in survivors:
            self.assertGreaterEqual(best.score.peak_accuracy, s.score.peak_accuracy)


if __name__ == "__main__":
    unittest.main()

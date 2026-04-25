"""Verify LLMProposer prompt includes token-budget guidance (framework-upgrades).

The hint biases the proposer toward strict-Pareto-dominance shapes:
"hit accuracy >= X at tokens <= Y" is in the prompt for every iteration
where the frontier has at least one point.
"""

from __future__ import annotations

import unittest

from meta_harness_plus.llm.proposer import DiagnosticContext


class TestTokenBudgetHint(unittest.TestCase):
    def test_hint_present_when_frontier_has_points(self):
        ctx = DiagnosticContext(
            frontier=[
                {
                    "candidate_id": "rag",
                    "score": {"accuracy": 0.88, "tokens": 174, "latency_ms": 1095},
                    "describe": [
                        {"kind": "retriever", "name": "bow_retriever", "k": 3},
                        {"kind": "predictor", "name": "llm_predictor", "n_samples": 1},
                    ],
                },
                {
                    "candidate_id": "discovered",
                    "score": {"accuracy": 0.92, "tokens": 287, "latency_ms": 1405},
                    "describe": [
                        {"kind": "retriever", "name": "bm25_retriever", "k": 3},
                    ],
                },
            ],
            attribution={},
            available=[],
            exploration_gap=[],
            iteration=2,
            n_requested=4,
        )
        prompt = ctx.to_user_prompt()
        self.assertIn("Token-budget guidance", prompt)
        self.assertIn("STRICT WIN target", prompt)
        # Best accuracy point: 0.92 @ 287 tokens
        self.assertIn("0.92", prompt)
        # Cheapest point: 0.88 @ 174 tokens
        self.assertIn("174 tokens", prompt)

    def test_hint_omitted_when_frontier_empty(self):
        ctx = DiagnosticContext(
            frontier=[],
            attribution={},
            available=[],
            exploration_gap=[],
            iteration=0,
            n_requested=4,
        )
        prompt = ctx.to_user_prompt()
        # Without a frontier we can't compute targets — section omitted.
        self.assertNotIn("STRICT WIN target", prompt)

    def test_hint_bounds_correct_with_one_point(self):
        ctx = DiagnosticContext(
            frontier=[{
                "candidate_id": "only",
                "score": {"accuracy": 0.75, "tokens": 100, "latency_ms": 50},
                "describe": [],
            }],
            attribution={},
            available=[],
            exploration_gap=[],
            iteration=0,
            n_requested=1,
        )
        prompt = ctx.to_user_prompt()
        self.assertIn("100 tokens", prompt)
        # Best == cheapest when only one point.
        self.assertIn("0.75", prompt)


if __name__ == "__main__":
    unittest.main()

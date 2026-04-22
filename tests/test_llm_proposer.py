"""LLMProposer: filesystem read + JSON-parsing + registry validation."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from meta_harness_plus.attribution import AttributionTracker
from meta_harness_plus.components import baseline_for
from meta_harness_plus.llm.client import ScriptedClient
from meta_harness_plus.llm.proposer import LLMProposer
from meta_harness_plus.llm.registry import default_registry
from meta_harness_plus.pareto import ParetoFrontier
from meta_harness_plus.scorer import Scorer
from meta_harness_plus.tasks import build_toy_task, mock_llm


def _good_response() -> str:
    return json.dumps({
        "rationale": "Try full retrieval + few-shot + voting.",
        "proposals": [
            {"components": [
                {"kind": "retriever", "name": "bow_retriever", "config": {"k": 3}},
                {"kind": "fewshot", "name": "topk_fewshot", "config": {"k": 2}},
                {"kind": "formatter", "name": "simple_formatter", "config": {}},
                {"kind": "predictor", "name": "mock_llm_predictor", "config": {"n_samples": 5}},
                {"kind": "voter", "name": "majority_voter", "config": {}},
            ]},
            {"components": [
                {"kind": "retriever", "name": "bow_retriever", "config": {"k": 5}},
                {"kind": "fewshot", "name": "topk_fewshot", "config": {"k": 3}},
                {"kind": "formatter", "name": "simple_formatter", "config": {}},
                {"kind": "predictor", "name": "mock_llm_predictor", "config": {"n_samples": 3}},
                {"kind": "voter", "name": "majority_voter", "config": {}},
            ]},
        ],
    })


class TestLLMProposer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.task = build_toy_task(seed=0)
        self.llm_fn = mock_llm()
        self.scorer = Scorer(self.task)
        self.attr = AttributionTracker(self.scorer, baseline_for)
        self.registry = default_registry(self.task, self.llm_fn)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parses_valid_json_and_builds_harnesses(self):
        client = ScriptedClient([_good_response()])
        proposer = LLMProposer(client=client, registry=self.registry, run_dir=self.tmp)
        result = proposer.propose(frontier=ParetoFrontier(), attribution=self.attr, n=2)
        self.assertEqual(len(result.harnesses), 2)
        self.assertIn("full retrieval", result.rationale)

    def test_parses_fenced_markdown_response(self):
        fenced = f"Here's my plan:\n```json\n{_good_response()}\n```\nDone."
        client = ScriptedClient([fenced])
        proposer = LLMProposer(client=client, registry=self.registry, run_dir=self.tmp)
        result = proposer.propose(frontier=ParetoFrontier(), attribution=self.attr, n=2)
        self.assertEqual(len(result.harnesses), 2)

    def test_invalid_json_returns_parse_error(self):
        client = ScriptedClient(["not json at all"])
        proposer = LLMProposer(client=client, registry=self.registry, run_dir=self.tmp)
        result = proposer.propose(frontier=ParetoFrontier(), attribution=self.attr, n=2)
        self.assertEqual(len(result.harnesses), 0)
        self.assertIn("parse_error", result.rationale)

    def test_invalid_component_name_skipped(self):
        payload = json.dumps({
            "rationale": "",
            "proposals": [
                {"components": [  # bad: unknown retriever
                    {"kind": "retriever", "name": "magic", "config": {}},
                    {"kind": "fewshot", "name": "null_fewshot", "config": {}},
                    {"kind": "formatter", "name": "simple_formatter", "config": {}},
                    {"kind": "predictor", "name": "mock_llm_predictor", "config": {}},
                    {"kind": "voter", "name": "null_voter", "config": {}},
                ]},
                {"components": [  # good
                    {"kind": "retriever", "name": "null_retriever", "config": {}},
                    {"kind": "fewshot", "name": "null_fewshot", "config": {}},
                    {"kind": "formatter", "name": "simple_formatter", "config": {}},
                    {"kind": "predictor", "name": "mock_llm_predictor", "config": {}},
                    {"kind": "voter", "name": "null_voter", "config": {}},
                ]},
            ],
        })
        client = ScriptedClient([payload])
        proposer = LLMProposer(client=client, registry=self.registry, run_dir=self.tmp)
        result = proposer.propose(frontier=ParetoFrontier(), attribution=self.attr, n=2)
        # Only the valid proposal survives.
        self.assertEqual(len(result.harnesses), 1)
        self.assertIn("invalid", result.rationale)

    def test_reads_filesystem_state_into_prompt(self):
        # Seed the run_dir with a fake frontier + attribution stats.
        (Path(self.tmp) / "frontier.json").write_text(json.dumps([{
            "candidate_id": "cand_0001",
            "score": {"accuracy": 0.75, "tokens": 33.0, "latency_ms": 17.0, "n_evaluated": 20},
            "describe": [
                {"kind": "retriever", "name": "null_retriever"},
                {"kind": "fewshot", "name": "null_fewshot"},
                {"kind": "formatter", "name": "simple_formatter"},
                {"kind": "predictor", "name": "mock_llm_predictor", "n_samples": 5},
                {"kind": "voter", "name": "majority_voter"},
            ],
        }]))
        (Path(self.tmp) / "attribution_stats.json").write_text(json.dumps({
            "voter": {"kind": "voter", "n": 3, "mean_delta": 0.2, "m2": 0.01},
        }))

        captured: dict = {}

        def scripted(system: str, user: str) -> str:
            captured["user"] = user
            return _good_response()

        client = ScriptedClient(scripted)
        proposer = LLMProposer(client=client, registry=self.registry, run_dir=self.tmp)
        proposer.propose(frontier=ParetoFrontier(), attribution=self.attr, n=1)

        self.assertIn("cand_0001", captured["user"])
        self.assertIn("voter", captured["user"])
        self.assertIn("mean_delta=+0.200", captured["user"])
        # Exploration gap should appear too — bow_retriever is registered but
        # not on the seeded frontier.
        self.assertIn("retriever/bow_retriever", captured["user"])


if __name__ == "__main__":
    unittest.main()

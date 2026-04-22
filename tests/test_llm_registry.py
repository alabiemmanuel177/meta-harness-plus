"""Component registry: JSON spec round-trip + validation."""

from __future__ import annotations

import unittest

from meta_harness_plus.llm.registry import default_registry
from meta_harness_plus.tasks import build_toy_task, mock_llm


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.task = build_toy_task(seed=0)
        self.reg = default_registry(self.task, mock_llm())

    def test_available_covers_all_kinds(self):
        kinds = {k for k, _ in self.reg.available()}
        self.assertSetEqual(kinds, {"retriever", "fewshot", "formatter", "predictor", "voter"})

    def test_build_full_harness(self):
        h = self.reg.build_harness([
            {"kind": "retriever", "name": "bow_retriever", "config": {"k": 3}},
            {"kind": "fewshot", "name": "topk_fewshot", "config": {"k": 2}},
            {"kind": "formatter", "name": "simple_formatter", "config": {}},
            {"kind": "predictor", "name": "mock_llm_predictor", "config": {"n_samples": 3}},
            {"kind": "voter", "name": "majority_voter", "config": {}},
        ])
        self.assertEqual([c.kind for c in h.components],
                         ["retriever", "fewshot", "formatter", "predictor", "voter"])

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            self.reg.instantiate({"kind": "retriever", "name": "magic_retriever"})

    def test_missing_kind_raises(self):
        with self.assertRaises(ValueError):
            self.reg.instantiate({"name": "bow_retriever"})

    def test_disallowed_field_raises(self):
        with self.assertRaises(ValueError):
            self.reg.instantiate({
                "kind": "retriever", "name": "bow_retriever",
                "config": {"k": 3, "unknown_field": True},
            })

    def test_config_defaults_accepted(self):
        c = self.reg.instantiate({"kind": "retriever", "name": "bow_retriever", "config": {}})
        self.assertEqual(getattr(c, "k"), 3)  # default


if __name__ == "__main__":
    unittest.main()

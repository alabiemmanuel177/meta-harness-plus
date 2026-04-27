"""Phase 0 deterministic smoke test for the cloud-Ollama wiring.

Doesn't hit a real network — uses a stub HTTP backend to verify that:
- ``HTTPClient`` routes a bearer-token endpoint as OpenAI-compatible
  (Authorization header set, OpenAI-style payload).
- The Phase 0 runner script is importable and exposes ``main()``.
- The cost-log path scaffolding works (write + read JSONL).

Real-API smoke is the runner script itself, gated on env vars.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from meta_harness_plus.llm.client import HTTPClient


_REPO = Path(__file__).resolve().parent.parent


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_phase0_runner", _REPO / "examples" / "run_ollama_cloud_search.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestHTTPClientBearerTokenRouting(unittest.TestCase):
    def test_openai_compat_url_uses_bearer_header(self):
        client = HTTPClient(
            api_url="https://ollama.com/v1/chat/completions",
            api_key="sk-fake-token",
            model="gpt-oss:120b",
        )
        headers = client._build_headers()
        self.assertEqual(headers["authorization"], "Bearer sk-fake-token")
        self.assertEqual(headers["content-type"], "application/json")

    def test_openai_compat_payload_shape(self):
        client = HTTPClient(
            api_url="https://ollama.com/v1/chat/completions",
            api_key="sk-fake",
            model="gpt-oss:120b",
        )
        payload = client._build_payload(
            system="sys", user="user", max_tokens=42, temperature=0.0,
        )
        self.assertEqual(payload["model"], "gpt-oss:120b")
        self.assertEqual(payload["max_tokens"], 42)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["messages"], [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ])

    def test_local_ollama_native_unchanged(self):
        # Local Ollama (no api_key): no auth header.
        client = HTTPClient(
            api_url="http://localhost:11434/api/chat",
            model="gpt-oss:20b",
        )
        self.assertTrue(client._is_ollama_native)
        headers = client._build_headers()
        self.assertNotIn("authorization", headers)

    def test_cloud_ollama_native_uses_bearer(self):
        # Ollama Cloud uses the same native /api/chat protocol but
        # requires bearer-token auth.
        client = HTTPClient(
            api_url="https://ollama.com/api/chat",
            api_key="sk-cloud-token",
            model="gpt-oss:120b",
        )
        self.assertTrue(client._is_ollama_native)
        headers = client._build_headers()
        self.assertEqual(headers["authorization"], "Bearer sk-cloud-token")


class TestPhase0RunnerStructure(unittest.TestCase):
    def test_runner_imports(self):
        mod = _load_runner()
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "_record_ollama_batch"))
        self.assertTrue(hasattr(mod, "_AccountingClient"))
        self.assertTrue(hasattr(mod, "_percentile"))
        self.assertTrue(hasattr(mod, "COST_LOG"))

    def test_missing_url_env_exits_clearly(self):
        mod = _load_runner()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_CLOUD_URL", None)
            os.environ.pop("OLLAMA_API_KEY", None)
            with mock.patch.object(sys, "argv",
                                   ["run_ollama_cloud_search.py"]):
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                # Message should mention the missing env var so the user
                # knows what to do.
                self.assertIn("OLLAMA_CLOUD_URL", str(cm.exception))

    def test_missing_key_env_exits_clearly(self):
        mod = _load_runner()
        with mock.patch.dict(os.environ, {"OLLAMA_CLOUD_URL": "https://x"},
                             clear=False):
            os.environ.pop("OLLAMA_API_KEY", None)
            with mock.patch.object(sys, "argv",
                                   ["run_ollama_cloud_search.py"]):
                with self.assertRaises(SystemExit) as cm:
                    mod.main()
                self.assertIn("OLLAMA_API_KEY", str(cm.exception))

    def test_cost_log_ollama_schema(self):
        """Ollama batches record (calls, in_tokens, out_tokens, p50, p95)
        and deliberately OMIT the ``usd`` field (Pro is flat-rate)."""
        mod = _load_runner()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "wow_push" / "cost_log.jsonl"
            with mock.patch.object(mod, "COST_LOG", path):
                mod._record_ollama_batch(
                    phase="phase0", label="ping",
                    model="gpt-oss:120b", url="https://ollama.com/v1/...",
                    calls=1, in_tokens=10, out_tokens=5,
                    latencies_ms=[120.0],
                )
                rec = json.loads(path.read_text())
                self.assertEqual(rec["provider"], "ollama")
                self.assertEqual(rec["phase"], "phase0")
                self.assertEqual(rec["calls"], 1)
                self.assertEqual(rec["in_tokens"], 10)
                self.assertEqual(rec["out_tokens"], 5)
                self.assertEqual(rec["latency_ms_p50"], 120.0)
                self.assertEqual(rec["latency_ms_p95"], 120.0)
                # No `usd` field on Ollama batches.
                self.assertNotIn("usd", rec)
                self.assertNotIn("est_usd", rec)

    def test_percentile_helper(self):
        mod = _load_runner()
        # Edge cases.
        self.assertEqual(mod._percentile([], 50), 0.0)
        self.assertEqual(mod._percentile([42.0], 95), 42.0)
        # Sorted check.
        xs = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(mod._percentile(xs, 50), 30.0)
        self.assertEqual(mod._percentile(xs, 100), 50.0)

    def test_accounting_client_tallies(self):
        mod = _load_runner()

        class FakeInner:
            def __init__(self):
                self.calls = 0
            def complete(self, **kw):
                self.calls += 1
                # Build a fake LLMResponse-shaped object.
                from types import SimpleNamespace
                return SimpleNamespace(
                    text="x", input_tokens=7, output_tokens=3,
                    latency_ms=100.0 + self.calls * 10, raw=None,
                )

        ac = mod._AccountingClient(FakeInner())
        for _ in range(3):
            ac.complete(system="s", user="u", max_tokens=8, temperature=0.0)
        self.assertEqual(ac.calls, 3)
        self.assertEqual(ac.in_tokens, 21)
        self.assertEqual(ac.out_tokens, 9)
        self.assertEqual(len(ac.latencies_ms), 3)


if __name__ == "__main__":
    unittest.main()

"""HTTPClient payload/header/parse logic per provider. No network."""

from __future__ import annotations

import unittest

from meta_harness_plus.llm.client import HTTPClient, ScriptedClient


class TestScriptedClient(unittest.TestCase):
    def test_list_consumed_in_order(self):
        c = ScriptedClient(["a", "b", "c"])
        self.assertEqual(c.complete(system="s", user="u").text, "a")
        self.assertEqual(c.complete(system="s", user="u").text, "b")

    def test_callable_form(self):
        c = ScriptedClient(lambda s, u: f"heard:{u}")
        self.assertEqual(c.complete(system="s", user="hello").text, "heard:hello")

    def test_reports_tokens_and_latency(self):
        c = ScriptedClient(["hi"], tokens_per_char=1.0, fake_latency_ms=42.0)
        r = c.complete(system="s", user="u")
        self.assertEqual(r.latency_ms, 42.0)
        self.assertGreater(r.total_tokens, 0)


class TestProviderDetection(unittest.TestCase):
    def test_anthropic_detection(self):
        c = HTTPClient(api_url="https://api.anthropic.com/v1/messages",
                       api_key="sk-x", model="claude-opus-4-7")
        p = c._build_payload(system="S", user="U", max_tokens=10, temperature=0.1)
        self.assertEqual(p["system"], "S")
        self.assertEqual(p["messages"], [{"role": "user", "content": "U"}])
        self.assertEqual(p["max_tokens"], 10)
        h = c._build_headers()
        self.assertEqual(h["x-api-key"], "sk-x")
        self.assertIn("anthropic-version", h)
        self.assertTrue(c._is_anthropic)
        self.assertFalse(c._is_ollama_native)

    def test_ollama_native_detection(self):
        c = HTTPClient(api_url="http://localhost:11434/api/chat", model="llama3.2")
        p = c._build_payload(system="S", user="U", max_tokens=10, temperature=0.1)
        self.assertEqual(p["stream"], False)
        self.assertEqual(p["messages"][0]["role"], "system")
        self.assertEqual(p["options"]["num_predict"], 10)
        self.assertEqual(p["options"]["temperature"], 0.1)
        h = c._build_headers()
        self.assertNotIn("authorization", h)   # no auth on local Ollama
        self.assertNotIn("x-api-key", h)
        self.assertTrue(c._is_ollama_native)

    def test_openai_compatible_default(self):
        c = HTTPClient(api_url="https://api.openai.com/v1/chat/completions",
                       api_key="sk-x", model="gpt-4o-mini")
        p = c._build_payload(system="S", user="U", max_tokens=10, temperature=0.1)
        self.assertEqual(p["messages"][0]["role"], "system")
        self.assertEqual(p["max_tokens"], 10)
        h = c._build_headers()
        self.assertEqual(h["authorization"], "Bearer sk-x")
        self.assertFalse(c._is_anthropic)
        self.assertFalse(c._is_ollama_native)


class TestResponseParsing(unittest.TestCase):
    def test_parse_anthropic(self):
        c = HTTPClient(api_url="https://api.anthropic.com/v1/messages",
                       api_key="x", model="m")
        resp = {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }
        text, in_tok, out_tok = c._parse(resp)
        self.assertEqual(text, "hello")
        self.assertEqual((in_tok, out_tok), (12, 3))

    def test_parse_ollama_native(self):
        c = HTTPClient(api_url="http://localhost:11434/api/chat", model="m")
        resp = {
            "message": {"content": "sports"},
            "prompt_eval_count": 20,
            "eval_count": 1,
        }
        text, in_tok, out_tok = c._parse(resp)
        self.assertEqual(text, "sports")
        self.assertEqual((in_tok, out_tok), (20, 1))

    def test_parse_openai(self):
        c = HTTPClient(api_url="https://api.openai.com/v1/chat/completions",
                       api_key="x", model="m")
        resp = {
            "choices": [{"message": {"content": "cooking"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 1},
        }
        text, in_tok, out_tok = c._parse(resp)
        self.assertEqual(text, "cooking")
        self.assertEqual((in_tok, out_tok), (15, 1))


if __name__ == "__main__":
    unittest.main()

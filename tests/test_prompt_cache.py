"""PromptCache + CachedLLMClient (ROADMAP option C)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meta_harness_plus.llm.cache import (
    CachedLLMClient,
    PromptCache,
    _canonical_key,
)
from meta_harness_plus.llm.client import LLMResponse, ScriptedClient


def _resp(text: str = "ok", in_t: int = 10, out_t: int = 5,
          lat: float = 1.0) -> LLMResponse:
    return LLMResponse(text=text, input_tokens=in_t,
                       output_tokens=out_t, latency_ms=lat)


class TestCanonicalKey(unittest.TestCase):
    def test_same_args_same_key(self):
        a = _canonical_key("m1", "S", "U", 0.0, 100)
        b = _canonical_key("m1", "S", "U", 0.0, 100)
        self.assertEqual(a, b)

    def test_different_args_different_key(self):
        base = _canonical_key("m1", "S", "U", 0.0, 100)
        self.assertNotEqual(base, _canonical_key("m2", "S", "U", 0.0, 100))
        self.assertNotEqual(base, _canonical_key("m1", "S2", "U", 0.0, 100))
        self.assertNotEqual(base, _canonical_key("m1", "S", "U2", 0.0, 100))
        self.assertNotEqual(base, _canonical_key("m1", "S", "U", 0.5, 100))
        self.assertNotEqual(base, _canonical_key("m1", "S", "U", 0.0, 200))

    def test_unicode_safe(self):
        # Doesn't crash on non-ASCII.
        a = _canonical_key("m1", "Sürs", "查询", 0.0, 100)
        self.assertIsInstance(a, str)
        self.assertEqual(len(a), 64)  # SHA256 hex


class TestPromptCacheBasics(unittest.TestCase):
    def test_miss_then_hit(self):
        c = PromptCache()
        self.assertIsNone(c.get("m", "S", "U", 0.0, 100))
        self.assertEqual(c.misses, 1)
        c.put("m", "S", "U", 0.0, 100, _resp("yes"))
        hit = c.get("m", "S", "U", 0.0, 100)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.text, "yes")
        self.assertEqual(c.hits, 1)

    def test_default_skips_temp_above_zero(self):
        c = PromptCache()  # cache_all=False
        c.put("m", "S", "U", 0.7, 100, _resp("a"))
        # Stochastic call shouldn't be cached.
        self.assertEqual(len(c), 0)
        self.assertIsNone(c.get("m", "S", "U", 0.7, 100))

    def test_cache_all_caches_everything(self):
        c = PromptCache(cache_all=True)
        c.put("m", "S", "U", 0.7, 100, _resp("a"))
        self.assertEqual(len(c), 1)
        hit = c.get("m", "S", "U", 0.7, 100)
        self.assertIsNotNone(hit)

    def test_hit_rate(self):
        c = PromptCache()
        c.put("m", "S", "U", 0.0, 100, _resp("yes"))
        c.get("m", "S", "U", 0.0, 100)  # hit
        c.get("m", "S", "U2", 0.0, 100)  # miss
        c.get("m", "S", "U", 0.0, 100)  # hit
        self.assertEqual(c.hits, 2)
        self.assertEqual(c.misses, 1)
        self.assertAlmostEqual(c.hit_rate, 2 / 3, places=3)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "cache.jsonl"

    def test_persists_and_reloads(self):
        c1 = PromptCache(path=self.path)
        c1.put("m", "S", "U", 0.0, 100, _resp("first"))
        # Reload in a fresh instance.
        c2 = PromptCache(path=self.path)
        hit = c2.get("m", "S", "U", 0.0, 100)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.text, "first")

    def test_appends_not_truncates(self):
        c1 = PromptCache(path=self.path)
        c1.put("m", "S", "A", 0.0, 100, _resp("a"))
        c2 = PromptCache(path=self.path)
        c2.put("m", "S", "B", 0.0, 100, _resp("b"))
        # Reload: both entries should be present.
        c3 = PromptCache(path=self.path)
        self.assertIsNotNone(c3.get("m", "S", "A", 0.0, 100))
        self.assertIsNotNone(c3.get("m", "S", "B", 0.0, 100))

    def test_corrupt_line_tolerated(self):
        # Write a half-line crash artifact.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            f.write(json.dumps({
                "key_hash": "deadbeef" + "0" * 56,
                "text": "fine",
                "input_tokens": 1, "output_tokens": 1, "latency_ms": 0.0,
            }) + "\n")
            f.write("{not-json")  # crashed mid-write
        c = PromptCache(path=self.path)
        # Should load the good record, skip the broken one.
        self.assertEqual(len(c), 1)


class TestCachedLLMClient(unittest.TestCase):
    def test_first_call_passes_through(self):
        scripted = ScriptedClient(["hello"])
        cache = PromptCache()
        client = CachedLLMClient(scripted, cache, model_id="m")
        r = client.complete(system="S", user="U", temperature=0.0)
        self.assertEqual(r.text, "hello")
        self.assertEqual(cache.misses, 1)

    def test_second_call_hits_cache(self):
        scripted = ScriptedClient(["one"])  # only 1 response queued
        cache = PromptCache()
        client = CachedLLMClient(scripted, cache, model_id="m")
        r1 = client.complete(system="S", user="U", temperature=0.0)
        # If the second call hit the network, ScriptedClient would IndexError.
        r2 = client.complete(system="S", user="U", temperature=0.0)
        self.assertEqual(r1.text, r2.text)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_different_args_miss(self):
        scripted = ScriptedClient(["a", "b"])
        cache = PromptCache()
        client = CachedLLMClient(scripted, cache, model_id="m")
        r1 = client.complete(system="S", user="U1", temperature=0.0)
        r2 = client.complete(system="S", user="U2", temperature=0.0)
        self.assertEqual(r1.text, "a")
        self.assertEqual(r2.text, "b")
        self.assertEqual(cache.hits, 0)

    def test_temp_above_zero_bypasses_cache(self):
        scripted = ScriptedClient(["x", "y"])
        cache = PromptCache()  # default: skip temp != 0
        client = CachedLLMClient(scripted, cache, model_id="m")
        r1 = client.complete(system="S", user="U", temperature=0.7)
        r2 = client.complete(system="S", user="U", temperature=0.7)
        # Both calls should pass through to ScriptedClient — different
        # responses despite identical args, because temperature > 0.
        self.assertEqual(r1.text, "x")
        self.assertEqual(r2.text, "y")
        self.assertEqual(len(cache), 0)


class TestCacheStatsReport(unittest.TestCase):
    def test_stats_dict_shape(self):
        cache = PromptCache()
        cache.put("m", "S", "U", 0.0, 100, _resp("a"))
        cache.get("m", "S", "U", 0.0, 100)  # hit
        cache.get("m", "S", "U2", 0.0, 100)  # miss
        s = cache.stats()
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["misses"], 1)
        self.assertEqual(s["cached_keys"], 1)
        self.assertAlmostEqual(s["hit_rate"], 0.5, places=3)


if __name__ == "__main__":
    unittest.main()

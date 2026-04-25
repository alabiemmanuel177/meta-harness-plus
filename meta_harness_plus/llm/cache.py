"""Prompt cache for LLM clients (ROADMAP option C).

Wraps any ``LLMClient`` with persistent (model, system, user, temperature,
max_tokens) → response caching. Cache hits skip the network call entirely.

Why this matters for our experiments:
- Multi-seed runs evaluate the same seed harness 5+ times; at temperature=0
  every (prompt, model) → response is identical, so 4 of 5 seed-eval calls
  become cache hits.
- Drop-one ablation re-evaluates a harness with one component swapped to
  baseline; the prompts on most eval items are unchanged from the full
  harness's prompts. Major cache-hit territory.
- Every hyperparameter sweep benefits — same prompt across configs hits
  the cache.

Default policy: only cache temperature == 0 calls. Temp > 0 is supposed
to be stochastic, so caching it would hide diversity. Override with
``cache_all=True`` if you know what you're doing.

On-disk format: append-only JSONL. Each line is a record with the key
hash + the response fields. Fast to load, easy to inspect, robust to
crashes (partial writes are detected and skipped at load time).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import LLMClient, LLMResponse


def _canonical_key(
    model: str, system: str, user: str,
    temperature: float, max_tokens: int,
) -> str:
    """SHA256 of the canonical JSON representation of the cache key."""
    payload = json.dumps(
        {
            "model": model, "system": system, "user": user,
            "temperature": float(temperature), "max_tokens": int(max_tokens),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class PromptCache:
    """Append-only JSONL prompt cache with in-memory index.

    ``path``: optional file path for persistence. If None, cache is
    in-memory only (cleared at process exit).
    ``cache_all``: when True, caches every call regardless of temperature.
    Default False — only temperature == 0 calls are cached.
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        *,
        cache_all: bool = False,
    ):
        self.path = Path(path) if path else None
        self.cache_all = cache_all
        self._mem: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.path and self.path.exists():
            self._load()

    # --- persistence ---

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    # Tolerate partial / corrupt lines from crashed writes.
                    continue
                key = rec.get("key_hash")
                if not key:
                    continue
                self._mem[key] = CacheEntry(
                    text=rec.get("text", ""),
                    input_tokens=int(rec.get("input_tokens", 0)),
                    output_tokens=int(rec.get("output_tokens", 0)),
                    latency_ms=float(rec.get("latency_ms", 0.0)),
                )

    def _persist(self, key_hash: str, entry: CacheEntry) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({
                "key_hash": key_hash,
                "text": entry.text,
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "latency_ms": entry.latency_ms,
            }, ensure_ascii=False) + "\n")
        self.writes += 1

    # --- public API ---

    def get(
        self,
        model: str, system: str, user: str,
        temperature: float, max_tokens: int,
    ) -> LLMResponse | None:
        if not self.cache_all and temperature != 0.0:
            return None
        key = _canonical_key(model, system, user, temperature, max_tokens)
        entry = self._mem.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return LLMResponse(
            text=entry.text,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            latency_ms=entry.latency_ms,
            raw=None,
        )

    def put(
        self,
        model: str, system: str, user: str,
        temperature: float, max_tokens: int,
        response: LLMResponse,
    ) -> None:
        if not self.cache_all and temperature != 0.0:
            return
        key = _canonical_key(model, system, user, temperature, max_tokens)
        entry = CacheEntry(
            text=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )
        self._mem[key] = entry
        self._persist(key, entry)

    def __len__(self) -> int:
        return len(self._mem)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "cached_keys": len(self._mem),
            "hit_rate": round(self.hit_rate, 3),
            "path": str(self.path) if self.path else None,
        }


class CachedLLMClient:
    """Drop-in wrapper for any LLMClient. Transparent to existing code.

    Construct with the inner client + a ``PromptCache``::

        cache = PromptCache(path="cache/news_hard.jsonl")
        client = CachedLLMClient(HTTPClient(...), cache)

    Then use ``client`` anywhere ``LLMClient`` is expected. On cache hit,
    returns the cached LLMResponse; on miss, delegates to the inner
    client and stores the result.
    """

    def __init__(
        self,
        inner: LLMClient,
        cache: PromptCache,
        *,
        model_id: str | None = None,
    ):
        self.inner = inner
        self.cache = cache
        # Most LLMClient implementations expose ``model`` (e.g. HTTPClient).
        # ScriptedClient doesn't — caller passes model_id explicitly.
        self.model_id = (
            model_id
            if model_id is not None
            else getattr(inner, "model", "unknown")
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        cached = self.cache.get(
            self.model_id, system, user, temperature, max_tokens,
        )
        if cached is not None:
            return cached
        resp = self.inner.complete(
            system=system, user=user,
            max_tokens=max_tokens, temperature=temperature,
        )
        self.cache.put(
            self.model_id, system, user, temperature, max_tokens, resp,
        )
        return resp

"""LLM client abstractions.

Keep this module dependency-free. Real API calls go through stdlib ``urllib``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict | None = None  # provider response blob, for debugging

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...


# ---------------- ScriptedClient (tests) ----------------

class ScriptedClient:
    """Deterministic fake. Constructed with either:
    - a list[str] consumed in order, or
    - a callable (system, user) -> str.
    Reports fake but consistent token/latency so cost math still works.
    """

    def __init__(
        self,
        responses: list[str] | Callable[[str, str], str],
        tokens_per_char: float = 0.25,
        fake_latency_ms: float = 10.0,
    ):
        self.responses = responses
        self.tokens_per_char = tokens_per_char
        self.fake_latency_ms = fake_latency_ms
        self._idx = 0
        self.call_log: list[tuple[str, str, str]] = []  # (system, user, response)

    def complete(self, *, system: str, user: str,
                 max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        if callable(self.responses):
            text = self.responses(system, user)
        else:
            if self._idx >= len(self.responses):
                raise IndexError(f"ScriptedClient out of responses (idx={self._idx})")
            text = self.responses[self._idx]
            self._idx += 1
        self.call_log.append((system, user, text))
        in_toks = int(self.tokens_per_char * (len(system) + len(user)))
        out_toks = int(self.tokens_per_char * len(text))
        return LLMResponse(
            text=text,
            input_tokens=in_toks,
            output_tokens=out_toks,
            latency_ms=self.fake_latency_ms,
        )


# ---------------- HTTPClient (real APIs) ----------------

class HTTPClient:
    """Minimal urllib-backed client. Supports four provider flavors:

    - **Anthropic** Messages API (auto-detected by ``anthropic.com`` in URL)
    - **Ollama** native chat API (auto-detected by ``/api/chat`` in URL — e.g.
      ``http://localhost:11434/api/chat``; no API key needed)
    - **Gemini** generateContent (auto-detected by
      ``generativelanguage.googleapis.com`` in URL; key passed as ``?key=``
      query param, not a header)
    - **OpenAI-compatible** Chat Completions (everything else, including
      Ollama's ``/v1/chat/completions`` endpoint, LM Studio, vLLM, etc.)

    Not used in tests — only when a real endpoint is configured.
    """

    # HTTP status codes worth retrying — transient backend issues + rate
    # limits. Anything else (400 bad request, 401 auth, 403 perm denied)
    # is a permanent failure and retrying just wastes calls.
    RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str = "",
        model: str,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.extra_headers = extra_headers or {}
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._is_anthropic = "anthropic.com" in api_url
        self._is_ollama_native = "/api/chat" in api_url
        self._is_gemini = "generativelanguage.googleapis.com" in api_url

    def _build_payload(self, system: str, user: str, max_tokens: int, temperature: float) -> dict:
        if self._is_anthropic:
            return {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        if self._is_ollama_native:
            return {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
        if self._is_gemini:
            return {
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
        # OpenAI-compatible (includes Ollama /v1/chat/completions, vLLM, LM Studio)
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def _build_headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._is_anthropic:
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self._is_ollama_native:
            pass  # local Ollama: no auth
        elif self._is_gemini:
            pass  # Gemini auth goes in the URL as ?key=...
        else:
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _resolve_url(self) -> str:
        """Gemini wants the API key as a URL query parameter, not a header."""
        if self._is_gemini and self.api_key and "key=" not in self.api_url:
            sep = "&" if "?" in self.api_url else "?"
            return f"{self.api_url}{sep}key={self.api_key}"
        return self.api_url

    def _parse(self, resp_json: dict) -> tuple[str, int, int]:
        if self._is_anthropic:
            parts = resp_json.get("content", [])
            text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            usage = resp_json.get("usage", {})
            return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        if self._is_ollama_native:
            msg = resp_json.get("message", {})
            text = msg.get("content", "") or ""
            # Reasoning models (gpt-oss, deepseek-r1, qwen-reasoning) put chain-
            # of-thought in "thinking" and only emit "content" after. If the
            # model was cut off mid-thinking, content is empty — fall back to
            # thinking so downstream parsers have something to work with.
            if not text.strip():
                text = msg.get("thinking", "") or ""
            return text, resp_json.get("prompt_eval_count", 0), resp_json.get("eval_count", 0)
        if self._is_gemini:
            cands = resp_json.get("candidates", [])
            text = ""
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)
            usage = resp_json.get("usageMetadata", {})
            return (text,
                    int(usage.get("promptTokenCount", 0)),
                    int(usage.get("candidatesTokenCount", 0)))
        # OpenAI
        choices = resp_json.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage = resp_json.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _do_request(self, data: bytes, headers: dict) -> tuple[str, float]:
        """Single POST attempt. Returns (body_text, wall_time_seconds)."""
        req = urllib.request.Request(
            self._resolve_url(), data=data, headers=headers, method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            body = r.read().decode("utf-8")
        t1 = time.perf_counter()
        return body, t1 - t0

    def complete(self, *, system: str, user: str,
                 max_tokens: int = 1024, temperature: float = 0.0) -> LLMResponse:
        payload = self._build_payload(system, user, max_tokens, temperature)
        headers = self._build_headers()
        data = json.dumps(payload).encode("utf-8")

        # Retry loop with exponential backoff + jitter on transient errors.
        # Uses ``RETRYABLE_STATUSES`` (429, 500, 502, 503, 504) and timeouts.
        # Permanent errors (400/401/403/404) raise immediately.
        last_err: Exception | None = None
        wall_seconds = 0.0
        body: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                body, wall_seconds = self._do_request(data, headers)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:500]
                if e.code in self.RETRYABLE_STATUSES and attempt < self.max_retries:
                    import random
                    delay = min(
                        self.retry_max_delay,
                        self.retry_base_delay * (2 ** attempt),
                    )
                    delay *= 0.5 + random.random()  # full-jitter
                    time.sleep(delay)
                    last_err = RuntimeError(f"LLM HTTP {e.code}: {detail}")
                    continue
                raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.max_retries:
                    import random
                    delay = min(
                        self.retry_max_delay,
                        self.retry_base_delay * (2 ** attempt),
                    )
                    delay *= 0.5 + random.random()
                    time.sleep(delay)
                    last_err = e
                    continue
                raise
        if body is None:
            assert last_err is not None
            raise RuntimeError(f"all {self.max_retries + 1} attempts failed: {last_err}")

        parsed = json.loads(body)
        text, in_toks, out_toks = self._parse(parsed)
        return LLMResponse(
            text=text,
            input_tokens=in_toks,
            output_tokens=out_toks,
            latency_ms=wall_seconds * 1000.0,
            raw=parsed,
        )

"""HTTPClient retry-on-transient-errors behavior."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from meta_harness_plus.llm.client import HTTPClient


def _http_error(code: int, body: str = ""):
    return HTTPError(
        url="http://x", code=code, msg="err",
        hdrs={}, fp=io.BytesIO(body.encode()),
    )


class TestRetry(unittest.TestCase):
    def setUp(self):
        # Fast retries so tests don't drag.
        self.client = HTTPClient(
            api_url="https://api.openai.com/v1/chat/completions",
            api_key="k", model="m",
            max_retries=3,
            retry_base_delay=0.0001,
            retry_max_delay=0.001,
        )

    def test_503_then_success(self):
        responses = [
            _http_error(503, '{"error": "transient"}'),
            ('{"choices":[{"message":{"content":"ok"}}],"usage":{}}', 0.01),
        ]

        def fake_request(data, headers):
            r = responses.pop(0)
            if isinstance(r, HTTPError):
                raise r
            return r

        with patch.object(self.client, "_do_request", side_effect=fake_request):
            resp = self.client.complete(system="S", user="U", max_tokens=10)
        self.assertEqual(resp.text, "ok")

    def test_400_no_retry(self):
        """Permanent failures fail fast — no retries on 4xx (except 429)."""
        calls = {"n": 0}
        def fake_request(data, headers):
            calls["n"] += 1
            raise _http_error(400, '{"error":"bad"}')
        with patch.object(self.client, "_do_request", side_effect=fake_request):
            with self.assertRaises(RuntimeError) as cm:
                self.client.complete(system="S", user="U", max_tokens=10)
        self.assertIn("HTTP 400", str(cm.exception))
        self.assertEqual(calls["n"], 1)  # no retries

    def test_429_retries(self):
        """429 (rate limit) IS retried."""
        calls = {"n": 0}
        def fake_request(data, headers):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _http_error(429, '{"error":"slow_down"}')
            return ('{"choices":[{"message":{"content":"ok"}}],"usage":{}}', 0.01)
        with patch.object(self.client, "_do_request", side_effect=fake_request):
            resp = self.client.complete(system="S", user="U", max_tokens=10)
        self.assertEqual(resp.text, "ok")
        self.assertEqual(calls["n"], 3)

    def test_exhausted_retries_raises(self):
        """After max_retries+1 attempts, give up and raise."""
        calls = {"n": 0}
        def fake_request(data, headers):
            calls["n"] += 1
            raise _http_error(503, '{"error":"down"}')
        with patch.object(self.client, "_do_request", side_effect=fake_request):
            with self.assertRaises(RuntimeError):
                self.client.complete(system="S", user="U", max_tokens=10)
        # max_retries=3 means 4 total attempts (initial + 3 retries).
        self.assertEqual(calls["n"], 4)

    def test_url_error_retried(self):
        """Network-level errors (URLError, OSError) also trigger retry."""
        calls = {"n": 0}
        def fake_request(data, headers):
            calls["n"] += 1
            if calls["n"] < 2:
                raise OSError("connection reset")
            return ('{"choices":[{"message":{"content":"recovered"}}],"usage":{}}', 0.01)
        with patch.object(self.client, "_do_request", side_effect=fake_request):
            resp = self.client.complete(system="S", user="U", max_tokens=10)
        self.assertEqual(resp.text, "recovered")
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()

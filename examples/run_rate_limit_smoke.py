"""Phase 2 amendment — Ollama Pro rate-limit ceiling probe.

Burst N calls with M concurrent workers against a cheap model
(gpt-oss:20b by default) to find the actual concurrency ceiling. Reports:

- HTTP errors by code (429s, 500s, etc.) with retry counts
- p50 / p95 / p99 per-call latency
- effective QPS achieved
- whether p95 / p50 ratio went above 2× (sign of throttling kicking in)

Usage:
    python3 examples/run_rate_limit_smoke.py --workers 16 --calls 60

Decision rule (per amendment):
- clean (no 429s, p95 < 2×p50): proceed at the requested worker count.
- throttled: fall back to the next-lower power of two (16→8).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import (  # type: ignore
    _AccountingClient, _percentile, _record_ollama_batch,
    _load_dotenv_if_present, COST_LOG,
)
from meta_harness_plus.llm.client import HTTPClient


PROBE = (
    "Reply with exactly one short sentence summarizing why the sky appears blue."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--calls", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    _load_dotenv_if_present()
    url = os.environ["OLLAMA_CLOUD_URL"]
    key = os.environ["OLLAMA_API_KEY"]

    raw = HTTPClient(api_url=url, api_key=key, model=args.model,
                     timeout_s=120.0, max_retries=0)
    client = _AccountingClient(raw)

    print(f"[rate] burst {args.calls} calls @ {args.workers} workers, "
          f"model={args.model}")

    errors_by_kind: dict[str, int] = {}
    successes = 0

    def one(_i):
        try:
            client.complete(system="You answer concisely.", user=PROBE,
                            max_tokens=args.max_tokens, temperature=0.0)
            return ("ok", None)
        except urllib.error.HTTPError as e:
            return (f"http_{e.code}", str(e))
        except Exception as e:
            return (f"exc_{type(e).__name__}", str(e))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed(pool.submit(one, i) for i in range(args.calls)):
            kind, msg = fut.result()
            if kind == "ok":
                successes += 1
            else:
                errors_by_kind[kind] = errors_by_kind.get(kind, 0) + 1
    wall = time.perf_counter() - t0

    p50 = _percentile(client.latencies_ms, 50)
    p95 = _percentile(client.latencies_ms, 95)
    p99 = _percentile(client.latencies_ms, 99)
    ratio = p95 / max(1.0, p50)
    qps = client.calls / max(0.001, wall)

    print(f"\n[rate] === RESULT ===")
    print(f"[rate] success: {successes}/{args.calls}")
    print(f"[rate] errors:  {errors_by_kind or '{}'}")
    print(f"[rate] latency  p50={p50:.0f}ms p95={p95:.0f}ms p99={p99:.0f}ms "
          f"(ratio p95/p50 = {ratio:.2f}x)")
    print(f"[rate] effective QPS: {qps:.2f} (theoretical {args.workers}/p50_ms*1000 ~"
          f" {args.workers / max(0.001, p50/1000):.2f})")
    print(f"[rate] wall: {wall:.1f}s; calls={client.calls} "
          f"in_tokens={client.in_tokens} out_tokens={client.out_tokens}")

    _record_ollama_batch(
        phase="phase2_rate_smoke", label=f"burst_w{args.workers}_n{args.calls}",
        model=args.model, url=url,
        calls=client.calls, in_tokens=client.in_tokens,
        out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
    )

    # Decision.
    print(f"\n[rate] === DECISION ===")
    has_429 = any(k.startswith("http_429") for k in errors_by_kind)
    has_5xx = any(k.startswith("http_5") for k in errors_by_kind)
    if has_429 or has_5xx:
        print(f"[rate] FAIL: rate-limited / server errors observed. "
              f"Recommend max_workers = {max(1, args.workers // 2)}.")
        sys.exit(2)
    if ratio > 2.0:
        print(f"[rate] FAIL: p95/p50 ratio {ratio:.2f}× > 2.0 — "
              f"throttling pressure visible. Recommend "
              f"max_workers = {max(1, args.workers // 2)}.")
        sys.exit(3)
    if errors_by_kind:
        print(f"[rate] WARN: non-429 errors {errors_by_kind} — investigate.")
    print(f"[rate] PASS: clean burst at {args.workers} workers; "
          f"safe to launch flattened sweep at this concurrency.")


if __name__ == "__main__":
    main()

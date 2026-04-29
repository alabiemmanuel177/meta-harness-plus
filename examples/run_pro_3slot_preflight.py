"""Pre-flight test for Ollama Pro 3-slot concurrent-model semantics.

Fires three simultaneous calls — one each to deepseek-v3.1:671b,
gpt-oss:20b, and kimi-k2:1t — from three threads. Reports per-model
latency and whether any return HTTP 429 / get queued behind another.

This is a $0 (Pro flat-rate), ~30s test that decides whether Phase 3
can really fan out across 3 model slots, or whether Pro's "3 concurrent
models" actually means a shared account-wide request quota.

Run it BEFORE assuming the 3-way restructure is safe.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import _load_dotenv_if_present
from meta_harness_plus.llm.client import HTTPClient


PROBE = "Reply with exactly: PONG"
MODELS = ["deepseek-v3.1:671b", "gpt-oss:20b", "kimi-k2:1t"]


def fire(model: str, url: str, key: str) -> dict:
    client = HTTPClient(api_url=url, api_key=key, model=model,
                        timeout_s=120.0, max_retries=0)
    t0 = time.perf_counter()
    try:
        resp = client.complete(system="You answer concisely.",
                               user=PROBE, max_tokens=8, temperature=0.0)
        wall = time.perf_counter() - t0
        return {
            "model": model, "ok": True, "wall_s": wall,
            "latency_ms": resp.latency_ms,
            "in_tokens": resp.input_tokens, "out_tokens": resp.output_tokens,
            "text": resp.text[:60],
        }
    except urllib.error.HTTPError as e:
        return {"model": model, "ok": False,
                "wall_s": time.perf_counter() - t0,
                "error": f"HTTP {e.code}", "detail": str(e)[:120]}
    except RuntimeError as e:
        return {"model": model, "ok": False,
                "wall_s": time.perf_counter() - t0,
                "error": str(e)[:200]}
    except Exception as e:
        return {"model": model, "ok": False,
                "wall_s": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


def main():
    _load_dotenv_if_present()
    url = os.environ["OLLAMA_CLOUD_URL"]
    key = os.environ["OLLAMA_API_KEY"]

    print(f"[pre] firing 3 simultaneous calls: {', '.join(MODELS)}")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fire, m, url, key): m for m in MODELS}
        results: list[dict] = []
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            stamp = time.perf_counter() - t0
            tag = "OK" if r["ok"] else f"FAIL ({r.get('error', '?')})"
            print(f"[pre] +{stamp:5.1f}s  {r['model']:25s}  {tag}  "
                  f"wall={r['wall_s']:.1f}s")
    total_wall = time.perf_counter() - t0

    print(f"\n[pre] total wall: {total_wall:.1f}s")
    n_ok = sum(1 for r in results if r["ok"])
    n_429 = sum(1 for r in results if not r["ok"] and "429" in str(r.get("error", "")))
    print(f"[pre] success: {n_ok}/{len(MODELS)}, 429s: {n_429}")

    # Verdict.
    individual_walls = sorted(r["wall_s"] for r in results if r["ok"])
    if not individual_walls:
        print("[pre] VERDICT: all calls failed — investigate before restructuring")
        sys.exit(1)
    max_individual = max(individual_walls)
    if total_wall > max_individual * 1.5:
        print(f"[pre] VERDICT: SERIAL — total wall ({total_wall:.1f}s) > "
              f"1.5 × slowest individual ({max_individual:.1f}s). "
              f"Pro is likely queuing across models. Stick with 2-slot or "
              f"sequential.")
    elif n_429 > 0:
        print(f"[pre] VERDICT: PARTIAL — {n_429} calls hit 429 in 3-way burst. "
              f"Pro has a shared cap; restructure but cap concurrency.")
    else:
        print(f"[pre] VERDICT: 3-WAY CONCURRENT — all 3 models served "
              f"in parallel ({total_wall:.1f}s ≤ {max_individual:.1f}s × "
              f"1.5). Restructure to use all 3 slots.")


if __name__ == "__main__":
    main()

"""Tier-1 model probe — measure cheap-model per-call latency on HMMT
problems so the two-tier Phase 3 wall can be projected accurately.

Runs 5 HMMT integer-only problems × 1 sample each on a candidate
"halving" model (qwen3-next:80b, gpt-oss:20b, etc.). Reports
per-call p50/p95 + n=1 baseline accuracy. The model with the better
latency-vs-accuracy tradeoff becomes the Tier-1 search workhorse.

Usage:
    python3 examples/run_hmmt_tier1_probe.py --model qwen3-next:80b
    python3 examples/run_hmmt_tier1_probe.py --model gpt-oss:20b
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import (  # type: ignore
    _AccountingClient, _percentile, _record_ollama_batch,
    _load_dotenv_if_present, COST_LOG,
)
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.tasks.hmmt_task import (
    build_hmmt_feb2025_task, parse_hmmt_int_answer,
)


SYSTEM_PROMPT = (
    "You are a careful mathematics solver. The answer is an integer "
    "(possibly large or negative).\n"
    "Reason step by step. End with: Answer: N (no other text on that line)."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Tier-1 candidate, e.g. qwen3-next:80b or gpt-oss:20b.")
    ap.add_argument("--n-problems", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    _load_dotenv_if_present()
    url = os.environ["OLLAMA_CLOUD_URL"]
    key = os.environ["OLLAMA_API_KEY"]

    print(f"[tier1-probe] model={args.model} n_problems={args.n_problems} (n=1)")
    task = build_hmmt_feb2025_task(integer_only=True, verify=True)
    problems = task.eval_set[: args.n_problems]

    raw = HTTPClient(api_url=url, api_key=key, model=args.model,
                     timeout_s=300.0, max_retries=3)
    client = _AccountingClient(raw)

    correct = 0
    from concurrent.futures import ThreadPoolExecutor

    def one(i_ex):
        i, ex = i_ex
        resp = client.complete(
            system=SYSTEM_PROMPT, user=ex.input,
            max_tokens=args.max_tokens, temperature=0.0,
        )
        pred = parse_hmmt_int_answer(resp.text)
        try:
            gold = int(str(ex.label).strip())
        except Exception:
            gold = None
        ok = pred is not None and pred == gold
        return i, ok, pred, ex.label, resp.latency_ms

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        results = list(pool.map(one, list(enumerate(problems))))
    wall = time.perf_counter() - t0

    for i, ok, pred, gold, lat in sorted(results):
        correct += int(ok)
        print(f"[tier1-probe] p={i+1}/{len(problems)} ok={ok} pred={pred} "
              f"gold={gold} lat={lat:.0f}ms")

    p50 = _percentile(client.latencies_ms, 50)
    p95 = _percentile(client.latencies_ms, 95)
    acc = correct / len(problems)
    print(f"\n[tier1-probe] {args.model}: acc(n=1)={acc:.3f} "
          f"p50={p50:.0f}ms p95={p95:.0f}ms wall={wall:.1f}s "
          f"in={client.in_tokens} out={client.out_tokens}")
    _record_ollama_batch(
        phase="phase1-tier1-probe", label=f"probe_n{args.n_problems}_s1",
        model=args.model, url=url,
        calls=client.calls, in_tokens=client.in_tokens,
        out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
    )


if __name__ == "__main__":
    main()

"""Phase 1 (HMMT-Feb-2025) zero-shot CoT smoke — primary headroom benchmark.

Loads HMMT-Feb-2025 from ``MathArena/hmmt_feb_2025`` (HuggingFace),
filters to integer-answer problems (default), runs deepseek-v3.1:671b
zero-shot Chain-of-Thought with self-consistency, and reports:

- problem count loaded (integer-only and full)
- ground-truth integer check
- baseline accuracy on the 5-problem smoke
- per-call latency p50 / p95
- Phase 3 wall projection at 4-worker AND 8-worker concurrency
- last 3 lines of cost_log.jsonl

Headroom gate (per WOW_PUSH_PLAN.md option A):
- Pivot to PutnamBench if smoke acc > 75%.
- Proceed if smoke acc 40-75%.
- Pivot if smoke acc < 20% (model can't even start; harness search
  has no signal to follow).

If projected Phase 3 wall > 3 days even with 4 workers, recommend
dropping seeds 5 -> 3.

Usage:
    python3 examples/run_hmmt_smoke.py
    python3 examples/run_hmmt_smoke.py --model deepseek-v3.1:671b \\
        --n-problems 5 --samples 8 --max-workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "You are a careful mathematics solver. You will be given a problem "
    "from the HMMT competition. The answer is an integer (it may be "
    "large, and may be negative).\n"
    "Reason step by step. Then on the final line, output exactly:\n"
    "    Answer: N\n"
    "where N is the integer answer (no other text on that line)."
)


def _majority(votes: list[int | None]) -> int | None:
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _evaluate_problem(
    client: _AccountingClient, problem: str, label: str,
    samples: int, max_tokens: int, temperature: float,
) -> tuple[bool, list[int | None], list[float]]:
    votes: list[int | None] = []
    latencies: list[float] = []
    for _ in range(samples):
        resp = client.complete(
            system=SYSTEM_PROMPT, user=problem,
            max_tokens=max_tokens, temperature=temperature,
        )
        votes.append(parse_hmmt_int_answer(resp.text))
        latencies.append(resp.latency_ms)
    pred = _majority(votes)
    try:
        gold = int(str(label).strip())
    except (TypeError, ValueError):
        gold = None
    return (pred is not None and pred == gold, votes, latencies)


def _project_phase3(
    *, per_call_p50_ms: float,
    halving_schedule: list[tuple[int, int]],
    seeds: int, samples_per_call: int,
    workers: int,
) -> tuple[float, int]:
    """Total calls × p50_ms, divided by workers."""
    calls = sum(n * sz for n, sz in halving_schedule) * samples_per_call * seeds
    seconds_seq = (calls * per_call_p50_ms) / 1000.0
    return (seconds_seq / max(1, workers), calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v3.1:671b")
    ap.add_argument("--n-problems", type=int, default=5)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--headroom-ceiling", type=float, default=0.75,
                    help="If acc > this, STOP — pivot to PutnamBench.")
    ap.add_argument("--headroom-floor", type=float, default=0.20,
                    help="If acc < this, STOP — model has no signal.")
    args = ap.parse_args()

    _load_dotenv_if_present()
    url = os.environ.get("OLLAMA_CLOUD_URL")
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not url:
        raise SystemExit("Missing OLLAMA_CLOUD_URL.")
    if not api_key:
        raise SystemExit("Missing OLLAMA_API_KEY.")

    print(f"[phase1-hmmt] HMMT-Feb-2025 smoke: model={args.model} "
          f"n_problems={args.n_problems} samples={args.samples}")
    print(f"[phase1-hmmt] dataset: HuggingFace MathArena/hmmt_feb_2025 "
          f"(commit a7ceac11a0b5b8c18a9b50d41d7ff96311b189c2)")
    print(f"[phase1-hmmt] cache: meta_harness_plus/tasks/data/hmmt/hmmt_feb2025.jsonl")

    full = build_hmmt_feb2025_task(integer_only=False, verify=True)
    int_only = build_hmmt_feb2025_task(integer_only=True, verify=True)
    print(f"[phase1-hmmt] loaded {len(full.eval_set)} problems total; "
          f"{len(int_only.eval_set)} have integer answers")
    print(f"[phase1-hmmt] integer-only check: PASS "
          f"(filtered {len(full.eval_set) - len(int_only.eval_set)} non-integer "
          f"answers — fractions/surds/expressions; deterministic-grader-safe subset)")

    if len(int_only.eval_set) < args.n_problems:
        raise SystemExit(
            f"Only {len(int_only.eval_set)} integer-answer problems; "
            f"--n-problems {args.n_problems} requested."
        )

    raw_client = HTTPClient(api_url=url, api_key=api_key, model=args.model,
                            timeout_s=300.0, max_retries=3)
    client = _AccountingClient(raw_client)

    problems = int_only.eval_set[: args.n_problems]
    correct = 0
    per_problem_latencies: list[list[float]] = []

    t0 = time.perf_counter()
    if args.max_workers <= 1:
        for i, ex in enumerate(problems):
            ok, votes, lats = _evaluate_problem(
                client, ex.input, ex.label,
                args.samples, args.max_tokens, args.temperature,
            )
            correct += int(ok)
            per_problem_latencies.append(lats)
            print(f"[phase1-hmmt] p={i+1}/{len(problems)} ok={ok} "
                  f"votes={votes} gold={ex.label}")
    else:
        results: dict[int, tuple[bool, list[int | None], list[float]]] = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {
                pool.submit(_evaluate_problem, client, ex.input, ex.label,
                            args.samples, args.max_tokens, args.temperature): i
                for i, ex in enumerate(problems)
            }
            for fut in as_completed(future_map):
                i = future_map[fut]
                results[i] = fut.result()
        for i in sorted(results):
            ok, votes, lats = results[i]
            correct += int(ok)
            per_problem_latencies.append(lats)
            print(f"[phase1-hmmt] p={i+1}/{len(problems)} ok={ok} "
                  f"votes={votes} gold={problems[i].label}")
    wall = time.perf_counter() - t0

    acc = correct / len(problems)
    flat = [l for ls in per_problem_latencies for l in ls]
    p50 = _percentile(flat, 50)
    p95 = _percentile(flat, 95)

    print(f"\n[phase1-hmmt] === RESULT ===")
    print(f"[phase1-hmmt] baseline acc (zero-shot CoT MAJ@{args.samples}): "
          f"{acc:.3f} ({correct}/{len(problems)})")
    print(f"[phase1-hmmt] per-call latency p50={p50:.0f}ms p95={p95:.0f}ms")
    print(f"[phase1-hmmt] total calls={client.calls} "
          f"in_tokens={client.in_tokens} out_tokens={client.out_tokens}")
    print(f"[phase1-hmmt] wall_seconds={wall:.1f}")

    _record_ollama_batch(
        phase="phase1-hmmt",
        label=f"smoke_n{args.n_problems}_s{args.samples}",
        model=args.model, url=url,
        calls=client.calls, in_tokens=client.in_tokens,
        out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
    )

    # === Phase 3 projections — both 4-worker and 8-worker configs ===
    # HMMT integer-only set is 14 problems, so the eval-size schedule
    # is rebalanced: 14→14→14→14→14→14 (no early-stopping; full eval
    # on 14 every rung). Halving still narrows candidates 32→16→8→4→2→1.
    schedule = [(32, 14), (16, 14), (8, 14), (4, 14), (2, 14), (1, 14)]
    seeds_5 = 5
    seeds_3 = 3

    print(f"\n[phase1-hmmt] === PHASE 3 PROJECTION (HMMT 14-problem subset) ===")
    print(f"[phase1-hmmt] schedule: {schedule}, samples={args.samples}")
    for seeds in (seeds_3, seeds_5):
        for workers in (4, 8):
            seconds, calls = _project_phase3(
                per_call_p50_ms=p50, halving_schedule=schedule,
                seeds=seeds, samples_per_call=args.samples, workers=workers,
            )
            hours = seconds / 3600.0
            print(f"[phase1-hmmt]   seeds={seeds} workers={workers}: "
                  f"{calls:,} calls -> {hours:.1f}h ({hours/24:.1f}d)")

    # Recommend seed cut if even seeds=3+workers=4 exceeds 3 days.
    secs_3w4, _ = _project_phase3(
        per_call_p50_ms=p50, halving_schedule=schedule,
        seeds=3, samples_per_call=args.samples, workers=4,
    )
    if secs_3w4 / 3600 > 72:
        print(f"[phase1-hmmt] WARN: even seeds=3 + 4-workers > 3 days. "
              f"Consider 8 workers or smaller halving (32->16->8->4 only).")

    # === Headroom check ===
    print(f"\n[phase1-hmmt] === HEADROOM CHECK ===")
    if acc > args.headroom_ceiling:
        print(f"[phase1-hmmt] STOP: acc {acc:.3f} > ceiling "
              f"{args.headroom_ceiling:.2f} — model saturates HMMT-Feb-2025 "
              f"too. Pivot to PutnamBench / Putnam-AXIOM.")
        sys.exit(2)
    if acc < args.headroom_floor:
        print(f"[phase1-hmmt] STOP: acc {acc:.3f} < floor "
              f"{args.headroom_floor:.2f} — model can't even start; "
              f"harness search has no signal to follow. Either pick a "
              f"stronger model or an easier benchmark.")
        sys.exit(3)
    print(f"[phase1-hmmt] OK: acc {acc:.3f} in band "
          f"({args.headroom_floor:.2f}, {args.headroom_ceiling:.2f}) — "
          f"headroom for harness gains.")
    print(f"\n[phase1-hmmt] cost log: {COST_LOG}")


if __name__ == "__main__":
    main()

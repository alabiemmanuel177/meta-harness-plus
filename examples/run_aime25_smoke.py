"""Phase 1 — AIME-25 zero-shot CoT smoke (deepseek-v3.1 via Ollama Cloud).

Loads AIME-25 from ``math-ai/aime25`` (HuggingFace; cached to
``meta_harness_plus/tasks/data/aime/aime25.jsonl`` after first load),
verifies all 30 ground-truth answers are integers 0–999, then runs
deepseek-v3.1 zero-shot Chain-of-Thought with self-consistency
(default 8 samples, majority vote) on the first ``--n-problems`` items.

Reports:
- problem count loaded
- ground-truth integrity (pass/fail per row)
- baseline accuracy
- per-call latency p50 / p95
- Phase 3 wall-clock projection from observed per-call latency
- last 3 lines of cost_log.jsonl

Usage:
    python3 examples/run_aime25_smoke.py
    python3 examples/run_aime25_smoke.py --model deepseek-v3.1 --n-problems 5 --samples 8

Phase 1 gate (per WOW_PUSH_PLAN.md):
- Loader green + 30 problems + GT verify pass.
- 5-problem baseline acc within ~50–70% expected band.
- IF acc > 75%, STOP — pivot to harder split before sinking time into
  baselines and search. Reported but not auto-pivoted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Reuse Phase 0 helpers (cost log + accounting client + .env loader).
from examples.run_ollama_cloud_search import (  # type: ignore
    _AccountingClient, _percentile, _record_ollama_batch, _load_dotenv_if_present,
    COST_LOG,
)
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.tasks.aime_task import (
    build_aime25_task, parse_aime_answer,
)


# Chain-of-Thought system prompt for AIME — keeps the model focused on
# integer 0–999 final answers and a definite final-line marker so the
# parser can extract reliably.
SYSTEM_PROMPT = (
    "You are a careful mathematics solver. You will be given an AIME "
    "problem whose answer is an integer between 0 and 999 inclusive.\n"
    "Reason step by step. Then on the final line, output exactly:\n"
    "    Answer: N\n"
    "where N is the integer answer (no other text on that line)."
)


def _majority(votes: list[int | None]) -> int | None:
    """Plurality vote across self-consistency samples; ignores None."""
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _evaluate_problem(
    client: _AccountingClient, problem: str, label: str,
    samples: int, max_tokens: int, temperature: float,
) -> tuple[bool, list[int | None], list[float]]:
    """Run ``samples`` self-consistency rolls; majority-vote correctness."""
    votes: list[int | None] = []
    latencies: list[float] = []
    for _ in range(samples):
        resp = client.complete(
            system=SYSTEM_PROMPT, user=problem,
            max_tokens=max_tokens, temperature=temperature,
        )
        votes.append(parse_aime_answer(resp.text))
        latencies.append(resp.latency_ms)
    pred = _majority(votes)
    try:
        gold = int(label)
    except (TypeError, ValueError):
        gold = None
    return (pred is not None and pred == gold, votes, latencies)


def _project_phase3(
    *,
    per_call_p50_ms: float,
    halving_schedule: list[tuple[int, int]],
    seeds: int,
    samples_per_call: int,
) -> tuple[float, int]:
    """Crude wall-clock projection for Phase 3 search.

    halving_schedule: list of (n_candidates, eval_size_per_candidate)
    per stage. Total LLM calls = sum(n*eval) * samples_per_call * seeds.
    Wall is calls × p50_ms / 1000 s, *assuming sequential*. Parallel
    workers divide it.
    """
    calls = sum(n * sz for n, sz in halving_schedule) * samples_per_call * seeds
    seconds = (calls * per_call_p50_ms) / 1000.0
    return (seconds, calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v3.1",
                    help="Ollama Cloud model (math-reasoning-tuned).")
    ap.add_argument("--n-problems", type=int, default=5,
                    help="Smoke size; gate uses this many problems.")
    ap.add_argument("--samples", type=int, default=8,
                    help="Self-consistency samples per problem.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Long enough for AIME chain-of-thought.")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Parallel HTTP workers (within Pro rate limits).")
    ap.add_argument("--headroom-ceiling", type=float, default=0.75,
                    help="If smoke acc > this, gate fails — pivot to harder split.")
    args = ap.parse_args()

    _load_dotenv_if_present()
    url = os.environ.get("OLLAMA_CLOUD_URL")
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not url:
        raise SystemExit("Missing OLLAMA_CLOUD_URL. Add to .env or export.")
    if not api_key:
        raise SystemExit("Missing OLLAMA_API_KEY. Add to .env or export.")

    print(f"[phase1] AIME-25 smoke: model={args.model} n_problems={args.n_problems} "
          f"samples={args.samples}")
    print(f"[phase1] dataset: HuggingFace math-ai/aime25 (test split, 30 problems)")
    print(f"[phase1] cache: meta_harness_plus/tasks/data/aime/aime25.jsonl")

    # Load with verification. Raises if any answer is non-int or out of range.
    task = build_aime25_task(verify=True)
    n_loaded = len(task.eval_set)
    if n_loaded < args.n_problems:
        raise SystemExit(
            f"AIME-25 only has {n_loaded} problems loaded; --n-problems "
            f"{args.n_problems} requested."
        )
    print(f"[phase1] loaded {n_loaded} problems")
    print(f"[phase1] ground-truth check: PASS (all 0-999 integers)")

    raw_client = HTTPClient(api_url=url, api_key=api_key, model=args.model,
                            timeout_s=300.0, max_retries=3)
    client = _AccountingClient(raw_client)

    problems = task.eval_set[: args.n_problems]
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
            print(f"[phase1] p={i+1}/{len(problems)} ok={ok} votes={votes} "
                  f"gold={ex.label}")
    else:
        # Each problem runs serial (samples are sequential) but
        # different problems run in parallel.
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
            print(f"[phase1] p={i+1}/{len(problems)} ok={ok} votes={votes} "
                  f"gold={problems[i].label}")
    wall = time.perf_counter() - t0

    acc = correct / len(problems)
    flat_latencies = [l for ls in per_problem_latencies for l in ls]
    p50 = _percentile(flat_latencies, 50)
    p95 = _percentile(flat_latencies, 95)

    print(f"\n[phase1] === RESULT ===")
    print(f"[phase1] baseline accuracy (zero-shot CoT MAJ@{args.samples}): "
          f"{acc:.3f} ({correct}/{len(problems)})")
    print(f"[phase1] per-call latency p50={p50:.0f}ms p95={p95:.0f}ms")
    print(f"[phase1] total calls={client.calls} "
          f"in_tokens={client.in_tokens} out_tokens={client.out_tokens}")
    print(f"[phase1] wall_seconds={wall:.1f}")

    _record_ollama_batch(
        phase="phase1", label=f"smoke_n{args.n_problems}_s{args.samples}",
        model=args.model, url=url,
        calls=client.calls, in_tokens=client.in_tokens,
        out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
    )

    # === Phase 3 wall projection ===
    # WOW_PUSH_PLAN Phase 3 schedule: 32→16→8→4→2→1, eval [5,10,15,25,30,30].
    schedule = [(32, 5), (16, 10), (8, 15), (4, 25), (2, 30), (1, 30)]
    seeds = 5
    proj_seconds_seq, proj_calls = _project_phase3(
        per_call_p50_ms=p50, halving_schedule=schedule,
        seeds=seeds, samples_per_call=args.samples,
    )
    proj_hours_seq = proj_seconds_seq / 3600.0
    workers = args.max_workers
    proj_hours_par = proj_hours_seq / max(1, workers)

    print(f"\n[phase1] === PHASE 3 PROJECTION ===")
    print(f"[phase1] schedule: {schedule}, seeds={seeds}, samples={args.samples}")
    print(f"[phase1] total calls projected: {proj_calls:,}")
    print(f"[phase1] sequential @ p50 ({p50:.0f}ms): {proj_hours_seq:.1f} h "
          f"({proj_hours_seq/24:.1f} days)")
    print(f"[phase1] parallel  @ {workers} workers: ~{proj_hours_par:.1f} h "
          f"({proj_hours_par/24:.1f} days)")
    if proj_hours_par > 24.0:
        print(f"[phase1] WARN: projected wall > 24h — flag before Phase 2 "
              f"closes; choose (a) overnight runs, (b) smaller halving, "
              f"or (c) more workers.")

    # === Headroom check ===
    print(f"\n[phase1] === HEADROOM CHECK ===")
    if acc > args.headroom_ceiling:
        print(f"[phase1] STOP: baseline acc {acc:.3f} > ceiling "
              f"{args.headroom_ceiling:.2f}. Pivot to harder split before "
              f"committing to Phase 2 baselines.")
        print(f"[phase1] options: AIME-24 + AIME-25 combined; AIME-II only; "
              f"USAMO; HMMT; or AIME-25 with weaker model.")
        sys.exit(2)
    print(f"[phase1] OK: acc {acc:.3f} ≤ ceiling {args.headroom_ceiling:.2f}; "
          f"headroom available for harness gains.")

    print(f"\n[phase1] cost log: {COST_LOG}")


if __name__ == "__main__":
    main()

"""Phase 3 Tier-1 — halving search over math-harness shapes.

Search space (math task):
- system_prompt_id ∈ {0, 1, 2}     # plain CoT / plan-execute / contest-math
- n_samples ∈ {1, 4, 8, 16}
- temperature ∈ {0.0, 0.3, 0.7}
- n_bootstrap_demos ∈ {0, 2, 4}    # leave-one-out from the train pool

Total = 3 × 4 × 3 × 3 = 108 candidates. We sample 32 randomly per seed
and run successive halving (32 → 16 → 8 → 4 → 2 → 1) using the cheap
Tier-1 model (default: gpt-oss:20b) at single-sample evaluation. The
lowest-acc half drops at each rung; rung eval-size grows.

Output: per-seed JSON with the discovered top-3 candidate configs +
their tier-1 accuracy curves. Phase 3 Tier-2 then validates the top-3
configs on the heavy model with full MAJ@8.

Halving schedules are task-aware:
- HMMT (14 problems):   rungs of [4, 6, 8, 12, 14, 14]
- AIME-25 (30):         rungs of [5, 10, 15, 20, 25, 30]

Resume-safe: persists each completed seed to
``runs/wow_push/phase3_tier1_<task>_<model>_seed<n>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import (  # type: ignore
    _AccountingClient, _percentile, _record_ollama_batch,
    _load_dotenv_if_present, COST_LOG,
)
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.tasks.aime_task import (
    build_aime25_task, parse_aime_answer,
)
from meta_harness_plus.tasks.hmmt_task import (
    build_hmmt_feb2025_task, parse_hmmt_int_answer,
)


# ---------- search space ----------

SYS_PROMPTS_HMMT = [
    "You are a careful mathematics solver. The answer is an integer "
    "(possibly large or negative).\nReason step by step. End with: "
    "Answer: N (integer only on the final line).",

    "You are a math olympiad expert. First sketch a plan in 1-2 "
    "sentences, then execute the algebra carefully. End with: "
    "Answer: N (integer only on the final line).",

    "You are a contest mathematics specialist. Decompose the problem "
    "into sub-claims, verify each, then combine. End with: "
    "Answer: N (integer only on the final line).",
]

SYS_PROMPTS_AIME = [
    "You are a careful mathematics solver. The answer is an integer "
    "0-999 inclusive.\nReason step by step. End with: Answer: N "
    "(integer only on the final line).",

    "You are an AIME problem solver. First sketch a plan in 1-2 "
    "sentences, then execute. End with: Answer: N (integer 0-999 only).",

    "You are a contest mathematics specialist. Decompose the problem "
    "into sub-claims, verify each, then combine. The answer is an "
    "integer 0-999. End with: Answer: N (integer only on the final line).",
]

TEMPERATURE_CHOICES = [0.0, 0.3, 0.7]
N_DEMOS_CHOICES = [0, 2, 4]


@dataclass(frozen=True)
class Candidate:
    """Tier-1 search variable: sys_prompt × temperature × n_bootstrap_demos.

    n_samples is a Tier-2 knob — at Tier 1 every candidate is evaluated
    at n=1, so it doesn't ride in the search space. The whole point of
    Tier 1 is a cheap ranking signal under single-sample CoT.
    """
    sys_prompt_id: int
    temperature: float
    n_bootstrap_demos: int

    def label(self) -> str:
        return (f"sp{self.sys_prompt_id}_"
                f"t{self.temperature:.1f}_d{self.n_bootstrap_demos}")


def sample_candidates(seed: int, n: int = 32) -> list[Candidate]:
    """Sample ``n`` distinct candidates. The full grid is 3×3×3 = 27 — when
    n > 27 we fall back to drawing duplicates, which we de-dup so callers
    still get up to 27 unique configurations."""
    rng = random.Random(seed)
    seen: set[Candidate] = set()
    out: list[Candidate] = []
    # Enumerate the full grid first, shuffle, take the first n unique.
    grid = [
        Candidate(sys_prompt_id=sp, temperature=t, n_bootstrap_demos=d)
        for sp in [0, 1, 2]
        for t in TEMPERATURE_CHOICES
        for d in N_DEMOS_CHOICES
    ]
    rng.shuffle(grid)
    for c in grid:
        if len(out) >= n:
            break
        seen.add(c)
        out.append(c)
    return out


# ---------- bootstrap-demo helpers ----------

def _bootstrap_demos(client, problems, parser, system, max_tokens, max_workers):
    """Run zero-shot CoT on every problem; return the (problem, label)
    list of those the model got right. Used as the demo pool for
    `n_bootstrap_demos > 0` candidates."""
    def per(idx_ex):
        i, ex = idx_ex
        resp = client.complete(system=system, user=ex.input,
                               max_tokens=max_tokens, temperature=0.0)
        pred = parser(resp.text)
        try:
            gold = int(str(ex.label).strip())
        except Exception:
            gold = None
        return i, (pred is not None and pred == gold)

    correct: list[int] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for i, ok in pool.map(per, list(enumerate(problems))):
            if ok:
                correct.append(i)
    return [(problems[i].input, problems[i].label) for i in correct]


# ---------- per-candidate Tier-1 eval ----------

def _eval_candidate(
    client, candidate: Candidate, problems, parser, sys_prompts,
    demo_pool, max_tokens, max_workers,
):
    """Evaluate one candidate at Tier-1 — single sample per problem
    (n=1) at the candidate's declared temperature. The Tier-1 ranking
    signal is per-problem correctness under one rollout. Tier-2 will
    re-validate top-3 candidates with MAJ@8."""
    sp = sys_prompts[candidate.sys_prompt_id]
    K = candidate.n_bootstrap_demos
    demos = demo_pool[:K]

    def build_user(problem_text: str) -> str:
        if not demos:
            return problem_text
        demo_block = "\n\n".join(
            f"Problem: {p}\nAnswer: {a}" for p, a in demos
        )
        return f"{demo_block}\n\nNow solve this problem:\n{problem_text}"

    def one_call(i_ex):
        i, ex = i_ex
        resp = client.complete(
            system=sp, user=build_user(ex.input),
            max_tokens=max_tokens, temperature=candidate.temperature,
        )
        return i, parser(resp.text)

    pairs = list(enumerate(problems))
    pred_by_idx: dict[int, int | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for fut in as_completed(pool.submit(one_call, p) for p in pairs):
            i, pred = fut.result()
            pred_by_idx[i] = pred

    per_problem = []
    correct_total = 0
    for i in range(len(problems)):
        pred = pred_by_idx.get(i)
        try:
            gold = int(str(problems[i].label).strip())
        except Exception:
            gold = None
        ok = pred is not None and pred == gold
        if ok:
            correct_total += 1
        per_problem.append({"idx": i, "ok": ok, "pred": pred})

    return {
        "accuracy": correct_total / max(1, len(problems)),
        "n_correct": correct_total,
        "n_total": len(problems),
        "per_problem": per_problem,
    }


# ---------- halving driver ----------

def run_halving_for_seed(
    *, client, task_name: str, problems, parser, sys_prompts, demo_pool_full,
    rungs: list[int], seed: int, max_tokens: int, max_workers: int,
) -> dict:
    candidates = sample_candidates(seed, n=32)
    rung_log: list[dict] = []

    def candidate_key(c: Candidate) -> str:
        return c.label()

    surviving = list(candidates)
    # Halve at each rung, but cap at ≥1 and never grow.
    n_init = len(surviving)
    rung_keep_sizes: list[int] = []
    n = n_init
    for _ in range(len(rungs)):
        n = max(1, n // 2)
        rung_keep_sizes.append(n)
    rung_keep_sizes[-1] = max(1, rung_keep_sizes[-1])

    for rung_idx, eval_size in enumerate(rungs):
        n_keep = rung_keep_sizes[rung_idx] if rung_idx < len(rung_keep_sizes) else 1
        rung_problems = problems[: min(eval_size, len(problems))]

        # FLAT parallelism: dispatch every (candidate × problem) pair to a
        # single ThreadPoolExecutor of size max_workers. Previous version
        # serialised across candidates which left workers idle whenever a
        # candidate had fewer problems than workers; that turned a 32s
        # rung into an 11-min rung.
        def build_user(problem_text: str, cand: Candidate) -> str:
            demos = demo_pool_full[: cand.n_bootstrap_demos]
            if not demos:
                return problem_text
            demo_block = "\n\n".join(
                f"Problem: {p}\nAnswer: {a}" for p, a in demos
            )
            return f"{demo_block}\n\nNow solve this problem:\n{problem_text}"

        def one_call(args):
            ci, pi = args
            cand = surviving[ci]
            ex = rung_problems[pi]
            sp = sys_prompts[cand.sys_prompt_id]
            resp = client.complete(
                system=sp, user=build_user(ex.input, cand),
                max_tokens=max_tokens, temperature=cand.temperature,
            )
            return ci, pi, parser(resp.text)

        pairs = [(ci, pi) for ci in range(len(surviving))
                          for pi in range(len(rung_problems))]
        preds: dict[tuple[int, int], int | None] = {}
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            for fut in as_completed(pool.submit(one_call, p) for p in pairs):
                ci, pi, pred = fut.result()
                preds[(ci, pi)] = pred

        rung_results = []
        for ci, cand in enumerate(surviving):
            correct = 0
            for pi in range(len(rung_problems)):
                pred = preds.get((ci, pi))
                try:
                    gold = int(str(rung_problems[pi].label).strip())
                except Exception:
                    gold = None
                if pred is not None and pred == gold:
                    correct += 1
            rung_results.append({
                "candidate": candidate_key(cand),
                "config": asdict(cand),
                "accuracy": correct / max(1, len(rung_problems)),
                "n_correct": correct,
                "n_total": len(rung_problems),
                "calls_total": client.calls,
            })
        rung_results.sort(key=lambda r: -r["accuracy"])
        survivors_keys = {r["candidate"] for r in rung_results[:n_keep]}
        survivor_configs = [c for c in surviving if candidate_key(c) in survivors_keys]
        rung_log.append({
            "rung": rung_idx,
            "eval_size": len(rung_problems),
            "n_candidates": len(surviving),
            "n_keep": n_keep,
            "ranked": rung_results,
            "survivors": [candidate_key(c) for c in survivor_configs],
        })
        surviving = survivor_configs

    return {
        "task": task_name,
        "seed": seed,
        "rung_log": rung_log,
        "winners_top3": rung_log[-3]["ranked"][:3] if len(rung_log) >= 3 else rung_log[-1]["ranked"][:3],
        "winner_top1": rung_log[-1]["ranked"][:1],
        "client_summary": {
            "calls": client.calls, "in_tokens": client.in_tokens,
            "out_tokens": client.out_tokens,
            "p50": _percentile(client.latencies_ms, 50),
            "p95": _percentile(client.latencies_ms, 95),
        },
    }


# ---------- main ----------

def _setup_task(task_name: str):
    if task_name == "hmmt":
        task = build_hmmt_feb2025_task(integer_only=True, verify=True)
        return (task, parse_hmmt_int_answer, SYS_PROMPTS_HMMT,
                [4, 6, 8, 12, 14, 14])
    if task_name == "aime25":
        task = build_aime25_task(verify=True)
        return (task, parse_aime_answer, SYS_PROMPTS_AIME,
                [5, 10, 15, 20, 25, 30])
    raise SystemExit(f"unknown --task {task_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["hmmt", "aime25"])
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--out-dir", default="runs/wow_push")
    args = ap.parse_args()

    _load_dotenv_if_present()
    url = os.environ["OLLAMA_CLOUD_URL"]
    key = os.environ["OLLAMA_API_KEY"]

    task, parser, sys_prompts, rungs = _setup_task(args.task)
    print(f"[t1] task={args.task} model={args.model} seeds={args.seeds} "
          f"workers={args.max_workers}")
    print(f"[t1] eval set size={len(task.eval_set)}; rungs={rungs}")

    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the demo pool ONCE (shared across all seeds + candidates).
    # Uses sys_prompts[0] (the plain CoT prompt) for demo selection.
    raw_demo_client = HTTPClient(api_url=url, api_key=key, model=args.model,
                                 timeout_s=300.0, max_retries=8,
                                 retry_base_delay=2.0, retry_max_delay=60.0)
    demo_client = _AccountingClient(raw_demo_client)
    demo_pool_path = out_dir / f"phase3_tier1_{args.task}_{safe_model}_demos.json"
    if demo_pool_path.exists():
        demo_pool_full = json.loads(demo_pool_path.read_text())
        print(f"[t1] reused {len(demo_pool_full)} demos from cache")
    else:
        print(f"[t1] building demo pool (zero-shot CoT on full eval set)...")
        t0 = time.perf_counter()
        demo_pool_full = _bootstrap_demos(
            demo_client, task.eval_set, parser, sys_prompts[0],
            args.max_tokens, args.max_workers,
        )
        wall = time.perf_counter() - t0
        demo_pool_path.write_text(json.dumps(demo_pool_full))
        print(f"[t1] {len(demo_pool_full)} demos collected in {wall:.0f}s")
        _record_ollama_batch(
            phase="phase3_tier1_demos", label=f"{args.task}",
            model=args.model, url=url,
            calls=demo_client.calls, in_tokens=demo_client.in_tokens,
            out_tokens=demo_client.out_tokens, latencies_ms=demo_client.latencies_ms,
        )

    for seed in range(args.seeds):
        out_path = out_dir / f"phase3_tier1_{args.task}_{safe_model}_seed{seed}.json"
        if out_path.exists():
            print(f"[t1] skip seed {seed}: {out_path} exists")
            continue

        raw = HTTPClient(api_url=url, api_key=key, model=args.model,
                         timeout_s=300.0, max_retries=8,
                         retry_base_delay=2.0, retry_max_delay=60.0)
        client = _AccountingClient(raw)
        print(f"[t1] === seed {seed} ===")
        t0 = time.perf_counter()
        result = run_halving_for_seed(
            client=client, task_name=args.task, problems=task.eval_set,
            parser=parser, sys_prompts=sys_prompts, demo_pool_full=demo_pool_full,
            rungs=rungs, seed=seed, max_tokens=args.max_tokens,
            max_workers=args.max_workers,
        )
        wall = time.perf_counter() - t0
        result["wall_seconds"] = wall

        print(f"[t1] seed {seed} done in {wall:.1f}s; "
              f"winner_top1 acc={result['winner_top1'][0]['accuracy']:.3f}, "
              f"config={result['winner_top1'][0]['config']}")
        print(f"[t1] top-3:")
        for r in result["winners_top3"]:
            print(f"[t1]   {r['accuracy']:.3f}  {r['config']}")

        out_path.write_text(json.dumps(result, indent=2))
        _record_ollama_batch(
            phase="phase3_tier1", label=f"{args.task}_seed{seed}",
            model=args.model, url=url,
            calls=client.calls, in_tokens=client.in_tokens,
            out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
        )

    print(f"[t1] === Phase 3 Tier-1 ({args.task}) complete ===")


if __name__ == "__main__":
    main()

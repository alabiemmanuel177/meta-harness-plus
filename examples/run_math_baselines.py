"""Phase 2 math baselines runner — open-generation math (AIME / HMMT).

A single CLI per (task, baseline, seed, model) so a sweep can be
parallelized as a bash matrix. Aggregation is a separate step.

Baselines implemented:
- ``cot``    Zero-shot Chain-of-Thought, n=1 sample.
- ``maj8``   Self-consistency MAJ@8 (8 samples, T=0.7, plurality vote).
- ``maj16``  Self-consistency MAJ@16 (16 samples, T=0.7).
- ``dspy``   Bootstrap-FewShot adapted for math: leave-one-out, the K
             other problems where zero-shot CoT got the right answer
             are used as few-shot demos (no separate train split for
             AIME/HMMT). Fall back to zero-shot if no demos qualify.
             Final response uses MAJ@8.
- ``opro``   Instruction-string optimization. Generate K candidate
             system prompts via meta-LLM, score each on a tiny calibration
             slice, pick the best, run final eval with MAJ@8.
- ``random`` Random hyperparameter search: sample (temperature ∈ [0, 1],
             samples ∈ {1, 2, 4, 8}, system_prompt ∈ {3 hand variants},
             vote ∈ {majority, sum-of-most-common}) for K candidates,
             pick best, run final eval.

Per-call cost is tracked via ``_AccountingClient``; the aggregate JSON
shape mirrors the existing rag_vs_mh aggregates so the same downstream
report machinery works.

Usage:
    python3 examples/run_math_baselines.py \\
        --task hmmt --baseline maj8 --seed 0 \\
        --model deepseek-v3.1:671b --max-workers 4

Each invocation writes:
    runs/wow_push/<task>_<baseline>_<model_safe>_seed<n>.json
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


SYS_AIME = (
    "You are a careful mathematics solver. The answer is an integer 0-999.\n"
    "Reason step by step. End with: Answer: N (integer only on the final line)."
)
SYS_HMMT = (
    "You are a careful mathematics solver. The answer is an integer (possibly "
    "large or negative).\n"
    "Reason step by step. End with: Answer: N (integer only on the final line)."
)

# Alternate system prompts the random / OPRO baselines pull from.
ALT_SYS_PROMPTS = [
    "Solve the problem. Show every algebraic step. End with: Answer: N",
    "You are a math olympiad expert. Think carefully. End with: Answer: N",
    "Decompose the problem into sub-claims, verify each, then conclude with: Answer: N",
]


# ---------------- task / parser dispatch ----------------

def _load_task(name: str):
    if name == "hmmt":
        return build_hmmt_feb2025_task(integer_only=True, verify=True), parse_hmmt_int_answer, SYS_HMMT
    if name == "aime25":
        return build_aime25_task(verify=True), parse_aime_answer, SYS_AIME
    raise SystemExit(f"unknown --task {name}")


def _gold_int(label: str) -> int | None:
    try:
        return int(str(label).strip())
    except (TypeError, ValueError):
        return None


def _correct(parsed: int | None, gold: int | None) -> bool:
    return parsed is not None and gold is not None and parsed == gold


def _majority(votes: list[int | None]) -> int | None:
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


# ---------------- baseline implementations ----------------

def _gen_samples(client, system, user, n, temperature, max_tokens):
    """Generate n samples sequentially. (Flattened parallelism happens
    one level up in ``_baseline_majN``, where every (problem, sample)
    pair is dispatched to a single ThreadPoolExecutor — the n inside a
    single problem still has to be serial when called this way, but
    we don't use this path in the hot baselines anymore.)"""
    out = []
    for _ in range(n):
        resp = client.complete(system=system, user=user,
                               max_tokens=max_tokens, temperature=temperature)
        out.append(resp)
    return out


def baseline_cot(client, problems, parser, system, max_tokens, _seed,
                 max_workers):
    return _baseline_majN(client, problems, parser, system, max_tokens,
                          n=1, temperature=0.0, max_workers=max_workers)


def baseline_maj8(client, problems, parser, system, max_tokens, _seed,
                  max_workers):
    return _baseline_majN(client, problems, parser, system, max_tokens,
                          n=8, temperature=0.7, max_workers=max_workers)


def baseline_maj16(client, problems, parser, system, max_tokens, _seed,
                   max_workers):
    return _baseline_majN(client, problems, parser, system, max_tokens,
                          n=16, temperature=0.7, max_workers=max_workers)


def _baseline_majN(client, problems, parser, system, max_tokens, *,
                   n, temperature, max_workers):
    """Self-consistency MAJ@n with FLAT parallelism.

    All (problem_idx, sample_idx) pairs are dispatched to a single
    ThreadPoolExecutor so workers fan out across both axes — not just
    across problems. This unlocks the model's full concurrency budget;
    on a 14-problem × 16-sample maj16 sweep, 4 workers got ~46 min wall
    by parallelizing across problems only; 16 workers across all 224
    calls gets ~12 min.
    """
    def one_call(args):
        i, _s = args
        ex = problems[i]
        resp = client.complete(system=system, user=ex.input,
                               max_tokens=max_tokens, temperature=temperature)
        return i, parser(resp.text)

    pairs = [(i, s) for i in range(len(problems)) for s in range(n)]
    votes_by_problem: dict[int, list[int | None]] = {i: [] for i in range(len(problems))}

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for fut in as_completed(pool.submit(one_call, p) for p in pairs):
            i, vote = fut.result()
            votes_by_problem[i].append(vote)

    results: dict[int, tuple] = {}
    for i, votes in votes_by_problem.items():
        pred = _majority(votes)
        ok = _correct(pred, _gold_int(problems[i].label))
        results[i] = (ok, pred, votes)

    correct = sum(1 for v in results.values() if v[0])
    return {
        "method": f"maj{n}_T{temperature}",
        "n_samples_per_problem": n,
        "temperature": temperature,
        "n_correct": correct,
        "n_total": len(problems),
        "accuracy": correct / max(1, len(problems)),
        "per_problem": [
            {"idx": i, "ok": results[i][0], "pred": results[i][1],
             "votes": results[i][2]}
            for i in sorted(results)
        ],
    }


def baseline_dspy(client, problems, parser, system, max_tokens, seed,
                  max_workers):
    """Bootstrap-FewShot adapted: leave-one-out, demos = correctly-solved peers.

    For each eval problem, run zero-shot CoT on the OTHER 13 problems (cached
    across the first call), then prompt the eval problem with the top-K
    correct demos as few-shot examples. Final answer uses MAJ@8.
    """
    # 1. Zero-shot pass on every problem to identify "correct" demos.
    n_problems = len(problems)
    demo_results = _baseline_majN(
        client, problems, parser, system, max_tokens,
        n=1, temperature=0.0, max_workers=max_workers,
    )
    # Demos = list of (problem_text, gold_label_text, model_response_text)
    correct_demos: list[tuple[str, str]] = []
    for i, pp in enumerate(demo_results["per_problem"]):
        if pp["ok"]:
            correct_demos.append((problems[i].input, problems[i].label))

    if not correct_demos:
        # No demos qualified; fall back to MAJ@8 zero-shot.
        return {
            **baseline_maj8(client, problems, parser, system, max_tokens, seed,
                            max_workers),
            "method": "dspy_fallback_no_demos",
        }

    rng = random.Random(seed)
    K = min(3, len(correct_demos))

    def with_demos_for(eval_idx):
        demos = [d for d in correct_demos if d[0] != problems[eval_idx].input]
        rng.shuffle(demos)
        chosen = demos[:K]
        demo_block = "\n\n".join(
            f"Problem: {p}\nAnswer: {a}" for p, a in chosen
        )
        return demo_block

    # Pre-build one prompt per problem (each gets its own demo set), then
    # fan out (problem × sample) pairs across the pool — same flatten
    # pattern as _baseline_majN.
    prompts = [
        f"{with_demos_for(i)}\n\nNow solve this problem:\n{problems[i].input}"
        for i in range(n_problems)
    ]
    n_samples = 8

    def one_call(args):
        i, _s = args
        resp = client.complete(system=system, user=prompts[i],
                               max_tokens=max_tokens, temperature=0.7)
        return i, parser(resp.text)

    pairs = [(i, s) for i in range(n_problems) for s in range(n_samples)]
    votes_by_problem: dict[int, list[int | None]] = {i: [] for i in range(n_problems)}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for fut in as_completed(pool.submit(one_call, p) for p in pairs):
            i, vote = fut.result()
            votes_by_problem[i].append(vote)

    results: dict[int, tuple] = {}
    for i, votes in votes_by_problem.items():
        pred = _majority(votes)
        ok = _correct(pred, _gold_int(problems[i].label))
        results[i] = (ok, pred, votes)

    correct = sum(1 for v in results.values() if v[0])
    return {
        "method": f"dspy_bootstrap_K{K}_maj8",
        "n_demos_pool": len(correct_demos),
        "n_correct": correct,
        "n_total": n_problems,
        "accuracy": correct / max(1, n_problems),
        "per_problem": [
            {"idx": i, "ok": results[i][0], "pred": results[i][1],
             "votes": results[i][2]}
            for i in sorted(results)
        ],
    }


def baseline_opro(client, problems, parser, system, max_tokens, seed,
                  max_workers, n_candidate_prompts: int = 6,
                  calibration_size: int = 5):
    """OPRO-style instruction search.

    Step 1: ask the model to propose K candidate system prompts.
    Step 2: score each candidate on a calibration slice (single sample, T=0).
    Step 3: pick the highest-scoring candidate; run final eval with MAJ@8.
    """
    rng = random.Random(seed)
    proposer_user = (
        "Propose a single, short system prompt for an LLM that solves "
        "competition math problems whose answer is an integer. The prompt "
        "should encourage careful step-by-step reasoning and end with the "
        "literal phrase 'Answer: N'. Output ONLY the prompt text — one line."
    )
    candidates: list[str] = [system]  # always include the default
    seed_proposer_prompt = (
        "You are an expert prompt engineer. Output only the new system "
        "prompt — no quotes, no commentary."
    )
    for _ in range(n_candidate_prompts - 1):
        try:
            resp = client.complete(
                system=seed_proposer_prompt, user=proposer_user,
                max_tokens=200, temperature=0.7 + rng.random() * 0.3,
            )
            cand = resp.text.strip().split("\n")[0]
            if cand and len(cand) < 600 and "Answer" in cand:
                candidates.append(cand)
        except Exception:
            continue

    cal_problems = problems[: min(calibration_size, len(problems))]

    def cal_score(cand):
        def one(ex):
            r = client.complete(system=cand, user=ex.input,
                                max_tokens=max_tokens, temperature=0.0)
            return _correct(parser(r.text), _gold_int(ex.label))
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            return sum(int(b) for b in pool.map(one, cal_problems))

    cal_scores = [(cand, cal_score(cand)) for cand in candidates]
    best = max(cal_scores, key=lambda kv: kv[1])
    best_cand = best[0]

    eval_result = _baseline_majN(
        client, problems, parser, best_cand, max_tokens,
        n=8, temperature=0.7, max_workers=max_workers,
    )
    return {
        "method": "opro_instruction_search_maj8",
        "n_candidate_prompts": n_candidate_prompts,
        "calibration_size": len(cal_problems),
        "calibration_scores": [(c[:80], int(s)) for c, s in cal_scores],
        "winning_prompt_excerpt": best_cand[:200],
        **{k: v for k, v in eval_result.items() if k not in ("method",)},
    }


def baseline_random(client, problems, parser, system, max_tokens, seed,
                    max_workers, n_candidates: int = 6,
                    calibration_size: int = 5):
    """Random hyperparameter search.

    Sample candidates over (system_prompt, n_samples, temperature, vote).
    Calibrate on a slice, pick the best, run final eval.
    """
    rng = random.Random(seed)
    n_choices = [1, 2, 4, 8]
    temps = [0.0, 0.3, 0.6, 0.9]
    sys_pool = [system] + ALT_SYS_PROMPTS

    candidates: list[dict] = []
    for _ in range(n_candidates):
        candidates.append({
            "system": rng.choice(sys_pool),
            "n_samples": rng.choice(n_choices),
            "temperature": rng.choice(temps),
        })

    cal_problems = problems[: min(calibration_size, len(problems))]

    def score(cand):
        def one(ex):
            resps = _gen_samples(client, cand["system"], ex.input,
                                 cand["n_samples"], cand["temperature"],
                                 max_tokens)
            votes = [parser(r.text) for r in resps]
            pred = _majority(votes)
            return _correct(pred, _gold_int(ex.label))
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            return sum(int(b) for b in pool.map(one, cal_problems))

    cal_scores = [(c, score(c)) for c in candidates]
    best = max(cal_scores, key=lambda kv: kv[1])
    best_cand = best[0]

    eval_result = _baseline_majN(
        client, problems, parser, best_cand["system"], max_tokens,
        n=best_cand["n_samples"],
        temperature=best_cand["temperature"], max_workers=max_workers,
    )
    return {
        "method": "random_search",
        "n_candidates": n_candidates,
        "calibration_scores": [
            ({"sys": c["system"][:60], "n": c["n_samples"], "T": c["temperature"]}, s)
            for c, s in cal_scores
        ],
        "winning_config": {
            "system_prompt_excerpt": best_cand["system"][:80],
            "n_samples": best_cand["n_samples"],
            "temperature": best_cand["temperature"],
        },
        **{k: v for k, v in eval_result.items() if k not in ("method",)},
    }


BASELINES = {
    "cot": baseline_cot,
    "maj8": baseline_maj8,
    "maj16": baseline_maj16,
    "dspy": baseline_dspy,
    "opro": baseline_opro,
    "random": baseline_random,
}


# ---------------- driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["hmmt", "aime25"])
    ap.add_argument("--baseline", required=True, choices=list(BASELINES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=None,
                    help="Cap eval set size; default = full task.")
    ap.add_argument("--out-dir", default="runs/wow_push")
    args = ap.parse_args()

    _load_dotenv_if_present()
    url = os.environ["OLLAMA_CLOUD_URL"]
    key = os.environ["OLLAMA_API_KEY"]

    task, parser, system = _load_task(args.task)
    problems = task.eval_set
    if args.n_eval is not None:
        problems = problems[: args.n_eval]
    print(f"[base] task={args.task} baseline={args.baseline} seed={args.seed} "
          f"model={args.model} n_problems={len(problems)}")

    raw = HTTPClient(api_url=url, api_key=key, model=args.model,
                     timeout_s=300.0, max_retries=8,
                     retry_base_delay=2.0, retry_max_delay=60.0)
    client = _AccountingClient(raw)

    fn = BASELINES[args.baseline]
    t0 = time.perf_counter()
    result = fn(client, problems, parser, system, args.max_tokens,
                args.seed, args.max_workers)
    wall = time.perf_counter() - t0

    record = {
        "task": args.task,
        "baseline": args.baseline,
        "model": args.model,
        "seed": args.seed,
        "wall_seconds": wall,
        "calls": client.calls,
        "in_tokens": client.in_tokens,
        "out_tokens": client.out_tokens,
        "latency_ms_p50": _percentile(client.latencies_ms, 50),
        "latency_ms_p95": _percentile(client.latencies_ms, 95),
        "result": result,
    }
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_{args.baseline}_{safe_model}_seed{args.seed}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"[base] PASS acc={result['accuracy']:.3f} wall={wall:.1f}s "
          f"calls={client.calls} p50={record['latency_ms_p50']:.0f}ms "
          f"-> {out_path}")

    _record_ollama_batch(
        phase=f"phase2_{args.task}",
        label=f"{args.baseline}_seed{args.seed}",
        model=args.model, url=url,
        calls=client.calls, in_tokens=client.in_tokens,
        out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
    )


if __name__ == "__main__":
    main()

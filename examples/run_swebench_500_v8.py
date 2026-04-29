"""V8 driver — V7 + Pass@N (best-of-N seed sampling).

V8 = V7 layers + multi-seed sampling at varied temperatures, with eval-driven
winner selection per instance.

Per task:
  1. Run N seeds (default 3) at T=0.0, 0.3, 0.7
  2. Each seed produces its own patch via the V7 actor + reviewer + revision pipeline
  3. After all N complete, run swebench eval on all N patches
  4. Per-instance winner = first seed that resolves;
     fallback = highest-quality stop_reason (done > tests_passed > others)

Cost note: with prefix caching at 99% hit, N seeds ~ N× cost on output tokens
(input is mostly cached). Realistic V8 spend on full 500 ≈ 3× V7 ≈ $20-30.

Usage:
    python3 examples/run_swebench_500_v8.py --n 500 --max-workers 12 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import _load_dotenv_if_present  # type: ignore
from examples.run_swebench_500_v7 import (
    _api_key_for, _build_chat_fn, run_one_v7_task,
    evaluate_v7_patches, disk_free_gb,
)
from meta_harness_plus.agent_swebench_loop import (
    BudgetGuard, RecoveryPolicy, price_for_model,
)
from meta_harness_plus.swebench_adapter import (
    SWEBenchInstance, load_swebench_verified, run_swebench_eval, write_predictions,
)


# Temperature schedule for Pass@N sampling. T=0 anchors deterministic best-shot,
# T=0.3 light exploration, T=0.7 wider sampling.
SEED_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]


def _stop_quality_score(stop_reason: str | None) -> int:
    """Higher = more confident the agent thinks it's done."""
    if not stop_reason:
        return 0
    s = stop_reason.lower()
    if "done" in s: return 4
    if "tests_passed" in s: return 3
    if "max_turns" in s: return 2
    if "tokens" in s: return 1
    return 0


def select_winner(per_seed_records: list[dict],
                  resolved_lookup: dict[tuple[str, int], bool]) -> dict:
    """Pick the best record across seeds. Priority:
    1. Any seed that's resolved by swebench eval.
    2. Highest stop-reason quality (done > tests_passed > max_turns > tokens).
    3. Larger non-empty patch.
    4. First seed."""
    iid = per_seed_records[0]["instance_id"]
    resolved_seeds = [
        r for r in per_seed_records
        if resolved_lookup.get((iid, r["seed"]), False)
    ]
    if resolved_seeds:
        return resolved_seeds[0]
    nonempty = [r for r in per_seed_records if r.get("patch_len", 0) > 0]
    if nonempty:
        nonempty.sort(
            key=lambda r: (
                _stop_quality_score(r.get("stop_reason")),
                r.get("patch_len", 0),
            ),
            reverse=True,
        )
        return nonempty[0]
    return per_seed_records[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--seeds", type=int, default=3,
                    help="Number of independent seeds per instance (Pass@N).")
    ap.add_argument("--temperatures", default=None,
                    help="Comma-separated temps; default: schedule by seed count.")
    ap.add_argument("--provider", default="deepseek",
                    choices=["anthropic", "deepseek", "ollama"])
    ap.add_argument("--model", default="deepseek-reasoner")
    ap.add_argument("--reviewer-provider", default=None)
    ap.add_argument("--reviewer-model", default=None)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--revision-turns", type=int, default=6)
    ap.add_argument("--max-tokens-per-turn", type=int, default=8192)
    ap.add_argument("--max-reviewer-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens-in", type=int, default=1_000_000)
    ap.add_argument("--max-wall-s", type=float, default=2700.0)
    ap.add_argument("--max-usd", type=float, default=999.0)
    ap.add_argument("--max-recoveries", type=int, default=2)
    ap.add_argument("--same-file-edit-threshold", type=int, default=3)
    ap.add_argument("--require-reproduction-before-edit", action="store_true",
                    default=False)
    ap.add_argument("--cleanup-images", action="store_true", default=True)
    ap.add_argument("--no-cleanup-images", action="store_false",
                    dest="cleanup_images")
    ap.add_argument("--out-dir", default="runs/swebench_500_v8")
    ap.add_argument("--run-id", default="v8")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    _load_dotenv_if_present()
    if args.temperatures:
        temps = [float(t) for t in args.temperatures.split(",")]
    else:
        temps = SEED_TEMPERATURES[: args.seeds] if args.seeds <= len(SEED_TEMPERATURES) \
                else SEED_TEMPERATURES + [0.7] * (args.seeds - len(SEED_TEMPERATURES))
    assert len(temps) >= args.seeds, "need at least seeds temperatures"
    temps = temps[: args.seeds]

    reviewer_provider = args.reviewer_provider or args.provider
    reviewer_model = args.reviewer_model or args.model
    actor_chat_fn = _build_chat_fn(args.provider, _api_key_for(args.provider), args.model)
    reviewer_chat_fn = _build_chat_fn(
        reviewer_provider, _api_key_for(reviewer_provider), reviewer_model)

    if args.instance:
        instances = load_swebench_verified(instance_ids=[args.instance])
    else:
        instances = load_swebench_verified(n=args.n)

    print("[v8] benchmark: SWE-bench Verified  (Pass@N)")
    print(f"[v8] instances={len(instances)} seeds={args.seeds} temps={temps}")
    print(f"[v8] workers={args.max_workers} actor={args.provider}/{args.model}")
    print(f"[v8] turns={args.max_turns}+{args.revision_turns}  usd_cap=${args.max_usd}")
    print(f"[v8] disk free: {disk_free_gb():.1f} GB")

    budget = BudgetGuard(
        max_turns=args.max_turns, max_tokens_in=args.max_tokens_in,
        max_usd=args.max_usd, max_wall_s=args.max_wall_s,
    )
    recovery = RecoveryPolicy(
        enabled=True, same_file_edit_threshold=args.same_file_edit_threshold,
        max_recoveries=args.max_recoveries,
    )

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    # Build the (instance, seed) cell list. Seed maps to a temperature.
    cells = [(inst, seed) for inst in instances for seed in range(args.seeds)]
    print(f"[v8] {len(cells)} (instance × seed) cells to run")

    # Re-route run_one_v7_task to a temperature-aware wrapper.
    def task_for_cell(inst: SWEBenchInstance, seed: int) -> dict:
        # Override temperature via a wrapped chat_fn that injects temperature.
        # Actually, run_one_v7_task uses its own AgentShape internally (default
        # temp=0.0). We monkey-patch the temperature by wrapping the chat_fn
        # to override `temperature` in the request — simpler than threading
        # a new arg through run_one_v7_task.
        seed_temp = temps[seed]
        if seed_temp == 0.0:
            seed_actor = actor_chat_fn
        else:
            def _seed_chat(*, system, messages, tools, max_tokens=4096, temperature=0.0):
                return actor_chat_fn(
                    system=system, messages=messages, tools=tools,
                    max_tokens=max_tokens, temperature=seed_temp,
                )
            seed_actor = _seed_chat
        return run_one_v7_task(
            inst=inst, seed=seed,
            actor_chat_fn=seed_actor, reviewer_chat_fn=reviewer_chat_fn,
            model=args.model, reviewer_model=reviewer_model,
            cache_dir=cache_dir,
            max_turns=args.max_turns, revision_turns=args.revision_turns,
            max_tokens_per_turn=args.max_tokens_per_turn,
            budget_guard=budget, recovery_policy=recovery,
            require_reproduction_before_edit=args.require_reproduction_before_edit,
            max_reviewer_tokens=args.max_reviewer_tokens,
            cleanup_image=args.cleanup_images,
        )

    t0 = time.perf_counter()
    records: list[dict] = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(task_for_cell, inst, seed): (inst.instance_id, seed)
                for inst, seed in cells}
        for fut in as_completed(futs):
            iid, seed = futs[fut]
            rec = fut.result()
            records.append(rec)
            n_done += 1
            elapsed = time.perf_counter() - t0
            print(f"[v8] +{elapsed:6.0f}s ({n_done:>4d}/{len(cells)}) "
                  f"{iid:38s} seed{seed} (T={temps[seed]}) "
                  f"turns={rec.get('n_turns',0):>2d} "
                  f"patch={rec.get('patch_len',0):>5d}b "
                  f"stop={str(rec.get('stop_reason',''))[:22]:<22s}",
                  flush=True)

    sweep_wall = time.perf_counter() - t0
    print(f"\n[v8] sweep wall: {sweep_wall:.0f}s ({sweep_wall/3600:.2f}h)")

    if args.skip_eval:
        out_path = out_dir / f"{args.run_id}.json"
        out_path.write_text(json.dumps({
            "run_id": args.run_id, "model": args.model,
            "n_instances": len(instances), "seeds": args.seeds, "temps": temps,
            "records": records, "sweep_wall_s": sweep_wall, "evaluated": False,
        }, indent=2))
        print(f"[v8] wrote {out_path} (eval skipped)")
        return

    # Eval each seed separately, then aggregate per instance to pick winner.
    print(f"\n[v8] grading {args.seeds} × {len(instances)} = "
          f"{args.seeds * len(instances)} patches via swebench eval...")
    per_seed_eval: dict[int, dict[str, dict]] = {}
    for seed in range(args.seeds):
        seed_recs = [r for r in records if r["seed"] == seed]
        run_id_full = f"{args.run_id}_seed{seed}"
        res = evaluate_v7_patches(
            records=seed_recs, run_id=run_id_full,
            predictions_dir=out_dir / "predictions", model=args.model,
        )
        per_seed_eval[seed] = res
        n_resolved = sum(1 for v in res.values() if v.get("resolved"))
        print(f"[v8] seed {seed} (T={temps[seed]}): {n_resolved}/{len(res)} resolved")

    # Build resolved_lookup and pick per-instance winner.
    resolved_lookup: dict[tuple[str, int], bool] = {}
    for seed, res in per_seed_eval.items():
        for iid, info in res.items():
            resolved_lookup[(iid, seed)] = bool(info.get("resolved"))

    # Group records by instance, pick winner per instance.
    by_inst: dict[str, list[dict]] = {}
    for r in records:
        by_inst.setdefault(r["instance_id"], []).append(r)
    winners: list[dict] = []
    for iid, recs in by_inst.items():
        winners.append(select_winner(recs, resolved_lookup))

    # Final headline = winners with patches that are resolved across any seed.
    n_resolved_winner = sum(
        1 for w in winners
        if any(resolved_lookup.get((w["instance_id"], s), False)
               for s in range(args.seeds))
    )
    pct = 100 * n_resolved_winner / max(1, len(winners))
    print()
    print("=" * 60)
    print(f"V8 PASS@{args.seeds} HEADLINE")
    print("=" * 60)
    for seed in range(args.seeds):
        seed_recs = [r for r in records if r["seed"] == seed]
        n_seed = sum(1 for r in seed_recs
                     if resolved_lookup.get((r["instance_id"], seed), False))
        print(f"  seed {seed} (T={temps[seed]:.1f}) alone: {n_seed}/{len(seed_recs)} "
              f"= {100*n_seed/max(1,len(seed_recs)):.1f}%")
    print(f"  Pass@{args.seeds} (any seed resolves): "
          f"{n_resolved_winner}/{len(winners)} = {pct:.1f}%")

    out_path = out_dir / f"{args.run_id}.json"
    out_path.write_text(json.dumps({
        "run_id": args.run_id, "model": args.model,
        "n_instances": len(instances), "seeds": args.seeds, "temps": temps,
        "records": records,
        "winners": [w["instance_id"] for w in winners],
        "per_seed_eval": {str(k): v for k, v in per_seed_eval.items()},
        "n_resolved_pass_at_n": n_resolved_winner,
        "pct_resolved_pass_at_n": pct,
        "sweep_wall_s": sweep_wall, "evaluated": True,
    }, indent=2))
    print(f"\n[v8] wrote {out_path}")


if __name__ == "__main__":
    main()

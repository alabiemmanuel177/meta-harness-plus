"""V5 driver — full 500-task SWE-bench Verified, leaderboard-credible config.

Conditions (per user spec):
- 500-task SWE-bench Verified (not the 50-slice)
- max_turns=50, temperature=0.0
- per-task wall cap = 2700s (45 min)
- single-command timeout = 210s (already set in agent_swebench_loop)
- start at 12 parallel workers, push to 16+ if Docker holds
- no golden patch, no external RAG, FAIL_TO_PASS selectors kept (standard practice)
- pull-as-you-go + post-task instance image cleanup (working set ≤150 GB)
- full trajectory logging to trajectories_full/
- value-per-hour summary at end

Resume-safe: per-(instance, shape, seed) JSON cache; reruns skip done work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import _load_dotenv_if_present  # type: ignore
from examples.run_swebench_agent import (
    SHAPES, AgentShape, run_one_task,
    _build_anthropic_chat_fn, _build_ollama_chat_fn, _build_deepseek_chat_fn,
    evaluate_patches,
)
from meta_harness_plus.agent_swebench_loop import BudgetGuard
from meta_harness_plus.swebench_adapter import load_swebench_verified


def disk_free_gb(path: str = "/") -> float:
    out = subprocess.check_output(["df", "-BG", path]).decode()
    line = out.splitlines()[1]
    avail = line.split()[3].rstrip("G")
    return float(avail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500,
                    help="Number of instances (500 = full SWE-bench Verified).")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--shape", default="baseline", choices=list(SHAPES.keys()))
    ap.add_argument("--provider", default="deepseek",
                    choices=["anthropic", "ollama", "deepseek"])
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--max-workers", type=int, default=12,
                    help="Parallel agent loops; ramp up cautiously.")
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--max-tokens-in", type=int, default=1_000_000)
    ap.add_argument("--max-usd", type=float, default=999.0,
                    help="Per-task USD budget cap (Ollama is free).")
    ap.add_argument("--max-wall-s", type=float, default=2700.0)
    ap.add_argument("--cleanup-images", action="store_true", default=True,
                    help="docker rmi instance image after each task.")
    ap.add_argument("--no-cleanup-images", action="store_false",
                    dest="cleanup_images")
    ap.add_argument("--verbose-log", action="store_true", default=True,
                    help="Write full-fidelity trajectory dumps.")
    ap.add_argument("--out-dir", default="runs/swebench_500_v5")
    ap.add_argument("--run-id", default="v5")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--instance", default=None,
                    help="Single instance for smoke; overrides --n.")
    args = ap.parse_args()

    _load_dotenv_if_present()
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY missing")
        chat_fn = _build_anthropic_chat_fn(api_key, args.model)
    elif args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY missing")
        chat_fn = _build_deepseek_chat_fn(api_key, args.model)
    else:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise SystemExit("OLLAMA_API_KEY missing")
        chat_fn = _build_ollama_chat_fn(api_key, args.model)

    if args.instance:
        instances = load_swebench_verified(instance_ids=[args.instance])
    else:
        instances = load_swebench_verified(n=args.n)
    print(f"[v5] benchmark: SWE-bench Verified")
    print(f"[v5] instances: {len(instances)}  shape={args.shape}  seeds={args.seeds}")
    print(f"[v5] provider={args.provider}  model={args.model}")
    print(f"[v5] max_turns={args.max_turns}  max_workers={args.max_workers}  "
          f"max_wall_s={args.max_wall_s}  cleanup_images={args.cleanup_images}")
    print(f"[v5] disk free: {disk_free_gb('/'):.1f} GB")

    shape = SHAPES[args.shape]
    if shape.max_turns != args.max_turns:
        shape = AgentShape(
            name=shape.name, system_prompt=shape.system_prompt,
            tool_set=shape.tool_set, temperature=shape.temperature,
            max_turns=args.max_turns,
            max_tokens_per_turn=shape.max_tokens_per_turn,
        )
    budget = BudgetGuard(
        max_turns=args.max_turns,
        max_tokens_in=args.max_tokens_in,
        max_usd=args.max_usd,
        max_wall_s=args.max_wall_s,
    )

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    pairs = [(inst, seed) for inst in instances for seed in range(args.seeds)]
    print(f"[v5] {len(pairs)} (instance × seed) cells")

    t0 = time.perf_counter()
    records: list[dict] = []
    n_done = 0
    n_resolved_so_far = 0  # rough estimate; real eval is at end
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {
            pool.submit(
                run_one_task,
                inst=inst, shape=shape, seed=seed, chat_fn=chat_fn,
                model=args.model, cache_dir=cache_dir, budget_guard=budget,
                cleanup_image=args.cleanup_images,
                verbose_log=args.verbose_log,
            ): (inst.instance_id, seed)
            for inst, seed in pairs
        }
        for fut in as_completed(futs):
            iid, seed = futs[fut]
            rec = fut.result()
            records.append(rec)
            n_done += 1
            elapsed = time.perf_counter() - t0
            tasks_per_hour = n_done / max(0.001, elapsed) * 3600
            disk = disk_free_gb('/')
            print(
                f"[v5] +{elapsed:6.0f}s  ({n_done:>3d}/{len(pairs)})  "
                f"{iid:38s} t={rec.get('n_turns',0):>2d} "
                f"patch={rec.get('patch_len',0):>5d}b "
                f"stop={rec.get('stop_reason','')[:25]:<25s}  "
                f"|  rate={tasks_per_hour:.1f}/h  disk={disk:.0f}GB",
                flush=True,
            )

    sweep_wall = time.perf_counter() - t0
    print(f"\n[v5] sweep wall: {sweep_wall:.0f}s ({sweep_wall/60:.1f} min, "
          f"{sweep_wall/3600:.2f} h)")
    total_usd = sum(r.get("usd", 0) for r in records)
    print(f"[v5] total inference cost: ${total_usd:.3f}")

    if args.skip_eval:
        out_path = out_dir / f"{args.run_id}_{args.shape}.json"
        out_path.write_text(json.dumps({
            "run_id": args.run_id, "shape": args.shape, "model": args.model,
            "provider": args.provider, "n_instances": len(instances),
            "records": records, "sweep_wall_s": sweep_wall,
            "total_usd": total_usd, "evaluated": False,
            "config": {
                "max_turns": args.max_turns, "max_tokens_in": args.max_tokens_in,
                "max_wall_s": args.max_wall_s, "max_workers": args.max_workers,
                "cleanup_images": args.cleanup_images,
            },
        }, indent=2))
        print(f"[v5] wrote {out_path} (eval skipped)")
        return

    print(f"\n[v5] grading {len(instances)} patches via swebench eval...")
    per_seed_resolved: dict[int, dict[str, bool]] = {}
    for seed in range(args.seeds):
        seed_records = [r for r in records if r["seed"] == seed]
        run_id_full = f"{args.run_id}_{args.shape}_seed{seed}"
        res = evaluate_patches(
            records=seed_records, run_id=run_id_full,
            predictions_dir=out_dir / "predictions", model=args.model,
        )
        per_seed_resolved[seed] = res
        n_resolved = sum(1 for v in res.values() if v)
        n_total = len(seed_records)
        pct = 100 * n_resolved / max(1, n_total)
        print(f"\n>>> V5 seed {seed}: {n_resolved}/{n_total} = {pct:.1f}% RESOLVED <<<")

    out_path = out_dir / f"{args.run_id}_{args.shape}.json"
    out_path.write_text(json.dumps({
        "run_id": args.run_id, "shape": args.shape, "model": args.model,
        "provider": args.provider, "n_instances": len(instances),
        "records": records,
        "per_seed_resolved": {str(k): v for k, v in per_seed_resolved.items()},
        "sweep_wall_s": sweep_wall, "total_usd": total_usd,
        "config": {
            "max_turns": args.max_turns, "max_tokens_in": args.max_tokens_in,
            "max_wall_s": args.max_wall_s, "max_workers": args.max_workers,
            "cleanup_images": args.cleanup_images,
        },
        "evaluated": True,
    }, indent=2))
    print(f"\n[v5] wrote {out_path}")

    # Headline summary block
    print("\n" + "=" * 70)
    print("V5 — SWE-bench Verified — Headline")
    print("=" * 70)
    seed0 = per_seed_resolved.get(0, {})
    total = len(seed0)
    n_ok = sum(1 for v in seed0.values() if v)
    n_empty = sum(1 for r in records if r.get('seed', 0) == 0
                  and r.get('patch_len', 0) == 0)
    pct = 100 * n_ok / max(1, total)
    tasks_per_hour = total / max(0.001, sweep_wall) * 3600
    resolved_per_hour = n_ok / max(0.001, sweep_wall) * 3600
    print(f"  resolved        : {n_ok}/{total} = {pct:.1f}%")
    print(f"  empty patches   : {n_empty}")
    print(f"  sweep wall      : {sweep_wall/3600:.2f} hours")
    print(f"  tasks/hour      : {tasks_per_hour:.1f}")
    print(f"  resolved/hour   : {resolved_per_hour:.1f}  ← value-per-hour")
    print(f"  total cost      : ${total_usd:.2f}")
    if total_usd > 0:
        print(f"  $/resolved      : ${total_usd / max(1, n_ok):.2f}")
    print(f"  config          : turns={args.max_turns}, tokens={args.max_tokens_in:,}, "
          f"wall={args.max_wall_s:.0f}s, workers={args.max_workers}")


if __name__ == "__main__":
    main()

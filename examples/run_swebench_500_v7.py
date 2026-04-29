"""V7 driver — SWE-bench Verified with reviewer-actor verification.

V7 layers over the v5 Docker agent:
- Actor loop with LSP-style tools, reproduction-first gating, and rollback.
- Independent reviewer model audits the final patch before official eval.
- Reviewer can generate an edge-case reproduction that runs in the same
  container before submission.
- If reviewer/reproduction finds a real risk, a short revision actor pass
  can repair the patch.
- Test-fix proposals are recorded as audit artifacts for noisy/impossible
  benchmark tasks; they are not mixed into the production patch.

Usage:
    python3 examples/run_swebench_500_v7.py --instance sympy__sympy-20916 --skip-eval
    python3 examples/run_swebench_500_v7.py --n 500 --max-workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import _load_dotenv_if_present  # type: ignore
from examples.run_swebench_agent import (
    AgentShape,
    _build_anthropic_chat_fn,
    _build_deepseek_chat_fn,
    _build_ollama_chat_fn,
)
from meta_harness_plus.agent_docker import DockerShellExecutor
from meta_harness_plus.agent_swebench_loop import (
    BudgetGuard,
    RecoveryPolicy,
    Trajectory,
    price_for_model,
    run_agent_loop,
)
from meta_harness_plus.swebench_adapter import (
    SWEBenchInstance,
    load_swebench_verified,
    run_swebench_eval,
    write_predictions,
)
from meta_harness_plus.swebench_v7 import (
    V7_ACTOR_SYSTEM_PROMPT,
    build_revision_problem_statement,
    run_patch_reviewer,
)


def disk_free_gb(path: str = "/") -> float:
    out = subprocess.check_output(["df", "-BG", path]).decode()
    line = out.splitlines()[1]
    return float(line.split()[3].rstrip("G"))


def _build_chat_fn(provider: str, api_key: str, model: str):
    if provider == "anthropic":
        return _build_anthropic_chat_fn(api_key, model)
    if provider == "deepseek":
        return _build_deepseek_chat_fn(api_key, model)
    return _build_ollama_chat_fn(api_key, model)


def _api_key_for(provider: str) -> str:
    env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "ollama": "OLLAMA_API_KEY",
    }[provider]
    key = os.environ.get(env)
    if not key:
        raise SystemExit(f"{env} missing")
    return key


def _abbrev(value: str, limit: int = 800) -> str:
    value = value or ""
    if len(value) <= limit:
        return value
    return value[:limit] + f"...[{len(value) - limit} more chars]"


def _has_pass_to_pass_regression(report: dict[str, Any]) -> bool:
    """Detect a real PASS_TO_PASS regression in a SWE-bench eval report.

    Schema (verified on actual reports):
        report["tests_status"]["PASS_TO_PASS"] = {"success": [...], "failure": [...]}
    A regression = at least one previously-passing test now in the
    ``failure`` list. We do NOT treat large ``success`` lists as failures.
    Older code had a bug that recursed into nested lists and flagged any
    non-empty collection as a failure — that's why V7 smoke v2 said 3/3
    regressed even though only one task was actually broken.
    """
    if not isinstance(report, dict):
        return False
    # Walk to the PASS_TO_PASS bucket regardless of where it sits.
    p2p = None
    if isinstance(report.get("tests_status"), dict):
        p2p = report["tests_status"].get("PASS_TO_PASS")
    if p2p is None:
        p2p = report.get("PASS_TO_PASS")
    if p2p is None:
        return False
    if isinstance(p2p, dict):
        failures = p2p.get("failure") or p2p.get("failed") or p2p.get("failures") or []
        return bool(failures)
    # Some older formats put the failure list directly here.
    if isinstance(p2p, list):
        return False  # treated as success list; no failure info
    return False


def evaluate_v7_patches(
    *,
    records: list[dict[str, Any]],
    run_id: str,
    predictions_dir: Path,
    model: str,
) -> dict[str, dict[str, Any]]:
    predictions = {r["instance_id"]: r.get("patch", "") for r in records}
    preds_path = predictions_dir / f"{run_id}.jsonl"
    write_predictions(predictions, model_name=model, out_path=preds_path)
    try:
        results = run_swebench_eval(
            predictions_path=preds_path,
            run_id=run_id,
            instance_ids=list(predictions.keys()),
            max_workers=8,
        )
    except Exception as exc:
        print(f"[v7] eval crashed: {type(exc).__name__}: {exc}")
        return {}
    out: dict[str, dict[str, Any]] = {}
    for iid, rec in results.items():
        report = rec.get("report", rec)
        out[iid] = {
            "resolved": bool(rec.get("resolved", False)),
            "regressed": _has_pass_to_pass_regression(report),
            "report": report,
        }
    return out


def trajectory_to_record(traj: Trajectory) -> dict[str, Any]:
    return {
        "n_turns": len(traj.turns),
        "done_emitted": traj.done_emitted,
        "stop_reason": traj.stop_reason,
        "in_tokens": traj.total_in_tokens,
        "out_tokens": traj.total_out_tokens,
        "wall_s": traj.total_wall_s,
        "usd": traj.total_usd,
        "patch_len": len(traj.final_patch),
        "reproduction_attempted": traj.reproduction_attempted,
        "reproduction_passed": traj.reproduction_passed,
        "recovery_events": [asdict(e) for e in traj.recovery_events],
        "test_fix_proposals": traj.test_fix_proposals,
        "turns_detail": [
            {
                "i": t.turn_idx,
                "tool_calls": [
                    {"name": c.get("name"), "input": c.get("input", {})}
                    for c in t.tool_calls
                ],
                "tool_results": [
                    {"obs": _abbrev(r.get("content", "") or r.get("text", ""), 500)}
                    for r in t.tool_results
                ],
                "in_tokens": t.in_tokens,
                "out_tokens": t.out_tokens,
                "wall_ms": round(t.wall_ms, 1),
                "stop_reason": t.stop_reason,
            }
            for t in traj.turns
        ],
    }


def run_one_v7_task(
    *,
    inst: SWEBenchInstance,
    seed: int,
    actor_chat_fn,
    reviewer_chat_fn,
    model: str,
    reviewer_model: str,
    cache_dir: Path,
    max_turns: int,
    revision_turns: int,
    max_tokens_per_turn: int,
    budget_guard: BudgetGuard,
    recovery_policy: RecoveryPolicy,
    require_reproduction_before_edit: bool,
    max_reviewer_tokens: int,
    cleanup_image: bool,
) -> dict[str, Any]:
    cache_path = cache_dir / "trajectories_v7" / f"{inst.instance_id}_seed{seed}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    shape = AgentShape(
        name="v7_sota",
        system_prompt=V7_ACTOR_SYSTEM_PROMPT,
        tool_set="minimal",
        temperature=0.0,
        max_turns=max_turns,
        max_tokens_per_turn=max_tokens_per_turn,
    )

    sh = DockerShellExecutor(
        inst.instance_id,
        no_network=False,
        ensure_pytest=True,
    )
    image_name = sh.image
    t0 = time.perf_counter()
    rec: dict[str, Any] = {
        "instance_id": inst.instance_id,
        "seed": seed,
        "shape": shape.label(),
        "model": model,
        "reviewer_model": reviewer_model,
        "config": {
            "actor": asdict(shape),
            "revision_turns": revision_turns,
            "require_reproduction_before_edit": require_reproduction_before_edit,
            "recovery_policy": asdict(recovery_policy),
        },
    }

    actor_traj: Trajectory | None = None
    revision_traj: Trajectory | None = None
    reviewer_repro: dict[str, Any] | None = None
    reviewer_verdict = None
    final_reviewer_verdict = None

    try:
        sh.start()
        actor_traj = run_agent_loop(
            instance_id=inst.instance_id,
            problem_statement=inst.problem_statement,
            fail_to_pass=inst.fail_to_pass,
            sh=sh,
            chat_fn=actor_chat_fn,
            system_prompt=shape.system_prompt,
            tool_set=shape.tool_set,
            temperature=shape.temperature,
            max_tokens_per_turn=shape.max_tokens_per_turn,
            budget_guard=budget_guard,
            price_per_mtok=price_for_model(model),
            recovery_policy=recovery_policy,
            require_reproduction_before_edit=require_reproduction_before_edit,
        )
        patch = actor_traj.final_patch
        if patch.strip():
            reviewer_verdict = run_patch_reviewer(
                inst=inst,
                patch=patch,
                actor_trajectory=actor_traj,
                reviewer_chat_fn=reviewer_chat_fn,
                model=reviewer_model,
                max_tokens=max_reviewer_tokens,
            )
            if reviewer_verdict.reproduction_code.strip():
                rr = sh.run_reproduction(
                    reviewer_verdict.reproduction_code,
                    timeout_s=90.0,
                )
                reviewer_repro = {
                    "exit_code": rr.exit_code,
                    "stdout": _abbrev(rr.stdout, 1600),
                    "stderr": _abbrev(rr.stderr, 1000),
                    "elapsed_s": rr.elapsed_s,
                }

            # Reviewer is now ADVISORY: text-based "needs_revision" is logged
            # but does not auto-trigger a revision pass. We only revise when
            # the reviewer's own reproduction objectively proves a bug
            # (exit_code != 0). This stops the reviewer from rejecting
            # working fixes and burning revision budget.
            reviewer_found_failure = bool(
                reviewer_repro and reviewer_repro.get("exit_code") != 0
            )
            if revision_turns > 0 and reviewer_found_failure:
                revision_budget = BudgetGuard(
                    max_turns=revision_turns,
                    max_tokens_in=budget_guard.max_tokens_in,
                    max_tokens_out=budget_guard.max_tokens_out,
                    max_wall_s=min(600.0, budget_guard.max_wall_s),
                    max_usd=budget_guard.max_usd,
                )
                revision_traj = run_agent_loop(
                    instance_id=inst.instance_id,
                    problem_statement=build_revision_problem_statement(
                        inst,
                        reviewer_verdict,
                    ),
                    fail_to_pass=inst.fail_to_pass,
                    sh=sh,
                    chat_fn=actor_chat_fn,
                    system_prompt=shape.system_prompt,
                    tool_set=shape.tool_set,
                    temperature=shape.temperature,
                    max_tokens_per_turn=shape.max_tokens_per_turn,
                    budget_guard=revision_budget,
                    price_per_mtok=price_for_model(model),
                    recovery_policy=recovery_policy,
                    require_reproduction_before_edit=False,
                )
                patch = revision_traj.final_patch
                if patch.strip():
                    final_reviewer_verdict = run_patch_reviewer(
                        inst=inst,
                        patch=patch,
                        actor_trajectory=revision_traj,
                        reviewer_chat_fn=reviewer_chat_fn,
                        model=reviewer_model,
                        max_tokens=max_reviewer_tokens,
                    )

        rec["patch"] = sh.get_diff(exclude_tests=True)
        rec["patch_len"] = len(rec["patch"])
        rec["actor"] = trajectory_to_record(actor_traj)
        if revision_traj is not None:
            rec["revision"] = trajectory_to_record(revision_traj)
        if reviewer_verdict is not None:
            rec["reviewer"] = reviewer_verdict.to_dict()
        if final_reviewer_verdict is not None:
            rec["final_reviewer"] = final_reviewer_verdict.to_dict()
        if reviewer_repro is not None:
            rec["reviewer_reproduction"] = reviewer_repro

        total_usd = actor_traj.total_usd
        if revision_traj is not None:
            total_usd += revision_traj.total_usd
        if reviewer_verdict is not None:
            total_usd += reviewer_verdict.usd
        if final_reviewer_verdict is not None:
            total_usd += final_reviewer_verdict.usd
        rec["usd"] = total_usd
        rec["n_turns"] = len(actor_traj.turns) + (
            len(revision_traj.turns) if revision_traj else 0
        )
        rec["stop_reason"] = (
            revision_traj.stop_reason if revision_traj else actor_traj.stop_reason
        )
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["patch"] = ""
        rec["patch_len"] = 0
    finally:
        rec["task_wall_s"] = round(time.perf_counter() - t0, 1)
        sh.cleanup()
        if cleanup_image:
            try:
                subprocess.run(["docker", "rmi", image_name],
                               capture_output=True, timeout=30)
            except Exception:
                pass

    cache_path.write_text(json.dumps(rec, indent=2))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--provider", default="deepseek",
                    choices=["anthropic", "deepseek", "ollama"])
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--reviewer-provider", default=None,
                    choices=["anthropic", "deepseek", "ollama"])
    ap.add_argument("--reviewer-model", default=None)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--max-turns", type=int, default=18)
    ap.add_argument("--revision-turns", type=int, default=6)
    ap.add_argument("--max-tokens-per-turn", type=int, default=8192)
    ap.add_argument("--max-reviewer-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens-in", type=int, default=500_000)
    ap.add_argument("--max-wall-s", type=float, default=1800.0)
    ap.add_argument("--max-usd", type=float, default=1.00,
                    help="Per-task USD cap. DeepSeek-reasoner thinking tokens "
                         "need ~2x of v4-flash; 1.00 still well under V6's $1.85.")
    ap.add_argument("--max-recoveries", type=int, default=2)
    ap.add_argument("--same-file-edit-threshold", type=int, default=3)
    # Default OFF: smoke v5 showed the gate over-restricts on simple tasks
    # (agent gets stuck in reproduction loops instead of editing).
    # Use --require-reproduction-before-edit to re-enable.
    ap.add_argument("--require-reproduction-before-edit", action="store_true",
                    default=False)
    ap.add_argument("--allow-edit-before-repro", action="store_false",
                    dest="require_reproduction_before_edit")
    ap.add_argument("--cleanup-images", action="store_true", default=True)
    ap.add_argument("--no-cleanup-images", action="store_false",
                    dest="cleanup_images")
    ap.add_argument("--out-dir", default="runs/swebench_500_v7")
    ap.add_argument("--run-id", default="v7")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    _load_dotenv_if_present()
    reviewer_provider = args.reviewer_provider or args.provider
    reviewer_model = args.reviewer_model or args.model
    actor_key = _api_key_for(args.provider)
    reviewer_key = _api_key_for(reviewer_provider)
    actor_chat_fn = _build_chat_fn(args.provider, actor_key, args.model)
    reviewer_chat_fn = _build_chat_fn(reviewer_provider, reviewer_key, reviewer_model)

    if args.instance:
        instances = load_swebench_verified(instance_ids=[args.instance])
    else:
        instances = load_swebench_verified(n=args.n)

    print("[v7] benchmark: SWE-bench Verified")
    print(f"[v7] instances={len(instances)} seeds={args.seeds} workers={args.max_workers}")
    print(f"[v7] actor={args.provider}/{args.model}")
    print(f"[v7] reviewer={reviewer_provider}/{reviewer_model}")
    print(f"[v7] turns={args.max_turns}+{args.revision_turns} "
          f"usd_cap=${args.max_usd:.2f} repro_gate={args.require_reproduction_before_edit}")
    print(f"[v7] disk free: {disk_free_gb('/'):.1f} GB")

    budget = BudgetGuard(
        max_turns=args.max_turns,
        max_tokens_in=args.max_tokens_in,
        max_usd=args.max_usd,
        max_wall_s=args.max_wall_s,
    )
    recovery_policy = RecoveryPolicy(
        enabled=True,
        same_file_edit_threshold=args.same_file_edit_threshold,
        max_recoveries=args.max_recoveries,
    )

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    pairs = [(inst, seed) for inst in instances for seed in range(args.seeds)]
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {
            pool.submit(
                run_one_v7_task,
                inst=inst,
                seed=seed,
                actor_chat_fn=actor_chat_fn,
                reviewer_chat_fn=reviewer_chat_fn,
                model=args.model,
                reviewer_model=reviewer_model,
                cache_dir=cache_dir,
                max_turns=args.max_turns,
                revision_turns=args.revision_turns,
                max_tokens_per_turn=args.max_tokens_per_turn,
                budget_guard=budget,
                recovery_policy=recovery_policy,
                require_reproduction_before_edit=args.require_reproduction_before_edit,
                max_reviewer_tokens=args.max_reviewer_tokens,
                cleanup_image=args.cleanup_images,
            ): (inst.instance_id, seed)
            for inst, seed in pairs
        }
        for fut in as_completed(futs):
            iid, seed = futs[fut]
            rec = fut.result()
            records.append(rec)
            elapsed = time.perf_counter() - t0
            reviewer = rec.get("final_reviewer") or rec.get("reviewer") or {}
            approved = reviewer.get("approved")
            print(
                f"[v7] +{elapsed:6.0f}s ({len(records):>3d}/{len(pairs)}) "
                f"{iid:38s} seed{seed} turns={rec.get('n_turns', 0):>2d} "
                f"patch={rec.get('patch_len', 0):>5d}b "
                f"review={approved!s:<5s} stop={str(rec.get('stop_reason', ''))[:22]:<22s} "
                f"usd={rec.get('usd', 0):.3f}",
                flush=True,
            )

    sweep_wall = time.perf_counter() - t0
    total_usd = sum(float(r.get("usd", 0) or 0) for r in records)
    avg_turns = sum(int(r.get("n_turns", 0) or 0) for r in records) / max(1, len(records))
    avg_cost = total_usd / max(1, len(records))
    print(f"\n[v7] sweep wall: {sweep_wall:.0f}s ({sweep_wall/3600:.2f} h)")
    print(f"[v7] avg turns/task: {avg_turns:.1f}")
    print(f"[v7] total inference cost: ${total_usd:.3f}  avg=${avg_cost:.3f}/task")

    if args.skip_eval:
        out_path = out_dir / f"{args.run_id}.json"
        out_path.write_text(json.dumps({
            "run_id": args.run_id,
            "model": args.model,
            "reviewer_model": reviewer_model,
            "records": records,
            "sweep_wall_s": sweep_wall,
            "total_usd": total_usd,
            "avg_turns": avg_turns,
            "avg_cost_per_task": avg_cost,
            "evaluated": False,
            "goals": {
                "resolved_rate_min": 0.81,
                "resolved_rate_elite": 0.877,
                "pass_at_1_min": 0.65,
                "pass_at_1_elite": 0.72,
                "avg_turns_min": 18,
                "avg_turns_elite": 12,
                "avg_cost_min": 0.50,
                "avg_cost_elite": 0.25,
                "regression_rate_min": 0.03,
                "regression_rate_elite": 0.01,
            },
        }, indent=2))
        print(f"[v7] wrote {out_path} (eval skipped)")
        return

    per_seed_resolved: dict[int, dict[str, bool]] = {}
    per_seed_regressed: dict[int, dict[str, bool]] = {}
    per_seed_eval: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in range(args.seeds):
        seed_records = [r for r in records if r["seed"] == seed]
        run_id_full = f"{args.run_id}_seed{seed}"
        print(f"\n[v7] grading seed {seed} via swebench eval...")
        res = evaluate_v7_patches(
            records=seed_records,
            run_id=run_id_full,
            predictions_dir=out_dir / "predictions",
            model=args.model,
        )
        per_seed_eval[seed] = res
        per_seed_resolved[seed] = {
            iid: bool(v.get("resolved", False)) for iid, v in res.items()
        }
        per_seed_regressed[seed] = {
            iid: bool(v.get("regressed", False)) for iid, v in res.items()
        }
        n_resolved = sum(1 for v in per_seed_resolved[seed].values() if v)
        n_regressed = sum(1 for v in per_seed_regressed[seed].values() if v)
        pct = n_resolved / max(1, len(res))
        reg_pct = n_regressed / max(1, len(res))
        print(f"[v7] seed {seed}: {n_resolved}/{len(res)} resolved ({pct:.1%}); "
              f"regression={n_regressed}/{len(res)} ({reg_pct:.1%})")

    out_path = out_dir / f"{args.run_id}.json"
    out_path.write_text(json.dumps({
        "run_id": args.run_id,
        "model": args.model,
        "reviewer_model": reviewer_model,
        "records": records,
        "per_seed_resolved": {str(k): v for k, v in per_seed_resolved.items()},
        "per_seed_regressed": {str(k): v for k, v in per_seed_regressed.items()},
        "per_seed_eval": {str(k): v for k, v in per_seed_eval.items()},
        "sweep_wall_s": sweep_wall,
        "total_usd": total_usd,
        "avg_turns": avg_turns,
        "avg_cost_per_task": avg_cost,
        "evaluated": True,
        "goals": {
            "resolved_rate_min": 0.81,
            "resolved_rate_elite": 0.877,
            "pass_at_1_min": 0.65,
            "pass_at_1_elite": 0.72,
            "avg_turns_min": 18,
            "avg_turns_elite": 12,
            "avg_cost_min": 0.50,
            "avg_cost_elite": 0.25,
            "regression_rate_min": 0.03,
            "regression_rate_elite": 0.01,
        },
    }, indent=2))
    print(f"[v7] wrote {out_path}")


if __name__ == "__main__":
    main()

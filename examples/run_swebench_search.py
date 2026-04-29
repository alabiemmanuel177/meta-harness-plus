"""SWE-bench Verified pilot — per-seed halving search over harness shapes.

Runs Tier-1-style halving (9 → 5 → 3 → 1) across candidate harness
configurations on a fixed N-task SWE-bench-Verified subset.

Per-seed flow:
1. Sample 9 candidates from the (sys_prompt × temperature × include_hints)
   grid. Total grid is 18; we sample 9 unique configs deterministically
   from the seed.
2. Rung schedule: rung_idx → eval_size:
     [10, 20, 35, 50, 50] (capped at full N if smaller).
3. At each rung, for every surviving candidate × every instance in the
   rung's eval slice, generate one patch with Sonnet. Then write a
   single predictions JSONL with the union of patches and run the
   swebench harness once to grade them all in one batch (max_workers=8
   inside the harness).
4. Rank candidates by ``resolved`` count, keep top half.
5. Final rung leaves 1 candidate; that's the per-seed winner.

Across-seed aggregation happens in a separate script.

Speed:
- Per-task Sonnet call ~5-10s, 50+ Anthropic concurrency.
- Per-rung swebench eval batched, ~30s/instance with cached images.

Durability:
- Per-(seed, rung, candidate, instance) JSON caches under
  ``runs/swebench_pilot/cache/`` — power-cut resume just skips done
  cells.
- Cost log appends a USD line per call.

Usage:
    python3 examples/run_swebench_search.py --n 50 --seeds 5 --max-tokens 8192
    python3 examples/run_swebench_search.py --n 10 --seeds 1   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import (  # type: ignore
    _percentile, _load_dotenv_if_present, COST_LOG,
)
from examples.run_swebench_smoke import _record_anthropic_call, SONNET_4_6_PRICE
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.swebench_adapter import (
    CONCISE_PATCH_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    PLAN_THEN_PATCH_PROMPT,
    SWEBenchInstance,
    build_user_prompt,
    extract_patch,
    load_swebench_verified,
    run_swebench_eval,
    write_predictions,
)


SYS_PROMPTS = {
    0: DEFAULT_SYSTEM_PROMPT,
    1: PLAN_THEN_PATCH_PROMPT,
    2: CONCISE_PATCH_PROMPT,
}
TEMPERATURE_CHOICES = [0.0, 0.3, 0.7]
HINTS_CHOICES = [False, True]


@dataclass(frozen=True)
class Candidate:
    sys_prompt_id: int
    temperature: float
    include_hints: bool

    def label(self) -> str:
        h = "h1" if self.include_hints else "h0"
        return f"sp{self.sys_prompt_id}_t{self.temperature:.1f}_{h}"


def sample_candidates(seed: int, n: int = 9) -> list[Candidate]:
    """Deterministic sample of `n` unique candidates from the 3×3×2 grid."""
    import random
    grid = [
        Candidate(sys_prompt_id=sp, temperature=t, include_hints=h)
        for sp in [0, 1, 2]
        for t in TEMPERATURE_CHOICES
        for h in HINTS_CHOICES
    ]
    rng = random.Random(seed)
    rng.shuffle(grid)
    return grid[:min(n, len(grid))]


# --------- single-call patch generation ---------

def _patch_cache_path(cache_dir: Path, seed: int, cand: Candidate, inst: str) -> Path:
    return cache_dir / "patches" / f"seed{seed}_{cand.label()}_{inst}.json"


def _generate_patch_cached(
    *, client, inst: SWEBenchInstance, cand: Candidate,
    cache_dir: Path, seed: int, max_tokens: int, model: str,
) -> dict:
    p = _patch_cache_path(cache_dir, seed, cand, inst.instance_id)
    if p.exists():
        return json.loads(p.read_text())
    user = build_user_prompt(inst, include_hints=cand.include_hints)
    sys_prompt = SYS_PROMPTS[cand.sys_prompt_id]
    t0 = time.perf_counter()
    try:
        resp = client.complete(
            system=sys_prompt, user=user,
            max_tokens=max_tokens, temperature=cand.temperature,
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        rec = {
            "instance_id": inst.instance_id,
            "candidate": cand.label(),
            "config": asdict(cand),
            "seed": seed,
            "in_tokens": resp.input_tokens,
            "out_tokens": resp.output_tokens,
            "latency_ms": resp.latency_ms,
            "wall_ms": wall_ms,
            "patch": extract_patch(resp.text),
            "patch_extracted": bool(extract_patch(resp.text)),
            "model": model,
        }
    except Exception as e:
        rec = {
            "instance_id": inst.instance_id,
            "candidate": cand.label(),
            "config": asdict(cand),
            "seed": seed,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "patch": "", "patch_extracted": False,
            "in_tokens": 0, "out_tokens": 0, "latency_ms": 0, "wall_ms": 0,
            "model": model,
        }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2))
    if rec.get("in_tokens", 0):
        _record_anthropic_call(
            phase="swebench_pilot",
            label=f"seed{seed}_{cand.label()}_{inst.instance_id}",
            model=model,
            in_tokens=rec["in_tokens"], out_tokens=rec["out_tokens"],
            latency_ms=rec["wall_ms"],
        )
    return rec


# --------- rung evaluation ---------

def _eval_rung(
    *, client, candidates: list[Candidate], instances: list[SWEBenchInstance],
    cache_dir: Path, seed: int, max_tokens: int, model: str, max_workers: int,
    run_id: str, rung_idx: int,
) -> dict[str, int]:
    """For each (candidate, instance) generate a patch, then run the
    swebench eval ONCE on the union. Returns ``{candidate_label: n_resolved}``."""
    print(f"[search] rung {rung_idx}: {len(candidates)} candidates × "
          f"{len(instances)} instances = {len(candidates)*len(instances)} patches")

    # Step 1: generate all patches (parallel).
    pairs = [(ci, ii) for ci in range(len(candidates)) for ii in range(len(instances))]
    patches: dict[tuple[int, int], dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(
            _generate_patch_cached, client=client,
            inst=instances[ii], cand=candidates[ci],
            cache_dir=cache_dir, seed=seed,
            max_tokens=max_tokens, model=model,
        ): (ci, ii) for ci, ii in pairs}
        for fut in as_completed(futs):
            ci, ii = futs[fut]
            patches[(ci, ii)] = fut.result()
    print(f"[search] rung {rung_idx}: all patches generated")

    # Step 2: evaluate each candidate's patches separately (so we get
    # per-candidate resolved counts). Could batch all in one harness
    # call with a tagged predictions file but per-candidate is simpler.
    resolved_per_cand: dict[str, int] = {}
    for ci, cand in enumerate(candidates):
        cand_predictions: dict[str, str] = {}
        for ii, inst in enumerate(instances):
            cand_predictions[inst.instance_id] = patches[(ci, ii)].get("patch", "")
        run_id_full = f"{run_id}_seed{seed}_rung{rung_idx}_{cand.label()}"
        preds_path = cache_dir / "predictions" / f"{run_id_full}.jsonl"
        write_predictions(cand_predictions, model_name=model, out_path=preds_path)

        # Run eval. Skip if the per-candidate report.json already exists
        # (resume after power-cut).
        cache_key = cache_dir / "rung_results" / f"{run_id_full}.json"
        if cache_key.exists():
            results = json.loads(cache_key.read_text())
        else:
            t0 = time.perf_counter()
            try:
                eval_results = run_swebench_eval(
                    predictions_path=preds_path,
                    run_id=run_id_full,
                    instance_ids=[i.instance_id for i in instances],
                    max_workers=8,
                )
            except Exception as e:
                print(f"[search] rung {rung_idx} cand {cand.label()}: "
                      f"eval crashed — {type(e).__name__}: {e}")
                eval_results = {}
            wall = time.perf_counter() - t0
            results = {
                "candidate": cand.label(),
                "rung": rung_idx, "seed": seed,
                "n_instances": len(instances),
                "eval_wall_seconds": round(wall, 1),
                "per_instance": {
                    iid: {"resolved": bool(r.get("resolved", False))}
                    for iid, r in eval_results.items()
                },
                "n_resolved": sum(1 for r in eval_results.values()
                                  if r.get("resolved", False)),
            }
            cache_key.parent.mkdir(parents=True, exist_ok=True)
            cache_key.write_text(json.dumps(results, indent=2))
        resolved_per_cand[cand.label()] = results["n_resolved"]
        print(f"[search] rung {rung_idx}/{cand.label():22s}: "
              f"{results['n_resolved']}/{len(instances)} resolved")

    return resolved_per_cand


# --------- driver ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Number of SWE-bench instances.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--candidates-per-seed", type=int, default=9)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-workers", type=int, default=8,
                    help="Anthropic-side parallel patch-gen workers.")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--out-dir", default="runs/swebench_pilot")
    ap.add_argument("--run-id", default="pilot_v1")
    args = ap.parse_args()

    _load_dotenv_if_present()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY missing")

    print(f"[search] loading {args.n} SWE-bench Verified instances")
    instances = load_swebench_verified(n=args.n)
    print(f"[search] loaded {len(instances)} instances")

    # Halving rung schedule. Start with 10 instances at rung 0, expand
    # at each rung up to the full N.
    rung_eval_sizes = [
        min(10, args.n),
        min(20, args.n),
        min(35, args.n),
        min(50, args.n),
    ]

    # Halving keep schedule.
    keep_at_rung = [5, 3, 1, 1]

    cache_dir = Path(args.out_dir) / "cache"
    out_path = Path(args.out_dir) / f"{args.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = HTTPClient(
        api_url="https://api.anthropic.com/v1/messages",
        api_key=api_key, model=args.model, timeout_s=300.0,
        max_retries=4, retry_base_delay=2.0, retry_max_delay=30.0,
    )

    all_seed_results: list[dict] = []
    for seed in range(args.seeds):
        print(f"\n[search] === seed {seed} ===")
        candidates = sample_candidates(seed, n=args.candidates_per_seed)
        print(f"[search] sampled {len(candidates)} candidates: "
              f"{[c.label() for c in candidates]}")

        rung_log: list[dict] = []
        survivors = list(candidates)
        for rung_idx, eval_size in enumerate(rung_eval_sizes):
            rung_inst = instances[:eval_size]
            t0 = time.perf_counter()
            resolved = _eval_rung(
                client=raw, candidates=survivors, instances=rung_inst,
                cache_dir=cache_dir, seed=seed, max_tokens=args.max_tokens,
                model=args.model, max_workers=args.max_workers,
                run_id=args.run_id, rung_idx=rung_idx,
            )
            wall = time.perf_counter() - t0
            ranked = sorted(survivors, key=lambda c: -resolved[c.label()])
            n_keep = keep_at_rung[rung_idx] if rung_idx < len(keep_at_rung) else 1
            survivors_next = ranked[:n_keep]
            rung_log.append({
                "rung": rung_idx, "eval_size": eval_size,
                "wall_seconds": round(wall, 1),
                "ranked": [{"candidate": c.label(),
                            "config": asdict(c),
                            "n_resolved": resolved[c.label()],
                            "accuracy": resolved[c.label()] / max(1, eval_size)}
                           for c in ranked],
                "survivors": [c.label() for c in survivors_next],
            })
            print(f"[search] rung {rung_idx} done in {wall:.0f}s; "
                  f"survivors: {[c.label() for c in survivors_next]}")
            survivors = survivors_next

        seed_result = {
            "seed": seed,
            "rung_log": rung_log,
            "winner": rung_log[-1]["ranked"][0],
        }
        all_seed_results.append(seed_result)
        print(f"[search] seed {seed} winner: {seed_result['winner']['candidate']} "
              f"({seed_result['winner']['n_resolved']}/{rung_eval_sizes[-1]})")

    out_path.write_text(json.dumps({
        "run_id": args.run_id, "model": args.model,
        "n_instances": args.n, "seeds": args.seeds,
        "rung_eval_sizes": rung_eval_sizes,
        "results": all_seed_results,
    }, indent=2))
    print(f"\n[search] wrote {out_path}")


if __name__ == "__main__":
    main()

"""Per-instance comparison of parallel vs serial dev_100 retrieval.

Commit 11d hard-stopped early (15/100 instances done) once the speedup
signal stayed below 1.0× across both astropy and django blocks. The
correctness piece of 11d is still meaningful on the 15 completed
instances — that's the parallel-correctness sample size.

For each completed parallel instance, compare:
  1. The aggregated reranked top-10 file list.
  2. The per-strategy retrieval lists.
  3. The number of files indexed.

A perfect match (rerank top-10 equal as ordered lists) is the success
criterion. Differences in non-deterministic reranker tie-breaks are
allowed if rare (the spec says "MUST match exactly" — so any
difference fails the gate).
"""

from __future__ import annotations

import json
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SERIAL_DIR = (
    PROJECT_ROOT / "runs" / "v10_dev_100_retr_eval"
    / "checkpoints" / "embed_noshortlist_tb_bs256_rerank"
)
PARALLEL_DIR = (
    PROJECT_ROOT / "runs" / "v10_dev_100_retr_eval"
    / "checkpoints" / "embed_noshortlist_tb_bs256_rerank_w4"
)


def main() -> int:
    parallel_files = sorted(PARALLEL_DIR.glob("*.json"))
    if not parallel_files:
        print(f"[compare] no parallel checkpoints at {PARALLEL_DIR}")
        return 2

    print(f"[compare] {len(parallel_files)} parallel checkpoints to verify")
    print(f"[compare] serial baseline: {SERIAL_DIR.relative_to(PROJECT_ROOT)}")
    print()

    n_compared = 0
    n_match_top10 = 0
    n_match_per_strategy = 0
    diffs: list[dict] = []

    for p_path in parallel_files:
        iid = p_path.stem
        s_path = SERIAL_DIR / p_path.name
        if not s_path.exists():
            print(f"[compare] {iid}: NO SERIAL BASELINE — skipping")
            continue
        n_compared += 1

        p = json.loads(p_path.read_text())
        s = json.loads(s_path.read_text())

        p_top10 = [r["file_path"] for r in (p.get("reranked_files") or [])][:10]
        s_top10 = [r["file_path"] for r in (s.get("reranked_files") or [])][:10]

        top10_match = (p_top10 == s_top10)
        if top10_match:
            n_match_top10 += 1

        # Per-strategy: compare each strategy's ranked list.
        p_per = p.get("per_strategy", {})
        s_per = s.get("per_strategy", {})
        per_strategy_match = True
        per_strategy_diffs = {}
        for sk in sorted(set(p_per.keys()) | set(s_per.keys())):
            p_list = p_per.get(sk, [])
            s_list = s_per.get(sk, [])
            if p_list != s_list:
                per_strategy_match = False
                # Find first divergence index
                first_diff = None
                for i, (pp, ss) in enumerate(zip(p_list, s_list)):
                    if pp != ss:
                        first_diff = i
                        break
                if first_diff is None:
                    first_diff = min(len(p_list), len(s_list))
                per_strategy_diffs[sk] = {
                    "first_diff_at": first_diff,
                    "p_len": len(p_list),
                    "s_len": len(s_list),
                }
        if per_strategy_match:
            n_match_per_strategy += 1

        n_files_match = (p.get("n_files_indexed") == s.get("n_files_indexed"))

        status = "MATCH" if top10_match and per_strategy_match else "DIFF"
        print(
            f"[compare] {iid:<48s}  top10={'==' if top10_match else '!='}  "
            f"per_strategy={'==' if per_strategy_match else '!='}  "
            f"n_files={'==' if n_files_match else '!='}  {status}"
        )
        if not top10_match or not per_strategy_match:
            diffs.append({
                "iid": iid,
                "top10_match": top10_match,
                "p_top10": p_top10,
                "s_top10": s_top10,
                "per_strategy_diffs": per_strategy_diffs,
                "n_files_p": p.get("n_files_indexed"),
                "n_files_s": s.get("n_files_indexed"),
            })

    print()
    print(f"[compare] compared:               {n_compared}")
    print(f"[compare] top-10 match:           {n_match_top10}/{n_compared}")
    print(f"[compare] per-strategy match:     {n_match_per_strategy}/{n_compared}")

    if diffs:
        print()
        print("=" * 60)
        print(f"[compare] {len(diffs)} INSTANCE(S) WITH DIFFERENCES — detail below")
        print("=" * 60)
        for d in diffs:
            print()
            print(f"--- {d['iid']} ---")
            if not d["top10_match"]:
                print("  top-10 lists differ:")
                for i, (p, s) in enumerate(zip(d["p_top10"], d["s_top10"])):
                    marker = "==" if p == s else "!="
                    print(f"    {i+1:>2}.  {marker}  parallel={p}")
                    if p != s:
                        print(f"            serial=  {s}")
            if d["per_strategy_diffs"]:
                print("  per-strategy lists differ:")
                for sk, info in d["per_strategy_diffs"].items():
                    print(
                        f"    {sk}: first_diff_at={info['first_diff_at']}  "
                        f"p_len={info['p_len']}  s_len={info['s_len']}"
                    )

    if n_match_top10 == n_compared and n_match_per_strategy == n_compared:
        print()
        print("[compare] PARALLEL = SERIAL (correctness verified on partial sample)")
        return 0
    print()
    print("[compare] PARALLEL != SERIAL (some instances differ)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure rerank top-K variance across multiple runs of the same model.

Commit 11d showed that DeepSeek-chat at temperature=0 produces
different rerank top-10 orderings on 6/15 instances (40%) between
two runs with identical inputs (the parallel and serial runs had
identical upstream signals). The question this script answers:

  When the rerank ordering differs, does the top-K HIT/MISS verdict
  also differ?

Two top-10 lists can be reordered yet both contain the gold file at
some position 1-10 — the top-10 hit count would be identical even if
the orderings are not. Likewise, top-1 is sensitive to position
exactly because it cares which file is rank #1.

For each pair of runs:
  - Compute top-1 / top-5 / top-10 hits per the production matcher
    (harness.eval._normalize_path equality with the gold patch's
    touched files).
  - Per-instance: did top-1 flip between hit and miss? top-5? top-10?
  - Report mean ± stddev across the run set.

Usage:

    PYTHONPATH=. .venv/bin/python3 scripts/measure_rerank_variance.py \\
        --split splits/dev_100.json \\
        --rerank-checkpoints \\
            runs/v10_dev_100_retr_eval/checkpoints/embed_noshortlist_tb_bs256_rerank \\
            runs/v10_dev_100_retr_eval/checkpoints/embed_noshortlist_tb_bs256_rerank_det_run2 \\
            runs/v10_dev_100_retr_eval/checkpoints/embed_noshortlist_tb_bs256_rerank_w4 \\
        --out docs/audits/rerank_variance_dev100.md
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _gold_top_k_hit(retrieved: list[str], gold: set[str], k: int) -> bool:
    """Production matcher semantic: a hit if any of top-K (after
    normalization) is in the gold set."""
    from harness.eval import _normalize_path
    g_norm = {_normalize_path(g) for g in gold}
    for r in retrieved[:k]:
        if _normalize_path(r) in g_norm:
            return True
    return False


def _load_run(rerank_dir: pathlib.Path, instance_ids: list[str]) -> dict[str, list[str]]:
    """Return {iid: [reranked file paths in rank order]}.

    Skips iids without checkpoints (e.g. partial runs).
    """
    out: dict[str, list[str]] = {}
    for iid in instance_ids:
        ckpt = rerank_dir / f"{iid}.json"
        if not ckpt.exists():
            continue
        d = json.loads(ckpt.read_text())
        rr = d.get("reranked_files") or []
        out[iid] = [r["file_path"] for r in rr]
    return out


def _stats(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument(
        "--rerank-checkpoints", nargs="+", required=True,
        help="Two or more rerank checkpoint dirs to compare.",
    )
    ap.add_argument(
        "--labels", nargs="*", default=None,
        help="Optional labels for each run, in --rerank-checkpoints order. "
             "Defaults to the dir basename.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.rerank_checkpoints) < 2:
        print("[variance] need at least 2 rerank checkpoint dirs to compare")
        return 2

    split_path = pathlib.Path(args.split)
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    dev = json.loads(split_path.read_text())
    instance_ids = [e["instance_id"] for e in dev["instances"]]

    # Resolve labels.
    if args.labels:
        if len(args.labels) != len(args.rerank_checkpoints):
            print("[variance] labels must match rerank-checkpoints in count")
            return 2
        labels = list(args.labels)
    else:
        labels = [pathlib.Path(p).name for p in args.rerank_checkpoints]

    # Load each run.
    runs: list[tuple[str, dict[str, list[str]]]] = []
    for ck, label in zip(args.rerank_checkpoints, labels):
        ck_path = pathlib.Path(ck)
        if not ck_path.is_absolute():
            ck_path = PROJECT_ROOT / ck_path
        if not ck_path.is_dir():
            print(f"[variance] missing dir: {ck_path}")
            return 2
        runs.append((label, _load_run(ck_path, instance_ids)))

    # Gold metadata for the same instances.
    print(f"[variance] loading gold metadata for {len(instance_ids)} instances")
    from harness.eval import load_eval_metadata
    meta = load_eval_metadata(instance_ids)

    # Per-run top-K hit counts.
    K_VALUES = (1, 5, 10)
    per_run_hits: dict[str, dict[int, int]] = {}
    per_run_n: dict[str, int] = {}
    for label, run in runs:
        hits = {k: 0 for k in K_VALUES}
        n = 0
        for iid, retrieved in run.items():
            gold = meta.gold_touched_files.get(iid, set())
            if not gold:
                continue
            n += 1
            for k in K_VALUES:
                if _gold_top_k_hit(retrieved, gold, k):
                    hits[k] += 1
        per_run_hits[label] = hits
        per_run_n[label] = n
        print(
            f"[variance] {label:<60s}  n={n}  "
            f"top1={hits[1]:>3d}  top5={hits[5]:>3d}  top10={hits[10]:>3d}"
        )

    # Pair-wise overlap on the COMMON instances (those present in EVERY run).
    common = set(instance_ids)
    for _, run in runs:
        common &= set(run.keys())
    print(f"[variance] {len(common)} instances common to all {len(runs)} runs")

    # Per-instance flip analysis on the common set.
    flip_top1: list[str] = []
    flip_top5: list[str] = []
    flip_top10: list[str] = []
    for iid in sorted(common):
        gold = meta.gold_touched_files.get(iid, set())
        if not gold:
            continue
        hits_top1: list[bool] = []
        hits_top5: list[bool] = []
        hits_top10: list[bool] = []
        for _, run in runs:
            retrieved = run[iid]
            hits_top1.append(_gold_top_k_hit(retrieved, gold, 1))
            hits_top5.append(_gold_top_k_hit(retrieved, gold, 5))
            hits_top10.append(_gold_top_k_hit(retrieved, gold, 10))
        if len(set(hits_top1)) > 1:
            flip_top1.append(iid)
        if len(set(hits_top5)) > 1:
            flip_top5.append(iid)
        if len(set(hits_top10)) > 1:
            flip_top10.append(iid)

    # Recall stats across runs (mean + stddev) — only meaningful if all
    # runs cover the full split. Skip partial runs.
    full_run_recalls: dict[int, list[float]] = {k: [] for k in K_VALUES}
    full_labels: list[str] = []
    for label, run in runs:
        if len(run) >= len(instance_ids):
            for k in K_VALUES:
                # n_with_gold (those we counted)
                full_run_recalls[k].append(per_run_hits[label][k] / per_run_n[label])
            full_labels.append(label)

    # Build markdown report.
    out: list[str] = []
    out.append("# Rerank top-K variance audit (commit 12a)")
    out.append("")
    out.append(
        "Generated by `scripts/measure_rerank_variance.py`. Compares "
        "rerank top-1/5/10 hit counts across multiple runs of the same "
        "rerank model on the same dev_100 retrieval inputs. Designed to "
        "quantify the API-level non-determinism that commit 11d "
        "uncovered: at temperature=0, DeepSeek-chat returns different "
        "top-10 orderings 6/15 of the time."
    )
    out.append("")
    out.append("## Per-run summary")
    out.append("")
    out.append("| Label | n covered | top-1 | top-5 | top-10 |")
    out.append("|---|---|---|---|---|")
    for label, run in runs:
        n = per_run_n[label]
        h = per_run_hits[label]
        if n == 0:
            out.append(f"| `{label}` | 0 | n/a | n/a | n/a |")
            continue
        out.append(
            f"| `{label}` | {n} | "
            f"{h[1]}/{n} ({100.0*h[1]/n:.1f}%) | "
            f"{h[5]}/{n} ({100.0*h[5]/n:.1f}%) | "
            f"{h[10]}/{n} ({100.0*h[10]/n:.1f}%) |"
        )
    out.append("")

    if len(full_labels) >= 2:
        out.append("## Full-coverage variance (only runs covering all 100 instances)")
        out.append("")
        out.append(f"Runs included: {', '.join('`' + l + '`' for l in full_labels)}")
        out.append("")
        out.append("| Metric | Mean | Stddev | Min | Max | Range |")
        out.append("|---|---|---|---|---|---|")
        for k in K_VALUES:
            vals = full_run_recalls[k]
            mean, std = _stats(vals)
            mn = min(vals) if vals else 0
            mx = max(vals) if vals else 0
            out.append(
                f"| top-{k} | {100*mean:.1f}% | {100*std:.2f}pp | "
                f"{100*mn:.1f}% | {100*mx:.1f}% | {100*(mx-mn):.1f}pp |"
            )
        out.append("")

    out.append(f"## Per-instance flip analysis (intersection of all runs, n={len(common)})")
    out.append("")
    out.append(
        "An instance \"flips\" at K when the top-K hit/miss verdict "
        "differs between runs. A non-zero flip count is direct evidence "
        "that the rerank API non-determinism affects the headline "
        "score, not just internal ordering."
    )
    out.append("")
    out.append("| Metric | # flipped | % of common |")
    out.append("|---|---|---|")
    for k_label, flips in [("top-1", flip_top1), ("top-5", flip_top5), ("top-10", flip_top10)]:
        n = max(1, len(common))
        out.append(f"| {k_label} | {len(flips)} | {100*len(flips)/n:.1f}% |")
    out.append("")

    # Pairwise flip analysis for every (run_a, run_b) combination on
    # the union of their instance sets — gives a complete picture
    # without being restricted to the all-runs intersection.
    full_pair_flips: dict[tuple[str, str], dict[str, list[str]]] = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            la, ra = runs[i]
            lb, rb = runs[j]
            common_ab = set(ra.keys()) & set(rb.keys())
            ft1: list[str] = []
            ft5: list[str] = []
            ft10: list[str] = []
            for iid in sorted(common_ab):
                gold = meta.gold_touched_files.get(iid, set())
                if not gold:
                    continue
                ha1 = _gold_top_k_hit(ra[iid], gold, 1)
                hb1 = _gold_top_k_hit(rb[iid], gold, 1)
                ha5 = _gold_top_k_hit(ra[iid], gold, 5)
                hb5 = _gold_top_k_hit(rb[iid], gold, 5)
                ha10 = _gold_top_k_hit(ra[iid], gold, 10)
                hb10 = _gold_top_k_hit(rb[iid], gold, 10)
                if ha1 != hb1: ft1.append(iid)
                if ha5 != hb5: ft5.append(iid)
                if ha10 != hb10: ft10.append(iid)
            full_pair_flips[(la, lb)] = {
                "n_compared": len(common_ab),
                "top1": ft1, "top5": ft5, "top10": ft10,
            }

    out.append("## Pairwise flip analysis (each pair on its own intersection)")
    out.append("")
    out.append(
        "More informative than the all-runs intersection above when "
        "runs cover different subsets — each pair is scored on every "
        "instance both runs cover."
    )
    out.append("")
    out.append("| Run A | Run B | n compared | top-1 flips | top-5 flips | top-10 flips |")
    out.append("|---|---|---|---|---|---|")
    for (la, lb), info in full_pair_flips.items():
        out.append(
            f"| `{la}` | `{lb}` | {info['n_compared']} | "
            f"{len(info['top1'])} | {len(info['top5'])} | {len(info['top10'])} |"
        )
    out.append("")
    # If any pair has flips, list them
    any_pair_flips = any(
        info["top1"] or info["top5"] or info["top10"]
        for info in full_pair_flips.values()
    )
    if any_pair_flips:
        out.append("### Per-pair flip details")
        out.append("")
        for (la, lb), info in full_pair_flips.items():
            if info["top1"] or info["top5"] or info["top10"]:
                out.append(f"**`{la}` vs `{lb}`** ({info['n_compared']} compared):")
                if info["top1"]:
                    out.append(f"  - top-1 flips ({len(info['top1'])}): " + ", ".join(f"`{i}`" for i in info["top1"]))
                if info["top5"]:
                    out.append(f"  - top-5 flips ({len(info['top5'])}): " + ", ".join(f"`{i}`" for i in info["top5"]))
                if info["top10"]:
                    out.append(f"  - top-10 flips ({len(info['top10'])}): " + ", ".join(f"`{i}`" for i in info["top10"]))
                out.append("")

    if flip_top1:
        out.append("### top-1 flips")
        out.append("")
        out.append("| Instance | per-run top-1 hit |")
        out.append("|---|---|")
        for iid in flip_top1:
            gold = meta.gold_touched_files.get(iid, set())
            cells = []
            for label, run in runs:
                retrieved = run[iid]
                hit = _gold_top_k_hit(retrieved, gold, 1)
                cells.append(f"{label}: {'✓' if hit else '✗'}")
            out.append(f"| `{iid}` | " + ", ".join(cells) + " |")
        out.append("")

    if flip_top5 or flip_top10:
        out.append("### top-5 / top-10 flips")
        out.append("")
        out.append("| Instance | top-5 flip | top-10 flip |")
        out.append("|---|---|---|")
        for iid in sorted(set(flip_top5) | set(flip_top10)):
            gold = meta.gold_touched_files.get(iid, set())
            top5_per_run = []
            top10_per_run = []
            for label, run in runs:
                retrieved = run[iid]
                top5_per_run.append((label, _gold_top_k_hit(retrieved, gold, 5)))
                top10_per_run.append((label, _gold_top_k_hit(retrieved, gold, 10)))
            top5_str = ", ".join(f"{l}: {'✓' if h else '✗'}" for l, h in top5_per_run)
            top10_str = ", ".join(f"{l}: {'✓' if h else '✗'}" for l, h in top10_per_run)
            out.append(f"| `{iid}` | {top5_str} | {top10_str} |")
        out.append("")

    # Total flips across all pairs (the meaningful signal — covers
    # everything, not just the all-runs intersection).
    total_pair_top1 = sum(len(p["top1"]) for p in full_pair_flips.values())
    total_pair_top5 = sum(len(p["top5"]) for p in full_pair_flips.values())
    total_pair_top10 = sum(len(p["top10"]) for p in full_pair_flips.values())
    any_flips = total_pair_top1 + total_pair_top5 + total_pair_top10 > 0

    out.append("## Verdict & implications")
    out.append("")
    if not any_flips:
        out.append(
            "**Zero hit/miss flips on every pair, every K.** Rerank "
            "ordering DOES vary between runs (the prior 11d audit "
            "found ~40% of instances had different top-10 orderings), "
            "but the reordering is purely WITHIN the top-K window — "
            "files don't cross the K boundary. Top-1, top-5, and "
            "top-10 verdicts are bit-stable across reruns."
        )
        out.append("")
        out.append(
            "**The dev_100 headline (84/97/98) is reproducible to ±0pp "
            "across DeepSeek-chat reruns at temperature=0.** The "
            "earlier worry from 11d (\"the published numbers carry "
            "implicit ±1-2pp\") was overstated: ordering jitter does "
            "NOT translate to recall jitter at K=1, 5, or 10."
        )
    else:
        # Compute worst pairwise flip rate
        worst_rate = 0.0
        for (la, lb), info in full_pair_flips.items():
            n = max(1, info["n_compared"])
            for k_field in ("top1", "top5", "top10"):
                rate = 100.0 * len(info[k_field]) / n
                if rate > worst_rate:
                    worst_rate = rate
        out.append(
            f"Hit/miss flips DO occur. Worst pairwise flip rate: "
            f"{worst_rate:.1f}%. The dev_100 headline numbers carry "
            f"an implicit band of ±{worst_rate:.1f}pp at temperature=0."
        )
    out.append("")
    out.append("## Recommendations for test_500")
    out.append("")
    if not any_flips:
        out.append(
            "  - The headline run can be a single run; numbers are "
            "reproducible.\n"
            "  - The Sonnet ablation (commit 12b) is now a QUALITY "
            "question, not a variance one — does Sonnet give better "
            "top-1 than DeepSeek's 84%? Worth running for the quality "
            "answer, but not required to fix a variance problem."
        )
    else:
        out.append(
            "  - Either run test_500 multiple times and report "
            "mean ± stddev, OR\n"
            "  - Switch to a reranker model with tighter determinism "
            "guarantees (commit 12b ablates Sonnet 4.5)."
        )
    out.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    try:
        print(f"[variance] wrote {out_path.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"[variance] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

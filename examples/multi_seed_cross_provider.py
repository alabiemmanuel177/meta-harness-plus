"""Multi-seed cross-provider bakeoff aggregator.

Reads per-seed bakeoff_summary.json files produced by rag_vs_mh_bakeoff.py
across multiple --run-name dirs, computes bootstrap confidence intervals
on (MH++ peak accuracy - RAG accuracy) per provider, and prints a
comparative table.

Usage::

    python3 examples/multi_seed_cross_provider.py \\
        --runs runs/openai_news_hard_50_seed0 runs/openai_news_hard_50_seed1 ... \\
        --label "OpenAI gpt-4.1-nano"

Or pass a glob pattern::

    python3 examples/multi_seed_cross_provider.py \\
        --pattern "runs/openai_news_hard_50_seed*"

Bootstrap CIs use the tier-1 statistics module — same code path as the
``MultiSeedRunner``, just applied to summary files instead of running
the searches inline.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.statistics import (
    bootstrap_ci,
    paired_bootstrap_diff,
    paired_t_test,
)


def _load_summary(run_dir: Path) -> dict | None:
    # bakeoff_summary.json is per-model; pick whichever single one exists.
    for sub in run_dir.iterdir():
        summary = sub / "bakeoff_summary.json"
        if summary.exists():
            return json.loads(summary.read_text())
    return None


def aggregate(run_dirs: list[Path], label: str) -> dict:
    summaries = []
    for d in run_dirs:
        s = _load_summary(d)
        if s is None:
            print(f"  WARN: no summary in {d}", flush=True)
            continue
        summaries.append((d.name, s))

    if not summaries:
        return {"label": label, "n_seeds": 0}

    rag_accs: list[float] = []
    mh_peak_accs: list[float] = []
    bare_accs: list[float] = []
    rag_tokens: list[float] = []
    mh_peak_tokens: list[float] = []
    n_dominate: list[int] = []
    n_match_cheaper: list[int] = []

    for name, s in summaries:
        rag_accs.append(s["rag_baseline"]["accuracy"])
        rag_tokens.append(s["rag_baseline"]["tokens"])
        bare_accs.append(s["bare_baseline"]["accuracy"])
        # MH++ peak: highest-accuracy non-seed entry on the frontier.
        peaks = [r for r in s["frontier"]
                 if r.get("label") in ("discovered",)]
        if peaks:
            best = max(peaks, key=lambda r: r["accuracy"])
            mh_peak_accs.append(best["accuracy"])
            mh_peak_tokens.append(best["tokens"])
        else:
            # No discovered entries — frontier just has seeds. Use
            # whichever seed has highest accuracy as a degenerate "peak".
            best = max(s["frontier"], key=lambda r: r["accuracy"])
            mh_peak_accs.append(best["accuracy"])
            mh_peak_tokens.append(best["tokens"])
        n_dominate.append(s["n_discovered_dominates_rag"])
        n_match_cheaper.append(s["n_discovered_ties_rag_acc_cheaper"])

    diff_ci = paired_bootstrap_diff(mh_peak_accs, rag_accs, n_resamples=2000)
    mh_acc_ci = bootstrap_ci(mh_peak_accs, n_resamples=2000)
    rag_acc_ci = bootstrap_ci(rag_accs, n_resamples=2000)
    t_test = paired_t_test(mh_peak_accs, rag_accs)

    # Per-class delta aggregation: collect per-class accuracies for the
    # MH++ peak harness and the RAG baseline across all seeds, compute
    # per-class mean delta. Lets us answer "which class did MH++ help
    # most?" — granular failure-mode signal beyond aggregate accuracy.
    mh_per_class: dict[str, list[float]] = {}
    rag_per_class: dict[str, list[float]] = {}
    for name, s in summaries:
        # MH++ peak harness's per-class accuracy.
        peaks = [r for r in s["frontier"] if r.get("label") in ("discovered",)]
        peak_row = max(peaks, key=lambda r: r["accuracy"]) if peaks \
                   else max(s["frontier"], key=lambda r: r["accuracy"])
        for kv in peak_row.get("per_class_accuracy", []) or []:
            if isinstance(kv, (list, tuple)) and len(kv) == 2:
                klass, acc = kv
                mh_per_class.setdefault(klass, []).append(float(acc))
        # RAG (cand_0002 is the seeded RAG row).
        rag_row = next((r for r in s["frontier"] if r.get("label") == "[RAG]"), None)
        if rag_row:
            for kv in rag_row.get("per_class_accuracy", []) or []:
                if isinstance(kv, (list, tuple)) and len(kv) == 2:
                    klass, acc = kv
                    rag_per_class.setdefault(klass, []).append(float(acc))

    per_class_summary = {}
    for klass in sorted(set(mh_per_class) | set(rag_per_class)):
        mh_vals = mh_per_class.get(klass, [])
        rag_vals = rag_per_class.get(klass, [])
        per_class_summary[klass] = {
            "mh_mean": sum(mh_vals) / len(mh_vals) if mh_vals else 0.0,
            "rag_mean": sum(rag_vals) / len(rag_vals) if rag_vals else 0.0,
            "delta_mean": (
                (sum(mh_vals) / len(mh_vals)) - (sum(rag_vals) / len(rag_vals))
                if mh_vals and rag_vals else 0.0
            ),
            "n_seeds": min(len(mh_vals), len(rag_vals)),
        }

    return {
        "label": label,
        "n_seeds": len(summaries),
        "per_seed_bare_acc": bare_accs,
        "per_seed_rag_acc": rag_accs,
        "per_seed_mh_peak_acc": mh_peak_accs,
        "per_seed_rag_tokens": rag_tokens,
        "per_seed_mh_peak_tokens": mh_peak_tokens,
        "per_seed_n_dominate": n_dominate,
        "per_seed_n_match_cheaper": n_match_cheaper,
        "ci": {
            "mh_minus_rag_acc": {
                "mean": diff_ci.mean, "low": diff_ci.low,
                "high": diff_ci.high, "confidence": diff_ci.confidence,
            },
            "mh_peak_acc": {
                "mean": mh_acc_ci.mean, "low": mh_acc_ci.low,
                "high": mh_acc_ci.high, "confidence": mh_acc_ci.confidence,
            },
            "rag_acc": {
                "mean": rag_acc_ci.mean, "low": rag_acc_ci.low,
                "high": rag_acc_ci.high, "confidence": rag_acc_ci.confidence,
            },
        },
        "paired_t_test": {
            "n": t_test.n,
            "mean_diff": t_test.mean_diff,
            "std_diff": t_test.std_diff,
            "t_statistic": t_test.t_statistic,
            "p_value_two_sided": t_test.p_value_two_sided,
            "cohens_d": t_test.cohens_d,
        },
        "per_class_delta": per_class_summary,
    }


def print_report(agg: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {agg['label']}  (n_seeds={agg['n_seeds']})")
    print(f"{'=' * 72}")
    if agg["n_seeds"] == 0:
        print("  (no seed data found)")
        return
    print(f"  per-seed BARE acc:    {agg['per_seed_bare_acc']}")
    print(f"  per-seed RAG  acc:    {agg['per_seed_rag_acc']}")
    print(f"  per-seed MH++ acc:    {agg['per_seed_mh_peak_acc']}")
    print(f"  per-seed RAG  tokens: {[round(x, 1) for x in agg['per_seed_rag_tokens']]}")
    print(f"  per-seed MH++ tokens: {[round(x, 1) for x in agg['per_seed_mh_peak_tokens']]}")
    print(f"  dominates_rag count:  {agg['per_seed_n_dominate']}")
    print(f"  match_rag_cheaper:    {agg['per_seed_n_match_cheaper']}")
    print()
    ci = agg["ci"]
    print(f"  RAG  acc CI:          {ci['rag_acc']['mean']:.3f} "
          f"[{ci['rag_acc']['low']:.3f}, {ci['rag_acc']['high']:.3f}] "
          f"({ci['rag_acc']['confidence']:.0%})")
    print(f"  MH++ acc CI:          {ci['mh_peak_acc']['mean']:.3f} "
          f"[{ci['mh_peak_acc']['low']:.3f}, {ci['mh_peak_acc']['high']:.3f}] "
          f"({ci['mh_peak_acc']['confidence']:.0%})")
    diff = ci["mh_minus_rag_acc"]
    print(f"  Δ (MH++ - RAG) CI:    {diff['mean']:+.3f} "
          f"[{diff['low']:+.3f}, {diff['high']:+.3f}] "
          f"({diff['confidence']:.0%})")
    excludes_zero = (diff["low"] > 0.0) or (diff["high"] < 0.0)
    if excludes_zero:
        print(f"  ==> CI excludes zero — significant accuracy difference")
    else:
        print(f"  ==> CI includes zero — accuracy difference not significant at this seed count")

    # Paired t-test (parametric complement to the bootstrap CI).
    t = agg.get("paired_t_test", {})
    if t and t.get("n", 0) > 1:
        # Cohen's d interpretation: |d| 0.2=small, 0.5=medium, 0.8=large.
        d_abs = abs(t["cohens_d"])
        if d_abs >= 0.8:
            d_label = "large"
        elif d_abs >= 0.5:
            d_label = "medium"
        elif d_abs >= 0.2:
            d_label = "small"
        else:
            d_label = "negligible"
        print()
        print(f"  Paired t-test (n={t['n']}):")
        print(f"    t = {t['t_statistic']:+.3f}  "
              f"p (two-sided) = {t['p_value_two_sided']:.4f}")
        print(f"    Cohen's d = {t['cohens_d']:+.3f} ({d_label} effect)")

    # Per-class breakdown.
    pc = agg.get("per_class_delta", {})
    if pc:
        print()
        print(f"  Per-class Δ (MH++ - RAG, mean across seeds):")
        # Sort by largest |delta| first.
        sorted_pc = sorted(pc.items(),
                           key=lambda kv: -abs(kv[1].get("delta_mean", 0)))
        for klass, info in sorted_pc:
            arrow = "+" if info["delta_mean"] >= 0 else ""
            print(f"    {klass:25s}  RAG={info['rag_mean']:.2f}  "
                  f"MH++={info['mh_mean']:.2f}  Δ={arrow}{info['delta_mean']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--pattern", default=None,
                    help="glob pattern for run dirs (e.g. 'runs/openai_*_seed*')")
    ap.add_argument("--label", default="aggregate", help="label for the report")
    ap.add_argument("--output", default=None,
                    help="optional JSON output path for the aggregate")
    args = ap.parse_args()

    paths: list[Path] = [Path(r) for r in args.runs]
    if args.pattern:
        paths.extend(Path(p) for p in sorted(glob.glob(args.pattern))
                     if Path(p).is_dir())
    if not paths:
        raise SystemExit("no run dirs supplied (--runs or --pattern)")

    agg = aggregate(paths, args.label)
    print_report(agg)
    if args.output:
        Path(args.output).write_text(json.dumps(agg, indent=2))
        print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()

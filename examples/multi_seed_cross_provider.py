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

from meta_harness_plus.statistics import bootstrap_ci, paired_bootstrap_diff


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

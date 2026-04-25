"""Aggregate the full-scale ablation experiment.

Reads `runs/ablation_<cond>_seed<i>/*/bakeoff_summary.json` for each
of (full, no-c1, no-c2, no-c3) × 5 seeds. Prints per-condition mean
accuracy + tokens + frontier-size, paired-bootstrap Δ vs full MH++,
and a paired t-test for each ablation against full.

This is the version of run_ablation_study.py used at full LLM scale —
the toy-task version only validated the wiring; this answers the
reviewer question "did each contribution help on a real task with a
real LLM?" with multi-seed CI evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_harness_plus.statistics import (
    bootstrap_ci, paired_bootstrap_diff, paired_t_test,
)


CONDITIONS = ["full", "no-c1", "no-c2", "no-c3"]


def _load_seed(cond: str, seed: int) -> dict | None:
    d = Path(f"runs/ablation_{cond}_seed{seed}")
    if not d.is_dir():
        return None
    for sub in d.iterdir():
        bs = sub / "bakeoff_summary.json"
        if bs.exists():
            return json.loads(bs.read_text())
    return None


def _peak(s: dict) -> tuple[float, float, float]:
    """Returns (best_accuracy, tokens@best, latency@best) of *discovered* candidates.

    Falls back to whichever frontier point has highest accuracy if no
    discovered ones exist.
    """
    if not s:
        return (0.0, 0.0, 0.0)
    discovered = [r for r in s["frontier"] if r.get("label") in ("discovered",)]
    if not discovered:
        # No discovered survived; return RAG seed acc to avoid zero-skewed stats.
        rag = next((r for r in s["frontier"] if r.get("label") == "[RAG]"), None)
        if rag:
            return (rag["accuracy"], rag["tokens"], rag["latency_ms"])
        return (s["frontier"][0]["accuracy"], s["frontier"][0]["tokens"],
                s["frontier"][0]["latency_ms"])
    best = max(discovered, key=lambda r: r["accuracy"])
    return (best["accuracy"], best["tokens"], best["latency_ms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--task", default="news_hard_50")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    by_cond: dict[str, list[tuple[float, float, float]]] = {}
    by_cond_full: dict[str, list[dict]] = {}
    for cond in CONDITIONS:
        peaks = []
        summaries = []
        for seed in args.seeds:
            s = _load_seed(cond, seed)
            if s is None:
                print(f"  WARN: missing {cond}/seed{seed}", flush=True)
                continue
            peaks.append(_peak(s))
            summaries.append(s)
        by_cond[cond] = peaks
        by_cond_full[cond] = summaries

    print(f"\n{'=' * 76}")
    print(f"  Ablation full-scale results — task={args.task}, seeds={args.seeds}")
    print(f"{'=' * 76}")
    print(f"{'Condition':<12} {'mean acc':>10} {'mean tok':>10} {'mean lat':>10} {'n':>4}")
    print("-" * 56)
    for cond, peaks in by_cond.items():
        n = len(peaks)
        if n == 0:
            print(f"{cond:<12} {'—':>10} {'—':>10} {'—':>10} {n:>4}")
            continue
        mean_acc = sum(p[0] for p in peaks) / n
        mean_tok = sum(p[1] for p in peaks) / n
        mean_lat = sum(p[2] for p in peaks) / n
        print(f"{cond:<12} {mean_acc:>10.3f} {mean_tok:>10.1f} {mean_lat:>10.1f} {n:>4}")

    full_accs = [p[0] for p in by_cond.get("full", [])]
    print()
    print("Δ (ablation - full MH++) on best accuracy [paired CI + t-test]:")
    deltas: dict[str, dict] = {}
    if full_accs:
        for cond in CONDITIONS:
            if cond == "full" or not by_cond.get(cond):
                continue
            ab_accs = [p[0] for p in by_cond[cond]]
            n = min(len(ab_accs), len(full_accs))
            if n == 0:
                continue
            ab = ab_accs[:n]
            full = full_accs[:n]
            diff = paired_bootstrap_diff(ab, full, n_resamples=2000)
            t = paired_t_test(ab, full)
            sig = (diff.low > 0) or (diff.high < 0)
            print(f"  {cond:<10}  Δ_mean={diff.mean:+.3f}  CI=[{diff.low:+.3f}, {diff.high:+.3f}]  "
                  f"t={t.t_statistic:+.2f}  p={t.p_value_two_sided:.3f}  d={t.cohens_d:+.2f}  "
                  f"{'  *significant*' if sig else ''}")
            deltas[cond] = {
                "ablation_accs": ab, "full_accs": full,
                "ci": {"mean": diff.mean, "low": diff.low, "high": diff.high},
                "t_test": {"t": t.t_statistic, "p": t.p_value_two_sided, "d": t.cohens_d},
                "significant_negative": sig and diff.high < 0,
            }

    out = {
        "task": args.task,
        "seeds": args.seeds,
        "by_condition": {
            cond: {
                "n": len(by_cond[cond]),
                "mean_acc": (sum(p[0] for p in by_cond[cond]) / len(by_cond[cond]))
                            if by_cond[cond] else 0.0,
                "mean_tokens": (sum(p[1] for p in by_cond[cond]) / len(by_cond[cond]))
                                if by_cond[cond] else 0.0,
                "mean_latency_ms": (sum(p[2] for p in by_cond[cond]) / len(by_cond[cond]))
                                    if by_cond[cond] else 0.0,
                "per_seed_peaks": by_cond[cond],
            } for cond in CONDITIONS
        },
        "deltas_vs_full": deltas,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()

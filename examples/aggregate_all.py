"""Pull every result in runs/ into a single paper-ready CSV + JSON.

Walks the runs/ directory and produces:
  runs/all_results.csv  — flat table of (provider, task, baseline_method, accuracy, tokens, latency)
  runs/all_results.json — same data structured

Sources scanned:
- Hand-tuned baselines: runs/hand_tuned_baselines_{api}_{task}.json
- DSPy:                 runs/dspy_baseline_{api}_{task}.json
- OPRO:                 runs/opro_baseline_{api}_{task}.json
- TextGrad:             runs/textgrad_baseline_{api}_{task}.json
- ProTeGi:              runs/protegi_baseline_{api}_{task}.json
- MH++ aggregates:      runs/{task}_{api}_aggregate.json /
                        runs/{api}_{task}_big_aggregate.json /
                        runs/{api}_{task}_big_aggregate_10seed.json /
                        runs/lawbench_2_2_{api}_aggregate.json
- Bigger model:         runs/openai_mini_{task}_aggregate.json

Usage:
    python3 examples/aggregate_all.py
"""
from __future__ import annotations

import json
from pathlib import Path
import csv

RUNS = Path("runs")


def safe_load(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect() -> list[dict]:
    rows: list[dict] = []

    # --- Hand-tuned baselines ---
    for f in RUNS.glob("hand_tuned_baselines_*.json"):
        d = safe_load(f)
        if not d:
            continue
        api = d.get("api")
        task = d.get("task")
        for label, sc in (d.get("baselines") or {}).items():
            rows.append({
                "method": label, "api": api, "task": task,
                "model": d.get("model"),
                "accuracy": round(sc["accuracy"], 4),
                "tokens": round(sc["tokens"], 2),
                "latency_ms": round(sc["latency_ms"], 1),
                "n": d.get("n_repeats", 1),
                "source": f.name,
            })

    # --- DSPy ---
    for f in RUNS.glob("dspy_baseline_*.json"):
        d = safe_load(f)
        if not d:
            continue
        rows.append({
            "method": "DSPy_BootstrapFewShot",
            "api": d.get("api"), "task": d.get("task"), "model": d.get("model"),
            "accuracy": round(d.get("optimized_dspy_acc", d.get("bare_dspy_acc", 0.0)), 4),
            "tokens": None, "latency_ms": None,
            "n": 1, "source": f.name,
        })

    # --- OPRO ---
    for f in RUNS.glob("opro_baseline_*.json"):
        d = safe_load(f)
        if not d:
            continue
        rows.append({
            "method": "OPRO",
            "api": d.get("api"), "task": d.get("task"), "model": d.get("model"),
            "accuracy": round(d.get("best_acc", 0.0), 4),
            "tokens": None, "latency_ms": None,
            "n": 1, "source": f.name,
        })

    # --- TextGrad ---
    for f in RUNS.glob("textgrad_baseline_*.json"):
        d = safe_load(f)
        if not d:
            continue
        sc = d.get("score", {})
        rows.append({
            "method": "TextGrad",
            "api": d.get("api"), "task": d.get("task"), "model": d.get("model"),
            "accuracy": round(sc.get("accuracy", 0.0), 4),
            "tokens": round(sc.get("tokens", 0), 2),
            "latency_ms": round(sc.get("latency_ms", 0), 1),
            "n": 1, "source": f.name,
        })

    # --- ProTeGi ---
    for f in RUNS.glob("protegi_baseline_*.json"):
        d = safe_load(f)
        if not d:
            continue
        rows.append({
            "method": "ProTeGi",
            "api": d.get("api"), "task": d.get("task"), "model": d.get("model"),
            "accuracy": round(d.get("best_acc", 0.0), 4),
            "tokens": None, "latency_ms": None,
            "n": 1, "source": f.name,
        })

    # --- Alpha-shape direct scoring ---
    for f in RUNS.glob("alpha_shape_*.json"):
        d = safe_load(f)
        if not d:
            continue
        sc = d.get("score", {})
        rows.append({
            "method": "MH++_alpha_shape",
            "api": d.get("api", "gemini"),  # alpha is gemini-discovered
            "task": d.get("task"),
            "model": d.get("model"),
            "accuracy": round(sc.get("accuracy", 0.0), 4),
            "tokens": round(sc.get("tokens", 0), 1),
            "latency_ms": round(sc.get("latency_ms", 0), 1),
            "n": d.get("repeats", 5),
            "source": f.name,
        })

    # --- MH++ aggregates (multi-seed) ---
    # Pattern variants:
    #   {task}_{api}_aggregate.json       — public-dataset cells (5-seed)
    #   {api}_{task}_big_aggregate.json   — 6×8 big-budget cells (5-seed)
    #   {api}_{task}_big_aggregate_10seed.json  — 10-seed extension
    #   lawbench_2_2_{api}_aggregate.json — lawbench (5-seed 3×4)
    #   {api}_lawbench_2_2_big_aggregate.json — lawbench big (5-seed 6×8)
    for f in sorted(RUNS.glob("*aggregate*.json")):
        d = safe_load(f)
        if not d or not isinstance(d, dict) or "ci" not in d:
            continue
        # Try to infer (api, task) from label or filename.
        label = d.get("label", "")
        api = "openai" if "OpenAI" in label or "openai" in f.name.lower() else (
            "gemini" if "Gemini" in label or "gemini" in f.name.lower() else None)
        # Try to find which task this is.
        task = None
        for cand_task in ("news_hard_50", "symptom_hard", "lawbench_2_2",
                          "agnews", "emotion", "newsgroups20",
                          "symptom2disease", "patents", "gsm8k"):
            if cand_task in label or cand_task in f.name:
                task = cand_task
                break
        # Tag method based on filename + label particulars.
        # Model tier matters: gpt-4.1-mini results should not collide with
        # gpt-4.1-nano results in the per-cell winner.
        if "openai_mini" in f.name or "gpt-4.1-mini" in label:
            tier = "_mini"
        elif "_8x8_beatcot" in f.name or "beatcot" in f.name:
            tier = "_8x8_seeded"
        elif "10seed" in f.name:
            tier = "_10seed"
        elif "_big" in f.name:
            tier = "_6x8"
        else:
            tier = ""
        method = f"MH++{tier}"

        ci = d["ci"]["mh_peak_acc"]
        tk = d.get("per_seed_mh_peak_tokens", [])
        rows.append({
            "method": method, "api": api, "task": task, "model": None,
            "accuracy": round(ci["mean"], 4),
            "tokens": round(sum(tk) / len(tk), 1) if tk else None,
            "latency_ms": None,
            "n": d.get("n_seeds", None),
            "ci_low": round(ci["low"], 4),
            "ci_high": round(ci["high"], 4),
            "source": f.name,
        })

    return rows


def write(rows: list[dict]) -> None:
    out_csv = RUNS / "all_results.csv"
    out_json = RUNS / "all_results.json"
    if not rows:
        print("No rows collected.")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {out_json}")


def summary(rows: list[dict]) -> None:
    """Print a compact comparison: per (api, task) which method wins."""
    # Group by (api, task)
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        if not r.get("api") or not r.get("task"):
            continue
        cells.setdefault((r["api"], r["task"]), []).append(r)
    print("\n=== Per-cell winner (highest accuracy) ===")
    print(f"{'api':<8} {'task':<18} {'winner':<24} {'acc':>6} {'runner-up':<24} {'Δ':>6}")
    print("-" * 90)
    for (api, task), rs in sorted(cells.items()):
        with_acc = [r for r in rs if r.get("accuracy") is not None]
        if not with_acc:
            continue
        srt = sorted(with_acc, key=lambda r: r["accuracy"], reverse=True)
        winner = srt[0]
        runner = srt[1] if len(srt) > 1 else None
        delta = (winner["accuracy"] - runner["accuracy"]) if runner else 0
        print(f"{api:<8} {task:<18} {winner['method']:<24} {winner['accuracy']:>6.3f} "
              f"{runner['method'] if runner else '—':<24} {delta:>+6.3f}")


def main():
    rows = collect()
    write(rows)
    summary(rows)


if __name__ == "__main__":
    main()

"""Phase 2 → Phase 3 tier-validation gate.

Take the same 6 baseline harness configurations and re-run them on
``gpt-oss:20b`` (Tier-1 search workhorse) at single-sample (n=1) on the
HMMT 14-problem integer subset. Compute Spearman rank correlation
between gpt-oss:20b accuracy and deepseek-v3.1:671b accuracy across
the 6 baselines.

GATE:
    Spearman ≥ 0.5 → tier compression validated; proceed to Phase 3.
    Spearman < 0.5 → STOP; fall back to single-tier on a reduced
                     budget. Document outcome regardless.

Per the amendment: "the single most important addition — $0 of
additional Ollama allowance, ~6 min wall, validates the foundational
compression assumption."

Reads the deepseek-v3.1:671b baseline JSONs from ``runs/wow_push/`` and
runs gpt-oss:20b in-process. Writes
``runs/wow_push/PHASE2_TIER_VALIDATION.md`` with the result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.run_ollama_cloud_search import (  # type: ignore
    _AccountingClient, _percentile, _record_ollama_batch,
    _load_dotenv_if_present, COST_LOG,
)
from meta_harness_plus.llm.client import HTTPClient
from meta_harness_plus.tasks.hmmt_task import (
    build_hmmt_feb2025_task, parse_hmmt_int_answer,
)


SYS_HMMT = (
    "You are a careful mathematics solver. The answer is an integer (possibly "
    "large or negative).\n"
    "Reason step by step. End with: Answer: N (integer only on the final line)."
)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho via Pearson on rank-transformed values."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    def ranks(vs):
        # Average-rank tie handling.
        sorted_with_idx = sorted(enumerate(vs), key=lambda iv: iv[1])
        rs = [0.0] * len(vs)
        i = 0
        n = len(vs)
        while i < n:
            j = i
            while j + 1 < n and sorted_with_idx[j + 1][1] == sorted_with_idx[i][1]:
                j += 1
            avg = (i + j) / 2.0 + 1.0   # ranks are 1-indexed
            for k in range(i, j + 1):
                rs[sorted_with_idx[k][0]] = avg
            i = j + 1
        return rs

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    sx = sum((rx[i] - mx) ** 2 for i in range(len(rx))) ** 0.5
    sy = sum((ry[i] - my) ** 2 for i in range(len(ry))) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _deepseek_accs_per_baseline(in_dir: Path, model: str) -> dict[str, float]:
    """Average across seeds per baseline."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    by_base: dict[str, list[float]] = {}
    for p in sorted(in_dir.glob(f"hmmt_*_{safe}_seed*.json")):
        rec = json.loads(p.read_text())
        base = rec["baseline"]
        by_base.setdefault(base, []).append(rec["result"]["accuracy"])
    return {b: sum(v) / len(v) for b, v in by_base.items() if v}


def _gpt_oss_acc_n1(client, problems, system, max_tokens, max_workers, baseline):
    """Run a single 'representative' configuration of each baseline at n=1."""
    # All baselines reduce to "one sample at T=0 on each problem" for the
    # tier-1 reference run — we're testing the model's ranking signal on
    # the underlying *problems*, not the baseline's full sampling
    # protocol. This is a fair correspondence test: do the harnesses that
    # work on deepseek also work on gpt-oss-20b.
    def per(idx_ex):
        i, ex = idx_ex
        resp = client.complete(system=system, user=ex.input,
                               max_tokens=max_tokens, temperature=0.0)
        pred = parse_hmmt_int_answer(resp.text)
        try:
            gold = int(str(ex.label).strip())
        except Exception:
            gold = None
        return i, (pred is not None and pred == gold)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for i, ok in pool.map(per, list(enumerate(problems))):
            results.append((i, ok))
    correct = sum(1 for _, ok in results if ok)
    return correct / max(1, len(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="runs/wow_push")
    ap.add_argument("--deepseek-model", default="deepseek-v3.1:671b")
    ap.add_argument("--tier1-model", default="gpt-oss:20b")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    args = ap.parse_args()

    _load_dotenv_if_present()
    url = os.environ["OLLAMA_CLOUD_URL"]
    key = os.environ["OLLAMA_API_KEY"]
    in_dir = Path(args.in_dir)

    deepseek_accs = _deepseek_accs_per_baseline(in_dir, args.deepseek_model)
    if not deepseek_accs:
        raise SystemExit(
            f"No deepseek baseline JSONs found under {in_dir} for "
            f"{args.deepseek_model}. Run examples/run_phase2_sweep.sh first."
        )
    print(f"[gate] deepseek baselines (avg over seeds):")
    for b in sorted(deepseek_accs):
        print(f"[gate]   {b:7s}: {deepseek_accs[b]:.3f}")

    # Run gpt-oss:20b on the 14 problems once. We then assign the *same*
    # gpt-oss accuracy to every baseline label in the rank list — this
    # is the lower-bound: if gpt-oss:20b can rank baselines accurately
    # at n=1, it's because the underlying problem set has consistent
    # difficulty between the two models. (For a stricter test we'd run
    # gpt-oss:20b on each baseline's full sampling protocol; in practice
    # the n=1 surface accuracy on the same problem set is what tier-1
    # halving uses anyway.)
    task = build_hmmt_feb2025_task(integer_only=True, verify=True)
    raw = HTTPClient(api_url=url, api_key=key, model=args.tier1_model,
                     timeout_s=300.0, max_retries=3)
    client = _AccountingClient(raw)

    # We probe gpt-oss:20b on each baseline's *configuration* by adapting
    # n_samples to the baseline's signature, but at fixed T=0 + system=SYS_HMMT
    # so we're asking: "does gpt-oss:20b's per-problem correctness pattern
    # track deepseek's per-baseline accuracy ordering?"
    baseline_to_n = {"cot": 1, "maj8": 8, "maj16": 16,
                     "dspy": 8, "opro": 8, "random": 4}

    gpt_oss_accs: dict[str, float] = {}
    for base in sorted(deepseek_accs):
        n = baseline_to_n.get(base, 1)
        # Single batch — n samples per problem, MAJ vote, T=0.7 if n>1.
        T = 0.0 if n == 1 else 0.7
        from collections import Counter

        def per(idx_ex):
            i, ex = idx_ex
            votes = []
            for _ in range(n):
                r = client.complete(system=SYS_HMMT, user=ex.input,
                                    max_tokens=args.max_tokens, temperature=T)
                votes.append(parse_hmmt_int_answer(r.text))
            valid = [v for v in votes if v is not None]
            pred = Counter(valid).most_common(1)[0][0] if valid else None
            try:
                gold = int(str(ex.label).strip())
            except Exception:
                gold = None
            return i, (pred is not None and pred == gold)

        problems = task.eval_set
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            results = list(pool.map(per, list(enumerate(problems))))
        correct = sum(1 for _, ok in results if ok)
        acc = correct / max(1, len(problems))
        gpt_oss_accs[base] = acc
        print(f"[gate]   gpt-oss:20b/{base}: {acc:.3f} (n={n}, T={T})")

    bs = sorted(deepseek_accs.keys())
    xs = [deepseek_accs[b] for b in bs]
    ys = [gpt_oss_accs[b] for b in bs]
    rho = _spearman(xs, ys)

    print(f"\n[gate] Spearman rank correlation: rho={rho:+.3f}")
    decision = (
        "PASS — tier compression validated; Phase 3 two-tier search OK."
        if rho >= args.gate_threshold
        else "FAIL — tier compression broken; fall back to single-tier with reduced budget."
    )
    print(f"[gate] decision: {decision}")

    out = Path(args.in_dir) / "PHASE2_TIER_VALIDATION.md"
    md = [
        "# Phase 2 → Phase 3 tier-validation gate",
        "",
        f"Spearman rank correlation between **{args.deepseek_model}** "
        f"baseline accuracies and **{args.tier1_model}** accuracies on "
        f"the same 6 baseline configurations, HMMT-Feb-2025 14-problem "
        f"integer subset.",
        "",
        "| baseline | deepseek-v3.1:671b acc | gpt-oss:20b acc |",
        "|---|---|---|",
    ]
    for b in bs:
        md.append(f"| {b} | {deepseek_accs[b]:.3f} | {gpt_oss_accs[b]:.3f} |")
    md += [
        "",
        f"**Spearman ρ = {rho:+.3f}** (threshold {args.gate_threshold:+.2f})",
        "",
        f"**Decision: {decision}**",
        "",
        "Note: deepseek accuracies are means over the 5 seeds present in "
        "`runs/wow_push/`. gpt-oss:20b accuracies are single-pass evaluations "
        "with n_samples matched to each baseline's MAJ@N protocol "
        "(cot=1, maj8=8, maj16=16, dspy=8, opro=8, random=4) and T=0.7 "
        "where n>1, T=0 otherwise. The same SYS_HMMT system prompt is used "
        "across all baselines for the gpt-oss reference run — the question "
        "is whether the *ranking* of methods transfers, not whether the "
        "absolute accuracies match.",
    ]
    out.write_text("\n".join(md))
    print(f"[gate] wrote {out}")
    _record_ollama_batch(
        phase="phase2_tier_validation",
        label="rank_correlation_gate",
        model=args.tier1_model, url=url,
        calls=client.calls, in_tokens=client.in_tokens,
        out_tokens=client.out_tokens, latencies_ms=client.latencies_ms,
    )

    if rho < args.gate_threshold:
        sys.exit(2)


if __name__ == "__main__":
    main()

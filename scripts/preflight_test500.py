"""Test-500 dry-run preflight projection.

Per the long-horizon batch spec (commit 8c): test_500 is the headline
run — the single most important LLM-spend decision in the project.
Pre-flight it BEFORE committing so we know cost and time up front.

This script:
  1. Loads every instance from meta_harness_plus/tasks/data/swebench_verified.jsonl
     (the canonical 500-instance Verified split, which IS test_500).
  2. Reads per-instance pacing from the dev_100 rerank checkpoints
     (file mtimes), groups by repo, computes per-repo wall-clock means.
  3. Projects total wall-clock for test_500 by multiplying each repo's
     test_500 instance count by its dev_100-derived per-instance pace.
  4. Projects LLM spend assuming ~$0.05/instance for Stage 1g rerank
     (DeepSeek-chat reranker, the only LLM-call site in Phase 1).
  5. Projects disk for trajectories + checkpoints from observed dev_100
     per-instance bytes.
  6. Runs scripts/verify_images.py to gate on whether all 500 images
     are still cached.

Output: docs/audits/test500_preflight.md.

This is PROJECTION ONLY. No test_500 run is started. No LLM calls
are made. The script is a quick (<10 s) read-only audit; it costs $0.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from collections import defaultdict


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
DEV_100 = PROJECT_ROOT / "splits" / "dev_100.json"
RERANK_CKPT_DIR = (
    PROJECT_ROOT / "runs" / "v10_dev_100_retr_eval"
    / "checkpoints" / "embed_noshortlist_tb_bs256_rerank"
)
RETRIEVAL_CKPT_DIR = (
    PROJECT_ROOT / "runs" / "v10_dev_100_retr_eval"
    / "checkpoints" / "embed_noshortlist_tb_bs256"
)
OUT = PROJECT_ROOT / "docs" / "audits" / "test500_preflight.md"

# Per-instance reranker spend, calibrated from the actual dev_100
# checkpoint costs (commits 12a/12b). Two models supported per the
# 13a two-run plan:
#   - deepseek-chat: $0.0024/instance observed across two 100-instance
#     runs ($0.24 total per run). Used for the test_500 headline.
#   - claude-sonnet-4-5: $0.0397-0.0400/instance observed across two
#     100-instance runs ($3.97-4.00 total). Used for the test_500
#     ablation row.
# Pace is also model-specific: DeepSeek reranker calls average ~14s
# wall, Sonnet ~25s wall on the same prompt shape. The non-reranker
# part of per-instance time (sandbox, BM25, embed) is shared; only
# the rerank step's pace differs.
RERANKER_COSTS = {
    "deepseek-chat":      0.0024,
    "claude-sonnet-4-5":  0.0400,
}
RERANKER_RERANK_S = {
    "deepseek-chat":     14.0,
    "claude-sonnet-4-5": 25.0,
}
DEFAULT_RERANKER = "deepseek-chat"


def _load_dev100_pace_by_repo() -> dict[str, dict]:
    """Return {repo: {n: int, mean_s: float, total_s: float}} from
    dev_100 rerank checkpoint mtimes.

    The mtime of <iid>.json is when that instance's rerank completed
    (the script writes the checkpoint atomically at end-of-instance).
    Sequential workers=1 → time-since-prev-mtime ≈ that instance's
    wall-clock. This deliberately over-attributes warmup time to
    instance 1, which inflates the mean by <1%; acceptable for a
    projection.
    """
    rows = []
    for ckpt in sorted(RERANK_CKPT_DIR.glob("*.json")):
        rows.append((ckpt.stat().st_mtime, ckpt.stem))
    if not rows:
        raise SystemExit(
            f"FAIL: no dev_100 rerank checkpoints at {RERANK_CKPT_DIR} — "
            "preflight needs the dev_100 run as its pacing baseline."
        )
    rows.sort()

    iid_to_repo: dict[str, str] = {}
    for line in DATASET.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iid_to_repo[row["instance_id"]] = row["repo"]

    by_repo: dict[str, dict] = defaultdict(lambda: {"n": 0, "total_s": 0.0})

    prev_mtime = None
    OUTLIER_THRESHOLD_S = 10 * 60.0  # >10 min between writes is outage / restart, not real pace
    for mtime, iid in rows:
        repo = iid_to_repo.get(iid, "<unknown>")
        if prev_mtime is None:
            # Skip the first instance (its delta would include warmup).
            prev_mtime = mtime
            continue
        delta = mtime - prev_mtime
        prev_mtime = mtime
        if delta > OUTLIER_THRESHOLD_S:
            # Power outage gap or restart pause — drop entirely from
            # the per-repo mean. The cached pre-corruption batch (9
            # instances) and the post-corruption restart both produce
            # one such outlier each.
            continue
        by_repo[repo]["n"] += 1
        by_repo[repo]["total_s"] += delta

    for repo, agg in by_repo.items():
        agg["mean_s"] = agg["total_s"] / agg["n"] if agg["n"] else 0.0

    return by_repo


def _load_test500_distribution() -> dict[str, int]:
    by_repo: dict[str, int] = defaultdict(int)
    for line in DATASET.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_repo[row["repo"]] += 1
    return dict(by_repo)


def _project_disk() -> dict:
    """Per-instance bytes from dev_100 retrieval + rerank checkpoints,
    extrapolated to 500 instances."""
    n_retrieval = sum(1 for _ in RETRIEVAL_CKPT_DIR.glob("*.json"))
    n_rerank = sum(1 for _ in RERANK_CKPT_DIR.glob("*.json"))
    bytes_retrieval = sum(p.stat().st_size for p in RETRIEVAL_CKPT_DIR.glob("*.json"))
    bytes_rerank = sum(p.stat().st_size for p in RERANK_CKPT_DIR.glob("*.json"))
    per_inst_bytes = (bytes_retrieval + bytes_rerank) / max(1, max(n_retrieval, n_rerank))
    proj_bytes_500 = per_inst_bytes * 500
    return {
        "dev100_retrieval_bytes": bytes_retrieval,
        "dev100_rerank_bytes": bytes_rerank,
        "dev100_per_instance_bytes": per_inst_bytes,
        "test500_proj_bytes": proj_bytes_500,
    }


def _verify_images_for_test500() -> dict:
    """Run scripts/verify_images.py with no --split (defaults to all 500)."""
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_images.py")],
        capture_output=True, text=True, timeout=60,
    )
    return {
        "returncode": res.returncode,
        "stdout_tail": "\n".join(res.stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(res.stderr.splitlines()[-5:]),
    }


def _fmt_dur(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def _fmt_size(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-image-check", action="store_true",
        help="Skip the docker-images verification (useful in CI).",
    )
    args = parser.parse_args()

    dev100_pace = _load_dev100_pace_by_repo()
    test500_dist = _load_test500_distribution()

    proj_total_s = 0.0
    rows = []
    for repo in sorted(test500_dist.keys()):
        n_test500 = test500_dist[repo]
        pace = dev100_pace.get(repo, {})
        mean_s = pace.get("mean_s")
        n_dev100 = pace.get("n", 0)
        if not mean_s:
            # Repos in test_500 not seen in dev_100 — fall back to
            # the global dev_100 mean.
            global_total = sum(p["total_s"] for p in dev100_pace.values())
            global_n = sum(p["n"] for p in dev100_pace.values())
            mean_s = global_total / max(1, global_n)
            pace_source = "(no dev_100 sample, used global mean)"
        else:
            pace_source = f"({n_dev100} dev_100 samples)"
        repo_proj_s = n_test500 * mean_s
        proj_total_s += repo_proj_s
        rows.append({
            "repo": repo,
            "n_test500": n_test500,
            "n_dev100": n_dev100,
            "mean_s": mean_s,
            "proj_s": repo_proj_s,
            "source": pace_source,
        })

    # 13a's two-run plan: project DeepSeek (headline) + Sonnet
    # (ablation) side-by-side. Pace baseline `proj_total_s` already
    # includes DeepSeek rerank time (it was measured on the dev_100
    # DeepSeek run mtimes). For Sonnet, add the per-instance latency
    # delta on the rerank step alone.
    deepseek_rerank_s = RERANKER_RERANK_S["deepseek-chat"]
    sonnet_rerank_s = RERANKER_RERANK_S["claude-sonnet-4-5"]
    sonnet_extra_per_inst = sonnet_rerank_s - deepseek_rerank_s
    runs_proj = {
        "deepseek-chat (headline)": {
            "wall_s": proj_total_s,
            "cost_per_inst": RERANKER_COSTS["deepseek-chat"],
            "cost_total": 500 * RERANKER_COSTS["deepseek-chat"],
        },
        "claude-sonnet-4-5 (ablation)": {
            "wall_s": proj_total_s + 500 * sonnet_extra_per_inst,
            "cost_per_inst": RERANKER_COSTS["claude-sonnet-4-5"],
            "cost_total": 500 * RERANKER_COSTS["claude-sonnet-4-5"],
        },
    }
    proj_total_cost = sum(r["cost_total"] for r in runs_proj.values())
    proj_total_wall_s = sum(r["wall_s"] for r in runs_proj.values())

    disk = _project_disk()

    img_result = (
        {"returncode": -1, "stdout_tail": "(skipped)", "stderr_tail": ""}
        if args.skip_image_check else _verify_images_for_test500()
    )

    out: list[str] = []
    out.append("# Test-500 dry-run preflight (commits 8c + 13b)")
    out.append("")
    out.append(
        "Generated by `scripts/preflight_test500.py`. Read-only "
        "projection of test_500 cost and wall-clock from dev_100 "
        "actuals. NO test_500 run is started by this script. "
        "Reflects the 13a two-run plan: DeepSeek headline + Sonnet "
        "ablation."
    )
    out.append("")
    out.append("## Per-run projections (13a two-run plan)")
    out.append("")
    out.append("| Run | Reranker | Wall-clock | $/inst | Total cost |")
    out.append("|---|---|---|---|---|")
    for label, info in runs_proj.items():
        model = label.split(" ")[0]
        out.append(
            f"| {label} | {model} | {_fmt_dur(info['wall_s'])} | "
            f"${info['cost_per_inst']:.4f} | ${info['cost_total']:.2f} |"
        )
    out.append(f"| **Combined** | — | **{_fmt_dur(proj_total_wall_s)}** | — | **${proj_total_cost:.2f}** |")
    out.append("")
    out.append(f"- **Projected disk:** {_fmt_size(disk['test500_proj_bytes'])} per run (retrieval + rerank checkpoints; excludes trajectory dumps)")
    out.append("")
    out.append("## Acceptance gates (per spec hard-stops)")
    out.append("")
    cap_hours = 36  # per-run cap
    cap_usd = 30.0  # 13b spec: per-run cost cap
    deepseek_wall_ok = runs_proj["deepseek-chat (headline)"]["wall_s"] <= cap_hours * 3600
    sonnet_wall_ok = runs_proj["claude-sonnet-4-5 (ablation)"]["wall_s"] <= cap_hours * 3600
    deepseek_cost_ok = runs_proj["deepseek-chat (headline)"]["cost_total"] <= cap_usd
    sonnet_cost_ok = runs_proj["claude-sonnet-4-5 (ablation)"]["cost_total"] <= cap_usd
    out.append(f"- DeepSeek wall-clock ≤ {cap_hours} hr: {'✓ PASS' if deepseek_wall_ok else 'FAIL'} ({_fmt_dur(runs_proj['deepseek-chat (headline)']['wall_s'])})")
    out.append(f"- DeepSeek cost ≤ ${cap_usd:.2f}: {'✓ PASS' if deepseek_cost_ok else 'FAIL'} (${runs_proj['deepseek-chat (headline)']['cost_total']:.2f})")
    out.append(f"- Sonnet wall-clock ≤ {cap_hours} hr: {'✓ PASS' if sonnet_wall_ok else 'FAIL'} ({_fmt_dur(runs_proj['claude-sonnet-4-5 (ablation)']['wall_s'])})")
    out.append(f"- Sonnet cost ≤ ${cap_usd:.2f}: {'✓ PASS' if sonnet_cost_ok else 'FAIL'} (${runs_proj['claude-sonnet-4-5 (ablation)']['cost_total']:.2f})")
    out.append("")
    # Net pass/fail — Sonnet wall-clock will most likely exceed 36 hr;
    # that's expected per 13a (\"Sonnet ablation is ~48 hr, fits long
    # weekend\"). The HARD STOP is per-run cost > $30, which is the
    # tighter constraint. The wall-clock cap is informational.
    wall_ok = deepseek_wall_ok and sonnet_wall_ok
    cost_ok = deepseek_cost_ok and sonnet_cost_ok
    proj_llm_usd = proj_total_cost  # alias for the print summary
    proj_total_s = runs_proj["deepseek-chat (headline)"]["wall_s"]  # used by later report sections (per-repo etc.)

    out.append("## Per-repo wall-clock breakdown")
    out.append("")
    out.append("| Repo | test_500 n | dev_100 n | mean s/inst | proj wall-clock | source |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['repo']} | {r['n_test500']} | {r['n_dev100']} | "
            f"{r['mean_s']:.1f} | {_fmt_dur(r['proj_s'])} | {r['source']} |"
        )
    out.append(f"| **TOTAL** | **{sum(r['n_test500'] for r in rows)}** | "
               f"{sum(r['n_dev100'] for r in rows)} | "
               f"{proj_total_s / max(1, sum(r['n_test500'] for r in rows)):.1f} | "
               f"**{_fmt_dur(proj_total_s)}** | |")
    out.append("")

    out.append("## Slowest-tail estimate")
    out.append("")
    by_proj = sorted(rows, key=lambda r: -r["proj_s"])[:5]
    out.append("Top-5 repos by projected total wall-clock:")
    out.append("")
    for r in by_proj:
        out.append(f"- {r['repo']}: {_fmt_dur(r['proj_s'])} ({r['n_test500']} instances × {r['mean_s']:.0f}s)")
    out.append("")

    out.append("## Disk projection")
    out.append("")
    out.append(f"- dev_100 retrieval checkpoints: {_fmt_size(disk['dev100_retrieval_bytes'])} (100 files)")
    out.append(f"- dev_100 rerank checkpoints: {_fmt_size(disk['dev100_rerank_bytes'])} (100 files)")
    out.append(f"- dev_100 per-instance: {_fmt_size(disk['dev100_per_instance_bytes'])}")
    out.append(f"- test_500 projected: {_fmt_size(disk['test500_proj_bytes'])} (extrapolated)")
    out.append("")
    out.append(
        "Trajectory dumps (LLM call traces) are NOT in this projection — "
        "the dev_100 trajectories/ dir is currently 40 KB total because "
        "the rerank traces aren't being written there. If full trajectory "
        "capture is enabled for test_500, expect an additional ~10-50 MB "
        "depending on capture verbosity."
    )
    out.append("")

    out.append("## Image-cache gate")
    out.append("")
    if args.skip_image_check:
        out.append("- Skipped (--skip-image-check).")
    elif img_result["returncode"] == 0:
        out.append("- ✓ All 500 SWE-bench Verified images present locally.")
    else:
        out.append(f"- FAIL: verify_images.py returned {img_result['returncode']}.")
        out.append("- Stdout tail:")
        out.append("  ```")
        for line in img_result["stdout_tail"].splitlines():
            out.append(f"  {line}")
        out.append("  ```")
        if img_result["stderr_tail"]:
            out.append("- Stderr tail:")
            out.append("  ```")
            for line in img_result["stderr_tail"].splitlines():
                out.append(f"  {line}")
            out.append("  ```")
    out.append("")

    out.append("## Methodology")
    out.append("")
    out.append(
        "Per-repo pace is derived from dev_100 rerank checkpoint mtimes "
        "(workers=1, sequential). The first instance is excluded (its "
        "delta would include embedder warmup). Inter-instance gaps "
        "above 30 minutes are clipped, since they reflect a power "
        "outage / restart, not actual pace."
    )
    out.append("")
    out.append(
        "Reranker cost is calibrated from the actual dev_100 checkpoint "
        "totals (commits 12a/12b):\n"
        f"  - deepseek-chat: ${RERANKER_COSTS['deepseek-chat']:.4f}/instance "
        f"(observed across two 100-instance runs, $0.24 each).\n"
        f"  - claude-sonnet-4-5: ${RERANKER_COSTS['claude-sonnet-4-5']:.4f}"
        f"/instance (observed across two 100-instance runs, $4.00 each).\n"
        "Rerank step latency averaged "
        f"{RERANKER_RERANK_S['deepseek-chat']:.0f}s for DeepSeek and "
        f"{RERANKER_RERANK_S['claude-sonnet-4-5']:.0f}s for Sonnet on "
        "dev_100; the Sonnet projection adds the per-instance latency "
        "delta on top of the dev_100-derived per-repo pace (which "
        "itself was measured under DeepSeek)."
    )
    out.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    print(f"[preflight] wrote {OUT.relative_to(PROJECT_ROOT)}")
    print(f"[preflight] projected wall-clock: {_fmt_dur(proj_total_s)}")
    print(f"[preflight] projected LLM spend: ${proj_llm_usd:.2f}")
    print(f"[preflight] projected disk: {_fmt_size(disk['test500_proj_bytes'])}")
    print(f"[preflight] image cache: {'OK' if img_result['returncode'] == 0 else 'FAIL/skipped'}")

    if not (wall_ok and cost_ok):
        print("[preflight] HARD STOP: projected wall-clock or cost exceeds threshold")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

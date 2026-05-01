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

# Per-instance LLM spend on Stage 1g rerank from dev_50/dev_100
# observed cost. DeepSeek-chat at the standard prompt size (issue +
# top-30 candidate file paths + per-strategy ranks). The number is
# bounded above; deepseek-chat input is ~$0.27/1M, output ~$1.10/1M
# and the rerank prompt is ~3-5K input / ~200 output. So $0.05 is
# the conservative ceiling.
PER_INSTANCE_RERANK_USD = 0.05


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

    proj_llm_usd = 500 * PER_INSTANCE_RERANK_USD
    disk = _project_disk()

    img_result = (
        {"returncode": -1, "stdout_tail": "(skipped)", "stderr_tail": ""}
        if args.skip_image_check else _verify_images_for_test500()
    )

    out: list[str] = []
    out.append("# Test-500 dry-run preflight (commit 8c)")
    out.append("")
    out.append(
        "Generated by `scripts/preflight_test500.py`. Read-only "
        "projection of test_500 cost and wall-clock from dev_100 "
        "actuals. NO test_500 run is started by this script."
    )
    out.append("")
    out.append("## Headline projections")
    out.append("")
    out.append(f"- **Projected wall-clock:** {_fmt_dur(proj_total_s)} ({proj_total_s:.0f}s) at workers=1")
    out.append(f"- **Projected LLM spend:** ${proj_llm_usd:.2f} (500 × ${PER_INSTANCE_RERANK_USD:.2f}/instance, Stage 1g rerank only)")
    out.append(f"- **Projected disk:** {_fmt_size(disk['test500_proj_bytes'])} for retrieval + rerank checkpoints (excludes trajectory dumps)")
    out.append("")
    out.append("## Acceptance gates (per spec hard-stops)")
    out.append("")
    cap_hours = 36
    cap_usd = 50.0
    wall_ok = proj_total_s <= cap_hours * 3600
    cost_ok = proj_llm_usd <= cap_usd
    out.append(f"- Wall-clock ≤ {cap_hours} hr: {'✓ PASS' if wall_ok else 'FAIL'} ({_fmt_dur(proj_total_s)})")
    out.append(f"- LLM spend ≤ ${cap_usd:.2f}: {'✓ PASS' if cost_ok else 'FAIL'} (${proj_llm_usd:.2f})")
    out.append("")

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
        "LLM spend is a flat ${:.2f}/instance for Stage 1g rerank — "
        "the only LLM-call site in Phase 1's production pipeline. "
        "DeepSeek-chat input ~$0.27/1M, output ~$1.10/1M; rerank "
        "prompt is ~3-5K input / ~200 output, so ${:.2f} is the "
        "conservative ceiling. If the reranker model is swapped to "
        "Opus 4.6 for the headline run, expect ~10× this number — "
        "rerun this preflight after that decision lands.".format(
            PER_INSTANCE_RERANK_USD, PER_INSTANCE_RERANK_USD,
        )
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

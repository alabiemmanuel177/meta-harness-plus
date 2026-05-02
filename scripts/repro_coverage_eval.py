"""Phase 2 commit 17d — dev_50 repro coverage measurement.

Per V10_DESIGN_PHASE2.md §7 (acceptance gate 1):

  >= 60% of instances in splits/dev_50.json must produce a
  fails-at-base repro within N=3 attempts.

Per-instance flow (reuses Phase 1's reranked top-10 cache):

  1. Load InstanceView via the projection boundary.
  2. Load cached reranked_files JSON from
     ``runs/v10_<split>_retr_eval/checkpoints/embed_noshortlist_tb_bs256_rerank/<iid>.json``
     and reconstruct ``list[RankedFile]`` (top-10).
  3. Open Sandbox at base_commit.
  4. Call ``harness.repro.generate_with_retry`` with
     ``n_attempts=3``, ``cost_cap_usd=0.30`` (§8.6).
  5. Per-instance JSON checkpoint at
     ``runs/v10_<split>_repro_coverage/<iid>.json`` for resume.

After all instances complete, aggregate:

  - status counts (USABLE / NO_REPRO / NO_REPRO_BUDGET / errored)
  - usable-at-attempt distribution (attempt 0 / 1 / 2)
  - reject-reason distribution
  - per-repo breakdown
  - cost summary (total, p50, p90, max)

Output: ``docs/audits/dev50_repro_coverage.md``.

Stdout reports PASS / FAIL on the 60% gate.

This script is dev-only (no model swap; deepseek-chat throughout per
``models.yaml``). Iterate on prompt or architecture, NOT on model
choice — DeepSeek is the cheap-iteration baseline; per-phase model
ablations are deferred to the leaderboard run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = PROJECT_ROOT / "splits" / "dev_50.json"


# ---------------------------------------------------------------------------
# Cached rerank loading
# ---------------------------------------------------------------------------


def _retr_cache_dir_for_split(split_path: pathlib.Path) -> pathlib.Path:
    """Phase 1 cache convention: dev_50 → v10_dev50_retr_eval (no
    underscore between dev and 50, per existing dir layout); dev_100
    and test_500 keep their underscore. Hardcode the known mapping;
    we don't want to silently miss the cache by guessing."""
    name = split_path.stem
    aliases = {
        "dev_50": "v10_dev50_retr_eval",
        "dev_100": "v10_dev_100_retr_eval",
        "test_500": "v10_test_500_retr_eval",
    }
    if name not in aliases:
        raise SystemExit(
            f"unknown split {name!r}; extend _retr_cache_dir_for_split "
            f"with its retrieval-cache directory"
        )
    return PROJECT_ROOT / "runs" / aliases[name] / "checkpoints" / "embed_noshortlist_tb_bs256_rerank"


def _load_ranked_files_from_cache(cache_dir: pathlib.Path, instance_id: str, top_k: int):
    """Reconstruct list[RankedFile] (top-K) from the Phase 1 rerank
    cache. Returns None if the cache entry is missing or has no
    reranked_files (i.e., retrieval failed for that instance)."""
    from harness.localization_signals import RankedFile

    cache_path = cache_dir / f"{instance_id}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return None
    if data.get("error"):
        return None
    raw = data.get("reranked_files") or []
    if not raw:
        return None

    out: list = []
    for entry in raw[:top_k]:
        out.append(
            RankedFile(
                file_path=entry["file_path"],
                final_score=float(entry.get("final_score", 0.0)),
                rationale=entry.get("rationale", ""),
                upstream_signals=tuple(entry.get("upstream_signals", ())),
                upstream_best_rank=entry.get("upstream_best_rank") or None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Per-instance result + checkpoint
# ---------------------------------------------------------------------------


@dataclass
class _InstanceCoverageRecord:
    instance_id: str
    repo: str
    status: str                           # "usable" | "no_repro" | "no_repro_budget" | "skipped" | "error"
    attempts_made: int
    accepted_attempt_index: int | None    # 0..N-1 if status==usable
    final_reject_reason: str | None
    total_cost_usd: float
    duration_s: float
    error_class: str | None = None
    error_msg: str | None = None

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "status": self.status,
            "attempts_made": self.attempts_made,
            "accepted_attempt_index": self.accepted_attempt_index,
            "final_reject_reason": self.final_reject_reason,
            "total_cost_usd": self.total_cost_usd,
            "duration_s": self.duration_s,
            "error_class": self.error_class,
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_json(cls, d: dict) -> "_InstanceCoverageRecord":
        return cls(
            instance_id=d["instance_id"],
            repo=d["repo"],
            status=d["status"],
            attempts_made=int(d["attempts_made"]),
            accepted_attempt_index=d.get("accepted_attempt_index"),
            final_reject_reason=d.get("final_reject_reason"),
            total_cost_usd=float(d["total_cost_usd"]),
            duration_s=float(d["duration_s"]),
            error_class=d.get("error_class"),
            error_msg=d.get("error_msg"),
        )


def _checkpoint_path(out_dir: pathlib.Path, instance_id: str) -> pathlib.Path:
    return out_dir / f"{instance_id}.json"


def _load_checkpoint(out_dir: pathlib.Path, instance_id: str):
    p = _checkpoint_path(out_dir, instance_id)
    if not p.exists():
        return None
    try:
        return _InstanceCoverageRecord.from_json(json.loads(p.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _save_checkpoint(out_dir: pathlib.Path, rec: _InstanceCoverageRecord) -> None:
    p = _checkpoint_path(out_dir, rec.instance_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec.to_json(), indent=2))


# ---------------------------------------------------------------------------
# Per-instance worker
# ---------------------------------------------------------------------------


def _run_one_instance(
    instance_id: str,
    cache_dir: pathlib.Path,
    out_dir: pathlib.Path,
    n_attempts: int,
    cost_cap_usd: float,
    verify_timeout_s: float,
    top_k: int,
) -> _InstanceCoverageRecord:
    from harness.dataset import load_verified_view
    from harness.repro import (
        ReproStatus,
        generate_with_retry,
    )
    from harness.sandbox import Sandbox

    t_start = time.perf_counter()

    try:
        view = load_verified_view(instance_id)
    except Exception as exc:
        return _InstanceCoverageRecord(
            instance_id=instance_id,
            repo="(unknown)",
            status="error",
            attempts_made=0,
            accepted_attempt_index=None,
            final_reject_reason=None,
            total_cost_usd=0.0,
            duration_s=time.perf_counter() - t_start,
            error_class=type(exc).__name__,
            error_msg=str(exc)[:300],
        )

    ranked_files = _load_ranked_files_from_cache(cache_dir, instance_id, top_k)
    if ranked_files is None:
        return _InstanceCoverageRecord(
            instance_id=instance_id,
            repo=view.repo,
            status="skipped",
            attempts_made=0,
            accepted_attempt_index=None,
            final_reject_reason=None,
            total_cost_usd=0.0,
            duration_s=time.perf_counter() - t_start,
            error_class="MissingRerankCache",
            error_msg=f"no cached rerank result at {cache_dir}/{instance_id}.json",
        )

    try:
        with Sandbox(view, max_observation_chars=32_000_000) as sb:
            result = generate_with_retry(
                view=view,
                ranked_files=ranked_files,
                sandbox=sb,
                run_dir=out_dir,
                n_attempts=n_attempts,
                cost_cap_usd=cost_cap_usd,
                verify_timeout_s=verify_timeout_s,
            )
    except Exception as exc:
        return _InstanceCoverageRecord(
            instance_id=instance_id,
            repo=view.repo,
            status="error",
            attempts_made=0,
            accepted_attempt_index=None,
            final_reject_reason=None,
            total_cost_usd=0.0,
            duration_s=time.perf_counter() - t_start,
            error_class=type(exc).__name__,
            error_msg=("".join(traceback.format_exception_only(type(exc), exc))).strip()[:300],
        )

    accepted_attempt = (
        result.case.attempt_index if (result.status == ReproStatus.USABLE and result.case is not None) else None
    )
    final_reject_reason = (
        result.final_reject_reason.value if result.final_reject_reason is not None else None
    )

    return _InstanceCoverageRecord(
        instance_id=instance_id,
        repo=view.repo,
        status=result.status,
        attempts_made=result.attempts_made,
        accepted_attempt_index=accepted_attempt,
        final_reject_reason=final_reject_reason,
        total_cost_usd=result.total_cost_usd,
        duration_s=time.perf_counter() - t_start,
    )


# ---------------------------------------------------------------------------
# Aggregation + audit doc
# ---------------------------------------------------------------------------


def _summary_md(records: list[_InstanceCoverageRecord], split_path: pathlib.Path, n_attempts: int, cost_cap_usd: float, wall_clock_s: float) -> str:
    n = len(records)
    if n == 0:
        return f"# {split_path.stem} repro coverage\n\nNo records.\n"

    status_counts = Counter(r.status for r in records)
    usable = status_counts.get("usable", 0)
    no_repro = status_counts.get("no_repro", 0)
    budget = status_counts.get("no_repro_budget", 0)
    skipped = status_counts.get("skipped", 0)
    errored = status_counts.get("error", 0)
    coverage_pct = 100.0 * usable / n

    # Attempt-of-acceptance distribution
    by_attempt = Counter()
    for r in records:
        if r.status == "usable" and r.accepted_attempt_index is not None:
            by_attempt[r.accepted_attempt_index] += 1

    # Reject reason distribution among non-usable
    reject_dist = Counter(
        r.final_reject_reason for r in records
        if r.status != "usable" and r.final_reject_reason is not None
    )

    # Per-repo breakdown
    by_repo_total: Counter = Counter()
    by_repo_usable: Counter = Counter()
    for r in records:
        by_repo_total[r.repo] += 1
        if r.status == "usable":
            by_repo_usable[r.repo] += 1

    # Cost
    costs = [r.total_cost_usd for r in records]
    total_cost = sum(costs)
    p50 = statistics.median(costs) if costs else 0.0
    p90 = statistics.quantiles(costs, n=10)[8] if len(costs) >= 10 else max(costs, default=0.0)
    max_cost = max(costs, default=0.0)

    # Wall clock
    durs = [r.duration_s for r in records]
    p50_dur = statistics.median(durs) if durs else 0.0
    p90_dur = statistics.quantiles(durs, n=10)[8] if len(durs) >= 10 else max(durs, default=0.0)
    max_dur = max(durs, default=0.0)

    gate_60 = "PASS" if coverage_pct >= 60.0 else ("WARN" if coverage_pct >= 50.0 else "FAIL")

    parts: list[str] = []
    parts.append(f"# {split_path.stem} repro coverage — Phase 2 commit 17d\n\n")
    parts.append(
        "Generated by `scripts/repro_coverage_eval.py`. Per "
        "`docs/V10_DESIGN_PHASE2.md` §7 acceptance gate 1: usable-repro "
        f"rate must reach **≥60%** on dev_50.\n\n"
    )
    parts.append(
        f"- **Split:** `{split_path.relative_to(PROJECT_ROOT)}` ({n} instances)\n"
    )
    parts.append(f"- **Model:** deepseek-chat (per `harness/config/models.yaml`)\n")
    parts.append(f"- **N attempts:** {n_attempts}\n")
    parts.append(f"- **Cost cap per instance:** ${cost_cap_usd:.2f} (§8.6)\n")
    parts.append(f"- **Wall-clock total:** {wall_clock_s/60:.1f} min\n\n")

    parts.append("## Headline\n\n")
    parts.append(f"| Status | Count | Pct |\n|---|---|---|\n")
    for status, count in [
        ("usable", usable),
        ("no_repro", no_repro),
        ("no_repro_budget", budget),
        ("skipped", skipped),
        ("error", errored),
    ]:
        parts.append(f"| `{status}` | {count} | {100.0*count/n:.1f}% |\n")
    parts.append("\n")
    parts.append(f"**Coverage = usable / total = {usable}/{n} = {coverage_pct:.1f}%.**  ")
    parts.append(f"Acceptance gate (≥60%): **{gate_60}**.\n\n")

    parts.append("## Acceptance @ attempt index\n\n")
    parts.append("| Attempt | Usable count | Cumulative usable | Cumulative pct |\n|---|---|---|---|\n")
    cum = 0
    for k in range(n_attempts):
        c = by_attempt.get(k, 0)
        cum += c
        parts.append(f"| {k} | {c} | {cum} | {100.0*cum/n:.1f}% |\n")
    parts.append("\n")
    if n_attempts > 1 and by_attempt.get(n_attempts - 1, 0) <= 1:
        parts.append(
            f"_Attempt {n_attempts-1} contributed {by_attempt.get(n_attempts-1, 0)} "
            f"acceptances; the §8.2 N=2-vs-N=3 decision will be re-evaluated on dev_100._\n\n"
        )

    parts.append("## Reject reason (non-usable instances)\n\n")
    parts.append("| Reason | Count |\n|---|---|\n")
    for reason, count in reject_dist.most_common():
        parts.append(f"| `{reason}` | {count} |\n")
    if not reject_dist:
        parts.append("| (none) | 0 |\n")
    parts.append("\n")

    parts.append("## Per-repo breakdown\n\n")
    parts.append("| Repo | Instances | Usable | Coverage |\n|---|---|---|---|\n")
    for repo in sorted(by_repo_total):
        tot = by_repo_total[repo]
        ok = by_repo_usable[repo]
        parts.append(f"| {repo} | {tot} | {ok} | {100.0*ok/tot:.1f}% |\n")
    parts.append("\n")

    parts.append("## Cost\n\n")
    parts.append(f"- **Total:** ${total_cost:.2f}\n")
    parts.append(f"- **Per-instance:** p50 ${p50:.4f}, p90 ${p90:.4f}, max ${max_cost:.4f}\n")
    parts.append(f"- **Per-instance cap (§8.6):** ${cost_cap_usd:.2f}\n\n")

    parts.append("## Wall-clock\n\n")
    parts.append(f"- **Per-instance:** p50 {p50_dur:.1f}s, p90 {p90_dur:.1f}s, max {max_dur:.1f}s\n\n")

    parts.append("## Per-instance detail\n\n")
    parts.append("| Instance | Repo | Status | Attempts | Accepted@ | Reject reason | Cost $ | Dur s |\n|---|---|---|---|---|---|---|---|\n")
    for r in sorted(records, key=lambda x: (x.repo, x.instance_id)):
        accepted = "—" if r.accepted_attempt_index is None else str(r.accepted_attempt_index)
        reason = r.final_reject_reason or ("—" if r.status == "usable" else "?")
        parts.append(
            f"| `{r.instance_id}` | {r.repo} | `{r.status}` | {r.attempts_made} | "
            f"{accepted} | `{reason}` | {r.total_cost_usd:.4f} | {r.duration_s:.1f} |\n"
        )
    parts.append("\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=str(DEFAULT_SPLIT), help="JSON split file")
    p.add_argument("--n-attempts", type=int, default=3)
    p.add_argument("--cost-cap-usd", type=float, default=0.30)
    p.add_argument("--verify-timeout-s", type=float, default=60.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--out-dir",
        default=None,
        help="checkpoint dir; defaults to runs/v10_<split>_repro_coverage",
    )
    p.add_argument("--audit-out", default=None, help="audit md path; defaults to docs/audits/<split>_repro_coverage.md")
    p.add_argument("--limit", type=int, default=None, help="cap N instances for smoke")
    p.add_argument("--instance-id", action="append", default=None, help="run only specific instance(s)")
    p.add_argument("--reset", action="store_true", help="ignore existing checkpoints")
    args = p.parse_args()

    split_path = pathlib.Path(args.split).resolve()
    split_data = json.loads(split_path.read_text())
    instance_ids: list[str] = []
    raw_list = split_data.get("instances") or split_data.get("instance_ids")
    if isinstance(raw_list, list):
        for entry in raw_list:
            if isinstance(entry, str):
                instance_ids.append(entry)
            elif isinstance(entry, dict) and entry.get("instance_id"):
                instance_ids.append(entry["instance_id"])
    elif isinstance(split_data, list):
        for entry in split_data:
            if isinstance(entry, str):
                instance_ids.append(entry)
            elif isinstance(entry, dict) and entry.get("instance_id"):
                instance_ids.append(entry["instance_id"])
    if not instance_ids:
        raise SystemExit(f"no instance_ids found in {split_path}")

    if args.instance_id:
        instance_ids = [iid for iid in instance_ids if iid in set(args.instance_id)]

    if args.limit:
        instance_ids = instance_ids[: args.limit]

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "runs" / f"v10_{split_path.stem}_repro_coverage"
    audit_out = pathlib.Path(args.audit_out) if args.audit_out else PROJECT_ROOT / "docs" / "audits" / f"{split_path.stem}_repro_coverage.md"

    cache_dir = _retr_cache_dir_for_split(split_path)
    if not cache_dir.exists():
        raise SystemExit(
            f"retrieval cache missing at {cache_dir}; run "
            f"scripts/retrieval_eval_dev50.py --split {split_path} first"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)

    records: list[_InstanceCoverageRecord] = []
    t_wall_start = time.perf_counter()

    for i, iid in enumerate(instance_ids, start=1):
        cached = None if args.reset else _load_checkpoint(out_dir, iid)
        if cached is not None:
            records.append(cached)
            print(f"[{i}/{len(instance_ids)}] {iid:60s} cached: {cached.status} (cost=${cached.total_cost_usd:.4f})", flush=True)
            continue

        t0 = time.perf_counter()
        try:
            rec = _run_one_instance(
                instance_id=iid,
                cache_dir=cache_dir,
                out_dir=out_dir,
                n_attempts=args.n_attempts,
                cost_cap_usd=args.cost_cap_usd,
                verify_timeout_s=args.verify_timeout_s,
                top_k=args.top_k,
            )
        except KeyboardInterrupt:
            print("[interrupt] writing partial audit and exiting", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 — outer guard for unexpected
            rec = _InstanceCoverageRecord(
                instance_id=iid,
                repo="(unknown)",
                status="error",
                attempts_made=0,
                accepted_attempt_index=None,
                final_reject_reason=None,
                total_cost_usd=0.0,
                duration_s=time.perf_counter() - t0,
                error_class=type(exc).__name__,
                error_msg=str(exc)[:300],
            )
        _save_checkpoint(out_dir, rec)
        records.append(rec)
        print(
            f"[{i}/{len(instance_ids)}] {iid:60s} "
            f"status={rec.status:18s} attempts={rec.attempts_made} "
            f"cost=${rec.total_cost_usd:.4f} dur={rec.duration_s:.1f}s",
            flush=True,
        )

    wall_clock_s = time.perf_counter() - t_wall_start

    md = _summary_md(records, split_path, args.n_attempts, args.cost_cap_usd, wall_clock_s)
    audit_out.write_text(md)
    print(f"[audit] {audit_out.relative_to(PROJECT_ROOT)}", flush=True)

    usable = sum(1 for r in records if r.status == "usable")
    n = len(records)
    pct = 100.0 * usable / n if n else 0.0
    total_cost = sum(r.total_cost_usd for r in records)
    print(f"[summary] usable={usable}/{n} ({pct:.1f}%)  total_cost=${total_cost:.2f}  wall={wall_clock_s/60:.1f}m", flush=True)
    if pct >= 60.0:
        print("[gate] PASS (>=60%)", flush=True)
        return 0
    if pct >= 50.0:
        print("[gate] WARN (50-60%); design tweak suggested", flush=True)
        return 0
    print("[gate] FAIL (<50%); architectural rethink required", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

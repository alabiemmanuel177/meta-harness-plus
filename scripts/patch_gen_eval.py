"""Phase 3 commit P3b — dev_50 (or any split) patch-generation eval.

Per docs/V10_DESIGN_PHASE3.md §6 acceptance criteria:

  - P3b dev_50 pipeline-only floor: ≥40% correct (architecture-soundness).
  - P3d dev_50 full-routing: ≥50%.
  - P3e dev_100 generalization within ±5pp of dev_50.

This script runs Phase 3's pipeline path over a split, writes the
predictions.jsonl in the official SWE-bench format, calls
``harness.eval.grade`` (the eval-only firewall-allowed channel), and
reports per-strategy correctness:

  - **single-shot (T=0)**: only the first candidate is submitted.
    The honest "pipeline produced one patch, did it work" number.
  - **oracle best-of-K**: submit each candidate independently; count
    the instance correct if ANY candidate resolves it. Upper bound
    on the pipeline path; Phase 5 selection's job is to pick the
    right one out of K.

For each instance:
  1. Load InstanceView + cached Phase 1 reranked top-10 files.
  2. Open Sandbox at base_commit.
  3. Call generate_pipeline (K=2 by default, T=(0, 0.5)).
  4. Catch ContextOversizeError → status=skip-oversize.
  5. Catch other exceptions → status=error.
  6. Write per-instance JSON checkpoint for resume.

After all candidates land, write two predictions.jsonl files (one
"single-shot" with the T=0 candidate, one "oracle" with all K),
call grade() once per file, parse per-instance resolved verdict,
and emit a single audit doc.

Output: docs/audits/<split>_patch_gen_eval.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = PROJECT_ROOT / "splits" / "dev_50.json"


# ---------------------------------------------------------------------------
# Cached rerank loading (same alias map as repro_coverage_eval.py)
# ---------------------------------------------------------------------------


def _retr_cache_dir_for_split(split_path: pathlib.Path) -> pathlib.Path:
    aliases = {
        "dev_50": "v10_dev50_retr_eval",
        "dev_100": "v10_dev_100_retr_eval",
        "test_500": "v10_test_500_retr_eval",
    }
    name = split_path.stem
    if name not in aliases:
        raise SystemExit(
            f"unknown split {name!r}; extend _retr_cache_dir_for_split"
        )
    return PROJECT_ROOT / "runs" / aliases[name] / "checkpoints" / "embed_noshortlist_tb_bs256_rerank"


def _load_ranked_files_from_cache(cache_dir: pathlib.Path, instance_id: str, top_k: int):
    from harness.localization_signals import RankedFile

    p = cache_dir / f"{instance_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    if data.get("error"):
        return None
    raw = data.get("reranked_files") or []
    if not raw:
        return None
    return [
        RankedFile(
            file_path=e["file_path"],
            final_score=float(e.get("final_score", 0.0)),
            rationale=e.get("rationale", ""),
            upstream_signals=tuple(e.get("upstream_signals", ())),
            upstream_best_rank=e.get("upstream_best_rank") or None,
        )
        for e in raw[:top_k]
    ]


# ---------------------------------------------------------------------------
# Per-instance generation
# ---------------------------------------------------------------------------


@dataclass
class _InstanceGenRecord:
    instance_id: str
    repo: str
    status: str                          # "ok" | "skip-oversize" | "skip-no-cache" | "error"
    candidates: list[dict]               # [{candidate_id, diff, model, cost_usd, temperature, tokens_in, tokens_out, duration_s}]
    total_cost_usd: float
    duration_s: float
    error_class: str | None = None
    error_msg: str | None = None

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "status": self.status,
            "candidates": self.candidates,
            "total_cost_usd": self.total_cost_usd,
            "duration_s": self.duration_s,
            "error_class": self.error_class,
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_json(cls, d: dict) -> "_InstanceGenRecord":
        return cls(
            instance_id=d["instance_id"],
            repo=d["repo"],
            status=d["status"],
            candidates=list(d.get("candidates") or []),
            total_cost_usd=float(d["total_cost_usd"]),
            duration_s=float(d["duration_s"]),
            error_class=d.get("error_class"),
            error_msg=d.get("error_msg"),
        )


def _checkpoint_path(out_dir: pathlib.Path, instance_id: str) -> pathlib.Path:
    return out_dir / f"{instance_id}.json"


def _load_checkpoint(out_dir: pathlib.Path, iid: str):
    p = _checkpoint_path(out_dir, iid)
    if not p.exists():
        return None
    try:
        return _InstanceGenRecord.from_json(json.loads(p.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _save_checkpoint(out_dir: pathlib.Path, rec: _InstanceGenRecord) -> None:
    p = _checkpoint_path(out_dir, rec.instance_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec.to_json(), indent=2))


def _generate_for_instance(
    iid: str,
    cache_dir: pathlib.Path,
    temperatures: tuple[float, ...],
    cost_cap_usd: float,
    top_k: int,
) -> _InstanceGenRecord:
    from harness.dataset import load_verified_view
    from harness.patch_gen import (
        ContextOversizeError,
        generate_pipeline,
    )
    from harness.sandbox import Sandbox

    t_start = time.perf_counter()

    try:
        view = load_verified_view(iid)
    except Exception as exc:
        return _InstanceGenRecord(
            instance_id=iid, repo="(unknown)", status="error",
            candidates=[], total_cost_usd=0.0,
            duration_s=time.perf_counter() - t_start,
            error_class=type(exc).__name__, error_msg=str(exc)[:300],
        )

    rfs = _load_ranked_files_from_cache(cache_dir, iid, top_k)
    if rfs is None:
        return _InstanceGenRecord(
            instance_id=iid, repo=view.repo, status="skip-no-cache",
            candidates=[], total_cost_usd=0.0,
            duration_s=time.perf_counter() - t_start,
            error_class="MissingRerankCache",
            error_msg=f"no cached rerank result at {cache_dir}/{iid}.json",
        )

    try:
        with Sandbox(view, max_observation_chars=32_000_000) as sb:
            try:
                result = generate_pipeline(
                    view=view, ranked_files=rfs, sandbox=sb,
                    temperatures=temperatures, cost_cap_usd=cost_cap_usd,
                )
            except ContextOversizeError as exc:
                return _InstanceGenRecord(
                    instance_id=iid, repo=view.repo, status="skip-oversize",
                    candidates=[], total_cost_usd=0.0,
                    duration_s=time.perf_counter() - t_start,
                    error_class="ContextOversizeError",
                    error_msg=str(exc)[:300],
                )
    except Exception as exc:
        return _InstanceGenRecord(
            instance_id=iid, repo=view.repo, status="error",
            candidates=[], total_cost_usd=0.0,
            duration_s=time.perf_counter() - t_start,
            error_class=type(exc).__name__,
            error_msg=("".join(traceback.format_exception_only(type(exc), exc))).strip()[:300],
        )

    cands = [
        {
            "candidate_id": c.candidate_id,
            "diff": c.diff,
            "model": c.generator_model,
            "cost_usd": c.generation_cost_usd,
            "temperature": c.source_temperature,
            "tokens_in": c.generator_input_tokens,
            "tokens_out": c.generator_output_tokens,
            "duration_s": c.duration_s,
        }
        for c in result.candidates
    ]
    return _InstanceGenRecord(
        instance_id=iid, repo=view.repo,
        status="ok" if cands else "error",
        candidates=cands,
        total_cost_usd=result.total_cost_usd,
        duration_s=time.perf_counter() - t_start,
    )


# ---------------------------------------------------------------------------
# Predictions + grading
# ---------------------------------------------------------------------------


def _write_predictions_jsonl(
    records: list[_InstanceGenRecord],
    out_path: pathlib.Path,
    *,
    candidate_index: int | None,
    model_name_or_path: str,
) -> int:
    """Write one row per (instance, candidate) into ``out_path``.

    If ``candidate_index`` is None, write ALL candidates (one prediction
    per (instance, candidate_id)). If an int, write only that index
    (typically 0 = T=0 single-shot).

    The official SWE-bench format expects one prediction per
    instance_id. For oracle-best-of-K we'd need to evaluate each
    candidate independently, which means K separate run_ids. For V0,
    we keep candidate_index=0 only (the honest single-shot number).

    Returns the row count written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fh:
        for rec in records:
            if rec.status != "ok" or not rec.candidates:
                continue
            if candidate_index is not None:
                if candidate_index >= len(rec.candidates):
                    continue
                cand = rec.candidates[candidate_index]
                row = {
                    "instance_id": rec.instance_id,
                    "model_name_or_path": model_name_or_path,
                    "model_patch": cand["diff"],
                }
                fh.write(json.dumps(row) + "\n")
                n += 1
            else:
                # All candidates (one row per): caller MUST split into
                # K separate prediction files for K-way grading.
                for cand in rec.candidates:
                    row = {
                        "instance_id": rec.instance_id,
                        "model_name_or_path": f"{model_name_or_path}_{cand['candidate_id']}",
                        "model_patch": cand["diff"],
                    }
                    fh.write(json.dumps(row) + "\n")
                    n += 1
    return n


def _grade_predictions(predictions_path: pathlib.Path, run_id: str, instance_ids: list[str], max_workers: int) -> pathlib.Path:
    from harness.eval import grade
    return grade(
        predictions_path=predictions_path,
        run_id=run_id,
        instance_ids=instance_ids,
        max_workers=max_workers,
    )


def _parse_resolved(eval_outputs_dir: pathlib.Path, model_name_or_path: str, run_id: str) -> dict[str, bool]:
    """Read the official SWE-bench harness's resolved verdicts.

    The grader writes its top-level summary report at the process
    CWD (NOT under eval_outputs/) with the filename pattern
    ``<model_name_or_path>.v10_<run_id>.json``. The schema is
    ``{"resolved_ids": [...], "unresolved_ids": [...], ...}``.

    We search in three locations to be robust to the grader's
    actual write path:
      1. CWD: ``<model>.v10_<run>.json``
      2. ``eval_outputs/v10_<run_id>/`` (the report_dir we pass)
      3. ``logs/run_evaluation/v10_<run_id>/`` (the SWE-bench harness'
         own cache layout)
    """
    resolved: dict[str, bool] = {}
    candidates: list[pathlib.Path] = []
    cwd = pathlib.Path.cwd()
    # 1. CWD top-level summary
    candidates.extend(cwd.glob(f"{model_name_or_path}.v10_{run_id}.json"))
    candidates.extend(cwd.glob(f"*v10_{run_id}.json"))
    # 2. report_dir
    if eval_outputs_dir.exists():
        candidates.extend(eval_outputs_dir.rglob("*.json"))
    # 3. logs/run_evaluation/ — the harness' own cache
    logs_dir = pathlib.Path("logs") / "run_evaluation" / f"v10_{run_id}"
    if logs_dir.exists():
        candidates.extend(logs_dir.rglob("*.json"))
    # Deduplicate (set semantics) and parse.
    seen: set[pathlib.Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "resolved_ids" in data:
            for iid in data.get("resolved_ids", []):
                resolved[iid] = True
            for iid in data.get("unresolved_ids", []):
                resolved.setdefault(iid, False)
            for iid in data.get("error_ids", []):
                resolved.setdefault(iid, False)
    return resolved


# ---------------------------------------------------------------------------
# Audit doc
# ---------------------------------------------------------------------------


def _summary_md(
    records: list[_InstanceGenRecord],
    resolved: dict[str, bool],
    split_path: pathlib.Path,
    temperatures: tuple[float, ...],
    cost_cap_usd: float,
    wall_clock_s: float,
    *,
    candidate_index: int,
    floor_pct: float,
) -> str:
    n = len(records)
    if n == 0:
        return f"# {split_path.stem} patch gen eval\n\nNo records.\n"

    status_counts = Counter(r.status for r in records)
    ok = status_counts.get("ok", 0)
    skip_oversize = status_counts.get("skip-oversize", 0)
    skip_no_cache = status_counts.get("skip-no-cache", 0)
    errored = status_counts.get("error", 0)

    n_resolved = sum(1 for v in resolved.values() if v)
    n_attempted = sum(1 for r in records if r.status == "ok" and r.candidates)
    pct_of_n = 100.0 * n_resolved / n if n else 0.0
    pct_of_attempted = 100.0 * n_resolved / n_attempted if n_attempted else 0.0

    gate = "PASS" if pct_of_n >= floor_pct else (
        "WARN" if pct_of_n >= max(0.0, floor_pct - 10) else "FAIL"
    )

    costs = [r.total_cost_usd for r in records]
    total_cost = sum(costs)
    p50_cost = statistics.median(costs) if costs else 0.0
    p90_cost = statistics.quantiles(costs, n=10)[8] if len(costs) >= 10 else max(costs, default=0.0)

    durs = [r.duration_s for r in records]
    p50_dur = statistics.median(durs) if durs else 0.0
    p90_dur = statistics.quantiles(durs, n=10)[8] if len(durs) >= 10 else max(durs, default=0.0)

    by_repo_total: Counter = Counter()
    by_repo_resolved: Counter = Counter()
    for r in records:
        by_repo_total[r.repo] += 1
        if resolved.get(r.instance_id, False):
            by_repo_resolved[r.repo] += 1

    parts: list[str] = []
    parts.append(f"# {split_path.stem} patch generation eval — Phase 3 commit P3b\n\n")
    parts.append(
        f"Generated by `scripts/patch_gen_eval.py`. Pipeline-path only "
        f"(K=len(temperatures)). Per `docs/V10_DESIGN_PHASE3.md` §6 "
        f"acceptance gate: pipeline-only correct-patch rate must reach "
        f"**≥{floor_pct:.0f}%** as the architecture-soundness floor.\n\n"
    )
    parts.append(f"- **Split:** `{split_path.relative_to(PROJECT_ROOT)}` ({n} instances)\n")
    parts.append(f"- **Model:** deepseek-chat (per `harness/config/models.yaml`)\n")
    parts.append(f"- **Temperatures (K):** {temperatures}\n")
    parts.append(f"- **Submitted candidate:** index={candidate_index} ({'T=' + str(temperatures[candidate_index]) if candidate_index < len(temperatures) else 'oracle-K'})\n")
    parts.append(f"- **Cost cap per instance:** ${cost_cap_usd:.2f}\n")
    parts.append(f"- **Wall-clock total (gen):** {wall_clock_s/60:.1f} min\n\n")

    parts.append("## Headline\n\n")
    parts.append(
        f"| Metric | Count | Pct |\n|---|---|---|\n"
        f"| Resolved (correct patch) | {n_resolved}/{n} | {pct_of_n:.1f}% |\n"
        f"| Resolved (of submitted only) | {n_resolved}/{n_attempted} | {pct_of_attempted:.1f}% |\n"
    )
    parts.append("\n")
    parts.append(f"**Acceptance gate (≥{floor_pct:.0f}% of total): {gate}.**\n\n")

    parts.append("## Generation status\n\n")
    parts.append(f"| Status | Count | Pct |\n|---|---|---|\n")
    for label, count in [
        ("ok", ok), ("skip-oversize", skip_oversize),
        ("skip-no-cache", skip_no_cache), ("error", errored),
    ]:
        parts.append(f"| `{label}` | {count} | {100.0*count/n:.1f}% |\n")
    parts.append("\n")

    parts.append("## Cost\n\n")
    parts.append(f"- **Total LLM spend:** ${total_cost:.2f}\n")
    parts.append(f"- **Per-instance:** p50 ${p50_cost:.4f}, p90 ${p90_cost:.4f}, max ${max(costs, default=0):.4f}\n")
    parts.append(f"- **Per-instance gen cap:** ${cost_cap_usd:.2f}\n\n")

    parts.append("## Wall-clock (per instance, gen only — excludes grading)\n\n")
    parts.append(f"- p50 {p50_dur:.1f}s, p90 {p90_dur:.1f}s, max {max(durs, default=0):.1f}s\n\n")

    parts.append("## Per-repo breakdown\n\n")
    parts.append("| Repo | Instances | Resolved | Pct |\n|---|---|---|---|\n")
    for repo in sorted(by_repo_total):
        tot = by_repo_total[repo]
        res = by_repo_resolved[repo]
        parts.append(f"| {repo} | {tot} | {res} | {100.0*res/tot:.1f}% |\n")
    parts.append("\n")

    parts.append("## Per-instance detail\n\n")
    parts.append("| Instance | Repo | Status | Candidates | Cost $ | Resolved? |\n|---|---|---|---|---|---|\n")
    for r in sorted(records, key=lambda x: (x.repo, x.instance_id)):
        n_c = len(r.candidates)
        res_str = "✓" if resolved.get(r.instance_id) else ("—" if r.status != "ok" else "✗")
        parts.append(
            f"| `{r.instance_id}` | {r.repo} | `{r.status}` | {n_c} | "
            f"{r.total_cost_usd:.4f} | {res_str} |\n"
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_split_iids(split_path: pathlib.Path) -> list[str]:
    data = json.loads(split_path.read_text())
    out: list[str] = []
    raw = data.get("instances") or data.get("instance_ids") or data
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                out.append(entry)
            elif isinstance(entry, dict) and entry.get("instance_id"):
                out.append(entry["instance_id"])
    if not out:
        raise SystemExit(f"no instance_ids in {split_path}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=str(DEFAULT_SPLIT))
    p.add_argument("--temperatures", default="0.0,0.5",
                   help="comma-sep list, default '0.0,0.5'")
    p.add_argument("--cost-cap-usd", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--instance-id", action="append", default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--gen-only", action="store_true",
                   help="generate candidates but skip grading")
    p.add_argument("--grade-only", action="store_true",
                   help="skip generation; just grade existing checkpoints")
    p.add_argument("--grader-workers", type=int, default=4)
    p.add_argument("--candidate-index", type=int, default=0,
                   help="which K-candidate to submit; 0 = T=0 single-shot")
    p.add_argument("--floor-pct", type=float, default=40.0,
                   help="P3b pipeline-only acceptance floor; default 40%")
    p.add_argument("--run-id-suffix", default="patch_gen_eval")
    p.add_argument("--audit-out", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    split_path = pathlib.Path(args.split).resolve()
    instance_ids = _load_split_iids(split_path)
    if args.instance_id:
        wanted = set(args.instance_id)
        instance_ids = [iid for iid in instance_ids if iid in wanted]
    if args.limit:
        instance_ids = instance_ids[: args.limit]

    temperatures = tuple(float(x) for x in args.temperatures.split(","))

    out_dir = PROJECT_ROOT / "runs" / f"v10_{split_path.stem}_{args.run_id_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_out = pathlib.Path(args.audit_out) if args.audit_out else PROJECT_ROOT / "docs" / "audits" / f"{split_path.stem}_{args.run_id_suffix}.md"

    cache_dir = _retr_cache_dir_for_split(split_path)
    if not cache_dir.exists():
        raise SystemExit(f"retrieval cache missing at {cache_dir}")

    # ---- Generation ----
    records: list[_InstanceGenRecord] = []
    t_wall = time.perf_counter()

    if not args.grade_only:
        for i, iid in enumerate(instance_ids, start=1):
            cached = None if args.reset else _load_checkpoint(out_dir, iid)
            if cached is not None:
                records.append(cached)
                print(f"[{i}/{len(instance_ids)}] {iid:60s} cached: {cached.status} "
                      f"(K={len(cached.candidates)}, cost=${cached.total_cost_usd:.4f})", flush=True)
                continue
            try:
                rec = _generate_for_instance(
                    iid=iid, cache_dir=cache_dir,
                    temperatures=temperatures,
                    cost_cap_usd=args.cost_cap_usd,
                    top_k=args.top_k,
                )
            except KeyboardInterrupt:
                print("[interrupt] writing partial audit and exiting", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                rec = _InstanceGenRecord(
                    instance_id=iid, repo="(unknown)", status="error",
                    candidates=[], total_cost_usd=0.0,
                    duration_s=0.0,
                    error_class=type(exc).__name__, error_msg=str(exc)[:300],
                )
            _save_checkpoint(out_dir, rec)
            records.append(rec)
            print(
                f"[{i}/{len(instance_ids)}] {iid:60s} status={rec.status:14s} "
                f"K={len(rec.candidates)} cost=${rec.total_cost_usd:.4f} dur={rec.duration_s:.1f}s",
                flush=True,
            )
    else:
        # Load all checkpoints
        for iid in instance_ids:
            rec = _load_checkpoint(out_dir, iid)
            if rec is not None:
                records.append(rec)

    wall_clock_s = time.perf_counter() - t_wall

    # ---- Grading ----
    resolved: dict[str, bool] = {}
    if not args.gen_only:
        run_id = f"{split_path.stem}_{args.run_id_suffix}"
        model_name_or_path = f"v10_{run_id}"
        preds_path = out_dir / "predictions.jsonl"
        n_rows = _write_predictions_jsonl(
            records, preds_path,
            candidate_index=args.candidate_index,
            model_name_or_path=model_name_or_path,
        )
        if n_rows == 0:
            print("[grade] no candidates to grade; skipping grader", flush=True)
        else:
            ok_iids = [r.instance_id for r in records if r.status == "ok" and r.candidates and (args.candidate_index is None or args.candidate_index < len(r.candidates))]
            print(f"[grade] {n_rows} predictions to grade; calling SWE-bench harness…", flush=True)
            try:
                eval_outputs_dir = _grade_predictions(
                    preds_path, run_id, ok_iids, max_workers=args.grader_workers,
                )
            except Exception as exc:
                print(f"[grade] FAILED: {type(exc).__name__}: {exc}", flush=True)
                eval_outputs_dir = None
            if eval_outputs_dir:
                resolved = _parse_resolved(eval_outputs_dir, model_name_or_path, run_id)
                print(f"[grade] eval_outputs at {eval_outputs_dir}: "
                      f"{sum(1 for v in resolved.values() if v)}/{len(resolved)} resolved", flush=True)

    # ---- Audit ----
    md = _summary_md(
        records, resolved, split_path, temperatures, args.cost_cap_usd,
        wall_clock_s,
        candidate_index=args.candidate_index,
        floor_pct=args.floor_pct,
    )
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(md)
    print(f"[audit] {audit_out.relative_to(PROJECT_ROOT)}", flush=True)

    n_resolved = sum(1 for v in resolved.values() if v)
    n = len(records)
    pct = 100.0 * n_resolved / n if n else 0.0
    total_cost = sum(r.total_cost_usd for r in records)
    print(
        f"[summary] resolved={n_resolved}/{n} ({pct:.1f}%)  "
        f"total_cost=${total_cost:.2f}  wall={wall_clock_s/60:.1f}m",
        flush=True,
    )
    if pct >= args.floor_pct:
        print(f"[gate] PASS (>= {args.floor_pct:.0f}%)", flush=True)
        return 0
    if pct >= max(0.0, args.floor_pct - 10):
        print(f"[gate] WARN ({pct:.1f}% in [{args.floor_pct-10:.0f}, {args.floor_pct:.0f}))", flush=True)
        return 0
    print(f"[gate] FAIL (< {args.floor_pct-10:.0f}%); architecture rethink required", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 4 P4c-light — validator dry-run over Phase 3's K=1 candidates.

Loads the dev_100 (or any split) routed-eval checkpoints, opens a
Sandbox per instance, runs ``harness.validation.validate_candidate``,
writes the ValidationResult to disk, and emits a confusion-matrix
audit comparing the validator's "good/broken" verdict to the
grader's resolved/unresolved set.

NO LLM CALLS. Validator wall is bounded by the public test suite
(default 480s/run, with a baseline cache so K candidates per instance
share one baseline run — at K=1 the cache doesn't help).

Usage:
    python scripts/validation_eval.py \\
        --candidates-dir runs/v10_dev_100_routed \\
        --grader-cwd-report v10_dev_100_routed.v10_dev_100_routed.json \\
        --out-dir runs/v10_dev_100_validated \\
        --audit-out docs/audits/dev_100_validated.md
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
from dataclasses import asdict, dataclass


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Loading + saving
# ---------------------------------------------------------------------------


def _load_resolved_ids(report_paths: list[pathlib.Path]) -> set[str]:
    """Read the SWE-bench harness's top-level summary report(s) and
    collect every resolved_ids entry. The grader writes this file at
    CWD with the pattern <model>.<run_id>.json."""
    out: set[str] = set()
    for p in report_paths:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            for iid in data.get("resolved_ids", []):
                out.add(iid)
    return out


def _load_candidate_records(candidates_dir: pathlib.Path) -> list[dict]:
    """Load all per-instance JSON checkpoints from the routed eval."""
    out: list[dict] = []
    for p in sorted(candidates_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Per-instance validation
# ---------------------------------------------------------------------------


@dataclass
class _ValidationCheckpoint:
    instance_id: str
    repo: str
    skipped_no_candidate: bool
    apply_clean: bool | None
    suite_ran_at_base: bool | None
    new_failures_count: int | None
    new_passes_count: int | None
    repro_signal_status: str | None
    static_ruff_clean: bool | None
    static_mypy_clean: bool | None
    cluster_id: str
    diff_files: list[str]
    diff_additions: int
    diff_deletions: int
    notes: str
    duration_s: float
    error_class: str | None = None
    error_msg: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def _run_one(rec: dict, out_dir: pathlib.Path, suite_timeout_s: int) -> _ValidationCheckpoint:
    iid = rec["instance_id"]
    repo = rec.get("repo", "(unknown)")
    cands = rec.get("candidates") or []
    if not cands:
        return _ValidationCheckpoint(
            instance_id=iid, repo=repo, skipped_no_candidate=True,
            apply_clean=None, suite_ran_at_base=None,
            new_failures_count=None, new_passes_count=None,
            repro_signal_status=None, static_ruff_clean=None,
            static_mypy_clean=None, cluster_id="", diff_files=[],
            diff_additions=0, diff_deletions=0, notes="no_candidate",
            duration_s=0.0,
        )

    cand_dict = cands[0]

    t_start = time.perf_counter()
    try:
        from harness.dataset import load_verified_view
        from harness.patch_gen.views import PatchCandidate
        from harness.sandbox import Sandbox
        from harness.validation import validate_candidate
    except Exception as exc:  # noqa: BLE001
        return _ValidationCheckpoint(
            instance_id=iid, repo=repo, skipped_no_candidate=False,
            apply_clean=None, suite_ran_at_base=None,
            new_failures_count=None, new_passes_count=None,
            repro_signal_status=None, static_ruff_clean=None,
            static_mypy_clean=None, cluster_id="", diff_files=[],
            diff_additions=0, diff_deletions=0,
            notes="import_error", duration_s=0.0,
            error_class=type(exc).__name__, error_msg=str(exc)[:300],
        )

    try:
        view = load_verified_view(iid)
        cand = PatchCandidate(
            instance_id=iid,
            candidate_id=cand_dict["candidate_id"],
            diff=cand_dict["diff"],
            source_route="agent" if "agent" in cand_dict["candidate_id"] else "pipeline",
            source_temperature=cand_dict.get("temperature"),
            source_attempt_index=0,
            generator_model=cand_dict.get("model", "deepseek-chat"),
            generator_input_tokens=int(cand_dict.get("tokens_in", 0)),
            generator_output_tokens=int(cand_dict.get("tokens_out", 0)),
            generation_cost_usd=float(cand_dict.get("cost_usd", 0.0)),
            duration_s=float(cand_dict.get("duration_s", 0.0)),
        )
    except Exception as exc:  # noqa: BLE001
        return _ValidationCheckpoint(
            instance_id=iid, repo=repo, skipped_no_candidate=False,
            apply_clean=None, suite_ran_at_base=None,
            new_failures_count=None, new_passes_count=None,
            repro_signal_status=None, static_ruff_clean=None,
            static_mypy_clean=None, cluster_id="", diff_files=[],
            diff_additions=0, diff_deletions=0,
            notes="candidate_construction_error",
            duration_s=time.perf_counter() - t_start,
            error_class=type(exc).__name__, error_msg=str(exc)[:300],
        )

    try:
        with Sandbox(view, max_observation_chars=32_000_000) as sb:
            result = validate_candidate(
                view=view, sandbox=sb, candidate=cand,
                repro_signal=None,  # P4c-light: no Phase 2 repro for dev_100
                run_dir=out_dir,
                suite_timeout_s=suite_timeout_s,
                static_timeout_s=60,
                skip_static=False,
            )
    except Exception as exc:  # noqa: BLE001
        return _ValidationCheckpoint(
            instance_id=iid, repo=repo, skipped_no_candidate=False,
            apply_clean=None, suite_ran_at_base=None,
            new_failures_count=None, new_passes_count=None,
            repro_signal_status=None, static_ruff_clean=None,
            static_mypy_clean=None, cluster_id="", diff_files=[],
            diff_additions=0, diff_deletions=0,
            notes="validate_candidate_raised",
            duration_s=time.perf_counter() - t_start,
            error_class=type(exc).__name__,
            error_msg=("".join(traceback.format_exception_only(type(exc), exc))).strip()[:300],
        )

    cv = result.candidate_view
    return _ValidationCheckpoint(
        instance_id=iid, repo=repo, skipped_no_candidate=False,
        apply_clean=result.apply_result.applied,
        suite_ran_at_base=cv.public_suite_signal.suite_ran_at_base,
        new_failures_count=cv.public_suite_signal.new_failures_count,
        new_passes_count=cv.public_suite_signal.new_passes_count,
        repro_signal_status=(
            cv.repro_signal.status if cv.repro_signal is not None else None
        ),
        static_ruff_clean=cv.static_signal.ruff_clean,
        static_mypy_clean=cv.static_signal.mypy_clean,
        cluster_id=cv.cluster_id,
        diff_files=list(cv.diff_stats.files_touched),
        diff_additions=cv.diff_stats.additions,
        diff_deletions=cv.diff_stats.deletions,
        notes=cv.notes,
        duration_s=time.perf_counter() - t_start,
    )


# ---------------------------------------------------------------------------
# Audit doc
# ---------------------------------------------------------------------------


def _validator_says_good(c: _ValidationCheckpoint) -> bool:
    """Validator's verdict: candidate is "likely good" iff it applied
    cleanly AND the public suite ran AND no new failures were
    introduced. Repro signal is N/A (None) for the dev_100 dry-run.
    """
    return bool(
        c.apply_clean is True
        and c.suite_ran_at_base is True
        and (c.new_failures_count or 0) == 0
    )


def _summary_md(
    checkpoints: list[_ValidationCheckpoint],
    resolved_ids: set[str],
    candidates_dir: pathlib.Path,
    wall_clock_s: float,
) -> str:
    n_total = len(checkpoints)
    n_no_cand = sum(1 for c in checkpoints if c.skipped_no_candidate)
    validated = [c for c in checkpoints if not c.skipped_no_candidate]
    n_v = len(validated)

    n_apply = sum(1 for c in validated if c.apply_clean is True)
    n_apply_fail = sum(1 for c in validated if c.apply_clean is False)
    n_apply_unknown = sum(1 for c in validated if c.apply_clean is None)

    n_suite_ran = sum(1 for c in validated if c.suite_ran_at_base is True)
    n_no_regress = sum(
        1 for c in validated
        if c.suite_ran_at_base is True and (c.new_failures_count or 0) == 0
    )

    n_ruff_clean = sum(1 for c in validated if c.static_ruff_clean is True)
    n_ruff_fail = sum(1 for c in validated if c.static_ruff_clean is False)
    n_ruff_unconfigured = sum(1 for c in validated if c.static_ruff_clean is None)

    n_mypy_clean = sum(1 for c in validated if c.static_mypy_clean is True)
    n_mypy_fail = sum(1 for c in validated if c.static_mypy_clean is False)
    n_mypy_unconfigured = sum(1 for c in validated if c.static_mypy_clean is None)

    # Confusion matrix vs grader resolved
    tp = fp = tn = fn = 0
    for c in validated:
        validator_good = _validator_says_good(c)
        grader_resolved = c.instance_id in resolved_ids
        if validator_good and grader_resolved:
            tp += 1
        elif validator_good and not grader_resolved:
            fp += 1
        elif not validator_good and grader_resolved:
            fn += 1
        else:
            tn += 1
    total_cm = tp + fp + tn + fn
    accuracy = (tp + tn) / total_cm if total_cm else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    durs = [c.duration_s for c in validated]
    p50 = statistics.median(durs) if durs else 0.0
    p90 = statistics.quantiles(durs, n=10)[8] if len(durs) >= 10 else max(durs, default=0.0)

    by_repo_total: Counter = Counter()
    by_repo_apply: Counter = Counter()
    for c in validated:
        by_repo_total[c.repo] += 1
        if c.apply_clean is True:
            by_repo_apply[c.repo] += 1

    parts: list[str] = []
    parts.append(f"# {candidates_dir.name} validator dry-run — Phase 4 P4c-light\n\n")
    parts.append(
        f"Generated by `scripts/validation_eval.py`. Validates each "
        f"K=1 candidate from Phase 3's routed eval against the public "
        f"test suite + static analysis. **NO LLM CALLS.** Sanity-check "
        f"only — no acceptance gate.\n\n"
    )
    parts.append(f"- **Source candidates:** `{candidates_dir.relative_to(PROJECT_ROOT)}` ({n_total} instances)\n")
    parts.append(f"- **Skipped (no candidate):** {n_no_cand}\n")
    parts.append(f"- **Validated:** {n_v}\n")
    parts.append(f"- **Wall-clock:** {wall_clock_s/60:.1f} min\n\n")

    parts.append("## Per-signal distribution (validated only)\n\n")
    parts.append("| Signal | True | False | Unknown/N/A |\n|---|---|---|---|\n")
    parts.append(f"| `apply_clean` | {n_apply} | {n_apply_fail} | {n_apply_unknown} |\n")
    parts.append(f"| `suite_ran_at_base` | {n_suite_ran} | {n_v - n_suite_ran} | 0 |\n")
    parts.append(f"| `no_regression` | {n_no_regress} | {n_v - n_no_regress} | 0 |\n")
    parts.append(f"| `static_ruff_clean` | {n_ruff_clean} | {n_ruff_fail} | {n_ruff_unconfigured} |\n")
    parts.append(f"| `static_mypy_clean` | {n_mypy_clean} | {n_mypy_fail} | {n_mypy_unconfigured} |\n\n")

    parts.append("## Confusion matrix vs grader\n\n")
    parts.append(
        "Validator says \"good\" = `apply_clean ∧ suite_ran ∧ no_regression`. "
        "Grader says \"resolved\" = SWE-bench harness verdict.\n\n"
    )
    parts.append("| | Grader: resolved | Grader: unresolved |\n|---|---|---|\n")
    parts.append(f"| Validator: good   | TP={tp} | FP={fp} |\n")
    parts.append(f"| Validator: broken | FN={fn} | TN={tn} |\n\n")
    parts.append(
        f"- Accuracy:  {accuracy:.1%} ({tp+tn}/{total_cm})\n"
        f"- Precision: {precision:.1%} ({tp}/{tp+fp if tp+fp else 0})\n"
        f"- Recall:    {recall:.1%} ({tp}/{tp+fn if tp+fn else 0})\n\n"
    )

    parts.append("## Wall-clock per validated instance\n\n")
    parts.append(f"- p50 {p50:.1f}s, p90 {p90:.1f}s, max {max(durs, default=0):.1f}s\n\n")

    parts.append("## Per-repo apply rates\n\n")
    parts.append("| Repo | Validated | Apply clean | Pct |\n|---|---|---|---|\n")
    for repo in sorted(by_repo_total):
        tot = by_repo_total[repo]
        ok = by_repo_apply[repo]
        parts.append(f"| {repo} | {tot} | {ok} | {100.0*ok/tot:.0f}% |\n")
    parts.append("\n")

    parts.append("## Per-instance detail (truncated rows for readability)\n\n")
    parts.append("| Instance | Repo | apply | suite | new_fail | static_ruff | static_mypy | Resolved? |\n|---|---|---|---|---|---|---|---|\n")
    for c in sorted(checkpoints, key=lambda x: (x.repo, x.instance_id)):
        if c.skipped_no_candidate:
            row = (
                f"| `{c.instance_id}` | {c.repo} | — | — | — | — | — | "
                f"{'✓' if c.instance_id in resolved_ids else '—'} |\n"
            )
        else:
            row = (
                f"| `{c.instance_id}` | {c.repo} | "
                f"{'✓' if c.apply_clean else '✗'} | "
                f"{'✓' if c.suite_ran_at_base else '✗'} | "
                f"{c.new_failures_count if c.new_failures_count is not None else '?'} | "
                f"{'✓' if c.static_ruff_clean is True else ('✗' if c.static_ruff_clean is False else 'n/a')} | "
                f"{'✓' if c.static_mypy_clean is True else ('✗' if c.static_mypy_clean is False else 'n/a')} | "
                f"{'✓' if c.instance_id in resolved_ids else ' '} |\n"
            )
        parts.append(row)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates-dir", default="runs/v10_dev_100_routed",
                   help="dir with per-instance JSON checkpoints from a routed run")
    p.add_argument("--grader-cwd-report", default="v10_dev_100_routed.v10_dev_100_routed.json")
    p.add_argument("--grader-eval-outputs", default="eval_outputs/v10_dev_100_routed")
    p.add_argument("--out-dir", default="runs/v10_dev_100_validated")
    p.add_argument("--audit-out", default="docs/audits/dev_100_validated.md")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--suite-timeout-s", type=int, default=480)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    candidates_dir = pathlib.Path(args.candidates_dir).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    audit_out = pathlib.Path(args.audit_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)

    # Resolved set from grader
    grader_paths = [pathlib.Path(args.grader_cwd_report)]
    eval_outputs_dir = pathlib.Path(args.grader_eval_outputs)
    if eval_outputs_dir.exists():
        grader_paths.extend(eval_outputs_dir.rglob("*.json"))
    resolved_ids = _load_resolved_ids(grader_paths)
    print(f"[grader] {len(resolved_ids)} resolved instance ids", flush=True)

    # Load candidate records
    records = _load_candidate_records(candidates_dir)
    if args.limit:
        records = records[: args.limit]
    print(f"[validator] {len(records)} candidate checkpoints", flush=True)

    checkpoints: list[_ValidationCheckpoint] = []
    t_wall_start = time.perf_counter()

    for i, rec in enumerate(records, start=1):
        iid = rec["instance_id"]
        ckpt_path = out_dir / f"{iid}.json"
        if not args.reset and ckpt_path.exists():
            try:
                cached = json.loads(ckpt_path.read_text())
                cp = _ValidationCheckpoint(**cached)
                checkpoints.append(cp)
                print(f"[{i}/{len(records)}] {iid:60s} cached", flush=True)
                continue
            except Exception:
                pass
        try:
            cp = _run_one(rec, out_dir, args.suite_timeout_s)
        except KeyboardInterrupt:
            print("[interrupt] writing partial audit", flush=True)
            break
        ckpt_path.write_text(json.dumps(cp.to_json(), indent=2))
        checkpoints.append(cp)
        print(
            f"[{i}/{len(records)}] {iid:60s} "
            f"apply={cp.apply_clean} suite={cp.suite_ran_at_base} "
            f"new_fail={cp.new_failures_count} dur={cp.duration_s:.1f}s",
            flush=True,
        )

    wall_clock_s = time.perf_counter() - t_wall_start
    md = _summary_md(checkpoints, resolved_ids, candidates_dir, wall_clock_s)
    audit_out.write_text(md)
    print(f"[audit] {audit_out.relative_to(PROJECT_ROOT)}", flush=True)

    n_validated = sum(1 for c in checkpoints if not c.skipped_no_candidate)
    n_apply = sum(1 for c in checkpoints if c.apply_clean is True)
    print(f"[summary] validated={n_validated} apply_clean={n_apply} wall={wall_clock_s/60:.1f}m", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

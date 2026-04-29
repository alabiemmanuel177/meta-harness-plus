"""Analyze V7 per-instance behavior to produce a difficulty prior for
each instance. Output: splits/v7_difficulty_priors.json.

Per V10_DESIGN.md §13.3: V7's success/failure pattern is OUR observation
about OUR runs — legitimate, oracle-free signal. Specifically:

  - easy   = V7 resolved cleanly: resolved=True, n_turns <= 15,
             stop_reason='done', no apparent retry struggle.
  - hard   = V7 failed: resolved=False. V7 had FAIL_TO_PASS in the
             actor prompt; if V7 couldn't fix it WITH that leak, V10
             without it will struggle more. Most reliable signal in
             this analysis.
  - medium = everything else: V7 resolved but with effort indicators
             (high n_turns, non-'done' stop, high cost, large patch).
  - unknown = instances missing a V7 trajectory.

The output is read by ``scripts/build_dev_split.py`` (later: by the
dev-50 stratifier) and by ablation scripts. Production routing logic
at inference does NOT consume this file directly — it re-derives
difficulty from issue features (length, traceback, candidate-file
count) and uses these priors only as calibration ground truth on
dev-50.
"""

from __future__ import annotations

import collections
import json
import pathlib
import statistics


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAJ_ROOT = PROJECT_ROOT / "runs" / "swebench_500_v7" / "cache" / "trajectories_v7"
V7_FINAL = PROJECT_ROOT / "runs" / "swebench_500_v7" / "v7_full500_FINAL.json"
OUT_PATH = PROJECT_ROOT / "splits" / "v7_difficulty_priors.json"


# Effort thresholds. Calibrated against the actual V7 turn-count
# distribution: resolved-instance n_turns has p25=24, median=34, p75=44.
# Setting EASY_TURN_THRESHOLD = 25 catches the bottom-quartile-by-turns
# resolved instances, ~99/500 = 20% easy. This is a coarse signal and
# Phase 1 will recalibrate via dev-50 sweeps.
EASY_TURN_THRESHOLD = 25


def _classify(traj: dict) -> tuple[str, str]:
    """Return (difficulty, signal_basis_label) for one instance."""
    resolved = traj.get("resolved")
    if resolved is None:
        # Trajectory present but eval result missing (e.g., 33 missing
        # eval reports from commit 12). Trust the trajectory's stop_reason.
        if traj.get("stop_reason") == "done":
            return "unknown_traj_done_no_report", "v7_no_report"
        return "unknown_traj_incomplete", "v7_incomplete"

    if resolved is False:
        return "hard", "v7_unresolved"

    # resolved is True — sub-classify by effort. Turn count is the
    # cleanest signal (cost/wall correlate with turns; patch_len is
    # noisier — large correct patches happen).
    n_turns = int(traj.get("n_turns", 0))
    stop_reason = traj.get("stop_reason", "")
    if n_turns <= EASY_TURN_THRESHOLD and stop_reason == "done":
        return "easy", "v7_resolved_clean"
    return "medium", "v7_resolved_with_effort"


def main() -> int:
    if not V7_FINAL.exists():
        print(f"FAIL: V7 final results not found at {V7_FINAL}")
        return 2
    final = json.loads(V7_FINAL.read_text())
    per_instance = final["per_instance"]

    priors: dict[str, dict] = {}
    by_difficulty = collections.Counter()
    for iid, entry in per_instance.items():
        # Try to read this instance's trajectory.
        traj_path = TRAJ_ROOT / f"{iid}_seed0.json"
        if not traj_path.exists():
            # Fallback: classify purely from V7 final's per-instance entry.
            resolved = entry.get("resolved")
            if resolved is False:
                difficulty, basis = "hard", "v7_unresolved_no_traj"
            elif resolved is True:
                difficulty, basis = "medium", "v7_resolved_no_traj"
            else:
                difficulty, basis = "unknown", "no_traj_no_report"
            priors[iid] = {
                "estimated_difficulty": difficulty,
                "signal_basis": basis,
                "n_turns": None,
                "usd": None,
                "task_wall_s": None,
            }
            by_difficulty[difficulty] += 1
            continue

        try:
            traj = json.loads(traj_path.read_text())
        except Exception as exc:
            priors[iid] = {
                "estimated_difficulty": "unknown",
                "signal_basis": f"traj_load_error:{type(exc).__name__}",
                "n_turns": None,
                "usd": None,
                "task_wall_s": None,
            }
            by_difficulty["unknown"] += 1
            continue

        # Merge trajectory and per_instance verdict.
        traj_with_verdict = dict(traj)
        if "resolved" not in traj_with_verdict:
            traj_with_verdict["resolved"] = entry.get("resolved")
        difficulty, basis = _classify(traj_with_verdict)
        priors[iid] = {
            "estimated_difficulty": difficulty,
            "signal_basis": basis,
            "n_turns": int(traj.get("n_turns", 0)),
            "usd": round(float(traj.get("usd", 0.0) or 0.0), 4),
            "task_wall_s": round(float(traj.get("task_wall_s", 0.0) or 0.0), 1),
        }
        by_difficulty[difficulty] += 1

    out = {
        "version": "v10-v7-priors-r1",
        "description": (
            "Per-instance difficulty estimates derived from V7's "
            "behavior. NOT to be consumed by production routing — "
            "for dev-50 calibration only. See V10_DESIGN.md §13.3."
        ),
        "thresholds": {
            "easy_turn_threshold": EASY_TURN_THRESHOLD,
            "easy_stop_reason_required": "done",
        },
        "summary": dict(by_difficulty),
        "priors": priors,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[priors] wrote {OUT_PATH.relative_to(PROJECT_ROOT)} with {len(priors)} priors")
    print(f"[priors] summary: {dict(by_difficulty)}")
    # Diagnostics
    by_diff_resolved = {
        d: sum(1 for p in priors.values() if p["estimated_difficulty"] == d)
        for d in ("easy", "medium", "hard", "unknown",
                  "unknown_traj_done_no_report", "unknown_traj_incomplete")
    }
    print(f"[priors] by_difficulty: {by_diff_resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

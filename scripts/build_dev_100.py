"""Construct splits/dev_100.json — held out from dev_50 with zero
overlap, repo-balanced, difficulty-balanced via V7 priors.

Per the V10_DESIGN.md split discipline: dev-50 is for component sweeps
(iterate freely), dev-100 is for decision-making (touched once after
component-tuning is locked), test-500 is the headline run (touched
once). Zero-overlap between dev-50 and dev-100 is a hard rule.

Selection process:
  1. Load splits/dev_50.json → exclude its 50 instance_ids.
  2. Load splits/v7_difficulty_priors.json as a calibration target
     (NOT a constraint — V10 production routing re-derives difficulty
     from issue features, not these priors).
  3. Apply REPO_QUOTA (≈ 2x dev-50, adjusted for repos that ran out
     of available instances after the dev-50 carveout).
  4. Within each repo, target a 25:45:30 easy:medium:hard mix using
     the V7 prior labels.
  5. Stable sort by instance_id within (repo, difficulty) for
     determinism.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
DEV_50 = PROJECT_ROOT / "splits" / "dev_50.json"
PRIORS = PROJECT_ROOT / "splits" / "v7_difficulty_priors.json"
OUT = PROJECT_ROOT / "splits" / "dev_100.json"


# 2× dev-50 quotas where the repo has the headroom; reduced where
# Verified runs out (flask: only 1 instance in Verified, already in
# dev_50; seaborn: only 2, 1 in dev_50). Surplus reallocated to
# django (most instances available in Verified).
REPO_QUOTA: dict[str, int] = {
    "django/django":              26,  # 24 from 2x + 2 from flask/seaborn shortfall
    "sympy/sympy":                14,
    "sphinx-doc/sphinx":          10,
    "matplotlib/matplotlib":      10,
    "scikit-learn/scikit-learn":  10,
    "astropy/astropy":             8,
    "pydata/xarray":               6,
    "pytest-dev/pytest":           6,
    "pylint-dev/pylint":           4,
    "psf/requests":                4,
    "mwaskom/seaborn":             1,
    "pallets/flask":               0,  # only 1 instance in Verified, in dev_50
    # Sum: 99. One spare slot → bumped django to 26 above? No, 26+14+10+10+10+8+6+6+4+4+1+0 = 99
}
# fix sum to 100 by adding one more to a large-enough repo
REPO_QUOTA["django/django"] = 27
assert sum(REPO_QUOTA.values()) == 100, f"REPO_QUOTA must sum to 100, got {sum(REPO_QUOTA.values())}"

DIFFICULTY_MIX = {
    "easy":   0.25,  # ~25/100
    "medium": 0.45,  # ~45/100
    "hard":   0.30,  # ~30/100
}


def main() -> int:
    rows = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    dev_50 = json.loads(DEV_50.read_text())
    priors = json.loads(PRIORS.read_text())["priors"]

    used_ids: set[str] = {p["instance_id"] for p in dev_50["instances"]}

    # Bucket: bucket[repo][difficulty] -> list of instance_ids, sorted.
    bucket: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        iid = row["instance_id"]
        if iid in used_ids:
            continue
        repo = row["repo"]
        # Use V7 prior label, defaulting to medium if absent.
        prior = priors.get(iid, {})
        diff = prior.get("estimated_difficulty", "medium")
        if diff not in DIFFICULTY_MIX:
            diff = "medium"  # fold unknown / unknown_traj_* into medium
        bucket[repo][diff].append(iid)
    for repo in bucket:
        for diff in bucket[repo]:
            bucket[repo][diff].sort()

    picked: list[dict] = []
    for repo, quota in REPO_QUOTA.items():
        if quota == 0:
            continue
        if repo not in bucket:
            print(f"WARN: {repo} has no remaining instances after dev_50 carveout")
            continue
        per_diff = {
            d: max(0, int(quota * frac))
            for d, frac in DIFFICULTY_MIX.items()
        }
        # Round to quota.
        while sum(per_diff.values()) > quota:
            for d in ("medium", "easy", "hard"):
                if per_diff[d] > 0:
                    per_diff[d] -= 1
                    break
        while sum(per_diff.values()) < quota:
            per_diff["medium"] += 1

        for diff, want in per_diff.items():
            available = bucket[repo].get(diff, [])
            picked_ids = available[:want]
            for iid in picked_ids:
                picked.append({"instance_id": iid, "repo": repo, "difficulty": diff})
            if want > len(available):
                shortfall = want - len(available)
                for alt_diff in ("medium", "hard", "easy"):
                    if alt_diff == diff:
                        continue
                    for iid in bucket[repo].get(alt_diff, []):
                        if any(p["instance_id"] == iid for p in picked):
                            continue
                        picked.append({
                            "instance_id": iid,
                            "repo": repo,
                            "difficulty": diff + "(fill)",
                        })
                        shortfall -= 1
                        if shortfall == 0:
                            break
                    if shortfall == 0:
                        break

    picked.sort(key=lambda p: p["instance_id"])
    assert len(picked) == 100, f"expected 100, got {len(picked)}"

    overlap = {p["instance_id"] for p in picked} & used_ids
    assert not overlap, f"dev_100 overlaps dev_50: {overlap}"

    by_diff = defaultdict(int)
    by_repo = defaultdict(int)
    for p in picked:
        by_diff[p["difficulty"].replace("(fill)", "")] += 1
        by_repo[p["repo"]] += 1

    out = {
        "version": "v10-dev-100-r1",
        "description": (
            "V10 dev-100 split for decision-making after component-tuning "
            "is locked on dev-50. Held out from dev_50 with zero overlap. "
            "Repo-balanced (~2x dev-50 quotas, redistributed where Verified "
            "ran out). Difficulty-balanced via V7 priors as a calibration "
            "target only — production routing re-derives difficulty from "
            "issue features, NOT these priors."
        ),
        "selection_criteria": {
            "n": 100,
            "difficulty_mix": DIFFICULTY_MIX,
            "repo_quota": REPO_QUOTA,
            "deterministic": True,
            "held_out_from": "splits/dev_50.json",
            "overlap_with_dev_50": 0,
        },
        "summary": {
            "by_difficulty": dict(by_diff),
            "by_repo": dict(by_repo),
        },
        "instances": picked,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} with {len(picked)} instances")
    print(f"  by difficulty: {dict(by_diff)}")
    print(f"  by repo: {dict(by_repo)}")
    print(f"  overlap with dev_50: {len(overlap)} (must be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

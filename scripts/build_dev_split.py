"""Construct splits/dev_50.json (and a placeholder dev_100.json) from
the cached SWE-bench Verified rows.

Per V10_DESIGN.md §3.1 the dev-50 split is for component sweeps and
must be:

  - difficulty-balanced: 15 easy, 20 medium, 15 hard
  - repo-diverse: every Verified repo represented, weighted to not
    over-concentrate on django/sympy
  - deterministic: the same `make build-split` produces the same
    instances on any machine

The selection is deterministic via stable sort by instance_id within
each (repo, difficulty) bucket. No randomness, no oracle metadata.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict


DATASET_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
)
SPLITS_DIR = pathlib.Path(__file__).resolve().parent.parent / "splits"

# Per-repo target counts for dev_50 (sums to 50). Picks favor repo
# diversity over the natural Verified distribution (django is 46% of the
# full set; here it's 24%). This keeps ablation signal honest across
# repos rather than letting django dominate.
REPO_QUOTA: dict[str, int] = {
    "django/django": 12,
    "sympy/sympy": 7,
    "sphinx-doc/sphinx": 5,
    "matplotlib/matplotlib": 5,
    "scikit-learn/scikit-learn": 5,
    "astropy/astropy": 4,
    "pydata/xarray": 3,
    "pytest-dev/pytest": 3,
    "pylint-dev/pylint": 2,
    "psf/requests": 2,
    "mwaskom/seaborn": 1,
    "pallets/flask": 1,
}
assert sum(REPO_QUOTA.values()) == 50, "REPO_QUOTA must sum to 50"

# Within each repo, target this difficulty mix (sums to 1.0). We round
# down per repo and absorb remainders into "medium".
DIFFICULTY_MIX = {
    "easy":   0.30,   # ~15/50
    "medium": 0.40,   # ~20/50
    "hard":   0.30,   # ~15/50
}

DIFFICULTY_BUCKET = {
    "<15 min fix":   "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours":     "hard",
    ">4 hours":      "hard",
}


def main() -> None:
    SPLITS_DIR.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]

    # bucket[repo][difficulty] -> list of instance_ids, sorted deterministically
    bucket: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        repo = row["repo"]
        diff = DIFFICULTY_BUCKET.get(row.get("difficulty", "?"), "medium")
        bucket[repo][diff].append(row["instance_id"])
    for repo in bucket:
        for diff in bucket[repo]:
            bucket[repo][diff].sort()

    # Pick per repo: round-down quotas, absorb remainder into medium.
    picked: list[dict] = []
    for repo, quota in REPO_QUOTA.items():
        if repo not in bucket:
            print(f"WARN: {repo} not in dataset; skipping")
            continue
        per_diff = {
            d: max(1 if quota > 0 else 0, int(quota * frac))
            for d, frac in DIFFICULTY_MIX.items()
        }
        # If sum > quota (due to rounding floor on tiny quotas), reduce medium first.
        while sum(per_diff.values()) > quota:
            for d in ("medium", "easy", "hard"):
                if per_diff[d] > 0:
                    per_diff[d] -= 1
                    break
        # If sum < quota, top up medium.
        while sum(per_diff.values()) < quota:
            per_diff["medium"] += 1

        for diff, want in per_diff.items():
            available = bucket[repo].get(diff, [])
            for iid in available[:want]:
                picked.append({"instance_id": iid, "repo": repo, "difficulty": diff})
            if want > len(available):
                # Fill from any other difficulty in same repo.
                shortfall = want - len(available)
                for alt_diff in ("medium", "easy", "hard"):
                    if alt_diff == diff:
                        continue
                    for iid in bucket[repo].get(alt_diff, []):
                        if any(p["instance_id"] == iid for p in picked):
                            continue
                        picked.append({"instance_id": iid, "repo": repo, "difficulty": diff + "(fill)"})
                        shortfall -= 1
                        if shortfall == 0:
                            break
                    if shortfall == 0:
                        break

    # Sort final list by instance_id for stable output.
    picked.sort(key=lambda p: p["instance_id"])
    assert len(picked) == 50, f"expected 50 picks, got {len(picked)}"

    out = {
        "version": "v10-dev-50-r1",
        "description": (
            "V10 dev-50 split for component sweeps. Difficulty-balanced "
            "(15 easy / 20 medium / 15 hard) and repo-diverse. "
            "Deterministic: stable sort by instance_id within (repo, difficulty). "
            "DOES NOT touch FAIL_TO_PASS, PASS_TO_PASS, or any oracle field."
        ),
        "selection_criteria": {
            "n": 50,
            "difficulty_mix": DIFFICULTY_MIX,
            "repo_quota": REPO_QUOTA,
            "deterministic": True,
        },
        "instances": picked,
    }
    out_path = SPLITS_DIR / "dev_50.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path} with {len(picked)} instances")
    # Diagnostics
    by_diff: dict[str, int] = defaultdict(int)
    by_repo: dict[str, int] = defaultdict(int)
    for p in picked:
        by_diff[p["difficulty"].replace("(fill)", "")] += 1
        by_repo[p["repo"]] += 1
    print(f"  by difficulty: {dict(by_diff)}")
    print(f"  by repo: {dict(by_repo)}")


if __name__ == "__main__":
    main()

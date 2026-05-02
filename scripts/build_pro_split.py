"""Construct splits/test_pro.json — the SWE-bench Pro public test set.

Mirror of build_test_500.py for the Pro dataset. Pro's
swebench_pro.jsonl IS the canonical 731-instance public test set; this
script just shapes the entries into the {instances: [...]} format that
the eval scripts expect via --split.

Pro vs. Verified deltas (see V10_DESIGN.md):
  - 731 instances (vs. 500 for Verified)
  - 11 repos (vs. 12 for Verified)
  - Schema: lowercase fail_to_pass/pass_to_pass; no hints_text/version
  - Image tag lives in row['dockerhub_tag']; image is hosted at
    jefzda/sweap-images:{dockerhub_tag} (NOT swebench/sweb.eval.*)

Per V10_DESIGN.md §9 split discipline: the Pro test split is the
leaderboard run, touched ONCE.
"""

from __future__ import annotations

import json
import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "meta_harness_plus" / "tasks" / "data" / "swebench_pro.jsonl"
OUT = PROJECT_ROOT / "splits" / "test_pro.json"

EXPECTED_ROW_COUNT = 731


def main() -> int:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_ROW_COUNT:
        raise SystemExit(f"expected {EXPECTED_ROW_COUNT} rows in Pro dataset, got {len(rows)}")

    instances = sorted(
        (
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "dockerhub_tag": r["dockerhub_tag"],
            }
            for r in rows
        ),
        key=lambda e: e["instance_id"],
    )
    out = {
        "version": "v10-test-pro-r1",
        "description": (
            "V10 test_pro split — the SWE-bench Pro public test set (731 instances). "
            "Leaderboard run, touched ONCE."
        ),
        "selection_criteria": {
            "n": EXPECTED_ROW_COUNT,
            "deterministic": True,
            "source": "meta_harness_plus/tasks/data/swebench_pro.jsonl",
            "image_repo": "jefzda/sweap-images",
        },
        "instances": instances,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} with {len(instances)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Construct splits/test_500.json — the full SWE-bench Verified set.

The dataset's swebench_verified.jsonl IS test_500; this script just
shapes the entries into the {instances: [...]} format that
scripts/retrieval_eval_dev50.py expects via --split.

Per V10_DESIGN.md §9 split discipline: test-500 is the headline run.
Touched ONCE.
"""

from __future__ import annotations

import json
import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
OUT = PROJECT_ROOT / "splits" / "test_500.json"


def main() -> int:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if len(rows) != 500:
        raise SystemExit(f"expected 500 rows in dataset, got {len(rows)}")

    instances = sorted(
        ({"instance_id": r["instance_id"], "repo": r["repo"]} for r in rows),
        key=lambda e: e["instance_id"],
    )
    out = {
        "version": "v10-test-500-r1",
        "description": (
            "V10 test_500 split — the full SWE-bench Verified set. "
            "This is the headline run, touched ONCE."
        ),
        "selection_criteria": {
            "n": 500,
            "deterministic": True,
            "source": "meta_harness_plus/tasks/data/swebench_verified.jsonl",
        },
        "instances": instances,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} with {len(instances)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

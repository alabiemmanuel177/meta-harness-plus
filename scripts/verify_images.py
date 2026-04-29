"""Verify all SWE-bench Verified Docker images are present locally.

Per V10_DESIGN.md §13.1: V10 sandbox refuses to pull at runtime, so a
pre-flight check on the image cache is required before any large eval.
This script must run in <5 s.

Usage:

    python scripts/verify_images.py                  # check all 500 Verified
    python scripts/verify_images.py --split splits/dev_50.json
    python scripts/verify_images.py --print-missing  # show pull commands

Exit code 0 if all required images are present; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Iterable


def _swebench_image_name(instance_id: str) -> str:
    """Mirror meta_harness_plus.agent_docker._swebench_hub_image."""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def _list_local_images() -> set[str]:
    """One ``docker images`` call returning every locally-stored image
    in repository:tag form. O(1) docker call → O(N) python compare."""
    res = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, timeout=10,
    )
    if res.returncode != 0:
        raise RuntimeError(f"docker images failed: {res.stderr.strip()}")
    return {line.strip() for line in res.stdout.splitlines() if line.strip()}


def _required_instance_ids(split_path: pathlib.Path | None) -> list[str]:
    if split_path is not None:
        data = json.loads(split_path.read_text())
        return [entry["instance_id"] for entry in data["instances"]]
    # Default: all 500 Verified instance_ids from the canonical jsonl.
    cache = (
        pathlib.Path(__file__).resolve().parent.parent
        / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
    )
    out: list[str] = []
    for line in cache.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line)["instance_id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default=None,
                    help="splits/<name>.json to check (default: full Verified)")
    ap.add_argument("--print-missing", action="store_true",
                    help="print docker pull commands for missing images")
    args = ap.parse_args()

    t0 = time.perf_counter()
    split_path = pathlib.Path(args.split) if args.split else None
    required = _required_instance_ids(split_path)
    local_images = _list_local_images()

    missing: list[str] = []
    for iid in required:
        image = _swebench_image_name(iid)
        if image not in local_images:
            missing.append(iid)

    elapsed = time.perf_counter() - t0
    label = split_path.name if split_path else "test_500 (full Verified)"
    if not missing:
        print(f"[verify-images] OK {len(required)}/{len(required)} present "
              f"for {label} ({elapsed:.2f}s)")
        return 0
    print(f"[verify-images] FAIL {len(missing)}/{len(required)} missing "
          f"for {label} ({elapsed:.2f}s)")
    if args.print_missing:
        for iid in missing:
            print(f"docker pull {_swebench_image_name(iid)}")
    else:
        for iid in missing[:10]:
            print(f"  missing: {iid}")
        if len(missing) > 10:
            print(f"  …and {len(missing) - 10} more (re-run with --print-missing)")
    return 2


if __name__ == "__main__":
    sys.exit(main())

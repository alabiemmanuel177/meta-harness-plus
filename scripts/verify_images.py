"""Verify SWE-bench Docker images are present locally (Verified or Pro).

Per V10_DESIGN.md §13.1: V10 sandbox refuses to pull at runtime, so a
pre-flight check on the image cache is required before any large eval.
This script must run in <5 s.

Two image-naming schemes are supported:
  - Verified: swebench/sweb.eval.x86_64.{iid_munged}:latest
  - Pro:      jefzda/sweap-images:{dockerhub_tag}

Pro is auto-detected when the split entries carry a ``dockerhub_tag``
field (which build_pro_split.py emits) or when the split path's name
contains ``pro``.

Usage:

    python scripts/verify_images.py                            # all 500 Verified
    python scripts/verify_images.py --split splits/dev_50.json
    python scripts/verify_images.py --split splits/test_pro.json
    python scripts/verify_images.py --print-missing            # show pull commands

Exit code 0 if all required images are present; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time


PRO_IMAGE_REPO = "jefzda/sweap-images"


def _verified_image_name(instance_id: str) -> str:
    """Mirror meta_harness_plus.agent_docker._swebench_hub_image."""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def _pro_image_name(dockerhub_tag: str) -> str:
    return f"{PRO_IMAGE_REPO}:{dockerhub_tag}"


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


def _required_entries(split_path: pathlib.Path | None) -> tuple[str, list[dict]]:
    """Return (variant, [entry, ...]). variant is 'verified' or 'pro'.
    Each entry has keys instance_id and (for pro) dockerhub_tag.
    """
    if split_path is not None:
        data = json.loads(split_path.read_text())
        entries = data["instances"]
        # Pro detection: explicit dockerhub_tag in the split entries.
        if entries and "dockerhub_tag" in entries[0]:
            return "pro", entries
        return "verified", entries
    # Default: all 500 Verified instance_ids from the canonical jsonl.
    cache = (
        pathlib.Path(__file__).resolve().parent.parent
        / "meta_harness_plus" / "tasks" / "data" / "swebench_verified.jsonl"
    )
    entries = []
    for line in cache.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        entries.append({"instance_id": row["instance_id"], "repo": row["repo"]})
    return "verified", entries


def _image_for_entry(variant: str, entry: dict) -> str:
    if variant == "pro":
        return _pro_image_name(entry["dockerhub_tag"])
    return _verified_image_name(entry["instance_id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default=None,
                    help="splits/<name>.json to check (default: full Verified)")
    ap.add_argument("--print-missing", action="store_true",
                    help="print docker pull commands for missing images")
    args = ap.parse_args()

    t0 = time.perf_counter()
    split_path = pathlib.Path(args.split) if args.split else None
    variant, entries = _required_entries(split_path)
    local_images = _list_local_images()

    missing: list[dict] = []
    for entry in entries:
        image = _image_for_entry(variant, entry)
        if image not in local_images:
            missing.append(entry)

    elapsed = time.perf_counter() - t0
    if split_path:
        label = f"{split_path.name} ({variant})"
    else:
        label = "test_500 (full Verified)"
    if not missing:
        print(f"[verify-images] OK {len(entries)}/{len(entries)} present "
              f"for {label} ({elapsed:.2f}s)")
        return 0
    print(f"[verify-images] FAIL {len(missing)}/{len(entries)} missing "
          f"for {label} ({elapsed:.2f}s)")
    if args.print_missing:
        for entry in missing:
            print(f"docker pull {_image_for_entry(variant, entry)}")
    else:
        for entry in missing[:10]:
            print(f"  missing: {entry['instance_id']}")
        if len(missing) > 10:
            print(f"  …and {len(missing) - 10} more (re-run with --print-missing)")
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""Pull pre-built SWE-bench Verified instance images from Docker Hub.

Reads the first ``--n`` instances from SWE-bench Verified, computes the
Docker Hub image name for each (using the ``_1776_`` separator that
the swebench harness rewrites ``__`` to), and pulls them in parallel.

Speed: ~30 min for 50 images on typical home internet (each image
~2-3 GB; pulls happen in parallel with 8 workers by default).

Resume-safe: skips images that are already present locally. If you
rerun this script, only newly-needed images are pulled.

Cost: $0 — Pro flat-rate doesn't apply; this is just Docker Hub
bandwidth (free for pulls).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from meta_harness_plus.swebench_adapter import load_swebench_verified


def hub_image_name(instance_id: str) -> str:
    """Convert an instance_id like ``sympy__sympy-22914`` to the Docker
    Hub image ``swebench/sweb.eval.x86_64.sympy_1776_sympy-22914:latest``."""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def image_present(name: str) -> bool:
    out = subprocess.run(
        ["docker", "image", "inspect", name],
        capture_output=True, text=True,
    )
    return out.returncode == 0


def pull_one(instance_id: str) -> dict:
    image = hub_image_name(instance_id)
    if image_present(image):
        return {"id": instance_id, "image": image, "ok": True,
                "skipped": True, "wall_s": 0.0, "size_mb": _image_size_mb(image)}
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["docker", "pull", image],
        capture_output=True, text=True, timeout=900,
    )
    wall = time.perf_counter() - t0
    return {
        "id": instance_id, "image": image,
        "ok": proc.returncode == 0,
        "skipped": False, "wall_s": round(wall, 1),
        "stderr_first_200": proc.stderr[:200] if proc.stderr else "",
        "size_mb": _image_size_mb(image) if proc.returncode == 0 else 0,
    }


def _image_size_mb(name: str) -> float:
    out = subprocess.run(
        ["docker", "image", "inspect", name, "--format", "{{.Size}}"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return 0.0
    try:
        return round(int(out.stdout.strip()) / 1024 / 1024, 1)
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50,
                    help="How many of the 500 Verified instances to pull.")
    ap.add_argument("--instance-ids", nargs="*",
                    help="Specific instance IDs to pull (overrides --n).")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--out-json", default="runs/swebench_smoke/image_pull_log.json")
    args = ap.parse_args()

    insts = load_swebench_verified(
        n=args.n if not args.instance_ids else None,
        instance_ids=args.instance_ids,
    )
    print(f"[pull] {len(insts)} instances to ensure-pulled "
          f"({args.max_workers} parallel workers)")

    results: list[dict] = []
    n_done = 0
    n_failed = 0
    n_skipped = 0
    total_size_mb = 0.0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(pull_one, inst.instance_id): inst for inst in insts}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_done += 1
            if r["ok"]:
                if r.get("skipped"):
                    n_skipped += 1
                total_size_mb += r["size_mb"]
            else:
                n_failed += 1
            elapsed = time.perf_counter() - t0
            tag = "skip" if r.get("skipped") else ("OK " if r["ok"] else "FAIL")
            print(f"[pull] +{elapsed:6.1f}s  {tag}  {r['id']:40s}  "
                  f"wall={r['wall_s']}s  size={r['size_mb']}MB  "
                  f"({n_done}/{len(insts)} done, {n_failed} failed)")

    total = time.perf_counter() - t0
    print(f"\n[pull] === SUMMARY ===")
    print(f"[pull] total wall: {total:.1f}s")
    print(f"[pull] succeeded: {n_done - n_failed}/{len(insts)} "
          f"({n_skipped} already cached)")
    print(f"[pull] failed: {n_failed}")
    print(f"[pull] total disk used: {total_size_mb / 1024:.1f} GB")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_attempted": len(insts),
        "n_succeeded": n_done - n_failed,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "wall_s": round(total, 1),
        "total_size_gb": round(total_size_mb / 1024, 2),
        "results": results,
    }, indent=2))
    print(f"[pull] wrote {out}")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

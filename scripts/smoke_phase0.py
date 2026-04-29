"""Phase 0 end-to-end smoke.

This is the gate from V10_DESIGN.md §3.1 / §10 commit 6:

  - Load one Verified instance via harness.dataset.load_verified_view
  - Assert clean V10 caches at startup (tightening 4)
  - Start a Sandbox at base_commit
  - Run the public test suite via the leak-free run_public_suite() —
    NOT via the legacy run_tests(test_targets) API
  - Write trajectory events to trajectories/v10_smoke/{instance_id}/
  - Verify the trajectory file landed on disk

No LLM calls, no patch generation. Just a structural shakedown of the
Phase 0 plumbing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from harness.cache import assert_clean_cache_at_startup
from harness.dataset import load_verified_view
from harness.sandbox import Sandbox
from harness.trajectory import trajectory


# Default to a small repo with a fast public suite; overridable.
DEFAULT_INSTANCE = "psf__requests-2317"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    ap.add_argument("--timeout-s", type=int, default=240,
                    help="public suite timeout (s); generous for smoke")
    ap.add_argument("--trajectories-root", default="trajectories/v10_smoke")
    ap.add_argument("--no-network", action="store_true", default=True)
    args = ap.parse_args()

    print(f"[smoke] instance_id={args.instance_id}")

    # 1. Cache hygiene gate. Phase 0 uses a SCOPED check on the smoke's
    # own write paths only — production-strict global hygiene fires in
    # Phase 1+. See V10_DESIGN.md §12.7 and harness/cache.py.
    print("[smoke] checking V10 cache hygiene (scoped)…")
    assert_clean_cache_at_startup(
        scope=(pathlib.Path(args.trajectories_root),),
    )

    # 2. Project the instance to InstanceView via the asserted boundary.
    print("[smoke] loading InstanceView…")
    view = load_verified_view(args.instance_id)
    print(f"        repo={view.repo!r}")
    print(f"        base_commit={view.base_commit!r}")
    print(f"        test_dirs={view.test_directives.dirs!r}")
    print(f"        test_dirs_source={view.test_directives.source!r}")

    traj_root = pathlib.Path(args.trajectories_root)
    traj_root.mkdir(parents=True, exist_ok=True)

    # 3. Open a trajectory writer for this instance.
    with trajectory(traj_root, view.instance_id, candidate_id="phase0_smoke") as tw:
        tw.write("phase0_start", {"timestamp": time.time()})
        tw.write("instance_loaded", {
            "repo": view.repo,
            "test_dirs": list(view.test_directives.dirs),
            "test_dirs_source": view.test_directives.source,
        })

        # 4. Start the sandbox at base_commit.
        print(f"[smoke] starting sandbox at base_commit={view.base_commit[:12]}…")
        with Sandbox(view, no_network=args.no_network) as sb:
            tw.write("sandbox_started", {"image": sb._exec.image})

            # 5. Run the public suite. THIS IS THE LEAK-FREE CALL.
            print(f"[smoke] running public test suite (collect-only, timeout={args.timeout_s}s)…")
            tw.write("public_suite_start", {"dirs": list(view.test_directives.dirs)})
            t0 = time.perf_counter()
            res = sb.run_public_suite(
                state_label="base",
                timeout_s=args.timeout_s,
                collect_only=True,
            )
            dur = time.perf_counter() - t0
            tw.write("public_suite_done", {
                "exit_code": res.exit_code,
                "duration_s": round(dur, 2),
                "truncated": res.truncated,
                "effective_test_paths": list(res.test_dirs),
                "log_tail_chars": len(res.log_excerpt),
                "log_tail": res.log_excerpt[-1500:],
            })
            print(f"        effective_paths={res.test_dirs!r}")
            print(f"        exit_code={res.exit_code}  duration={dur:.1f}s  "
                  f"log_chars={len(res.log_excerpt)}  truncated={res.truncated}")
            # Print the last few lines of pytest output for the operator.
            for line in res.log_excerpt.splitlines()[-6:]:
                print(f"        | {line}")

        tw.write("phase0_end", {"timestamp": time.time()})

    # 6. Verify trajectory landed.
    expected_dir = traj_root / view.instance_id / "phase0_smoke"
    assert expected_dir.exists(), f"trajectory dir missing: {expected_dir}"
    files = sorted(expected_dir.iterdir())
    assert files, f"no trajectory files in {expected_dir}"
    print(f"[smoke] trajectory landed: {files[0]}")

    # Print a one-line summary the Makefile target can grep on.
    print(f"[smoke] OK instance={args.instance_id} dur={dur:.1f}s "
          f"events_dir={expected_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

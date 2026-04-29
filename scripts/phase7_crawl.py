"""Phase 7 corpus crawler — public, pre-cutoff bug-fix commits across
~5,000 license-filtered Python repos. See V10_DESIGN.md §10.5.

Status: scaffolded but NOT auto-started. Running this is a deliberate
operator decision because of disk + bandwidth + GH-token cost
(roughly 50 GB and several hours for an initial 100-repo run; ~500 GB
and multi-week wall-clock for the full 5,000-repo corpus).

Usage:

    # Sanity check (no network, validates seed list + paths):
    python scripts/phase7_crawl.py --dry-run

    # Tiny initial run (10 repos, ~5 GB, ~30 min):
    GH_TOKEN=... python scripts/phase7_crawl.py --limit-repos 10

    # Full run (gated; must be confirmed by operator):
    GH_TOKEN=... python scripts/phase7_crawl.py --confirm-full

Output goes to ``data/phase7_corpus/{repo}/{sha}.jsonl`` in append
mode, resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


CORPUS_ROOT = pathlib.Path("data/phase7_corpus")
SEED_PATH = pathlib.Path("data/phase7_seed_repos.json")
ACCEPTABLE_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause"}


def _validate_environment(*, dry_run: bool) -> int:
    print("[phase7] validating environment…")
    if not dry_run and not os.environ.get("GH_TOKEN"):
        print("[phase7] FAIL: GH_TOKEN not set. Without it, the GitHub REST API")
        print("[phase7]       caps at 60 req/hr — the crawl will stall.")
        return 2
    if not SEED_PATH.exists():
        print(f"[phase7] FAIL: seed list missing at {SEED_PATH}.")
        print("[phase7]       Generate it once via the (unimplemented) seed builder")
        print("[phase7]       before running the crawl. See §10.5.")
        if dry_run:
            print("[phase7] (dry-run: continuing anyway)")
        else:
            return 2
    return 0


def _check_disk_budget() -> tuple[float, str]:
    """Return (free_gb, msg) with whether disk budget is acceptable."""
    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024 ** 3)
    msg = f"{free_gb:.1f} GB free; minimum recommended for a real run is 50 GB"
    return free_gb, msg


def _crawl_one_repo(repo_full: str, *, limit_commits: int, output_root: pathlib.Path) -> int:
    """Stub. Phase 1 work item: implement.

    Outline:
      1. Resolve repo metadata + license via GH API; reject if not in
         ACCEPTABLE_LICENSES.
      2. git clone --depth=N --filter=blob:none into a scratch dir.
      3. git log with regex matching `fix:`, `bug`, `closes #`, `fixes #`
         BEFORE model-cutoff date; cap at limit_commits.
      4. For each candidate commit:
           - Resolve linked issue text via GH API (if PR linked an issue).
           - Compute pre-commit skeleton at parent_sha.
           - Emit JSONL: {repo, sha, parent_sha, issue_text, files_changed, lines_changed, pre_commit_skeleton}.
      5. Atomic rename to output_root/{repo}/{sha}.jsonl.
    """
    print(f"[phase7] STUB: would crawl {repo_full!r} (up to {limit_commits} commits) "
          f"→ {output_root / repo_full}/")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="validate environment, print plan, exit 0")
    ap.add_argument("--limit-repos", type=int, default=10,
                    help="max repos to crawl (default 10 for safety)")
    ap.add_argument("--limit-commits", type=int, default=50,
                    help="max bug-fix commits per repo")
    ap.add_argument("--confirm-full", action="store_true",
                    help="run the full 5,000-repo corpus build (gated)")
    args = ap.parse_args()

    if args.confirm_full and args.limit_repos != 10:
        print("[phase7] WARN: --confirm-full overrides --limit-repos. Full run.")

    rc = _validate_environment(dry_run=args.dry_run)
    if rc != 0:
        return rc

    free_gb, disk_msg = _check_disk_budget()
    print(f"[phase7] disk: {disk_msg}")
    if not args.dry_run and free_gb < 50:
        print("[phase7] FAIL: less than 50 GB free; refuse to start a real run.")
        return 2

    if args.dry_run:
        print("[phase7] dry-run OK — environment is suitable, plan accepted.")
        return 0

    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    if not SEED_PATH.exists():
        print(f"[phase7] FAIL: cannot crawl without seed list at {SEED_PATH}.")
        return 2

    seed = json.loads(SEED_PATH.read_text())
    repos = seed.get("repos", [])
    if not args.confirm_full:
        repos = repos[: args.limit_repos]

    print(f"[phase7] starting crawl of {len(repos)} repos (limit={args.limit_commits} commits each)…")
    n_done = 0
    for repo_full in repos:
        rc = _crawl_one_repo(
            repo_full,
            limit_commits=args.limit_commits,
            output_root=CORPUS_ROOT,
        )
        if rc != 0:
            print(f"[phase7] WARN: {repo_full} failed with rc={rc}; continuing")
        else:
            n_done += 1

    print(f"[phase7] done; {n_done}/{len(repos)} repos succeeded")
    print(f"[phase7] output: {CORPUS_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

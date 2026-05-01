#!/usr/bin/env bash
# Roll a git bundle of the entire repo (all branches, all tags) to a
# host-local backup directory and prune bundles older than 14 days.
#
# Why this exists: the 2026-04-30 power outage corrupted 5 git objects
# mid-commit on the v10/phase-1 branch (zero-byte files at
# .git/objects/{c9,2e,30,a9,ac}/...). Recovery was clean only because
# the corruption hit objects that were re-creatable from the working
# tree, AND we had push history on origin that captured prior state.
# A bundle is the belt-and-braces version: a single self-contained
# file with every reachable object, restorable with
#   git clone meta-harness-YYYYMMDD.bundle restored-repo
# even if both .git and origin are simultaneously hosed.
#
# OPERATOR MUST ENABLE: this script is not on a cron yet. To run it
# nightly (recommended after the AMD outage post-mortem), add to your
# user crontab (e.g. `crontab -e`):
#   30 3 * * *  /home/eao/workplace/projects/Meta-Harness/scripts/backup_git_bundle.sh >> /var/log/meta-harness-backup.log 2>&1
# or invoke it manually from a Makefile target after each long-horizon
# batch.
#
# Cost: a fresh bundle of this repo is ~150-300 MB; 14 days × 1/day
# = ~2-4 GB on disk. Adjust the retention via BACKUP_KEEP_DAYS.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$HOME/meta-harness-backups}"
BACKUP_KEEP_DAYS="${BACKUP_KEEP_DAYS:-14}"

mkdir -p "$BACKUP_DIR"

cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_PATH="$BACKUP_DIR/v10-$STAMP.bundle"

# --all = every ref (branches, tags, remote-tracking, notes). We
# explicitly DO NOT include the working tree or stash — those aren't
# bundle-able and shouldn't be relied on for durability anyway.
git bundle create "$BUNDLE_PATH" --all
git bundle verify "$BUNDLE_PATH"

# Prune anything older than BACKUP_KEEP_DAYS.
find "$BACKUP_DIR" -maxdepth 1 -type f -name "v10-*.bundle" -mtime "+$BACKUP_KEEP_DAYS" -print -delete

echo "[backup_git_bundle] wrote $BUNDLE_PATH"
echo "[backup_git_bundle] retained bundles:"
ls -lh "$BACKUP_DIR"/v10-*.bundle 2>/dev/null || echo "  (none)"

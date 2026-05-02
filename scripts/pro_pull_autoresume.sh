#!/usr/bin/env bash
# Auto-resume the Pro bulk image pull.
#
# Designed for two scenarios:
#   1. Power loss + reboot: an @reboot cron entry runs this once.
#   2. Mid-run crash / SSH disconnect: a */5 cron entry runs this every
#      5 min as a safety net.
#
# Idempotent: does nothing if a pull is already running, or if the
# previous run completed (progress file shows "DONE"). Otherwise
# launches scripts/pull_pro_images.sh detached from any TTY.
#
# Logs:
#   /tmp/pro_pull_autoresume.log — this script's bookkeeping
#   /tmp/pro_image_pull.log      — the actual docker-pull output
#   /tmp/pro_pull_progress.txt   — "<ok> <fail> <skip> [STATE]"

set -u

PROJECT=/home/eao/workplace/projects/Meta-Harness
PULL_SCRIPT="$PROJECT/scripts/pull_pro_images.sh"
PROGRESS=/tmp/pro_pull_progress.txt
LOG=/tmp/pro_pull_autoresume.log
LOCK=/tmp/pro_pull_autoresume.lock

# docker / system binaries are not in the cron PATH by default.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

# Single-flight: don't race with another autoresume invocation.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Is)] another autoresume is running; exiting" >> "$LOG"
  exit 0
fi

# 1. Already running? Bail. Match by basename only — the script can be
# launched as either "bash scripts/pull_pro_images.sh" (foreground) or
# with the absolute path (cron); pgrep -f sees both forms in argv.
if pgrep -f "pull_pro_images.sh" >/dev/null 2>&1; then
  echo "[$(date -Is)] pull script already running; nothing to do" >> "$LOG"
  exit 0
fi

# 2. Previous run finished cleanly? Bail.
if [ -f "$PROGRESS" ] && grep -q "DONE" "$PROGRESS" 2>/dev/null; then
  echo "[$(date -Is)] previous run DONE: $(cat "$PROGRESS"); nothing to do" >> "$LOG"
  exit 0
fi

# 3. Previous run hit a hard stop? Bail (operator must re-evaluate).
if [ -f "$PROGRESS" ] && grep -qE "ABORT_DISK|ABORT_FAIL" "$PROGRESS" 2>/dev/null; then
  echo "[$(date -Is)] previous run aborted: $(cat "$PROGRESS"); operator must clear progress file to resume" >> "$LOG"
  exit 0
fi

# 4. Otherwise: relaunch.
cd "$PROJECT" || {
  echo "[$(date -Is)] cannot cd to $PROJECT" >> "$LOG"
  exit 1
}

echo "[$(date -Is)] launching pull script" >> "$LOG"
nohup bash "$PULL_SCRIPT" >>/tmp/pro_pull_main.out 2>&1 &
new_pid=$!
disown 2>/dev/null || true
echo "[$(date -Is)] launched pid=$new_pid" >> "$LOG"

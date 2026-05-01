#!/usr/bin/env bash
# Auto-resume the test_500 DeepSeek headline run after a reboot.
#
# Why this exists: the test_500 run is ~24h wall-clock. Power outages
# during the run would lose hours of progress without auto-resume.
# The retrieval_eval_dev50.py script writes per-instance checkpoints
# atomically, so resuming from any boundary is lossless — this
# wrapper just kicks the eval off again with the SAME signature so
# completed checkpoints get reused (Tier-1 cache hits skip the
# instance entirely).
#
# Hooked via the user crontab:
#   @reboot /home/eao/workplace/projects/Meta-Harness/scripts/test500_autoresume.sh
#
# Idempotency: this script can fire multiple times without harm.
#   1. If an eval is already running (pgrep), exit immediately.
#   2. If all 500 checkpoints exist, exit immediately (run is done).
#   3. Otherwise, sleep 60s for the system to settle (network,
#      docker daemon), then re-launch the eval in the foreground.
#      Cron runs detached; the foreground process inherits the cron
#      session and persists.
#
# Logs append to /tmp/test500_deepseek_autoresume.log with a date
# stamp at each invocation so you can audit what fired and when.

set -euo pipefail

REPO=/home/eao/workplace/projects/Meta-Harness
LOG=/tmp/test500_deepseek_autoresume.log
RUN_LOG=/tmp/test500_deepseek.log
CKPT="$REPO/runs/v10_test_500_retr_eval/checkpoints/embed_noshortlist_tb_bs256_rerank"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { printf '[%s] %s\n' "$(stamp)" "$*" >> "$LOG"; }

say "autoresume invoked (uptime: $(uptime -p 2>/dev/null || echo unknown))"

# Idempotency 1: don't double-launch if an eval is already running.
if pgrep -f 'retrieval_eval_dev50.py.*splits/test_500.json' > /dev/null; then
  say "eval already running, exiting"
  exit 0
fi

# Idempotency 2: don't relaunch if the run is complete.
if [ -d "$CKPT" ]; then
  N=$(ls "$CKPT" 2>/dev/null | wc -l)
  if [ "$N" -ge 500 ]; then
    say "run already complete (500/500 checkpoints), exiting"
    exit 0
  fi
  say "resuming with $N/500 existing checkpoints"
else
  say "no checkpoint dir yet — starting fresh"
fi

# Settle delay. Cron @reboot fires before docker daemon is fully up
# on some configurations; 60s is a safe margin.
sleep 60

# Re-source the environment and launch with the EXACT same args as
# the original invocation. Same signature → cached checkpoints are
# reused (Tier-1 cache hit per scripts/retrieval_eval_dev50.py:_run_one_instance).
cd "$REPO"
set -a
. ./.env
set +a

say "launching test_500 DeepSeek run (output appends to $RUN_LOG)"
exec >> "$RUN_LOG" 2>&1
echo
echo "=== AUTORESUME LAUNCH at $(stamp) ==="
echo
exec env PYTHONPATH=. .venv/bin/python3 scripts/retrieval_eval_dev50.py \
  --split splits/test_500.json \
  --workers 1 --batch-size 256 \
  --no-shortlist --include-traceback --rerank

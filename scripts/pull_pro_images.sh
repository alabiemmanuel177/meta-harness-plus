#!/usr/bin/env bash
# Serial Pro-image pull with disk + failure guards.
# Logs to /tmp/pro_image_pull.log. Skips already-cached images.
# Hard-stops:
#   - free disk on / < 200 GB
#   - cumulative pull failures (rc != 0) > 50
# Usage:
#   bash scripts/pull_pro_images.sh
# Re-runnable: re-running just resumes (cached images are skipped).

set -u

LOG=/tmp/pro_image_pull.log
TAGS_FILE=/tmp/pro_tags_all.txt
PROGRESS=/tmp/pro_pull_progress.txt
MIN_FREE_GB=200
MAX_FAIL=50
TIMEOUT_S=300

# Generate the tag list once (idempotent).
python3 - <<'PY'
import json
data = json.load(open("splits/test_pro.json"))
with open("/tmp/pro_tags_all.txt", "w") as fh:
    for inst in data["instances"]:
        fh.write(inst["dockerhub_tag"] + "\n")
PY

total=$(wc -l < "$TAGS_FILE")
echo "[$(date -Is)] starting bulk pull: $total images" | tee -a "$LOG"

# Build the set of images already cached so we can skip without
# invoking docker per-image (faster startup).
docker images --format "{{.Repository}}:{{.Tag}}" \
  | grep "^jefzda/sweap-images:" \
  | sed 's|^jefzda/sweap-images:||' \
  | sort -u > /tmp/pro_tags_cached.txt
cached=$(wc -l < /tmp/pro_tags_cached.txt)
echo "[$(date -Is)] $cached images already cached; will skip those" | tee -a "$LOG"

ok=0
fail=0
skip=0
i=0
echo "0 0 0" > "$PROGRESS"  # ok fail skip

while IFS= read -r tag; do
  i=$((i+1))
  if grep -qxF "$tag" /tmp/pro_tags_cached.txt; then
    skip=$((skip+1))
    echo "$ok $fail $skip" > "$PROGRESS"
    continue
  fi

  # Disk check every 10 pulls. df reports in 1K blocks; convert to GB.
  if [ $((i % 10)) -eq 0 ]; then
    free_gb=$(df --output=avail / | tail -1 | awk '{print int($1/1024/1024)}')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
      echo "[$(date -Is)] HARD STOP: free disk ${free_gb} GB < ${MIN_FREE_GB} GB. Aborting at i=$i (ok=$ok fail=$fail skip=$skip)." | tee -a "$LOG"
      echo "$ok $fail $skip ABORT_DISK" > "$PROGRESS"
      exit 3
    fi
  fi

  if [ "$fail" -gt "$MAX_FAIL" ]; then
    echo "[$(date -Is)] HARD STOP: failures=$fail > $MAX_FAIL. Aborting at i=$i (ok=$ok skip=$skip)." | tee -a "$LOG"
    echo "$ok $fail $skip ABORT_FAIL" > "$PROGRESS"
    exit 4
  fi

  echo "[$(date -Is)] [$i/$total] pulling jefzda/sweap-images:$tag" >> "$LOG"
  if timeout "$TIMEOUT_S" docker pull "jefzda/sweap-images:$tag" >>"$LOG" 2>&1; then
    ok=$((ok+1))
    echo "[$(date -Is)] [$i/$total] OK  $tag" >> "$LOG"
  else
    rc=$?
    fail=$((fail+1))
    echo "[$(date -Is)] [$i/$total] FAIL rc=$rc $tag" >> "$LOG"
  fi
  echo "$ok $fail $skip" > "$PROGRESS"
done < "$TAGS_FILE"

echo "[$(date -Is)] DONE: ok=$ok fail=$fail skip=$skip" | tee -a "$LOG"
echo "$ok $fail $skip DONE" > "$PROGRESS"

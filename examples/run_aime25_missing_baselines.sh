#!/usr/bin/env bash
# Launch the 4 AIME-25 baselines that the original Phase 2 sweep skipped
# (back when AIME-25 was thought to be saturated). Phase 2 proved it's
# NOT saturated (best of 2 baselines = 0.367), so Phase 3 needs a real
# best-baseline comparator including dspy/opro/maj16/random.
#
# Pinned at 8 workers because Pro returned HTTP 429 at 16 on
# deepseek-v3.1:671b. Resume-safe via run_one's existing skip check.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs runs/wow_push

MODEL="deepseek-v3.1:671b"
# Bumped to 16 (the empirical Pro account-wide cap) since we're now
# running serially — no concurrent gpt-oss:20b job to share with.
WORKERS=16
PY=".venv/bin/python3"
ts() { date +"%H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

run_one() {
    local task="$1" base="$2" seed="$3"
    local model_safe="${MODEL//[:\/]/_}"
    local key="${task}_${base}_${model_safe}_seed${seed}"
    local outpath="runs/wow_push/${key}.json"
    if [ -f "$outpath" ]; then
        log "skip ${key}: already complete"
        return 0
    fi
    log "running ${key}"
    "$PY" examples/run_math_baselines.py \
        --task "$task" --baseline "$base" --seed "$seed" \
        --model "$MODEL" --max-workers "$WORKERS" \
        > "logs/${key}.log" 2>&1
    log "  done ${key}"
}

log "=== AIME-25 missing baselines (dspy, opro, maj16, random; 5 seeds each) ==="
for seed in 0 1 2 3 4; do
    for base in maj16 dspy opro random; do
        run_one aime25 "$base" "$seed"
    done
done

log "=== AIME-25 missing baselines complete ==="

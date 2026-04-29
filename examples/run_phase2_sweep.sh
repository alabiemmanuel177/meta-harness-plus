#!/usr/bin/env bash
# Phase 2 baseline sweep: HMMT integer-only (14 problems) × 6 baselines × 5
# seeds × deepseek-v3.1:671b, then AIME-25 × 2 baselines × 5 seeds. Output
# per-(task,baseline,seed) JSON under runs/wow_push/. Tail -f the chosen log.
#
# Wall projection (4 workers): HMMT ~10h, AIME-25 ~6h, total ~16h sequential.
# DON'T parallelize the two task families (Pro rate limit headroom).
#
# Usage:
#     bash examples/run_phase2_sweep.sh                    # full sweep
#     bash examples/run_phase2_sweep.sh hmmt_only          # HMMT only
#     bash examples/run_phase2_sweep.sh aime_only          # AIME only
set -e
cd "$(dirname "$0")/.."
mkdir -p logs runs/wow_push

MODEL="deepseek-v3.1:671b"
# Concurrency is PER-MODEL on Ollama Pro:
# - gpt-oss:20b   passed 16-worker smoke (60/60, p95/p50=1.20x, 3.56 QPS)
# - deepseek-v3.1:671b returned HTTP 429 "too many concurrent requests"
#   at 16 workers; falling back to 8 per the WOW_PUSH_PLAN amendment.
# Document this Pro ceiling in PHASE2_TIER_VALIDATION.md when it lands.
WORKERS=8
PY=".venv/bin/python3"
ts() { date +"%H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

MODE="${1:-full}"

run_one() {
    local task="$1" base="$2" seed="$3"
    # IMPORTANT: this substitution MUST match run_math_baselines.py's
    # `re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)` — i.e. only ':'
    # and '/' get replaced (not '.'), so deepseek-v3.1:671b becomes
    # deepseek-v3.1_671b on disk.
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

if [ "$MODE" != "aime_only" ]; then
    log "=== HMMT-Feb-2025 baselines (14-problem integer subset) ==="
    for seed in 0 1 2 3 4; do
        for base in cot maj8 maj16 dspy opro random; do
            run_one hmmt "$base" "$seed"
        done
    done
fi

if [ "$MODE" != "hmmt_only" ]; then
    log "=== AIME-25 baselines (30 problems, ceiling check) ==="
    for seed in 0 1 2 3 4; do
        for base in cot maj8; do
            run_one aime25 "$base" "$seed"
        done
    done
fi

log "=== sweep complete ==="

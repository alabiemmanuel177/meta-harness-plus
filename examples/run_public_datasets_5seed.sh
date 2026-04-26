#!/usr/bin/env bash
# 5-seed × 2-task × 2-provider on the public AG News + emotion datasets.
# Breaks the 'all-our-benchmarks-are-hand-curated' criticism by replicating
# on two public HuggingFace classification benchmarks.

set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

mkdir -p runs/cache logs

ts() { date +"%H:%M:%S"; }

run_one_seed() {
  local api=$1
  local model=$2
  local task=$3
  local seed=$4
  local cache=$5
  local run_name="${api}_${task}_seed${seed}"
  python3 examples/rag_vs_mh_bakeoff.py \
    --api "$api" --models "$model" \
    --task "$task" \
    --run-name "$run_name" \
    --iterations 6 --proposals 8 \
    --eval-size 48 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 \
    --attribution-screen-size 12 \
    --cache-path "$cache" \
    --max-workers 8 \
    --screen-seed "$seed" \
    > "logs/${run_name}.log" 2>&1
}

run_5_seeds() {
  local api=$1
  local model=$2
  local task=$3
  local cache=$4
  echo "[$(ts)] === ${api} × ${task} 5 seeds ==="
  for seed in 0 1 2 3 4; do
    echo "[$(ts)]   seed=$seed ..."
    run_one_seed "$api" "$model" "$task" "$seed" "$cache"
  done
  echo "[$(ts)] ${api} × ${task} done."
}

# AG News: parallel pair (different APIs)
run_5_seeds "gemini" "gemini-2.5-flash-lite" "agnews" "runs/cache/gemini_agnews.jsonl" &
PID_G_AG=$!
run_5_seeds "openai" "gpt-4.1-nano" "agnews" "runs/cache/openai_agnews.jsonl" &
PID_O_AG=$!
wait $PID_G_AG $PID_O_AG

# Emotion: parallel pair
run_5_seeds "gemini" "gemini-2.5-flash-lite" "emotion" "runs/cache/gemini_emotion.jsonl" &
PID_G_EM=$!
run_5_seeds "openai" "gpt-4.1-nano" "emotion" "runs/cache/openai_emotion.jsonl" &
PID_O_EM=$!
wait $PID_G_EM $PID_O_EM

echo "[$(ts)] all 20 seeds complete; aggregating..."

aggregate_one() {
  local api=$1
  local task=$2
  local label=$3
  local out=$4
  python3 examples/multi_seed_cross_provider.py \
    --runs runs/${api}_${task}_seed0 runs/${api}_${task}_seed1 runs/${api}_${task}_seed2 runs/${api}_${task}_seed3 runs/${api}_${task}_seed4 \
    --label "$label" \
    --output "$out"
}

aggregate_one "gemini" "agnews" "Gemini 2.5-flash-lite × AG News (5 seeds)" runs/agnews_gemini_aggregate.json
aggregate_one "openai" "agnews" "OpenAI gpt-4.1-nano × AG News (5 seeds)" runs/agnews_openai_aggregate.json
aggregate_one "gemini" "emotion" "Gemini 2.5-flash-lite × emotion (5 seeds)" runs/emotion_gemini_aggregate.json
aggregate_one "openai" "emotion" "OpenAI gpt-4.1-nano × emotion (5 seeds)" runs/emotion_openai_aggregate.json

echo "[$(ts)] all done."

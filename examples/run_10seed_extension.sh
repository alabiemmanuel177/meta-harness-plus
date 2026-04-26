#!/usr/bin/env bash
# 10-seed extension: run seeds 5..9 on the 4 headline 6×8-budget cells
# (Gemini news_hard_50 + symptom_hard, OpenAI news_hard_50 + symptom_hard).
#
# Reuses existing prompt caches (runs/cache/*_big.jsonl) so most LLM
# calls hit the cache → cheap and fast. Aggregates the combined 10 seeds
# (0..9) into a single per-cell aggregate for paper-grade evidence.

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
  local run_name="${api}_${task}_seed${seed}_big"
  python3 examples/rag_vs_mh_bakeoff.py \
    --api "$api" --models "$model" \
    --task "$task" \
    --run-name "$run_name" \
    --iterations 6 --proposals 8 \
    --eval-size 50 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 \
    --attribution-screen-size 15 \
    --cache-path "$cache" \
    --max-workers 8 \
    --screen-seed "$seed" \
    > "logs/${run_name}.log" 2>&1
}

run_5_more_seeds() {
  local api=$1
  local model=$2
  local task=$3
  local cache=$4
  echo "[$(ts)] === ${api} × ${task} seeds 5..9 ==="
  for seed in 5 6 7 8 9; do
    echo "[$(ts)]   seed=$seed ..."
    run_one_seed "$api" "$model" "$task" "$seed" "$cache"
  done
  echo "[$(ts)] ${api} × ${task} done."
}

# Run all 4 cells in parallel pairs (different APIs don't conflict).
run_5_more_seeds "gemini" "gemini-2.5-flash-lite" "news_hard_50" "runs/cache/gemini_news_hard_50_big.jsonl" &
PID_G_NEWS=$!
run_5_more_seeds "openai" "gpt-4.1-nano" "news_hard_50" "runs/cache/openai_news_hard_50_big.jsonl" &
PID_O_NEWS=$!
wait $PID_G_NEWS $PID_O_NEWS

run_5_more_seeds "gemini" "gemini-2.5-flash-lite" "symptom_hard" "runs/cache/gemini_symptom_hard_big.jsonl" &
PID_G_SYM=$!
run_5_more_seeds "openai" "gpt-4.1-nano" "symptom_hard" "runs/cache/openai_symptom_hard_big.jsonl" &
PID_O_SYM=$!
wait $PID_G_SYM $PID_O_SYM

echo "[$(ts)] all 20 new seeds complete; re-aggregating with combined seeds 0..9..."

aggregate_one() {
  local api=$1
  local task=$2
  local label=$3
  local out=$4
  python3 examples/multi_seed_cross_provider.py \
    --runs runs/${api}_${task}_seed0_big runs/${api}_${task}_seed1_big runs/${api}_${task}_seed2_big runs/${api}_${task}_seed3_big runs/${api}_${task}_seed4_big runs/${api}_${task}_seed5_big runs/${api}_${task}_seed6_big runs/${api}_${task}_seed7_big runs/${api}_${task}_seed8_big runs/${api}_${task}_seed9_big \
    --label "$label (10 seeds)" \
    --output "$out"
}

aggregate_one "gemini" "news_hard_50" "Gemini 2.5-flash-lite × news_hard_50 6×8" runs/gemini_news_hard_50_big_aggregate_10seed.json
aggregate_one "openai" "news_hard_50" "OpenAI gpt-4.1-nano × news_hard_50 6×8" runs/openai_news_hard_50_big_aggregate_10seed.json
aggregate_one "gemini" "symptom_hard" "Gemini 2.5-flash-lite × symptom_hard 6×8" runs/gemini_symptom_hard_big_aggregate_10seed.json
aggregate_one "openai" "symptom_hard" "OpenAI gpt-4.1-nano × symptom_hard 6×8" runs/openai_symptom_hard_big_aggregate_10seed.json

echo "[$(ts)] all done."

#!/usr/bin/env bash
# 4-cell experiment: 5 seeds × 2 providers × 2 tasks at BIGGER search
# budget (6 iter × 8 proposals instead of 3 × 4). Goal: bigger Δ over
# RAG and cross-task replication.
#
# Usage:
#   bash examples/run_5seed_2task_2provider.sh
#
# Reads keys from .env. Expects OPENAI_API_KEY + (GOOGLE_API_KEY or
# GEMINI_API_KEY).

set -e

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

mkdir -p runs/cache logs

run_one_seed() {
  local api=$1
  local model=$2
  local task=$3
  local seed=$4
  local cache=$5
  local task_short=$6
  python3 examples/rag_vs_mh_bakeoff.py \
    --api "$api" --models "$model" \
    --task "$task" \
    --run-name "${api}_${task_short}_seed${seed}_big" \
    --iterations 6 --proposals 8 \
    --eval-size 50 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 \
    --attribution-screen-size 15 \
    --cache-path "$cache" \
    --max-workers 8 \
    --screen-seed "$seed" \
    > "logs/${api}_${task_short}_seed${seed}_big.log" 2>&1
}

run_provider_task() {
  local api=$1
  local model=$2
  local task=$3
  local task_short=$4
  local cache="runs/cache/${api}_${task_short}_big.jsonl"
  echo "[$(date +%H:%M:%S)] $api / $task_short — 5 seeds at 6×8 budget..."
  for seed in 0 1 2 3 4; do
    echo "[$(date +%H:%M:%S)]   $api/$task_short seed=$seed ..."
    run_one_seed "$api" "$model" "$task" "$seed" "$cache" "$task_short"
  done
  echo "[$(date +%H:%M:%S)] $api/$task_short done."
}

# Two parallel queues — providers don't share cache files, no contention.
# Within a provider, run all tasks sequentially so the cache builds up.
{
  run_provider_task openai gpt-4.1-nano news_hard_50  news_hard_50
  run_provider_task openai gpt-4.1-nano symptom_hard  symptom_hard
} &
PID_OPENAI=$!

{
  run_provider_task gemini gemini-2.5-flash-lite news_hard_50 news_hard_50
  run_provider_task gemini gemini-2.5-flash-lite symptom_hard symptom_hard
} &
PID_GEMINI=$!

wait $PID_OPENAI $PID_GEMINI

echo "[$(date +%H:%M:%S)] all 20 runs complete; aggregating..."

for cell in \
  "openai_news_hard_50:OpenAI gpt-4.1-nano news_hard_50 (6×8, 5 seeds)" \
  "openai_symptom_hard:OpenAI gpt-4.1-nano symptom_hard (6×8, 5 seeds)" \
  "gemini_news_hard_50:Gemini 2.5-flash-lite news_hard_50 (6×8, 5 seeds)" \
  "gemini_symptom_hard:Gemini 2.5-flash-lite symptom_hard (6×8, 5 seeds)"; do
  prefix="${cell%%:*}"
  label="${cell##*:}"
  python3 examples/multi_seed_cross_provider.py \
    --pattern "runs/${prefix}_seed*_big" \
    --label "$label" \
    --output "runs/${prefix}_big_aggregate.json"
done

echo "[$(date +%H:%M:%S)] done."

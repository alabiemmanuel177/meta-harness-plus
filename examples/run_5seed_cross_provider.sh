#!/usr/bin/env bash
# Run 5-seed news_hard_50 multi-seed bakeoff across both cloud providers
# in parallel, then aggregate for bootstrap CIs.
#
# Usage:
#   bash examples/run_5seed_cross_provider.sh
#
# Reads OPENAI_API_KEY + GOOGLE_API_KEY (or GEMINI_API_KEY) from .env or env.
# Produces:
#   runs/{openai,gemini}_news_hard_50_seed{0..4}/        per-seed artifacts
#   runs/cache/{openai,gemini}_news_hard_50.jsonl        shared per-provider caches
#   runs/cross_provider_*.json                            aggregate JSON

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
  local seed=$3
  local cache=$4
  python3 examples/rag_vs_mh_bakeoff.py \
    --api "$api" --models "$model" \
    --task news_hard_50 \
    --run-name "${api}_news_hard_50_seed${seed}" \
    --iterations 3 --proposals 4 \
    --eval-size 50 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 \
    --attribution-screen-size 15 \
    --cache-path "$cache" \
    --max-workers 8 \
    --screen-seed "$seed" \
    > "logs/${api}_seed${seed}.log" 2>&1
}

run_provider() {
  local api=$1
  local model=$2
  local cache=$3
  echo "[$(date +%H:%M:%S)] starting 5 seeds for $api ($model)..."
  for seed in 0 1 2 3 4; do
    echo "[$(date +%H:%M:%S)]   $api seed=$seed ..."
    run_one_seed "$api" "$model" "$seed" "$cache"
  done
  echo "[$(date +%H:%M:%S)] $api done."
}

run_provider "openai" "gpt-4.1-nano" "runs/cache/openai_news_hard_50.jsonl" &
PID_OPENAI=$!

run_provider "gemini" "gemini-2.5-flash-lite" "runs/cache/gemini_news_hard_50.jsonl" &
PID_GEMINI=$!

wait $PID_OPENAI $PID_GEMINI

echo "[$(date +%H:%M:%S)] all 10 seeds complete; aggregating..."

python3 examples/multi_seed_cross_provider.py \
  --pattern "runs/openai_news_hard_50_seed*" \
  --label "OpenAI gpt-4.1-nano (5 seeds)" \
  --output runs/cross_provider_openai.json

python3 examples/multi_seed_cross_provider.py \
  --pattern "runs/gemini_news_hard_50_seed*" \
  --label "Gemini 2.5 Flash Lite (5 seeds)" \
  --output runs/cross_provider_gemini.json

echo "[$(date +%H:%M:%S)] done."

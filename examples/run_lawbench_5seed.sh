#!/usr/bin/env bash
# 5-seed LawBench 2-2 multi-seed bakeoff. Both providers in parallel.
# This is the "Beat MH on its bed" cell of the four-paper-gaps matrix —
# a public dataset (open-compass/LawBench) that the original Meta-Harness
# paper specifically reports on.

set -e

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Activate venv (datasets pkg installed, urllib unblocked).
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

mkdir -p runs/cache logs

run_one_seed() {
  local api=$1
  local model=$2
  local seed=$3
  local cache=$4
  python3 examples/rag_vs_mh_bakeoff.py \
    --api "$api" --models "$model" \
    --task lawbench_2_2 \
    --run-name "${api}_lawbench_2_2_seed${seed}" \
    --iterations 3 --proposals 4 \
    --eval-size 48 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 \
    --attribution-screen-size 12 \
    --cache-path "$cache" \
    --max-workers 8 \
    --screen-seed "$seed" \
    > "logs/${api}_lawbench_2_2_seed${seed}.log" 2>&1
}

run_provider() {
  local api=$1
  local model=$2
  local cache=$3
  echo "[$(date +%H:%M:%S)] starting 5 seeds for $api lawbench_2_2 ($model)..."
  for seed in 0 1 2 3 4; do
    echo "[$(date +%H:%M:%S)]   $api lawbench_2_2 seed=$seed ..."
    run_one_seed "$api" "$model" "$seed" "$cache"
  done
  echo "[$(date +%H:%M:%S)] $api lawbench_2_2 done."
}

run_provider "openai" "gpt-4.1-nano" "runs/cache/openai_lawbench_2_2.jsonl" &
PID_OPENAI=$!

run_provider "gemini" "gemini-2.5-flash-lite" "runs/cache/gemini_lawbench_2_2.jsonl" &
PID_GEMINI=$!

wait $PID_OPENAI $PID_GEMINI

echo "[$(date +%H:%M:%S)] all 10 seeds complete; aggregating..."

python3 examples/multi_seed_cross_provider.py \
  --pattern "runs/openai_lawbench_2_2_seed*" \
  --label "OpenAI gpt-4.1-nano LawBench 2-2 (5 seeds)" \
  --output runs/lawbench_2_2_openai_aggregate.json

python3 examples/multi_seed_cross_provider.py \
  --pattern "runs/gemini_lawbench_2_2_seed*" \
  --label "Gemini 2.5-flash-lite LawBench 2-2 (5 seeds)" \
  --output runs/lawbench_2_2_gemini_aggregate.json

echo "[$(date +%H:%M:%S)] done."

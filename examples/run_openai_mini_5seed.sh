#!/usr/bin/env bash
# Bigger-model experiment: gpt-4.1-mini (one tier above nano) on the same
# news_hard_50 + symptom_hard tasks. Tests whether the framework's gains
# hold when we move past the cheapest-tier criticism.
#
# Cost estimate: ~$1 for 2 cells × 5 seeds at 6×8.

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
  local task=$1
  local seed=$2
  local cache=$3
  local run_name="openai_mini_${task}_seed${seed}"
  python3 examples/rag_vs_mh_bakeoff.py \
    --api openai --models gpt-4.1-mini \
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

run_5_seeds() {
  local task=$1
  local cache=$2
  echo "[$(ts)] === gpt-4.1-mini × ${task} 5 seeds ==="
  for seed in 0 1 2 3 4; do
    echo "[$(ts)]   seed=$seed ..."
    run_one_seed "$task" "$seed" "$cache"
  done
  echo "[$(ts)] gpt-4.1-mini × ${task} done."
}

run_5_seeds "news_hard_50" "runs/cache/openai_mini_news_hard_50.jsonl"
run_5_seeds "symptom_hard" "runs/cache/openai_mini_symptom_hard.jsonl"

echo "[$(ts)] all 10 seeds complete; aggregating..."

python3 examples/multi_seed_cross_provider.py \
  --runs runs/openai_mini_news_hard_50_seed0 runs/openai_mini_news_hard_50_seed1 runs/openai_mini_news_hard_50_seed2 runs/openai_mini_news_hard_50_seed3 runs/openai_mini_news_hard_50_seed4 \
  --label "OpenAI gpt-4.1-mini × news_hard_50 (5 seeds, 6×8)" \
  --output runs/openai_mini_news_hard_50_aggregate.json

python3 examples/multi_seed_cross_provider.py \
  --runs runs/openai_mini_symptom_hard_seed0 runs/openai_mini_symptom_hard_seed1 runs/openai_mini_symptom_hard_seed2 runs/openai_mini_symptom_hard_seed3 runs/openai_mini_symptom_hard_seed4 \
  --label "OpenAI gpt-4.1-mini × symptom_hard (5 seeds, 6×8)" \
  --output runs/openai_mini_symptom_hard_aggregate.json

echo "[$(ts)] all done."

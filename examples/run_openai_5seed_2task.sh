#!/usr/bin/env bash
# Re-run OpenAI's side of the big-budget bakeoff (which crashed mid-run
# on a SimpleFormatter bool-system_hint bug now fixed). Runs all 5 seeds
# of both news_hard_50 and symptom_hard sequentially, with 6×8 budget,
# parallel scoring, and shared cache.

set -e

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

mkdir -p runs/cache logs

run_one() {
  local task=$1
  local seed=$2
  python3 examples/rag_vs_mh_bakeoff.py \
    --api openai --models gpt-4.1-nano \
    --task "$task" \
    --run-name "openai_${task}_seed${seed}_big" \
    --iterations 6 --proposals 8 \
    --eval-size 50 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 \
    --attribution-screen-size 15 \
    --cache-path "runs/cache/openai_${task}_big.jsonl" \
    --max-workers 8 \
    --screen-seed "$seed" \
    > "logs/openai_${task}_seed${seed}_big.log" 2>&1
}

for task in news_hard_50 symptom_hard; do
  echo "[$(date +%H:%M:%S)] OpenAI / $task — 5 seeds at 6×8 budget..."
  for seed in 0 1 2 3 4; do
    echo "[$(date +%H:%M:%S)]   $task seed=$seed ..."
    run_one "$task" "$seed"
  done
  echo "[$(date +%H:%M:%S)] OpenAI/$task done."
done

echo "[$(date +%H:%M:%S)] aggregating..."
for task in news_hard_50 symptom_hard; do
  python3 examples/multi_seed_cross_provider.py \
    --pattern "runs/openai_${task}_seed*_big" \
    --label "OpenAI gpt-4.1-nano $task (6×8, 5 seeds, post-fix)" \
    --output "runs/openai_${task}_big_aggregate.json"
done
echo "[$(date +%H:%M:%S)] done."

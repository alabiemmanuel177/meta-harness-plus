#!/usr/bin/env bash
# Full-scale ablation study: 4 conditions × 5 seeds on news_hard_50 with
# OpenAI gpt-4.1-nano. Each condition disables one of the three MH++
# contributions (or none for the full baseline). Conditions:
#
#   - full        — full MH++ (Pareto + halving + LLM proposer)
#   - no-c1       — ScalarAccuracyFrontier (accuracy-only, ignores tokens/latency)
#   - no-c2       — full-eval every candidate (no successive halving)
#   - no-c3       — RandomProposer instead of attribution-guided LLM proposer
#
# Reports per-condition mean accuracy and Δ vs full MH++ to demonstrate
# which contribution carries the result.

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

API="openai"
MODEL="gpt-4.1-nano"
TASK="news_hard_50"
ITERS=3
PROPS=4
CACHE="runs/cache/ablation_${API}_${TASK}.jsonl"

ts() { date +"%H:%M:%S"; }

run_one() {
  local cond=$1
  local seed=$2
  local flag=""
  if [ "$cond" != "full" ]; then
    flag="--ablation $cond"
  fi
  python3 examples/rag_vs_mh_bakeoff.py \
    --api "$API" --models "$MODEL" \
    --task "$TASK" \
    --run-name "ablation_${cond}_seed${seed}" \
    --iterations "$ITERS" --proposals "$PROPS" \
    --eval-size 50 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 12 \
    --cache-path "$CACHE" \
    --max-workers 8 \
    --screen-seed "$seed" \
    $flag \
    > "logs/ablation_${cond}_seed${seed}.log" 2>&1
}

for COND in full no-c1 no-c2 no-c3; do
  echo "[$(ts)] === condition: $COND ==="
  for SEED in 0 1 2 3 4; do
    echo "[$(ts)]   seed=$SEED ..."
    run_one "$COND" "$SEED"
  done
  echo "[$(ts)] $COND done."
done

echo "[$(ts)] all 20 ablation runs complete; aggregating..."

python3 examples/aggregate_ablation.py \
  --task "$TASK" \
  --output runs/ablation_${API}_${TASK}_aggregate.json

echo "[$(ts)] done."

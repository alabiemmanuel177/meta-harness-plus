#!/bin/bash
# Gemini LawBench 6×8 rerun — tests small-budget hypothesis behind the
# original 3×4 negative result (-8.3pt vs RAG). If the gap closes at
# 6×8, we have evidence that the failure was budget-bound, not framework-bound.
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

echo "[$(ts)] starting 5 seeds for gemini lawbench_2_2 6×8 budget..."
for SEED in 0 1 2 3 4; do
  echo "[$(ts)]   gemini lawbench_2_2 seed=$SEED ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api gemini --models gemini-2.5-flash-lite \
    --task lawbench_2_2 \
    --run-name gemini_lawbench_2_2_seed${SEED}_big \
    --iterations 6 --proposals 8 \
    --eval-size 48 --screen-size 12 \
    --halving-k0 6 --halving-final-keep 3 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 15 \
    --cache-path runs/cache/gemini_lawbench_2_2_big.jsonl \
    --max-workers 8 \
    --screen-seed $SEED \
    > runs/gemini_lawbench_2_2_seed${SEED}_big_stdout.log 2>&1 || echo "[$(ts)] seed $SEED failed!"
done

echo "[$(ts)] gemini lawbench_2_2 6×8 done. aggregating..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/gemini_lawbench_2_2_seed0_big runs/gemini_lawbench_2_2_seed1_big runs/gemini_lawbench_2_2_seed2_big runs/gemini_lawbench_2_2_seed3_big runs/gemini_lawbench_2_2_seed4_big \
  --label "Gemini 2.5-flash-lite LawBench 2-2 6×8 (5 seeds)" \
  --output runs/lawbench_2_2_gemini_big_aggregate.json
echo "[$(ts)] all done."

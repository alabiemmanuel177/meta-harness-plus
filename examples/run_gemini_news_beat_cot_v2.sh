#!/usr/bin/env bash
# Beat-CoT-RAG v2: bigger 12×12 budget + ensemble proposer (4 children at
# different temperatures) + extra-baseline seeding. Goal: find shapes
# beating CoT-RAG (0.94) on more seeds than the 1-of-5 we got at 8×8.
#
# Cost: ~2x v1 (8×8) due to ensemble × bigger budget. Estimated ~$0.60
# and ~60-80 min wall.

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

echo "[$(ts)] starting 5 seeds for gemini news_hard_50 12×12 ensemble + extra-seeded..."
for SEED in 0 1 2 3 4; do
  echo "[$(ts)]   gemini news_hard_50 seed=$SEED ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api gemini --models gemini-2.5-flash-lite \
    --task news_hard_50 \
    --run-name gemini_news_hard_50_seed${SEED}_beatcot_v2 \
    --iterations 12 --proposals 12 \
    --eval-size 50 --screen-size 15 \
    --halving-k0 8 --halving-final-keep 4 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 15 \
    --proposer-mode ensemble --ensemble-size 4 \
    --proposer-temperature 0.6 \
    --cache-path runs/cache/gemini_news_hard_50_beatcot_v2.jsonl \
    --max-workers 8 \
    --screen-seed $SEED \
    --seed-extra-baselines \
    > runs/gemini_news_hard_50_seed${SEED}_beatcot_v2_stdout.log 2>&1 || echo "[$(ts)] seed $SEED failed!"
done

echo "[$(ts)] gemini news_hard_50 12×12 ensemble done. aggregating..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/gemini_news_hard_50_seed0_beatcot_v2 runs/gemini_news_hard_50_seed1_beatcot_v2 runs/gemini_news_hard_50_seed2_beatcot_v2 runs/gemini_news_hard_50_seed3_beatcot_v2 runs/gemini_news_hard_50_seed4_beatcot_v2 \
  --label "Gemini 2.5-flash-lite news_hard_50 12×12 ensemble + extra-seeded (5 seeds)" \
  --output runs/gemini_news_hard_50_beatcot_v2_aggregate.json
echo "[$(ts)] all done."

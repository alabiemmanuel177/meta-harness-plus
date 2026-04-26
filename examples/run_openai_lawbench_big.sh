#!/usr/bin/env bash
# OpenAI LawBench 6×8 rerun + extra-baseline seeding. The original 3×4
# run scored 0.267, losing to DSPy's 0.333. This rerun gives the search
# more budget + seeded strong baselines for the alpha-shape advantage.

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

echo "[$(ts)] OpenAI LawBench 6×8 + extra-baseline seeding (5 seeds)..."
for SEED in 0 1 2 3 4; do
  echo "[$(ts)]   seed=$SEED ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api openai --models gpt-4.1-nano \
    --task lawbench_2_2 \
    --run-name openai_lawbench_2_2_seed${SEED}_big \
    --iterations 6 --proposals 8 \
    --eval-size 48 --screen-size 12 \
    --halving-k0 6 --halving-final-keep 3 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 15 \
    --cache-path runs/cache/openai_lawbench_2_2_big.jsonl \
    --max-workers 8 \
    --screen-seed $SEED \
    --seed-extra-baselines \
    > "logs/openai_lawbench_2_2_seed${SEED}_big.log" 2>&1
done

echo "[$(ts)] OpenAI LawBench 6×8 done. Aggregating..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/openai_lawbench_2_2_seed0_big runs/openai_lawbench_2_2_seed1_big runs/openai_lawbench_2_2_seed2_big runs/openai_lawbench_2_2_seed3_big runs/openai_lawbench_2_2_seed4_big \
  --label "OpenAI gpt-4.1-nano LawBench 2-2 6×8 + extra-seeded (5 seeds)" \
  --output runs/openai_lawbench_2_2_big_aggregate.json
echo "[$(ts)] all done."

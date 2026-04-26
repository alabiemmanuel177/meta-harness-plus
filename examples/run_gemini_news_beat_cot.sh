#!/usr/bin/env bash
# Beat-CoT-RAG experiment: seed the search with CoT-RAG (and other strong
# baselines) so MH++ at minimum retains them, then run a bigger 8×8
# budget so the search has more iterations to extend past CoT-RAG.
#
# Goal: MH++ peak accuracy substantially above CoT-RAG's 0.940 on
# Gemini × news_hard_50, ideally ≥ 0.96.

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

echo "[$(ts)] starting 5 seeds for gemini news_hard_50 8×8 budget with extra-baseline seeding..."
for SEED in 0 1 2 3 4; do
  echo "[$(ts)]   gemini news_hard_50 seed=$SEED ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api gemini --models gemini-2.5-flash-lite \
    --task news_hard_50 \
    --run-name gemini_news_hard_50_seed${SEED}_beatcot \
    --iterations 8 --proposals 8 \
    --eval-size 50 --screen-size 12 \
    --halving-k0 6 --halving-final-keep 3 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 15 \
    --cache-path runs/cache/gemini_news_hard_50_beatcot.jsonl \
    --max-workers 8 \
    --screen-seed $SEED \
    --seed-extra-baselines \
    > runs/gemini_news_hard_50_seed${SEED}_beatcot_stdout.log 2>&1 || echo "[$(ts)] seed $SEED failed!"
done

echo "[$(ts)] gemini news_hard_50 8×8 + extra-seeded done. aggregating..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/gemini_news_hard_50_seed0_beatcot runs/gemini_news_hard_50_seed1_beatcot runs/gemini_news_hard_50_seed2_beatcot runs/gemini_news_hard_50_seed3_beatcot runs/gemini_news_hard_50_seed4_beatcot \
  --label "Gemini 2.5-flash-lite news_hard_50 8×8 with extra-baseline seeding (5 seeds)" \
  --output runs/gemini_news_hard_50_beatcot_aggregate.json
echo "[$(ts)] all done."

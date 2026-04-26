#!/usr/bin/env bash
# 5-seed × Gemini GSM8K MH++ search.
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs
ts() { date +"%H:%M:%S"; }

for SEED in 0 1 2 3 4; do
  echo "[$(ts)] gemini gsm8k seed=$SEED ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api gemini --models gemini-2.5-flash-lite \
    --task gsm8k --run-name gemini_gsm8k_seed${SEED} \
    --iterations 6 --proposals 8 \
    --eval-size 48 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 12 \
    --cache-path runs/cache/gemini_gsm8k.jsonl \
    --max-workers 6 --screen-seed $SEED \
    > "logs/gemini_gsm8k_seed${SEED}.log" 2>&1
done

echo "[$(ts)] aggregating..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/gemini_gsm8k_seed0 runs/gemini_gsm8k_seed1 runs/gemini_gsm8k_seed2 runs/gemini_gsm8k_seed3 runs/gemini_gsm8k_seed4 \
  --label "Gemini gemini-2.5-flash-lite × GSM8K (5 seeds, 6×8)" \
  --output runs/gsm8k_gemini_aggregate.json
echo "[$(ts)] done."

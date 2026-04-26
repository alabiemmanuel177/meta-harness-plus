#!/usr/bin/env bash
# OpenAI AG News retry with --bootstrap-instructions 16 (double the
# pool from the previous 8-pool retry) to test whether a richer
# instruction-pool closes the residual -1.5pt vs OPRO 0.896.
#
# Reads OPENAI_API_KEY from .env. Tokens used: ~2× the previous
# bootstrap-instructions 8 retry on the same model.
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs
ts() { date +"%H:%M:%S"; }

for SEED in 0 1 2 3 4; do
  echo "[$(ts)] agnews seed=$SEED with --bootstrap-instructions 16 ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api openai --models gpt-4.1-nano \
    --task agnews \
    --run-name openai_agnews_seed${SEED}_oproboot16 \
    --iterations 6 --proposals 8 \
    --eval-size 48 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 12 \
    --cache-path runs/cache/openai_agnews_oproboot16.jsonl \
    --max-workers 2 --screen-seed $SEED \
    --bootstrap-instructions 16 --seed-extra-baselines \
    > "logs/openai_agnews_seed${SEED}_oproboot16.log" 2>&1
done

echo "[$(ts)] aggregating..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/openai_agnews_seed0_oproboot16 runs/openai_agnews_seed1_oproboot16 \
         runs/openai_agnews_seed2_oproboot16 runs/openai_agnews_seed3_oproboot16 \
         runs/openai_agnews_seed4_oproboot16 \
  --label "OpenAI gpt-4.1-nano AG News + --bootstrap-instructions 16 (5 seeds)" \
  --output runs/openai_agnews_oproboot16_aggregate.json
echo "[$(ts)] done."

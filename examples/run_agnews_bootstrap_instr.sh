#!/usr/bin/env bash
# OpenAI AG News retry with --bootstrap-instructions to absorb OPRO's
# instruction-tuning specialty into MH++'s search space.
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs
ts() { date +"%H:%M:%S"; }

for SEED in 0 1 2 3 4; do
  echo "[$(ts)] agnews seed=$SEED ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api openai --models gpt-4.1-nano \
    --task agnews \
    --run-name openai_agnews_seed${SEED}_oproboot \
    --iterations 6 --proposals 8 \
    --eval-size 48 --screen-size 8 \
    --halving-k0 4 --halving-final-keep 2 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 12 \
    --cache-path runs/cache/openai_agnews_oproboot.jsonl \
    --max-workers 4 --screen-seed $SEED \
    --bootstrap-instructions 8 --seed-extra-baselines \
    > "logs/openai_agnews_seed${SEED}_oproboot.log" 2>&1
done

echo "[$(ts)] aggregating..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/openai_agnews_seed0_oproboot runs/openai_agnews_seed1_oproboot runs/openai_agnews_seed2_oproboot runs/openai_agnews_seed3_oproboot runs/openai_agnews_seed4_oproboot \
  --label "OpenAI gpt-4.1-nano AG News + bootstrap-instructions (5 seeds)" \
  --output runs/openai_agnews_oproboot_aggregate.json
echo "[$(ts)] done."

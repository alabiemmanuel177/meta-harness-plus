#!/usr/bin/env bash
# Resume lawbench-bootstrap from seed 2 (seeds 0+1 already done, both
# beating DSPy 0.333). Then aggregate.
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs
ts() { date +"%H:%M:%S"; }

for SEED in 2 3 4; do
  echo "[$(ts)] resuming seed=$SEED ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api openai --models gpt-4.1-nano \
    --task lawbench_2_2 \
    --run-name openai_lawbench_2_2_seed${SEED}_dspyboot \
    --iterations 6 --proposals 8 \
    --eval-size 48 --screen-size 12 \
    --halving-k0 6 --halving-final-keep 3 \
    --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 15 \
    --cache-path runs/cache/openai_lawbench_2_2_dspyboot.jsonl \
    --max-workers 2 --screen-seed $SEED \
    --bootstrap-demos --seed-extra-baselines \
    > "logs/openai_lawbench_2_2_seed${SEED}_dspyboot.log" 2>&1
done

echo "[$(ts)] aggregating all 5 seeds..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/openai_lawbench_2_2_seed0_dspyboot runs/openai_lawbench_2_2_seed1_dspyboot runs/openai_lawbench_2_2_seed2_dspyboot runs/openai_lawbench_2_2_seed3_dspyboot runs/openai_lawbench_2_2_seed4_dspyboot \
  --label "OpenAI gpt-4.1-nano LawBench 2-2 6×8 + bootstrap-demos (5 seeds)" \
  --output runs/openai_lawbench_2_2_dspyboot_aggregate.json
echo "[$(ts)] done."

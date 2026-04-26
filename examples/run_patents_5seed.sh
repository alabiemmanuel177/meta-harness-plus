#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs
ts() { date +"%H:%M:%S"; }

run_5_seeds() {
  local api=$1 model=$2 cache=$3
  echo "[$(ts)] === ${api} × patents 5 seeds ==="
  for seed in 0 1 2 3 4; do
    echo "[$(ts)]   ${api} patents seed=$seed"
    python3 examples/rag_vs_mh_bakeoff.py \
      --api "$api" --models "$model" \
      --task patents --run-name "${api}_patents_seed${seed}" \
      --iterations 6 --proposals 8 --eval-size 48 --screen-size 8 \
      --halving-k0 4 --halving-final-keep 2 \
      --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 12 \
      --cache-path "$cache" --max-workers 8 --screen-seed "$seed" \
      > "logs/${api}_patents_seed${seed}.log" 2>&1 || echo "[$(ts)] $api seed $seed failed"
  done
}

run_5_seeds gemini gemini-2.5-flash-lite runs/cache/gemini_patents.jsonl &
PID_G=$!
run_5_seeds openai gpt-4.1-nano runs/cache/openai_patents.jsonl &
PID_O=$!
wait $PID_G $PID_O

echo "[$(ts)] aggregating..."
for api in gemini openai; do
  python3 examples/multi_seed_cross_provider.py \
    --runs runs/${api}_patents_seed0 runs/${api}_patents_seed1 runs/${api}_patents_seed2 runs/${api}_patents_seed3 runs/${api}_patents_seed4 \
    --label "${api} × patents (5 seeds, 6×8)" \
    --output runs/patents_${api}_aggregate.json || echo "[$(ts)] $api aggregate failed"
done
echo "[$(ts)] all done."

#!/usr/bin/env bash
# Resume what aborted: openai newsgroups20 seed 4, then symptom2disease
# both providers, then aggregate.
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs
ts() { date +"%H:%M:%S"; }

echo "[$(ts)] resume: openai newsgroups20 seed 4..."
python3 examples/rag_vs_mh_bakeoff.py \
  --api openai --models gpt-4.1-nano \
  --task newsgroups20 --run-name openai_newsgroups20_seed4 \
  --iterations 6 --proposals 8 --eval-size 48 --screen-size 8 \
  --halving-k0 4 --halving-final-keep 2 \
  --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 12 \
  --cache-path runs/cache/openai_newsgroups20.jsonl \
  --max-workers 8 --screen-seed 4 \
  > logs/openai_newsgroups20_seed4_retry.log 2>&1 || echo "[$(ts)] seed 4 failed"

# Symptom2Disease: parallel pair
echo "[$(ts)] starting symptom2disease 5 seeds × 2 providers..."
run_5_seeds() {
  local api=$1 model=$2 cache=$3
  for seed in 0 1 2 3 4; do
    echo "[$(ts)]   ${api} symptom2disease seed=$seed"
    python3 examples/rag_vs_mh_bakeoff.py \
      --api "$api" --models "$model" \
      --task symptom2disease --run-name "${api}_symptom2disease_seed${seed}" \
      --iterations 6 --proposals 8 --eval-size 48 --screen-size 8 \
      --halving-k0 4 --halving-final-keep 2 \
      --eval-repeats 2 --attribution-repeats 2 --attribution-screen-size 12 \
      --cache-path "$cache" --max-workers 8 --screen-seed "$seed" \
      > "logs/${api}_symptom2disease_seed${seed}.log" 2>&1 || echo "[$(ts)] $api seed $seed failed"
  done
}
run_5_seeds gemini gemini-2.5-flash-lite runs/cache/gemini_symptom2disease.jsonl &
PID_G=$!
run_5_seeds openai gpt-4.1-nano runs/cache/openai_symptom2disease.jsonl &
PID_O=$!
wait $PID_G $PID_O

echo "[$(ts)] aggregating..."
for api in gemini openai; do
  for task in newsgroups20 symptom2disease; do
    python3 examples/multi_seed_cross_provider.py \
      --runs runs/${api}_${task}_seed0 runs/${api}_${task}_seed1 runs/${api}_${task}_seed2 runs/${api}_${task}_seed3 runs/${api}_${task}_seed4 \
      --label "${api} × ${task} (5 seeds, 6×8)" \
      --output runs/${task}_${api}_aggregate.json || echo "[$(ts)] aggregate $api $task failed"
  done
done
echo "[$(ts)] all done."

#!/usr/bin/env bash
# 5-seed × 2-provider on 20 Newsgroups (top-8) + Symptom2Disease (top-8) —
# the two public benchmarks Task 2 listed that we wired but hadn't run yet.
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs
ts() { date +"%H:%M:%S"; }

run_5_seeds() {
  local api=$1 model=$2 task=$3 cache=$4
  echo "[$(ts)] === ${api} × ${task} 5 seeds ==="
  for seed in 0 1 2 3 4; do
    echo "[$(ts)]   seed=$seed ..."
    python3 examples/rag_vs_mh_bakeoff.py \
      --api "$api" --models "$model" \
      --task "$task" \
      --run-name "${api}_${task}_seed${seed}" \
      --iterations 6 --proposals 8 \
      --eval-size 48 --screen-size 8 \
      --halving-k0 4 --halving-final-keep 2 \
      --eval-repeats 2 --attribution-repeats 2 \
      --attribution-screen-size 12 \
      --cache-path "$cache" \
      --max-workers 8 \
      --screen-seed "$seed" \
      > "logs/${api}_${task}_seed${seed}.log" 2>&1
  done
  echo "[$(ts)] ${api} × ${task} done."
}

# 20 Newsgroups: parallel pair
run_5_seeds "gemini" "gemini-2.5-flash-lite" "newsgroups20" "runs/cache/gemini_newsgroups20.jsonl" &
PID1=$!
run_5_seeds "openai" "gpt-4.1-nano" "newsgroups20" "runs/cache/openai_newsgroups20.jsonl" &
PID2=$!
wait $PID1 $PID2

# Symptom2Disease: parallel pair
run_5_seeds "gemini" "gemini-2.5-flash-lite" "symptom2disease" "runs/cache/gemini_symptom2disease.jsonl" &
PID3=$!
run_5_seeds "openai" "gpt-4.1-nano" "symptom2disease" "runs/cache/openai_symptom2disease.jsonl" &
PID4=$!
wait $PID3 $PID4

echo "[$(ts)] aggregating..."
for api in gemini openai; do
  for task in newsgroups20 symptom2disease; do
    python3 examples/multi_seed_cross_provider.py \
      --runs runs/${api}_${task}_seed0 runs/${api}_${task}_seed1 runs/${api}_${task}_seed2 runs/${api}_${task}_seed3 runs/${api}_${task}_seed4 \
      --label "${api} × ${task} (5 seeds, 6×8)" \
      --output runs/${task}_${api}_aggregate.json
  done
done
echo "[$(ts)] all done."

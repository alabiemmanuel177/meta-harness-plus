#!/usr/bin/env bash
# 5-seed × 2-provider on dair-ai/emotion (was missing from initial public_datasets run).
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs
ts() { date +"%H:%M:%S"; }

run_5_seeds() {
  local api=$1 model=$2 cache=$3
  echo "[$(ts)] === ${api} × emotion 5 seeds ==="
  for seed in 0 1 2 3 4; do
    echo "[$(ts)]   seed=$seed ..."
    python3 examples/rag_vs_mh_bakeoff.py \
      --api "$api" --models "$model" \
      --task emotion \
      --run-name "${api}_emotion_seed${seed}" \
      --iterations 6 --proposals 8 \
      --eval-size 48 --screen-size 8 \
      --halving-k0 4 --halving-final-keep 2 \
      --eval-repeats 2 --attribution-repeats 2 \
      --attribution-screen-size 12 \
      --cache-path "$cache" \
      --max-workers 8 \
      --screen-seed "$seed" \
      > "logs/${api}_emotion_seed${seed}.log" 2>&1
  done
  echo "[$(ts)] ${api} × emotion done."
}

run_5_seeds "gemini" "gemini-2.5-flash-lite" "runs/cache/gemini_emotion.jsonl" &
PID_G=$!
run_5_seeds "openai" "gpt-4.1-nano" "runs/cache/openai_emotion.jsonl" &
PID_O=$!
wait $PID_G $PID_O

echo "[$(ts)] aggregating..."
python3 examples/multi_seed_cross_provider.py \
  --runs runs/openai_emotion_seed0 runs/openai_emotion_seed1 runs/openai_emotion_seed2 runs/openai_emotion_seed3 runs/openai_emotion_seed4 \
  --label "OpenAI gpt-4.1-nano × emotion (5 seeds)" \
  --output runs/emotion_openai_aggregate.json
python3 examples/multi_seed_cross_provider.py \
  --runs runs/gemini_emotion_seed0 runs/gemini_emotion_seed1 runs/gemini_emotion_seed2 runs/gemini_emotion_seed3 runs/gemini_emotion_seed4 \
  --label "Gemini gemini-2.5-flash-lite × emotion (5 seeds)" \
  --output runs/emotion_gemini_aggregate.json
echo "[$(ts)] all done."

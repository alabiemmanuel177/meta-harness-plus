#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p runs/cache
ts() { date +"%H:%M:%S"; }

for task in symptom_hard agnews emotion lawbench_2_2; do
  echo "[$(ts)] TextGrad on openai/$task ..."
  python3 examples/textgrad_baseline.py \
    --api openai --model gpt-4.1-nano \
    --task $task --epochs 2 --n-train-batch 4 \
    --cache-path runs/cache/textgrad_openai_${task}.jsonl \
    --output runs/textgrad_baseline_openai_${task}.json \
    > /tmp/textgrad_${task}.log 2>&1 || echo "[$(ts)] $task failed"
done
echo "[$(ts)] all TextGrad done."

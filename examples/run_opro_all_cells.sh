#!/usr/bin/env bash
# Run OPRO baseline on all OpenAI cells.
set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p runs/cache
ts() { date +"%H:%M:%S"; }

for task in symptom_hard agnews emotion lawbench_2_2; do
  echo "[$(ts)] OPRO on openai/$task ..."
  python3 examples/opro_baseline.py \
    --api openai --model gpt-4.1-nano \
    --task $task --iterations 4 --proposals 4 \
    --cache-path runs/cache/opro_openai_${task}.jsonl \
    --output runs/opro_baseline_openai_${task}.json \
    > /tmp/opro_${task}.log 2>&1 || echo "[$(ts)] $task failed"
done
echo "[$(ts)] all OPRO done."

#!/usr/bin/env bash
# GSM8K BARE multi-seed on Gemini for variance estimate.
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache
ts() { date +"%H:%M:%S"; }

# Single 50-eval run is enough for a directional result; the smoke
# already showed 0.40 on n=20. Now run on n=50 for a cleaner number.
echo "[$(ts)] GSM8K BARE on Gemini, n_eval=50 ..."
python3 examples/gsm8k_smoke.py \
  --api gemini --model gemini-2.5-flash-lite \
  --n-train 30 --n-eval 50 \
  --cache-path runs/cache/gsm8k_gemini.jsonl \
  --output runs/gsm8k_gemini_50eval.json
echo "[$(ts)] done."

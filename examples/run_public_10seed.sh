#!/usr/bin/env bash
# Standardized 10-seed runner for any public-dataset cell.
#
# Usage:
#   API=openai MODEL=gpt-4.1-nano TASK=agnews \
#       bash examples/run_public_10seed.sh
#   API=gemini MODEL=gemini-2.5-flash-lite TASK=newsgroups20 \
#       bash examples/run_public_10seed.sh
#
# Reads required:
#   API     — openai|gemini|anthropic|ollama
#   MODEL   — model id
#   TASK    — name in TASK_FACTORIES (run with --task --help to list)
# Optional:
#   SEEDS         — number of seeds (default 10)
#   ITER PROP     — iterations × proposals (default 6 × 8)
#   EVAL_SIZE     — final eval size (default 48)
#   SCREEN_SIZE   — screen subset size (default 8)
#   K0 KFINAL     — halving sizes (default 4 / 2)
#   ATTRIB_REPEATS — attribution repeats (default 2)
#   MAX_WORKERS   — parallel score workers (default 4)
#   EXTRA_FLAGS   — passed through to rag_vs_mh_bakeoff.py
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p runs/cache logs

: "${API:?must set API= openai|gemini|anthropic|ollama}"
: "${MODEL:?must set MODEL= model id}"
: "${TASK:?must set TASK= task name}"
: "${SEEDS:=10}"
: "${ITER:=6}"
: "${PROP:=8}"
: "${EVAL_SIZE:=48}"
: "${SCREEN_SIZE:=8}"
: "${K0:=4}"
: "${KFINAL:=2}"
: "${ATTRIB_REPEATS:=2}"
: "${MAX_WORKERS:=4}"
: "${EXTRA_FLAGS:=}"

ts() { date +"%H:%M:%S"; }

echo "[$(ts)] starting $SEEDS-seed run for ${API}_${MODEL} on ${TASK}..."

RUN_NAMES=()
for ((SEED=0; SEED<SEEDS; SEED++)); do
  RUN_NAME="${API}_${TASK}_seed${SEED}_10s"
  RUN_NAMES+=("runs/${RUN_NAME}")
  echo "[$(ts)] ${TASK} seed=${SEED} ..."
  python3 examples/rag_vs_mh_bakeoff.py \
    --api "${API}" --models "${MODEL}" \
    --task "${TASK}" \
    --run-name "${RUN_NAME}" \
    --iterations "${ITER}" --proposals "${PROP}" \
    --eval-size "${EVAL_SIZE}" --screen-size "${SCREEN_SIZE}" \
    --halving-k0 "${K0}" --halving-final-keep "${KFINAL}" \
    --eval-repeats 2 --attribution-repeats "${ATTRIB_REPEATS}" \
    --attribution-screen-size 12 \
    --cache-path "runs/cache/${API}_${TASK}_10s.jsonl" \
    --max-workers "${MAX_WORKERS}" --screen-seed "${SEED}" \
    ${EXTRA_FLAGS} \
    > "logs/${RUN_NAME}.log" 2>&1
done

echo "[$(ts)] aggregating $SEEDS seeds..."
python3 examples/multi_seed_cross_provider.py \
  --runs "${RUN_NAMES[@]}" \
  --label "${API} ${MODEL} × ${TASK} (${SEEDS} seeds, ${ITER}×${PROP})" \
  --output "runs/${API}_${TASK}_10seed_aggregate.json"
echo "[$(ts)] done. Wrote runs/${API}_${TASK}_10seed_aggregate.json"

# V10 SWE-bench harness Makefile
# Phase 0 targets — see docs/V10_DESIGN.md §10.

PYTHON ?= PYTHONPATH=. .venv/bin/python3
PYTEST ?= .venv/bin/python3 -m pytest

.PHONY: help firewall smoke build-split test-v10 ablate eval clean-v10-cache

help:
	@echo "V10 Phase 0 targets:"
	@echo "  make firewall        - run the contamination firewall test"
	@echo "  make test-v10        - run all V10 unit tests"
	@echo "  make build-split     - rebuild splits/dev_50.json"
	@echo "  make smoke           - end-to-end Phase 0 shakedown (one instance)"
	@echo "  make clean-v10-cache - delete trajectories/v10_* and runs/v10_*"
	@echo
	@echo "Phase 1+ targets (placeholders):"
	@echo "  make ablate          - run the dev-50 ablation matrix"
	@echo "  make eval            - run the post-submission grader on a run"

firewall:
	$(PYTEST) tests/test_no_oracle_leak.py -q

test-v10:
	$(PYTEST) tests/test_no_oracle_leak.py \
	          tests/test_dataset_and_conventions.py \
	          tests/test_sandbox_and_cache.py -q

build-split:
	$(PYTHON) scripts/build_dev_split.py

smoke: firewall
	$(PYTHON) scripts/smoke_phase0.py

smoke-exec: firewall
	$(PYTHON) scripts/smoke_phase0.py --exec --timeout-s 480

smoke-negative: firewall
	$(PYTHON) scripts/negative_smoke.py

smoke-strict: firewall
	@echo "[smoke-strict] verifying global import-time cache hygiene"
	@$(PYTHON) -c "import harness; print('[smoke-strict] harness import OK — V10_EXCLUSIVE_DIRS clean')"
	@echo "[smoke-strict] running smoke under strict hygiene"
	$(PYTHON) scripts/smoke_phase0.py
	@echo "[smoke-strict] OK"

eval-fast: firewall
	@echo "[eval-fast] dev-50 retrieval eval, GPU + 2 parallel workers + batch 256"
	@echo "[eval-fast]   NOTE: --workers 4 crashed the host on 2026-04-30 (30 GB RAM,"
	@echo "[eval-fast]   4 × bge-large + 4 × Docker memory_gb=4 + 4 × Python = ~25 GB);"
	@echo "[eval-fast]   --workers 2 is the verified-safe parallel target. Can override:"
	@echo "[eval-fast]     PYTHONPATH=. .venv/bin/python3 scripts/retrieval_eval_dev50.py --workers N ..."
	$(PYTHON) scripts/retrieval_eval_dev50.py \
		--workers 2 --batch-size 256 \
		--no-shortlist --include-traceback

eval-bm25-only: firewall
	@echo "[eval-bm25-only] dev-50 retrieval eval, BM25 only (no embedding, no traceback)"
	$(PYTHON) scripts/retrieval_eval_dev50.py --no-embedding

verify-images:
	$(PYTHON) scripts/verify_images.py

verify-images-dev50:
	$(PYTHON) scripts/verify_images.py --split splits/dev_50.json

clean-v10-cache:
	rm -rf trajectories/v10_* runs/v10_*

ablate:
	@echo "Phase 1 placeholder — run dev-50 sweeps once Phase 1 lands."
	@false

eval:
	@echo "Post-submission grader; see harness/eval.py."
	@false

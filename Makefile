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
	@echo "[eval-fast] dev-50 retrieval eval, GPU + 1 worker + batch 256"
	@echo "[eval-fast]   NOTE: --workers > 1 fails on this AMD ROCm hardware:"
	@echo "[eval-fast]     --workers 4 crashed the host (RAM exhaustion);"
	@echo "[eval-fast]     --workers 2 hits 'GPU Hang' from concurrent CUDA-equivalent calls"
	@echo "[eval-fast]     (AMD ROCm PyTorch does not serialize multi-process kernels cleanly)."
	@echo "[eval-fast]   See V10_DESIGN.md §9 for the full evidence chain."
	$(PYTHON) scripts/retrieval_eval_dev50.py \
		--workers 1 --batch-size 256 \
		--no-shortlist --include-traceback

eval-bm25-only: firewall
	@echo "[eval-bm25-only] dev-50 retrieval eval, BM25 only (no embedding, no traceback)"
	$(PYTHON) scripts/retrieval_eval_dev50.py --no-embedding

verify-images:
	$(PYTHON) scripts/verify_images.py

verify-images-dev50:
	$(PYTHON) scripts/verify_images.py --split splits/dev_50.json

# test_500 launch shape (commit 14a). The actual launch happens via
# the long-horizon batch orchestration; these targets document the
# reproducible invocation. NOT typically run as `make eval-test500-*`
# — they're a reference for what the long-running command looks like.
eval-test500-deepseek: firewall
	@echo "[eval-test500-deepseek] full pipeline + DeepSeek reranker (~24h, ~\$$1.20)"
	@echo "[eval-test500-deepseek] background launch recommended:"
	@echo "    nohup PYTHONPATH=. .venv/bin/python3 scripts/retrieval_eval_dev50.py \\"
	@echo "        --split splits/test_500.json \\"
	@echo "        --no-shortlist --include-traceback --rerank \\"
	@echo "        --workers 1 \\"
	@echo "        > /tmp/test500_deepseek.log 2>&1 &"
	$(PYTHON) scripts/retrieval_eval_dev50.py \
		--split splits/test_500.json \
		--no-shortlist --include-traceback --rerank \
		--workers 1

eval-test500-sonnet: firewall
	@echo "[eval-test500-sonnet] Sonnet rerank ablation, reuses retrieval cache (~4h, ~\$$20)"
	$(PYTHON) scripts/retrieval_eval_dev50.py \
		--split splits/test_500.json \
		--no-shortlist --include-traceback --rerank \
		--reranker-model claude-sonnet-4-5 \
		--signature-suffix _test500_sonnet \
		--audit-suffix _test500_sonnet \
		--workers 1

clean-v10-cache:
	rm -rf trajectories/v10_* runs/v10_*

ablate:
	@echo "Phase 1 placeholder — run dev-50 sweeps once Phase 1 lands."
	@false

eval:
	@echo "Post-submission grader; see harness/eval.py."
	@false

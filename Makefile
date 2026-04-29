# V10 SWE-bench harness Makefile
# Phase 0 targets — see docs/V10_DESIGN.md §10.

PYTHON ?= .venv/bin/python3
PYTEST ?= $(PYTHON) -m pytest

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

clean-v10-cache:
	rm -rf trajectories/v10_* runs/v10_*

ablate:
	@echo "Phase 1 placeholder — run dev-50 sweeps once Phase 1 lands."
	@false

eval:
	@echo "Post-submission grader; see harness/eval.py."
	@false

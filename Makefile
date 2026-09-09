PYTHON ?= python
RUN ?=

.PHONY: help verify compile test summarize release-check require-run baseline nested tabm ctboost

help:
	@printf '%s\n' \
	  'make verify     Validate local data and artifact contracts' \
	  'make compile    Compile Python entry points' \
	  'make test       Run offline unit tests' \
	  'make release-check  Check staged file contents for data, artifacts, and credentials' \
	  'make summarize  Print local run metrics' \
	  'make baseline RUN=<new-id>  Train the canonical XGBoost baseline' \
	  'make nested RUN=<new-id>    Train the ten-fold XGBoost variant' \
	  'make tabm RUN=<new-id>      Train TabM after creating five-fold assignments' \
	  'make ctboost RUN=<new-id>   Train CTBoost after creating five-fold assignments'

verify:
	$(PYTHON) tools/verify_project.py

compile:
	$(PYTHON) -m compileall -q *.py tools tests

test:
	$(PYTHON) -m unittest discover -s tests -v

release-check:
	$(PYTHON) tools/check_git_release.py

summarize:
	$(PYTHON) tools/summarize_runs.py

require-run:
	$(if $(strip $(RUN)),,$(error Set RUN to a new experiment ID, e.g. make baseline RUN=repro_xgb_01))

baseline: require-run
	$(PYTHON) train_encoded.py --model xgb --run "$(RUN)"

nested: require-run
	$(PYTHON) train_10fold_nested.py --run "$(RUN)"

tabm: require-run
	$(PYTHON) train_tabm.py --run "$(RUN)"

ctboost: require-run
	$(PYTHON) train_ctboost.py --run "$(RUN)"

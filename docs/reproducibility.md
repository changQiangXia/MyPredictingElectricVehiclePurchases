# Reproducibility Guide

## Reference Environment

Recorded on September 9, 2026, from the existing machine, not from a fresh installation:

| Component | Reference |
| --- | --- |
| Training Python | 3.10.8 |
| GPU | One NVIDIA RTX 4090, 24 GB |
| NVIDIA driver | 535.129.03 |
| PyTorch | 2.1.2+cu121, CUDA runtime 12.1 |
| Tree/validation packages | Direct version pins in `requirements.txt` |
| Optional models | Direct version pins in `requirements-optional.txt` |
| Separate Kaggle CLI Python | 3.11.16 |
| Kaggle / requests in CLI environment | 2.2.4 / 2.34.2 |

For the reference Linux/Python 3.10 neural environment, PyTorch can be installed from its CUDA wheel index before installing the optional requirements:

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-optional.txt
```

The dependency files are not complete environment locks. They omit transitive pins and system libraries, do not guarantee newer Python support, and have not been tested in a fresh environment. Do not replace the working environment just to tidy the repository. Use `requirements-kaggle.txt` in a separate Python 3.11 environment for the authenticated helper; the older Kaggle package in the training environment is not the proven submission setup.

## Data Contract

The runners expect `train.csv`, `test.csv`, and `sample_submission.csv` in the repository root. The local verifier checks the target name, ID order, row count, uniqueness, and allowed target values. Data is intentionally ignored by Git.

The optional `original/` source dataset is used only by experiments that explicitly request it. It is not needed for the canonical nested-encoding baseline.

## Fold Contract

The default comparison surface is a fixed `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. `scripts/train_encoded.py` generates `artifacts/folds_seed42.npy` if absent. `scripts/train_tabm.py`, `scripts/train_ctboost.py`, `scripts/train_pairwise.py`, and `scripts/evaluate_offline.py` require this file to exist already. The ten-fold runner creates/uses a separate `artifacts/folds_seed42_10.npy` and does not satisfy that prerequisite.

Keep data row order and folds together. The current scripts use fixed fold paths, so a new run ID alone does not create a new validation split. Changing data or fold definitions needs a separate workspace or explicit fold-path changes. Row-paired OOF comparisons are possible across fold schemes, but the five-fold slices are not the ten-fold model's training folds; reported conditional uncertainty excludes shared-training and model-selection effects.

## Encoding Contract

Categorical and numeric keys are represented in the feature builder. Target encoders are fitted only on the outer training partition. The training partition itself is cross-fitted by scikit-learn's `TargetEncoder`; validation and test rows receive transformations from encoders fitted without their labels.

Frequency statistics in the canonical feature builder are deliberately transductive over unlabeled train plus test features. The canonical runner records that choice in its config; legacy runners do not all record it. Review this choice against the competition rules before release.

## Run Artifacts

A completed model run should contain:

- `config.json`: code/data/fold hashes and model parameters;
- `fold_*.json`: per-fold AUC, selected iteration, and elapsed time;
- `oof.npy`: one prediction per training row;
- `test.npy`: averaged test prediction;
- `submission.csv`: ID-aligned prediction file;
- `metrics.json`: pooled AUC, fold mean/std, and submission checksum;
- model and preprocessing files when the runner persists them.

These artifacts stay local and are ignored by Git. Important existing exceptions:

- `scripts/train_10fold_nested.py` saves fold predictions, metrics, encoders, and final prediction arrays, but no model weights or `config.json`. Its resume path does not verify config/data/code hashes or saved validation indices. Never reuse one of its old run directories after changing inputs or arguments; exact checkpoint inference cannot be reconstructed without retraining.
- TabM, CTBoost, and pairwise runners may stop after selected folds and record `status=pilot`. A fold score is not a full OOF score; only all-fold completion emits final OOF/test/submission files.
- Historical `run_*.py` scripts predate this contract. Some write flat artifacts or fixed run directories. External submission-only artifacts and public OOF blends do not establish reproducibility of the upstream model or folds.
- Offline comparisons may write only a report. Exported blend candidates record their source prediction hashes, but do not contain new model weights.

`tools/verify_project.py` checks data schema/IDs, Python syntax, selected required prediction files, stored submission checksums, and credential patterns without network calls. It does not reload every model, inspect every OOF array's shape, recompute all AUCs, or prove leakage freedom. `tools/check_git_release.py` checks staged blob contents, not commit history, licenses, or live sharing rules.

## Comparing Models

Use OOF predictions for rank blending and error analysis. Keep weight searches small, record the anchor, candidate, weight, fold deltas, and selection caveat. A public leaderboard score is an external observation, not a replacement for OOF validation. Never train a stacker from in-sample predictions.

## Re-running A Candidate

```bash
python scripts/train_encoded.py --model xgb --run repro_xgb_01
python tools/summarize_runs.py
python tools/verify_project.py
```

Use an unused run ID. The canonical, TabM, CTBoost, and pairwise runners have configuration checks, but their hashes are not a complete dependency/source lock. They may reject old runs after code changes; do not edit saved configs to bypass those checks. GPU/library changes can also change results even when a config matches.

`scripts/audit_distribution.py` needs existing anchor/candidate OOF artifacts named inside the script. `scripts/confirm_margin.py` reads the raw data and trains fresh models on a new split; it does not need existing prediction files. Both use fixed output directories and can overwrite previous diagnostics. Neither is part of the cheap checks, and neither should be run merely to inspect this repository.

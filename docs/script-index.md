# Script Index

All experiment entry points live in `scripts/` so the repository root stays clean. Each script resolves the repository root as `Path(__file__).resolve().parents[1]`, so run them from the repository root; data and `artifacts/` resolve relative to that root. This index separates maintained workflows from historical experiments.

## Primary Workflow

| Script | Purpose | Main output |
| --- | --- | --- |
| `scripts/train_encoded.py` | Five-fold nested target encoding with XGBoost, LightGBM, or CatBoost | `artifacts/<run>/` |
| `scripts/train_10fold_nested.py` | Legacy ten-fold XGBoost confirmation; no model weights/config and weak resume checks | `artifacts/<new-run>/` |
| `scripts/evaluate_offline.py` | Rank-blend candidates, fold deltas, and paired AUC comparison | `artifacts/<run>/report.json` |
| `scripts/submit_checked.py` | Validate and create a real Kaggle submission after explicit authorization | `artifacts/<run>/submission_receipt.json` |

`scripts/submit_checked.py` is deliberately absent from the Makefile. Generic verification, CI, or reproduction commands must never submit.

`scripts/evaluate_offline.py` needs completed OOF/test predictions plus the shared five-fold assignment. Its default anchor refers to a local historical artifact; a new workspace must pass an explicit `--anchor` as in the README.

## Optional Models

| Script | Purpose | Notes |
| --- | --- | --- |
| `scripts/train_tabm.py` | TabM ensemble with fold-isolated preprocessing | Needs optional PyTorch/TabM dependencies and CUDA |
| `scripts/train_ctboost.py` | CTBoost diversity model | Uses a fixed round count and persists reloadable models |
| `scripts/train_pairwise.py` | XGBoost pairwise-ranking pilot | Experimental; one large global ranking group is expensive |
| `scripts/train_formula.py` | Add the reverse-engineered formula as ordinary features | Wraps the canonical runner |
| `scripts/train_conditional.py` | Add conditional target-encoding keys | Wraps the canonical runner |
| `scripts/train_income_te.py` | Income-target-encoding ablation | Wraps the canonical runner |
| `scripts/train_income_cond.py` | Conditional income ablation | Wraps the canonical runner |
| `scripts/train_orig_lift.py` | Source-data distribution lift feature | Requires optional source data |
| `scripts/train_orig_model.py` | Source-data supervised prior | Requires optional source data |

TabM, CTBoost, and pairwise runners load `artifacts/folds_seed42.npy` without creating it. Run the canonical five-fold workflow first. Selected-fold pilots are not complete OOF models. Use new run IDs rather than names of preserved historical experiments.

## Diagnostics

| Script | Purpose |
| --- | --- |
| `scripts/audit_distribution.py` | Trains a drift model and uses existing anchor/candidate OOF for error slices; fixed output path |
| `scripts/confirm_margin.py` | Trains new models from raw data with inner-only early stopping; fixed output path |
| `tools/verify_project.py` | Offline data/code/artifact/credential contract checks |
| `tools/summarize_runs.py` | Local metrics, pilot/report distinctions, and cached submission scores; optional ignored report |
| `tools/check_git_release.py` | Inspects staged blobs for forbidden paths, large/binary files, and credential patterns |

## Historical Baselines

The following scripts are retained for provenance. They predate the canonical artifact contract and are not the recommended starting point for new work:

- `scripts/run_baseline.py`: native-categorical CatBoost baseline;
- `scripts/run_lgbm_features.py`: early frequency/digit LightGBM experiment;
- `scripts/run_gam.py`: spline plus one-hot logistic experiment;
- `scripts/run_rank_logistic.py`: rank/quantile logistic experiment.

Historical scripts may write files directly under `artifacts/` rather than a run directory. Do not use them as templates for new experiments.

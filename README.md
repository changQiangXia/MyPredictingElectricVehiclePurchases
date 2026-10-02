# Playground Series S6E9 Research Workspace

Local experiments and reproduction entry points for Kaggle's **Playground Series - Season 6, Episode 9: Predicting Electric Vehicle Purchases**.

This workspace is intentionally data-free at the repository boundary. Competition data, downloaded public artifacts, model checkpoints, predictions, submission receipts, API credentials, and Kaggle notebook downloads remain local and are ignored by Git.

## Current Status

The competition closed on September 30, 2026. Final result: **public 0.94691 (6th at the close)**
and **private 0.94551 (213th, tied with eight teams)**. The two selected final submissions were the
deadline-day public blend `B10` (`ref 56711566`) and the honest OOF candidate `priv_cand_v5_rl3_win`
(`ref 56557482`); Kaggle scored the better of the two. The best private-robust file held that day was
an OOF stack (`ref 56710849`, public 0.94672 / private 0.94565, roughly 79th place) and was not
selected. See [`docs/postmortem-s6e9.md`](docs/postmortem-s6e9.md) for the full deadline-day
selection postmortem and the reusable candidate-selection checklist.

This is now a code-only archive. Competition data, downloaded public artifacts, model checkpoints,
predictions, submission receipts, and credentials remain local and ignored by Git. The cached
competition rules still require publicly shared competition code to be posted on Kaggle's
discussion forum or notebooks; read [`docs/release-checklist.md`](docs/release-checklist.md)
before changing repository visibility or publishing.

The canonical local workflow has fixed stratified folds, nested target encoding, OOF predictions, fold metrics, model checkpoints, and submission-file validation.

## Layout

```text
.
|-- train_encoded.py              # Canonical nested-encoding XGBoost/LightGBM/CatBoost runner
|-- train_10fold_nested.py        # Ten-fold XGBoost variant
|-- train_tabm.py                 # TabM neural tabular experiment
|-- train_ctboost.py              # CTBoost diversity experiment
|-- train_pairwise.py             # XGBoost pairwise-ranking experiment
|-- evaluate_offline.py           # OOF-only rank blending and paired comparisons
|-- audit_distribution.py         # Target-free train/test drift audit
|-- confirm_margin.py             # New margin-link split of previously explored data
|-- submit_checked.py             # Explicitly gated Kaggle submission helper
|-- run_*.py / train_*.py         # Earlier baselines and ablations
|-- tools/                        # Local verification and run-report helpers
|-- docs/                         # Reproduction and release documentation
|-- train.csv                     # Local only, ignored by Git
|-- test.csv                      # Local only, ignored by Git
|-- sample_submission.csv         # Local only, ignored by Git
`-- artifacts/                    # Local only, ignored by Git
```

The scripts currently live at the repository root because they resolve data and artifacts relative to `Path(__file__).parent`. This preserves existing artifact paths. Historical runs are not guaranteed to resume with today's code; use fresh run IDs for reproduction.

## Environment

The reference training environment uses Python 3.10.8. Install the core tree/validation packages with:

```bash
python -m pip install -r requirements.txt
```

TabM and CTBoost are optional experiment groups. Install a suitable CUDA-enabled PyTorch build first as described in [`docs/reproducibility.md`](docs/reproducibility.md), then:

```bash
python -m pip install -r requirements-optional.txt
```

The reference machine has one RTX 4090 (24 GB). XGBoost/CatBoost/TabM runners use CUDA; the canonical LightGBM runner uses CPU. These requirements pin direct packages, not every transitive/system dependency, and a clean installation has not been validated.

Kaggle authentication is separate and not required for training or offline checks. The working CLI uses Python 3.11.16 with `requirements-kaggle.txt`; install it into a separate environment only when downloads or explicitly authorized submissions are needed.

## Data Setup

Place the competition files in the repository root:

```text
train.csv
test.csv
sample_submission.csv
```

Optional public source data belongs in `original/` for experiments that explicitly use it. Do not commit either location. Obtain data through Kaggle's official download flow and review the competition rules and data license first.

## Reproduction

Run the cheap checks first:

```bash
python tools/verify_project.py
python -m compileall -q *.py tools tests
python -m unittest discover -s tests -v
python tools/check_git_release.py
```

Without local data, use `python tools/verify_project.py --skip-data --skip-artifacts`. The release checker reads the Git index, so new or edited files must be staged before that check.

Run the canonical nested-encoding baseline with an unused run ID:

```bash
python train_encoded.py \
  --model xgb \
  --run repro_xgb_01
```

Run the ten-fold variant in a new directory. This legacy runner saves predictions and encoders, but no model weights or config file:

```bash
python train_10fold_nested.py --run repro_xgb_10fold_01
```

The completed canonical run above creates `artifacts/folds_seed42.npy`, which must already exist for these optional models. They do not generate folds on a clean clone:

```bash
python train_tabm.py --run repro_tabm_01
python train_ctboost.py --run repro_ctboost_01
python evaluate_offline.py \
  --run repro_comparison_01 \
  --anchor repro_xgb_01 \
  --models repro_tabm_01 repro_ctboost_01
```

The comparison command needs completed OOF/test artifacts from all named runs. It applies competition-specific edge rules and is exploratory, not a generic clean validation benchmark.

Summarize local runs, pilots, diagnostics, and saved Kaggle receipts without calling Kaggle:

```bash
python tools/summarize_runs.py
python tools/summarize_runs.py --write-internal-report
```

The optional report is written only to ignored `docs/internal/run-index.md`. Scores come from cached receipts, not a live leaderboard. The local handoff and parent strategy notes remain outside the tracked repository contents.

## Submission Safety

`submit_checked.py` validates IDs, columns, row count, finite probabilities, SHA256, and Kaggle quota before uploading. Creating a real submission additionally requires the explicit `--confirm-upload` flag. Do not run it as part of a generic CI job or Make target. Submission requires an explicit human decision and a separate Kaggle credential setup.

Never put `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, `kaggle.json`, cookies, or any other credential in this repository, shell scripts, notebooks, issue comments, or commit history.

## Validation Contract

- The primary metric is ROC AUC.
- The default fold assignment is five stratified folds with seed `42`, stored locally as `artifacts/folds_seed42.npy`.
- Target encoding is fitted inside each outer training fold; validation rows never provide their targets to an encoder.
- New model candidates should save OOF/test predictions, per-fold metrics, configuration, model/preprocessor files, and a submission checksum. Historical exceptions and partial-fold pilots are listed in the reproduction guide; a public submission alone is not a locally validated model.
- Rank blending is evaluated from OOF predictions only. A public leaderboard result is recorded separately from local OOF.
- No candidate is considered validated solely because a single public score improved.

## Scope And Attribution

Public notebooks and discussions were used as scouting references, not as a license to copy private code, data, or artifacts. Public reference links are recorded in [`docs/sources.md`](docs/sources.md); adaptation and license review is still required before release. The included MIT license is a proposed license for original project code only, pending the owner's release decision. It does not relicense data or third-party material.

See [`docs/script-index.md`](docs/script-index.md) for the maintained, diagnostic, and historical entry points.

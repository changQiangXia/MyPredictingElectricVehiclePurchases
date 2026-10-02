# GitHub Release Checklist

This competition is active in the current workspace. Treat the following as a release gate, not optional cleanup.

## Competition Rule Gate

The cached competition rules state that:

- private sharing of competition code is restricted between separate teams unless a team merger occurs;
- public competition code sharing is permitted only under the competition's public-sharing conditions and must be made available on the relevant Kaggle forum or notebook for the benefit of all competitors;
- open-source code used to generate a submission must have an OSI-approved license that does not limit commercial use;
- external data and tools must be publicly/equally accessible or satisfy the competition's reasonableness requirements;
- the competition has a daily submission limit and a final-submission limit.

The exact current rules are saved locally in ignored `intel/pages.json`; re-read the live Kaggle page before any public GitHub push because rules can change.

## Never Commit

- `train.csv`, `test.csv`, `sample_submission.csv`, source datasets, archives, or any derived data table;
- OOF/test predictions, model checkpoints, encoders, feature dumps, or private Kaggle dataset outputs;
- `submission_receipt.json`, score exports containing private metadata, cookies, `kaggle.json`, access tokens, `.env` files, or shell history;
- downloaded public notebooks or copied source material whose license and attribution have not been checked;
- local absolute paths, account credentials, private dataset handles, or unpublished leaderboard strategy.

The repository `.gitignore` is intentionally conservative. Verify with `git status --ignored` and inspect the staged file list before committing.

## Before Public Visibility

1. Re-read the live competition rules and confirm whether public GitHub publication satisfies the required Kaggle forum/notebook sharing condition.
2. Confirm repository visibility and the exact people granted access. A private repository is not permission to share competition work with non-team collaborators.
3. Check every dependency and any adapted public code for an OSI-approved license and attribution requirements.
4. Remove current competition-specific internal notes if they reveal unpublished strategy or data-derived artifacts.
5. Run `python tools/verify_project.py --skip-data --skip-artifacts` on the intended code-only working tree; this command does not inspect staged blobs or prove model reproducibility.
6. Run `python tools/check_git_release.py` against the Git index.
7. Inspect the exact staged file list and search it for `KGAT_`, `KAGGLE_API_TOKEN`, `access_token`, and private URLs.
8. Confirm the owner's intended license and copyright attribution. The included `LICENSE` is a proposed MIT license for original project code, not a completed third-party license audit. It does not relicense competition data, public notebook code, or third-party artifacts. See `docs/sources.md` for references still requiring review.

The index checker reads the exact staged contents, including files changed or removed in the working tree after staging. It is a conservative pattern/path check, not a guarantee that every secret format is detected. It does not scan earlier commits; review history before a later push if commits have been added.

## GitHub Authentication

Use SSH keys or a GitHub credential helper. Do not paste a GitHub token into a remote URL, script, notebook, commit message, or chat transcript. Confirm the remote URL and repository visibility before pushing. For Kaggle, `scripts/submit_checked.py` requires `--confirm-upload`; never weaken that guard to make automation convenient.

## After The Competition

At the end of the competition, record the final public/private result separately from local OOF, review the winning-artifact selection, and only then decide whether to publish a sanitized code-only snapshot. Keep the raw data and data-derived artifacts out of the public repository unless both Kaggle rules and the data license permit redistribution.

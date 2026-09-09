"""Validate local data, code, and candidate artifact contracts without network calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from .check_git_release import SECRET_PATTERNS
except ImportError:
    from check_git_release import SECRET_PATTERNS


TARGET = "Will_Buy_EV"
ID = "id"
EXPECTED_ROWS = 286_571


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_data(root: Path) -> list[str]:
    errors: list[str] = []
    paths = {name: root / name for name in ("train.csv", "test.csv", "sample_submission.csv")}
    if not all(path.exists() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.exists()]
        return [f"missing local data file(s): {', '.join(missing)}"]

    import pandas as pd

    train = pd.read_csv(paths["train.csv"])
    test = pd.read_csv(paths["test.csv"])
    sample = pd.read_csv(paths["sample_submission.csv"])
    if ID not in train or ID not in test or ID not in sample:
        errors.append("all data files must contain an id column")
    if TARGET not in train:
        errors.append(f"train.csv must contain {TARGET}")
    if TARGET in test or TARGET not in sample:
        errors.append("test/sample target-column contract is invalid")
    if len(test) != EXPECTED_ROWS or len(sample) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS:,} test/sample rows, found {len(test):,}/{len(sample):,}")
    if ID in test and ID in sample and not test[ID].equals(sample[ID]):
        errors.append("test.csv and sample_submission.csv id order differs")
    if ID in train and not train[ID].is_unique:
        errors.append("train.csv ids are not unique")
    if ID in test and not test[ID].is_unique:
        errors.append("test.csv ids are not unique")
    if TARGET in train and (train[TARGET].isna().any() or set(train[TARGET].unique()) != {"Yes", "No"}):
        errors.append(f"{TARGET} contains values outside Yes/No")
    if list(sample) != [ID, TARGET]:
        errors.append("sample_submission.csv must have exactly id and Will_Buy_EV columns")
    if TARGET in train and list(train.drop(columns=TARGET)) != list(test):
        errors.append("train/test feature names or order differ")
    if ID in train and ID in test and train[ID].isin(test[ID]).any():
        errors.append("train/test ids overlap")
    return errors


def check_code(root: Path) -> list[str]:
    errors: list[str] = []
    sources = sorted(root.glob("*.py")) + sorted((root / "tools").glob("*.py")) + sorted((root / "tests").glob("*.py"))
    if not sources:
        return ["no Python entry points found"]
    for source in sources:
        try:
            compile(source.read_text(encoding="utf-8"), str(source), "exec")
        except SyntaxError as exc:
            errors.append(f"{source}: {exc}")
    return errors


def check_secrets(root: Path) -> list[str]:
    errors: list[str] = []
    excluded_dirs = {".git", "artifacts", "intel", "original", "internal", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in excluded_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json", ".ini"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible credential in {path}")
                break
    return errors


def check_artifacts(root: Path) -> list[str]:
    errors: list[str] = []
    artifacts = root / "artifacts"
    if not artifacts.exists():
        return []
    for metrics_path in sorted(artifacts.glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON: {metrics_path}: {exc}")
            continue
        is_oof_run = metrics.get("status") == "local_candidate" or isinstance(
            metrics.get("oof_auc"), (int, float)
        )
        if is_oof_run:
            run_dir = metrics_path.parent
            for name in ("oof.npy", "test.npy", "submission.csv"):
                if not (run_dir / name).exists():
                    errors.append(f"{metrics_path}: completed run missing {name}")
        elif metrics.get("status") == "complete" and not (metrics_path.parent / "submission.csv").exists():
            errors.append(f"{metrics_path}: completed external run missing submission.csv")
        submission = metrics_path.parent / "submission.csv"
        if submission.exists() and metrics.get("submission_sha256"):
            if digest(submission) != metrics["submission_sha256"]:
                errors.append(f"{metrics_path}: submission.csv checksum mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    if not args.skip_data:
        errors.extend(check_data(root))
    errors.extend(check_code(root))
    errors.extend(check_secrets(root))
    if not args.skip_artifacts:
        errors.extend(check_artifacts(root))
    if errors:
        print("PROJECT_CHECK_FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print(json.dumps({"status": "ok", "root": str(root),
                      "data_checked": not args.skip_data,
                      "artifacts_checked": not args.skip_artifacts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

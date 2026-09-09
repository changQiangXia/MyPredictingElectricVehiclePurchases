"""Reject forbidden files or credentials from the Git index before a push."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


FORBIDDEN_PARTS = {
    "artifacts",
    "intel",
    "original",
    ".kaggle",
    "internal",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
}
FORBIDDEN_NAMES = {
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "kaggle.json",
    "access_token",
    "submission_receipt.json",
    ".env",
}
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".zip",
    ".parquet",
    ".feather",
    ".npy",
    ".npz",
    ".joblib",
    ".ubj",
    ".cbm",
    ".pt",
    ".pth",
    ".onnx",
    ".bin",
    ".safetensors",
    ".log",
    ".pem",
    ".key",
    ".pyc",
    ".ipynb",
}
SECRET_PATTERNS = (
    re.compile(r"KGAT_[A-Za-z0-9]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
)
MAX_FILE_BYTES = 2 * 1024 * 1024


def git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def check_index(root: Path) -> dict:
    entries = [entry for entry in git_output(root, "ls-files", "--stage", "-z").split(b"\0") if entry]
    errors: list[str] = []
    checked_text = 0
    total_bytes = 0
    if not entries:
        errors.append("Git index is empty; stage the intended repository files first")
    for entry in entries:
        metadata, filename = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        relative = PurePosixPath(filename.decode("utf-8"))
        if stage != "0":
            errors.append(f"unresolved merge in index: {relative}")
            continue
        if mode not in {"100644", "100755"}:
            errors.append(f"symlink or non-regular Git entry: {relative}")
            continue
        if set(relative.parts).intersection(FORBIDDEN_PARTS):
            errors.append(f"forbidden competition/private directory: {relative}")
            continue
        if relative.name in FORBIDDEN_NAMES or relative.name.startswith(".env."):
            errors.append(f"forbidden competition/private file: {relative}")
            continue
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES or relative.name.startswith("offline_experiments_"):
            errors.append(f"forbidden data/model or internal report file: {relative}")
            continue
        size = int(git_output(root, "cat-file", "-s", object_id))
        total_bytes += size
        if size > MAX_FILE_BYTES:
            errors.append(f"file exceeds the code-only size limit: {relative}")
            continue
        # Read the staged blob, even if the working copy was edited or removed.
        blob = git_output(root, "cat-file", "blob", object_id)
        try:
            content = blob.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"binary or non-UTF8 content in code-only repository: {relative}")
            continue
        if "\0" in content:
            errors.append(f"binary content in code-only repository: {relative}")
            continue
        checked_text += 1
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                errors.append(f"possible credential in tracked file: {relative}")
                break
    return {"status": "failed" if errors else "ok", "tracked_files": len(entries),
            "text_files_scanned": checked_text, "indexed_bytes": total_bytes, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        report = check_index(args.root.resolve())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"RELEASE_CHECK_FAILED: cannot inspect Git index ({type(exc).__name__})")
        return 1
    if report["errors"]:
        print("RELEASE_CHECK_FAILED")
        for error in report["errors"]:
            print(f"- {error}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

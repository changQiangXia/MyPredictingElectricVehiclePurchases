"""Summarize local runs and cached Kaggle receipts; never calls Kaggle."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def public_score(receipt: dict[str, Any]) -> str:
    kaggle = receipt.get("kaggle")
    score = kaggle.get("public_score") if isinstance(kaggle, dict) else None
    return "" if score is None else str(score)


def submitted(receipt: dict[str, Any], metrics: dict[str, Any]) -> str:
    status = str(receipt.get("status", ""))
    if status in {"scored", "complete", "SubmissionStatus.COMPLETE"} or receipt.get("submission_ref"):
        return "yes"
    if status:
        return "attempted"
    if metrics.get("submitted") is False:
        return "no"
    return "unknown"


def run_kind(metrics: dict[str, Any], run_dir: Path, receipt: dict[str, Any]) -> str:
    status = str(metrics.get("status", ""))
    if status in {"failed", "pilot"}:
        return status
    if status == "local_candidate":
        return "local_candidate"
    has_oof = (run_dir / "oof.npy").exists()
    has_submission = (run_dir / "submission.csv").exists() or bool(receipt)
    if isinstance(metrics.get("oof_auc"), (int, float)) and has_oof:
        return "local_oof"
    if has_oof:
        return "oof_unscored"
    if has_submission:
        return "submission_only"
    if (run_dir / "report.json").exists():
        return "report_only"
    return "artifact_only"


def collect_rows(artifacts: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(path for path in artifacts.glob("*") if path.is_dir()):
        metrics = load_json(run_dir / "metrics.json")
        report = load_json(run_dir / "report.json")
        receipt = load_json(run_dir / "submission_receipt.json")
        if not (metrics or report or receipt):
            continue
        metadata = metrics or report
        kaggle = receipt.get("kaggle")
        kaggle = kaggle if isinstance(kaggle, dict) else {}
        rows.append({
            "run": run_dir.name,
            "status": metrics.get("status", ""),
            "kind": run_kind(metrics, run_dir, receipt),
            "oof_auc": metrics.get("oof_auc"),
            "public_score": public_score(receipt),
            "submitted": submitted(receipt, metadata),
            "submission_ref": receipt.get("submission_ref") or kaggle.get("ref", ""),
            "source": metadata.get("source", metadata.get("anchor", metadata.get("base", ""))),
            "artifact": run_dir.relative_to(artifacts.parent).as_posix(),
        })
    return rows


def auc_text(value: Any) -> str:
    return f"{value:.9f}" if isinstance(value, (int, float)) else ""


def write_internal_report(root: Path, rows: list[dict[str, Any]]) -> Path:
    destination = root / "docs" / "internal" / "run-index.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Local Run Index",
        "",
        "> Generated from ignored local `artifacts/` on demand. This file is not a release artifact and is never submitted to Kaggle.",
        "> Scores are cached submission receipts; an empty score means no public score was recorded locally.",
        "",
        f"Generated at {datetime.now(timezone.utc).isoformat()}; {len(rows)} run directories.",
        "",
        "| Run | Kind | Local OOF AUC | Public score | Submitted | Ref | Source | Artifact |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        artifact_link = quote("../../" + row["artifact"] + "/", safe="/")
        values = [
            row["run"], row["kind"], auc_text(row["oof_auc"]), row["public_score"],
            row["submitted"], str(row["submission_ref"]), row["source"], f"[files]({artifact_link})",
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    lines.extend([
        "",
        "## Reading This Table",
        "",
        "- `local_oof` has a pooled OOF metric and saved OOF array; it may still be historical and not reproducible with current code.",
        "- `oof_unscored` has saved OOF predictions but no recorded pooled OOF AUC; inspect its report before comparing it.",
        "- `local_candidate` was selected from reused OOF predictions and should not be treated as an untouched validation result.",
        "- `pilot` is incomplete and must not be compared as a full five-fold model.",
        "- `submission_only` has a submission/receipt but no local OOF array; it does not identify the upstream model or its folds.",
        "- `report_only` is a diagnostic or comparison report, not an exported full OOF candidate.",
        "- `unknown` submission status means no explicit local record; absence of a receipt does not establish non-submission.",
        "- The index reads saved metadata and checks file presence; it does not recalculate AUC, validate array contents, or list flat historical artifacts.",
        "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--write-internal-report", action="store_true",
                        help="write ignored docs/internal/run-index.md")
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    rows = collect_rows(artifacts)
    if not rows:
        print("No local run metrics, reports, or receipts found.")
        return 0
    print(f"{'run':42} {'kind':20} {'oof_auc':12} {'public':8} {'submitted':10} ref")
    print("-" * 112)
    for row in rows:
        print(f"{row['run']:42} {row['kind']:20} {auc_text(row['oof_auc']):12} "
              f"{row['public_score']:8} {row['submitted']:10} {row['submission_ref']}")
    if args.write_internal_report:
        destination = write_internal_report(artifacts.parent, rows)
        print(f"Wrote ignored local report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

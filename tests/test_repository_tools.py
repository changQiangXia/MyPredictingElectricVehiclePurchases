import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.check_git_release import check_index
from tools.summarize_runs import collect_rows, write_internal_report


class GitIndexTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init", "-q")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def stage(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.git("add", "--", name)
        return path

    def test_clean_index_ignores_untracked_artifacts(self):
        self.stage("README.md", "# Example\n")
        (self.root / "train.csv").write_text("local only\n")
        self.assertEqual(check_index(self.root)["status"], "ok")

    def test_staged_credential_is_detected_after_worktree_is_cleaned(self):
        fake = "K" + "GAT_" + "0" * 32
        path = self.stage("config.ini", f"credential={fake}\n")
        path.write_text("credential=unset\n")
        report = check_index(self.root)
        self.assertEqual(report["status"], "failed")
        self.assertIn("possible credential", " ".join(report["errors"]))
        self.assertNotIn(fake, str(report))

    def test_index_is_checked_when_worktree_file_is_missing(self):
        self.stage("README.md", "# Example\n").unlink()
        self.assertEqual(check_index(self.root)["status"], "ok")

    def test_forbidden_files_are_rejected_when_staged(self):
        for name in (".env.production", "artifacts/model.json", "docs/internal/notes.md", "train.csv"):
            self.stage(name, "example\n")
        report = check_index(self.root)
        self.assertEqual(len(report["errors"]), 4)

    def test_extensionless_file_is_scanned(self):
        self.stage(".gitattributes", "K" + "GAT_" + "0" * 32)
        self.assertEqual(check_index(self.root)["status"], "failed")


class SubmissionGuardTest(unittest.TestCase):
    def invoke(self, arguments):
        script = Path(__file__).resolve().parents[1] / "scripts" / "submit_checked.py"
        # No third-party packages are available with -S, so this also detects
        # accidental Kaggle imports before argument parsing and the upload guard.
        return subprocess.run([sys.executable, "-S", str(script), *arguments], capture_output=True, text=True, timeout=10)

    def test_help_does_not_import_kaggle(self):
        result = self.invoke(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--confirm-upload", result.stdout)

    def test_upload_requires_explicit_flag_before_authentication(self):
        result = self.invoke(["--run", "example"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Upload disabled", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class RunSummaryTest(unittest.TestCase):
    def test_receipt_without_metric_does_not_establish_absence_of_oof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            run = artifacts / "external"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text(json.dumps({
                "run": "external", "status": "complete",
            }), encoding="utf-8")
            (run / "submission_receipt.json").write_text(json.dumps({
                "status": "scored", "submission_ref": 123,
                "kaggle": {"public_score": "0.90000"},
            }), encoding="utf-8")
            rows = collect_rows(artifacts)
            self.assertEqual(rows[0]["kind"], "submission_only")
            self.assertEqual(rows[0]["public_score"], "0.90000")
            self.assertEqual(rows[0]["submitted"], "yes")
            report = write_internal_report(root, rows)
            self.assertIn("submission_only", report.read_text(encoding="utf-8"))
            # Presence is sufficient for the inventory, which never loads arrays.
            (run / "oof.npy").touch()
            rows = collect_rows(artifacts)
            self.assertEqual(rows[0]["kind"], "oof_unscored")
            self.assertIsNone(rows[0]["oof_auc"])

    def test_pilot_and_diagnostic_are_not_full_oof_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            pilot, diagnostic = artifacts / "pilot", artifacts / "diagnostic"
            pilot.mkdir(parents=True)
            diagnostic.mkdir()
            (pilot / "metrics.json").write_text(json.dumps({
                "status": "pilot", "folds": [{"fold": 0, "auc": 0.9}], "submitted": False,
            }), encoding="utf-8")
            (diagnostic / "report.json").write_text(json.dumps({
                "submitted": False, "linear_auc": 0.9,
            }), encoding="utf-8")
            rows = {row["run"]: row for row in collect_rows(artifacts)}
            self.assertEqual(rows["pilot"]["kind"], "pilot")
            self.assertEqual(rows["diagnostic"]["kind"], "report_only")
            self.assertTrue(all(row["oof_auc"] is None for row in rows.values()))


if __name__ == "__main__":
    unittest.main()

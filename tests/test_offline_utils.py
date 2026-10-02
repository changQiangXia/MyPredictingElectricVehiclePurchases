import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# The experiment scripts live in scripts/ and import each other as flat modules
# (for example ``from train_encoded import ROOT``). Mirror that layout here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_offline import auc_placements, edges, paired_delta  # noqa: E402
from tools.check_git_release import SECRET_PATTERNS


class OfflineEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.target = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        self.anchor = np.array([0.1, 0.4, 0.3, 0.4, 0.2, 0.8, 0.7, 0.4, 0.5, 0.6, 0.8, 0.5])
        self.candidate = np.array([0.1, 0.5, 0.2, 0.4, 0.3, 0.8, 0.9, 0.2, 0.6, 0.5, 0.7, 0.4])

    def test_placements_match_sklearn_with_ties(self):
        positive, negative = auc_placements(self.target, self.anchor)
        expected = roc_auc_score(self.target, self.anchor)
        self.assertAlmostEqual(positive.mean(), expected, places=12)
        self.assertAlmostEqual(negative.mean(), expected, places=12)

    def test_paired_delta_matches_auc_difference(self):
        result = paired_delta(
            self.target,
            self.anchor,
            self.candidate,
            auc_placements(self.target, self.anchor),
        )
        expected = roc_auc_score(self.target, self.candidate) - roc_auc_score(self.target, self.anchor)
        self.assertAlmostEqual(result["delta"], expected, places=12)
        self.assertGreater(result["paired_se"], 0)

    def test_hard_edges_override_predictions(self):
        frame = pd.DataFrame(
            {
                "Annual_Income_USD": [200_000, 35_000, 80_000, 90_000],
                "Daily_Commute_km": [10, 10, 90, 20],
            }
        )
        actual = edges(frame, np.full(4, 0.5))
        np.testing.assert_array_equal(actual, np.array([1.0, 0.0, 0.0, 0.5]))

    def test_repository_does_not_contain_literal_token(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertFalse(any(pattern.search(text) for pattern in SECRET_PATTERNS), path)


if __name__ == "__main__":
    unittest.main()

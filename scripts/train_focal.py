"""A focal-loss contrast on the unchanged nested XGBoost feature pipeline."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
import xgboost

import train_encoded as runner


def focal_derivatives(y, margin, gamma):
    signed = 2 * np.asarray(y) - 1
    correct = np.clip(expit(signed * margin), 1e-7, 1 - 1e-7)
    wrong = 1 - correct
    logp = np.log(correct)
    weight = wrong ** gamma
    gradient = signed * weight * (gamma * correct * logp - wrong)
    hessian = weight * correct * (
        gamma * logp * (wrong - gamma * correct) + (2 * gamma + 1) * wrong)
    return gradient, hessian


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    custom, remaining = parser.parse_known_args()
    if custom.focal_gamma < 0:
        raise ValueError("Focal gamma must be nonnegative")
    run = remaining[remaining.index("--run") + 1]
    out = runner.ROOT / "artifacts" / run
    out.mkdir(exist_ok=False)
    manifest = {
        "hypothesis": "A different per-example loss may expose ranking information hidden by logloss's weighting of easy examples",
        "gamma": custom.focal_gamma,
        "alpha": "None: no class weighting",
        "hessian": "Analytic focal Hessian clipped at 1e-6 for nonconvex tails",
        "script_sha256": runner.digest(Path(__file__)),
        "runner_sha256": runner.digest(Path(runner.__file__)),
        "features": "Unchanged canonical nested target encoding",
        "blend_weights_prespecified": [0.05, 0.1, 0.2], "submitted": False,
    }
    (out / "objective_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    classifier = xgboost.XGBClassifier
    def objective(y, pred):
        g, h = focal_derivatives(y, pred, custom.focal_gamma)
        return g, np.maximum(h, 1e-6)
    def build_classifier(**kwargs):
        kwargs["objective"] = objective
        return classifier(**kwargs)
    xgboost.XGBClassifier = build_classifier
    sys.argv = [sys.argv[0]] + remaining
    runner.main()

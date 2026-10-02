"""Build low-weight external probes on top of the locked local candidate.

These artifacts are for rank/disagreement inspection only.  Nina, Ravi and
Talha provide test submissions without compatible OOF predictions, so this
script records no local AUC and never uploads a file.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parent
TARGET = "Will_Buy_EV"
BASE = "offline_wave7_tabm_transfer25_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values), method="average") / len(values)


def apply_edges(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    values = values.copy()
    values[frame.Annual_Income_USD.ge(170537).to_numpy()] = 1.0
    values[(frame.Annual_Income_USD.between(31004, 41970) |
            frame.Daily_Commute_km.ge(83)).to_numpy()] = 0.0
    return values


def main() -> None:
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    base_frame = pd.read_csv(ROOT / "artifacts" / BASE / "submission.csv")
    sources = {
        "nina_latest": ROOT / "intel/live_20260910/nina2025_output/submission.csv",
        "ravi_latest": ROOT / "intel/live_20260910/ravi_output/submission.csv",
        "talha_latest": ROOT / "artifacts/talha_current_20260909/submission.csv",
    }
    assert sample.id.equals(test.id) and base_frame.id.equals(sample.id)
    base = rank(base_frame[TARGET].to_numpy())
    ext = {}
    for name, path in sources.items():
        frame = pd.read_csv(path)
        assert frame.columns.tolist() == sample.columns.tolist()
        assert frame.id.equals(sample.id)
        ext[name] = rank(frame[TARGET].to_numpy())

    candidates = {}
    for name in sources:
        for weight in (0.02, 0.05, 0.10):
            candidates[f"offline_wave8_{name}_{int(weight * 100):02d}_v1"] = {
                "external": name,
                "external_weight": weight,
            }
    candidates["offline_wave8_nina_ravi_talha05_v1"] = {
        "nina_latest": 0.05,
        "ravi_latest": 0.05,
        "talha_latest": 0.05,
    }

    base_raw = base_frame[TARGET].to_numpy()
    for run, spec in candidates.items():
        if "external" in spec:
            name, weight = spec["external"], spec["external_weight"]
            prediction = (1.0 - weight) * base + weight * ext[name]
            weights = {"base": 1.0 - weight, name: weight}
        else:
            total = sum(spec.values())
            prediction = (1.0 - total) * base
            weights = {"base": 1.0 - total, **spec}
            for name, weight in spec.items():
                prediction += weight * ext[name]
        prediction = apply_edges(test, prediction)
        out = ROOT / "artifacts" / run
        out.mkdir(parents=True, exist_ok=False)
        sub = sample.copy()
        sub[TARGET] = prediction
        sub.to_csv(out / "submission.csv", index=False)
        edge_mask = (
            test.Annual_Income_USD.ge(170537) |
            test.Annual_Income_USD.between(31004, 41970) |
            test.Daily_Commute_km.ge(83)
        ).to_numpy()
        metrics = {
            "run": run,
            "status": "external_probe",
            "submitted": False,
            "public_auc": None,
            "base": BASE,
            "weights": weights,
            "rank_spearman_vs_base": float(spearmanr(base, prediction).statistic),
            "changed_fraction_vs_base": float(np.mean(np.abs(prediction - base_raw) > 1e-15)),
            "edge_rows_reapplied": int(edge_mask.sum()),
            "source_sha256": {BASE: digest(ROOT / "artifacts" / BASE / "submission.csv")},
            "script_sha256": digest(Path(__file__)),
            "test_sha256": digest(ROOT / "test.csv"),
            "submission_sha256": digest(out / "submission.csv"),
            "limitation": "No compatible OOF for external submissions; rank/disagreement only",
        }
        for name, path in sources.items():
            if name in weights:
                metrics["source_sha256"][name] = digest(path)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out / "builder.py").write_bytes(Path(__file__).read_bytes())
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()

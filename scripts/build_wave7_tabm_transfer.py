"""Build local TabM-transfer blends from the public-score wave5 anchor.

This script only writes local artifacts.  The TabM weight is fixed before
looking at the candidate's test rows, and the historical wave5 edge map is
reapplied after blending so the candidate keeps the anchor's explicit support
constraints.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
TARGET = "Will_Buy_EV"
ANCHOR = "offline_wave5_conservative"
TABM = "tabm_formula_nested_10fold_reg_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(x: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(x), method="average") / len(x)


def apply_edges(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    values = values.copy()
    values[frame.Annual_Income_USD.ge(170537).to_numpy()] = 1.0
    values[(frame.Annual_Income_USD.between(31004, 41970) |
            frame.Daily_Commute_km.ge(83)).to_numpy()] = 0.0
    return values


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = train[TARGET].eq("Yes").to_numpy()
    folds = np.load(ROOT / "artifacts/folds_seed42.npy")
    anchor_oof = np.load(ROOT / "artifacts" / ANCHOR / "oof.npy")
    anchor_test = np.load(ROOT / "artifacts" / ANCHOR / "test.npy")
    tabm_oof = np.load(ROOT / "artifacts" / TABM / "oof.npy")
    tabm_test = np.load(ROOT / "artifacts" / TABM / "test.npy")
    assert anchor_oof.shape == tabm_oof.shape == (len(train),)
    assert anchor_test.shape == tabm_test.shape == (len(test),)
    assert np.isfinite(anchor_oof).all() and np.isfinite(anchor_test).all()
    assert np.isfinite(tabm_oof).all() and np.isfinite(tabm_test).all()

    anchor_rank_oof, anchor_rank_test = rank(anchor_oof), rank(anchor_test)
    tabm_rank_oof, tabm_rank_test = rank(tabm_oof), rank(tabm_test)
    base_auc = float(roc_auc_score(y, anchor_oof))
    records = []
    # Transfer weights were fixed from the completed test-style diagnosis;
    # 0.25 is the conservative center of the useful range on the anchor.
    for weight in (0.20, 0.25, 0.30):
        name = f"offline_wave7_tabm_transfer{int(weight * 100):02d}_v1"
        out = ROOT / "artifacts" / name
        out.mkdir(parents=True, exist_ok=False)
        oof = (1.0 - weight) * anchor_rank_oof + weight * tabm_rank_oof
        test_pred = (1.0 - weight) * anchor_rank_test + weight * tabm_rank_test
        oof = apply_edges(train, oof)
        test_pred = apply_edges(test, test_pred)
        np.save(out / "oof.npy", oof)
        np.save(out / "test.npy", test_pred)
        sample = pd.read_csv(ROOT / "sample_submission.csv")
        assert sample.id.equals(test.id) and sample.id.is_unique
        sample[TARGET] = test_pred
        sample.to_csv(out / "submission.csv", index=False)
        auc = float(roc_auc_score(y, oof))
        fold_auc = [float(roc_auc_score(y[folds == f], oof[folds == f]))
                    for f in range(5)]
        metrics = {
            "run": name,
            "status": "local_candidate",
            "submitted": False,
            "public_auc": None,
            "anchor": ANCHOR,
            "tabm": TABM,
            "tabm_rank_weight": weight,
            "oof_auc": auc,
            "anchor_oof_auc": base_auc,
            "delta_vs_anchor": auc - base_auc,
            "fold_auc": fold_auc,
            "edge_policy": "wave5 income/commute edges reapplied after rank blend",
            "weights_fixed_from": "diagnose_tabm_transfer_v1 mean heldout results",
            "sources": {
                ANCHOR: {
                    "oof_sha256": digest(ROOT / "artifacts" / ANCHOR / "oof.npy"),
                    "test_sha256": digest(ROOT / "artifacts" / ANCHOR / "test.npy"),
                },
                TABM: {
                    "oof_sha256": digest(ROOT / "artifacts" / TABM / "oof.npy"),
                    "test_sha256": digest(ROOT / "artifacts" / TABM / "test.npy"),
                },
            },
            "script_sha256": digest(Path(__file__)),
            "train_sha256": digest(ROOT / "train.csv"),
            "test_sha256": digest(ROOT / "test.csv"),
            "oof_sha256": digest(out / "oof.npy"),
            "test_prediction_sha256": digest(out / "test.npy"),
            "submission_sha256": digest(out / "submission.csv"),
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out / "builder.py").write_bytes(Path(__file__).read_bytes())
        records.append(metrics)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

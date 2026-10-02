"""Build conservative local seed-transfer blends; never uploads."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
TARGET = "Will_Buy_EV"
BASE = "offline_wave7_tabm_transfer25_v1"
SEED_MODEL = "xgb_probit_10fold_seed2026_v1"


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
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    folds = np.load(ROOT / "artifacts/folds_seed42.npy")
    y = train[TARGET].eq("Yes").to_numpy()
    base_oof = np.load(ROOT / "artifacts" / BASE / "oof.npy")
    base_test = np.load(ROOT / "artifacts" / BASE / "test.npy")
    seed_oof = np.load(ROOT / "artifacts" / SEED_MODEL / "oof.npy")
    seed_test = np.load(ROOT / "artifacts" / SEED_MODEL / "test.npy")
    assert base_oof.shape == seed_oof.shape == (len(train),)
    assert base_test.shape == seed_test.shape == (len(test),)
    base_rank_oof, base_rank_test = rank(base_oof), rank(base_test)
    seed_rank_oof, seed_rank_test = rank(seed_oof), rank(seed_test)
    base_auc = float(roc_auc_score(y, base_oof))
    records = []
    for weight in (0.05, 0.10, 0.15):
        name = f"offline_wave9_seed_transfer{int(weight * 100):02d}_v1"
        out = ROOT / "artifacts" / name
        out.mkdir(parents=True, exist_ok=False)
        oof = apply_edges(train, (1.0 - weight) * base_rank_oof + weight * seed_rank_oof)
        test_pred = apply_edges(test, (1.0 - weight) * base_rank_test + weight * seed_rank_test)
        np.save(out / "oof.npy", oof)
        np.save(out / "test.npy", test_pred)
        sample = pd.read_csv(ROOT / "sample_submission.csv")
        assert sample.id.equals(test.id)
        sample[TARGET] = test_pred
        sample.to_csv(out / "submission.csv", index=False)
        auc = float(roc_auc_score(y, oof))
        base_fold = [float(roc_auc_score(y[folds == f], base_oof[folds == f])) for f in range(5)]
        fold_auc = [float(roc_auc_score(y[folds == f], oof[folds == f])) for f in range(5)]
        metrics = {
            "run": name,
            "status": "local_candidate",
            "submitted": False,
            "public_auc": None,
            "base": BASE,
            "seed_model": SEED_MODEL,
            "seed_model_rank_weight": weight,
            "oof_auc": auc,
            "base_oof_auc": base_auc,
            "delta_vs_base": auc - base_auc,
            "base_fold_auc": base_fold,
            "fold_auc": fold_auc,
            "fold_delta": [a - b for a, b in zip(fold_auc, base_fold)],
            "test_style_caveat": "Seed transfer gain is expected to shrink after fold averaging; weight kept conservative",
            "sources": {
                BASE: {"oof_sha256": digest(ROOT / "artifacts" / BASE / "oof.npy"),
                       "test_sha256": digest(ROOT / "artifacts" / BASE / "test.npy")},
                SEED_MODEL: {"oof_sha256": digest(ROOT / "artifacts" / SEED_MODEL / "oof.npy"),
                             "test_sha256": digest(ROOT / "artifacts" / SEED_MODEL / "test.npy")},
            },
            "script_sha256": digest(Path(__file__)),
            "train_sha256": digest(ROOT / "train.csv"),
            "test_sha256": digest(ROOT / "test.csv"),
            "submission_sha256": digest(out / "submission.csv"),
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out / "builder.py").write_bytes(Path(__file__).read_bytes())
        records.append(metrics)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

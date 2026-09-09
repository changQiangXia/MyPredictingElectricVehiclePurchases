from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parent
SEED = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"
ID = "id"


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = (train[TARGET] == "Yes").astype(np.int8)
    features = [c for c in test.columns if c != ID]
    cat_cols = [c for c in features if train[c].dtype == "object"]
    X = train[features].copy()
    X_test = test[features].copy()

    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    fold_scores = []
    start = time.time()

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(folds.split(X, y)):
        fold_start = time.time()
        model = CatBoostClassifier(
            iterations=1200,
            depth=8,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED + fold,
            l2_leaf_reg=5.0,
            random_strength=1.0,
            bootstrap_type="Bayesian",
            verbose=200,
            allow_writing_files=False,
            task_type="GPU",
            devices="0",
        )
        model.fit(
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            cat_features=cat_cols,
            eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
            early_stopping_rounds=100,
        )
        va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        te_pred = model.predict_proba(X_test)[:, 1]
        oof[va_idx] = va_pred
        test_pred += te_pred / N_SPLITS
        score = roc_auc_score(y.iloc[va_idx], va_pred)
        fold_scores.append(float(score))
        print(json.dumps({"fold": fold, "auc": score, "best_iteration": model.get_best_iteration(), "seconds": time.time() - fold_start}))

    cv = float(roc_auc_score(y, oof))
    summary = {
        "model": "catboost_gpu_baseline",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "features": features,
        "categorical_features": cat_cols,
        "fold_auc": fold_scores,
        "cv_auc_oof": cv,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "elapsed_seconds": time.time() - start,
    }
    np.save(out / "oof_catboost_baseline.npy", oof)
    np.save(out / "test_catboost_baseline.npy", test_pred)
    pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(out / "submission_catboost_baseline.csv", index=False)
    (out / "baseline_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

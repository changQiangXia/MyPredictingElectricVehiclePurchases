"""Pilot feature selection inspired by cdeotte's gain-reranked XGB recipe.

This is an exploratory fold-0 check. Candidate columns are selected from the
already trained fold-0 canonical model, then retrained at its fixed iteration
count. It is not promoted to a full candidate unless a later nested selection
protocol confirms it.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import train_encoded as runner

ROOT = runner.ROOT
RUN = "xgb_selected_pilot_v1"


def main():
    tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = tr[runner.TARGET].eq("Yes").to_numpy(dtype="int8")
    folds = np.load(ROOT / "artifacts/folds_seed42.npy")
    ti, vi = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    base, keys, _ = runner.features(tr, te, False)
    X, Xt = base.iloc[:len(tr)].to_numpy(), base.iloc[len(tr):].to_numpy()
    K, Kt = keys.iloc[:len(tr)].to_numpy(), keys.iloc[len(tr):].to_numpy()
    train_parts, valid_parts, test_parts = [X[ti]], [X[vi]], [Xt]
    for smoothing in ("auto", 10.0, 100.0):
        enc = TargetEncoder(target_type="binary", smooth=smoothing, cv=5,
                            shuffle=True, random_state=42)
        train_parts.append(enc.fit_transform(K[ti], y[ti]).astype("float32"))
        valid_parts.append(enc.transform(K[vi]).astype("float32"))
        test_parts.append(enc.transform(Kt).astype("float32"))
    A, B, C = [np.concatenate(p, axis=1) for p in (train_parts, valid_parts, test_parts)]
    imp = pd.read_csv(ROOT / "artifacts/xgb_nested_v1/importance_0.csv", index_col=0)
    ranked = [x for x in imp.iloc[:, 0].sort_values(ascending=False).index if x in runner.features(tr, te, False)[0].columns]
    # The canonical run has 110 columns: feature names are reconstructed in
    # the same order as the runner's feature list and TE keys.
    names = list(base.columns) + [f"{c}_TE_{s}" for s in ("auto", 10.0, 100.0) for c in keys]
    imp = pd.read_csv(ROOT / "artifacts/xgb_nested_v1/importance_0.csv", index_col=0)
    ranked = [n for n in imp.index.tolist() if n in names]
    rank_indices = [names.index(n) for n in ranked]
    if len(rank_indices) != len(names):
        # Any old importance CSV has the same complete feature set; fail closed
        # instead of silently selecting the wrong columns after code changes.
        raise RuntimeError(f"importance names mismatch: {len(rank_indices)} / {len(names)}")
    out = ROOT / "artifacts" / RUN
    out.mkdir(exist_ok=True)
    recs = []
    t0 = time.time()
    for k in (40, 60, 80):
        ix = rank_indices[:k]
        model = XGBClassifier(device="cuda", tree_method="hist", max_bin=1024,
                              n_estimators=662, max_depth=5, learning_rate=0.03,
                              min_child_weight=10, subsample=0.9, colsample_bytree=0.85,
                              reg_alpha=0.071, reg_lambda=2.0,
                              objective="binary:logistic", eval_metric="auc",
                              n_jobs=8, random_state=42)
        model.fit(A[:, ix], y[ti], eval_set=[(B[:, ix], y[vi])], verbose=False)
        vp = model.predict_proba(B[:, ix])[:, 1]
        rec = {"top_k": k, "auc": float(roc_auc_score(y[vi], vp)),
               "seconds": time.time() - t0, "features": [names[i] for i in ix]}
        recs.append(rec); print(json.dumps(rec), flush=True)
    manifest = {"run": RUN, "status": "pilot", "submitted": False,
                "selection_source": "canonical fold-0 importance; validation-aware exploratory pilot",
                "fixed_iterations": 662, "records": recs,
                "code_sha256": runner.digest(Path(__file__)),
                "canonical_importance_sha256": runner.digest(ROOT / "artifacts/xgb_nested_v1/importance_0.csv"),
                "note": "No full OOF or submission emitted because selection is not outer-fold independent"}
    (out / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

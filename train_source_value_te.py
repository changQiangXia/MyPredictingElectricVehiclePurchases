"""Nested XGBoost with source-label priors for exact numeric values."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

import train_encoded as runner

ROOT = runner.ROOT
TARGET = runner.TARGET
ALPHAS = (1.0, 5.0, 20.0, 100.0)
VALUE_COLUMNS = ("Annual_Income_USD", "Daily_Commute_km")


def source_maps(source):
    prior = float(source[TARGET].eq("Yes").mean())
    maps = {}
    for col in VALUE_COLUMNS:
        y = source[TARGET].eq("Yes").astype(float)
        stats = pd.DataFrame({"v": source[col], "y": y}).groupby("v").y.agg(["size", "sum"])
        maps[col] = (stats, prior)
    return maps


def source_features(fit_values, fit_y, query_values, maps):
    parts, names = [], []
    for col in VALUE_COLUMNS:
        values = np.asarray(fit_values[col])
        query = np.asarray(query_values[col])
        source_stats, source_prior = maps[col]
        source_rate = ((source_stats["sum"] + 20.0 * source_prior) /
                       (source_stats["size"] + 20.0))
        q_source_rate = pd.Series(query).map(source_rate).fillna(source_prior).to_numpy(float)
        q_source_n = pd.Series(query).map(source_stats["size"]).fillna(0).to_numpy(float)
        comp = pd.DataFrame({"v": values, "y": np.asarray(fit_y, float)}).groupby("v").y.agg(["size", "sum"])
        q_n = pd.Series(query).map(comp["size"]).fillna(0).to_numpy(float)
        q_sum = pd.Series(query).map(comp["sum"]).fillna(0).to_numpy(float)
        for alpha in ALPHAS:
            parts.append(((q_sum + alpha * q_source_rate) / (q_n + alpha)).astype("float32"))
            names.append(f"{col}_source_prior_te_{alpha:g}")
        parts.append(q_source_rate.astype("float32")); names.append(f"{col}_source_rate")
        parts.append(np.log1p(q_source_n).astype("float32")); names.append(f"{col}_source_log_count")
        parts.append(np.log1p(q_n).astype("float32")); names.append(f"{col}_competition_log_count")
    return np.column_stack(parts).astype("float32"), names


def main():
    run = "xgb_source_value_te_v1"
    out = ROOT / "artifacts" / run
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    source = pd.read_csv(next((ROOT / "original").glob("*.csv")))
    y = train[TARGET].eq("Yes").to_numpy(dtype="int8")
    folds = np.load(ROOT / "artifacts" / "folds_seed42.npy")
    n = len(train)
    base, keys, mappings = runner.features(train, test, False)
    X, Xt = base.iloc[:n].to_numpy(), base.iloc[n:].to_numpy()
    K, Kt = keys.iloc[:n].to_numpy(), keys.iloc[n:].to_numpy()
    all_values = {c: pd.concat([train[c], test[c]], ignore_index=True).to_numpy() for c in VALUE_COLUMNS}
    source_prior_maps = source_maps(source)
    config = {"run": run, "status": "running", "submitted": False,
              "alphas": ALPHAS, "value_columns": VALUE_COLUMNS,
              "validation": "5 outer folds; competition statistics for outer training rows use inner 5-fold cross-fit; source labels fixed externally",
              "code_sha256": runner.digest(Path(__file__)), "runner_sha256": runner.digest(ROOT / "train_encoded.py"),
              "source_sha256": runner.digest(next((ROOT / "original").glob("*.csv"))),
              "fold_sha256": runner.digest(ROOT / "artifacts/folds_seed42.npy"),
              "train_sha256": runner.digest(ROOT / "train.csv"), "test_sha256": runner.digest(ROOT / "test.csv")}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    oof, test_pred, records = np.full(n, np.nan), np.zeros(len(test)), []
    for fold in range(5):
        t0 = time.time(); ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inner = StratifiedKFold(5, shuffle=True, random_state=6200 + fold)
        inner_extra = np.zeros((len(ti), 0), dtype="float32")
        inner_names = None
        for fit_pos, query_pos in inner.split(ti, y[ti]):
            fit_idx, query_idx = ti[fit_pos], ti[query_pos]
            fit_values = {c: all_values[c][fit_idx] for c in VALUE_COLUMNS}
            query_values = {c: all_values[c][query_idx] for c in VALUE_COLUMNS}
            extra, names = source_features(fit_values, y[fit_idx], query_values, source_prior_maps)
            if inner_names is None:
                inner_names = names; inner_extra = np.zeros((len(ti), extra.shape[1]), dtype="float32")
            assert names == inner_names
            inner_extra[query_pos] = extra
        valid_values = {c: all_values[c][vi] for c in VALUE_COLUMNS}
        test_values = {c: all_values[c][n:] for c in VALUE_COLUMNS}
        outer_values = {c: all_values[c][ti] for c in VALUE_COLUMNS}
        valid_extra, names = source_features(outer_values, y[ti], valid_values, source_prior_maps)
        test_extra, names2 = source_features(outer_values, y[ti], test_values, source_prior_maps)
        assert names == names2 == inner_names
        train_parts, valid_parts, test_parts = [X[ti], inner_extra], [X[vi], valid_extra], [Xt, test_extra]
        encoders = []
        for smoothing in ("auto", 10.0, 100.0):
            enc = TargetEncoder(target_type="binary", smooth=smoothing, cv=5, shuffle=True, random_state=42)
            train_parts.append(enc.fit_transform(K[ti], y[ti]).astype("float32")); valid_parts.append(enc.transform(K[vi]).astype("float32")); test_parts.append(enc.transform(Kt).astype("float32")); encoders.append(enc)
        A, B, C = [np.concatenate(z, axis=1) for z in (train_parts, valid_parts, test_parts)]
        model = XGBClassifier(device="cuda", tree_method="hist", max_bin=1024, n_estimators=6500,
                              max_depth=5, learning_rate=0.03, min_child_weight=10, subsample=0.9,
                              colsample_bytree=0.85, reg_alpha=0.071, reg_lambda=2.0,
                              objective="binary:logistic", eval_metric="auc", early_stopping_rounds=300,
                              n_jobs=8, random_state=42)
        model.fit(A, y[ti], eval_set=[(B, y[vi])], verbose=500); model.set_params(device="cpu")
        vp, tp = model.predict_proba(B)[:, 1], model.predict_proba(C)[:, 1]
        oof[vi], test_pred = vp, test_pred + tp / 5
        rec = {"fold": fold, "auc": float(roc_auc_score(y[vi], vp)), "best_iteration": int(model.best_iteration) + 1, "seconds": time.time() - t0}
        records.append(rec); np.savez_compressed(out / f"fold_{fold}.npz", valid_indices=vi, valid_prediction=vp, test_prediction=tp); (out / f"fold_{fold}.json").write_text(json.dumps(rec, indent=2) + "\n"); model.save_model(out / f"model_{fold}.ubj"); joblib.dump(encoders, out / f"encoders_{fold}.joblib"); print(json.dumps(rec), flush=True)
        del A, B, C, model, encoders
    np.save(out / "oof.npy", oof); np.save(out / "test.npy", test_pred); pd.DataFrame({"id": test.id, TARGET: test_pred}).to_csv(out / "submission.csv", index=False)
    metrics = {"run": run, "status": "complete", "submitted": False, "oof_auc": float(roc_auc_score(y, oof)), "folds": records, "fold_mean": float(np.mean([r["auc"] for r in records])), "fold_std": float(np.std([r["auc"] for r in records])), "submission_sha256": runner.digest(out / "submission.csv"), "elapsed_seconds": time.time() - started}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n"); print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__": main()

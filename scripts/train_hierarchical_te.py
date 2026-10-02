"""Nested XGBoost with hierarchical empirical-Bayes value encodings.

Exact income/commute values are noisy when rare.  The usual TargetEncoder
shrinks them toward the global prior; this experiment shrinks them toward the
local floor-bin prior as well.  Every group statistic is recomputed inside the
reference partition for each outer/inner query.
"""
from __future__ import annotations

import hashlib
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
VALUE_SPECS = {
    "Annual_Income_USD": ("income_floor_100", "income_floor_1000"),
    "Daily_Commute_km": ("commute_floor",),
}
ALPHAS = (1.0, 5.0, 20.0, 50.0)


def hierarchical(values, y, query, parents, alphas=ALPHAS):
    """Return value posterior columns plus count and parent prior columns."""
    values = np.asarray(values)
    query = np.asarray(query)
    y = np.asarray(y, dtype=np.float64)
    result, names = [], []
    global_prior = float(y.mean())
    value_df = pd.DataFrame({"v": values, "y": y})
    value_stats = value_df.groupby("v").y.agg(["size", "sum"])
    query_count = pd.Series(query).map(value_stats["size"]).fillna(0).to_numpy(dtype=np.float64)
    result.append(np.log1p(query_count).astype("float32")); names.append("value_log_count")
    for parent in parents:
        parent_values = np.asarray(parent)
        parent_df = pd.DataFrame({"p": parent_values, "y": y})
        parent_stats = parent_df.groupby("p").y.agg(["size", "sum"])
        parent_rate = ((parent_stats["sum"] + 20.0 * global_prior) /
                       (parent_stats["size"] + 20.0))
        q_parent = pd.Series(np.asarray(parent_values)[0:0])
        # Parent query is supplied by the caller through the aligned dict below.
        raise RuntimeError("Use hierarchical_pair()")


def hierarchical_pair(values, fit_y, query_values, parent_values, query_parent, prefix):
    values = np.asarray(values)
    query_values = np.asarray(query_values)
    parent_values = np.asarray(parent_values)
    query_parent = np.asarray(query_parent)
    fit_y = np.asarray(fit_y, dtype=np.float64)
    prior = float(fit_y.mean())
    value_stats = pd.DataFrame({"v": values, "y": fit_y}).groupby("v").y.agg(["size", "sum"])
    parent_stats = pd.DataFrame({"p": parent_values, "y": fit_y}).groupby("p").y.agg(["size", "sum"])
    parent_rate = ((parent_stats["sum"] + 20.0 * prior) /
                   (parent_stats["size"] + 20.0))
    q_parent_rate = pd.Series(query_parent).map(parent_rate).fillna(prior).to_numpy(dtype=np.float64)
    q_n = pd.Series(query_values).map(value_stats["size"]).fillna(0).to_numpy(dtype=np.float64)
    q_sum = pd.Series(query_values).map(value_stats["sum"]).fillna(0).to_numpy(dtype=np.float64)
    output = []
    names = []
    for alpha in ALPHAS:
        output.append(((q_sum + alpha * q_parent_rate) / (q_n + alpha)).astype("float32"))
        names.append(f"{prefix}_hier_te_{alpha:g}")
    output.append(np.log1p(q_n).astype("float32")); names.append(f"{prefix}_value_log_count")
    output.append(q_parent_rate.astype("float32")); names.append(f"{prefix}_parent_rate")
    return np.column_stack(output), names


def make_hier(train_values, fit_idx, fit_y, query_idx, query_values):
    cols, names = [], []
    for column, parent_names in VALUE_SPECS.items():
        for parent_name in parent_names:
            values = train_values[column][fit_idx]
            q_values = query_values[column]
            parent = train_values[parent_name][fit_idx]
            q_parent = query_values[parent_name]
            a, n = hierarchical_pair(values, fit_y, q_values, parent, q_parent,
                                      f"{column}_by_{parent_name}")
            cols.append(a); names.extend(n)
    return np.column_stack(cols).astype("float32"), names


def main():
    run = "xgb_hierarchical_te_v1"
    out = ROOT / "artifacts" / run
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = train[TARGET].eq("Yes").to_numpy(dtype="int8")
    folds = np.load(ROOT / "artifacts" / "folds_seed42.npy")
    n = len(train)
    base, keys, mappings = runner.features(train, test, False)
    X, Xt = base.iloc[:n].to_numpy(), base.iloc[n:].to_numpy()
    K, Kt = keys.iloc[:n].to_numpy(), keys.iloc[n:].to_numpy()
    all_values = {}
    for name in set(VALUE_SPECS) | {p for ps in VALUE_SPECS.values() for p in ps}:
        if name == "income_floor_100": all_values[name] = np.floor(pd.concat([train.Annual_Income_USD, test.Annual_Income_USD]).to_numpy() / 100.0)
        elif name == "income_floor_1000": all_values[name] = np.floor(pd.concat([train.Annual_Income_USD, test.Annual_Income_USD]).to_numpy() / 1000.0)
        elif name == "commute_floor": all_values[name] = np.floor(pd.concat([train.Daily_Commute_km, test.Daily_Commute_km]).to_numpy())
        else: all_values[name] = pd.concat([train[name], test[name]]).to_numpy()
    config = {"run": run, "status": "running", "submitted": False,
              "value_specs": VALUE_SPECS, "alphas": ALPHAS,
              "validation": "outer five folds; outer training rows via inner five-fold cross-fit",
              "code_sha256": runner.digest(Path(__file__)), "runner_sha256": runner.digest(ROOT / "train_encoded.py"),
              "fold_sha256": runner.digest(ROOT / "artifacts/folds_seed42.npy"),
              "train_sha256": runner.digest(ROOT / "train.csv"), "test_sha256": runner.digest(ROOT / "test.csv")}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    oof, test_pred, records = np.full(n, np.nan), np.zeros(len(test)), []
    for fold in range(5):
        t0 = time.time(); ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inner = StratifiedKFold(5, shuffle=True, random_state=5200 + fold)
        inner_hier = None
        for fit_pos, query_pos in inner.split(ti, y[ti]):
            fit_idx, query_idx = ti[fit_pos], ti[query_pos]
            fit_vals = {c: all_values[c][fit_idx] for c in all_values}
            q_vals = {c: all_values[c][query_idx] for c in all_values}
            a, names = make_hier(all_values, fit_idx, y[fit_idx], query_idx, q_vals)
            if inner_hier is None: inner_hier = np.zeros((len(ti), a.shape[1]), dtype="float32")
            inner_hier[query_pos] = a
        outer_query = {c: all_values[c][vi] for c in all_values}
        test_query = {c: all_values[c][n:] for c in all_values}
        valid_hier, hier_names = make_hier(all_values, ti, y[ti], vi, outer_query)
        test_hier, hier_names2 = make_hier(all_values, ti, y[ti], np.arange(n, n + len(test)), test_query)
        assert hier_names == hier_names2 and inner_hier.shape[1] == len(hier_names)
        train_parts, valid_parts, test_parts = [X[ti], inner_hier], [X[vi], valid_hier], [Xt, test_hier]
        encoders = []
        for smoothing in ("auto", 10.0, 100.0):
            enc = TargetEncoder(target_type="binary", smooth=smoothing, cv=5, shuffle=True, random_state=42)
            train_parts.append(enc.fit_transform(K[ti], y[ti]).astype("float32"))
            valid_parts.append(enc.transform(K[vi]).astype("float32"))
            test_parts.append(enc.transform(Kt).astype("float32")); encoders.append(enc)
        A, B, C = [np.concatenate(z, axis=1) for z in (train_parts, valid_parts, test_parts)]
        model = XGBClassifier(device="cuda", tree_method="hist", max_bin=1024,
                              n_estimators=6500, max_depth=5, learning_rate=0.03,
                              min_child_weight=10, subsample=0.9, colsample_bytree=0.85,
                              reg_alpha=0.071, reg_lambda=2.0, objective="binary:logistic",
                              eval_metric="auc", early_stopping_rounds=300, n_jobs=8, random_state=42)
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

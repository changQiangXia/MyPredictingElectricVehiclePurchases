"""Canonical nested XGBoost augmented with leakage-safe support-purity flags.

Support rules are discovered from labels in a reference split, but every
outer-training row receives them through an inner cross-fit map.  Validation
and test rows use maps fit on the complete outer-training partition only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

import train_encoded as runner

ROOT = runner.ROOT
TARGET = runner.TARGET


RULES = {
    "Annual_Income_USD": {
        "exact": (20, None),
        "bin": (20, (100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0)),
    },
    "Daily_Commute_km": {
        "exact": (20, None),
        "bin": (20, (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)),
    },
}


def _query_mask(values, spec):
    if spec["kind"] == "exact":
        return np.isclose(values, spec["value"], rtol=0.0, atol=1e-10)
    return (values >= spec["lo"]) & (values < spec["hi"])


def support_features(fit_values, fit_y, query_values):
    """Return compact zero/one purity indicators for two numeric columns."""
    columns = []
    names = []
    for column in ("Annual_Income_USD", "Daily_Commute_km"):
        x = fit_values[column]
        q = query_values[column]
        # Aggregates are maxima across rule resolutions.  This avoids creating
        # thousands of one-hot columns while retaining generator identity.
        zero_any = np.zeros(len(q), dtype="float32")
        one_any = np.zeros(len(q), dtype="float32")
        zero_strength = np.zeros(len(q), dtype="float32")
        one_strength = np.zeros(len(q), dtype="float32")
        for kind, (minimum, widths) in RULES[column].items():
            width_list = (None,) if kind == "exact" else widths
            for width in width_list:
                if kind == "exact":
                    keys_fit, keys_query = x, q
                else:
                    width = float(width)
                    keys_fit, keys_query = np.floor(x / width) * width, np.floor(q / width) * width
                groups = pd.DataFrame({"v": keys_fit, "y": fit_y}).groupby("v").y.agg(["size", "sum"])
                groups = groups[groups["size"] >= minimum]
                pure_zero = groups.index[groups["sum"] == 0]
                pure_one = groups.index[groups["sum"] == groups["size"]]
                # Map each query key to its support count only for pure groups.
                zmap = groups.loc[pure_zero, "size"]
                omap = groups.loc[pure_one, "size"]
                zcount = pd.Series(keys_query).map(zmap).to_numpy()
                ocount = pd.Series(keys_query).map(omap).to_numpy()
                zmask, omask = np.isfinite(zcount), np.isfinite(ocount)
                zero_any[zmask] = 1.0
                one_any[omask] = 1.0
                if zmask.any():
                    zero_strength[zmask] = np.maximum(zero_strength[zmask],
                        np.minimum(1.0, np.log1p(zcount[zmask]) / np.log1p(500.0)).astype("float32"))
                if omask.any():
                    one_strength[omask] = np.maximum(one_strength[omask],
                        np.minimum(1.0, np.log1p(ocount[omask]) / np.log1p(500.0)).astype("float32"))
        columns.extend([zero_any, one_any, zero_strength, one_strength])
        names.extend([f"{column}_support_zero", f"{column}_support_one",
                      f"{column}_support_zero_strength", f"{column}_support_one_strength"])
    return np.column_stack(columns).astype("float32"), names


def main():
    run = "xgb_support_rules_v1"
    out = ROOT / "artifacts" / run
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = train[TARGET].eq("Yes").to_numpy(dtype="int8")
    n = len(train)
    folds = np.load(ROOT / "artifacts" / "folds_seed42.npy")
    base, keys, mappings = runner.features(train, test, False)
    X, Xt = base.iloc[:n].to_numpy(), base.iloc[n:].to_numpy()
    K, Kt = keys.iloc[:n].to_numpy(), keys.iloc[n:].to_numpy()
    values = {c: train[c].to_numpy(float) for c in RULES}
    test_values = {c: test[c].to_numpy(float) for c in RULES}
    names = list(base.columns)
    for smoothing in ("auto", 10.0, 100.0):
        names.extend([f"{c}_TE_{smoothing}" for c in keys])
    support_names = [f"{c}_support_{k}" for c in RULES for k in ("zero", "one", "zero_strength", "one_strength")]
    names.extend(support_names)
    config = {
        "run": run, "status": "running", "submitted": False,
        "validation": "5 outer folds; outer training rows receive support maps via inner 5-fold cross-fit",
        "rules": RULES, "support_features": support_names,
        "code_sha256": runner.digest(Path(__file__)), "runner_sha256": runner.digest(ROOT / "train_encoded.py"),
        "fold_sha256": runner.digest(ROOT / "artifacts/folds_seed42.npy"),
        "train_sha256": runner.digest(ROOT / "train.csv"), "test_sha256": runner.digest(ROOT / "test.csv"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    oof, test_pred = np.full(n, np.nan), np.zeros(len(test))
    records = []
    for fold in range(5):
        t0 = time.time()
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inner_support = np.zeros((len(ti), len(support_names)), dtype="float32")
        inner = StratifiedKFold(5, shuffle=True, random_state=4100 + fold)
        for fit_pos, query_pos in inner.split(ti, y[ti]):
            fit_idx, query_idx = ti[fit_pos], ti[query_pos]
            fit_vals = {c: values[c][fit_idx] for c in RULES}
            query_vals = {c: values[c][query_idx] for c in RULES}
            sf, sn = support_features(fit_vals, y[fit_idx], query_vals)
            assert sn == support_names
            inner_support[query_pos] = sf
        outer_fit_vals = values
        valid_vals = {c: values[c][vi] for c in RULES}
        test_vals = test_values
        sf_valid, sn = support_features({c: values[c][ti] for c in RULES}, y[ti], valid_vals)
        sf_test, sn2 = support_features({c: values[c][ti] for c in RULES}, y[ti], test_vals)
        assert sn == sn2 == support_names
        train_parts, valid_parts, test_parts = [X[ti], inner_support], [X[vi], sf_valid], [Xt, sf_test]
        encoders = []
        for smoothing in ("auto", 10.0, 100.0):
            enc = TargetEncoder(target_type="binary", smooth=smoothing, cv=5, shuffle=True, random_state=42)
            train_parts.append(enc.fit_transform(K[ti], y[ti]).astype("float32"))
            valid_parts.append(enc.transform(K[vi]).astype("float32"))
            test_parts.append(enc.transform(Kt).astype("float32"))
            encoders.append(enc)
        A, B, C = [np.concatenate(parts, axis=1) for parts in (train_parts, valid_parts, test_parts)]
        model = XGBClassifier(device="cuda", tree_method="hist", max_bin=1024,
                              n_estimators=6500, max_depth=5, learning_rate=0.03,
                              min_child_weight=10, subsample=0.9, colsample_bytree=0.85,
                              reg_alpha=0.071, reg_lambda=2.0, objective="binary:logistic",
                              eval_metric="auc", early_stopping_rounds=300, n_jobs=8, random_state=42)
        model.fit(A, y[ti], eval_set=[(B, y[vi])], verbose=500)
        model.set_params(device="cpu")
        vp, tp = model.predict_proba(B)[:, 1], model.predict_proba(C)[:, 1]
        oof[vi], test_pred = vp, test_pred + tp / 5
        record = {"fold": fold, "auc": float(roc_auc_score(y[vi], vp)),
                  "best_iteration": int(model.best_iteration) + 1, "seconds": time.time() - t0}
        np.savez_compressed(out / f"fold_{fold}.npz", valid_indices=vi, valid_prediction=vp, test_prediction=tp)
        (out / f"fold_{fold}.json").write_text(json.dumps(record, indent=2) + "\n")
        model.save_model(out / f"model_{fold}.ubj")
        joblib.dump(encoders, out / f"encoders_{fold}.joblib")
        records.append(record)
        print(json.dumps(record), flush=True)
        del A, B, C, model, encoders
    np.save(out / "oof.npy", oof); np.save(out / "test.npy", test_pred)
    sub = pd.DataFrame({"id": test.id, TARGET: test_pred})
    sub.to_csv(out / "submission.csv", index=False)
    metrics = {"run": run, "status": "complete", "submitted": False,
               "oof_auc": float(roc_auc_score(y, oof)), "folds": records,
               "fold_mean": float(np.mean([r["auc"] for r in records])),
               "fold_std": float(np.std([r["auc"] for r in records])),
               "submission_sha256": runner.digest(out / "submission.csv"),
               "elapsed_seconds": time.time() - started}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    import joblib
    main()

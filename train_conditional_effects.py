"""Nested income effects estimated conditionally on a nuisance response model.

For every encoding split, fit a small response model on that split's reference
rows, then estimate a regularized income-specific log-odds shift with its margin
as an offset. Query labels never enter either model or the effect estimates.
"""
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

from train_encoded import ROOT, TARGET, digest, features


def posterior_effect(codes, y, offset, n_codes, penalties):
    """Penalized conditional binomial likelihood; Newton updates by category."""
    effects = []
    for lam in penalties:
        effect = np.zeros(n_codes, dtype=np.float64)
        for _ in range(20):
            p = expit(offset + effect[codes])
            gradient = np.bincount(codes, weights=p-y, minlength=n_codes) + lam * effect
            curvature = np.bincount(codes, weights=p*(1-p), minlength=n_codes) + lam
            step = np.clip(gradient / curvature, -1, 1)
            effect -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        effects.append(effect)
    return np.column_stack(effects)


def encode(X, codes, y, fit, queries, args, out, name):
    teacher = XGBClassifier(
        objective="binary:logistic", device="cuda", tree_method="hist",
        n_estimators=args.teacher_iterations, learning_rate=0.04,
        max_depth=3, max_bin=128, min_child_weight=20,
        reg_lambda=5.0, subsample=1.0, colsample_bytree=1.0,
        n_jobs=8, random_state=42)
    teacher.fit(X[fit], y[fit], verbose=False)
    teacher.set_params(device="cpu")
    teacher.save_model(out / f"teacher_{name}.ubj")
    offset = teacher.predict(X[fit], output_margin=True).astype(np.float64)
    effects = posterior_effect(codes[fit], y[fit], offset, int(codes.max())+1, args.penalties)
    np.savez_compressed(out / f"effects_{name}.npz", effects=effects,
                        fit_indices=fit, penalties=args.penalties)
    result = []
    for query in queries:
        margin = teacher.predict(X[query], output_margin=True)
        result.append(np.column_stack([effects[codes[query]], margin]).astype("float32"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--penalties", type=float, nargs="+", default=[1.0, 5.0])
    parser.add_argument("--teacher-iterations", type=int, default=500)
    args = parser.parse_args()
    assert len(set(args.folds)) == len(args.folds) and all(f in range(5) for f in args.folds)
    assert all(p > 0 for p in args.penalties)
    start = time.time()
    out = ROOT / "artifacts" / args.run
    out.mkdir(exist_ok=True)
    tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = tr[TARGET].eq("Yes").to_numpy(dtype="float64")
    n = len(tr)
    fold_path = ROOT / "artifacts/folds_seed42.npy"
    fold_ids = np.load(fold_path)
    base, keys, mappings = features(tr, te, False)
    X, K = base.to_numpy(), keys.to_numpy()
    # Only raw columns: coarse histograms prevent the nuisance model from
    # deliberately fitting the high-cardinality income identity.
    raw = base[[c for c in te if c != "id"]].to_numpy()
    code, support = pd.factorize(pd.concat([tr.Annual_Income_USD, te.Annual_Income_USD]), sort=True)
    code = np.asarray(code, dtype=np.int64)
    test_ids = np.arange(n, n + len(te))
    config = {k:v for k,v in vars(args).items() if k != "folds"}
    config.update(
        code_sha256=digest(Path(__file__)), runner_sha256=digest(ROOT / "train_encoded.py"),
        train_sha256=digest(ROOT / "train.csv"), test_sha256=digest(ROOT / "test.csv"),
        fold_sha256=digest(fold_path), category_mappings=mappings,
        hypothesis="Conditioning on nuisance covariates reduces sampling noise in exact income effects",
        validation="Five outer folds, five inner folds. Every teacher and effect map sees only its reference rows",
        teacher="Depth 3, 128 histogram bins, 500 fixed rounds unless overridden; raw features only",
        source_labels=False, submitted=False,
        blend_weights_prespecified=[0.05, 0.1, 0.2])
    cp = out / "config.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == config, "Use a fresh run ID for changed code/config"
    else:
        cp.write_text(json.dumps(config, indent=2) + "\n")
        np.save(out / "income_support.npy", np.asarray(support))
    oof, test_prediction = np.full(n, np.nan), np.zeros(len(te))
    records = []
    for fold in args.folds:
        fs = time.time()
        ti, vi = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        checkpoint = out / f"fold_{fold}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint)
            assert np.array_equal(saved["valid_indices"], vi)
            vp, tp = saved["valid_prediction"], saved["test_prediction"]
            record = json.loads((out / f"fold_{fold}.json").read_text())
        else:
            extra_train = np.empty((len(ti), len(args.penalties)+1), dtype="float32")
            inner = StratifiedKFold(5, shuffle=True, random_state=42)
            for k, (fit_pos, query_pos) in enumerate(inner.split(ti, y[ti])):
                extra_train[query_pos] = encode(raw, code, y, ti[fit_pos], [ti[query_pos]],
                                                args, out, f"{fold}_inner{k}")[0]
                print(json.dumps({"event":"conditional_encoded", "fold":fold,
                                  "inner":k,"seconds":time.time()-fs}), flush=True)
            extra_valid, extra_test = encode(raw, code, y, ti, [vi, test_ids], args, out, f"{fold}_outer")
            parts = [[X[ti], extra_train], [X[vi], extra_valid], [X[test_ids], extra_test]]
            encoders = []
            for smoothing in ["auto", 10.0, 100.0]:
                enc = TargetEncoder(target_type="binary", smooth=smoothing, cv=5,
                                    shuffle=True, random_state=42)
                parts[0].append(enc.fit_transform(K[ti], y[ti]).astype("float32"))
                parts[1].append(enc.transform(K[vi]).astype("float32"))
                parts[2].append(enc.transform(K[test_ids]).astype("float32"))
                encoders.append(enc)
            A, B, C = [np.concatenate(p, axis=1) for p in parts]
            del parts
            model = XGBClassifier(device="cuda", tree_method="hist", max_bin=1024,
                                  n_estimators=6500, max_depth=5, learning_rate=0.03,
                                  min_child_weight=10, subsample=0.9, colsample_bytree=0.85,
                                  reg_alpha=0.071, reg_lambda=2.0, objective="binary:logistic",
                                  eval_metric="auc", early_stopping_rounds=300,
                                  n_jobs=8, random_state=42)
            model.fit(A, y[ti], eval_set=[(B, y[vi])], verbose=500)
            model.set_params(device="cpu")
            model.save_model(out / f"model_{fold}.ubj")
            joblib.dump(encoders, out / f"encoders_{fold}.joblib")
            vp, tp = model.predict_proba(B)[:,1], model.predict_proba(C)[:,1]
            names = list(base.columns) + [f"conditional_income_{p}" for p in args.penalties] + ["nuisance_margin"]
            names += [f"{c}_TE_{p}" for p in ["auto",10.0,100.0] for c in keys]
            pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).to_csv(out/f"importance_{fold}.csv")
            reloaded = XGBClassifier()
            reloaded.load_model(out / f"model_{fold}.ubj")
            check = reloaded.predict_proba(B[:1024])[:,1]
            assert np.allclose(check, vp[:1024], atol=1e-7)
            record = {"fold":fold, "auc":float(roc_auc_score(y[vi], vp)),
                      "best_iteration":int(model.best_iteration)+1,
                      "seconds":time.time()-fs,
                      "reload_max_error":float(np.max(np.abs(check-vp[:1024])))}
            np.savez_compressed(checkpoint, valid_indices=vi, valid_prediction=vp, test_prediction=tp)
            (out/f"fold_{fold}.json").write_text(json.dumps(record,indent=2)+"\n")
            del A, B, C, model, encoders
        oof[vi], test_prediction = vp, test_prediction + tp/len(args.folds)
        records.append(record)
        print(json.dumps(record),flush=True)
    complete = np.isfinite(oof).all()
    metrics = {"run":args.run, "status":"complete" if complete else "pilot",
               "submitted":False, "public_auc":None, "folds":records,
               "elapsed_seconds":time.time()-start}
    if complete:
        np.save(out/"oof.npy",oof);np.save(out/"test.npy",test_prediction)
        submission = pd.read_csv(ROOT/"sample_submission.csv")
        assert submission.id.equals(te.id) and np.isfinite(test_prediction).all()
        assert ((test_prediction >= 0) & (test_prediction <= 1)).all()
        submission[TARGET] = test_prediction
        submission.to_csv(out/"submission.csv",index=False)
        metrics.update(oof_auc=float(roc_auc_score(y,oof)),
                       fold_mean=float(np.mean([r["auc"] for r in records])),
                       fold_std=float(np.std([r["auc"] for r in records])),
                       submission_sha256=digest(out/"submission.csv"))
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics),flush=True)


if __name__ == "__main__":
    main()

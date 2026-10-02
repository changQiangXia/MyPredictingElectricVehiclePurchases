"""Regularized additive category effects with logistic or probit link.

Income and commute have smooth backgrounds plus shrunk exact-value effects.
All coefficients and spline/category mappings are fitted on the outer training
partition. No target-derived feature or previously fitted model is used.
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
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.metrics import roc_auc_score, log_loss

from train_encoded import ROOT, TARGET, digest


def likelihood(y, z, link):
    if link == "logistic":
        return np.logaddexp(0, z) - y * z, expit(z) - y
    sign = 2 * y - 1
    signed = sign * z
    logp = log_ndtr(signed)
    gradient = -sign * np.exp(-0.5 * signed ** 2 - 0.5 * np.log(2 * np.pi) - logp)
    return -logp, gradient


def design(fit, valid, test, args):
    columns = [c for c in test if c != "id"]
    smooth_columns = ["Annual_Income_USD", "Daily_Commute_km"]
    category = OneHotEncoder(handle_unknown="ignore", dtype=np.float64,
                             sparse_output=True)
    spline = SplineTransformer(n_knots=args.knots, degree=3,
                               knots="uniform", include_bias=False,
                               extrapolation="linear", sparse_output=True)
    af = category.fit_transform(fit[columns])
    av, at = category.transform(valid[columns]), category.transform(test[columns])
    sf = spline.fit_transform(fit[smooth_columns])
    sv, st = spline.transform(valid[smooth_columns]), spline.transform(test[smooth_columns])
    matrices = [sparse.hstack([np.ones((len(c), 1)), s, a], format="csr")
                for c, s, a in [(fit, sf, af), (valid, sv, av), (test, st, at)]]
    penalties = [np.zeros(1), np.full(sf.shape[1], args.smooth_penalty)]
    names = ["intercept"] + list(spline.get_feature_names_out(smooth_columns))
    for column, categories in zip(columns, category.categories_):
        lam = args.income_penalty if column == "Annual_Income_USD" else (
            args.commute_penalty if column == "Daily_Commute_km" else args.other_penalty)
        penalties.append(np.full(len(categories), lam))
        names.extend([f"{column}={v}" for v in categories])
    return matrices, np.concatenate(penalties), {
        "category": category, "spline": spline, "columns": columns,
        "smooth_columns": smooth_columns, "names": names}


def fit_model(X, y, penalty, link, iterations):
    # Jacobi scaling reduces the imbalance between frequent and rare values.
    scale = 1 / np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel() * 0.15 + penalty + 1e-8)
    scaled = X.multiply(scale).tocsr()
    scaled_penalty = penalty * scale ** 2
    def objective(w):
        losses, dz = likelihood(y, scaled @ w, link)
        value = losses.sum() + 0.5 * np.dot(scaled_penalty, w * w)
        gradient = np.asarray(scaled.T @ dz).ravel() + scaled_penalty * w
        return value, gradient
    initial = np.zeros(X.shape[1])
    if link == "logistic":
        intercept = np.log(y.mean() / (1 - y.mean()))
    else:
        from scipy.special import ndtri
        intercept = ndtri(y.mean())
    initial[0] = intercept / scale[0]
    result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": iterations, "ftol": 1e-10,
                               "gtol": 1e-4, "maxcor": 15})
    return result.x * scale, {"success": bool(result.success),
                             "message": str(result.message),
                             "iterations": int(result.nit),
                             "objective": float(result.fun),
                             "max_scaled_gradient": float(np.max(np.abs(result.jac)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--link", choices=["logistic", "probit"], default="probit")
    parser.add_argument("--income-penalty", type=float, default=5.0)
    parser.add_argument("--commute-penalty", type=float, default=10.0)
    parser.add_argument("--other-penalty", type=float, default=1.0)
    parser.add_argument("--smooth-penalty", type=float, default=0.1)
    parser.add_argument("--knots", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    start = time.time()
    out = ROOT / "artifacts" / args.run
    out.mkdir(exist_ok=True)
    fold_path = ROOT / "artifacts/folds_seed42.npy"
    tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = tr[TARGET].eq("Yes").to_numpy(dtype=np.float64)
    folds = np.load(fold_path)
    config = {k: v for k, v in vars(args).items() if k != "folds"}
    config.update(code_sha256=digest(Path(__file__)), train_sha256=digest(ROOT / "train.csv"),
                  test_sha256=digest(ROOT / "test.csv"), fold_sha256=digest(fold_path),
                  hypothesis="Joint nuisance-adjusted exact-value effects reduce marginal target-rate sampling noise",
                  validation="All supervised fitting and preprocessing inside the outer training fold; no TE",
                  selection="Pilot on fold 0; confirm other four folds only if competitive",
                  submitted=False, blend_weights_prespecified=[0.05, 0.1, 0.2])
    cp = out / "config.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == config, "Changed configuration requires a fresh run ID"
    else:
        cp.write_text(json.dumps(config, indent=2) + "\n")
    oof, test_prediction = np.full(len(tr), np.nan), np.zeros(len(te))
    records = []
    transform = expit if args.link == "logistic" else ndtr
    for fold in args.folds:
        assert fold in range(5)
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        checkpoint = out / f"fold_{fold}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint)
            assert np.array_equal(saved["valid_indices"], vi)
            vp, tp = saved["valid_prediction"], saved["test_prediction"]
            record = json.loads((out / f"fold_{fold}.json").read_text())
        else:
            fs = time.time()
            (A, B, C), penalty, preprocess = design(tr.iloc[ti], tr.iloc[vi], te, args)
            print(json.dumps({"event": "encoded", "fold": fold, "parameters": A.shape[1],
                              "seconds": time.time() - fs}), flush=True)
            coef, optimizer = fit_model(A, y[ti], penalty, args.link, args.iterations)
            vp, tp = transform(B @ coef), transform(C @ coef)
            model_path = out / f"model_{fold}.joblib"
            joblib.dump(dict(preprocess=preprocess, coefficient=coef, link=args.link), model_path)
            loaded = joblib.load(model_path)
            check = transform(B[:1024] @ loaded["coefficient"])
            assert np.allclose(check, vp[:1024], atol=1e-12)
            record = {"fold": fold, "auc": float(roc_auc_score(y[vi], vp)),
                      "logloss": float(log_loss(y[vi], vp)), "optimizer": optimizer,
                      "seconds": time.time() - fs, "reload_max_error": float(np.max(np.abs(check-vp[:1024])))}
            np.savez_compressed(checkpoint, valid_indices=vi, valid_prediction=vp, test_prediction=tp)
            (out / f"fold_{fold}.json").write_text(json.dumps(record, indent=2) + "\n")
            del A, B, C, preprocess
        oof[vi] = vp
        test_prediction += tp / len(args.folds)
        records.append(record)
        print(json.dumps(record), flush=True)
    complete = np.isfinite(oof).all()
    metrics = {"run": args.run, "status": "complete" if complete else "pilot",
               "submitted": False, "public_auc": None, "folds": records,
               "elapsed_seconds": time.time() - start}
    if complete:
        metrics.update(oof_auc=float(roc_auc_score(y, oof)),
                       fold_mean=float(np.mean([r["auc"] for r in records])),
                       fold_std=float(np.std([r["auc"] for r in records])))
        np.save(out / "oof.npy", oof)
        np.save(out / "test.npy", test_prediction)
        submission = pd.read_csv(ROOT / "sample_submission.csv")
        assert submission.id.equals(te.id) and np.isfinite(test_prediction).all()
        assert ((test_prediction >= 0) & (test_prediction <= 1)).all()
        submission[TARGET] = test_prediction
        submission.to_csv(out / "submission.csv", index=False)
        metrics["submission_sha256"] = digest(out / "submission.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

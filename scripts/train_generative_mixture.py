"""Pilot a class-conditional mixture of categorical densities on one outer fold.

This is a generative alternative to the discriminative GBDT/TabM pipeline.
Each label has its own mixture of product distributions. Exact numeric values
are categorical supports, and component densities shrink toward the class-wide
distribution. No target encoding, source labels, or existing predictions enter
model fitting. The outer labels are read only for the final comparison.
"""
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

from train_encoded import ROOT, TARGET, digest


def rank(v):
    return rankdata(v, method="average") / len(v)


def log(record):
    print(json.dumps(record), flush=True)


def class_density(codes, cardinalities, init, prior, components, iterations,
                  component_shrink, columns):
    """Fit by EM with hierarchical Dirichlet shrinkage; no validation labels."""
    x = torch.as_tensor(codes, dtype=torch.int64, device="cuda")
    class_prob = []
    for j, size in enumerate(cardinalities):
        pseudo = (10. if columns[j] == "Annual_Income_USD" else 20.) * prior
        count = torch.bincount(x[:, j], minlength=size).float() + pseudo
        class_prob.append(count / count.sum())
    if components == 1:
        return dict(log_mix=torch.zeros(1),
                    log_theta=[p.log()[None].cpu() for p in class_prob], history=[])
    resp = torch.nn.functional.one_hot(torch.as_tensor(init, dtype=torch.int64, device="cuda"),
                                       components).float() * .98 + .02 / components
    theta, mix, history = [], None, []
    for iteration in range(iterations):
        masses = resp.sum(0)
        mix = (masses + 1.) / (masses.sum() + components)
        theta = []
        for j, size in enumerate(cardinalities):
            shrink = component_shrink * (5. if columns[j] == "Annual_Income_USD" else 1.)
            count = torch.zeros((components, size), dtype=torch.float32, device="cuda")
            count.index_add_(1, x[:, j], resp.T.contiguous())
            count += shrink * class_prob[j][None]
            theta.append((count / count.sum(1, keepdim=True)).clamp_min(1e-15).log())
        score = mix.log()[None].expand(len(x), -1).clone()
        for j in range(x.shape[1]):
            score += theta[j][:, x[:, j]].T
        history.append(dict(iteration=iteration,
                            mean_log_density=float(torch.logsumexp(score, 1).mean()),
                            minimum_component_mass=float(masses.min())))
        resp = score.softmax(1)
    return dict(log_mix=mix.log().cpu(), log_theta=[p.cpu() for p in theta], history=history)


@torch.inference_mode()
def log_density(state, codes, batch=32768):
    theta = [p.cuda() for p in state["log_theta"]]
    mix = state["log_mix"].cuda()
    result = []
    for start in range(0, len(codes), batch):
        x = torch.as_tensor(codes[start:start + batch], dtype=torch.int64, device="cuda")
        score = mix[None].expand(len(x), -1).clone()
        for j in range(x.shape[1]):
            score += theta[j][:, x[:, j]].T
        result.append(torch.logsumexp(score, 1).cpu().numpy())
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--components", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--component-shrink", type=float, default=1000.)
    args = p.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    out = ROOT / "artifacts" / args.run
    out.mkdir(exist_ok=False)
    start = time.time()
    tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = tr[TARGET].eq("Yes").to_numpy("int8")
    columns = [c for c in te if c != "id"]
    all_rows = pd.concat([tr[columns], te[columns]], ignore_index=True)
    arrays, support = [], {}
    for c in columns:
        code, levels = pd.factorize(all_rows[c], sort=True)
        arrays.append(code.astype("int64"))
        support[c] = levels.tolist()
    codes = np.column_stack(arrays)
    cardinalities = [len(support[c]) for c in columns]
    folds = np.load(ROOT / "artifacts/folds_seed42.npy")
    fit, valid = np.flatnonzero(folds != args.fold), np.flatnonzero(folds == args.fold)
    # K-means is initialization only. Its scaler and clusters fit outer-train.
    small = [c for c in columns if all_rows[c].nunique() <= 20]
    continuous = [c for c in columns if c not in small]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    cat_init = enc.fit_transform(tr.iloc[fit][small])
    scaler = StandardScaler().fit(tr.iloc[fit][continuous])
    num_init = scaler.transform(tr.iloc[fit][continuous]).astype("float32")
    clustering = np.column_stack([cat_init, num_init])
    config = dict(vars(args), columns=columns, support=support,
                  model="Class-conditional mixture of categorical product densities",
                  protocol="Fixed 30 EM iterations unless overridden; outer validation labels excluded from all fitting and initialization",
                  selection="Three prespecified capacities on canonical fold 0; exploratory pilot, not full OOF",
                  vocabulary_scope="Unlabeled train+test; no test labels",
                  class_smoothing="income: 10 * class prior per category; other fields: 20 * prior",
                  blend_weights=[.01, .05, .1, .2], submitted=False,
                  code_sha256=digest(Path(__file__)), train_sha256=digest(ROOT / "train.csv"),
                  test_sha256=digest(ROOT / "test.csv"),
                  fold_sha256=digest(ROOT / "artifacts/folds_seed42.npy"))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "training_script.py").write_bytes(Path(__file__).read_bytes())
    anchor = np.load(ROOT / "artifacts/offline_wave7_tabm_transfer25_v1/oof.npy")[valid]
    records = []
    for k in args.components:
        states = []
        t0 = time.time()
        for label in [0, 1]:
            positions = np.flatnonzero(y[fit] == label)
            prior = float((y[fit] == label).mean())
            if k > 1:
                km = MiniBatchKMeans(n_clusters=k, random_state=615 + label,
                                    batch_size=8192, n_init=3, max_iter=100)
                initial = km.fit_predict(clustering[positions])
            else:
                initial = np.zeros(len(positions), dtype="int64")
            state = class_density(codes[fit[positions]], cardinalities, initial, prior,
                                  k, args.iterations, args.component_shrink, columns)
            state["log_prior"] = np.log(prior)
            states.append(state)
            log(dict(event="density_fitted", components=k, label=label,
                     last_iteration=state["history"][-1] if state["history"] else None))
        path = out / f"model_k{k}.pt"
        torch.save(states, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        margin = (log_density(loaded[1], codes[valid]) + loaded[1]["log_prior"]
                  - log_density(loaded[0], codes[valid]) - loaded[0]["log_prior"])
        test_margin = (log_density(loaded[1], codes[len(tr):]) + loaded[1]["log_prior"]
                       - log_density(loaded[0], codes[len(tr):]) - loaded[0]["log_prior"])
        assert np.isfinite(margin).all() and np.isfinite(test_margin).all()
        rec = dict(components=k, auc=float(roc_auc_score(y[valid], margin)),
                   anchor_auc=float(roc_auc_score(y[valid], anchor)),
                   rank_correlation=float(np.corrcoef(rank(margin), rank(anchor))[0, 1]),
                   seconds=time.time() - t0, model_sha256=digest(path), blends=[])
        for w in config["blend_weights"]:
            score = (1 - w) * rank(anchor) + w * rank(margin)
            auc = float(roc_auc_score(y[valid], score))
            rec["blends"].append(dict(weight=w, auc=auc, gain=auc-rec["anchor_auc"]))
        np.savez_compressed(out / f"fold_{args.fold}_k{k}.npz", valid_indices=valid,
                            valid_margin=margin, test_margin=test_margin,
                            valid_prediction=expit(margin), test_prediction=expit(test_margin))
        records.append(rec)
        log(rec)
    report = dict(run=args.run, status="pilot_complete", submitted=False, public_auc=None,
                  complete_outer_folds=[args.fold], comparisons=records,
                  elapsed_seconds=time.time()-start,
                  limitation="One previously explored outer fold; vocabulary uses unlabeled train+test. No independent leaderboard evidence.")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

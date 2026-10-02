"""Discriminatively refine a class-conditional categorical density ratio.

The generative pilot supplies fixed class density priors. Small trainable
deviations and smooth numerical backgrounds optimize binary log loss directly.
This tests whether latent product-mixture interactions help without spending
capacity on fitting target-irrelevant parts of the feature distribution.
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer, StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from train_encoded import ROOT, TARGET, digest


class DensityRatio(nn.Module):
    def __init__(self, states, n_numeric, columns):
        super().__init__()
        self.prior = nn.ModuleList()
        self.delta = nn.ModuleList()
        self.columns = columns
        self.k = states[0]["log_mix"].shape[0]
        for j in range(len(columns)):
            weights = torch.cat([state["log_theta"][j] for state in states], dim=0).T
            self.prior.append(nn.Embedding.from_pretrained(weights, freeze=True))
            embedding = nn.Embedding(weights.shape[0], weights.shape[1])
            nn.init.zeros_(embedding.weight)
            self.delta.append(embedding)
        self.register_buffer("prior_mix", torch.cat([s["log_mix"] for s in states]))
        self.mix_delta = nn.Parameter(torch.zeros(2*self.k))
        self.bias = nn.Parameter(torch.tensor(float(states[1]["log_prior"]-states[0]["log_prior"])))
        self.temperature = nn.Parameter(torch.ones(()))
        self.numeric = nn.Linear(n_numeric, 1, bias=False)
        nn.init.zeros_(self.numeric.weight)

    def forward(self, categories, numeric):
        scores = (self.prior_mix + self.mix_delta)[None].expand(len(categories), -1)
        for j, (prior, delta) in enumerate(zip(self.prior, self.delta)):
            scores = scores + prior(categories[:, j]) + delta(categories[:, j])
        negative = torch.logsumexp(scores[:, :self.k], dim=1)
        positive = torch.logsumexp(scores[:, self.k:], dim=1)
        return self.temperature * (positive-negative) + self.bias + self.numeric(numeric).squeeze(1)

    def penalty(self, n, strength):
        value = 0.
        for name, embedding in zip(self.columns, self.delta):
            multiplier = 2.5 if name == "Daily_Commute_km" else 1.
            value = value + multiplier * embedding.weight.square().sum()
        return strength*value/(2*n) + self.numeric.weight.square().sum()/n


@torch.inference_mode()
def predict(model, cats, nums):
    model.eval()
    return np.concatenate([model(cats[i:i+32768], nums[i:i+32768]).cpu().numpy()
                           for i in range(0, len(cats), 32768)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--prior-run", default="generative_mixture_pilot_v2")
    parser.add_argument("--components", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--penalty", type=float, default=20.)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=.01)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(611)
    assert torch.cuda.is_available()
    out = ROOT / "artifacts" / args.run
    out.mkdir(exist_ok=False)
    prior_dir = ROOT / "artifacts" / args.prior_run
    prior_config = json.loads((prior_dir / "config.json").read_text())
    tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = tr[TARGET].eq("Yes").to_numpy("float32")
    all_rows = pd.concat([tr, te], ignore_index=True)
    columns = prior_config["columns"]
    categories = np.column_stack([pd.Categorical(all_rows[c], categories=prior_config["support"][c]).codes
                                  for c in columns]).astype("int64")
    assert (categories >= 0).all()
    folds = np.load(ROOT / "artifacts/folds_seed42.npy")
    fit = np.flatnonzero(folds != prior_config["fold"])
    valid = np.flatnonzero(folds == prior_config["fold"])
    numeric_columns = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
    spline = SplineTransformer(n_knots=32, degree=3, knots="quantile", include_bias=False)
    spline.fit(tr.iloc[fit][numeric_columns])
    raw_numeric = spline.transform(all_rows[numeric_columns]).astype("float32")
    scaler = StandardScaler().fit(raw_numeric[fit])
    numeric = scaler.transform(raw_numeric).astype("float32")
    train_c, train_n, target = [torch.as_tensor(v, device="cuda") for v in
                               (categories[fit], numeric[fit], y[fit])]
    valid_c, valid_n = [torch.as_tensor(v, device="cuda") for v in (categories[valid], numeric[valid])]
    test_c, test_n = [torch.as_tensor(v, device="cuda") for v in (categories[len(tr):], numeric[len(tr):])]
    config = dict(vars(args), outer_fold=prior_config["fold"], seed=611,
                  prior_config_sha256=digest(prior_dir / "config.json"),
                  code_sha256=digest(Path(__file__)),
                  protocol="Fixed epoch budget; no outer validation score used during fitting; generative priors and splines fit outer training labels/rows only",
                  comparison_weights=[.01, .05, .1, .2], submitted=False,
                  train_sha256=digest(ROOT/"train.csv"), test_sha256=digest(ROOT/"test.csv"))
    (out/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    (out/"training_script.py").write_bytes(Path(__file__).read_bytes())
    import joblib
    joblib.dump(dict(spline=spline, scaler=scaler, numeric_columns=numeric_columns,
                     columns=columns, support=prior_config["support"]), out/"preprocessor.joblib")
    anchor = np.load(ROOT/"artifacts/offline_wave7_tabm_transfer25_v1/oof.npy")[valid]
    ar = rankdata(anchor)/len(anchor)
    base_auc = float(roc_auc_score(y[valid], anchor))
    reports = []
    for k in args.components:
        started = time.time()
        states = torch.load(prior_dir/f"model_k{k}.pt", map_location="cpu", weights_only=False)
        model = DensityRatio(states, numeric.shape[1], columns).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        history = []
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in torch.randperm(len(fit), device="cuda").split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                z = model(train_c[batch], train_n[batch])
                loss = F.binary_cross_entropy_with_logits(z, target[batch]) + model.penalty(len(fit), args.penalty)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch, train_objective=float(np.mean(losses))))
            if (epoch+1) % 10 == 0:
                print(json.dumps(dict(event="trained", components=k, **history[-1])), flush=True)
        path = out/f"model_k{k}.pt"
        torch.save(model.state_dict(), path)
        margin, test_margin = predict(model, valid_c, valid_n), predict(model, test_c, test_n)
        replay = DensityRatio(states, numeric.shape[1], columns).cuda()
        replay.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        check = predict(replay, valid_c[:1024], valid_n[:1024])
        np.testing.assert_allclose(check, margin[:1024], rtol=1e-5, atol=1e-5)
        np.savez_compressed(out/f"fold_{prior_config['fold']}_k{k}.npz", valid_indices=valid,
                            valid_margin=margin, test_margin=test_margin,
                            valid_prediction=expit(margin), test_prediction=expit(test_margin))
        rec = dict(components=k, auc=float(roc_auc_score(y[valid], margin)),
                   anchor_auc=base_auc, rank_correlation=float(np.corrcoef(ar, rankdata(margin)/len(margin))[0,1]),
                   seconds=time.time()-started, model_sha256=digest(path),
                   reload_max_error=float(np.max(np.abs(check-margin[:1024]))), history=history, blends=[])
        for w in config["comparison_weights"]:
            auc = float(roc_auc_score(y[valid], (1-w)*ar+w*rankdata(margin)/len(margin)))
            rec["blends"].append(dict(weight=w, auc=auc, gain=auc-base_auc))
        reports.append(rec)
        print(json.dumps({a:b for a,b in rec.items() if a != "history"}), flush=True)
    (out/"report.json").write_text(json.dumps(dict(run=args.run, status="pilot_complete", submitted=False,
        public_auc=None, comparisons=reports, limitation="One previously explored outer fold; no independent Public evidence"), indent=2)+"\n")


if __name__ == "__main__":
    main()

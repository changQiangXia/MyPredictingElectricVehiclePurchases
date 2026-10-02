"""Complete five-fold validation for the generative density-ratio probe."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

from train_encoded import ROOT, TARGET, digest
from train_generative_mixture import class_density, log_density


def rank(v):
    return rankdata(v, method="average") / len(v)


def main():
    run = "generative_mixture_oof_v1"
    out = ROOT / "artifacts" / run
    out.mkdir(exist_ok=False)
    started = time.time()
    tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = tr[TARGET].eq("Yes").to_numpy("int8")
    columns = [c for c in te if c != "id"]
    all_rows = pd.concat([tr[columns], te[columns]], ignore_index=True)
    arrays, support = [], {}
    for c in columns:
        code, levels = pd.factorize(all_rows[c], sort=True)
        arrays.append(code.astype("int64")); support[c] = levels.tolist()
    codes = np.column_stack(arrays)
    cardinalities = [len(support[c]) for c in columns]
    folds = np.load(ROOT / "artifacts/folds_seed42.npy")
    anchor_oof = np.load(ROOT / "artifacts/offline_wave7_tabm_transfer25_v1/oof.npy")
    anchor_test = np.load(ROOT / "artifacts/offline_wave7_tabm_transfer25_v1/test.npy")
    columns_small = [c for c in columns if all_rows[c].nunique() <= 20]
    columns_num = [c for c in columns if c not in columns_small]
    records, all_oof, all_test = [], {}, {}
    for k in [1, 8]:
        oof, test_pred = np.full(len(tr), np.nan), np.zeros(len(te))
        for fold in range(5):
            fit, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            cat = encoder.fit_transform(tr.iloc[fit][columns_small])
            scaler = StandardScaler().fit(tr.iloc[fit][columns_num])
            num = scaler.transform(tr.iloc[fit][columns_num]).astype("float32")
            clustering = np.column_stack([cat, num])
            states = []
            for label in [0, 1]:
                positions = np.flatnonzero(y[fit] == label)
                if k > 1:
                    km = MiniBatchKMeans(n_clusters=k, random_state=615 + label + fold,
                                         batch_size=8192, n_init=3, max_iter=100)
                    init = km.fit_predict(clustering[positions]).astype("int64")
                else:
                    init = np.zeros(len(positions), dtype="int64")
                state = class_density(codes[fit[positions]], cardinalities, init,
                                      float((y[fit] == label).mean()), k, 30, 1000., columns)
                state["log_prior"] = np.log(float((y[fit] == label).mean()))
                states.append(state)
            valid_margin = (log_density(states[1], codes[valid]) + states[1]["log_prior"]
                            - log_density(states[0], codes[valid]) - states[0]["log_prior"])
            test_margin = (log_density(states[1], codes[len(tr):]) + states[1]["log_prior"]
                           - log_density(states[0], codes[len(tr):]) - states[0]["log_prior"])
            oof[valid], test_pred = valid_margin, test_pred + test_margin / 5
            rec = dict(components=k, fold=fold, auc=float(roc_auc_score(y[valid], valid_margin)),
                       valid_rows=len(valid), seconds=time.time()-started)
            records.append(rec); print(json.dumps(rec), flush=True)
            torch.cuda.empty_cache()
        all_oof[f"k{k}"] = oof; all_test[f"k{k}"] = test_pred
    base_auc = float(roc_auc_score(y, anchor_oof))
    blends = []
    for k in [1, 8]:
        p, t = all_oof[f"k{k}"], all_test[f"k{k}"]
        entry = dict(components=k, auc=float(roc_auc_score(y, p)), correlation=float(np.corrcoef(rank(anchor_oof), rank(p))[0,1]), weights=[])
        for w in [.002, .005, .01, .02, .05]:
            score = (1-w)*rank(anchor_oof) + w*rank(p)
            auc = float(roc_auc_score(y, score)); folds_gain=[]
            for f in range(5):
                m=folds==f; folds_gain.append(float(roc_auc_score(y[m],score[m])-roc_auc_score(y[m],anchor_oof[m])))
            entry["weights"].append(dict(weight=w, auc=auc, gain=auc-base_auc, fold_gains=folds_gain))
        blends.append(entry)
        np.save(out/f"oof_k{k}.npy",p); np.save(out/f"test_k{k}.npy",t)
    report=dict(run=run,status="complete",submitted=False,public_auc=None,base_auc=base_auc,
                folds=records,blends=blends,elapsed_seconds=time.time()-started,
                protocol="Five fixed outer folds; class density and mixture initialization use outer training rows only; no outer labels used for selection",
                components=[1,8],iterations=30,component_shrink=1000.,
                train_sha256=digest(ROOT/"train.csv"),test_sha256=digest(ROOT/"test.csv"),
                fold_sha256=digest(ROOT/"artifacts/folds_seed42.npy"))
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"training_script.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(blends, indent=2), flush=True)


if __name__ == "__main__": main()

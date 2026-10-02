"""Outer-fold-safe gain-reranked feature selection.

The canonical fold models already contain feature importances generated from
their respective outer training partitions. We use each fold's ranking only
for that same fold's validation prediction, then retrain a fixed-size model
on the selected columns. The feature count is fixed before evaluation.
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

import train_encoded as runner

ROOT = runner.ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=800)
    args = parser.parse_args()
    start = time.time()
    out = ROOT / "artifacts" / args.run
    out.mkdir(exist_ok=False)
    tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = tr[runner.TARGET].eq("Yes").to_numpy(dtype="int8")
    folds = np.load(ROOT / "artifacts/folds_seed42.npy")
    base, keys, mappings = runner.features(tr, te, False)
    X, Xt = base.iloc[:len(tr)].to_numpy(dtype="float32"), base.iloc[len(tr):].to_numpy(dtype="float32")
    K, Kt = keys.iloc[:len(tr)].to_numpy(), keys.iloc[len(tr):].to_numpy()
    names = list(base.columns) + [f"{c}_TE_{s}" for s in ("auto", 10.0, 100.0) for c in keys]
    assert len(names) == X.shape[1] + 3 * K.shape[1]
    oof = np.full(len(tr), np.nan); test_pred = np.zeros(len(te)); records=[]
    selected_all=[]
    for fold in range(5):
        fs=time.time(); ti=np.flatnonzero(folds != fold); vi=np.flatnonzero(folds == fold)
        train_parts, valid_parts, test_parts=[X[ti]],[X[vi]],[Xt]
        encoders=[]
        for smoothing in ("auto",10.0,100.0):
            enc=TargetEncoder(target_type="binary",smooth=smoothing,cv=5,shuffle=True,random_state=42)
            train_parts.append(enc.fit_transform(K[ti],y[ti]).astype("float32"))
            valid_parts.append(enc.transform(K[vi]).astype("float32"))
            test_parts.append(enc.transform(Kt).astype("float32"));encoders.append(enc)
        A,B,C=[np.concatenate(p,axis=1) for p in (train_parts,valid_parts,test_parts)]
        imp=pd.read_csv(ROOT/f"artifacts/xgb_nested_v1/importance_{fold}.csv",index_col=0)
        ranking=[n for n in imp.iloc[:,0].sort_values(ascending=False).index if n in names]
        assert len(ranking)==len(names),f"importance mismatch fold {fold}: {len(ranking)} / {len(names)}"
        selected=ranking[:args.top_k];ix=np.array([names.index(n) for n in selected],dtype=np.int64)
        model=XGBClassifier(device="cuda",tree_method="hist",max_bin=1024,
            n_estimators=args.iterations,max_depth=5,learning_rate=0.03,min_child_weight=10,
            subsample=0.9,colsample_bytree=0.85,reg_alpha=0.071,reg_lambda=2.0,
            objective="binary:logistic",eval_metric="auc",n_jobs=8,random_state=42)
        model.fit(A[:,ix],y[ti],verbose=False);model.set_params(device="cpu")
        vp=model.predict_proba(B[:,ix])[:,1];tp=model.predict_proba(C[:,ix])[:,1]
        oof[vi]=vp;test_pred += tp/5
        model.save_model(out/f"model_{fold}.ubj");joblib.dump(encoders,out/f"encoders_{fold}.joblib")
        rec={"fold":fold,"auc":float(roc_auc_score(y[vi],vp)),"top_k":args.top_k,
             "iterations":args.iterations,"seconds":time.time()-fs,
             "selected_features":selected}
        records.append(rec);selected_all.append(selected)
        (out/f"fold_{fold}.json").write_text(json.dumps(rec,indent=2)+"\n")
        print(json.dumps({k:v for k,v in rec.items() if k!='selected_features'}),flush=True)
        del A,B,C,model,encoders
    assert np.isfinite(oof).all()
    np.save(out/"oof.npy",oof);np.save(out/"test.npy",test_pred)
    sub=pd.read_csv(ROOT/"sample_submission.csv");assert sub.id.equals(te.id);sub[runner.TARGET]=test_pred;sub.to_csv(out/"submission.csv",index=False)
    metrics={"run":args.run,"status":"complete","submitted":False,"public_auc":None,
             "oof_auc":float(roc_auc_score(y,oof)),"folds":records,"fold_mean":float(np.mean([r['auc'] for r in records])),
             "fold_std":float(np.std([r['auc'] for r in records])),"top_k":args.top_k,"iterations":args.iterations,
             "selection_protocol":"Each fold ranking loaded from canonical model trained on that fold's outer training rows; fixed top_k and fixed iterations; exploratory because the ranking source used validation-driven early stopping.",
             "code_sha256":runner.digest(Path(__file__)),"train_sha256":runner.digest(ROOT/"train.csv"),
             "test_sha256":runner.digest(ROOT/"test.csv"),"submitted":False,"submission_sha256":runner.digest(out/"submission.csv"),
             "elapsed_seconds":time.time()-start}
    (out/"config.json").write_text(json.dumps({"top_k":args.top_k,"iterations":args.iterations,"code_sha256":runner.digest(Path(__file__))},indent=2)+"\n")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['folds']}),flush=True)


if __name__=="__main__":main()

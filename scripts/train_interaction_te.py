"""Nested target encodings for exact numeric values conditioned on regimes."""
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


def key_frame(train, test):
    all_rows = pd.concat([train.drop(columns=[runner.TARGET]), test], ignore_index=True)
    income = all_rows.Annual_Income_USD.round().astype(str)
    commute = all_rows.Daily_Commute_km.round(1).astype(str)
    env = all_rows.Environmental_Concern_Level.astype(str)
    subsidy = all_rows.Subsidy_Available.astype(str)
    anxiety = all_rows.Range_Anxiety_Level.astype(str)
    pairs = {
        "income_x_env": (income, env),
        "income_x_subsidy": (income, subsidy),
        "income_x_anxiety": (income, anxiety),
        "income_x_env_subsidy": (income, env, subsidy),
        "income_x_env_anxiety": (income, env, anxiety),
        "income_x_subsidy_anxiety": (income, subsidy, anxiety),
        "income_x_env_subsidy_anxiety": (income, env, subsidy, anxiety),
        "commute_x_env": (commute, env),
        "commute_x_subsidy": (commute, subsidy),
        "commute_x_anxiety": (commute, anxiety),
    }
    result = {}
    for name, parts in pairs.items():
        key = parts[0]
        for part in parts[1:]: key = key + "\x1f" + part
        result[name] = pd.factorize(key, sort=True)[0].astype("int64")
    return pd.DataFrame(result)


def main():
    run = "xgb_interaction_te_v1"
    out = runner.ROOT / "artifacts" / run
    out.mkdir(exist_ok=False)
    started = time.time()
    train, test = pd.read_csv(runner.ROOT/"train.csv"), pd.read_csv(runner.ROOT/"test.csv")
    y = train[runner.TARGET].eq("Yes").to_numpy("int8")
    folds = np.load(runner.ROOT/"artifacts/folds_seed42.npy")
    base, base_keys, _ = runner.features(train, test, False)
    X, Xt = base.iloc[:len(train)].to_numpy("float32"), base.iloc[len(train):].to_numpy("float32")
    K = key_frame(train, test).to_numpy(); n = len(train)
    oof, tp, records = np.full(n, np.nan), np.zeros(len(test)), []
    config = dict(run=run, interaction_keys=list(key_frame(train,test).columns), smooth=["auto",10.,100.],
                  validation="five outer folds; inner fivefold cross-fit for every interaction encoding",
                  submitted=False, code_sha256=runner.digest(Path(__file__)),
                  runner_sha256=runner.digest(Path(runner.__file__)),
                  train_sha256=runner.digest(runner.ROOT/"train.csv"), test_sha256=runner.digest(runner.ROOT/"test.csv"))
    (out/"config.json").write_text(json.dumps(config, indent=2)+"\n"); (out/"training_script.py").write_bytes(Path(__file__).read_bytes())
    for fold in range(5):
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        parts=[[X[ti]],[X[vi]],[Xt]]; encoders=[]; t0=time.time()
        for smooth in ("auto",10.,100.):
            enc=TargetEncoder(target_type="binary",smooth=smooth,cv=5,shuffle=True,random_state=42)
            parts[0].append(enc.fit_transform(K[ti],y[ti]).astype("float32")); parts[1].append(enc.transform(K[vi]).astype("float32")); parts[2].append(enc.transform(K[n:]).astype("float32")); encoders.append(enc)
        A,B,C=[np.concatenate(v,axis=1) for v in parts]
        model=XGBClassifier(device="cuda",tree_method="hist",max_bin=1024,n_estimators=6500,max_depth=5,learning_rate=.03,min_child_weight=10,subsample=.9,colsample_bytree=.85,reg_alpha=.071,reg_lambda=2.,objective="binary:logistic",eval_metric="auc",early_stopping_rounds=300,n_jobs=8,random_state=42)
        model.fit(A,y[ti],eval_set=[(B,y[vi])],verbose=False);model.set_params(device="cpu");vp=model.predict_proba(B)[:,1];pred=model.predict_proba(C)[:,1]
        oof[vi],tp=vp,tp+pred/5
        rec=dict(fold=fold,auc=float(roc_auc_score(y[vi],vp)),best_iteration=int(model.best_iteration)+1,seconds=time.time()-t0);records.append(rec);print(json.dumps(rec),flush=True)
        np.savez_compressed(out/f"fold_{fold}.npz",valid_indices=vi,valid_prediction=vp,test_prediction=pred);model.save_model(out/f"model_{fold}.ubj");joblib.dump(encoders,out/f"encoders_{fold}.joblib")
        del A,B,C,model,encoders
    np.save(out/"oof.npy",oof);np.save(out/"test.npy",tp);pd.DataFrame({"id":test.id,runner.TARGET:tp}).to_csv(out/"submission.csv",index=False)
    metrics=dict(run=run,status="complete",submitted=False,public_auc=None,oof_auc=float(roc_auc_score(y,oof)),fold_mean=float(np.mean([r["auc"] for r in records])),fold_std=float(np.std([r["auc"] for r in records])),folds=records,submission_sha256=runner.digest(out/"submission.csv"),elapsed_seconds=time.time()-started)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics,indent=2),flush=True)


if __name__ == "__main__": main()

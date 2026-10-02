"""Strict test-style transfer for full-data TE; outer holdout never used for stopping."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier
from train_encoded import ROOT, TARGET, features
from train_xgb_formula_cv import margin_values

OUT = ROOT / "artifacts/experimental/full_te_teststyle_strict_v1"
OUT.mkdir(parents=True, exist_ok=True)
def rk(x): return rankdata(x, method="average") / len(x)

tr, te = pd.read_csv(ROOT/"train.csv"), pd.read_csv(ROOT/"test.csv")
y = tr[TARGET].eq("Yes").to_numpy("int8")
base, keys, _ = features(tr, te, False)
X, K = base.to_numpy()[:len(tr)], keys.to_numpy()[:len(tr)]
margin = margin_values(tr, "probit")
outer = np.load(ROOT/"artifacts/diagnosis_bagging_transfer_v1/outer_folds.npy")
records=[]
for f in [0,1,2]:
    hold=np.flatnonzero(outer==f); dev=np.flatnonzero(outer!=f)
    fit_local, stop_local=next(StratifiedKFold(10,shuffle=True,random_state=812+f).split(dev,y[dev]))
    fit,stop=dev[fit_local],dev[stop_local]
    parts=[[X[idx]] for idx in [fit,stop,hold]]
    for smooth in ("auto",10.,100.):
        enc=TargetEncoder(target_type="binary",smooth=smooth,cv=5,shuffle=True,random_state=42)
        parts[0].append(enc.fit_transform(K[fit],y[fit]).astype("float32"))
        parts[1].append(enc.transform(K[stop]).astype("float32"))
        parts[2].append(enc.transform(K[hold]).astype("float32"))
    A,B,C=[np.column_stack(v) for v in parts]
    params=dict(device="cuda",tree_method="hist",max_bin=1024,n_estimators=6500,max_depth=5,learning_rate=.03,min_child_weight=10,subsample=.9,colsample_bytree=.85,reg_alpha=.071,reg_lambda=2.,objective="binary:logistic",eval_metric="auc",early_stopping_rounds=300,n_jobs=8,random_state=42)
    m=XGBClassifier(**params); t=time.time()
    m.fit(A,y[fit],base_margin=margin[fit],eval_set=[(B,y[stop])],base_margin_eval_set=[margin[stop]],verbose=False)
    m.set_params(device="cpu"); fp=m.predict_proba(C,base_margin=margin[hold])[:,1]
    bag=np.mean([np.load(ROOT/f"artifacts/diagnosis_bagging_transfer_v1/outer_{f}_seed_42_member_{j}.npz")["heldout_prediction"] for j in range(10)],axis=0)
    mod=np.load(ROOT/f"artifacts/experimental/modulo_transfer_audit_v1/outer_{f}.npz")["mod1000_s100_full_outer80"]
    rbag,rfull,rmod=rk(bag),rk(fp),rk(mod); base_auc=roc_auc_score(y[hold],rbag)
    for wf in [.1,.15,.2,.25,.3]:
        for wm in [.004,.006,.008]:
            auc=roc_auc_score(y[hold],(1-wf-wm)*rbag+wf*rfull+wm*rmod)
            records.append(dict(outer=f,w_full=wf,w_mod=wm,bag_auc=base_auc,full_auc=roc_auc_score(y[hold],rfull),auc=auc,delta=auc-base_auc,best_iteration=int(m.best_iteration)+1,seconds=time.time()-t))
    np.save(OUT/f"full_{f}.npy",fp)
    print(json.dumps(dict(outer=f,bag_auc=base_auc,full_auc=roc_auc_score(y[hold],rfull),best_iteration=int(m.best_iteration)+1,seconds=time.time()-t)),flush=True)
df=pd.DataFrame(records); df.to_csv(OUT/"comparison.csv",index=False)
g=df.groupby(["w_full","w_mod"]).delta.agg(["mean","min","max","std"]).reset_index()
report=dict(run="full_te_teststyle_strict_v1",status="complete",protocol="outer holdout labels unused for TE, fitting, or early stopping; 10% inner stop split inside outer development",summary=g.to_dict("records"),best_mean=g.sort_values("mean",ascending=False).head(20).to_dict("records"),best_min=g.sort_values(["min","mean"],ascending=False).head(20).to_dict("records"),submitted=False)
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

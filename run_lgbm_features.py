from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
TARGET, ID, SEED, N_SPLITS = "Will_Buy_EV", "id", 42, 3

def make_features(train, test):
    tr = train.drop(columns=[TARGET]).copy()
    te = test.copy()
    allx = pd.concat([tr, te], axis=0, ignore_index=True)
    num = [c for c in te.columns if c != ID and pd.api.types.is_numeric_dtype(te[c])]
    # Frequency encodings and decimal/integer digit features capture synthetic generator artifacts.
    for c in num:
        vc = allx[c].value_counts(dropna=False)
        tr[c + "__freq"] = tr[c].map(vc).astype("float32")
        te[c + "__freq"] = te[c].map(vc).astype("float32")
        vals = allx[c].fillna(-999999).astype(float)
        ints = np.floor(np.abs(vals)).astype(np.int64)
        for k in range(6):
            d = ((ints // (10 ** k)) % 10).astype("int8")
            tr[c + f"__digit{k}"] = d.iloc[:len(tr)].to_numpy()
            te[c + f"__digit{k}"] = d.iloc[len(tr):].to_numpy()
        # tenths and hundredths, useful for commute values
        for k in (1, 2):
            d = (np.floor(np.abs(vals) * (10 ** k) + 1e-5) % 10).astype("int8")
            tr[c + f"__frac{k}"] = d.iloc[:len(tr)].to_numpy()
            te[c + f"__frac{k}"] = d.iloc[len(tr):].to_numpy()
    # Explicit deterministic edges and a few high-signal interactions.
    for d in (tr, te):
        inc = d["Annual_Income_USD"]
        commute = d["Daily_Commute_km"]
        d["income_high_edge"] = (inc >= 170537).astype("int8")
        d["income_dead_zone"] = ((inc >= 31004) & (inc <= 41970)).astype("int8")
        d["commute_long_edge"] = (commute >= 83).astype("int8")
        d["commute_is_5"] = (commute == 5).astype("int8")
        d["subsidy_yes"] = (d["Subsidy_Available"] == "Yes").astype("int8")
        d["home_charge_yes"] = (d["Home_Charging_Possible"] == "Yes").astype("int8")
        d["range_high"] = (d["Range_Anxiety_Level"] == "High").astype("int8")
        d["subsidy_env"] = d["subsidy_yes"] * d["Environmental_Concern_Level"]
        d["subsidy_range"] = d["subsidy_yes"] * (d["Range_Anxiety_Level"] == "Low").astype("int8")
        d["income_env"] = d["Annual_Income_USD"] / 100000.0 * d["Environmental_Concern_Level"]
    cat = [c for c in te.columns if te[c].dtype == "object"]
    for c in cat:
        # Shared category codes ensure train/test alignment.
        allv = pd.concat([tr[c], te[c]], ignore_index=True).astype("category")
        codes = allv.cat.codes.astype("int16")
        tr[c] = codes.iloc[:len(tr)].to_numpy()
        te[c] = codes.iloc[len(tr):].to_numpy()
    return tr.drop(columns=[ID]), te.drop(columns=[ID])

def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = train[TARGET].eq("Yes").astype("int8").to_numpy()
    X, Xt = make_features(train, test)
    oof = np.zeros(len(train), dtype="float32"); pred = np.zeros(len(test), dtype="float32")
    scores=[]; start=time.time()
    folds=StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    for fold,(ti,vi) in enumerate(folds.split(X,y)):
        model=lgb.LGBMClassifier(
            n_estimators=1800, learning_rate=0.04, num_leaves=31,
            max_depth=-1, min_child_samples=80, subsample=0.9,
            colsample_bytree=0.85, reg_lambda=2.0, reg_alpha=0.1,
            max_bin=1023, objective="binary", random_state=SEED+fold,
            n_jobs=64, verbosity=-1)
        model.fit(X.iloc[ti], y[ti], eval_set=[(X.iloc[vi], y[vi])],
                  callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        vp=model.predict_proba(X.iloc[vi], num_iteration=model.best_iteration_)[:,1]
        tp=model.predict_proba(Xt, num_iteration=model.best_iteration_)[:,1]
        oof[vi]=vp; pred += tp/N_SPLITS
        sc=roc_auc_score(y[vi],vp); scores.append(float(sc))
        print(json.dumps({"fold":fold,"auc":sc,"best_iteration":int(model.best_iteration_),"seconds":time.time()-start}))
    cv=float(roc_auc_score(y,oof)); out=ROOT/"artifacts"; out.mkdir(exist_ok=True)
    np.save(out/"oof_lgbm_features.npy",oof); np.save(out/"test_lgbm_features.npy",pred)
    pd.DataFrame({ID:test[ID],TARGET:pred}).to_csv(out/"submission_lgbm_features.csv",index=False)
    metrics={"model":"lightgbm_frequency_digits","cv_auc_oof":cv,"fold_auc":scores,"fold_mean":float(np.mean(scores)),"fold_std":float(np.std(scores)),"n_features":X.shape[1],"elapsed_seconds":time.time()-start}
    (out/"lgbm_features_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()

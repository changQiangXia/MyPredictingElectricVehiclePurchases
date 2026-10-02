"""LightGBM counterpart with nested value encodings and formula features."""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy.special import logit, ndtr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import lightgbm as lgb
from train_encoded import ROOT, TARGET, digest, features


def formula(frame):
    anx=frame.Range_Anxiety_Level.map({'Low':0.,'Medium':-1.,'High':-3.}).fillna(-1.)
    score=(1.2*frame.Annual_Income_USD/1e5 + .6*frame.Environmental_Concern_Level
           + 2.*frame.Subsidy_Available.eq('Yes').astype(float) + anx - 5.5)
    return np.column_stack([score.to_numpy('float32'), logit(np.clip(ndtr(score),1e-6,1-1e-6)).astype('float32')])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--n-splits',type=int,default=10);ap.add_argument('--fold-seed',type=int,default=42);ap.add_argument('--folds',nargs='+',type=int);ap.add_argument('--learning-rate',type=float,default=.03);ap.add_argument('--iterations',type=int,default=6000);ap.add_argument('--num-leaves',type=int,default=32);ap.add_argument('--colsample',type=float,default=.45);ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
    requested=list(range(args.n_splits)) if args.folds is None else args.folds
    assert requested and set(requested).issubset(range(args.n_splits))
    out=ROOT/'artifacts'/args.run;out.mkdir(parents=True,exist_ok=True);started=time.time()
    tr,te=pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv');n=len(tr);y=tr[TARGET].eq('Yes').to_numpy('int8')
    fold_path=ROOT/'artifacts'/f'folds_seed{args.fold_seed}_{args.n_splits}.npy';expected=np.full(n,-1,dtype='int8')
    for f,(_,vi) in enumerate(StratifiedKFold(args.n_splits,shuffle=True,random_state=args.fold_seed).split(tr,y)):expected[vi]=f
    if fold_path.exists():folds=np.load(fold_path);np.testing.assert_array_equal(folds,expected)
    else:folds=expected;np.save(fold_path,folds)
    base,keys,mappings=features(tr,te,False);X=base.to_numpy();K=keys.to_numpy();F=formula(pd.concat([tr,te],ignore_index=True));names=list(base.columns)+[f'{c}_TE_{s}' for s in ('auto',10.,100.) for c in keys]+['formula_linear','formula_probit']
    params=dict(n_estimators=args.iterations,num_leaves=args.num_leaves,max_depth=5,learning_rate=args.learning_rate,min_child_samples=20,colsample_bytree=args.colsample,subsample=.9,subsample_freq=1,reg_alpha=.071,reg_lambda=2.,objective='binary',metric='auc',verbosity=-1,n_jobs=8,random_state=args.seed,force_col_wise=True,max_bin=1024)
    config={k:v for k,v in vars(args).items() if k!='folds'};config.update(code_sha256=digest(Path(__file__)),feature_code_sha256=digest(ROOT/'train_encoded.py'),train_sha256=digest(ROOT/'train.csv'),test_sha256=digest(ROOT/'test.csv'),fold_sha256=digest(fold_path),inner_splits=5,smoothing=['auto',10.,100.],features=names,model_params=params,validation='outer folds; inner 5-fold target encoding',submitted=False)
    cp=out/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Changed configuration requires fresh run ID'
    else:cp.write_text(json.dumps(config,indent=2)+'\n');(out/'training_script.py').write_bytes(Path(__file__).read_bytes())
    for f in requested:
        ck=out/f'fold_{f}.npz'
        if ck.exists():print(json.dumps({'event':'resumed','fold':f}),flush=True);continue
        t0=time.time();ti,vi=np.flatnonzero(folds!=f),np.flatnonzero(folds==f);parts=[[X[ti],F[ti]],[X[vi],F[vi]],[X[n:],F[n:]]];encoders=[]
        for smooth in ('auto',10.,100.):
            e=TargetEncoder(target_type='binary',smooth=smooth,cv=5,shuffle=True,random_state=42);parts[0].append(e.fit_transform(K[ti],y[ti]).astype('float32'));parts[1].append(e.transform(K[vi]).astype('float32'));parts[2].append(e.transform(K[n:]).astype('float32'));encoders.append(e)
        A,B,C=[np.column_stack(z) for z in parts]
        model=lgb.LGBMClassifier(**params);model.fit(A,y[ti],eval_set=[(B,y[vi])],callbacks=[lgb.early_stopping(300,verbose=False),lgb.log_evaluation(1000)])
        vp,tp=model.predict_proba(B)[:,1],model.predict_proba(C)[:,1];model.booster_.save_model(str(out/f'model_{f}.txt'));joblib.dump(encoders,out/f'encoders_{f}.joblib')
        replay=lgb.Booster(model_file=str(out/f'model_{f}.txt')).predict(C[:1024]);np.testing.assert_allclose(replay,tp[:1024],atol=1e-7,rtol=1e-6)
        rec=dict(fold=f,auc=float(roc_auc_score(y[vi],vp)),best_iteration=int(model.best_iteration_),seconds=time.time()-t0,reload_checked_rows=1024,reload_max_error=float(np.max(np.abs(replay-tp[:1024]))),model_sha256=digest(out/f'model_{f}.txt'))
        np.savez_compressed(ck,valid_indices=vi,valid_prediction=vp,test_prediction=tp);(out/f'fold_{f}.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True);del A,B,C,model,encoders
    available=[f for f in range(args.n_splits) if (out/f'fold_{f}.npz').exists()];oof=np.full(n,np.nan);pred=np.zeros(len(te));records=[]
    for f in available:
        z=np.load(out/f'fold_{f}.npz');vi=np.flatnonzero(folds==f);np.testing.assert_array_equal(z['valid_indices'],vi);oof[vi]=z['valid_prediction'];pred+=z['test_prediction']/len(available);records.append(json.loads((out/f'fold_{f}.json').read_text()))
    complete=np.isfinite(oof).all();metrics=dict(run=args.run,status='complete' if complete else 'pilot',submitted=False,folds=records,n_splits=args.n_splits,elapsed_seconds=time.time()-started)
    if complete:
        np.save(out/'oof.npy',oof);np.save(out/'test.npy',pred);sub=pd.read_csv(ROOT/'sample_submission.csv');assert sub.id.equals(te.id);sub[TARGET]=pred;sub.to_csv(out/'submission.csv',index=False);metrics.update(oof_auc=float(roc_auc_score(y,oof)),submission_sha256=digest(out/'submission.csv'))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()

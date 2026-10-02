"""Resumable XGBoost with nested encodings, formula offsets, and saved models."""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, ndtr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

from train_encoded import ROOT, TARGET, digest, features


def margin_values(frame, link):
    anxiety = frame.Range_Anxiety_Level.map({'Low': 0., 'Medium': -1., 'High': -3.}).fillna(-1.)
    score = (1.2*frame.Annual_Income_USD/1e5 + .6*frame.Environmental_Concern_Level
             + 2.*frame.Subsidy_Available.eq('Yes').astype(float) + anxiety - 5.5)
    result = score.to_numpy('float64')
    if link == 'probit':
        result = logit(np.clip(ndtr(result), 1e-6, 1-1e-6))
    return result.astype('float32')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--n-splits', type=int, default=10)
    parser.add_argument('--fold-seed', type=int, default=42)
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--folds', type=int, nargs='+')
    parser.add_argument('--link', choices=['probit','linear'], default='probit')
    parser.add_argument('--iterations', type=int, default=6500)
    args = parser.parse_args()
    requested = list(range(args.n_splits)) if args.folds is None else args.folds
    assert len(requested)==len(set(requested)) and set(requested).issubset(range(args.n_splits))
    started = time.time()
    out = ROOT/'artifacts'/args.run
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(ROOT/'train.csv'), pd.read_csv(ROOT/'test.csv')
    n = len(tr)
    y = tr[TARGET].eq('Yes').to_numpy('int8')
    fold_path = ROOT/'artifacts'/f'folds_seed{args.fold_seed}_{args.n_splits}.npy'
    expected = np.full(n, -1, dtype='int8')
    for f, (_,vi) in enumerate(StratifiedKFold(args.n_splits, shuffle=True, random_state=args.fold_seed).split(tr,y)):
        expected[vi] = f
    if fold_path.exists():
        folds = np.load(fold_path)
        np.testing.assert_array_equal(folds, expected)
    else:
        folds = expected
        np.save(fold_path, folds)
    base, keys, mappings = features(tr, te, False)
    X, K = base.to_numpy(), keys.to_numpy()
    raw = pd.concat([tr,te], ignore_index=True)
    margin = margin_values(raw, args.link)
    names = list(base.columns) + [f'{c}_TE_{s}' for s in ('auto',10.,100.) for c in keys]
    model_params = dict(device='cuda', tree_method='hist', max_bin=1024,
        n_estimators=args.iterations, max_depth=5, learning_rate=.03,
        min_child_weight=10., subsample=.9, colsample_bytree=.85, reg_alpha=.071,
        reg_lambda=2., objective='binary:logistic', eval_metric='auc',
        early_stopping_rounds=300, n_jobs=8, random_state=args.model_seed)
    config = {k:v for k,v in vars(args).items() if k != 'folds'}
    config.update(code_sha256=digest(Path(__file__)), feature_code_sha256=digest(ROOT/'train_encoded.py'),
        train_sha256=digest(ROOT/'train.csv'), test_sha256=digest(ROOT/'test.csv'),
        fold_sha256=digest(fold_path), inner_splits=5, inner_seed=42, smoothing=['auto',10.,100.],
        features=names, category_mappings=mappings, model_params=model_params,
        frequency_scope='unlabeled train+test', validation='outer folds; inner crossfit for training TE',
        comparison_weights=[.1,.2,.3], submitted=False)
    cp = out/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config, 'Changed configuration requires a fresh run ID'
    else:
        cp.write_text(json.dumps(config,indent=2)+'\n')
        (out/'training_script.py').write_bytes(Path(__file__).read_bytes())
    for f in requested:
        checkpoint = out/f'fold_{f}.npz'
        if checkpoint.exists():
            print(json.dumps(dict(event='resumed',fold=f)),flush=True)
            continue
        t0 = time.time()
        ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        parts = [[X[ti]], [X[vi]], [X[n:]]]
        encoders=[]
        for smooth in ('auto',10.,100.):
            enc = TargetEncoder(target_type='binary', smooth=smooth, cv=5, shuffle=True, random_state=42)
            parts[0].append(enc.fit_transform(K[ti],y[ti]).astype('float32'))
            parts[1].append(enc.transform(K[vi]).astype('float32'))
            parts[2].append(enc.transform(K[n:]).astype('float32'))
            encoders.append(enc)
        A,B,C = [np.column_stack(p) for p in parts]
        model=XGBClassifier(**model_params)
        model.fit(A,y[ti],base_margin=margin[ti],eval_set=[(B,y[vi])],
                  base_margin_eval_set=[margin[vi]],verbose=1000)
        model.set_params(device='cpu')
        vp=model.predict_proba(B,base_margin=margin[vi])[:,1]
        tp=model.predict_proba(C,base_margin=margin[n:])[:,1]
        model_path=out/f'model_{f}.ubj'
        model.save_model(model_path)
        joblib.dump(encoders,out/f'encoders_{f}.joblib')
        # Rebuild both encoded rows and the predictor from disk.
        reloaded=joblib.load(out/f'encoders_{f}.joblib')
        picks=np.linspace(n,len(raw)-1,1024,dtype=int)
        replay_X=np.column_stack([X[picks]]+[e.transform(K[picks]).astype('float32') for e in reloaded])
        replay_model=XGBClassifier()
        replay_model.load_model(model_path)
        replay_model.set_params(device='cpu',n_jobs=8)
        replay=replay_model.predict_proba(replay_X,base_margin=margin_values(raw.iloc[picks],args.link))[:,1]
        np.testing.assert_allclose(replay,tp[picks-n],atol=1e-7,rtol=1e-6)
        rec=dict(fold=f,auc=float(roc_auc_score(y[vi],vp)), best_iteration=int(model.best_iteration)+1,
            seconds=time.time()-t0, reload_checked_rows=1024,
            reload_max_error=float(np.max(np.abs(replay-tp[picks-n]))),model_sha256=digest(model_path))
        np.savez_compressed(checkpoint,valid_indices=vi,valid_prediction=vp,test_prediction=tp)
        (out/f'fold_{f}.json').write_text(json.dumps(rec,indent=2)+'\n')
        print(json.dumps(rec),flush=True)
        del A,B,C,model,replay_model,parts,encoders
    available=[f for f in range(args.n_splits) if (out/f'fold_{f}.npz').exists()]
    oof,test_pred,count,records=np.full(n,np.nan),np.zeros(len(te)),np.zeros(n,dtype='int8'),[]
    for f in available:
        z=np.load(out/f'fold_{f}.npz');vi=np.flatnonzero(folds==f)
        np.testing.assert_array_equal(z['valid_indices'],vi)
        oof[vi]=z['valid_prediction'];count[vi]+=1
        test_pred+=z['test_prediction']/len(available)
        records.append(json.loads((out/f'fold_{f}.json').read_text()))
    complete=np.isfinite(oof).all()
    metrics=dict(run=args.run,status='complete' if complete else 'pilot',submitted=False,folds=records,
        elapsed_seconds=time.time()-started, n_splits=args.n_splits)
    if complete:
        assert (count==1).all() and np.isfinite(test_pred).all()
        np.save(out/'oof.npy',oof);np.save(out/'test.npy',test_pred)
        sub=pd.read_csv(ROOT/'sample_submission.csv')
        assert sub.id.equals(te.id) and ((test_pred>=0)&(test_pred<=1)).all()
        sub[TARGET]=test_pred;sub.to_csv(out/'submission.csv',index=False)
        metrics.update(oof_auc=float(roc_auc_score(y,oof)), fold_mean=float(np.mean([r['auc'] for r in records])),
            oof_sha256=digest(out/'oof.npy'),test_sha256=digest(out/'test.npy'),
            submission_sha256=digest(out/'submission.csv'))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    main()

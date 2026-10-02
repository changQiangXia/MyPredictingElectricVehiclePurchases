"""Confirm linear/probit margins on a new split with inner-only early stopping."""
import os

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, ndtr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

from evaluate_offline import auc_placements, paired_delta
from train_encoded import ROOT, TARGET, digest, features


def main():
    start = time.time()
    out = ROOT / 'artifacts' / 'offline_wave2_margin_confirmation'
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    y = tr[TARGET].eq('Yes').to_numpy('int8')
    base, keys, _ = features(tr, te, False)
    X, K = base.iloc[:len(tr)].to_numpy(), keys.iloc[:len(tr)].to_numpy()
    del base, keys
    linear = (1.2 * tr.Annual_Income_USD / 1e5 + 0.6 * tr.Environmental_Concern_Level
              + 2 * tr.Subsidy_Available.eq('Yes').astype(float)
              + tr.Range_Anxiety_Level.map({'Low': 0., 'Medium': -1., 'High': -3.}) - 5.5).to_numpy('float32')
    margins = {'linear': linear, 'probit': logit(np.clip(ndtr(linear.astype('float64')), 1e-6, 1 - 1e-6)).astype('float32')}
    ti, vi = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=y, random_state=20260909)
    inner_train, inner_stop = train_test_split(ti, test_size=0.2, stratify=y[ti], random_state=60909)
    np.savez_compressed(out / 'split_indices.npz', outer_train=ti, outer_valid=vi,
                        inner_train=inner_train, inner_stop=inner_stop)
    params = dict(device='cuda', tree_method='hist', max_bin=1024, max_depth=5,
                  learning_rate=0.03, min_child_weight=10, subsample=0.9,
                  colsample_bytree=0.85, reg_alpha=0.071, reg_lambda=2.,
                  objective='binary:logistic', eval_metric='auc', n_jobs=8, random_state=42)

    def matrices(fit_indices, valid_indices):
        a, b, encoders = [X[fit_indices]], [X[valid_indices]], []
        for smoothing in ('auto', 10.0, 100.0):
            encoder = TargetEncoder(target_type='binary', smooth=smoothing, cv=5,
                                    shuffle=True, random_state=42)
            a.append(encoder.fit_transform(K[fit_indices], y[fit_indices]).astype('float32'))
            b.append(encoder.transform(K[valid_indices]).astype('float32'))
            encoders.append(encoder)
        return np.concatenate(a, axis=1), np.concatenate(b, axis=1), encoders

    a, b, _ = matrices(inner_train, inner_stop)
    rounds = {}
    for name, margin in margins.items():
        model = XGBClassifier(**params, n_estimators=5000, early_stopping_rounds=300)
        model.fit(a, y[inner_train], base_margin=margin[inner_train],
                  eval_set=[(b, y[inner_stop])], base_margin_eval_set=[margin[inner_stop]], verbose=False)
        rounds[name] = int(model.best_iteration) + 1
        print(json.dumps({'stage': 'inner_stop', 'model': name, 'rounds': rounds[name]}), flush=True)
    del a, b, model
    a, b, encoders = matrices(ti, vi)
    joblib.dump(encoders, out / 'outer_encoders.joblib')
    predictions = {}
    for name, margin in margins.items():
        model = XGBClassifier(**params, n_estimators=rounds[name])
        model.fit(a, y[ti], base_margin=margin[ti], verbose=False)
        model.set_params(device='cpu')
        predictions[name] = model.predict_proba(b, base_margin=margin[vi])[:, 1]
        model.save_model(out / f'{name}.ubj')
        print(json.dumps({'stage': 'outer_holdout', 'model': name,
                          'auc': float(roc_auc_score(y[vi], predictions[name]))}), flush=True)
    np.savez_compressed(out / 'predictions.npz', indices=vi, target=y[vi], **predictions)
    report = {'submitted': False, 'train_sha256': digest(ROOT / 'train.csv'),
              'code_sha256': digest(Path(__file__)), 'outer_split_seed': 20260909,
              'inner_split_seed': 60909, 'outer_holdout_rows': len(vi),
              'early_stopping': 'Inner training partition only; outer holdout scored after refitting fixed rounds',
              'params': params, 'rounds': rounds,
              'linear_auc': float(roc_auc_score(y[vi], predictions['linear'])),
              'probit_auc': float(roc_auc_score(y[vi], predictions['probit'])),
              **paired_delta(y[vi].astype(bool), predictions['linear'], predictions['probit'],
                             auc_placements(y[vi].astype(bool), predictions['linear'])),
              'elapsed_seconds': time.time() - start,
              'limitation': 'New random split of previously explored data, not a wholly untouched external validation sample.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

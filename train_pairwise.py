"""Binary purchase ranking using XGBoost's pairwise logistic ranking objective."""
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
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

from train_encoded import ROOT, TARGET, digest, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--iterations', type=int, default=4500)
    parser.add_argument('--pairs', type=int, default=4)
    args = parser.parse_args()
    assert set(args.folds).issubset(set(range(5)))
    start = time.time()
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    y = tr[TARGET].eq('Yes').to_numpy('int8')
    n = len(tr)
    fold_path = ROOT / 'artifacts' / 'folds_seed42.npy'
    folds = np.load(fold_path)
    assert folds.shape == y.shape and set(folds) == set(range(5))
    frame, keys, mappings = features(tr, te, False)
    X, K = frame.to_numpy('float32'), keys.to_numpy()
    names = list(frame)
    for smoothing in ('auto', 10.0, 100.0):
        names.extend(f'{c}_TE_{smoothing}' for c in keys)
    del frame, keys
    params = dict(objective='rank:pairwise', eval_metric='auc', device='cuda',
                  tree_method='hist', max_bin=1024,
                  max_depth=5, learning_rate=0.03, min_child_weight=10,
                  subsample=0.9, colsample_bytree=0.85, reg_alpha=0.071,
                  reg_lambda=2.0, nthread=8, seed=42,
                  lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=args.pairs)
    config = {'run': args.run, 'params': params, 'features': names, 'category_mappings': mappings,
              'code_sha256': digest(Path(__file__)), 'feature_code_sha256': digest(ROOT / 'train_encoded.py'),
              'train_sha256': digest(ROOT / 'train.csv'), 'test_sha256': digest(ROOT / 'test.csv'),
              'fold_sha256': digest(fold_path), 'n_splits': 5,
              'ranking_groups': 'One global training group; validation has no groups and uses ordinary binary AUC',
              'max_rounds': args.iterations, 'early_stopping_rounds': 300,
              'output_transform': 'sigmoid of ranking score, uncalibrated; assess/blend by rank',
              'submitted': False}
    config_path = out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Use a new run ID'
    else:
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'event': 'start', 'run': args.run}), flush=True)
    for fold in args.folds:
        checkpoint = out / f'fold_{fold}.npz'
        if checkpoint.exists():
            continue
        fold_start = time.time()
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        parts, encoders = [[X[ti]], [X[vi]], [X[n:]]], []
        for smoothing in ('auto', 10.0, 100.0):
            enc = TargetEncoder(target_type='binary', smooth=smoothing, cv=5, shuffle=True, random_state=42)
            parts[0].append(enc.fit_transform(K[ti], y[ti]).astype('float32'))
            parts[1].append(enc.transform(K[vi]).astype('float32'))
            parts[2].append(enc.transform(K[n:]).astype('float32'))
            encoders.append(enc)
        A, B, C = [np.concatenate(p, axis=1) for p in parts]
        del parts
        print(json.dumps({'event': 'encoded', 'fold': fold, 'seconds': time.time() - fold_start}), flush=True)
        dtrain = xgb.DMatrix(A, label=y[ti], nthread=8)
        dtrain.set_group([len(ti)])
        # The ranking AUC implementation enumerates pairs within query groups.
        # An ungrouped validation matrix computes the intended ordinary binary AUC.
        dvalid = xgb.DMatrix(B, label=y[vi], nthread=8)
        dtest = xgb.DMatrix(C, nthread=8)
        model = xgb.train(params, dtrain, num_boost_round=args.iterations,
                          evals=[(dvalid, 'valid')], early_stopping_rounds=300, verbose_eval=500)
        best_iteration = int(model.best_iteration) + 1
        model = model[:best_iteration]
        model.set_param({'device': 'cpu'})
        vp, tp = expit(model.predict(dvalid).astype('float64')), expit(model.predict(dtest).astype('float64'))
        model_path = out / f'model_{fold}.ubj'
        model.save_model(model_path)
        replay = xgb.Booster()
        replay.load_model(model_path)
        replay.set_param({'device': 'cpu'})
        np.testing.assert_allclose(expit(replay.predict(xgb.DMatrix(C[:1024])).astype('float64')), tp[:1024], rtol=1e-7)
        assert np.isfinite(vp).all() and np.isfinite(tp).all()
        np.savez_compressed(checkpoint, valid_indices=vi, valid_prediction=vp, test_prediction=tp)
        joblib.dump(encoders, out / f'encoders_{fold}.joblib')
        record = {'fold': fold, 'auc': float(roc_auc_score(y[vi], vp)),
                  'best_iteration': best_iteration, 'seconds': time.time() - fold_start,
                  'model_sha256': digest(model_path), 'reload_checked_rows': 1024}
        (out / f'fold_{fold}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        del A, B, C, model, replay, encoders, dtrain, dvalid, dtest
    available = [f for f in range(5) if (out / f'fold_{f}.npz').exists()]
    oof, test_prediction, records = np.full(n, np.nan), np.zeros(len(te)), []
    for fold in available:
        saved = np.load(out / f'fold_{fold}.npz')
        vi = np.flatnonzero(folds == fold)
        assert np.array_equal(vi, saved['valid_indices'])
        oof[vi] = saved['valid_prediction']
        test_prediction += saved['test_prediction'] / len(available)
        records.append(json.loads((out / f'fold_{fold}.json').read_text()))
    complete = len(available) == 5
    metrics = {'run': args.run, 'status': 'complete' if complete else 'pilot', 'submitted': False,
               'folds': records, 'elapsed_seconds_this_invocation': time.time() - start}
    if complete:
        metrics.update({'oof_auc': float(roc_auc_score(y, oof)),
                        'fold_mean': float(np.mean([r['auc'] for r in records])),
                        'fold_std': float(np.std([r['auc'] for r in records]))})
        np.save(out / 'oof.npy', oof)
        np.save(out / 'test.npy', test_prediction)
        sub = pd.read_csv(ROOT / 'sample_submission.csv')
        assert sub.id.equals(te.id) and np.isfinite(test_prediction).all()
        assert ((test_prediction >= 0) & (test_prediction <= 1)).all()
        sub[TARGET] = test_prediction
        sub.to_csv(out / 'submission.csv', index=False)
        metrics['submission_sha256'] = digest(out / 'submission.csv')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()

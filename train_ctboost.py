"""CTBoost diversity experiment on the existing nested-encoding feature set."""
import os

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
import json
import time
from pathlib import Path

import ctboost
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

from train_encoded import ROOT, TARGET, digest, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--iterations', type=int, default=1426)
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--seed', type=int, default=20260904)
    args = parser.parse_args()
    assert set(args.folds).issubset(set(range(5)))
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    n = len(tr)
    y = tr[TARGET].eq('Yes').to_numpy('int8')
    fold_path = ROOT / 'artifacts' / 'folds_seed42.npy'
    folds = np.load(fold_path)
    assert folds.shape == y.shape and set(folds) == set(range(5))
    frame, keys, mappings = features(tr, te, False)
    names = list(frame)
    X, K = frame.to_numpy('float32'), keys.to_numpy()
    for smoothing in ('auto', 10.0, 100.0):
        names.extend(f'{c}_TE_{smoothing}' for c in keys)
    del frame, keys
    params = dict(iterations=args.iterations, task_type='GPU', devices='0',
                  random_seed=args.seed, eval_metric='AUC', verbose=False,
                  learning_rate=0.039, max_depth=4, max_leaves=16,
                  grow_policy='LeafWise', alpha=0.5, lambda_l2=8.0,
                  min_data_in_leaf=100, min_child_weight=0.1, subsample=0.85,
                  bootstrap_type='Bernoulli', colsample_bytree=0.3,
                  feature_test='quadratic', feature_test_adjustment='none',
                  feature_test_bins=8, max_bins=1024, leaf_estimation_iterations=3)
    config = {'run': args.run, 'params': params, 'code_sha256': digest(Path(__file__)),
              'feature_code_sha256': digest(ROOT / 'train_encoded.py'),
              'train_sha256': digest(ROOT / 'train.csv'), 'test_sha256': digest(ROOT / 'test.csv'),
              'fold_sha256': digest(fold_path), 'n_splits': 5, 'ctboost_version': ctboost.__version__,
              'features': names, 'category_mappings': mappings,
              'round_selection': 'fixed from public notebook; no outer-validation early stopping',
              'parameter_source': 'https://www.kaggle.com/code/maiernator/s6e9-ctboost-not-catboost-astra-baseline',
              'submitted': False}
    config_path = out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Use a new run ID'
    else:
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'event': 'start', 'run': args.run, 'features': len(names)}), flush=True)
    for fold in args.folds:
        checkpoint_path = out / f'fold_{fold}.npz'
        if checkpoint_path.exists():
            continue
        fold_start = time.time()
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        parts = [[X[ti]], [X[vi]], [X[n:]]]
        encoders = []
        for smoothing in ('auto', 10.0, 100.0):
            enc = TargetEncoder(target_type='binary', smooth=smoothing, cv=5,
                                shuffle=True, random_state=42)
            parts[0].append(enc.fit_transform(K[ti], y[ti]).astype('float32'))
            parts[1].append(enc.transform(K[vi]).astype('float32'))
            parts[2].append(enc.transform(K[n:]).astype('float32'))
            encoders.append(enc)
        A, B, C = [np.concatenate(p, axis=1) for p in parts]
        del parts
        print(json.dumps({'event': 'encoded', 'fold': fold, 'seconds': time.time() - fold_start}), flush=True)
        model = ctboost.CTBoostClassifier(**params).fit(A, y[ti])
        vp, tp = model.predict_proba(B)[:, 1], model.predict_proba(C)[:, 1]
        assert np.isfinite(vp).all() and np.isfinite(tp).all()
        model_path = out / f'model_{fold}.json'
        model.save_model(str(model_path))
        replay = ctboost.CTBoostClassifier.load_model(str(model_path)).predict_proba(C[:1024])[:, 1]
        np.testing.assert_allclose(replay, tp[:1024], rtol=1e-7, atol=1e-8)
        joblib.dump(encoders, out / f'encoders_{fold}.joblib')
        np.savez_compressed(checkpoint_path, valid_indices=vi, valid_prediction=vp, test_prediction=tp)
        record = {'fold': fold, 'auc': float(roc_auc_score(y[vi], vp)),
                  'seconds': time.time() - fold_start, 'model_sha256': digest(model_path),
                  'reload_checked_rows': 1024}
        (out / f'fold_{fold}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        del A, B, C, model, encoders
    available = [f for f in range(5) if (out / f'fold_{f}.npz').exists()]
    oof, prediction, records = np.full(n, np.nan), np.zeros(len(te)), []
    for fold in available:
        saved = np.load(out / f'fold_{fold}.npz')
        vi = np.flatnonzero(folds == fold)
        assert np.array_equal(vi, saved['valid_indices'])
        oof[vi] = saved['valid_prediction']
        prediction += saved['test_prediction'] / len(available)
        records.append(json.loads((out / f'fold_{fold}.json').read_text()))
    complete = len(available) == 5
    metrics = {'run': args.run, 'folds': records, 'status': 'complete' if complete else 'pilot',
               'submitted': False, 'elapsed_seconds_this_invocation': time.time() - start}
    if complete:
        metrics.update({'oof_auc': float(roc_auc_score(y, oof)),
                        'fold_mean': float(np.mean([r['auc'] for r in records])),
                        'fold_std': float(np.std([r['auc'] for r in records]))})
        np.save(out / 'oof.npy', oof)
        np.save(out / 'test.npy', prediction)
        sub = pd.read_csv(ROOT / 'sample_submission.csv')
        assert sub.id.equals(te.id) and np.isfinite(prediction).all()
        assert ((prediction >= 0) & (prediction <= 1)).all()
        sub[TARGET] = prediction
        sub.to_csv(out / 'submission.csv', index=False)
        metrics['submission_sha256'] = digest(out / 'submission.csv')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()

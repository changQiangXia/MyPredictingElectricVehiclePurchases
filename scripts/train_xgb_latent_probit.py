"""XGBoost on the latent additive probit scale, with nested target encoding.

The current production comparator uses a probit-derived logit offset and a
logistic likelihood. This model instead optimizes Bernoulli probit likelihood
with the generator's untransformed latent formula as its offset.
"""
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
from scipy.special import log_ndtr, ndtr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

from train_encoded import ROOT, TARGET, digest, features
from train_xgb_formula_cv import margin_values


def probit_derivatives(y, margin):
    """Gradient and positive Hessian of -log Phi((2y-1) * margin)."""
    sign = 2 * np.asarray(y, dtype='float64') - 1
    signed = sign * np.asarray(margin, dtype='float64')
    mills = np.exp(-.5 * signed**2 - .5 * np.log(2 * np.pi) - log_ndtr(signed))
    gradient = -sign * mills
    hessian = mills * (mills + signed)
    return gradient.astype('float32'), np.maximum(hessian, 1e-7).astype('float32')


def make_model(iterations=6500):
    return XGBClassifier(objective=probit_derivatives, device='cuda', tree_method='hist',
        max_bin=1024, n_estimators=iterations, max_depth=5, learning_rate=.03,
        min_child_weight=10., subsample=.9, colsample_bytree=.85,
        reg_alpha=.071, reg_lambda=2., eval_metric='auc', early_stopping_rounds=300,
        n_jobs=8, random_state=42)


def latent_predict(model, X, margin):
    # predict_proba would apply the sklearn wrapper's logistic output transform.
    # Raw output + ndtr is required, especially before averaging test folds.
    raw = model.predict(X, base_margin=margin, output_margin=True)
    return ndtr(np.asarray(raw, dtype='float64'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--folds', type=int, nargs='+')
    parser.add_argument('--n-splits', type=int, default=10)
    parser.add_argument('--fold-seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=6500)
    args = parser.parse_args()
    requested = list(range(args.n_splits)) if args.folds is None else args.folds
    assert len(requested) == len(set(requested)) and set(requested).issubset(range(args.n_splits))
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    y = tr[TARGET].eq('Yes').to_numpy('int8')
    n = len(tr)
    fold_path = ROOT / 'artifacts' / f'folds_seed{args.fold_seed}_{args.n_splits}.npy'
    folds = np.full(n, -1, dtype='int8')
    for f, (_, vi) in enumerate(StratifiedKFold(args.n_splits, shuffle=True,
                                              random_state=args.fold_seed).split(tr, y)):
        folds[vi] = f
    if fold_path.exists():
        np.testing.assert_array_equal(folds, np.load(fold_path))
    else:
        np.save(fold_path, folds)
    base, keys, mappings = features(tr, te, False)
    X, K = base.to_numpy(), keys.to_numpy()
    margin = margin_values(pd.concat([tr, te], ignore_index=True), 'linear')
    names = list(base) + [f'{c}_TE_{s}' for s in ('auto', 10., 100.) for c in keys]
    config = {k: v for k, v in vars(args).items() if k != 'folds'}
    config.update(code_sha256=digest(Path(__file__)),
        feature_code_sha256=digest(ROOT / 'train_encoded.py'),
        margin_code_sha256=digest(ROOT / 'train_xgb_formula_cv.py'),
        train_sha256=digest(ROOT / 'train.csv'), test_sha256=digest(ROOT / 'test.csv'),
        fold_sha256=digest(fold_path), features=names, category_mappings=mappings,
        objective='Bernoulli probit likelihood; exact gradient and Hessian',
        output_transform='ndtr(raw margin), then probability average across folds',
        offset='1.2*income/1e5 + .6*concern + 2*subsidy + anxiety - 5.5',
        validation='outer folds, inner fivefold TE; outer early stopping reused for model selection',
        frequency_scope='unlabeled full train+test', submitted=False,
        comparator='xgb_probit_10fold_v1', comparison_weights=[.1, .2, .3])
    cp = out / 'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config, 'Changed configuration requires fresh run ID'
    else:
        cp.write_text(json.dumps(config, indent=2) + '\n')
        (out / 'training_script.py').write_bytes(Path(__file__).read_bytes())
    for f in requested:
        checkpoint = out / f'fold_{f}.npz'
        if checkpoint.exists():
            print(json.dumps(dict(event='resumed', fold=f)), flush=True)
            continue
        t0 = time.time()
        ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        parts, encoders = [[X[ti]], [X[vi]], [X[n:]]], []
        for smooth in ('auto', 10., 100.):
            enc = TargetEncoder(target_type='binary', smooth=smooth, cv=5, shuffle=True, random_state=42)
            parts[0].append(enc.fit_transform(K[ti], y[ti]).astype('float32'))
            parts[1].append(enc.transform(K[vi]).astype('float32'))
            parts[2].append(enc.transform(K[n:]).astype('float32'))
            encoders.append(enc)
        A, B, C = [np.column_stack(p) for p in parts]
        model = make_model(args.iterations)
        model.fit(A, y[ti], base_margin=margin[ti], eval_set=[(B, y[vi])],
                  base_margin_eval_set=[margin[vi]], verbose=1000)
        model.set_params(device='cpu')
        vp, tp = latent_predict(model, B, margin[vi]), latent_predict(model, C, margin[n:])
        mp, ep = out / f'model_{f}.ubj', out / f'encoders_{f}.joblib'
        model.save_model(mp)
        joblib.dump(encoders, ep)
        loaded = joblib.load(ep)
        picks = np.linspace(n, len(X) - 1, 1024, dtype=int)
        replay_X = np.column_stack([X[picks]] + [e.transform(K[picks]).astype('float32') for e in loaded])
        replay_model = XGBClassifier()
        replay_model.load_model(mp)
        replay_model.set_params(device='cpu', n_jobs=8)
        replay = latent_predict(replay_model, replay_X, margin[picks])
        np.testing.assert_allclose(replay, tp[picks - n], atol=1e-7, rtol=1e-6)
        assert np.isfinite(vp).all() and np.isfinite(tp).all()
        rec = dict(fold=f, auc=float(roc_auc_score(y[vi], vp)),
            best_iteration=int(model.best_iteration) + 1, seconds=time.time() - t0,
            model_sha256=digest(mp), reload_checked_rows=1024,
            reload_max_error=float(np.max(np.abs(replay - tp[picks - n]))))
        np.savez_compressed(checkpoint, valid_indices=vi, valid_prediction=vp, test_prediction=tp)
        (out / f'fold_{f}.json').write_text(json.dumps(rec, indent=2) + '\n')
        print(json.dumps(rec), flush=True)
        del A, B, C, parts, encoders, model, replay_model
    available = [f for f in range(args.n_splits) if (out / f'fold_{f}.npz').exists()]
    oof, tp, count, records = np.full(n, np.nan), np.zeros(len(te)), np.zeros(n, dtype='int8'), []
    for f in available:
        z = np.load(out / f'fold_{f}.npz')
        vi = np.flatnonzero(folds == f)
        np.testing.assert_array_equal(z['valid_indices'], vi)
        oof[vi] = z['valid_prediction']
        count[vi] += 1
        tp += z['test_prediction'] / len(available)
        records.append(json.loads((out / f'fold_{f}.json').read_text()))
    complete = len(available) == args.n_splits
    metrics = dict(run=args.run, status='complete' if complete else 'pilot',
        submitted=False, public_auc=None, folds=records, elapsed_seconds=time.time() - start)
    if complete:
        assert (count == 1).all() and np.isfinite(oof).all() and np.isfinite(tp).all()
        np.save(out / 'oof.npy', oof)
        np.save(out / 'test.npy', tp)
        sub = pd.read_csv(ROOT / 'sample_submission.csv')
        assert sub.id.equals(te.id) and sub.id.is_unique and ((tp >= 0) & (tp <= 1)).all()
        sub[TARGET] = tp
        sub.to_csv(out / 'submission.csv', index=False)
        metrics.update(oof_auc=float(roc_auc_score(y, oof)),
            fold_mean=float(np.mean([r['auc'] for r in records])),
            oof_sha256=digest(out / 'oof.npy'), test_sha256=digest(out / 'test.npy'),
            submission_sha256=digest(out / 'submission.csv'))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()

"""Fixed five-fold training with nested target encoding and resumable artifacts."""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'Will_Buy_EV'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(train, test, original):
    columns = [c for c in test if c != 'id']
    raw = pd.concat([train[columns], test[columns]], ignore_index=True)
    numeric = list(raw.select_dtypes('number').columns)
    base, keys = {}, {}
    mappings = {}
    for c in columns:
        if c in numeric:
            base[c] = raw[c].to_numpy(dtype='float32')
            keys[c] = raw[c].to_numpy()
        else:
            categories = sorted(raw[c].unique())
            mappings[c] = {v: i for i, v in enumerate(categories)}
            base[c] = raw[c].map(mappings[c]).to_numpy(dtype='float32')
            keys[c] = base[c]
    # Floating floor residues are deliberate generator fingerprints. They are
    # distinct from mathematical decimal digits, especially at negative powers.
    for c in numeric:
        values = raw[c].to_numpy(dtype='float64')
        for k in range(-4, 4):
            base[f'{c}_floor_residue_{k}'] = (values // (10.0 ** k) % 10).astype('float32')
    seen, unique = set(), {}
    for c, values in base.items():
        if np.min(values) == np.max(values):
            continue
        key = hashlib.sha256(values.tobytes()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique[c] = values
    base = unique
    for c, values in list(base.items()):
        frequency = pd.Series(values).value_counts(normalize=True)
        base[f'{c}_frequency'] = pd.Series(values).map(frequency).to_numpy(dtype='float32')
    for divisor in (100, 1000):
        keys[f'income_floor_{divisor}'] = np.floor(raw.Annual_Income_USD / divisor).to_numpy()
    keys['commute_floor'] = np.floor(raw.Daily_Commute_km).to_numpy()
    if original:
        orig_path = next((ROOT / 'original').glob('*.csv'))
        orig = pd.read_csv(orig_path)
        orig[TARGET] = orig[TARGET].eq('Yes').astype('int8')
        prior = orig[TARGET].mean()
        for c in columns:
            means = orig.groupby(c)[TARGET].mean()
            base[f'{c}_original_mean'] = raw[c].map(means).fillna(prior).to_numpy(dtype='float32')
    return pd.DataFrame(base), pd.DataFrame(keys), mappings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['xgb', 'lgbm', 'cat'], required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.03)
    parser.add_argument('--iterations', type=int, default=6500)
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--min-child-weight', type=float, default=10.0)
    parser.add_argument('--subsample', type=float, default=0.9)
    parser.add_argument('--colsample-bytree', type=float, default=0.85)
    parser.add_argument('--reg-alpha', type=float, default=0.071)
    parser.add_argument('--reg-lambda', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--grow-policy', choices=['depthwise', 'lossguide'], default='depthwise')
    parser.add_argument('--max-leaves', type=int, default=0)
    parser.add_argument('--formula-margin', action='store_true')
    parser.add_argument('--margin-link', choices=['linear', 'probit'], default='linear')
    parser.add_argument('--max-bin', type=int, default=1024)
    parser.add_argument('--smoothing-values', nargs='+', type=str,
                        default=['auto', '10.0', '100.0'])
    parser.add_argument('--fold-seed', type=int, default=42)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train, test = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    y = train[TARGET].eq('Yes').to_numpy(dtype='int8')
    fold_path = ROOT / 'artifacts' / f'folds_seed{args.fold_seed}.npy'
    if not fold_path.exists():
        fold_ids = np.full(len(train), -1, dtype='int8')
        for f, (_, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=args.fold_seed).split(train, y)):
            fold_ids[vi] = f
        np.save(fold_path, fold_ids)
    fold_ids = np.load(fold_path)
    assert fold_ids.shape == y.shape and set(fold_ids) == set(range(5))
    base, keys, category_mappings = features(train, test, args.original)
    n = len(train)
    X, Xt = base.iloc[:n].to_numpy(), base.iloc[n:].to_numpy()
    K, Kt = keys.iloc[:n].to_numpy(), keys.iloc[n:].to_numpy()
    margin = margin_t = None
    if args.formula_margin:
        all_rows = pd.concat([train, test], ignore_index=True)
        anxiety = all_rows['Range_Anxiety_Level'].map({'Low': 0.0, 'Medium': -1.0, 'High': -3.0}).fillna(-1.0)
        score = (1.2 * all_rows['Annual_Income_USD'] / 1e5 +
                 0.6 * all_rows['Environmental_Concern_Level'] +
                 2.0 * all_rows['Subsidy_Available'].eq('Yes').astype(float) + anxiety)
        formula_margin = (score - 5.5).to_numpy(dtype='float64')
        if args.margin_link == 'probit':
            from scipy.special import logit, ndtr
            formula_margin = logit(np.clip(ndtr(formula_margin), 1e-6, 1 - 1e-6))
        margin = formula_margin[:n].astype('float32')
        margin_t = formula_margin[n:].astype('float32')
    names = list(base.columns)
    smoothing_values = tuple('auto' if s == 'auto' else float(s)
                             for s in args.smoothing_values)
    if not smoothing_values:
        raise ValueError('at least one smoothing value is required')
    for smoothing in smoothing_values:
        names += [f'{c}_TE_{smoothing}' for c in keys]
    config = dict(vars(args), n_splits=5, fold_seed=args.fold_seed, inner_splits=5,
                  smoothing=list(smoothing_values), features=names,
                  frequency_scope='unlabeled train+test (transductive)',
                  category_mappings=category_mappings,
                  code_sha256=digest(Path(__file__)), fold_sha256=digest(fold_path),
                  train_sha256=digest(ROOT / 'train.csv'), test_sha256=digest(ROOT / 'test.csv'))
    # `smoothing` is the canonical persisted setting.  Keep the newer CLI
    # spelling out of the manifest so historical default runs remain resumable.
    config.pop('smoothing_values', None)
    config_path = out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Run config changed: use a new run ID'
    else:
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'event': 'start', 'run': args.run, 'features': len(names)}), flush=True)
    oof, test_pred = np.full(n, np.nan), np.zeros(len(test))
    scores = []
    for fold in range(5):
        fold_start = time.time()
        ti, vi = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        checkpoint = out / f'fold_{fold}.npz'
        if checkpoint.exists():
            saved = np.load(checkpoint)
            assert np.array_equal(saved['valid_indices'], vi)
            oof[vi], tp = saved['valid_prediction'], saved['test_prediction']
            test_pred += tp / 5
            scores.append(json.loads((out / f'fold_{fold}.json').read_text()))
            print(json.dumps({'event': 'resumed', 'fold': fold}), flush=True)
            continue
        # fit_transform cross-fits training rows; validation/test see only
        # mappings fitted on the outer training fold, never its held-out labels.
        train_parts, valid_parts, test_parts = [X[ti]], [X[vi]], [Xt]
        encoders = []
        for smoothing in smoothing_values:
            enc = TargetEncoder(target_type='binary', smooth=smoothing, cv=5,
                                shuffle=True, random_state=42)
            train_parts.append(enc.fit_transform(K[ti], y[ti]).astype('float32'))
            valid_parts.append(enc.transform(K[vi]).astype('float32'))
            test_parts.append(enc.transform(Kt).astype('float32'))
            encoders.append(enc)
        A, B, C = (np.concatenate(parts, axis=1) for parts in (train_parts, valid_parts, test_parts))
        del train_parts, valid_parts, test_parts
        print(json.dumps({'event': 'encoded', 'fold': fold, 'seconds': time.time()-fold_start}), flush=True)
        if args.model == 'xgb':
            from xgboost import XGBClassifier
            model = XGBClassifier(device='cuda', tree_method='hist', max_bin=args.max_bin,
                                  n_estimators=args.iterations, max_depth=args.depth,
                                  learning_rate=args.learning_rate, min_child_weight=args.min_child_weight,
                                  subsample=args.subsample, colsample_bytree=args.colsample_bytree,
                                  reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda, gamma=args.gamma,
                                  grow_policy=args.grow_policy,
                                  **({'max_leaves': args.max_leaves} if args.max_leaves else {}),
                                  objective='binary:logistic', eval_metric='auc',
                                  early_stopping_rounds=300, n_jobs=8, random_state=args.model_seed)
            fit_kwargs = {'eval_set': [(B, y[vi])], 'verbose': 500}
            if args.formula_margin:
                fit_kwargs['base_margin'] = margin[ti]
                fit_kwargs['base_margin_eval_set'] = [margin[vi]]
            model.fit(A, y[ti], **fit_kwargs)
            # Predict using CPU to avoid host/device mismatch copies at inference.
            model.set_params(device='cpu')
            best = int(model.best_iteration) + 1
            model.save_model(out / f'model_{fold}.ubj')
        elif args.model == 'lgbm':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(n_estimators=args.iterations, max_depth=args.depth,
                                      num_leaves=2**args.depth, learning_rate=args.learning_rate,
                                      min_child_samples=10, colsample_bytree=0.3,
                                      reg_alpha=0.071, reg_lambda=2.0, max_bin=1024,
                                      metric='auc', n_jobs=8, verbosity=-1, random_state=args.model_seed,
                                      force_col_wise=True)
            model.fit(A, y[ti], eval_set=[(B, y[vi])],
                      callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(500)])
            best = int(model.best_iteration_)
            model.booster_.save_model(str(out / f'model_{fold}.txt'))
        else:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(iterations=args.iterations, depth=args.depth,
                                       learning_rate=args.learning_rate, l2_leaf_reg=5,
                                       task_type='GPU', devices='0', eval_metric='AUC',
                                       loss_function='Logloss', random_seed=args.model_seed,
                                       thread_count=8, allow_writing_files=False, verbose=500)
            model.fit(A, y[ti], eval_set=(B, y[vi]), early_stopping_rounds=300)
            best = int(model.best_iteration_) + 1
            model.save_model(str(out / f'model_{fold}.cbm'))
        if args.formula_margin and args.model == 'xgb':
            vp = model.predict_proba(B, base_margin=margin[vi])[:, 1]
            tp = model.predict_proba(C, base_margin=margin_t)[:, 1]
        else:
            vp, tp = model.predict_proba(B)[:, 1], model.predict_proba(C)[:, 1]
        assert np.isfinite(vp).all() and np.isfinite(tp).all()
        oof[vi], test_pred = vp, test_pred + tp / 5
        record = {'fold': fold, 'auc': float(roc_auc_score(y[vi], vp)),
                  'best_iteration': best, 'seconds': time.time()-fold_start}
        joblib.dump(encoders, out / f'encoders_{fold}.joblib')
        np.savez_compressed(checkpoint, valid_indices=vi, valid_prediction=vp, test_prediction=tp)
        (out / f'fold_{fold}.json').write_text(json.dumps(record, indent=2) + '\n')
        pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).to_csv(out / f'importance_{fold}.csv')
        scores.append(record)
        print(json.dumps(record), flush=True)
        del A, B, C, model, encoders
    assert np.isfinite(oof).all()
    np.save(out / 'oof.npy', oof)
    np.save(out / 'test.npy', test_pred)
    submission = pd.DataFrame({'id': test.id, TARGET: test_pred})
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    assert submission.id.equals(sample.id) and list(submission) == list(sample)
    assert submission[TARGET].between(0, 1).all()
    submission.to_csv(out / 'submission.csv', index=False)
    metrics = {'run': args.run, 'oof_auc': float(roc_auc_score(y, oof)),
               'folds': scores, 'fold_mean': float(np.mean([s['auc'] for s in scores])),
               'fold_std': float(np.std([s['auc'] for s in scores])),
               'elapsed_seconds_this_invocation': time.time()-start,
               'status': 'complete', 'submission_sha256': digest(out / 'submission.csv')}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()

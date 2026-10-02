"""Measure reseed gains with single and bagged inference on excluded holdouts.

This is a mechanism diagnosis on reused competition labels, not new independent
data and not a submission generator. No outer score is evaluated until all
prespecified model fits finish. Outer labels never enter TE, fit, or stopping.
"""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

from train_encoded import ROOT, TARGET, digest, features
from train_xgb_formula_cv import margin_values


def rank(values):
    return rankdata(values, method='average') / len(values)


def log(record):
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    started = time.time()
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    y = tr[TARGET].eq('Yes').to_numpy('int8')
    # Only features of the real test set are used, for the same label-free
    # transductive frequencies as the production model. No test predictions.
    base, keys, _ = features(tr, te, False)
    X, K = base.to_numpy()[:len(tr)], keys.to_numpy()[:len(tr)]
    margin = margin_values(tr, 'probit')
    outer = np.full(len(tr), -1, dtype='int8')
    for f, (_, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=731).split(tr, y)):
        outer[vi] = f
    params = dict(device='cuda', tree_method='hist', max_bin=1024,
        n_estimators=6500, max_depth=5, learning_rate=.03,
        min_child_weight=10., subsample=.9, colsample_bytree=.85,
        reg_alpha=.071, reg_lambda=2., objective='binary:logistic',
        eval_metric='auc', early_stopping_rounds=300, n_jobs=8, random_state=42)
    config = dict(run=args.run, outer_splits=5, outer_seed=731, diagnosed_outer_folds=[0, 1, 2],
        inner_bag_splits=10, bag_fold_seeds=[42, 2026], te_splits=5, te_seed=42,
        smoothing=['auto', 10., 100.], comparison_bag_sizes=[1, 2, 5, 10],
        seed_blend_weight=.5, model_params=params,
        protocol='All 60 fits fixed in advance; outer labels excluded from TE/training/early stopping; no weight search',
        limitation='Reuses competition labels previously explored; 72% train rows per member; three diagnostic outer groups, not full fivefold',
        frequency_scope='unlabeled full train+test',
        code_sha256=digest(Path(__file__)), feature_code_sha256=digest(ROOT / 'train_encoded.py'),
        margin_code_sha256=digest(ROOT / 'train_xgb_formula_cv.py'),
        train_sha256=digest(ROOT / 'train.csv'), test_sha256=digest(ROOT / 'test.csv'))
    cp = out / 'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config, 'Changed diagnostic requires fresh run ID'
        np.testing.assert_array_equal(np.load(out / 'outer_folds.npy'), outer)
    else:
        cp.write_text(json.dumps(config, indent=2) + '\n')
        np.save(out / 'outer_folds.npy', outer)
        (out / 'training_script.py').write_bytes(Path(__file__).read_bytes())
    for f in config['diagnosed_outer_folds']:
        dev, heldout = np.flatnonzero(outer != f), np.flatnonzero(outer == f)
        for seed in config['bag_fold_seeds']:
            bag = StratifiedKFold(10, shuffle=True, random_state=seed)
            for j, (fit_local, stop_local) in enumerate(bag.split(dev, y[dev])):
                stem = f'outer_{f}_seed_{seed}_member_{j}'
                ck = out / f'{stem}.npz'
                fit, stop = dev[fit_local], dev[stop_local]
                assert len(np.unique(np.concatenate([fit, stop, heldout]))) == len(y)
                assert len(fit) + len(stop) + len(heldout) == len(y)
                if ck.exists():
                    with np.load(ck) as z:
                        np.testing.assert_array_equal(z['fit_indices'], fit)
                        np.testing.assert_array_equal(z['stop_indices'], stop)
                        np.testing.assert_array_equal(z['heldout_indices'], heldout)
                    log(dict(event='resumed', outer=f, seed=seed, member=j))
                    continue
                t0 = time.time()
                parts = [[X[fit]], [X[stop]], [X[heldout]]]
                encoders = []
                for smooth in config['smoothing']:
                    enc = TargetEncoder(target_type='binary', smooth=smooth,
                                        cv=5, shuffle=True, random_state=42)
                    parts[0].append(enc.fit_transform(K[fit], y[fit]).astype('float32'))
                    parts[1].append(enc.transform(K[stop]).astype('float32'))
                    parts[2].append(enc.transform(K[heldout]).astype('float32'))
                    encoders.append(enc)
                A, B, C = [np.column_stack(x) for x in parts]
                model = XGBClassifier(**params)
                model.fit(A, y[fit], base_margin=margin[fit], eval_set=[(B, y[stop])],
                          base_margin_eval_set=[margin[stop]], verbose=False)
                model.set_params(device='cpu')
                pred = model.predict_proba(C, base_margin=margin[heldout])[:, 1]
                assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
                model_path = out / f'{stem}.ubj'
                encoder_path = out / f'{stem}.joblib'
                model.save_model(model_path)
                joblib.dump(encoders, encoder_path)
                # A bounded reconstruction check, including the saved encoders.
                picks = np.linspace(0, len(heldout) - 1, 512, dtype=int)
                enc2 = joblib.load(encoder_path)
                rebuilt = np.column_stack([X[heldout[picks]]] +
                    [e.transform(K[heldout[picks]]).astype('float32') for e in enc2])
                reload_model = XGBClassifier()
                reload_model.load_model(model_path)
                reload_model.set_params(device='cpu', n_jobs=8)
                replay = reload_model.predict_proba(rebuilt, base_margin=margin[heldout[picks]])[:, 1]
                np.testing.assert_allclose(replay, pred[picks], rtol=1e-6, atol=1e-7)
                np.savez_compressed(ck, fit_indices=fit, stop_indices=stop,
                    heldout_indices=heldout, heldout_prediction=pred)
                rec = dict(outer=f, seed=seed, member=j, fit_rows=len(fit), stop_rows=len(stop),
                    heldout_rows=len(heldout), inner_stop_auc=float(model.best_score),
                    best_iteration=int(model.best_iteration) + 1,
                    reload_max_error=float(np.max(np.abs(replay - pred[picks]))),
                    checkpoint_sha256=digest(ck), model_sha256=digest(model_path),
                    encoder_sha256=digest(encoder_path), seconds=time.time() - t0)
                (out / f'{stem}.json').write_text(json.dumps(rec, indent=2) + '\n')
                log(dict(event='fit_completed', **rec))
                del A, B, C, parts, encoders, enc2, model, reload_model, rebuilt
                gc.collect()
    # Scoring starts only after every planned fit is available.
    records, contributions, spliced_records = [], [], []
    for f in config['diagnosed_outer_folds']:
        heldout = np.flatnonzero(outer == f)
        labels = y[heldout]
        bags = []
        for seed in config['bag_fold_seeds']:
            bags.append(np.stack([np.load(out / f'outer_{f}_seed_{seed}_member_{j}.npz')
                                  ['heldout_prediction'] for j in range(10)]).astype('float64'))
        # Prefixes are deterministic and fixed; 1-member averages enumerate all
        # ten matched member indices to avoid selecting a lucky single member.
        for size in config['comparison_bag_sizes']:
            gains = []
            for rotation in (range(10) if size < 10 else [0]):
                members = (np.arange(size) + rotation) % 10
                p, q = [bag[members].mean(axis=0) for bag in bags]
                r = .5 * rank(p) + .5 * rank(q)
                a, b, c = [float(roc_auc_score(labels, v)) for v in (p, q, r)]
                rec = dict(outer=f, bag_size=size, rotation=rotation,
                    seed42_auc=a, seed2026_auc=b, blend_auc=c,
                    blend_gain_over_seed42=c - a, blend_gain_over_mean_single_seed=c - (a + b) / 2,
                    seed_rank_correlation=float(np.corrcoef(rank(p), rank(q))[0, 1]))
                records.append(rec)
                gains.append(rec)
            log(dict(event='holdout_scored', outer=f, bag_size=size,
                seed42_auc_mean=float(np.mean([r['seed42_auc'] for r in gains])),
                seed2026_auc_mean=float(np.mean([r['seed2026_auc'] for r in gains])),
                blend_auc_mean=float(np.mean([r['blend_auc'] for r in gains])),
                mean_gain_over_seed42=float(np.mean([r['blend_gain_over_seed42'] for r in gains]))))
        rng = np.random.default_rng(7733 + f)
        spliced = [bag[rng.integers(10, size=len(labels)), np.arange(len(labels))] for bag in bags]
        a, b = [float(roc_auc_score(labels, v)) for v in spliced]
        c = float(roc_auc_score(labels, .5 * rank(spliced[0]) + .5 * rank(spliced[1])))
        spliced_records.append(dict(outer=f, seed42_auc=a, seed2026_auc=b, blend_auc=c,
            blend_gain_over_seed42=c - a, blend_gain_over_mean_single_seed=c - (a + b) / 2))
    table = pd.DataFrame(records)
    table.to_csv(out / 'comparison.csv', index=False)
    summary = table.groupby('bag_size')[['seed42_auc', 'seed2026_auc', 'blend_auc',
        'blend_gain_over_seed42', 'blend_gain_over_mean_single_seed', 'seed_rank_correlation']].mean()
    report = dict(run=args.run, status='diagnosis_complete', submitted=False,
        n_models=60, diagnosed_rows=int(np.isin(outer, [0, 1, 2]).sum()),
        elapsed_seconds=time.time() - started, comparison=records,
        mean_across_outer_groups=summary.reset_index().to_dict('records'),
        spliced_single_member_diagnostic=spliced_records,
        uncertainty_note='Three fixed diagnostic holdouts on previously explored labels; 72% fit size differs from production 90%; no claim of independent Public validation',
        config_sha256=digest(cp), code_sha256=digest(Path(__file__)))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    log(dict(event='diagnosis_complete', summary=report['mean_across_outer_groups'],
             spliced=spliced_records, elapsed_seconds=report['elapsed_seconds']))


if __name__ == '__main__':
    main()

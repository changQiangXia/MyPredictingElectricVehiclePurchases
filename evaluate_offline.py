"""Compare local OOF candidates without any Kaggle API or submission calls."""
import os

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from train_encoded import ROOT, TARGET, digest


def auc_placements(y, predictions):
    positive, negative = predictions[y], predictions[~y]
    pos_sorted, neg_sorted = np.sort(positive), np.sort(negative)
    positive_placement = (np.searchsorted(neg_sorted, positive, side='left')
                          + np.searchsorted(neg_sorted, positive, side='right')) / (2.0 * len(negative))
    negative_placement = 1 - (np.searchsorted(pos_sorted, negative, side='left')
                              + np.searchsorted(pos_sorted, negative, side='right')) / (2.0 * len(positive))
    assert abs(positive_placement.mean() - roc_auc_score(y, predictions)) < 1e-10
    return positive_placement, negative_placement


def paired_delta(y, baseline, prediction, baseline_placements):
    a, b = auc_placements(y, prediction)
    da, db = a - baseline_placements[0], b - baseline_placements[1]
    se = float(np.sqrt(np.var(da, ddof=1) / len(da) + np.var(db, ddof=1) / len(db)))
    delta = float(roc_auc_score(y, prediction) - roc_auc_score(y, baseline))
    return {'delta': delta, 'paired_se': se, 'conditional_ci95': [delta - 1.96 * se, delta + 1.96 * se]}


def edges(frame, values):
    values = values.copy()
    values[frame.Annual_Income_USD.ge(170537).to_numpy()] = 1
    values[(frame.Annual_Income_USD.between(31004, 41970) | frame.Daily_Commute_km.ge(83)).to_numpy()] = 0
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--anchor', default='public01_ours10_rank_edges_v1')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--weights', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument('--candidate-model')
    parser.add_argument('--candidate-weight', type=float)
    args = parser.parse_args()
    assert all(0 <= w <= 1 for w in args.weights)
    assert (args.candidate_model is None) == (args.candidate_weight is None)
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    y = tr[TARGET].eq('Yes').to_numpy()
    folds = np.load(ROOT / 'artifacts' / 'folds_seed42.npy')
    anchor_path = ROOT / 'artifacts' / args.anchor
    anchor = np.load(anchor_path / 'oof.npy')
    anchor_test = np.load(anchor_path / 'test.npy')
    assert len(anchor) == len(y) and len(anchor_test) == len(te)
    assert np.isfinite(anchor).all() and np.isfinite(anchor_test).all()
    anchor_auc = float(roc_auc_score(y, anchor))
    anchor_rank, anchor_test_rank = rankdata(anchor) / len(anchor), rankdata(anchor_test) / len(anchor_test)
    placements = auc_placements(y, anchor)
    anchor_folds = [float(roc_auc_score(y[folds == f], anchor[folds == f])) for f in range(5)]
    results = []
    sources = {args.anchor: {'oof_sha256': digest(anchor_path / 'oof.npy'),
                            'test_sha256': digest(anchor_path / 'test.npy')}}
    for run in args.models:
        path = ROOT / 'artifacts' / run
        prediction, test_prediction = np.load(path / 'oof.npy'), np.load(path / 'test.npy')
        assert prediction.shape == y.shape and test_prediction.shape == anchor_test.shape
        assert np.isfinite(prediction).all() and np.isfinite(test_prediction).all()
        rank, test_rank = rankdata(prediction) / len(prediction), rankdata(test_prediction) / len(test_prediction)
        sources[run] = {'oof_sha256': digest(path / 'oof.npy'), 'test_sha256': digest(path / 'test.npy')}
        single = {'run': run, 'kind': 'single', 'auc': float(roc_auc_score(y, prediction)),
                  'spearman_with_anchor': float(np.corrcoef(rank, anchor_rank)[0, 1])}
        results.append(single)
        print(json.dumps(single), flush=True)
        for w in args.weights:
            blend = edges(tr, (1 - w) * anchor_rank + w * rank)
            auc = float(roc_auc_score(y, blend))
            fold_delta = [float(roc_auc_score(y[folds == f], blend[folds == f]) - anchor_folds[f]) for f in range(5)]
            rec = {'run': run, 'kind': 'rank_blend', 'weight': w, 'auc': auc,
                   **paired_delta(y, anchor, blend, placements), 'fold_deltas': fold_delta,
                   'positive_folds': int(np.sum(np.array(fold_delta) > 0))}
            results.append(rec)
            print(json.dumps(rec), flush=True)
            if args.candidate_model == run and args.candidate_weight == w:
                pred = edges(te, (1 - w) * anchor_test_rank + w * test_rank)
                assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
                np.save(out / 'oof.npy', blend)
                np.save(out / 'test.npy', pred)
                sub = pd.read_csv(ROOT / 'sample_submission.csv')
                assert sub.id.equals(te.id) and sub.id.is_unique
                sub[TARGET] = pred
                sub.to_csv(out / 'submission.csv', index=False)
                metrics = {'run': args.run, 'oof_auc': auc, 'status': 'local_candidate',
                           'submitted': False, 'public_auc': None, 'anchor': args.anchor,
                           'added_model': run, 'weight': w,
                           'selection_note': 'Exploratory selection on reused OOF, not an untouched validation set',
                           'submission_sha256': digest(out / 'submission.csv'), **rec}
                metrics['run'] = args.run
                (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    report = {'run': args.run, 'anchor': args.anchor, 'anchor_auc': anchor_auc,
              'anchor_fold_aucs': anchor_folds, 'results': results, 'sources': sources,
              'train_sha256': digest(ROOT / 'train.csv'), 'test_sha256': digest(ROOT / 'test.csv'),
              'code_sha256': digest(Path(__file__)), 'submitted': False,
              'uncertainty_note': 'Paired DeLong CI conditional on fixed OOF predictions; excludes retraining, model/weight selection and shared-fold dependence.',
              'public_oof_note': 'External OOF generation is not fully reproducible from the available artifacts; the public anchor is an exploratory comparator.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.DataFrame(results).to_csv(out / 'comparison.csv', index=False)


if __name__ == '__main__':
    main()

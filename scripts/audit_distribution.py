"""Target-free train/test drift check and local error slices; never submits."""
import os

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train_encoded import ROOT, TARGET, digest


def main():
    out = ROOT / 'artifacts' / 'offline_wave2_distribution'
    out.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    cols = [c for c in test if c != 'id']
    raw = pd.concat([train[cols], test[cols]], ignore_index=True)
    numeric = raw.select_dtypes('number').columns
    distributions = []
    for c in cols:
        if c in numeric:
            distributions.append({'feature': c, 'type': 'numeric',
                                  'ks_statistic': float(ks_2samp(train[c], test[c]).statistic),
                                  'standardized_mean_delta': float((test[c].mean() - train[c].mean()) / train[c].std())})
        else:
            categories = sorted(raw[c].unique())
            a = train[c].value_counts(normalize=True).reindex(categories, fill_value=0)
            b = test[c].value_counts(normalize=True).reindex(categories, fill_value=0)
            distributions.append({'feature': c, 'type': 'categorical',
                                  'total_variation': float(np.abs(a - b).sum() / 2)})
            raw[c] = raw[c].map({v: i for i, v in enumerate(categories)}).astype('int32')
    source = np.r_[np.zeros(len(train), dtype='int8'), np.ones(len(test), dtype='int8')]
    development, holdout = train_test_split(np.arange(len(raw)), test_size=0.25, random_state=20260909, stratify=source)
    fit_indices, stop_indices = train_test_split(development, test_size=0.2, random_state=42, stratify=source[development])
    model = lgb.LGBMClassifier(n_estimators=500, num_leaves=15, learning_rate=0.05,
                              min_child_samples=200, reg_lambda=10, colsample_bytree=0.8,
                              random_state=42, n_jobs=4, verbosity=-1)
    model.fit(raw.iloc[fit_indices], source[fit_indices], eval_set=[(raw.iloc[stop_indices], source[stop_indices])],
              eval_metric='auc', callbacks=[lgb.early_stopping(50, verbose=False)])
    prediction = model.predict_proba(raw.iloc[holdout])[:, 1]
    auc = float(roc_auc_score(source[holdout], prediction))
    model.booster_.save_model(str(out / 'adversarial_lgbm.txt'))
    np.savez_compressed(out / 'adversarial_holdout.npz', indices=holdout, source=source[holdout], prediction=prediction)
    report = {'adversarial_holdout_auc': auc, 'holdout_rows': len(holdout),
              'early_stopping_rows': len(stop_indices), 'best_iteration': model.best_iteration_,
              'excluded_columns': ['id', TARGET], 'features': cols,
              'fit_seed': 42, 'holdout_seed': 20260909,
              'distribution_statistics': distributions, 'submitted': False,
              'train_sha256': digest(ROOT / 'train.csv'), 'test_sha256': digest(ROOT / 'test.csv'),
              'code_sha256': digest(Path(__file__)),
              'interpretation': 'AUC near 0.5 gives no evidence for useful covariate-shift weighting at this model capacity.'}
    pd.DataFrame(distributions).to_csv(out / 'feature_drift.csv', index=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    y = train[TARGET].eq('Yes').to_numpy()
    base = np.load(ROOT / 'artifacts' / 'public01_ours10_rank_edges_v1' / 'oof.npy')
    candidate = np.load(ROOT / 'artifacts' / 'offline_wave2_initial' / 'oof.npy')
    slices = {'income': pd.cut(train.Annual_Income_USD, [0, 30000, 50000, 75000, 100000, 150000, np.inf], include_lowest=True).astype(str),
              'environmental_concern': train.Environmental_Concern_Level.astype(str),
              'anxiety': train.Range_Anxiety_Level, 'city': train.City_Type}
    records = []
    for name, groups in slices.items():
        for group in sorted(groups.unique()):
            mask = groups.eq(group).to_numpy()
            if len(np.unique(y[mask])) < 2:
                continue
            baseline_auc = float(roc_auc_score(y[mask], base[mask]))
            candidate_auc = float(roc_auc_score(y[mask], candidate[mask]))
            records.append({'slice': name, 'group': group, 'rows': int(mask.sum()),
                            'positive_rows': int(y[mask].sum()), 'baseline_auc': baseline_auc,
                            'candidate_auc': candidate_auc, 'delta': candidate_auc - baseline_auc})
    pd.DataFrame(records).to_csv(out / 'error_slices.csv', index=False)
    print(json.dumps({'adversarial_holdout_auc': auc, 'best_iteration': model.best_iteration_,
                      'max_numeric_ks': max(d.get('ks_statistic', 0) for d in distributions),
                      'slices': records}), flush=True)


if __name__ == '__main__':
    main()

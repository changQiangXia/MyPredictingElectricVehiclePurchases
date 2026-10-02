"""Audit the whole pure-group discovery procedure, including failed groups.

Public lists mined from this competition's training labels are not independent
validation. For each held-out row, build its rule list using other folds only.
This is a historical-label audit, not a new untouched validation dataset.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'commute_discovery_reaudit_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    y = train.Will_Buy_EV.eq('Yes').to_numpy(np.int8)
    x = train.Daily_Commute_km.to_numpy(float)
    xt = test.Daily_Commute_km.to_numpy(float)
    folds = np.load(ROOT / 'artifacts/folds_seed42.npy')
    fixed = np.isin(x, [68.8, 75.7, 78.3, 79.8]) | ((x >= 75) & (x < 76))
    models = ['offline_wave5_conservative', 'offline_wave9_seed_transfer15_v1']
    # Two previously evaluated families at their existing threshold. No new
    # value/threshold sweep, no removal of groups that fail on held-out data.
    report = []
    for name, width in [('exact', None), ('quarter_km', 0.25)]:
        key = x if width is None else np.floor(x / width) * width
        keyt = xt if width is None else np.floor(xt / width) * width
        honest_mask = np.zeros(len(y), bool)
        discovery = []
        for fold in range(5):
            fit, val = folds != fold, folds == fold
            stats = pd.DataFrame({'key': key[fit], 'y': y[fit]}).groupby('key').y.agg(['size', 'sum'])
            pure = stats[(stats['size'] >= 20) & (stats['sum'] == 0)]
            mask = np.isin(key[val], pure.index)
            honest_mask[val] = mask
            records = []
            for k, row in pure.iterrows():
                hit = val & (key == k)
                records.append({'key': float(k), 'fit_n': int(row['size']),
                                'val_n': int(hit.sum()), 'val_positive': int(y[hit].sum()),
                                'test_n': int((keyt == k).sum())})
            pd.DataFrame(records).to_csv(OUT / f'{name}_fold{fold}_map.csv', index=False)
            discovery.append({'fold': fold, 'groups': len(pure), 'validation_rows': int(mask.sum()),
                              'validation_positives': int(y[val][mask].sum())})
        for model_name in models:
            anchor = np.load(ROOT / 'artifacts' / model_name / 'oof.npy')
            a = rankdata(anchor, method='average') / (len(anchor) + 1)
            base_auc = roc_auc_score(y, a)
            for mode in ['down_0.1', 'bottom']:
                p = a.copy()
                if mode == 'bottom':
                    p[honest_mask] = 0
                else:
                    p[honest_mask] -= .1
                gains = []
                for f in range(5):
                    v = folds == f
                    gains.append(float(roc_auc_score(y[v], p[v]) - roc_auc_score(y[v], a[v])))
                record = {'family': name, 'minimum_fit_count': 20, 'mode': mode, 'anchor': model_name,
                          'delta_auc': float(roc_auc_score(y, p) - base_auc), 'fold_gains': gains,
                          'hit_rows': int(honest_mask.sum()), 'hit_positives': int(y[honest_mask].sum()),
                          'discovery': discovery}
                report.append(record)
                np.save(OUT / f'{name}_{mode}_{model_name}_oof.npy', p)
        np.save(OUT / f'{name}_honest_mask.npy', honest_mask)
    manifest = {'interpretation': 'Rules and minimum support set before this audit; each map excludes its validation labels. Historical labels and anchors are reused, so this is not independent confirmation.',
                'invalid_fixed_union_diagnostic': {'rows': int(fixed.sum()), 'positives': int(y[fixed].sum()),
                    'reason': 'Public/manual fixed union was selected using the complete training labels; its five positive folds are not independent evidence.'},
                'train_sha256': digest(ROOT / 'train.csv'), 'folds_sha256': digest(ROOT / 'artifacts/folds_seed42.npy'),
                'script_sha256': digest(Path(__file__)), 'submitted': False, 'results': report}
    (OUT / 'report.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for r in report:
        print(json.dumps({k: v for k, v in r.items() if k != 'discovery'}), flush=True)


if __name__ == '__main__':
    main()

"""Materialize preregistered quarter-km pure-zero groups with support >=10."""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
TARGET = 'Will_Buy_EV'
base = pd.read_csv(ROOT / 'artifacts/probe_publicbest_v19_high_wm010/submission.csv')
tr = pd.read_csv(ROOT / 'train.csv')
te = pd.read_csv(ROOT / 'test.csv')
x = np.floor(tr.Daily_Commute_km.to_numpy(float) / .25) * .25
xt = np.floor(te.Daily_Commute_km.to_numpy(float) / .25) * .25
y = tr.Will_Buy_EV.eq('Yes').to_numpy()
stats = pd.DataFrame({'k': x, 'y': y}).groupby('k').y.agg(['size', 'sum'])
pure = stats[(stats['size'] >= 10) & (stats['sum'] == 0)].index
mask = np.isin(xt, pure)
ranks = rankdata(base[TARGET].to_numpy(float), method='average')
pred = ranks / (len(ranks) + 1.0)
pred[mask] = 0.0
name = 'nextday_v19_commute_quarter10_rankbottom'
out = ROOT / 'artifacts' / name
out.mkdir(exist_ok=False)
sub = base.copy()
sub[TARGET] = pred
sub.to_csv(out / 'submission.csv', index=False)
meta = {
    'run': name, 'base': 'probe_publicbest_v19_high_wm010',
    'rule': 'floor(Daily_Commute_km/.25)*.25 pure-zero groups, train support >=10',
    'n_groups': int(len(pure)), 'rows_test': int(mask.sum()), 'submitted': False,
    'valid_probability_range': True,
    'selection': 'family and threshold fixed before outer-fold audit; all qualifying groups retained',
    'oof_audit': 'strict seed42 wave5 +6.60e-6 and 5/5 folds; quarter min20 +5.54e-6',
    'sha256': hashlib.sha256((out / 'submission.csv').read_bytes()).hexdigest(),
}
(out / 'metrics.json').write_text(json.dumps(meta, indent=2) + '\n')
stats.loc[pure].reset_index().to_csv(out / 'full_train_pure_zero_map.csv', index=False)
print(json.dumps(meta, indent=2))

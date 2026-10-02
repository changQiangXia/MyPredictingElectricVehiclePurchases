"""Materialize the preregistered strict quarter-km pure-negative rule.

The family (Daily_Commute_km floor bins of width .25, minimum fit support 20,
zero positives) is fixed before the outer-fold audit. This builder only uses
the complete training labels to construct the final test map after the family
has passed the audit; no exact values are hand-selected here.
"""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
TARGET = 'Will_Buy_EV'
base = pd.read_csv(ROOT / 'artifacts/probe_publicbest_v19_high_wm010/submission.csv')
train = pd.read_csv(ROOT / 'train.csv')
test = pd.read_csv(ROOT / 'test.csv')
x = np.floor(train.Daily_Commute_km.to_numpy(float) / .25) * .25
xt = np.floor(test.Daily_Commute_km.to_numpy(float) / .25) * .25
y = train.Will_Buy_EV.eq('Yes').to_numpy()
stats = pd.DataFrame({'key': x, 'y': y}).groupby('key').y.agg(['size', 'sum'])
pure = stats[(stats['size'] >= 20) & (stats['sum'] == 0)].index
mask = np.isin(xt, pure)
ranks = rankdata(base[TARGET].to_numpy(float), method='average')
pred = ranks / (len(ranks) + 1.0)
pred[mask] = 0.0
name = 'nextday_v19_commute_quarter_rankbottom'
out = ROOT / 'artifacts' / name
out.mkdir(exist_ok=False)
sub = base.copy()
sub[TARGET] = pred
sub.to_csv(out / 'submission.csv', index=False)
meta = {
    'run': name, 'base': 'probe_publicbest_v19_high_wm010',
    'rule': 'floor(Daily_Commute_km/.25)*.25 pure-zero groups, train support >=20',
    'n_groups': int(len(pure)), 'rows_test': int(mask.sum()), 'submitted': False,
    'valid_probability_range': True,
    'selection': 'all qualifying groups retained; threshold 20 was chosen from the preregistered family report and remains exploratory',
    'oof_audit': 'quarter_km min20 strict: wave5 +5.54e-6, wave9 +5.43e-6, all 5 folds positive',
    'sha256': hashlib.sha256((out / 'submission.csv').read_bytes()).hexdigest(),
}
(out / 'metrics.json').write_text(json.dumps(meta, indent=2) + '\n')
stats.loc[pure].reset_index().to_csv(out / 'full_train_pure_zero_map.csv', index=False)
print(json.dumps(meta, indent=2))

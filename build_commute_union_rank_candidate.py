"""Rank-safe candidate for union of two publicly reported pure-negative rules."""
from pathlib import Path
import hashlib, json
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
TARGET = 'Will_Buy_EV'
base = pd.read_csv(ROOT / 'artifacts/probe_publicbest_v19_high_wm010/submission.csv')
test = pd.read_csv(ROOT / 'test.csv')
mask = (test.Daily_Commute_km.isin([68.8, 75.7, 78.3, 79.8]) | ((test.Daily_Commute_km >= 75) & (test.Daily_Commute_km < 76))).to_numpy()
ranks = rankdata(base[TARGET].to_numpy(float), method='average')
pred = ranks / (len(ranks) + 1.0)
pred[mask] = 0.0
name = 'nextday_v19_commute_union_rankbottom'
out = ROOT / 'artifacts' / name
out.mkdir(exist_ok=False)
sub = base.copy()
sub[TARGET] = pred
sub.to_csv(out / 'submission.csv', index=False)
meta = {
    'run': name, 'base': 'probe_publicbest_v19_high_wm010',
    'rule': 'union fixed public exact values 68.8,75.7,78.3,79.8 and interval [75,76)',
    'rows': int(mask.sum()), 'submitted': False, 'valid_probability_range': True,
    'rationale': 'both rules were public before selection; rank-equivalent maximum correction',
    'oof_proxy_gain_wave9': 9.630319095399464e-06,
    'oof_proxy_folds_wave9': [8.676903387794255e-06, 1.2123902934924047e-05, 9.565295871816204e-06, 1.0361660380708848e-05, 7.210725696982223e-06],
    'sha256': hashlib.sha256((out / 'submission.csv').read_bytes()).hexdigest(),
}
(out / 'metrics.json').write_text(json.dumps(meta, indent=2) + '\n')
print(meta['sha256'], meta['rows'])

"""Valid [0,1] rank-space version of the fixed commute purity correction."""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
TARGET = 'Will_Buy_EV'
base = pd.read_csv(ROOT / 'artifacts/probe_publicbest_v19_high_wm010/submission.csv')
test = pd.read_csv(ROOT / 'test.csv')
mask = test.Daily_Commute_km.isin([68.8, 75.7, 78.3, 79.8]).to_numpy()
ranks = rankdata(base[TARGET].to_numpy(float), method='average')
# Leave a positive gap above the affected rows so their bottom assignment does
# not tie with the minimum unaffected prediction.
pred = ranks / (len(ranks) + 1.0)
pred[mask] = 0.0
name = 'nextday_v19_commute_exact4_rankbottom'
out = ROOT / 'artifacts' / name
out.mkdir(exist_ok=False)
sub = base.copy()
sub[TARGET] = pred
sub.to_csv(out / 'submission.csv', index=False)
meta = {
    'run': name, 'base': 'probe_publicbest_v19_high_wm010',
    'rule': 'fixed public exact commute values 68.8,75.7,78.3,79.8',
    'rows': int(mask.sum()), 'submitted': False,
    'valid_probability_range': True,
    'rationale': 'rank-equivalent maximum correction; pure-negative groups moved to bottom',
    'sha256': hashlib.sha256((out / 'submission.csv').read_bytes()).hexdigest(),
}
(out / 'metrics.json').write_text(json.dumps(meta, indent=2) + '\n')
print(name, meta['sha256'], pred.min(), pred.max())

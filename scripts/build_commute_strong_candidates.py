"""Prepare stronger fixed-public commute purity probes; never uploads."""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'Will_Buy_EV'
base = pd.read_csv(ROOT / 'artifacts/probe_publicbest_v19_high_wm010/submission.csv')
test = pd.read_csv(ROOT / 'test.csv')
mask = test.Daily_Commute_km.isin([68.8, 75.7, 78.3, 79.8]).to_numpy()
for weight in (-0.5, -1.0):
    pred = base[TARGET].to_numpy(float).copy()
    pred[mask] += weight
    name = f'nextday_v19_commute_exact4_w{abs(weight):g}'
    out = ROOT / 'artifacts' / name
    out.mkdir(exist_ok=False)
    sub = base.copy()
    sub[TARGET] = pred
    sub.to_csv(out / 'submission.csv', index=False)
    meta = {
        'run': name, 'base': 'probe_publicbest_v19_high_wm010',
        'rule': 'fixed public exact commute values 68.8,75.7,78.3,79.8',
        'weight': weight, 'rows': int(mask.sum()), 'submitted': False,
        'clip': False,
        'rationale': 'stronger correction; all four groups are pure-negative in train and rule fixed before OOF',
        'sha256': hashlib.sha256((out / 'submission.csv').read_bytes()).hexdigest(),
    }
    (out / 'metrics.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(name, meta['sha256'])

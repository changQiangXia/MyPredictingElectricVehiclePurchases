"""Ablate conditional target-encoding keys using the unchanged training harness."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import train_encoded as runner

base_features = runner.features
PAIRS = [
    ('Annual_Income_USD', 'Subsidy_Available'),
    ('Annual_Income_USD', 'Environmental_Concern_Level'),
    ('Annual_Income_USD', 'Range_Anxiety_Level'),
    ('Daily_Commute_km', 'Range_Anxiety_Level'),
    ('Charging_Stations_Near_Home', 'Home_Charging_Possible'),
    ('City_Type', 'Home_Charging_Possible'),
]


def conditional_features(train, test, original):
    base, keys, mappings = base_features(train, test, original)
    for a, b in PAIRS:
        pairs = pd.MultiIndex.from_arrays([keys[a], keys[b]])
        keys[f'{a}_BY_{b}'] = pd.factorize(pairs, sort=True)[0].astype('float64')
    return base, keys, mappings


if __name__ == '__main__':
    run = sys.argv[sys.argv.index('--run') + 1]
    out = runner.ROOT / 'artifacts' / run
    out.mkdir(parents=True, exist_ok=True)
    record = {'pairs': PAIRS, 'addon_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'description': 'Conditional keys encoded only with nested outer/inner training labels'}
    path = out / 'conditional_config.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(record))
    else:
        path.write_text(json.dumps(record, indent=2) + '\n')
    runner.features = conditional_features
    runner.main()

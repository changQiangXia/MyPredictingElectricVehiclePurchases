"""Add target-free train/test-versus-original distribution lift features."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import train_encoded as runner

ORIG = Path(runner.ROOT) / 'original'
raw_features = runner.features


def features(train, test, original):
    base, keys, mappings = raw_features(train, test, original)
    orig_path = next(ORIG.glob('*.csv'))
    orig_df = pd.read_csv(orig_path)
    cols = [c for c in test.columns if c != 'id']
    combined = pd.concat([train[cols], test[cols]], ignore_index=True)
    for c in cols:
        all_counts = combined[c].value_counts(normalize=True, dropna=False)
        orig_counts = orig_df[c].value_counts(normalize=True, dropna=False)
        all_p = combined[c].map(all_counts).fillna(0).to_numpy(dtype='float64')
        orig_p = combined[c].map(orig_counts).fillna(0).to_numpy(dtype='float64')
        # Log lift is target-free; +1e-7 handles values absent from original.
        base[f'{c}_orig_lift'] = np.log((all_p + 1e-7) / (orig_p + 1e-7)).astype('float32')
        base[f'{c}_orig_novel'] = (orig_p == 0).astype('float32')
    return base, keys, mappings


if __name__ == '__main__':
    run = sys.argv[sys.argv.index('--run') + 1]
    cfg = {'run': run, 'features': 'target-free normalized frequency lift versus public original',
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = runner.ROOT / 'artifacts' / run; out.mkdir(parents=True, exist_ok=True)
    (out / 'feature_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    runner.features = features
    runner.main()

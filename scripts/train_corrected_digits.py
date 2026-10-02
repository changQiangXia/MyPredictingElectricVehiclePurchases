"""Evaluate corrected decimal digits after the 740824 public discussion.

The canonical runner deliberately retains float floor residues. This view adds
integer-scaled digits for numeric columns, especially Daily_Commute_km, without
removing the canonical features. Target encodings remain nested in the runner.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
import train_encoded as runner

RAW = runner.features

def features(train, test, original):
    base, keys, mappings = RAW(train, test, original)
    all_rows = pd.concat([train.drop(columns=[runner.TARGET]), test], ignore_index=True)
    for c in all_rows.select_dtypes('number').columns:
        if c == 'id':
            continue
        vals = all_rows[c].to_numpy(float)
        # Values in this competition have at most four decimal places. Round
        # before integer conversion to avoid binary representation artefacts.
        scaled = np.rint(vals * 10**4).astype(np.int64)
        for k in range(-4, 4):
            d = (scaled // (10 ** (k + 4)) % 10).astype('float32')
            name = f'{c}_correct_digit_{k}'
            base[name] = d
            keys[name] = d
    return base, keys, mappings

if __name__ == '__main__':
    run = sys.argv[sys.argv.index('--run') + 1]
    out = runner.ROOT / 'artifacts' / run
    out.mkdir(parents=True, exist_ok=False)
    (out / 'feature_config.json').write_text(json.dumps({
        'run': run, 'hypothesis': 'integer-scaled decimal digits avoid float // artefact',
        'discussion': 740824, 'scale': 10000,
        'label_boundary': 'digits label-free; canonical TE outer/inner cross-fit',
        'submitted': False,
    }, indent=2) + '\n')
    runner.features = features
    runner.main()

"""Ablation: add only corrected integer-scaled commute digits."""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
import train_encoded as runner
RAW = runner.features
def features(train, test, original):
    base, keys, mappings = RAW(train, test, original)
    all_rows = pd.concat([train.drop(columns=[runner.TARGET]), test], ignore_index=True)
    vals = all_rows.Daily_Commute_km.to_numpy(float)
    scaled = np.rint(vals * 10**4).astype(np.int64)
    for k in range(-4, 4):
        d = (scaled // (10 ** (k + 4)) % 10).astype('float32')
        name = f'Daily_Commute_km_correct_digit_{k}'
        base[name] = d; keys[name] = d
    return base, keys, mappings
if __name__ == '__main__':
    run = sys.argv[sys.argv.index('--run') + 1]; out = runner.ROOT/'artifacts'/run; out.mkdir(parents=True, exist_ok=False)
    (out/'feature_config.json').write_text(json.dumps({'run':run,'hypothesis':'only corrected integer-scaled commute digits','discussion':740824,'scale':10000,'submitted':False},indent=2)+'\n')
    runner.features=features; runner.main()

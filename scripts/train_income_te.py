"""Ablation: retain only high-signal target-encoding keys."""
import hashlib
import json
import sys
from pathlib import Path
import train_encoded as runner

raw_features = runner.features
KEEP = {'Annual_Income_USD', 'income_floor_100', 'income_floor_1000',
        'Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level'}


def features(train, test, original):
    base, keys, mappings = raw_features(train, test, original)
    keys = keys[[c for c in keys.columns if c in KEEP]]
    return base, keys, mappings


if __name__ == '__main__':
    run = sys.argv[sys.argv.index('--run') + 1]
    out = runner.ROOT / 'artifacts' / run; out.mkdir(parents=True, exist_ok=True)
    cfg = {'keep_target_encoding_keys': sorted(KEEP),
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'feature_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    runner.features = features
    runner.main()

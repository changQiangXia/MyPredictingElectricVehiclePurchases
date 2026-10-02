"""Add the public reverse-engineered EV purchase score as a regular feature."""
import hashlib
import json
import sys
from pathlib import Path
import train_encoded as runner

raw_features = runner.features

def features(train, test, original):
    base, keys, mappings = raw_features(train, test, original)
    both = __import__('pandas').concat([train, test], ignore_index=True)
    range_map = {'Low': 0.0, 'Medium': -1.0, 'High': -3.0}
    score = (1.2 * both['Annual_Income_USD'] / 1e5
             + 0.6 * both['Environmental_Concern_Level']
             + 2.0 * both['Subsidy_Available'].eq('Yes').astype(float)
             + both['Range_Anxiety_Level'].map(range_map).fillna(-1.0))
    base['recovered_formula_score'] = score.to_numpy(dtype='float32')
    base['recovered_formula_logit'] = (score - 5.5).to_numpy(dtype='float32')
    return base, keys, mappings

if __name__ == '__main__':
    run = sys.argv[sys.argv.index('--run') + 1]
    out = runner.ROOT / 'artifacts' / run; out.mkdir(parents=True, exist_ok=True)
    (out / 'feature_config.json').write_text(json.dumps({
        'formula': '1.2 income/1e5 + 0.6 concern + 2 subsidy - range penalty',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    runner.features = features
    runner.main()

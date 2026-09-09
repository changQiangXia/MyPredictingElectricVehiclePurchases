"""Target-encode one conditional income key while retaining the validated keys."""
import hashlib, json, sys
from pathlib import Path
import pandas as pd
import train_encoded as runner

raw_features = runner.features

def features(train, test, original):
    base, keys, mappings = raw_features(train, test, original)
    all_rows = pd.concat([train, test], ignore_index=True)
    # Numeric code is only a key identifier; TargetEncoder receives it as a category.
    keys['income_by_subsidy'] = (all_rows['Annual_Income_USD'].astype(str) + '|' +
                                 all_rows['Subsidy_Available'].astype(str)).factorize(sort=True)[0]
    keys['income_by_concern'] = (all_rows['Annual_Income_USD'].astype(str) + '|' +
                                 all_rows['Environmental_Concern_Level'].astype(str)).factorize(sort=True)[0]
    return base, keys, mappings

if __name__ == '__main__':
    run = sys.argv[sys.argv.index('--run') + 1]
    out = runner.ROOT / 'artifacts' / run; out.mkdir(parents=True, exist_ok=True)
    (out/'feature_config.json').write_text(json.dumps({'keys':['income_by_subsidy','income_by_concern'], 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    runner.features = features
    runner.main()

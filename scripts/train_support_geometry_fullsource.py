"""Canonical XGBoost plus support geometry from fully reconstructed source.

The public original file has missing income/commute cells. The EDA notebook
reproduces the original generator exactly with RandomState(101), so those cells
can be restored without reading any target labels. This wrapper reuses the
strict source-geometry implementation and changes only the source support set.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

import train_encoded as runner
import train_support_geometry as geometry


def reconstruct_source(source):
    out = source.copy()
    n = len(out)
    rng = np.random.RandomState(101)
    age = rng.randint(25, 70, n)
    gender = rng.choice(['Male', 'Female', 'Other'], n, p=[.52, .45, .03])
    income = np.clip(rng.normal(85_000, 35_000, n), 30_000, None).astype(int)
    city = rng.choice(['Urban', 'Suburban', 'Rural'], n, p=[.50, .35, .15])
    cars = rng.choice([1, 2, 3, 4], n, p=[.40, .40, .15, .05])
    ctype = rng.choice(['Sedan', 'SUV', 'Hatchback', 'Truck'], n, p=[.40, .35, .15, .10])
    commute = np.clip(rng.normal(40, 25, n), 5, None).round(1)
    # Verify the reconstructed sequence against every nonmissing source cell
    # before filling missing values, preventing a silent seed/order mismatch.
    checks = {
        'Age': age, 'Gender': gender, 'Annual_Income_USD': income,
        'City_Type': city, 'Number_of_Cars_Owned': cars,
        'Current_Car_Type': ctype, 'Daily_Commute_km': commute,
    }
    for col, values in checks.items():
        m = source[col].notna().to_numpy()
        if col == 'Daily_Commute_km':
            ok = np.isclose(source.loc[m, col].to_numpy(float), values[m])
        else:
            ok = source.loc[m, col].to_numpy() == values[m]
        assert float(np.mean(ok)) == 1.0, f'generator reconstruction failed for {col}'
        out.loc[~m, col] = values[~m]
    return out


def features(train, test, original):
    if original:
        raise ValueError('Source labels are not used by this experiment')
    base, keys, mappings = geometry.BASE_FEATURES(train, test, False)
    source_path = next((runner.ROOT / 'original').glob('*.csv'))
    source = reconstruct_source(pd.read_csv(source_path))
    source = source[list(geometry.COLUMNS)]
    raw = pd.concat([train[list(geometry.COLUMNS)], test[list(geometry.COLUMNS)]], ignore_index=True)
    extra = {}
    for column in geometry.COLUMNS:
        extra.update({f'{column}_{name}': v for name, v in geometry.support_view(
            raw[column].to_numpy(np.float64), source[column].to_numpy(np.float64),
            geometry.WINDOWS[column]).items()})
    return pd.concat([base, pd.DataFrame(extra)], axis=1), keys, mappings


if __name__ == '__main__':
    run = sys.argv[sys.argv.index('--run') + 1]
    out = runner.ROOT / 'artifacts' / run
    out.mkdir(parents=False, exist_ok=False)
    source_path = next((runner.ROOT / 'original').glob('*.csv'))
    manifest = {
        'hypothesis': 'Reconstructing missing original source values improves support geometry',
        'generator': 'RandomState(101), exact seven-column reproduction before filling missing cells',
        'source_sha256': runner.digest(source_path),
        'script_sha256': runner.digest(Path(__file__)),
        'geometry_script_sha256': runner.digest(Path(geometry.__file__)),
        'source_labels': False, 'submitted': False,
        'validation': 'Canonical outer fivefold / inner fivefold target encoding',
    }
    (out / 'feature_config.json').write_text(json.dumps(manifest, indent=2) + '\n')
    runner.features = features
    geometry.runner.features = features
    geometry.runner.main()

"""Test label-free original-support geometry with canonical nested encoding.

Only the two source feature columns are read; source labels never enter this
view. Counts over competition train+test are deliberately transductive.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

import train_encoded as runner

BASE_FEATURES = runner.features
COLUMNS = ("Annual_Income_USD", "Daily_Commute_km")
WINDOWS = {"Annual_Income_USD": (100.0, 1000.0, 5000.0),
           "Daily_Commute_km": (0.5, 2.0, 10.0)}


def support_view(values, source, windows):
    """Build each value's lookup once; strict neighbors exclude exact matches."""
    unique, inverse, count = np.unique(values, return_inverse=True, return_counts=True)
    source = np.sort(np.asarray(source, dtype=np.float64))
    support = np.unique(source)
    left = np.searchsorted(support, unique, side="left")
    right = np.searchsorted(support, unique, side="right")
    below = unique - support[np.maximum(left - 1, 0)]
    above = support[np.minimum(right, len(support) - 1)] - unique
    below[left == 0] = np.nan
    above[right == len(support)] = np.nan
    lo = np.searchsorted(source, unique, side="left")
    hi = np.searchsorted(source, unique, side="right")
    exact_count = hi - lo
    nearest = np.minimum(
        np.abs(unique - support[np.maximum(left - 1, 0)]),
        np.abs(unique - support[np.minimum(left, len(support) - 1)]))
    cp = count / len(values)
    sp = exact_count / len(source)
    view = {
        "source_count": np.log1p(exact_count),
        "source_novel": (exact_count == 0).astype(float),
        "source_min": (unique == support[0]).astype(float),
        "source_max": (unique == support[-1]).astype(float),
        "source_nearest_distance": np.log1p(nearest),
        "source_predecessor_gap": np.log1p(below),
        "source_successor_gap": np.log1p(above),
        "source_gap_asymmetry": np.log1p(below) - np.log1p(above),
        "source_interval_position": below / (below + above),
        "source_mid_quantile": (lo + hi) / (2 * len(source)),
        "source_quantile_mass": sp,
        "exact_log_lift": np.log((cp + 1 / len(values)) / (sp + 1 / len(source))),
    }
    competition = np.sort(values)
    for window in windows:
        sl = np.searchsorted(source, unique - window, side="left")
        sr = np.searchsorted(source, unique + window, side="right")
        cl = np.searchsorted(competition, unique - window, side="left")
        cr = np.searchsorted(competition, unique + window, side="right")
        sn, cn = sr - sl, cr - cl
        key = f"window_{window:g}"
        view[key + "_source_density"] = sn / (len(source) * 2 * window)
        view[key + "_local_log_lift"] = np.log(
            ((cn + 1) / len(values)) / ((sn + 1) / len(source)))
        view[key + "_competition_atom_share"] = count / np.maximum(cn, 1)
        view[key + "_source_atom_share"] = exact_count / np.maximum(sn, 1)
    return {name: v[inverse].astype("float32") for name, v in view.items()}


def features(train, test, original):
    if original:
        raise ValueError("This experiment excludes original targets")
    base, keys, mappings = BASE_FEATURES(train, test, False)
    source_path = next((runner.ROOT / "original").glob("*.csv"))
    source = pd.read_csv(source_path, usecols=list(COLUMNS))
    raw = pd.concat([train[list(COLUMNS)], test[list(COLUMNS)]], ignore_index=True)
    extra = {}
    for column in COLUMNS:
        values = support_view(raw[column].to_numpy(np.float64),
                              source[column].to_numpy(np.float64), WINDOWS[column])
        extra.update({f"{column}_{name}": v for name, v in values.items()})
    return pd.concat([base, pd.DataFrame(extra)], axis=1), keys, mappings


if __name__ == "__main__":
    run = sys.argv[sys.argv.index("--run") + 1]
    out = runner.ROOT / "artifacts" / run
    out.mkdir(parents=True, exist_ok=False)
    source_path = next((runner.ROOT / "original").glob("*.csv"))
    manifest = {
        "hypothesis": "Strict neighboring source gaps and local mass ratios expose generator value identity beyond exact frequency",
        "script_sha256": runner.digest(Path(__file__)),
        "runner_sha256": runner.digest(Path(runner.__file__)),
        "source_sha256": runner.digest(source_path),
        "source_columns": list(COLUMNS), "windows": WINDOWS,
        "label_boundary": "Source labels never read; competition target encoding uses canonical outer/inner folds",
        "frequency_scope": "Unlabeled competition train+test and public original feature support",
        "comparison": "xgb_nested_v1 single and offline_wave5_conservative rank anchor",
        "blend_weights_prespecified": [0.05, 0.1, 0.2],
        "submitted": False,
    }
    (out / "feature_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runner.features = features
    runner.main()

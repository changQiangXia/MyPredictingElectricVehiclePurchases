"""External original-label pair priors, evaluated with canonical outer folds.

The public source labels are independent of competition validation labels. We
only use them to map feature combinations to smoothed source purchase rates;
competition target encoding remains inside the runner's outer/inner folds.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

import train_encoded as runner

RAW = runner.features
TARGET = runner.TARGET


def key_string(frame, columns):
    # Explicit separators avoid collisions between stringified values.
    return frame[list(columns)].astype(str).agg("\x1f".join, axis=1)


def features(train, test, original):
    base, keys, mappings = RAW(train, test, False)
    orig = pd.read_csv(next((runner.ROOT / "original").glob("*.csv")))
    orig_y = orig[TARGET].eq("Yes").to_numpy(dtype="float64")
    prior = float(orig_y.mean())
    all_rows = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    # Include high-cardinality exact keys with the strongest low-cardinality
    # drivers, plus all low-cardinality pairs. The map is target-derived only
    # from the public source, never from competition labels.
    pairs = []
    cats = [c for c in test.columns if c != "id" and train[c].dtype == object]
    for numeric in ["Annual_Income_USD", "Daily_Commute_km"]:
        for category in ["Subsidy_Available", "Environmental_Concern_Level",
                         "Range_Anxiety_Level", "Home_Charging_Possible",
                         "City_Type", "Current_Car_Type"]:
            pairs.append((numeric, category))
    for i, left in enumerate(cats):
        for right in cats[i + 1:]:
            pairs.append((left, right))
    # Coarse numeric support pairs are less sparse than exact combinations.
    all_rows = all_rows.copy(); orig = orig.copy()
    for divisor in [100, 1000]:
        name = f"Annual_Income_USD_floor_{divisor}"
        # The public source has a small amount of missingness; keep it as one
        # explicit support bucket instead of dropping those rows.
        all_rows[name] = np.floor(all_rows.Annual_Income_USD.fillna(-1) / divisor).astype(int)
        orig[name] = np.floor(orig.Annual_Income_USD.fillna(-1) / divisor).astype(int)
        for category in ["Subsidy_Available", "Environmental_Concern_Level", "Range_Anxiety_Level"]:
            pairs.append((name, category))
    for window in ["exact"]:
        del window
    used = set()
    for left, right in pairs:
        if left not in all_rows or right not in all_rows or (left, right) in used:
            continue
        used.add((left, right))
        source_key = key_string(orig, (left, right))
        query_key = key_string(all_rows, (left, right))
        stats = pd.DataFrame({"key": source_key, "y": orig_y}).groupby("key").y.agg(["sum", "count"])
        counts = query_key.map(stats["count"]).fillna(0).to_numpy(dtype="float64")
        sums = query_key.map(stats["sum"]).fillna(0).to_numpy(dtype="float64")
        # Empirical-Bayes smoothing keeps rare source combinations near the
        # source base rate while preserving larger exact groups.
        for alpha in [5.0, 25.0]:
            rate = (sums + alpha * prior) / (counts + alpha)
            base[f"source_pair_{left}_x_{right}_mean_{int(alpha)}"] = rate.astype("float32")
        base[f"source_pair_{left}_x_{right}_count"] = np.log1p(counts).astype("float32")
    return base, keys, mappings


if __name__ == "__main__":
    run = sys.argv[sys.argv.index("--run") + 1]
    out = runner.ROOT / "artifacts" / run
    out.mkdir(parents=True, exist_ok=False)
    source_path = next((runner.ROOT / "original").glob("*.csv"))
    manifest = {
        "run": run,
        "feature_family": "smoothed external original-label pair priors",
        "pair_scope": "income/commute with categorical drivers, all categorical pairs, income floors with drivers",
        "smoothing": [5.0, 25.0],
        "source_labels": True,
        "competition_labels": "Only canonical nested TE; never used for source prior maps",
        "source_sha256": runner.digest(source_path),
        "script_sha256": runner.digest(Path(__file__)),
        "submitted": False,
        "hypothesis": "Generator preserves enough source conditional response structure that pair priors improve exact-value identity",
    }
    (out / "feature_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runner.features = features
    runner.main()

"""Strict five-fold test of the genuinely new Hermengardo income views.

Adds only income ten-thousands digit and an unlabeled 600-quantile bin to the
canonical nested-TE runner.  The quantile cut points are fit on train+test
features (transductive but label-free); target encoding remains outer/inner
cross-fitted in train_encoded.main.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import train_encoded as runner


def features(train, test, original):
    base, keys, mappings = runner._BASE_FEATURES(train, test, original) if hasattr(runner, '_BASE_FEATURES') else runner.features(train, test, original)
    # The original function is saved below before monkey patching.  Add only
    # two genuinely new columns from the public notebook.
    all_income = pd.concat([train['Annual_Income_USD'], test['Annual_Income_USD']], ignore_index=True).astype(float)
    d4 = (all_income // 10_000 % 10).to_numpy(dtype='float32')
    # Quantile bins are label-free and fitted on the complete competition
    # feature distribution, matching the notebook's transductive protocol.
    q = np.linspace(0.0, 1.0, 601)
    edges = np.quantile(all_income.to_numpy(), q, method='linear')
    edges = np.unique(edges)
    b = np.searchsorted(edges, all_income.to_numpy(), side='right') - 1
    b = np.clip(b, 0, len(edges)-2).astype('float32')
    base['Annual_Income_USD_inc_d4'] = d4
    base['Annual_Income_USD_inc_bin_600'] = b
    keys['inc_d4'] = d4
    keys['inc_bin_600'] = b
    return base, keys, mappings


_base = runner.features
runner._BASE_FEATURES = _base
runner.features = features

if __name__ == '__main__':
    import sys
    run = sys.argv[sys.argv.index('--run') + 1]
    out = runner.ROOT / 'artifacts' / run
    out.mkdir(parents=True, exist_ok=False)
    (out / 'feature_config.json').write_text(json.dumps({
        'hypothesis': 'Hermengardo genuinely new views: income ten-thousands digit and 600-quantile bin',
        'label_boundary': 'quantile edges use unlabeled train+test; all TE remains canonical nested outer/inner',
        'source_not_used': True,
        'submitted': False,
    }, indent=2) + '\n')
    runner.main()

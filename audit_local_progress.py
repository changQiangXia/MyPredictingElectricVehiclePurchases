"""Audit recent local experiments without using submission services.

Reconstructs saved fold predictions, reloads XGBoost models on held-out/test
rows, checks whole-family rule discovery, and compares fixed blend weights.
It never adds the historical full-label edge rules during model comparison.
"""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from train_encoded import ROOT, TARGET, digest, features
from evaluate_offline import auc_placements, paired_delta
from train_hierarchical_te import make_hier
from train_source_value_te import source_features, source_maps
from train_support_rules_model import support_features

RECENT = ['xgb_support_rules_v1', 'xgb_hierarchical_te_v1', 'xgb_source_value_te_v1']
ANCHOR = 'offline_wave2_local_neural'


def reload_recent(tr, te, y, folds):
    n = len(tr)
    base, key, _ = features(tr, te, False)
    X, K = base.to_numpy(), key.to_numpy()
    raw = pd.concat([tr, te], ignore_index=True)
    values = {c: raw[c].to_numpy() for c in ('Annual_Income_USD', 'Daily_Commute_km')}
    hierarchy = dict(values, income_floor_100=np.floor(values['Annual_Income_USD']/100),
                     income_floor_1000=np.floor(values['Annual_Income_USD']/1000),
                     commute_floor=np.floor(values['Daily_Commute_km']))
    maps = source_maps(pd.read_csv(ROOT / 'original/EV_Adoption_and_Range_Anxiety_Dataset.csv'))
    checks = []
    for run in RECENT:
        p = ROOT / 'artifacts' / run
        oof, test_sum = np.full(n, np.nan), np.zeros(len(te))
        count = np.zeros(n, dtype='int8')
        errors = []
        for f in range(5):
            ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            saved = np.load(p / f'fold_{f}.npz')
            np.testing.assert_array_equal(saved['valid_indices'], vi)
            oof[vi] = saved['valid_prediction']; count[vi] += 1
            test_sum += saved['test_prediction']/5
            # Replay disjoint samples of both held-out and test rows.
            va_take = np.linspace(0, len(vi)-1, 512, dtype=int)
            te_take = np.linspace(0, len(te)-1, 512, dtype=int)
            qi = np.concatenate([vi[va_take], n+te_take])
            fit_values, q_values = ({c: values[c][ti] for c in values},
                                    {c: values[c][qi] for c in values})
            if run == 'xgb_support_rules_v1':
                extra, _ = support_features(fit_values, y[ti], q_values)
            elif run == 'xgb_source_value_te_v1':
                extra, _ = source_features(fit_values, y[ti], q_values, maps)
            else:
                extra, _ = make_hier(hierarchy, ti, y[ti], qi, {c: hierarchy[c][qi] for c in hierarchy})
            encoders = joblib.load(p / f'encoders_{f}.joblib')
            Q = np.column_stack([X[qi], extra] + [e.transform(K[qi]).astype('float32') for e in encoders])
            model = XGBClassifier()
            model.load_model(p / f'model_{f}.ubj')
            model.set_params(device='cpu', n_jobs=8)
            replay = model.predict_proba(Q)[:, 1]
            expected = np.concatenate([saved['valid_prediction'][va_take], saved['test_prediction'][te_take]])
            np.testing.assert_allclose(replay, expected, atol=1e-7, rtol=1e-6)
            errors.append(float(np.max(np.abs(replay-expected))))
        assert (count == 1).all()
        np.testing.assert_array_equal(oof, np.load(p / 'oof.npy'))
        np.testing.assert_allclose(test_sum, np.load(p / 'test.npy'), atol=1e-12, rtol=0)
        sub = pd.read_csv(p / 'submission.csv')
        np.testing.assert_array_equal(sub.id, te.id)
        np.testing.assert_allclose(sub[TARGET], test_sum, atol=1e-15, rtol=1e-12)
        checks.append(dict(run=run, valid_rows_per_fold=512, test_rows_per_fold=512,
                           reload_max_errors=errors, oof_assignment_count=1,
                           oof_auc=float(roc_auc_score(y, oof)),
                           oof_sha256=digest(p/'oof.npy'), test_sha256=digest(p/'test.npy'),
                           submission_sha256=digest(p/'submission.csv')))
        print(json.dumps(checks[-1]), flush=True)
    return checks


def audit_families(tr, te, y, folds, out):
    source = ROOT / 'artifacts/experimental/support_rule_family_audit_v1'
    report = pd.read_csv(source/'family_report.csv')
    anchor = np.load(ROOT/'artifacts'/ANCHOR/'oof.npy')
    canonical = np.load(ROOT/'artifacts/xgb_nested_v1/oof.npy')
    canonical_auc = float(roc_auc_score(y, canonical))
    rows, group_diagnostics = [], []
    for spec in report.to_dict('records'):
        c, width, minimum = spec['column'], spec['width'], spec['minimum']
        x = tr[c].to_numpy()
        k = x if spec['kind'] == 'exact' else np.floor(x/width)*width
        z = te[c].to_numpy()
        kt = z if spec['kind'] == 'exact' else np.floor(z/width)*width
        directory = source/spec['family']/f'min_{minimum}'
        p, q = anchor.copy(), canonical.copy()
        wrong_zero, wrong_one, covers, wrong_groups = 0, 0, [], 0
        for f in range(5):
            ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            stats = pd.DataFrame(dict(k=k[ti], y=y[ti])).groupby('k').y.agg(['size','sum'])
            expected = stats[(stats['size'] >= minimum) & ((stats['sum']==0) | (stats['sum']==stats['size']))]
            try:
                saved_map = pd.read_csv(directory/f'fold_{f}_map.csv')
            except pd.errors.EmptyDataError:
                saved_map = pd.DataFrame(columns=['key','fit_n','fit_pos','pure'])
            # Equality checks include all false discoveries; no global-purity filter.
            assert set(saved_map.key) == set(expected.index)
            for row in saved_map.itertuples():
                st = expected.loc[row.key]
                assert int(st['size']) == row.fit_n and int(st['sum']) == row.fit_pos
                assert row.pure == ('zero' if row.fit_pos == 0 else 'one')
            zero = expected.index[expected['sum']==0]
            one = expected.index[expected['sum']==expected['size']]
            zm, om = np.isin(k[vi], zero), np.isin(k[vi], one)
            p[vi[zm]], p[vi[om]] = 0., 1.
            q[vi[zm]], q[vi[om]] = 0., 1.
            wz, wo = int(y[vi[zm]].sum()), int((1-y[vi[om]]).sum())
            wrong_zero += wz; wrong_one += wo
            covers.append(int(np.isin(kt, expected.index).sum()))
            held = pd.DataFrame(dict(k=k[vi], y=y[vi])).groupby('k').y.agg(['size','sum'])
            for val, st in expected.iterrows():
                valid_n = int(held.loc[val, 'size']) if val in held.index else 0
                valid_pos = int(held.loc[val, 'sum']) if val in held.index else 0
                failure = valid_pos if st['sum']==0 else valid_n-valid_pos
                wrong_groups += int(failure > 0)
                if failure > 0:
                    group_diagnostics.append(dict(family=spec['family'], minimum=minimum, fold=f,
                        key=float(val), train_n=int(st['size']), train_pos=int(st['sum']),
                        validation_n=valid_n, validation_pos=valid_pos, contradicted_rows=failure))
            checkpoint = np.load(directory/f'fold_{f}.npz')
            np.testing.assert_array_equal(checkpoint['valid_indices'], vi)
            np.testing.assert_array_equal(checkpoint['valid_prediction'], p[vi])
            np.testing.assert_array_equal(checkpoint['valid_rule_mask'], zm|om)
        auc = float(roc_auc_score(y, p))
        assert abs(auc-spec['oof_auc']) < 1e-12
        rows.append(dict(family=spec['family'], minimum=minimum, self_anchor_auc=auc,
                         self_anchor_delta=spec['oof_delta'], canonical_delta=float(roc_auc_score(y,q))-canonical_auc,
                         wrong_zero_rows=wrong_zero, wrong_one_rows=wrong_one, contradicted_fold_groups=wrong_groups,
                         test_coverage_min=min(covers), test_coverage_max=max(covers),
                         test_coverage_mean=float(np.mean(covers)), positive_folds=spec['positive_folds']))
    pd.DataFrame(rows).to_csv(out/'family_diagnostics.csv', index=False)
    pd.DataFrame(group_diagnostics).to_csv(out/'contradicted_rules.csv', index=False)
    return dict(families_checked=len(rows), maps_checked=5*len(rows), discovery_maps_complete=True,
                validation_predictions_reconstructed=True,
                notes=['Selection among 70 family/threshold combinations is exploratory.',
                       'Historical self anchor already contains full-label-selected income/commute edges.',
                       'canonical_delta also uses raw xgb_nested_v1 OOF, with no added historical edges.',
                       'Different random splits of these labels are stability checks, not independent data.'])


def compare_models(y, folds, runs, out):
    a = np.load(ROOT/'artifacts'/ANCHOR/'oof.npy')
    ar = rankdata(a)/len(a)
    fa = [float(roc_auc_score(y[folds==f], a[folds==f])) for f in range(5)]
    placements = auc_placements(y.astype(bool), a)
    rows = []
    for run in runs:
        p = np.load(ROOT/'artifacts'/run/'oof.npy')
        pr = rankdata(p)/len(p)
        for w in (0.05, 0.1, 0.2):
            b = (1-w)*ar+w*pr
            fd = [float(roc_auc_score(y[folds==f], b[folds==f]))-fa[f] for f in range(5)]
            rec = dict(run=run, weight=w, single_auc=float(roc_auc_score(y,p)),
                       spearman=float(np.corrcoef(ar,pr)[0,1]), auc=float(roc_auc_score(y,b)),
                       fold_deltas=fd, positive_folds=sum(v>0 for v in fd),
                       **paired_delta(y.astype(bool), a, b, placements))
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out/'self_model_comparison.csv', index=False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--additional-models', nargs='*', default=[])
    args = parser.parse_args()
    out = ROOT/'artifacts'/args.run
    out.mkdir(exist_ok=False)
    tr, te = pd.read_csv(ROOT/'train.csv'), pd.read_csv(ROOT/'test.csv')
    y = tr[TARGET].eq('Yes').to_numpy('int8')
    folds = np.load(ROOT/'artifacts/folds_seed42.npy')
    checks = reload_recent(tr, te, y, folds)
    family = audit_families(tr, te, y, folds, out)
    comparison = compare_models(y, folds, RECENT+args.additional_models, out)
    report = dict(run=args.run, status='complete', submitted=False, anchor=ANCHOR,
        self_anchor_auc=float(roc_auc_score(y,np.load(ROOT/'artifacts'/ANCHOR/'oof.npy'))),
        model_checks=checks, family_audit=family, comparison=comparison,
        selection_notes=['80/20 added-XGB blend was selected after a weight scan; it was not prespecified.',
                         'Support-family candidates are screens, not independently confirmed generalization gains.',
                         'Conditional DeLong intervals exclude selection, retraining and shared-fold dependence.'],
        script_sha256=digest(Path(__file__)), train_sha256=digest(ROOT/'train.csv'),
        test_sha256=digest(ROOT/'test.csv'), fold_sha256=digest(ROOT/'artifacts/folds_seed42.npy'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(run=args.run, status='complete', family_audit=family)),flush=True)


if __name__ == '__main__':
    main()

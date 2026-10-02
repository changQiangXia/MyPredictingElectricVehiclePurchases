"""Compare fold-averaged TabM and XGBoost on the same excluded outer groups.

Reuses the fixed outer/inner splits of diagnosis_bagging_transfer_v1. Each
heldout row is excluded from all ten members, including TE and early stopping.
This diagnostic does not generate a competition submission.
"""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
import gc
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import rtdl_num_embeddings
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_offline import auc_placements, paired_delta
from train_encoded import ROOT, TARGET, digest
from train_tabm_residual import ResidualTabM, prepare, predict


def log(record):
    print(json.dumps(record), flush=True)


def rank(pred):
    return rankdata(pred, method='average') / len(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--outer-folds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    assert len(set(args.outer_folds)) == len(args.outer_folds)
    assert set(args.outer_folds).issubset({0, 1, 2})
    assert torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    start = time.time()
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    reference = ROOT / 'artifacts/diagnosis_bagging_transfer_v1'
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    y = tr[TARGET].eq('Yes').to_numpy('float32')
    outer = np.load(reference / 'outer_folds.npy')
    expected = np.full(len(y), -1, dtype='int8')
    for f, (_, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=731).split(tr, y)):
        expected[vi] = f
    np.testing.assert_array_equal(expected, outer)
    numeric, categorical, exact, margin, meta, keys = prepare(tr, te)
    model_args = SimpleNamespace(epochs=90, patience=12, batch_size=4096,
        learning_rate=.002, weight_decay=.02, width=192, members=8,
        embedding_dim=16, bins=48, dropout=.2, effect_dropout=.1,
        income_penalty=10., commute_penalty=50., seed=20260910,
        no_formula=False, no_effects=True, encoding=True)
    config = dict(run=args.run, model_args=vars(model_args),
        outer_seed=731, outer_splits=5, diagnosed_outer_folds=[0, 1, 2],
        bag_splits=10, bag_fold_seed=42, te_splits=5, te_seed=42,
        weights=[0., .1, .2, .3, .5, 1.],
        reference='diagnosis_bagging_transfer_v1 seed42 ten-member probability mean',
        protocol='Heldout labels excluded from every encoding, fit, scale/bin fit and early stopping',
        limitation='Previously explored labels; three diagnostic groups; each member fits 72% of full training data',
        frequency_scope='unlabeled full train+test',
        code_sha256=digest(Path(__file__)),
        model_code_sha256=digest(ROOT / 'train_tabm_residual.py'),
        feature_code_sha256=digest(ROOT / 'train_encoded.py'),
        train_sha256=digest(ROOT / 'train.csv'), test_sha256=digest(ROOT / 'test.csv'),
        outer_fold_sha256=digest(reference / 'outer_folds.npy'), submitted=False)
    cp = out / 'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config, 'Changed diagnostic requires a fresh run ID'
    else:
        cp.write_text(json.dumps(config, indent=2) + '\n')
        (out / 'training_script.py').write_bytes(Path(__file__).read_bytes())
    for f in args.outer_folds:
        dev, heldout = np.flatnonzero(outer != f), np.flatnonzero(outer == f)
        for j, (fit_local, stop_local) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(dev, y[dev])):
            fit, stop = dev[fit_local], dev[stop_local]
            stem = f'outer_{f}_member_{j}'
            ck = out / f'{stem}.npz'
            with np.load(reference / f'outer_{f}_seed_42_member_{j}.npz') as z:
                np.testing.assert_array_equal(z['fit_indices'], fit)
                np.testing.assert_array_equal(z['stop_indices'], stop)
                np.testing.assert_array_equal(z['heldout_indices'], heldout)
            if ck.exists():
                with np.load(ck) as z:
                    np.testing.assert_array_equal(z['heldout_indices'], heldout)
                log(dict(event='resumed', outer=f, member=j))
                continue
            t0 = time.time()
            seed = model_args.seed + j
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            parts, encoders = [[numeric[fit]], [numeric[stop]], [numeric[heldout]]], []
            for smooth in ('auto', 10., 100.):
                enc = TargetEncoder(target_type='binary', smooth=smooth, cv=5, shuffle=True, random_state=42)
                parts[0].append(enc.fit_transform(keys[fit], y[fit]).astype('float32'))
                parts[1].append(enc.transform(keys[stop]).astype('float32'))
                parts[2].append(enc.transform(keys[heldout]).astype('float32'))
                encoders.append(enc)
            raw_A, raw_B, raw_C = [np.column_stack(p) for p in parts]
            active = np.ptp(raw_A, axis=0) > 0
            scaler = StandardScaler().fit(raw_A[:, active])
            A, B, C = [scaler.transform(v[:, active]).astype('float32') for v in (raw_A, raw_B, raw_C)]
            bins = rtdl_num_embeddings.compute_bins(torch.from_numpy(A), n_bins=model_args.bins)
            preprocessor = dict(scaler=scaler, active=active, bins=[b.numpy() for b in bins],
                                metadata=meta, encoders=encoders)
            preprocessor_path = out / f'{stem}.joblib'
            joblib.dump(preprocessor, preprocessor_path)
            model = ResidualTabM(A.shape[1], meta['cat_cardinalities'], bins,
                                  meta['exact_cardinalities'], model_args).cuda()
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(), 'weight_decay': model_args.weight_decay},
                {'params': model.effects.parameters(), 'weight_decay': 0.},
                {'params': [model.formula_scale], 'weight_decay': 0.},
            ], lr=model_args.learning_rate)
            train_t = [torch.as_tensor(v, device='cuda') for v in (A, categorical[fit], exact[fit], margin[fit])]
            stop_t = [torch.as_tensor(v, device='cuda') for v in (B, categorical[stop], exact[stop], margin[stop])]
            heldout_t = [torch.as_tensor(v, device='cuda') for v in (C, categorical[heldout], exact[heldout], margin[heldout])]
            labels = torch.as_tensor(y[fit], device='cuda')
            best_auc, best_epoch, best_state = -np.inf, -1, None
            history = []
            for epoch in range(model_args.epochs):
                model.train()
                losses = []
                for batch in torch.randperm(len(fit), device='cuda').split(model_args.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model(*(v[batch] for v in train_t))
                        loss = F.binary_cross_entropy_with_logits(logits.float(), labels[batch, None].expand_as(logits))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10.)
                    optimizer.step()
                    losses.append(loss.detach())
                vp = predict(model, stop_t)
                auc = float(roc_auc_score(y[stop], vp))
                history.append(dict(epoch=epoch, stop_auc=auc, loss=float(torch.stack(losses).mean())))
                if auc > best_auc:
                    best_auc, best_epoch = auc, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch - best_epoch >= model_args.patience:
                    break
            model.load_state_dict(best_state)
            mp = out / f'{stem}.pt'
            torch.save(best_state, mp)
            hp = predict(model, heldout_t)
            state = joblib.load(preprocessor_path)
            picks = np.linspace(0, len(heldout) - 1, 512, dtype=int)
            raw_replay = np.column_stack([numeric[heldout[picks]]] +
                [e.transform(keys[heldout[picks]]).astype('float32') for e in state['encoders']])
            num_replay = state['scaler'].transform(raw_replay[:, state['active']]).astype('float32')
            replay_model = ResidualTabM(A.shape[1], meta['cat_cardinalities'],
                [torch.from_numpy(b) for b in state['bins']], meta['exact_cardinalities'], model_args).cuda()
            replay_model.load_state_dict(torch.load(mp, map_location='cpu', weights_only=True))
            replay_t = [torch.as_tensor(v, device='cuda') for v in
                        (num_replay, categorical[heldout[picks]], exact[heldout[picks]], margin[heldout[picks]])]
            replay = predict(replay_model, replay_t)
            np.testing.assert_allclose(replay, hp[picks], rtol=2e-4, atol=2e-6)
            assert np.isfinite(hp).all()
            np.savez_compressed(ck, heldout_indices=heldout, heldout_prediction=hp)
            rec = dict(outer=f, member=j, fit_rows=len(fit), stop_rows=len(stop), heldout_rows=len(heldout),
                stop_auc=best_auc, best_epoch=best_epoch, epochs_completed=len(history),
                model_sha256=digest(mp), checkpoint_sha256=digest(ck),
                preprocessor_sha256=digest(preprocessor_path),
                reload_max_error=float(np.max(np.abs(replay - hp[picks]))), seconds=time.time() - t0)
            (out / f'{stem}.json').write_text(json.dumps(rec, indent=2) + '\n')
            (out / f'{stem}_epochs.json').write_text(json.dumps(history, indent=2) + '\n')
            log(dict(event='fit_completed', **rec))
            del model, replay_model, optimizer, train_t, stop_t, heldout_t, replay_t, labels
            del A, B, C, raw_A, raw_B, raw_C, parts, encoders, preprocessor, state, best_state
            gc.collect()
            torch.cuda.empty_cache()
    complete = [f for f in range(3) if all((out / f'outer_{f}_member_{j}.npz').exists() for j in range(10))]
    records, fold_records = [], []
    for f in complete:
        heldout = np.flatnonzero(outer == f)
        labels = y[heldout].astype(bool)
        base = np.stack([np.load(reference / f'outer_{f}_seed_42_member_{j}.npz')['heldout_prediction']
                         for j in range(10)]).astype('float64').mean(0)
        new = np.stack([np.load(out / f'outer_{f}_member_{j}.npz')['heldout_prediction']
                        for j in range(10)]).astype('float64').mean(0)
        a, b = float(roc_auc_score(labels, base)), float(roc_auc_score(labels, new))
        placements = auc_placements(labels, base)
        fold_records.append(dict(outer=f, base_auc=a, tabm_auc=b,
            rank_correlation=float(np.corrcoef(rank(base), rank(new))[0, 1])))
        for weight in config['weights']:
            pred = (1 - weight) * rank(base) + weight * rank(new)
            rec = dict(outer=f, weight=weight, auc=float(roc_auc_score(labels, pred)),
                       **paired_delta(labels, base, pred, placements))
            records.append(rec)
            log(dict(event='scored', **rec))
    table = pd.DataFrame(records)
    table.to_csv(out / 'comparison.csv', index=False)
    report = dict(run=args.run, status='diagnosis_complete' if complete == [0, 1, 2] else 'diagnostic_pilot',
        submitted=False, complete_outer_groups=complete, fold_records=fold_records, comparisons=records,
        mean_by_weight=table.groupby('weight')[['auc', 'delta']].mean().reset_index().to_dict('records'),
        uncertainty_note='Conditional paired SE excludes model and weight selection and shared-training dependence',
        elapsed_seconds=time.time() - start, config_sha256=digest(cp))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    log(dict(event=report['status'], complete_outer_groups=complete,
             mean_by_weight=report['mean_by_weight'], fold_records=fold_records))


if __name__ == '__main__':
    main()

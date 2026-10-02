"""TabM with a formula offset and regularized, jointly fitted value effects.

All trainable effects see only outer-training labels. Exact income and commute
effects are model parameters, not globally computed target encodings. Category
vocabularies and frequency counts use unlabeled train+test; numeric preprocessing
uses outer training rows. Each completed fold is resumable and reload checked.
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

import joblib
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import tabm
import torch
from scipy.special import logit, ndtr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, TargetEncoder
from torch import nn
from torch.nn import functional as F

from train_encoded import ROOT, TARGET, digest, features


def log(record):
    print(json.dumps(record), flush=True)


def formula_margin(frame):
    anxiety = frame.Range_Anxiety_Level.map({'Low': 0., 'Medium': -1., 'High': -3.})
    score = (1.2 * frame.Annual_Income_USD / 1e5
             + 0.6 * frame.Environmental_Concern_Level
             + 2.0 * frame.Subsidy_Available.eq('Yes') + anxiety - 5.5)
    return np.asarray(logit(np.clip(ndtr(score), 1e-6, 1-1e-6)), dtype='float32')


class ResidualTabM(nn.Module):
    def __init__(self, n_num, cardinalities, bins, exact_cardinalities, args):
        super().__init__()
        self.backbone = tabm.TabM.make(
            n_num_features=n_num, cat_cardinalities=cardinalities, d_out=1,
            num_embeddings=rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins, d_embedding=args.embedding_dim, activation=False, version='B'),
            k=args.members, d_block=args.width, n_blocks=2, dropout=args.dropout)
        self.effects = nn.ModuleList([nn.Embedding(c, args.members) for c in exact_cardinalities])
        for effect in self.effects:
            nn.init.zeros_(effect.weight)
        self.effect_dropout = args.effect_dropout
        self.use_formula = not args.no_formula
        self.use_effects = not args.no_effects
        self.formula_scale = nn.Parameter(torch.ones(args.members))

    def forward(self, numeric, categorical, exact, margin):
        logits = self.backbone(numeric, categorical).squeeze(-1)
        if self.use_formula:
            logits = logits + margin[:, None] * self.formula_scale[None, :]
        if self.use_effects:
            for i, effect in enumerate(self.effects):
                delta = effect(exact[:, i])
                if self.training and self.effect_dropout:
                    keep = (torch.rand((len(delta), 1), device=delta.device) >= self.effect_dropout)
                    delta = delta * keep
                logits = logits + delta
        return logits

    def penalty(self, n_train, args):
        if not self.use_effects:
            return 0.
        # Objective is mean BCE. Dividing the sum of squared effects by N
        # gives fixed per-value pseudo-observation strength as support varies.
        return sum(lam * e.weight.square().sum() / (2 * n_train * args.members)
                   for lam, e in zip((args.income_penalty, args.commute_penalty), self.effects))


@torch.inference_mode()
def predict(model, tensors, batch_size=8192):
    model.eval()
    pieces = []
    for start in range(0, len(tensors[0]), batch_size):
        pieces.append(model(*(x[start:start+batch_size] for x in tensors))
                      .sigmoid().mean(1).cpu().numpy())
    return np.concatenate(pieces)


def prepare(tr, te):
    base, keys, mappings = features(tr, te, False)
    cat_columns = [c for c in base if not c.endswith('_frequency') and base[c].nunique() <= 20]
    num_columns = [c for c in base if c not in cat_columns]
    category_values, cat_arrays = {}, []
    for c in cat_columns:
        values = sorted(base[c].unique().tolist())
        category_values[c] = values
        cat_arrays.append(base[c].map({v: i for i, v in enumerate(values)}).to_numpy('int64'))
    categorical = np.column_stack(cat_arrays)
    cardinalities = [len(category_values[c]) for c in cat_columns]
    numeric = base[num_columns].to_numpy('float32')
    # Heavy frequency spikes are easier to model on a logarithmic scale.
    for j, c in enumerate(num_columns):
        if c.endswith('_frequency'):
            numeric[:, j] = np.log1p(numeric[:, j] * len(base))
    raw = pd.concat([tr, te], ignore_index=True)
    exact_columns = ['Annual_Income_USD', 'Daily_Commute_km']
    exact, exact_values = [], {}
    for c in exact_columns:
        values = np.sort(raw[c].unique())
        exact.append(np.searchsorted(values, raw[c]).astype('int64'))
        exact_values[c] = values.tolist()
    metadata = dict(numeric_features=num_columns, categorical_features=cat_columns,
                    categorical_values=category_values, raw_category_mappings=mappings,
                    exact_columns=exact_columns, exact_values=exact_values,
                    cat_cardinalities=cardinalities,
                    exact_cardinalities=[len(exact_values[c]) for c in exact_columns])
    return numeric, categorical, np.column_stack(exact), formula_margin(raw), metadata, keys.to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--fold-seed', type=int, default=42)
    parser.add_argument('--folds', nargs='+', type=int)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--width', type=int, default=192)
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--embedding-dim', type=int, default=16)
    parser.add_argument('--bins', type=int, default=48)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--effect-dropout', type=float, default=0.1)
    parser.add_argument('--income-penalty', type=float, default=10.)
    parser.add_argument('--commute-penalty', type=float, default=50.)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--no-formula', action='store_true')
    parser.add_argument('--no-effects', action='store_true')
    parser.add_argument('--encoding', action='store_true')
    args = parser.parse_args()
    if args.folds is None:
        args.folds = list(range(args.n_splits))
    assert args.n_splits >= 2 and args.folds and set(args.folds).issubset(range(args.n_splits))
    assert torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    n = len(tr)
    y = tr[TARGET].eq('Yes').to_numpy('float32')
    fold_path = ROOT / f'artifacts/folds_seed{args.fold_seed}_{args.n_splits}.npy'
    if fold_path.exists():
        folds = np.load(fold_path)
    else:
        from sklearn.model_selection import StratifiedKFold
        folds = np.full(n, -1, dtype='int8')
        for f, (_, vi) in enumerate(StratifiedKFold(args.n_splits, shuffle=True,
                                                     random_state=args.fold_seed).split(tr, y)):
            folds[vi] = f
        np.save(fold_path, folds)
    assert folds.shape == y.shape and set(folds) == set(range(args.n_splits))
    numeric, categorical, exact, margin, meta, keys = prepare(tr, te)
    config = {k: v for k, v in vars(args).items() if k != 'folds'}
    config.update(code_sha256=digest(Path(__file__)), feature_code_sha256=digest(ROOT / 'train_encoded.py'),
                  train_sha256=digest(ROOT / 'train.csv'), test_sha256=digest(ROOT / 'test.csv'),
                  fold_sha256=digest(fold_path), tabm_version=tabm.__version__, torch_version=torch.__version__,
                  frequency_scope='unlabeled train+test; log1p(count)',
                  supervised_encoding='outer 5 / inner 5, smooth auto/10/100' if args.encoding else 'none',
                  fold_seed=args.fold_seed, n_splits=args.n_splits,
                  validation='outer folds; all trainable effects see only outer training labels',
                  comparison_weights=[0.05, 0.1, 0.2], submitted=False,
                  features={k: v for k, v in meta.items() if k != 'exact_values'})
    cp = out / 'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config, 'Changed configuration requires a fresh run ID'
    else:
        cp.write_text(json.dumps(config, indent=2)+'\n')
    for fold in args.folds:
        checkpoint = out / f'fold_{fold}.npz'
        if checkpoint.exists():
            log(dict(event='resumed', fold=fold))
            continue
        t0 = time.time()
        seed = args.seed + fold
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        parts = [[numeric[ti]], [numeric[vi]], [numeric[n:]]]
        encoders = []
        if args.encoding:
            for smooth in ('auto', 10., 100.):
                encoder = TargetEncoder(target_type='binary', smooth=smooth, cv=5,
                                        shuffle=True, random_state=42)
                parts[0].append(encoder.fit_transform(keys[ti], y[ti]).astype('float32'))
                parts[1].append(encoder.transform(keys[vi]).astype('float32'))
                parts[2].append(encoder.transform(keys[n:]).astype('float32'))
                encoders.append(encoder)
        raw_A, raw_B, raw_C = [np.column_stack(p) for p in parts]
        active = np.ptp(raw_A, axis=0) > 0
        scaler = StandardScaler().fit(raw_A[:, active])
        A, B, C = [scaler.transform(x[:, active]).astype('float32') for x in (raw_A, raw_B, raw_C)]
        bins = rtdl_num_embeddings.compute_bins(torch.from_numpy(A), n_bins=args.bins)
        preprocess = dict(scaler=scaler, active=active, bins=[b.numpy() for b in bins], metadata=meta,
                          encoders=encoders)
        joblib.dump(preprocess, out / f'preprocessor_{fold}.joblib')
        model = ResidualTabM(A.shape[1], meta['cat_cardinalities'], bins, meta['exact_cardinalities'], args).cuda()
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'weight_decay': args.weight_decay},
            {'params': model.effects.parameters(), 'weight_decay': 0.},
            {'params': [model.formula_scale], 'weight_decay': 0.},
        ], lr=args.learning_rate)
        train_tensors = [torch.as_tensor(x, device='cuda') for x in (A, categorical[ti], exact[ti], margin[ti])]
        valid_tensors = [torch.as_tensor(x, device='cuda') for x in (B, categorical[vi], exact[vi], margin[vi])]
        test_tensors = [torch.as_tensor(x, device='cuda') for x in (C, categorical[n:], exact[n:], margin[n:])]
        labels = torch.as_tensor(y[ti], device='cuda')
        best_auc, best_epoch, best_state = -np.inf, -1, None
        history = []
        log(dict(event='prepared', fold=fold, parameters=sum(p.numel() for p in model.parameters()), seconds=time.time()-t0))
        for epoch in range(args.epochs):
            et = time.time()
            model.train()
            losses = []
            for batch in torch.randperm(len(ti), device='cuda').split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(*(x[batch] for x in train_tensors))
                    loss = F.binary_cross_entropy_with_logits(logits.float(), labels[batch, None].expand_as(logits))
                    loss = loss + model.penalty(len(ti), args)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                losses.append(loss.detach())
            vp = predict(model, valid_tensors)
            auc = float(roc_auc_score(y[vi], vp))
            rec = dict(fold=fold, epoch=epoch, auc=auc, loss=float(torch.stack(losses).mean()), seconds=time.time()-et)
            history.append(rec)
            if epoch % 5 == 0:
                log(rec)
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch-best_epoch >= args.patience:
                break
        model.load_state_dict(best_state)
        model_path = out / f'model_{fold}.pt'
        torch.save(best_state, model_path)
        vp, tp = predict(model, valid_tensors), predict(model, test_tensors)
        state = joblib.load(out / f'preprocessor_{fold}.joblib')
        replay_raw = np.column_stack([numeric[n:n+1024]] + [enc.transform(keys[n:n+1024]).astype('float32')
                                                         for enc in state['encoders']])
        replay_num = state['scaler'].transform(replay_raw[:, state['active']]).astype('float32')
        np.testing.assert_allclose(replay_num, C[:1024], atol=1e-6)
        replay_model = ResidualTabM(A.shape[1], meta['cat_cardinalities'],
                                   [torch.from_numpy(b) for b in state['bins']], meta['exact_cardinalities'], args).cuda()
        replay_model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        replay_tensors = [torch.as_tensor(replay_num, device='cuda')] + [x[:1024] for x in test_tensors[1:]]
        replay = predict(replay_model, replay_tensors)
        np.testing.assert_allclose(replay, tp[:1024], rtol=2e-4, atol=2e-6)
        assert np.isfinite(vp).all() and np.isfinite(tp).all()
        rec = dict(fold=fold, auc=float(roc_auc_score(y[vi], vp)), best_epoch=best_epoch,
                   epochs_completed=len(history), seconds=time.time()-t0, model_sha256=digest(model_path),
                   reload_checked_rows=1024, reload_max_error=float(np.max(np.abs(replay-tp[:1024]))))
        np.savez_compressed(checkpoint, valid_indices=vi, valid_prediction=vp, test_prediction=tp)
        (out / f'fold_{fold}.json').write_text(json.dumps(rec, indent=2)+'\n')
        (out / f'epochs_{fold}.json').write_text(json.dumps(history, indent=2)+'\n')
        log(rec)
        del model, replay_model, optimizer, train_tensors, valid_tensors, test_tensors, replay_tensors, labels
        del A, B, C, best_state
        gc.collect(); torch.cuda.empty_cache()
    available = [f for f in range(args.n_splits) if (out / f'fold_{f}.npz').exists()]
    oof, test_pred, records = np.full(n, np.nan), np.zeros(len(te)), []
    for fold in available:
        z = np.load(out / f'fold_{fold}.npz')
        vi = np.flatnonzero(folds == fold)
        assert np.array_equal(vi, z['valid_indices'])
        oof[vi] = z['valid_prediction']; test_pred += z['test_prediction'] / len(available)
        records.append(json.loads((out / f'fold_{fold}.json').read_text()))
    complete = np.isfinite(oof).all()
    metrics = dict(run=args.run, status='complete' if complete else 'pilot', submitted=False,
                   folds=records, elapsed_seconds=time.time()-started)
    if complete:
        metrics['oof_auc'] = float(roc_auc_score(y, oof))
        np.save(out / 'oof.npy', oof); np.save(out / 'test.npy', test_pred)
        sub = pd.read_csv(ROOT / 'sample_submission.csv')
        assert sub.id.equals(te.id) and np.isfinite(test_pred).all()
        assert ((test_pred >= 0) & (test_pred <= 1)).all()
        sub[TARGET] = test_pred
        sub.to_csv(out / 'submission.csv', index=False)
        metrics['submission_sha256'] = digest(out / 'submission.csv')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    log(metrics)


if __name__ == '__main__':
    main()

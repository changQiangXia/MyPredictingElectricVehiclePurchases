"""Local TabM experiments with fold-isolated preprocessing and saved checkpoints."""
import os

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import argparse
import copy
import gc
import hashlib
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, TargetEncoder
from torch.nn import functional as F

from train_encoded import ROOT, TARGET, digest, features


def log(record):
    print(json.dumps(record), flush=True)


def create_model(n_num, cardinalities, bins, args):
    return tabm.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=cardinalities,
        d_out=1,
        num_embeddings=rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, d_embedding=args.embedding_dim, activation=False, version='B'),
        k=args.members, d_block=args.width, n_blocks=2, dropout=args.dropout,
    )


@torch.inference_mode()
def predict(model, numeric, categorical, batch_size=8192):
    model.eval()
    predictions = []
    for start in range(0, len(numeric), batch_size):
        logits = model(numeric[start:start + batch_size], categorical[start:start + batch_size])
        predictions.append(logits.squeeze(-1).sigmoid().mean(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--width', type=int, default=192)
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--embedding-dim', type=int, default=16)
    parser.add_argument('--bins', type=int, default=48)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20260909)
    args = parser.parse_args()
    assert set(args.folds).issubset(set(range(5))) and args.folds
    assert torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    start_time = time.time()
    out = ROOT / 'artifacts' / args.run
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(ROOT / 'train.csv'), pd.read_csv(ROOT / 'test.csv')
    n = len(tr)
    y = tr[TARGET].eq('Yes').to_numpy(dtype='float32')
    fold_path = ROOT / 'artifacts' / 'folds_seed42.npy'
    fold_ids = np.load(fold_path)
    assert fold_ids.shape == y.shape and set(fold_ids) == set(range(5))
    base, keys, mappings = features(tr, te, False)
    cat_columns = [c for c in base if not c.endswith('_frequency') and base[c].nunique() <= 20]
    num_columns = [c for c in base if c not in cat_columns]
    cardinalities = []
    category_values = {}
    encoded_cats = []
    for c in cat_columns:
        values = sorted(base[c].unique().tolist())
        category_values[c] = values
        encoded_cats.append(base[c].map({v: i for i, v in enumerate(values)}).to_numpy('int64'))
        cardinalities.append(len(values))
    cat = np.column_stack(encoded_cats)
    numeric = base[num_columns].to_numpy('float32')
    key_columns = ['Annual_Income_USD', 'Daily_Commute_km', 'income_floor_100',
                   'income_floor_1000', 'commute_floor']
    key = keys[key_columns].to_numpy()
    del base, keys
    config = vars(args).copy()
    config.pop('folds')
    config.update({
        'code_sha256': digest(Path(__file__)), 'feature_code_sha256': digest(ROOT / 'train_encoded.py'),
        'train_sha256': digest(ROOT / 'train.csv'), 'test_sha256': digest(ROOT / 'test.csv'),
        'fold_sha256': digest(fold_path), 'fold_seed': 42, 'n_splits': 5,
        'tabm_version': tabm.__version__, 'torch_version': torch.__version__,
        'numeric_features': num_columns, 'categorical_features': cat_columns,
        'target_encoding_features': key_columns, 'target_encoding_smoothing': 10.0,
        'categorical_values': category_values, 'raw_category_mappings': mappings,
        'frequency_scope': 'unlabeled train+test, inherited baseline', 'submitted': False,
    })
    config_path = out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Use a new run ID after changing configuration'
    else:
        config_path.write_text(json.dumps(config, indent=2) + '\n')
    log({'event': 'start', 'run': args.run, 'n_numeric': len(num_columns) + len(key_columns),
         'n_categorical': len(cat_columns), 'folds_requested': args.folds})
    for fold in args.folds:
        checkpoint_path = out / f'fold_{fold}.npz'
        if checkpoint_path.exists():
            log({'event': 'resumed', 'fold': fold})
            continue
        fold_start = time.time()
        seed = args.seed + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ti, vi = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        encoder = TargetEncoder(target_type='binary', smooth=10.0, cv=5,
                                shuffle=True, random_state=42)
        A = np.column_stack([numeric[ti], encoder.fit_transform(key[ti], y[ti]).astype('float32')])
        B = np.column_stack([numeric[vi], encoder.transform(key[vi]).astype('float32')])
        C = np.column_stack([numeric[n:], encoder.transform(key[n:]).astype('float32')])
        active = np.ptp(A, axis=0) > 0
        scaler = StandardScaler().fit(A[:, active])
        A, B, C = [scaler.transform(x[:, active]).astype('float32') for x in (A, B, C)]
        bins = rtdl_num_embeddings.compute_bins(torch.from_numpy(A), n_bins=args.bins)
        state = {'encoder': encoder, 'scaler': scaler, 'active_columns': active,
                 'bins': [b.numpy() for b in bins], 'cat_cardinalities': cardinalities,
                 'config': config}
        joblib.dump(state, out / f'preprocessor_{fold}.joblib')
        model = create_model(A.shape[1], cardinalities, bins, args).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        tensors = [torch.as_tensor(x, device='cuda') for x in (A, B, C, cat[ti], cat[vi], cat[n:])]
        train_num, valid_num, test_num, train_cat, valid_cat, test_cat = tensors
        labels = torch.as_tensor(y[ti], device='cuda')
        best_auc, best_epoch, best_state = -np.inf, -1, None
        epochs = []
        log({'event': 'prepared', 'fold': fold, 'seconds': time.time() - fold_start,
             'parameters': sum(p.numel() for p in model.parameters())})
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            order = torch.randperm(len(ti), device='cuda')
            losses = []
            for indices in order.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(train_num[indices], train_cat[indices]).squeeze(-1)
                    # Each ensemble member learns the label independently.
                    loss = F.binary_cross_entropy_with_logits(logits.float(), labels[indices, None].expand_as(logits))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(loss.detach())
            vp = predict(model, valid_num, valid_cat)
            auc = float(roc_auc_score(y[vi], vp))
            record = {'fold': fold, 'epoch': epoch, 'auc': auc,
                      'loss': float(torch.stack(losses).mean()), 'seconds': time.time() - epoch_start}
            epochs.append(record)
            log(record)
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch >= args.patience:
                break
        model.load_state_dict(best_state)
        model_path = out / f'model_{fold}.pt'
        torch.save(best_state, model_path)
        vp, tp = predict(model, valid_num, valid_cat), predict(model, test_num, test_cat)
        assert np.isfinite(vp).all() and np.isfinite(tp).all()
        # Reconstruct the model and preprocessing from disk to verify the artifact handoff.
        reloaded = joblib.load(out / f'preprocessor_{fold}.joblib')
        raw_replay = np.column_stack([numeric[n:n + 1024],
            reloaded['encoder'].transform(key[n:n + 1024]).astype('float32')])
        replay_num = reloaded['scaler'].transform(raw_replay[:, reloaded['active_columns']]).astype('float32')
        np.testing.assert_allclose(replay_num, C[:1024], atol=1e-6)
        replay_model = create_model(A.shape[1], reloaded['cat_cardinalities'],
                                   [torch.from_numpy(b) for b in reloaded['bins']], args).cuda()
        replay_model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        replay = predict(replay_model, torch.as_tensor(replay_num, device='cuda'), test_cat[:1024])
        np.testing.assert_allclose(replay, tp[:1024], rtol=2e-4, atol=2e-6)
        record = {'fold': fold, 'auc': float(roc_auc_score(y[vi], vp)), 'best_epoch': best_epoch,
                  'epochs_completed': len(epochs), 'seconds': time.time() - fold_start,
                  'model_sha256': digest(model_path), 'reload_checked_rows': 1024}
        np.savez_compressed(checkpoint_path, valid_indices=vi, valid_prediction=vp, test_prediction=tp)
        (out / f'fold_{fold}.json').write_text(json.dumps(record, indent=2) + '\n')
        (out / f'epochs_{fold}.json').write_text(json.dumps(epochs, indent=2) + '\n')
        log(record)
        del model, replay_model, optimizer, tensors, train_num, valid_num, test_num
        del train_cat, valid_cat, test_cat, labels, A, B, C, best_state
        gc.collect()
        torch.cuda.empty_cache()
    available_folds = [f for f in range(5) if (out / f'fold_{f}.npz').exists()]
    records = []
    oof = np.full(n, np.nan)
    pred = np.zeros(len(te))
    for fold in available_folds:
        vi = np.flatnonzero(fold_ids == fold)
        saved = np.load(out / f'fold_{fold}.npz')
        assert np.array_equal(vi, saved['valid_indices'])
        oof[vi] = saved['valid_prediction']
        pred += saved['test_prediction'] / len(available_folds)
        records.append(json.loads((out / f'fold_{fold}.json').read_text()))
    complete = len(available_folds) == 5
    metrics = {'run': args.run, 'status': 'complete' if complete else 'pilot', 'submitted': False,
               'folds': records, 'folds_completed': available_folds,
               'elapsed_seconds_this_invocation': time.time() - start_time}
    if complete:
        assert np.isfinite(oof).all() and np.isfinite(pred).all()
        metrics.update({'oof_auc': float(roc_auc_score(y, oof)),
                        'fold_mean': float(np.mean([r['auc'] for r in records])),
                        'fold_std': float(np.std([r['auc'] for r in records]))})
        np.save(out / 'oof.npy', oof)
        np.save(out / 'test.npy', pred)
        submission = pd.read_csv(ROOT / 'sample_submission.csv')
        assert submission.id.equals(te.id) and ((pred >= 0) & (pred <= 1)).all()
        submission[TARGET] = pred
        submission.to_csv(out / 'submission.csv', index=False)
        metrics['submission_sha256'] = digest(out / 'submission.csv')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    log(metrics)


if __name__ == '__main__':
    main()

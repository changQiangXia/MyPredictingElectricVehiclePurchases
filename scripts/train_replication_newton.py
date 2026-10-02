#!/usr/bin/env python
# coding: utf-8

# # Replication-Aware Newton Boosting
# 
# This notebook tests a simple idea:
# 
# > Prefer splits that remain strong across multiple random subgroups of the training data.
# 
# Standard gradient boosting usually chooses splits mainly from their performance on the full training node.
# 
# Here, I first find strong split candidates using Newton gain on the full node. The same candidates are then evaluated separately across 16 random, disjoint subgroups.
# 
# The goal is to prefer splits whose behavior is more consistent across different parts of the training data.
# 
# ## How it works
# 
# For each tree node:
# 
# 1. Generate numerical and categorical split candidates.
# 2. Rank the strongest candidates globally using Newton gain.
# 3. Keep the top candidate shortlist.
# 4. Evaluate those same candidates separately in 16 random subgroups.
# 5. Rank candidates by Newton gain inside each subgroup.
# 6. Measure how often the Newton direction agrees with the global direction.
# 7. Prefer candidates with strong agreement, good subgroup ranks and stable performance across groups.
# 
# The replication score combines:
# 
# * directional agreement across subgroups
# * average subgroup rank
# * the fraction of subgroups where the split is valid
# * a penalty for unstable subgroup ranks
# 
# In this experiment, the final split choice is based entirely on this replication score among the globally shortlisted candidates.
# 
# ## Newton Boosting
# 
# For binary classification, the current predictions are converted into gradients and Hessians:
# 
# $$
# g_i = p_i - y_i
# $$
# 
# $$
# h_i = p_i(1-p_i)
# $$
# 
# Candidate splits are first evaluated using the usual second-order Newton gain.
# 
# After a tree is built, each leaf receives a Newton update and the tree is added to the ensemble with a learning rate.
# 
# ## Features
# 
# The feature set includes:
# 
# * original numerical features
# * original categorical features
# * digit features
# * a few simple domain flags
# * frequency encodings
# * fold-safe target encodings
# 
# The numerical candidate bank uses up to 64 quantile-based thresholds per feature where possible.
# 
# Categorical splits are tested as one-category-vs-rest splits.
# 
# ## Validation
# 
# I use 5-fold stratified cross-validation.
# 
# For each fold, all fold-dependent preprocessing is fit only on that fold's training rows. This includes medians, frequency encodings, target encodings, categorical mappings and the split candidate bank.
# 
# The held-out fold is used to measure ROC AUC and select the best boosting round with early stopping.
# 
# After all five folds, I report the individual fold scores and the full out-of-fold ROC AUC.
# 
# ## Submission
# 
# Each fold model also predicts the competition test set using the boosting round selected on that fold's validation data.
# 
# The five test prediction vectors are averaged to produce the final prediction.
# 
# At the end of the notebook:
# 
# `submission.csv`
# 
# is created using the competition's sample submission format and can be submitted directly to Kaggle.
# 
# The notebook also saves:
# 
# `oof_predictions.csv`
# 
# for the complete out-of-fold predictions.
# 
# ## Idea
# 
# The motivation is simple.
# 
# A split can look strong on the full training node because of noise or a pattern that is concentrated in only one part of the data.
# 
# If the same split also performs well across many separate random groups, that provides an additional signal that the pattern may be more stable.
# 
# This does not prove that the split is truly generalizable.
# 
# It is simply used here as an additional criterion for split selection.
# 

# In[ ]:


import gc
import math
import random
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")

TARGET = "Will_Buy_EV"

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = str(ROOT / "train.csv")
TEST_PATH = str(ROOT / "test.csv")
SAMPLE_SUB_PATH = str(ROOT / "sample_submission.csv")

SEED = 21
N_FOLDS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_BOOST_ROUNDS = 600
LEARNING_RATE = 0.08
MAX_DEPTH = 6
EARLY_STOPPING_PATIENCE = 75

MIN_NODE_SIZE = 1800
L2_LEAF = 4.0
MIN_CHILD_HESSIAN = 20.0
MIN_GLOBAL_NEWTON_GAIN = 1e-5
MAX_LEAF_VALUE = 3.0

N_THRESHOLDS_PER_NUMERIC = 64
TOP_GLOBAL_CANDIDATES = 48
MIN_GLOBAL_GAIN_RATIO = 0.05
GLOBAL_CANDIDATE_CHUNK = 256

N_REPLICATION_SUBSETS = 16
MIN_SUBSET_ROWS = 120
MIN_SUBSET_CHILD_ROWS = 20
MIN_SUBSET_CHILD_HESSIAN = 2.0

REPLICATION_WEIGHT = 1.0

GLOBAL_GAIN_RATIO_MIX = 0.70
GLOBAL_GAIN_RANK_MIX = 0.30

REP_AGREEMENT_MIX = 0.55
REP_MEAN_RANK_MIX = 0.35
REP_VALIDITY_MIX = 0.10
REP_RANK_STD_PENALTY = 0.15

MIN_VALID_SUBSET_FRACTION_SOFT = 0.50
MIN_DIRECTION_AGREEMENT_FRACTION_SOFT = 0.25

RAW_NUMERICALS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

RAW_CATS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

TE_SMOOTHS = ["auto", 10.0]
EPS = 1e-7


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit_scalar(p):
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return math.log(p / (1.0 - p))


def fit_medians(df):
    return {col: float(pd.to_numeric(df[col], errors='coerce').median()) for col in RAW_NUMERICALS}


def clean_frame(df, medians):
    out = df.copy()
    for col in RAW_NUMERICALS:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(medians[col]).astype(np.float32)
    for col in RAW_CATS:
        out[col] = out[col].fillna('__MISSING__').astype(str)
    return out


def make_base_work(df):
    out = df[RAW_NUMERICALS + RAW_CATS].copy()
    digit_cols = []
    for col in RAW_NUMERICALS:
        values = out[col].values.astype(np.float64)
        for k in range(-4, 4):
            name = f'{col}_digit_{k}'
            out[name] = (np.floor(values / 10.0 ** k) % 10).astype(np.float32)
            digit_cols.append(name)
    out['is_30k_spike'] = (out['Annual_Income_USD'] == 30000).astype(np.float32)
    out['is_millionaire_cliff'] = (out['Annual_Income_USD'] >= 170537).astype(np.float32)
    out['is_dead_zone'] = ((out['Annual_Income_USD'] >= 38000) & (out['Annual_Income_USD'] <= 42000)).astype(np.float32)
    out['is_env_hater'] = (out['Environmental_Concern_Level'] == 1).astype(np.float32)
    domain_cols = ['is_30k_spike', 'is_millionaire_cliff', 'is_dead_zone', 'is_env_hater']
    return (out, digit_cols, domain_cols)


def make_encoding_sources(work, digit_cols):
    out = pd.DataFrame(index=work.index)
    for col in RAW_CATS:
        out[col] = work[col].astype(str)
    for col in RAW_NUMERICALS:
        out[f'{col}_as_cat'] = work[col].astype(str)
    for col in digit_cols:
        out[f'{col}_as_cat'] = work[col].astype(str)
    return out


def build_strong_features(train_df, val_df, test_df, y_train, fold_seed):
    medians = fit_medians(train_df)
    tr = clean_frame(train_df, medians)
    va = clean_frame(val_df, medians)
    te = clean_frame(test_df, medians)
    work_tr, digit_cols, domain_cols = make_base_work(tr)
    work_va, _, _ = make_base_work(va)
    work_te, _, _ = make_base_work(te)
    num_tr = []
    num_va = []
    num_te = []
    numerical_names = []
    direct_cols = RAW_NUMERICALS + digit_cols + domain_cols
    for col in direct_cols:
        num_tr.append(work_tr[col].values.astype(np.float32))
        num_va.append(work_va[col].values.astype(np.float32))
        num_te.append(work_te[col].values.astype(np.float32))
        numerical_names.append(col)
    source_tr = make_encoding_sources(work_tr, digit_cols)
    source_va = make_encoding_sources(work_va, digit_cols)
    source_te = make_encoding_sources(work_te, digit_cols)
    print('Encoding sources:', source_tr.shape[1])
    for col in source_tr.columns:
        mapping = source_tr[col].value_counts(normalize=True).to_dict()
        num_tr.append(source_tr[col].map(mapping).fillna(0).values.astype(np.float32))
        num_va.append(source_va[col].map(mapping).fillna(0).values.astype(np.float32))
        num_te.append(source_te[col].map(mapping).fillna(0).values.astype(np.float32))
        numerical_names.append(f'{col}_freq')
    for smooth in TE_SMOOTHS:
        encoder = TargetEncoder(target_type='binary', smooth=smooth, cv=5, shuffle=True, random_state=fold_seed)
        enc_tr = encoder.fit_transform(source_tr, y_train).astype(np.float32)
        enc_va = encoder.transform(source_va).astype(np.float32)
        enc_te = encoder.transform(source_te).astype(np.float32)
        smooth_name = 'auto' if smooth == 'auto' else str(smooth)
        for j, col in enumerate(source_tr.columns):
            num_tr.append(enc_tr[:, j])
            num_va.append(enc_va[:, j])
            num_te.append(enc_te[:, j])
            numerical_names.append(f'{col}_te_{smooth_name}')
    X_num_train = np.column_stack(num_tr).astype(np.float32)
    X_num_val = np.column_stack(num_va).astype(np.float32)
    X_num_test = np.column_stack(num_te).astype(np.float32)
    keep = np.std(X_num_train, axis=0) > 1e-12
    X_num_train = X_num_train[:, keep]
    X_num_val = X_num_val[:, keep]
    X_num_test = X_num_test[:, keep]
    numerical_names = [name for name, flag in zip(numerical_names, keep) if flag]
    print('Numerical features retained:', len(numerical_names))
    cat_tr = []
    cat_va = []
    cat_te = []
    for col in RAW_CATS:
        categories = pd.Index(tr[col].astype(str).unique()).tolist()
        mapping = {value: j for j, value in enumerate(categories)}
        cat_tr.append(tr[col].astype(str).map(mapping).fillna(-1).values.astype(np.int16))
        cat_va.append(va[col].astype(str).map(mapping).fillna(-1).values.astype(np.int16))
        cat_te.append(te[col].astype(str).map(mapping).fillna(-1).values.astype(np.int16))
    return {'X_num_train': np.nan_to_num(X_num_train), 'X_num_val': np.nan_to_num(X_num_val), 'X_num_test': np.nan_to_num(X_num_test), 'X_cat_train': np.column_stack(cat_tr).astype(np.int16), 'X_cat_val': np.column_stack(cat_va).astype(np.int16), 'X_cat_test': np.column_stack(cat_te).astype(np.int16), 'numerical_names': numerical_names, 'categorical_names': RAW_CATS.copy()}


def build_candidate_bank(X_num_train, X_cat_train):
    num_feature_idx = []
    num_threshold = []
    q = np.linspace(0.03, 0.97, N_THRESHOLDS_PER_NUMERIC)
    for j in range(X_num_train.shape[1]):
        values = X_num_train[:, j]
        unique = np.unique(values)
        if len(unique) <= 1:
            continue
        thresholds = np.quantile(values, q)
        thresholds = np.unique(thresholds)
        maximum = np.max(values)
        thresholds = thresholds[thresholds < maximum]
        for threshold in thresholds:
            num_feature_idx.append(j)
            num_threshold.append(float(threshold))
    cat_feature_idx = []
    cat_value = []
    for j in range(X_cat_train.shape[1]):
        categories = np.unique(X_cat_train[:, j])
        categories = categories[categories >= 0]
        for category in categories:
            cat_feature_idx.append(j)
            cat_value.append(int(category))
    return {'num_feature_idx': np.asarray(num_feature_idx, dtype=np.int64), 'num_threshold': np.asarray(num_threshold, dtype=np.float32), 'cat_feature_idx': np.asarray(cat_feature_idx, dtype=np.int64), 'cat_value': np.asarray(cat_value, dtype=np.int16)}


def make_replication_groups(n_rows, seed):
    row = np.arange(n_rows, dtype=np.uint64)
    x = row + np.uint64(seed * 2654435761)
    x ^= x >> np.uint64(16)
    x *= np.uint64(2246822507)
    x ^= x >> np.uint64(13)
    x *= np.uint64(3266489909)
    x ^= x >> np.uint64(16)
    return (x % np.uint64(N_REPLICATION_SUBSETS)).astype(np.int8)


@dataclass
class Candidate:
    feature_type: int
    feature_idx: int
    value: float
    global_gain: float
    global_direction: float


@dataclass
class Node:
    is_leaf: bool = True
    value: float = 0.0
    feature_type: int = -1
    feature_idx: int = -1
    threshold: float = 0.0
    category: int = -1
    left: object = None
    right: object = None
    depth: int = 0


def newton_leaf_weight(G, H):
    weight = -G / (H + L2_LEAF)
    return torch.clamp(weight, -MAX_LEAF_VALUE, MAX_LEAF_VALUE)


def newton_gain(G_left, H_left, G_parent, H_parent):
    G_right = G_parent - G_left
    H_right = H_parent - H_left
    left_term = G_left.pow(2) / (H_left + L2_LEAF)
    right_term = G_right.pow(2) / (H_right + L2_LEAF)
    parent_term = G_parent.pow(2) / (H_parent + L2_LEAF)
    gain = 0.5 * (left_term + right_term - parent_term)
    w_left = newton_leaf_weight(G_left, H_left)
    w_right = newton_leaf_weight(G_right, H_right)
    direction = w_right - w_left
    return (gain, direction, H_right)


def global_top_candidates(X_num, X_cat, gradients, hessians, node_idx, bank):
    idx = torch.as_tensor(node_idx, dtype=torch.long, device=DEVICE)
    g = gradients[idx]
    h = hessians[idx]
    G_parent = g.sum()
    H_parent = h.sum()
    if float(H_parent.item()) < 2.0 * MIN_CHILD_HESSIAN:
        return []
    candidates = []
    n_num = len(bank['num_feature_idx'])
    for start in range(0, n_num, GLOBAL_CANDIDATE_CHUNK):
        end = min(start + GLOBAL_CANDIDATE_CHUNK, n_num)
        f_idx = torch.as_tensor(bank['num_feature_idx'][start:end], dtype=torch.long, device=DEVICE)
        threshold = torch.as_tensor(bank['num_threshold'][start:end], dtype=torch.float32, device=DEVICE)
        values = X_num[idx[:, None], f_idx[None, :]]
        mask = values <= threshold[None, :]
        n_left = mask.sum(dim=0)
        n_right = len(node_idx) - n_left
        G_left = (mask.float() * g[:, None]).sum(dim=0)
        H_left = (mask.float() * h[:, None]).sum(dim=0)
        gain, direction, H_right = newton_gain(G_left, H_left, G_parent, H_parent)
        valid = (n_left >= 50) & (n_right >= 50) & (H_left >= MIN_CHILD_HESSIAN) & (H_right >= MIN_CHILD_HESSIAN) & (gain >= MIN_GLOBAL_NEWTON_GAIN)
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        for local in valid_idx:
            candidates.append(Candidate(feature_type=0, feature_idx=int(f_idx[local].item()), value=float(threshold[local].item()), global_gain=float(gain[local].item()), global_direction=float(direction[local].item())))
    n_cat = len(bank['cat_feature_idx'])
    for start in range(0, n_cat, GLOBAL_CANDIDATE_CHUNK):
        end = min(start + GLOBAL_CANDIDATE_CHUNK, n_cat)
        f_idx = torch.as_tensor(bank['cat_feature_idx'][start:end], dtype=torch.long, device=DEVICE)
        category = torch.as_tensor(bank['cat_value'][start:end], dtype=torch.int16, device=DEVICE)
        values = X_cat[idx[:, None], f_idx[None, :]]
        mask = values == category[None, :]
        n_left = mask.sum(dim=0)
        n_right = len(node_idx) - n_left
        G_left = (mask.float() * g[:, None]).sum(dim=0)
        H_left = (mask.float() * h[:, None]).sum(dim=0)
        gain, direction, H_right = newton_gain(G_left, H_left, G_parent, H_parent)
        valid = (n_left >= 50) & (n_right >= 50) & (H_left >= MIN_CHILD_HESSIAN) & (H_right >= MIN_CHILD_HESSIAN) & (gain >= MIN_GLOBAL_NEWTON_GAIN)
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        for local in valid_idx:
            candidates.append(Candidate(feature_type=1, feature_idx=int(f_idx[local].item()), value=int(category[local].item()), global_gain=float(gain[local].item()), global_direction=float(direction[local].item())))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x.global_gain, reverse=True)
    best_gain = candidates[0].global_gain
    candidates = [candidate for candidate in candidates if candidate.global_gain >= MIN_GLOBAL_GAIN_RATIO * best_gain]
    return candidates[:TOP_GLOBAL_CANDIDATES]


def candidate_mask_matrix(candidates, X_num, X_cat, node_idx):
    idx = torch.as_tensor(node_idx, dtype=torch.long, device=DEVICE)
    masks = []
    for candidate in candidates:
        if candidate.feature_type == 0:
            mask = X_num[idx, candidate.feature_idx] <= candidate.value
        else:
            mask = X_cat[idx, candidate.feature_idx] == int(candidate.value)
        masks.append(mask)
    return torch.stack(masks, dim=1)


def soft_replication_newton_ranking(candidates, X_num, X_cat, gradients, hessians, node_idx, replication_groups):
    if not candidates:
        return None
    C = len(candidates)
    K = N_REPLICATION_SUBSETS
    idx = torch.as_tensor(node_idx, dtype=torch.long, device=DEVICE)
    g_node = gradients[idx]
    h_node = hessians[idx]
    subset_ids = torch.as_tensor(replication_groups[node_idx], dtype=torch.long, device=DEVICE)
    split_masks = candidate_mask_matrix(candidates, X_num, X_cat, node_idx)
    gains = torch.full((K, C), float('nan'), dtype=torch.float32, device=DEVICE)
    directions = torch.full_like(gains, float('nan'))
    valid = torch.zeros((K, C), dtype=torch.bool, device=DEVICE)
    for subset in range(K):
        row_mask = subset_ids == subset
        n_rows = int(row_mask.sum().item())
        if n_rows < MIN_SUBSET_ROWS:
            continue
        g = g_node[row_mask]
        h = h_node[row_mask]
        masks = split_masks[row_mask]
        G_parent = g.sum()
        H_parent = h.sum()
        n_left = masks.sum(dim=0)
        n_right = n_rows - n_left
        G_left = (masks.float() * g[:, None]).sum(dim=0)
        H_left = (masks.float() * h[:, None]).sum(dim=0)
        gain, direction, H_right = newton_gain(G_left, H_left, G_parent, H_parent)
        subset_valid = (n_left >= MIN_SUBSET_CHILD_ROWS) & (n_right >= MIN_SUBSET_CHILD_ROWS) & (H_left >= MIN_SUBSET_CHILD_HESSIAN) & (H_right >= MIN_SUBSET_CHILD_HESSIAN) & torch.isfinite(gain) & torch.isfinite(direction)
        gains[subset] = gain
        directions[subset] = direction
        valid[subset] = subset_valid
    ranks = torch.full((K, C), float(C + 1), dtype=torch.float32, device=DEVICE)
    for subset in range(K):
        valid_indices = torch.nonzero(valid[subset], as_tuple=False).flatten()
        if len(valid_indices) == 0:
            continue
        order = torch.argsort(gains[subset, valid_indices], descending=True)
        ranked_indices = valid_indices[order]
        ranks[subset, ranked_indices] = torch.arange(1, len(ranked_indices) + 1, dtype=torch.float32, device=DEVICE)
    valid_count = valid.sum(dim=0).to(torch.int32)
    valid_count_float = valid_count.float().clamp_min(1.0)
    valid_fraction = valid_count.float() / float(K)
    masked_rank_sum = torch.where(valid, ranks, torch.zeros_like(ranks)).sum(dim=0)
    mean_rank = masked_rank_sum / valid_count_float
    centered = ranks - mean_rank.unsqueeze(0)
    rank_var = torch.where(valid, centered.pow(2), torch.zeros_like(centered)).sum(dim=0) / valid_count_float
    std_rank = torch.sqrt(rank_var)
    global_direction = torch.tensor([c.global_direction for c in candidates], dtype=torch.float32, device=DEVICE)
    expected_sign = torch.sign(global_direction)
    sign_match = torch.sign(directions) == expected_sign.unsqueeze(0)
    sign_match &= valid
    agreement_count = sign_match.sum(dim=0).to(torch.int32)
    agreement_fraction = agreement_count.float() / float(K)
    global_gains = torch.tensor([c.global_gain for c in candidates], dtype=torch.float32, device=DEVICE)
    best_global_gain = global_gains.max().clamp_min(1e-12)
    gain_ratio = torch.clamp(global_gains / best_global_gain, 0.0, 1.0)
    global_order = torch.argsort(global_gains, descending=True)
    global_rank = torch.empty(C, dtype=torch.float32, device=DEVICE)
    global_rank[global_order] = torch.arange(1, C + 1, dtype=torch.float32, device=DEVICE)
    if C <= 1:
        global_rank_strength = torch.ones_like(global_rank)
    else:
        global_rank_strength = 1.0 - (global_rank - 1.0) / float(C - 1)
    global_score = GLOBAL_GAIN_RATIO_MIX * gain_ratio + GLOBAL_GAIN_RANK_MIX * global_rank_strength
    if C <= 1:
        mean_rank_strength = torch.ones_like(mean_rank)
        rank_std_normalized = torch.zeros_like(std_rank)
    else:
        mean_rank_strength = torch.clamp(1.0 - (mean_rank - 1.0) / float(C - 1), 0.0, 1.0)
        rank_std_normalized = torch.clamp(std_rank / float(C - 1), 0.0, 1.0)
    replication_score = REP_AGREEMENT_MIX * agreement_fraction + REP_MEAN_RANK_MIX * mean_rank_strength + REP_VALIDITY_MIX * valid_fraction - REP_RANK_STD_PENALTY * rank_std_normalized
    replication_score = torch.clamp(replication_score, 0.0, 1.0)
    min_valid_count = max(1, int(math.ceil(K * MIN_VALID_SUBSET_FRACTION_SOFT)))
    min_agreement_count = max(0, int(math.ceil(K * MIN_DIRECTION_AGREEMENT_FRACTION_SOFT)))
    eligible = (valid_count >= min_valid_count) & (agreement_count >= min_agreement_count) & torch.isfinite(global_score) & torch.isfinite(replication_score)
    if not bool(eligible.any()):
        return None
    final_score = (1.0 - REPLICATION_WEIGHT) * global_score + REPLICATION_WEIGHT * replication_score
    final_score = torch.where(eligible, final_score, torch.full_like(final_score, float('-inf')))
    winner = int(torch.argmax(final_score).item())
    return {
        'candidate': candidates[winner],
        'agreement_count': int(agreement_count[winner].item()),
        'agreement_fraction': float(agreement_fraction[winner].item()),
    }


class SoftReplicationNewtonTree:

    def __init__(self, tree_id, seed, candidate_bank, numerical_names, categorical_names):
        self.tree_id = tree_id
        self.seed = seed
        self.candidate_bank = candidate_bank
        self.numerical_names = numerical_names
        self.categorical_names = categorical_names
        self.root = None
        self.n_splits = 0
        self.consensus_counts = {}

    def fit(self, X_num, X_cat, gradients, hessians):
        self.X_num = X_num
        self.X_cat = X_cat
        self.gradients = gradients
        self.hessians = hessians
        self.replication_groups = make_replication_groups(X_num.shape[0], self.seed)
        indices = np.arange(X_num.shape[0], dtype=np.int64)
        self.root = self._grow(indices, depth=0)
        return self

    def _grow(self, node_idx, depth):
        idx_t = torch.as_tensor(node_idx, dtype=torch.long, device=DEVICE)
        G = self.gradients[idx_t].sum()
        H = self.hessians[idx_t].sum()
        leaf_value = float(newton_leaf_weight(G, H).item())
        node = Node(is_leaf=True, value=leaf_value, depth=depth)
        if depth >= MAX_DEPTH:
            return node
        if len(node_idx) < MIN_NODE_SIZE:
            return node
        if float(H.item()) < 2.0 * MIN_CHILD_HESSIAN:
            return node
        candidates = global_top_candidates(self.X_num, self.X_cat, self.gradients, self.hessians, node_idx, self.candidate_bank)
        if not candidates:
            return node
        replication = soft_replication_newton_ranking(candidates, self.X_num, self.X_cat, self.gradients, self.hessians, node_idx, self.replication_groups)
        if replication is None:
            return node
        winner = replication['candidate']
        if winner.feature_type == 0:
            left_gpu = self.X_num[idx_t, winner.feature_idx] <= winner.value
        else:
            left_gpu = self.X_cat[idx_t, winner.feature_idx] == int(winner.value)
        left_mask = left_gpu.detach().cpu().numpy()
        left_idx = node_idx[left_mask]
        right_idx = node_idx[~left_mask]
        if len(left_idx) < 50 or len(right_idx) < 50:
            return node
        node.is_leaf = False
        node.feature_type = winner.feature_type
        node.feature_idx = winner.feature_idx
        if winner.feature_type == 0:
            node.threshold = winner.value
            feature_name = self.numerical_names[winner.feature_idx]
            split_text = f'{feature_name} <= {winner.value:.7g}'
        else:
            node.category = int(winner.value)
            feature_name = self.categorical_names[winner.feature_idx]
            split_text = f'{feature_name} == {int(winner.value)}'
        self.n_splits += 1
        consensus_key = f"{replication['agreement_count']}/{N_REPLICATION_SUBSETS}"
        self.consensus_counts[consensus_key] = self.consensus_counts.get(consensus_key, 0) + 1
        node.left = self._grow(left_idx, depth + 1)
        node.right = self._grow(right_idx, depth + 1)
        return node

    def predict(self, X_num, X_cat):
        output = np.empty(X_num.shape[0], dtype=np.float32)
        stack = [(self.root, np.arange(X_num.shape[0], dtype=np.int64))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node.is_leaf:
                output[idx] = node.value
                continue
            if node.feature_type == 0:
                left = X_num[idx, node.feature_idx] <= node.threshold
            else:
                left = X_cat[idx, node.feature_idx] == node.category
            stack.append((node.right, idx[~left]))
            stack.append((node.left, idx[left]))
        return output


def train_model(features, y_train, y_valid, fold):
    bank = build_candidate_bank(
        features["X_num_train"],
        features["X_cat_train"],
    )

    print(f"Numeric thresholds: {len(bank['num_threshold'])}")
    print(f"Categorical candidates: {len(bank['cat_value'])}")

    X_num_gpu = torch.tensor(
        features["X_num_train"],
        dtype=torch.float32,
        device=DEVICE,
    )

    X_cat_gpu = torch.tensor(
        features["X_cat_train"],
        dtype=torch.int16,
        device=DEVICE,
    )

    y_gpu = torch.tensor(
        y_train,
        dtype=torch.float32,
        device=DEVICE,
    )

    base_logit = logit_scalar(float(y_train.mean()))

    train_logit = torch.full(
        (len(y_train),),
        base_logit,
        dtype=torch.float32,
        device=DEVICE,
    )

    valid_logit = np.full(
        len(y_valid),
        base_logit,
        dtype=np.float64,
    )

    test_logit = np.full(
        features["X_num_test"].shape[0],
        base_logit,
        dtype=np.float64,
    )

    best_auc = -np.inf
    best_round = 0
    best_valid_logit = valid_logit.copy()
    best_test_logit = test_logit.copy()
    stale = 0

    for boosting_round in range(1, N_BOOST_ROUNDS + 1):
        probability = torch.sigmoid(train_logit)
        gradients = probability - y_gpu
        hessians = (probability * (1.0 - probability)).clamp_min(1e-6)

        tree = SoftReplicationNewtonTree(
            tree_id=boosting_round,
            seed=SEED + fold * 100000 + boosting_round * 977,
            candidate_bank=bank,
            numerical_names=features["numerical_names"],
            categorical_names=features["categorical_names"],
        )

        tree.fit(
            X_num_gpu,
            X_cat_gpu,
            gradients,
            hessians,
        )

        train_tree = tree.predict(
            features["X_num_train"],
            features["X_cat_train"],
        )

        valid_tree = tree.predict(
            features["X_num_val"],
            features["X_cat_val"],
        )

        test_tree = tree.predict(
            features["X_num_test"],
            features["X_cat_test"],
        )

        train_logit += LEARNING_RATE * torch.as_tensor(
            train_tree,
            dtype=torch.float32,
            device=DEVICE,
        )

        valid_logit += LEARNING_RATE * valid_tree
        test_logit += LEARNING_RATE * test_tree

        valid_pred = sigmoid_np(valid_logit)
        valid_auc = roc_auc_score(y_valid, valid_pred)

        if tree.consensus_counts:
            agreement = ", ".join(
                f"{key}:{value}"
                for key, value in sorted(
                    tree.consensus_counts.items(),
                    key=lambda x: -int(x[0].split("/")[0]),
                )
            )
        else:
            agreement = "none"

        print(
            f"Round {boosting_round:03d}/{N_BOOST_ROUNDS} | "
            f"splits={tree.n_splits:2d} | "
            f"Val AUC={valid_auc:.7f} | "
            f"agreement=[{agreement}]"
        )

        if valid_auc > best_auc + 1e-8:
            best_auc = valid_auc
            best_round = boosting_round
            best_valid_logit = valid_logit.copy()
            best_test_logit = test_logit.copy()
            stale = 0
        else:
            stale += 1

        del tree, train_tree, valid_tree, test_tree
        gc.collect()

        if stale >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping. Best round: {best_round}")
            break

    del X_num_gpu, X_cat_gpu, y_gpu, train_logit
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        sigmoid_np(best_valid_logit),
        sigmoid_np(best_test_logit),
        best_auc,
        best_round,
    )


seed_everything(SEED)

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)

if train_raw[TARGET].dtype == object:
    y = (
        train_raw[TARGET]
        .map({"No": 0, "Yes": 1})
        .values
        .astype(np.float32)
    )
else:
    y = train_raw[TARGET].values.astype(np.float32)

print(f"Train rows: {len(train_raw):,}")
print(f"Test rows: {len(test_raw):,}")
print(f"Positive rate: {y.mean():.6f}")
print(f"Folds: {N_FOLDS}")

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

oof_pred = np.zeros(len(train_raw), dtype=np.float64)
test_pred_sum = np.zeros(len(test_raw), dtype=np.float64)

fold_scores = []
fold_best_rounds = []

for fold, (train_idx, valid_idx) in enumerate(
    skf.split(train_raw, y),
    start=1,
):
    print()
    print("=" * 100)
    print(f"FOLD {fold}/{N_FOLDS}")
    print("=" * 100)

    train_df = train_raw.iloc[train_idx].reset_index(drop=True)
    valid_df = train_raw.iloc[valid_idx].reset_index(drop=True)
    test_df = test_raw.reset_index(drop=True)

    y_train = y[train_idx]
    y_valid = y[valid_idx]

    print(f"Train rows: {len(train_idx):,}")
    print(f"Validation rows: {len(valid_idx):,}")

    # Everything is fit only on this fold's training rows:
    # medians, frequencies, TargetEncoders, category mappings, and candidate bank.
    features = build_strong_features(
        train_df=train_df,
        val_df=valid_df,
        test_df=test_df,
        y_train=y_train,
        fold_seed=SEED + fold,
    )

    print(f"Numerical features: {features['X_num_train'].shape[1]}")
    print(f"Categorical features: {features['X_cat_train'].shape[1]}")

    valid_pred, fold_test_pred, best_auc, best_round = train_model(
        features,
        y_train,
        y_valid,
        fold=fold,
    )

    oof_pred[valid_idx] = valid_pred
    test_pred_sum += fold_test_pred

    fold_scores.append(best_auc)
    fold_best_rounds.append(best_round)

    print()
    print(f"Fold {fold} best AUC: {best_auc:.8f}")
    print(f"Fold {fold} best round: {best_round}")

    del features, train_df, valid_df, test_df
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

test_pred = test_pred_sum / N_FOLDS
oof_auc = roc_auc_score(y, oof_pred)

print()
print("=" * 100)
print("5-FOLD RESULT")
print("=" * 100)

for fold, (score, best_round) in enumerate(
    zip(fold_scores, fold_best_rounds),
    start=1,
):
    print(
        f"Fold {fold}: "
        f"AUC={score:.8f} | "
        f"best_round={best_round}"
    )

print("-" * 100)
print(f"Mean fold AUC: {np.mean(fold_scores):.8f}")
print(f"Std fold AUC : {np.std(fold_scores):.8f}")
print(f"OOF AUC      : {oof_auc:.8f}")
print("=" * 100)

# Optional OOF file: useful for later blending / diagnostics.
oof_out = pd.DataFrame({
    "row_index": np.arange(len(train_raw), dtype=np.int64),
    "target": y.astype(np.int8),
    "replication_newton_oof": oof_pred,
})
oof_out.to_csv("oof_predictions.csv", index=False)

# Final Kaggle submission = mean probability across the 5 fold models.
submission = pd.read_csv(SAMPLE_SUB_PATH)
submission[TARGET] = test_pred
submission.to_csv("submission.csv", index=False)

print("Saved: oof_predictions.csv")
print("Saved: submission.csv")

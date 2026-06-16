"""
Utility functions for MTDriverGNN.

Contains:
- Seeding / device setup
- Graph & feature loading
- Label loading (single-disease and cross-disease pretrain labels)
- Loss / metric helpers
- Cross-disease pretraining routine
- One-epoch training and evaluation routines for the multi-task model
"""

import os
import random
import copy
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce

from model import GCN_Residual_TwoHeads, LearnableAlpha

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ----------------- DATA & LABEL LOADING ----------------- #

def load_graph_and_features(cfg):
    ppi_path = os.path.join(cfg.base_dir, "PPI_CPDB.csv")
    ppi_df = pd.read_csv(ppi_path)

    col1, col2 = ppi_df.columns[:2]
    g1 = ppi_df[col1].astype(str).values
    g2 = ppi_df[col2].astype(str).values

    genes_edge = np.unique(np.concatenate([g1, g2]))

    feat_path = os.path.join(cfg.base_dir, f"features_for_{cfg.target_disease}.csv")
    feat_df = pd.read_csv(feat_path, index_col=0)
    feat_df.index = feat_df.index.astype(str)
    feat_genes = feat_df.index.values

    all_genes = np.unique(np.concatenate([genes_edge, feat_genes]))
    node_names = all_genes
    num_nodes = len(node_names)
    print(f"[INFO] #nodes (CPDB union features) = {num_nodes}")

    node_to_idx: Dict[str, int] = {g: i for i, g in enumerate(node_names)}

    src = np.array([node_to_idx[g] for g in g1], dtype=np.int64)
    dst = np.array([node_to_idx[g] for g in g2], dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    print(f"[INFO] #edges after undirected+coalesce = {edge_index.size(1)}")

    feature_dim = feat_df.shape[1]
    x_mat = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    has_feat = np.zeros(num_nodes, dtype=bool)

    genes_in_both = feat_df.index.intersection(pd.Index(node_names))
    print(f"[INFO] #genes with features ∩ graph = {len(genes_in_both)}")

    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_df.loc[genes_in_both].values)
    feat_scaled_df = pd.DataFrame(feat_scaled, index=genes_in_both, columns=feat_df.columns)

    for g in genes_in_both:
        idx = node_to_idx[g]
        x_mat[idx] = feat_scaled_df.loc[g].values
        has_feat[idx] = True

    neighbors = {i: [] for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
        neighbors[v].append(u)

    for i in range(num_nodes):
        if not has_feat[i]:
            nb = [x_mat[n] for n in neighbors[i] if has_feat[n]]
            if nb:
                x_mat[i] = np.mean(nb, axis=0)

    x = torch.tensor(x_mat, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)
    return data, node_names, node_to_idx


def load_labels(cfg, data: Data, node_to_idx: Dict[str, int]):
    num_nodes = data.num_nodes
    y1 = torch.full((num_nodes,), -1, dtype=torch.long)
    y2 = torch.full((num_nodes,), -1, dtype=torch.long)

    y1_path = os.path.join(cfg.base_dir, f"{cfg.target_disease}_labels(0_1).csv")
    y1_df = pd.read_csv(y1_path)
    y1_df["Gene"] = y1_df["Gene"].astype(str)
    y1_map = dict(zip(y1_df['Gene'], y1_df['Labels']))
    for g, lab in y1_map.items():
        if g in node_to_idx:
            y1[node_to_idx[g]] = int(lab)

    y2_path = os.path.join(cfg.base_dir, "labels_telomere.csv")
    y2_df = pd.read_csv(y2_path)
    y2_df["Gene"] = y2_df["Gene"].astype(str)
    y2_map = dict(zip(y2_df['Gene'], y2_df['Labels']))
    for g, lab in y2_map.items():
        if g in node_to_idx:
            y2[node_to_idx[g]] = int(lab)

    data = data.to(DEVICE)
    data.y = y1.to(DEVICE)
    data.y2 = y2.to(DEVICE)
    labeled_idx = (data.y != -1).nonzero(as_tuple=True)[0]
    print(f"[INFO] #labeled (y1 != -1) = {len(labeled_idx)}")
    return data, labeled_idx


# ----------------- PRETRAIN META ----------------- #

def build_disease_dict(base_dir: str) -> Dict[str, Dict[str, str]]:
    disease_list = ["BRCA", "LUAD", "CESC", "BLCA", "LIHC", "THCA",
                    "ESCA", "PRAD", "STAD", "COAD", "UCEC", "LUSC"]
    diseases = {
        d: {
            "Y1": os.path.join(base_dir, f"{d}_labels(0_1).csv"),
        }
        for d in disease_list
    }
    return diseases


# ----------------- LOSS / METRIC UTILS ----------------- #

@torch.no_grad()
def auprc_on_mask(logits: torch.Tensor, y_long: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return 0.0
    labels = y_long[mask].detach().cpu().numpy()
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return 0.0
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    return float(average_precision_score(labels, probs))


def bce_pos_weight_from_mask(y_long: torch.Tensor, mask: torch.Tensor, device) -> torch.Tensor:
    yy = y_long[mask]
    if yy.numel() == 0:
        return None
    pos = (yy == 1).sum().item()
    neg = (yy == 0).sum().item()
    if pos == 0:
        return None
    return torch.tensor([neg / float(pos)], dtype=torch.float, device=device)


# ----------------- CROSS-DISEASE PRETRAIN ----------------- #

def build_cross_disease_pretrain_labels(
    node_to_idx: Dict[str, int],
    target_name: str,
    data_y: torch.Tensor,
    base_dir: str
):
    diseases = build_disease_dict(base_dir)
    num_nodes = len(node_to_idx)
    y_pre = torch.full((num_nodes,), -1, dtype=torch.long)
    has_any_label = torch.zeros(num_nodes, dtype=torch.bool)

    target_labeled_mask = (data_y.detach().cpu() != -1)

    for d, meta in diseases.items():
        if d == target_name:
            continue
        df = pd.read_csv(meta["Y1"])
        df["Gene"] = df["Gene"].astype(str)
        for g, lab in zip(df["Gene"], df["Labels"]):
            if g in node_to_idx:
                idx = node_to_idx[g]
                has_any_label[idx] = True
                if int(lab) == 1:
                    y_pre[idx] = 1

    unlabeled_any = (~has_any_label)
    y_pre[unlabeled_any] = 0

    pretrain_mask = ((y_pre == 1) | (y_pre == 0)) & (~target_labeled_mask)

    pos_ct = int(((y_pre == 1) & pretrain_mask).sum().item())
    neg_ct = int(((y_pre == 0) & pretrain_mask).sum().item())
    excl   = int((target_labeled_mask & ((y_pre == 1) | (y_pre == 0))).sum().item())
    print(f"[Pretrain-XDisease] Pos: {pos_ct} | Neg: {neg_ct} | Excluded target-labeled: {excl}")
    return y_pre.to(DEVICE), pretrain_mask.to(DEVICE)


def pretrain_on_cross_disease(
    data: Data,
    y_pre: torch.Tensor,
    pretrain_mask: torch.Tensor,
    hidden_dims: List[int],
    cfg,
    dropout: float,
    weight_decay: float,
):
    model = GCN_Residual_TwoHeads(
        in_dim=data.num_features,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_layernorm=False,
        head_hidden=None,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.pretrain_lr,
        weight_decay=weight_decay,
    )

    yy = y_pre[pretrain_mask]
    pos = int((yy == 1).sum().item())
    neg = int((yy == 0).sum().item())
    if pos == 0:
        print("[Pretrain-XDisease] No positive samples; skipping pretraining.")
        return None

    pos_weight = torch.tensor([neg / float(pos)], dtype=torch.float, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    wait = 0

    for ep in range(1, cfg.pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logit1, _ = model(data.x, data.edge_index)
        loss = criterion(logit1[pretrain_mask], y_pre[pretrain_mask].float())
        loss.backward()
        optimizer.step()

        cur_loss = float(loss.item())
        if cur_loss < best_loss - 1e-6:
            best_loss = cur_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.pretrain_patience:
                print(f"[Pretrain-XDisease] Early stop @ epoch {ep}, best_loss={best_loss:.4f}")
                break

    pre_state = {
        k: v
        for k, v in best_state.items()
        if k.startswith("encoder.") or k.startswith("shared.") or k.startswith("out_y1.")
    }
    return pre_state


# ----------------- TRAIN / EVAL (MULTITASK) ----------------- #

def train_one_epoch_dual(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    y2_mask_train: torch.Tensor,
    opt: torch.optim.Optimizer,
    crit_y1,
    crit_y2,
    alpha_module: LearnableAlpha,
    epoch: int,
    warmup: int = 10,
):
    model.train()
    opt.zero_grad()
    logit1, logit2 = model(data.x, data.edge_index)

    loss1 = (
        crit_y1(logit1[train_mask], data.y[train_mask].float())
        if (crit_y1 is not None and train_mask.sum().item() > 0)
        else torch.tensor(0.0, device=logit1.device)
    )
    loss2 = (
        crit_y2(logit2[y2_mask_train], data.y2[y2_mask_train].float())
        if (crit_y2 is not None and y2_mask_train.sum().item() > 0)
        else torch.tensor(0.0, device=logit2.device)
    )

    if epoch < warmup:
        total = loss1
        alpha_val = 1.0
    else:
        alpha = alpha_module()
        total = alpha * loss1 + (1.0 - alpha) * loss2
        alpha_val = float(alpha.item())

    total.backward()
    opt.step()
    return float(loss1.item()), float(loss2.item()), float(total.item()), alpha_val


@torch.no_grad()
def evaluate_y1(model: nn.Module, data: Data, mask: torch.Tensor, criterion_y1):
    model.eval()
    logit1, _ = model(data.x, data.edge_index)
    loss = (
        criterion_y1(logit1[mask], data.y[mask].float())
        if (criterion_y1 is not None and mask.sum().item() > 0)
        else 0.0
    )
    auprc = auprc_on_mask(logit1, data.y, mask)
    return float(loss), float(auprc)

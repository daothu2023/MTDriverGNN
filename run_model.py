"""
Entry point for MTDriverGNN.

Example usage:
    python run_model.py BRCA
    python run_model.py LUAD --kfold 5 --epochs 300 --results_dir ./results
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from model import GCN_Residual_TwoHeads, LearnableAlpha
from utils import (
    DEVICE,
    set_seed,
    load_graph_and_features,
    load_labels,
    build_cross_disease_pretrain_labels,
    pretrain_on_cross_disease,
    train_one_epoch_dual,
    evaluate_y1,
    auprc_on_mask,
    bce_pos_weight_from_mask,
)


@dataclass
class TrainConfig:
    # General training
    num_epochs: int = 300
    patience: int = 30
    kfold: int = 5
    warmup: int = 10

    # Pretrain (cross-disease)
    pretrain_epochs: int = 300
    pretrain_patience: int = 30
    pretrain_lr: float = 1e-2

    # Search space
    depth_options: Tuple[int, ...] = (2,)
    hidden_options: Tuple[Tuple[int, int], ...] = ((64, 128),)
    dropout_options: Tuple[float, ...] = (0.5,)
    lr_options: Tuple[float, ...] = (1e-2,)
    weight_decay_options: Tuple[float, ...] = (5e-4,)

    # IO
    base_dir: str = "./Data"
    target_disease: str = "BRCA"
    results_dir: str = "./results"


# ----------------- GRID SEARCH ----------------- #

def grid_search_for_outer_fold(
    data, outer_trainval_idx, outer_test_idx,
    y_pre, pretrain_mask, cfg: TrainConfig, base_seed: int,
):
    best_hp = None
    best_val_auc = -1.0

    y_tv = data.y[outer_trainval_idx].detach().cpu().numpy()

    for depth in cfg.depth_options:
        for hd_pair in cfg.hidden_options:
            hidden_dims = [hd_pair[0]] if depth == 1 else list(hd_pair)
            for dr in cfg.dropout_options:
                for lr in cfg.lr_options:
                    for wd in cfg.weight_decay_options:
                        print("\n[GRID-FOLD] depth={}, hidden={}, drop={}, lr={}, wd={}".format(
                            depth, hidden_dims, dr, lr, wd))

                        set_seed(base_seed + 1000)

                        pretrained_state = pretrain_on_cross_disease(
                            data=data, y_pre=y_pre, pretrain_mask=pretrain_mask,
                            hidden_dims=hidden_dims, cfg=cfg, dropout=dr, weight_decay=wd,
                        )

                        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=base_seed + 1234)
                        tr_sub, va_sub = next(sss.split(outer_trainval_idx.cpu(), y_tv))
                        inner_train_nodes = outer_trainval_idx[tr_sub].to(DEVICE)
                        inner_val_nodes = outer_trainval_idx[va_sub].to(DEVICE)

                        N = data.num_nodes
                        train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
                        val_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
                        test_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
                        train_mask[inner_train_nodes] = True
                        val_mask[inner_val_nodes] = True
                        test_mask[outer_test_idx.to(DEVICE)] = True

                        y2_mask_train = (data.y2 != -1) & (~test_mask)
                        pw_y1 = bce_pos_weight_from_mask(data.y, train_mask, DEVICE)
                        pw_y2 = bce_pos_weight_from_mask(data.y2, y2_mask_train, DEVICE)
                        crit_y1 = nn.BCEWithLogitsLoss(pos_weight=pw_y1) if pw_y1 is not None else None
                        crit_y2 = nn.BCEWithLogitsLoss(pos_weight=pw_y2) if pw_y2 is not None else None

                        model = GCN_Residual_TwoHeads(
                            in_dim=data.num_features, hidden_dims=hidden_dims,
                            dropout=dr, use_layernorm=False, head_hidden=None,
                        ).to(DEVICE)

                        if pretrained_state is not None:
                            model.load_state_dict(pretrained_state, strict=False)

                        alpha_module = LearnableAlpha(init_alpha=0.7).to(DEVICE)
                        optimizer = torch.optim.Adam(
                            list(model.parameters()) + list(alpha_module.parameters()),
                            lr=lr, weight_decay=wd,
                        )

                        best_val_auc_hp = -1.0
                        wait = 0
                        for ep in range(1, cfg.num_epochs + 1):
                            train_one_epoch_dual(
                                model, data, train_mask, y2_mask_train,
                                optimizer, crit_y1, crit_y2, alpha_module, ep, warmup=cfg.warmup
                            )
                            _, v_auc = evaluate_y1(model, data, val_mask, crit_y1)

                            if v_auc > best_val_auc_hp:
                                best_val_auc_hp = v_auc
                                wait = 0
                            else:
                                wait += 1
                                if wait >= cfg.patience:
                                    break

                        print(f"[GRID-FOLD] Val AUPRC = {best_val_auc_hp:.4f}")

                        if best_val_auc_hp > best_val_auc:
                            best_val_auc = best_val_auc_hp
                            best_hp = {
                                "depth": depth, "hidden_dims": hidden_dims,
                                "dropout": dr, "lr": lr, "weight_decay": wd,
                            }

    print(f"[GRID-FOLD] Best HP for this outer fold: {best_hp}, Val AUPRC={best_val_auc:.4f}")
    return best_hp


# ----------------- FINAL TRAIN + TEST FOR 1 OUTER FOLD ----------------- #

def train_and_test_one_outer_fold(
    data, outer_trainval_idx, outer_test_idx, y_pre, pretrain_mask,
    cfg: TrainConfig, hp: Dict[str, Any], base_seed: int,
):
    depth = hp["depth"]
    hidden_dims = hp["hidden_dims"]
    dr = hp["dropout"]
    lr = hp["lr"]
    wd = hp["weight_decay"]

    set_seed(base_seed + 2000)

    pretrained_state = pretrain_on_cross_disease(
        data=data, y_pre=y_pre, pretrain_mask=pretrain_mask,
        hidden_dims=hidden_dims, cfg=cfg, dropout=dr, weight_decay=wd,
    )

    y_tv = data.y[outer_trainval_idx].detach().cpu().numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=base_seed + 2345)
    tr_sub, va_sub = next(sss.split(outer_trainval_idx.cpu(), y_tv))
    inner_train_nodes = outer_trainval_idx[tr_sub].to(DEVICE)
    inner_val_nodes = outer_trainval_idx[va_sub].to(DEVICE)

    N = data.num_nodes
    train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    train_mask[inner_train_nodes] = True
    val_mask[inner_val_nodes] = True
    test_mask[outer_test_idx.to(DEVICE)] = True

    y2_mask_train = (data.y2 != -1) & (~test_mask)
    pw_y1 = bce_pos_weight_from_mask(data.y, train_mask, DEVICE)
    pw_y2 = bce_pos_weight_from_mask(data.y2, y2_mask_train, DEVICE)
    crit_y1 = nn.BCEWithLogitsLoss(pos_weight=pw_y1) if pw_y1 is not None else None
    crit_y2 = nn.BCEWithLogitsLoss(pos_weight=pw_y2) if pw_y2 is not None else None

    model = GCN_Residual_TwoHeads(
        in_dim=data.num_features, hidden_dims=hidden_dims,
        dropout=dr, use_layernorm=False, head_hidden=None,
    ).to(DEVICE)

    if pretrained_state is not None:
        model.load_state_dict(pretrained_state, strict=False)

    alpha_module = LearnableAlpha(init_alpha=0.7).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(alpha_module.parameters()),
        lr=lr, weight_decay=wd,
    )

    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    wait = 0

    for ep in range(1, cfg.num_epochs + 1):
        train_one_epoch_dual(
            model, data, train_mask, y2_mask_train,
            optimizer, crit_y1, crit_y2, alpha_module, ep, warmup=cfg.warmup
        )
        v_loss, v_auc = evaluate_y1(model, data, val_mask, crit_y1)

        if v_loss < best_val_loss - 1e-6:
            best_val_loss = v_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state, strict=True)
    logit1, _ = model(data.x, data.edge_index)
    test_auprc = auprc_on_mask(logit1, data.y, test_mask)
    print(f"[OUTER-FOLD] Test AUPRC = {test_auprc:.4f}")
    return test_auprc


def nested_cv_one_run(data, labeled_idx, y_pre, pretrain_mask, cfg: TrainConfig, base_seed: int):
    skf = StratifiedKFold(n_splits=cfg.kfold, shuffle=True, random_state=base_seed)
    y_np = data.y[labeled_idx].detach().cpu().numpy()
    fold_test_scores = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(labeled_idx.cpu(), y_np), start=1):
        print(f"\n========== RUN seed={base_seed} | OUTER FOLD {fold}/{cfg.kfold} ==========")
        outer_trainval_idx = labeled_idx[tr_idx]
        outer_test_idx = labeled_idx[te_idx]

        best_hp = grid_search_for_outer_fold(
            data, outer_trainval_idx, outer_test_idx, y_pre, pretrain_mask,
            cfg=cfg, base_seed=base_seed + fold * 10,
        )

        test_auc = train_and_test_one_outer_fold(
            data, outer_trainval_idx, outer_test_idx, y_pre, pretrain_mask,
            cfg=cfg, hp=best_hp, base_seed=base_seed + fold * 20,
        )
        fold_test_scores.append(test_auc)

    mean_auc = float(np.mean(fold_test_scores))
    std_auc = float(np.std(fold_test_scores))
    print(f"\n[RUN seed={base_seed}] Mean Test AUPRC over {cfg.kfold} folds: {mean_auc:.4f} ± {std_auc:.4f}")
    return fold_test_scores, mean_auc, std_auc


# ----------------- MAIN ----------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("disease", type=str, help="Target cancer type, e.g. BRCA, LUAD, UCEC")
    parser.add_argument("--data_dir", type=str, default="./Data", help="Path to data folder")
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to results folder")
    parser.add_argument("--kfold", type=int, default=5, help="Number of outer CV folds")
    parser.add_argument("--epochs", type=int, default=300, help="Max training epochs per fold")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of repeated runs (different seeds)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    cfg = TrainConfig(
        base_dir=args.data_dir,
        target_disease=args.disease,
        results_dir=args.results_dir,
        kfold=args.kfold,
        num_epochs=args.epochs,
        patience=args.patience,
    )
    os.makedirs(cfg.results_dir, exist_ok=True)

    print(f"Using device: {DEVICE}")

    data, node_names, node_to_idx = load_graph_and_features(cfg)
    data, labeled_idx = load_labels(cfg, data, node_to_idx)

    y_pre, pretrain_mask = build_cross_disease_pretrain_labels(
        node_to_idx=node_to_idx,
        target_name=cfg.target_disease,
        data_y=data.y,
        base_dir=cfg.base_dir,
    )

    seeds = list(range(args.seed, args.seed + args.num_runs))
    all_run_results = []

    for i, seed in enumerate(seeds, start=1):
        print("\n=========================================")
        print(f"======== NESTED CV RUN {i}/{args.num_runs} — seed = {seed} =========")
        print("=========================================")

        set_seed(seed)
        fold_scores, mean_auc, std_auc = nested_cv_one_run(
            data, labeled_idx, y_pre, pretrain_mask, cfg=cfg, base_seed=seed,
        )

        all_run_results.append({
            "seed": seed,
            "fold_test_auprc": fold_scores,
            "mean_test_auprc": mean_auc,
            "std_test_auprc": std_auc,
        })

    out_path = os.path.join(cfg.results_dir, f"nested_cv_runs_{cfg.target_disease}.json")
    with open(out_path, "w") as f:
        json.dump(all_run_results, f, indent=2)
    print(f"\n[INFO] Nested CV {args.num_runs}-run results saved to: {out_path}")

    all_means = [r["mean_test_auprc"] for r in all_run_results]
    print(f"\n========== {args.num_runs}-RUN SUMMARY (Nested CV) ==========")
    print(f"Mean of run-wise mean Test AUPRC: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")


if __name__ == "__main__":
    main()

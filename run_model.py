"""
MTDriverGNN training pipeline.

Two-head residual GCN (cancer-driver head + telomere-association auxiliary
head) with cross-disease supervised pretraining, nested 5-fold stratified
cross-validation, and grid search over the hyperparameter space reported in
the manuscript.

Grid search
-----------
For each outer cross-validation fold, every hyperparameter combination is
trained with early stopping based on validation loss. Among all
combinations, the configuration achieving the highest validation AUPRC is
selected and retrained to obtain the final test-set score for that fold.

Usage
-----
    python run_model.py BRCA
    python run_model.py LUAD --num_runs 3 --data_dir ./Data
"""

import os
import json
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from utils import (
    DEVICE,
    set_seed,
    build_cancer_data_paths,
    build_dataset_for_cancer,
    build_cross_disease_pretrain_labels,
    pretrain_on_cross_disease,
    train_single_configuration,
    auprc_on_mask,
    make_masks,
)

NUM_EPOCHS = 300
PATIENCE = 30
KFOLD = 5
WARMUP_EPOCHS = 10

PRETRAIN_LR = 1e-2
PRETRAIN_WEIGHT_DECAY = 5e-4
PRETRAIN_EPOCHS = 300
PRETRAIN_PATIENCE = 30

# Hyperparameter search space, as reported in the manuscript.
DEPTH_OPTIONS = (1, 2)
HIDDEN_OPTIONS = ((64, 64), (64, 128), (128, 64), (128, 128))
DROPOUT_OPTIONS = (0.3, 0.4, 0.5)
LR_OPTIONS = (1e-2, 3e-3, 1e-3)
WEIGHT_DECAY_OPTIONS = (1e-4, 5e-4, 1e-3)


def split_inner_train_val(outer_trainval_idx, labels, seed):
    labels_np = labels[outer_trainval_idx].detach().cpu().numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_sub_idx, val_sub_idx = next(splitter.split(outer_trainval_idx.cpu(), labels_np))
    inner_train_idx = outer_trainval_idx[train_sub_idx].to(DEVICE)
    inner_val_idx = outer_trainval_idx[val_sub_idx].to(DEVICE)
    return inner_train_idx, inner_val_idx


def grid_search_for_outer_fold(data, outer_trainval_idx, outer_test_idx,
                                pretrain_labels, pretrain_mask, seed):
    """
    Split outer_trainval_idx into an inner 80/20 train/val partition,
    iterate over the full hyperparameter grid (DEPTH_OPTIONS x
    HIDDEN_OPTIONS x DROPOUT_OPTIONS x LR_OPTIONS x WEIGHT_DECAY_OPTIONS),
    pretrain and train each configuration via
    `train_single_configuration`, and return the hyperparameter
    dictionary of the configuration with the highest validation AUPRC.
    """
    inner_train_idx, inner_val_idx = split_inner_train_val(outer_trainval_idx, data.y, seed)
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, inner_train_idx, inner_val_idx, outer_test_idx.to(DEVICE), DEVICE
    )
    telomere_train_mask = (data.y_telomere != -1) & (~test_mask)

    best_hp = None
    best_val_auprc = -1.0

    for depth in DEPTH_OPTIONS:
        for hidden_pair in HIDDEN_OPTIONS:
            hidden_dims = [hidden_pair[0]] if depth == 1 else list(hidden_pair)
            for dropout in DROPOUT_OPTIONS:
                for lr in LR_OPTIONS:
                    for weight_decay in WEIGHT_DECAY_OPTIONS:
                        print(f"[Grid search] depth={depth}, hidden={hidden_dims}, "
                              f"dropout={dropout}, lr={lr}, weight_decay={weight_decay}")

                        pretrained_state = pretrain_on_cross_disease(
                            data, pretrain_labels, pretrain_mask, hidden_dims, dropout,
                            PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
                        )
                        val_loss, val_auprc, _ = train_single_configuration(
                            data, train_mask, val_mask, telomere_train_mask,
                            hidden_dims, dropout, lr, weight_decay, pretrained_state,
                            NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
                        )
                        print(f"[Grid search] val_loss={val_loss:.4f}, val_auprc={val_auprc:.4f}")

                        if val_auprc > best_val_auprc:
                            best_val_auprc = val_auprc
                            best_hp = {
                                "hidden_dims": hidden_dims,
                                "dropout": dropout,
                                "lr": lr,
                                "weight_decay": weight_decay,
                            }

    print(f"[Grid search] Best configuration: {best_hp}, val_auprc={best_val_auprc:.4f}")
    return best_hp


def train_and_test_outer_fold(data, outer_trainval_idx, outer_test_idx,
                               pretrain_labels, pretrain_mask, best_hp, seed):
    """Retrain with the selected hyperparameters and evaluate on the
    held-out outer test set."""
    inner_train_idx, inner_val_idx = split_inner_train_val(outer_trainval_idx, data.y, seed)
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, inner_train_idx, inner_val_idx, outer_test_idx.to(DEVICE), DEVICE
    )
    telomere_train_mask = (data.y_telomere != -1) & (~test_mask)

    pretrained_state = pretrain_on_cross_disease(
        data, pretrain_labels, pretrain_mask, best_hp["hidden_dims"], best_hp["dropout"],
        PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
    )
    _, _, model = train_single_configuration(
        data, train_mask, val_mask, telomere_train_mask,
        best_hp["hidden_dims"], best_hp["dropout"], best_hp["lr"], best_hp["weight_decay"],
        pretrained_state, NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
    )

    driver_logit, _ = model(data.x, data.edge_index)
    test_auprc = auprc_on_mask(driver_logit, data.y, test_mask)
    print(f"[Outer fold] Best HP={best_hp}, test_auprc={test_auprc:.4f}")
    return test_auprc


def run_nested_cv_one_seed(data, labeled_idx, pretrain_labels, pretrain_mask, seed_offset):
    """Run 5-fold stratified cross-validation with grid search in each
    outer fold, for one random seed."""
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=41 + seed_offset)
    labels_np = data.y[labeled_idx].detach().cpu().numpy()
    fold_test_auprc = []

    for fold, (train_val_pos, test_pos) in enumerate(skf.split(labeled_idx.cpu(), labels_np), start=1):
        print(f"\n--- Fold {fold}/{KFOLD} ---")
        outer_trainval_idx = labeled_idx[train_val_pos]
        outer_test_idx = labeled_idx[test_pos]
        assert not set(outer_trainval_idx.tolist()) & set(outer_test_idx.tolist()), "Train/test split overlap detected."

        fold_seed = 43 + seed_offset + fold
        best_hp = grid_search_for_outer_fold(
            data, outer_trainval_idx, outer_test_idx, pretrain_labels, pretrain_mask, fold_seed
        )
        test_auprc = train_and_test_outer_fold(
            data, outer_trainval_idx, outer_test_idx, pretrain_labels, pretrain_mask, best_hp, fold_seed
        )
        fold_test_auprc.append(test_auprc)

    print(f"\nMean test AUPRC over {KFOLD} folds: {np.mean(fold_test_auprc):.4f} +/- {np.std(fold_test_auprc):.4f}")
    return fold_test_auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("disease", type=str, help="Target cancer type, e.g. BRCA, LUAD, UCEC")
    parser.add_argument("--data_dir", type=str, default="./Data", help="Path to the data folder")
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to save results")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of repeated runs (different seeds)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    target_cancer = args.disease
    ppi_cpdb_path = os.path.join(args.data_dir, "PPI_CPDB.csv")
    telomere_labels_path = os.path.join(args.data_dir, "labels_telomere.csv")
    cancer_data_paths = build_cancer_data_paths(args.data_dir)

    print("Using device:", DEVICE)

    run_mean_auprc = []

    for run in range(1, args.num_runs + 1):
        seed = 42 + run - 1
        set_seed(seed)

        print("\n" + "=" * 60)
        print(f"Run {run}/{args.num_runs} (seed={seed}), target cancer = {target_cancer}")
        print("=" * 60)

        data, node_to_idx, labeled_idx = build_dataset_for_cancer(
            target_cancer, ppi_cpdb_path, telomere_labels_path, cancer_data_paths
        )
        pretrain_labels, pretrain_mask = build_cross_disease_pretrain_labels(
            node_to_idx, target_cancer, data.y, cancer_data_paths
        )

        fold_test_auprc = run_nested_cv_one_seed(data, labeled_idx, pretrain_labels, pretrain_mask, seed_offset=run)
        run_mean_auprc.append(float(np.mean(fold_test_auprc)))
        print(f"\n[Run {run}/{args.num_runs}] Mean test AUPRC (5 folds) = {run_mean_auprc[-1]:.4f}")

    print("\n" + "=" * 60)
    print(f"Summary over {args.num_runs} runs")
    print("=" * 60)
    for i, mean_auprc in enumerate(run_mean_auprc, start=1):
        print(f"  Run {i} (seed={42 + i - 1}): mean test AUPRC = {mean_auprc:.4f}")

    overall_mean = float(np.mean(run_mean_auprc))
    overall_std = float(np.std(run_mean_auprc))
    print(f"\nOverall mean test AUPRC over {len(run_mean_auprc)} runs: {overall_mean:.4f} +/- {overall_std:.4f}")

    results = {
        "target_cancer": target_cancer,
        "num_runs": args.num_runs,
        "run_mean_auprc": run_mean_auprc,
        "overall_mean_auprc": overall_mean,
        "overall_std_auprc": overall_std,
    }
    out_path = os.path.join(args.results_dir, f"results_{target_cancer}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

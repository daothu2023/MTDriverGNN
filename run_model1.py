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
Full manuscript protocol for one cancer type:
    python run_model.py BRCA

Run fewer repeated runs:
    python run_model.py LUAD --num_runs 3 --data_dir ./Data

Reviewer/software quick test:
    python run_model.py BRCA --quick_test

Force CPU execution:
    python run_model.py BRCA --quick_test --device cpu

Predict driver genes using a trained model and new feature data:
    python run_model.py BRCA --predict --features ./Data/features_for_BRCA.csv
    python run_model.py BRCA --predict --features ./path/to/features.csv --top_k 100
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce

from model import MultiTaskGCN
import utils as utils_module
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


def format_configuration(depth, hidden_dims, dropout, lr, weight_decay):
    """Return one-line text describing the model/training configuration."""
    return (
        f"GCN layers={depth}, hidden dimensions={hidden_dims}, "
        f"dropout={dropout}, learning rate={lr}, weight decay={weight_decay}"
    )


def configure_device(device_arg):
    """
    Configure the execution device.

    Parameters
    ----------
    device_arg : {"auto", "cpu", "gpu"}
        auto: use GPU if available, otherwise CPU.
        cpu: force CPU execution.
        gpu: force GPU execution; raise an error if no GPU is available.

    Notes
    -----
    PyTorch uses the "cuda" device name for both NVIDIA/CUDA and AMD/ROCm
    backends. Therefore, "gpu" maps to torch.device("cuda") when available.
    """
    global DEVICE

    if device_arg == "cpu":
        selected_device = torch.device("cpu")
    elif device_arg == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but torch.cuda.is_available() is False.")
        selected_device = torch.device("cuda")
    else:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update both the local imported DEVICE and the DEVICE stored in utils.py,
    # because helper functions imported from utils use the module-level DEVICE.
    DEVICE = selected_device
    utils_module.DEVICE = selected_device

    return selected_device



def save_best_model(model, best_hp, cancer, results_dir, quick_test=False):
    """
    Save the final trained model to a .pt file for use with --predict mode.
    The checkpoint stores weights, architecture (hidden_dims, dropout),
    cancer type, and hyperparameters so the model can be reconstructed
    without the original training arguments.
    """
    prefix = "quick_test_model" if quick_test else "best_model"
    save_path = os.path.join(results_dir, f"{prefix}_{cancer}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dims": best_hp["hidden_dims"],
        "dropout": best_hp["dropout"],
        "cancer": cancer,
        "hyperparams": best_hp,
    }, save_path)
    print(f"[Model saved] {save_path}")
    return save_path


# ------------------------------------------------------------------ #
# Prediction helpers
# ------------------------------------------------------------------ #

def _load_graph_for_predict(ppi_path):
    """Build the fixed PPI graph from PPI_CPDB.csv."""
    ppi_df = pd.read_csv(ppi_path)
    gene_col_a, gene_col_b = ppi_df.columns[:2]
    genes_a = ppi_df[gene_col_a].astype(str).values
    genes_b = ppi_df[gene_col_b].astype(str).values
    node_names = np.unique(np.concatenate([genes_a, genes_b]))
    num_nodes = len(node_names)
    node_to_idx = {gene: i for i, gene in enumerate(node_names)}
    src = np.array([node_to_idx[g] for g in genes_a], dtype=np.int64)
    dst = np.array([node_to_idx[g] for g in genes_b], dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    print(f"[Graph] #nodes = {num_nodes}, #edges (undirected) = {edge_index.size(1)}")
    return node_names, node_to_idx, edge_index, num_nodes


def _build_features_for_predict(features_path, node_names, node_to_idx, num_nodes, edge_index):
    """
    Map the user-supplied feature matrix onto the fixed PPI node set.
    Genes present in the file are z-score standardized; genes absent
    are imputed by averaging their neighbors' feature vectors.
    """
    feat_df = pd.read_csv(features_path, index_col=0)
    feat_df.index = feat_df.index.astype(str)
    feature_dim = feat_df.shape[1]
    feature_matrix = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    has_feature = np.zeros(num_nodes, dtype=bool)
    genes_with_features = feat_df.index.intersection(pd.Index(node_names))
    print(f"[Features] #genes matched with PPI nodes = {len(genes_with_features)} / {num_nodes}")
    if len(genes_with_features) == 0:
        raise ValueError(
            "No genes in the feature file matched the PPI graph node set. "
            "Check that the CSV index column contains gene names."
        )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_df.loc[genes_with_features].values)
    scaled_df = pd.DataFrame(scaled, index=genes_with_features)
    for gene in genes_with_features:
        idx = node_to_idx[gene]
        feature_matrix[idx] = scaled_df.loc[gene].values
        has_feature[idx] = True
    neighbors = {i: [] for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
        neighbors[v].append(u)
    imputed = 0
    for idx in range(num_nodes):
        if not has_feature[idx]:
            neighbor_feats = [feature_matrix[n] for n in neighbors[idx] if has_feature[n]]
            if neighbor_feats:
                feature_matrix[idx] = np.mean(neighbor_feats, axis=0)
                imputed += 1
    print(f"[Features] #nodes imputed by neighbor averaging = {imputed}")
    return torch.tensor(feature_matrix, dtype=torch.float32)


def run_predict(args, selected_device):
    """
    Predict cancer driver gene scores on new feature data.

    The PPI graph (PPI_CPDB.csv) is fixed and shared across all cancer types.
    Only the feature matrix changes between users / cancer types.
    The trained model is loaded from --model_path (default: results/best_model_<CANCER>.pt).
    Results are written to --output (default: results/predictions_<CANCER>.csv).
    """
    ppi_path = os.path.join(args.data_dir, "PPI_CPDB.csv")
    if args.model_path is None:
        prefix = "quick_test_model" if args.quick_test else "best_model"
        args.model_path = os.path.join(args.results_dir, f"{prefix}_{args.disease}.pt")
    if args.output is None:
        os.makedirs(args.results_dir, exist_ok=True)
        args.output = os.path.join(args.results_dir, f"predictions_{args.disease}.csv")

    for path, label in [
        (ppi_path,        "PPI_CPDB.csv"),
        (args.features,   "features CSV (--features)"),
        (args.model_path, "trained model (--model_path or default best_model_<CANCER>.pt)"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found — {label}: {path}")

    node_names, node_to_idx, edge_index, num_nodes = _load_graph_for_predict(ppi_path)
    x = _build_features_for_predict(args.features, node_names, node_to_idx, num_nodes, edge_index)
    data = Data(x=x, edge_index=edge_index).to(selected_device)

    checkpoint = torch.load(args.model_path, map_location=selected_device)
    hidden_dims = checkpoint["hidden_dims"]
    dropout = checkpoint.get("dropout", 0.5)
    model = MultiTaskGCN(in_dim=data.num_features, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()
    print(f"[Model] Loaded: {args.model_path}")
    print(f"[Model] hidden_dims={hidden_dims}, dropout={dropout}")

    print("[Predict] Running forward pass...")
    with torch.no_grad():
        driver_logit, _ = model(data.x, data.edge_index)
        scores = torch.sigmoid(driver_logit).cpu().numpy()

    results_df = pd.DataFrame({"gene": node_names, "driver_score": scores})
    results_df = results_df.sort_values("driver_score", ascending=False).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1
    if args.top_k is not None:
        results_df = results_df.head(args.top_k)
        print(f"[Output] Keeping top {args.top_k} genes.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\n[Done] Predictions saved to: {args.output}")
    print(f"       Genes ranked: {len(results_df)}")
    print(f"\nTop 10 predicted driver genes:")
    print(results_df.head(10).to_string(index=False))

def apply_quick_test_settings(args):
    """
    Apply reduced settings for reviewer/software testing.

    This mode keeps the full CPDB/PPI graph and real processed cancer data,
    but reduces runs, folds, epochs, and hyperparameter combinations.
    It is not intended to reproduce manuscript results.
    """
    global NUM_EPOCHS, PATIENCE, KFOLD, WARMUP_EPOCHS
    global PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE
    global DEPTH_OPTIONS, HIDDEN_OPTIONS, DROPOUT_OPTIONS, LR_OPTIONS, WEIGHT_DECAY_OPTIONS

    print("Quick-test mode enabled")
    print("  Full CPDB/PPI graph is used.")
    print("  Reduced runs, folds, epochs, and hyperparameter search are used for software testing only.")

    args.num_runs = 1

    NUM_EPOCHS = 5
    PATIENCE = 3
    KFOLD = 2
    WARMUP_EPOCHS = 1

    PRETRAIN_LR = 1e-2
    PRETRAIN_WEIGHT_DECAY = 5e-4
    PRETRAIN_EPOCHS = 3
    PRETRAIN_PATIENCE = 2

    DEPTH_OPTIONS = (1,)
    HIDDEN_OPTIONS = ((64, 64),)
    DROPOUT_OPTIONS = (0.3,)
    LR_OPTIONS = (1e-3,)
    WEIGHT_DECAY_OPTIONS = (5e-4,)


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
    iterate over the hyperparameter grid, pretrain and train each
    configuration via `train_single_configuration`, and return the
    hyperparameter dictionary of the configuration with the highest
    validation AUPRC.
    """
    inner_train_idx, inner_val_idx = split_inner_train_val(outer_trainval_idx, data.y, seed)
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, inner_train_idx, inner_val_idx, outer_test_idx.to(DEVICE), DEVICE
    )
    telomere_train_mask = (data.y_telomere != -1) & (~test_mask)

    best_hp = None
    best_val_auprc = -1.0

    total_candidates = (
        len(DEPTH_OPTIONS)
        * len(HIDDEN_OPTIONS)
        * len(DROPOUT_OPTIONS)
        * len(LR_OPTIONS)
        * len(WEIGHT_DECAY_OPTIONS)
    )
    candidate_id = 0

    for depth in DEPTH_OPTIONS:
        for hidden_pair in HIDDEN_OPTIONS:
            hidden_dims = [hidden_pair[0]] if depth == 1 else list(hidden_pair)
            for dropout in DROPOUT_OPTIONS:
                for lr in LR_OPTIONS:
                    for weight_decay in WEIGHT_DECAY_OPTIONS:
                        candidate_id += 1
                        config_text = format_configuration(depth, hidden_dims, dropout, lr, weight_decay)
                        if total_candidates == 1:
                            print(f"  Training configuration: {config_text}")
                        else:
                            print(f"  Candidate configuration {candidate_id}/{total_candidates}: {config_text}")

                        pretrained_state = pretrain_on_cross_disease(
                            data, pretrain_labels, pretrain_mask, hidden_dims, dropout,
                            PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
                        )
                        val_loss, val_auprc, _ = train_single_configuration(
                            data, train_mask, val_mask, telomere_train_mask,
                            hidden_dims, dropout, lr, weight_decay, pretrained_state,
                            NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
                        )
                        print(f"  Validation: loss={val_loss:.4f}, AUPRC={val_auprc:.4f}")

                        if val_auprc > best_val_auprc:
                            best_val_auprc = val_auprc
                            best_hp = {
                                "depth": depth,
                                "hidden_dims": hidden_dims,
                                "dropout": dropout,
                                "lr": lr,
                                "weight_decay": weight_decay,
                            }

    if total_candidates > 1:
        selected_text = format_configuration(
            best_hp["depth"], best_hp["hidden_dims"], best_hp["dropout"],
            best_hp["lr"], best_hp["weight_decay"]
        )
        print(f"  Selected configuration: {selected_text}, validation AUPRC={best_val_auprc:.4f}")

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
    print(f"  Test: AUPRC={test_auprc:.4f}")
    return test_auprc


def run_nested_cv_one_seed(data, labeled_idx, pretrain_labels, pretrain_mask, seed_offset):
    """Run stratified cross-validation with grid search in each outer fold,
    for one random seed."""
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=41 + seed_offset)
    labels_np = data.y[labeled_idx].detach().cpu().numpy()
    fold_test_auprc = []

    for fold, (train_val_pos, test_pos) in enumerate(skf.split(labeled_idx.cpu(), labels_np), start=1):
        print("\n" + "-" * 40)
        print(f"Fold {fold}/{KFOLD}")
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
        print("-" * 40)

    print(f"\nFold summary: mean test AUPRC = {np.mean(fold_test_auprc):.4f} +/- {np.std(fold_test_auprc):.4f}")
    return fold_test_auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("disease", type=str, help="Target cancer type, e.g. BRCA, LUAD, UCEC")
    parser.add_argument("--data_dir", type=str, default="./Data", help="Path to the data folder")
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to save results")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of repeated runs with different seeds")
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help=(
            "Run a compact software test using the full CPDB/PPI graph, one run, "
            "two folds, reduced epochs, and one fixed hyperparameter setting. "
            "This mode is not intended to reproduce manuscript results."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Execution device: auto uses GPU if available; cpu forces CPU; gpu forces GPU.",
    )
    # ── Predict mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--predict",
        action="store_true",
        help=(
            "Prediction mode: skip training, load a saved model, and score all genes "
            "in the provided feature matrix. Requires --features."
        ),
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help=(
            "[--predict] Path to feature matrix CSV. "
            "Rows = genes (index column), columns = omics features "
            "(e.g. CNV, methylation, expression, mutation)."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "[--predict] Path to trained model .pt file. "
            "Default: <results_dir>/best_model_<CANCER>.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "[--predict] Output CSV path. "
            "Default: <results_dir>/predictions_<CANCER>.csv"
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="[--predict] Save only the top-K genes by driver score. Default: save all.",
    )
    args = parser.parse_args()

    if args.quick_test:
        apply_quick_test_settings(args)

    selected_device = configure_device(args.device)

    # ------------------------------------------------------------------
    # Predict mode: load saved model and score new feature data.
    # Skips training entirely.
    # ------------------------------------------------------------------
    if args.predict:
        if args.features is None:
            raise ValueError("--predict requires --features <path_to_features.csv>")
        run_predict(args, selected_device)
        return

    os.makedirs(args.results_dir, exist_ok=True)

    target_cancer = args.disease
    ppi_cpdb_path = os.path.join(args.data_dir, "PPI_CPDB.csv")
    telomere_labels_path = os.path.join(args.data_dir, "labels_telomere.csv")
    cancer_data_paths = build_cancer_data_paths(args.data_dir)

    print("Using device:", selected_device)

    run_mean_auprc = []
    all_fold_test_auprc = []

    for run in range(1, args.num_runs + 1):
        seed = 42 + run - 1
        set_seed(seed)

        print("\n" + "=" * 60)
        print(f"Run {run}/{args.num_runs} | seed={seed} | target cancer={target_cancer}")
        print("=" * 60)

        data, node_to_idx, labeled_idx = build_dataset_for_cancer(
            target_cancer, ppi_cpdb_path, telomere_labels_path, cancer_data_paths
        )
        pretrain_labels, pretrain_mask = build_cross_disease_pretrain_labels(
            node_to_idx, target_cancer, data.y, cancer_data_paths
        )

        fold_test_auprc = run_nested_cv_one_seed(data, labeled_idx, pretrain_labels, pretrain_mask, seed_offset=run)
        all_fold_test_auprc.append([float(x) for x in fold_test_auprc])
        run_mean_auprc.append(float(np.mean(fold_test_auprc)))
        print(f"\n[Run {run}/{args.num_runs}] Mean test AUPRC ({KFOLD} folds) = {run_mean_auprc[-1]:.4f}")

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
        "mode": "quick_test" if args.quick_test else "full",
        "device": str(selected_device),
        "num_runs": args.num_runs,
        "num_folds": KFOLD,
        "num_epochs": NUM_EPOCHS,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "quick_test_note": (
            "Quick-test mode uses the full CPDB/PPI graph and real processed data, "
            "but reduced runs/folds/epochs/grid search. It is intended only for "
            "software execution testing, not manuscript result reproduction."
        ) if args.quick_test else None,
        "fold_test_auprc": all_fold_test_auprc,
        "run_mean_auprc": run_mean_auprc,
        "overall_mean_auprc": overall_mean,
        "overall_std_auprc": overall_std,
    }

    prefix = "quick_test_results" if args.quick_test else "results"
    out_path = os.path.join(args.results_dir, f"{prefix}_{target_cancer}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # ------------------------------------------------------------------
    # Train final model on ALL labeled data with the best hyperparameters
    # (from the run with highest mean AUPRC) and save for --predict mode.
    # ------------------------------------------------------------------
    best_run_idx = int(np.argmax(run_mean_auprc))
    best_run_seed = 42 + best_run_idx
    print(f"\n[Final model] Retraining on all labeled data")
    print(f"              Best run: {best_run_idx + 1} (seed={best_run_seed}), "
          f"mean AUPRC = {run_mean_auprc[best_run_idx]:.4f}")

    set_seed(best_run_seed)
    data_final, node_to_idx_final, labeled_idx_final = build_dataset_for_cancer(
        target_cancer, ppi_cpdb_path, telomere_labels_path, cancer_data_paths
    )
    pretrain_labels_final, pretrain_mask_final = build_cross_disease_pretrain_labels(
        node_to_idx_final, target_cancer, data_final.y, cancer_data_paths
    )
    all_labeled_mask = torch.zeros(data_final.num_nodes, dtype=torch.bool, device=DEVICE)
    all_labeled_mask[labeled_idx_final] = True
    telomere_mask_final = (data_final.y_telomere != -1)

    # Pick best HP via a lightweight grid search on all labeled data
    best_hp_final = None
    best_val_auprc_final = -1.0
    for depth in DEPTH_OPTIONS:
        for hidden_pair in HIDDEN_OPTIONS:
            hidden_dims = [hidden_pair[0]] if depth == 1 else list(hidden_pair)
            for dropout in DROPOUT_OPTIONS:
                for lr in LR_OPTIONS:
                    for weight_decay in WEIGHT_DECAY_OPTIONS:
                        pretrained_state = pretrain_on_cross_disease(
                            data_final, pretrain_labels_final, pretrain_mask_final,
                            hidden_dims, dropout,
                            PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
                        )
                        _, val_auprc, _ = train_single_configuration(
                            data_final, all_labeled_mask, all_labeled_mask, telomere_mask_final,
                            hidden_dims, dropout, lr, weight_decay, pretrained_state,
                            NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
                        )
                        if val_auprc > best_val_auprc_final:
                            best_val_auprc_final = val_auprc
                            best_hp_final = {
                                "depth": depth, "hidden_dims": hidden_dims,
                                "dropout": dropout, "lr": lr, "weight_decay": weight_decay,
                            }

    pretrained_state_final = pretrain_on_cross_disease(
        data_final, pretrain_labels_final, pretrain_mask_final,
        best_hp_final["hidden_dims"], best_hp_final["dropout"],
        PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_PATIENCE,
    )
    _, _, final_model = train_single_configuration(
        data_final, all_labeled_mask, all_labeled_mask, telomere_mask_final,
        best_hp_final["hidden_dims"], best_hp_final["dropout"],
        best_hp_final["lr"], best_hp_final["weight_decay"],
        pretrained_state_final, NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS,
    )
    save_best_model(final_model, best_hp_final, target_cancer, args.results_dir, args.quick_test)


if __name__ == "__main__":
    main()

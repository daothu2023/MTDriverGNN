# MTDriverGNN: Multitask Learning Graph Neural Network for Cancer Driver Gene Prioritization

MTDriverGNN is a multi-task graph neural network for cancer driver gene
prioritization. A residual GCN encoder is shared between two prediction
heads, the cancer-driver head (primary task) and the telomere-association
head (auxiliary task), combined with cross-disease supervised pretraining
and a nested cross-validation protocol with grid search over the
hyperparameter space.

## Key Features

- **Residual GCN Encoder:** stacks GCN layers with a residual connection
  from the input node features to the final node embedding.
- **Multi-Task Heads:** a shared MLP layer feeds two linear output heads,
  the cancer-driver head and the telomere-association head, combined via
  a learnable weighting coefficient (alpha).
- **Cross-Disease Supervised Pretraining:** pretrains the encoder on
  driver-gene labels aggregated from other cancer types before
  fine-tuning on the target cancer.
- **Nested Cross-Validation with Grid Search:** 5-fold outer
  cross-validation, each fold preceded by a grid search over GCN depth,
  hidden dimensions, dropout rate, learning rate, and weight decay.
  Each configuration is trained with early stopping on validation loss;
  the configuration with the highest validation AUPRC is retrained to
  obtain the held-out test score for that fold.

## Requirements

- Python 3.9+
- torch >= 1.9.1
- torch-geometric >= 2.0.4
- numpy >= 1.21.5
- pandas >= 1.3.5
- scikit-learn >= 1.0.2

Install with:

```bash
pip install -r requirements.txt
```

## Data

All required data files are included in the `Data/` folder of this
repository, so no manual download is needed; just clone the repository
and run.

- `PPI_CPDB.csv`: protein-protein interaction edge list (two gene-name columns)
- `features_for_<CANCER>.csv`: node feature matrix for each cancer type (genes as index)
- `<CANCER>_labels(0_1).csv`: driver-gene labels for each cancer type, columns `Gene,Labels`
- `labels_telomere.csv`: auxiliary task labels, columns `Gene,Labels`

`<CANCER>` must match one of: `BRCA, LUAD, CESC, BLCA, LIHC, THCA, ESCA, PRAD, STAD, COAD, UCEC, LUSC`.

## Usage

```bash
# Run MTDriverGNN for BRCA with default settings (10 runs x 5-fold nested CV with grid search)
python run_model.py BRCA

# Run for LUAD with custom data/results folders and fewer runs
python run_model.py LUAD --data_dir ./Data --results_dir ./results --num_runs 3
```

The script will:

- Load the CPDB protein-protein interaction graph and the cancer-specific
  node feature matrix, imputing missing features by neighbor averaging.
- Load the driver-gene labels for the target cancer and the
  telomere-association labels.
- Build cross-disease pretraining labels from all other cancer types.
- For each random seed, run nested 5-fold cross-validation: for each
  outer fold, perform a grid search over the hyperparameter space
  (early stopping on validation loss per configuration, configuration
  selection by validation AUPRC), then retrain with the selected
  configuration and evaluate on the held-out test fold.
- Save per-run mean test AUPRC scores to `results/results_<CANCER>.json`
  and print a summary across all runs.

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./Data` | Path to the data folder |
| `--results_dir` | `./results` | Path to save results |
| `--num_runs` | `10` | Number of repeated runs with different seeds |

## Project Structure

```
MTDriverGNN/
├── README.md
├── requirements.txt
├── model.py        # model definitions (residual GCN encoder, multi-task heads, learnable alpha)
├── utils.py         # data loading, cross-disease pretraining, training/evaluation utilities
├── run_model.py      # main entry point (argparse, grid search, nested cross-validation)
├── Data/              # input data (included in this repository)
└── results/             # output JSON results
```

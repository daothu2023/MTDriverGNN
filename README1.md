# MTDriverGNN: Multi-Task GNN for Cancer Driver Gene Prediction

MTDriverGNN is a multi-task Graph Neural Network framework for cancer driver gene
prediction. It uses a residual GCN encoder shared between two prediction heads
(driver-gene label and an auxiliary label), combined with cross-disease pretraining
and nested cross-validation with hyperparameter grid search.

### Key Features

- **Residual GCN Encoder:** Stacks GCN layers with a residual connection from the
  input features to the final representation.
- **Multi-Task Heads:** A shared trunk feeds two output heads — the main driver-gene
  label (Y1) and an auxiliary label (Y2) — combined via a learnable weighting factor (alpha).
- **Cross-Disease Pretraining:** Pretrains the encoder on driver-gene labels from other
  cancer types before fine-tuning on the target disease.
- **Nested Cross-Validation:** Outer k-fold CV for unbiased test performance, with an
  inner grid search over depth, hidden dimensions, dropout, learning rate, and weight decay.

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

All required data files are already included in the `Data/` folder of this repository,
so no manual download is needed — just clone the repo and run.

- `PPI_CPDB.csv` — protein-protein interaction edge list (two gene-name columns)
- `features_for_<DISEASE>.csv` — node feature matrix for each cancer type (genes as index)
- `<DISEASE>_labels(0_1).csv` — driver-gene labels for each cancer type, columns `Gene,Labels`
- `labels_telomere.csv` — auxiliary task labels, columns `Gene,Labels`

`<DISEASE>` must match one of: `BRCA, LUAD, CESC, BLCA, LIHC, THCA, ESCA, PRAD, STAD, COAD, UCEC, LUSC`.

## Usage

```bash
# Run MTDriverGNN for BRCA with default settings (10 runs x 5-fold nested CV)
python run_model.py BRCA

# Run for LUAD with custom data/results folders and fewer runs
python run_model.py LUAD --data_dir ./Data --results_dir ./results --num_runs 3
```

The script will:

- Load the PPI graph and node features, building the gene-to-index mapping.
- Load disease-specific driver-gene labels and the auxiliary telomere label.
- Build cross-disease pretraining labels from all other cancer types.
- For each random seed, run nested cross-validation: an inner grid search picks the
  best hyperparameters per outer fold, then trains/evaluates a final model on that fold.
- Save per-run, per-fold Test AUPRC scores to `results/nested_cv_runs_<DISEASE>.json`
  and print a summary across all runs.

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./Data` | Path to the data folder |
| `--results_dir` | `./results` | Path to save results |
| `--kfold` | `5` | Number of outer CV folds |
| `--epochs` | `300` | Max training epochs per fold |
| `--patience` | `30` | Early stopping patience |
| `--num_runs` | `10` | Number of repeated runs with different seeds |
| `--seed` | `42` | Base random seed |

## Project Structure

```
MTDriverGNN/
├── README.md
├── requirements.txt
├── model.py        # model definitions (encoder, two-head model, learnable alpha)
├── utils.py         # data loading, pretraining, training/eval utilities
├── run_model.py      # main entry point (argparse + nested CV pipeline)
├── Data/              # input data (included in this repository)
└── results/             # output JSON results
```

# FolDE: Few-Shot Optimized Learning for Directed Evolution

This module provides infrastructure for simulating and evaluating machine learning models on protein engineering tasks, particularly for low-N protein engineering campaigns. FolDE combines protein language model embeddings with few-shot learning to guide directed evolution experiments.

## Quick Start

### System Requirements

- Python 3.8+
- ~152GB disk space for data
- 16GB+ RAM recommended for running simulations
- Google Cloud SDK (for data download)

### Installation

1. **Install the package:**
   ```bash
   pip install -e .
   ```

2. **Download required data (~152GB):**

   The simulation data is hosted in a public Google Cloud Storage bucket. You'll need the Google Cloud SDK installed:

   ```bash
   # Install gsutil if not already installed
   # See: https://cloud.google.com/sdk/docs/install

   # Download all required data
   mkdir -p backend/folde/data
   cd backend/folde/data

   # Download DMS datasets (~1.0GB)
   gsutil -m cp -r gs://foldedata/DMS_ProteinGym_substitutions .

   # Download embeddings (~151GB, this will take a while)
   gsutil -m cp -r gs://foldedata/embeddings .

   # Download naturalness scores (~257MB)
   gsutil -m cp -r gs://foldedata/naturalness .

   # Download metadata files
   gsutil cp gs://foldedata/DMS_substitutions.csv .
   gsutil cp gs://foldedata/FLIP-AAV_multimutant_dataset.csv .

   cd ../../..
   ```

3. **Verify installation:**
   ```bash
   python -c "from folde.data import get_dms_metadata; print(f'Found {len(get_dms_metadata())} DMS datasets')"
   ```

## Running Benchmark Simulations

This repository includes the benchmark simulations used in the FolDE paper. These simulations evaluate different model configurations across protein engineering datasets.

### Example 1: Test Benchmark (Single-Mutant Datasets)

This benchmark evaluates FolDE on 17 single-mutant DMS datasets with 6 rounds of 16 variants each:

```bash
python backend/notebooks/jacob/251003_test_benchmark.py
```

**Key parameters:**
- 17 DMS datasets from ProteinGym
- 10 simulations per dataset
- 6 rounds × 16 variants = 96 total measurements per simulation
- Compares: Random, RandomForest, Naturalness-only, and FolDE variants

**Expected runtime:** ~2-4 hours (depending on hardware and parallelization)

**Output:** Results saved to `backend/notebooks/jacob/model_evals/251003-test-benchmark_*.json`

### Example 2: Multimutant Benchmark

This benchmark tests FolDE on 3 multimutant DMS datasets with 10 rounds:

```bash
python backend/notebooks/jacob/251003_multimutant_benchmark.py
```

**Key parameters:**
- 3 multimutant DMS datasets (SPG1_STRSG, GRB2_HUMAN, PABP_YEAST)
- 10 simulations per dataset
- 10 rounds × 16 variants = 160 total measurements per simulation
- Tests various ablations and hyperparameter configurations

**Expected runtime:** ~4-8 hours

**Output:** Results saved to `backend/notebooks/jacob/model_evals/251003-multimutant-benchmark_*.json`

### Customizing Benchmarks

You can create custom benchmarks by modifying the configuration:

```python
from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig

# Define your model configuration
config = FolDEModelConfig(
    name="MyFolDE",
    naturalness_model_id="600m",  # ESM-2 650M
    embedding_model_id="300m",    # ESM-2 300M
    zero_shot_model_name="NaturalnessZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="TorchMLPFewShotModel",
    few_shot_model_params={
        "pretrain": True,
        "pretrain_epochs": 50,
        "ensemble_size": 5,
        "embedding_dim": 960,
        "hidden_dims": [100, 50],
        "dropout": 0.2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "train_epochs": 200,
        "train_patience": 40,
        "val_frequency": 10,
        "do_validation_with_pair_fraction": 0.2,
        "decision_mode": "constantliar",
        "lie_noise_stddev_multiplier_schedule": [6.0] * 2 + [100.0] * 8,
    },
)

# Run simulations
results = simulate_campaigns_with_config_checkpoints(
    eval_prefix="my-experiment",
    dms_ids=["BLAT_ECOLX_Stiffler_2015"],  # Choose your datasets
    config_list=[config],
    checkpoint_dir="results",
    round_size=16,           # Variants per round
    number_of_simulations=10,  # Number of simulation replicates
    max_rounds=6,            # Campaign rounds
    random_seed=42,
    num_workers=4,           # Parallel workers
)
```

## Understanding the Code

### Core Simulation Function

The main entry point for running simulations is `simulate_campaigns_with_config_checkpoints()` in [campaign.py](backend/folde/campaign.py:613). This function:

1. Loads protein datasets (DMS data, embeddings, naturalness scores)
2. Runs multiple simulated campaigns with different random seeds
3. Saves checkpoints after each DMS dataset (resumable)
4. Returns evaluation metrics for all configurations

### Model Configuration

`FolDEModelConfig` defines a complete model setup:

- **`naturalness_model_id`**: Which ESM-2 model to use for naturalness scores ("300m", "600m", "3b", "15b")
- **`embedding_model_id`**: Which ESM-2 model to use for sequence embeddings
- **`zero_shot_model_name`**: Model for first round (before any measurements)
  - `"NaturalnessZeroShotModel"`: Use naturalness scores only
  - `"RandomZeroShotModel"`: Random selection (baseline)
- **`few_shot_model_name`**: Model for subsequent rounds
  - `"TorchMLPFewShotModel"`: Neural network ensemble (FolDE)
  - `"RandomForestFewShotModel"`: Random forest baseline
  - `"NaturalnessFewShotModel"`: Naturalness-only (no learning)
- **`few_shot_model_params`**: Hyperparameters for the few-shot model
  - `decision_mode`: "mean", "constantliar", or "ucb" for ensemble aggregation
  - `ensemble_size`: Number of models in ensemble
  - `pretrain`: Whether to pretrain on single-mutant naturalness
  - See [types.py](backend/folde/types.py) for all options

### Key Modules

- [campaign.py](backend/folde/campaign.py) - Campaign simulation logic
- [data.py](backend/folde/data.py) - Data loading utilities
- [few_shot_models.py](backend/folde/few_shot_models.py) - Few-shot learning models (MLP, RandomForest)
- [zero_shot_models.py](backend/folde/zero_shot_models.py) - Zero-shot models (naturalness-based)
- [types.py](backend/folde/types.py) - Type definitions and configuration classes
- [util.py](backend/folde/util.py) - Utility functions for metrics and data processing

## ProteinGym Data

This module uses data from **ProteinGym**, a comprehensive benchmark for assessing protein fitness prediction models. ProteinGym was developed by Cheng et al. and provides a standardized collection of Deep Mutational Scanning (DMS) datasets.

**Citation**:
Cheng, Y., Raghuram, J., Aghazadeh, A., Huang, P.-S., & Russ, W. P. (2023). ProteinGym: Large-scale benchmarks for protein fitness prediction and design. *Nature Methods*.
DOI: [link](https://pubmed.ncbi.nlm.nih.gov/38106144/)

**ProteinGym Repository**:
[https://github.com/OATML-Markslab/ProteinGym](https://github.com/OATML-Markslab/ProteinGym)

## Data Structure

The `backend/folde/data` directory contains datasets and related files organized as follows:

### Directory Structure

```
backend/folde/data/
├── DMS_substitutions.csv           # Metadata file from ProteinGym (208KB)
├── FLIP-AAV_multimutant_dataset.csv # Multimutant dataset for AAV capsid (462MB)
├── DMS_ProteinGym_substitutions/   # DMS datasets from ProteinGym (~1.0GB)
│   ├── BLAT_ECOLX_Stiffler_2015.csv
│   ├── PTEN_HUMAN_Mighell_2018.csv
│   └── ... (219 total datasets)
├── embeddings/                     # Protein embeddings from ESM-2 models (~151GB)
│   ├── BLAT_ECOLX_Stiffler_2015_embedding_300m.csv
│   ├── BLAT_ECOLX_Stiffler_2015_embedding_600m.csv
│   └── ... (186 total files)
└── naturalness/                    # Protein naturalness scores from ESM-2 (~257MB)
    ├── BLAT_ECOLX_Stiffler_2015_naturalness_300m.csv
    ├── BLAT_ECOLX_Stiffler_2015_naturalness_600m.csv
    └── ... (145 total files)
```

**Data Download:** All files are available from our public GCS bucket at `gs://foldedata/`. See the Quick Start section above for download instructions.

**Original Sources:**
- DMS datasets from [ProteinGym](https://proteingym.org/) (Cheng et al., 2023)
- Embeddings & naturalness scores pre-computed using ESM-2 models (300M, 600M, 3B, 15B parameters)

### File Formats

#### DMS_substitutions.csv

This CSV file is directly from ProteinGym and contains metadata about Deep Mutational Scanning (DMS) datasets, with columns including:

- `DMS_id`: Unique identifier for each DMS dataset (e.g., "BLAT_ECOLX_Stiffler_2015")
- `DMS_filename`: Filename of the DMS data in the DMS_ProteinGym_substitutions directory
- `UniProt_ID`: UniProt identifier for the protein
- Various additional metadata columns about the dataset, protein, and experimental conditions

#### DMS Dataset Files (inside DMS_ProteinGym_substitutions/)

These files are sourced directly from ProteinGym. They are CSV files containing mutation data with columns:

- `mutant`: Mutation identifier (e.g., "H24C")
- `mutated_sequence`: Full protein sequence with mutation
- `DMS_score`: Experimental measurement of protein function/fitness
- Additional dataset-specific columns

The `mutant` column is mapped to `seq_id` in code by replacing any colons with underscores.

#### Embedding Files (inside embeddings/)

Embedding files contain protein embeddings with columns:

- `seq_id`: Sequence identifier matching the DMS dataset
- `seq`: Protein sequence
- `embedding`: Vector representation of the protein (stored as a string representation of a list)

File naming pattern: `{DMS_id}_embedding_{model_id}.csv`

#### Naturalness Files (inside naturalness/)

Naturalness files contain protein naturalness scores with columns:

- `seq_id`: Sequence identifier matching the DMS dataset
- `wt_marginal`: Naturalness score (renamed to `naturalness` in code)
- Additional columns may be present depending on the naturalness model

File naming pattern: `{DMS_id}_naturalness_{model_id}.csv`

## Usage

To use these datasets, you can load them with the provided module functions:

```python
from prediction import get_available_proteingym_datasets, get_proteingym_dataset

# Get available datasets for specific models
datasets = get_available_proteingym_datasets("300m", "esm2")

# Load a specific dataset
naturalness_df, embedding_df, activity_df = get_proteingym_dataset(
    "BLAT_ECOLX_Stiffler_2015", "300m", "esm2"
)
```

See the module documentation for more details on available functions and their usage.

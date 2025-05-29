# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

from folde.data import get_available_proteingym_datasets
from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig, ModelEvaluation, ModelDiff
from folde.util import apply_diff_list_to_config



# Example configuration
NAME = '250527-spg1-more'

base_config = FolDEModelConfig(
    name="FolDE",
    # Required parameters
    naturalness_model_id="600m",  # ESM-2 650M model
    embedding_model_id="300m",  # Same model for embeddings
    zero_shot_model_name="NaturalnessZeroShotModel",
    zero_shot_model_params={},
    # Few-shot model configuration (used after first round)
    few_shot_model_name="TorchMLPFewShotModel",
    few_shot_model_params={
        "pretrain": True,
        "pretrain_epochs": 50,

        "ensemble_size": 5,
        "decision_mode": "ucb",

        "embedding_dim": 960,
        "hidden_dims": [100, 50],
        "dropout": 0.2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "train_epochs": 200,
        "train_patience": 40,
        "val_frequency": 10,

        "do_validation_with_pair_fraction": 0.2,
    },
)

config_list = apply_diff_list_to_config(
    base_config,
    [
        ModelDiff(
            name="constantliarS05",
            diffs={
                "few_shot_model_params.decision_mode": "constantliar",
                "few_shot_model_params.lie_noise_stddev_multiplier": 0.5,
            }
        ),
        ModelDiff(
            name="constantliarS05-schedule",
            diffs={
                "few_shot_model_params.decision_mode": "constantliar",
                "few_shot_model_params.lie_noise_stddev_multiplier_schedule": [0.5] + [1000.0] * 20,
            }
        ),
        ModelDiff(
            name="constantliarS1",
            diffs={
                "few_shot_model_params.decision_mode": "constantliar",
                "few_shot_model_params.lie_noise_stddev_multiplier": 1.0,
            }
        ),
        ModelDiff(
            name="constantliarS2",
            diffs={
                "few_shot_model_params.decision_mode": "constantliar",
                "few_shot_model_params.lie_noise_stddev_multiplier": 2.0,
            }
        ),
        ModelDiff(
            name="constantliarS4",
            diffs={
                "few_shot_model_params.decision_mode": "constantliar",
                "few_shot_model_params.lie_noise_stddev_multiplier": 4.0,
            }
        ),
        ModelDiff(
            name="constantliarS2-ucb1",
            diffs={
                "few_shot_model_params.decision_mode": "constantliar",
                "few_shot_model_params.lie_noise_stddev_multiplier": 2.0,
                "few_shot_model_params.ucb_beta": 2.0,
            }
        ),
        ModelDiff(
            name="constantliarS2-reweightmin",
            diffs={
                "few_shot_model_params.decision_mode": "constantliar",
                "few_shot_model_params.lie_noise_stddev_multiplier": 2.0,
                "few_shot_model_params.importance_sampling_reweighting_strat": 'min',
                "few_shot_model_params.importance_sampling_temperature": 10.0,
            }
        ),
    ]
)


# EMBEDDING_MODEL_ID = '300m_extras'
EMBEDDING_MODEL_ID = '300m'
NATURALNESS_MODEL_ID = '600m'

print(f"Testing with embedding model: {EMBEDDING_MODEL_ID} and naturalness model: {NATURALNESS_MODEL_ID}")
available_datasets = get_available_proteingym_datasets(EMBEDDING_MODEL_ID, NATURALNESS_MODEL_ID)

assert not available_datasets.empty
print(f"Found {len(available_datasets)} available datasets:")
print(available_datasets)

print(f"Config 1/{len(config_list)}:")
print(config_list[0].model_dump_json(indent=2))

VIRUS_IDS = [
    'A0A140D2T1_ZIKV_Sourisseau_2019',
    'A0A2Z5U3Z0_9INFA_Doud_2016'
]
results = simulate_campaigns_with_config_checkpoints(
  eval_prefix=NAME,
  dms_ids=['SPG1_STRSG_Olson_2014'],# [v for v in available_datasets['DMS_id'].values if v not in VIRUS_IDS],
  config_list=config_list,
  checkpoint_dir="notebooks/jacob/model_evals",
  round_size=16,
  number_of_simulations=15,
  activity_column="DMS_score",
  max_rounds=10,
  random_seed=42,
  num_workers=5,
)
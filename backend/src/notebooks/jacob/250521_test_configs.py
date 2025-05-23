# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

from folde.data import get_available_proteingym_datasets
from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig, ModelEvaluation, ModelDiff
from folde.util import apply_diff_list_to_config



# Example configuration
NAME = '250523-final-error'

base_config = FolDEModelConfig(
    name="FolDE",
    # Required parameters
    naturalness_model_id="600m",  # ESM-2 650M model
    embedding_model_id="300m_extras",  # Same model for embeddings
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
    },
)

config_list = apply_diff_list_to_config(
    base_config,
    [
        ModelDiff(
            name="pairHoldout",
            diffs={
                "few_shot_model_params.do_validation_with_pair_fraction": 0.2,
            }
        ),
        ModelDiff(
            name="noNormal",
            diffs={
                "few_shot_model_params.disable_ensemble_normalization": True,
            }
        ),
        ModelDiff(
            name="pairHoldout-liar",
            diffs={
                "few_shot_model_params.do_validation_with_pair_fraction": 0.2,
                "few_shot_model_params.decision_mode": "constantliar",
            }
        ),
        ModelDiff(
            name="pairHoldout-krigingbeliever",
            diffs={
                "few_shot_model_params.do_validation_with_pair_fraction": 0.2,
                "few_shot_model_params.decision_mode": "krigingbeliever",
            }
        ),
        ModelDiff(
            name="liar",
            diffs={
                "few_shot_model_params.decision_mode": "constantliar",
            }
        ),
        ModelDiff(
            name="pairHoldout-nopatience",
            diffs={
                "few_shot_model_params.do_validation_with_pair_fraction": 0.2,
                "few_shot_model_params.train_patience": None,
            }
        ),
    ]
)


EMBEDDING_MODEL_ID = '300m_extras'
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
  dms_ids=[v for v in available_datasets['DMS_id'].values if v not in VIRUS_IDS],
  config_list=config_list,
  checkpoint_dir="notebooks/jacob/model_evals",
  round_size=16,
  number_of_simulations=10,
  activity_column="DMS_score",
  max_rounds=6,
  random_seed=42,
)
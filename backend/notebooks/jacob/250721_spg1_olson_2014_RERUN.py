# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import FolDEModelConfig, ModelDiff
from folde.util import apply_diff_list_to_config

dms_ids = ["SPG1_STRSG_Olson_2014"]

# Example configuration
NAME = "250812-spg1-olson-2014-RERUN2"

random_config = FolDEModelConfig(
    name="Random",
    naturalness_model_id="600m",
    embedding_model_id="300m",
    zero_shot_model_name="RandomZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="RandomFewShotModel",
    few_shot_model_params={},
)

random_forest_config = FolDEModelConfig(
    name="RandomToRandomForest",
    naturalness_model_id="600m",
    embedding_model_id="300m",
    zero_shot_model_name="RandomZeroShotModel",
    zero_shot_model_params={},
    few_shot_model_name="RandomForestFewShotModel",
    few_shot_model_params={
        "n_estimators": 100,
        "criterion": "friedman_mse",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": 1.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": None,
        "verbose": 0,
        "warm_start": False,
        "ccp_alpha": 0.0,
        "max_samples": None,
    },
)

folde_config = FolDEModelConfig(
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
        "lie_noise_stddev_multiplier": 2.0,
    },
)

config_list = (
    apply_diff_list_to_config(
        folde_config,
        [
            ModelDiff(name="1-vs-rest", diffs={"data_split_mode": "one_vs_many_split"}),
            ModelDiff(name="1-vs-rest-redo", diffs={"data_split_mode": "one_vs_many_split"}),
            ModelDiff(
                name="1-vs-rest-noCL",
                diffs={
                    "data_split_mode": "one_vs_many_split",
                    "few_shot_model_params.decision_mode": "mean",
                },
            ),
        ],
    )
    + apply_diff_list_to_config(
        random_config,
        [
            ModelDiff(name="1-vs-rest", diffs={"data_split_mode": "one_vs_many_split"}),
        ],
    )
    + apply_diff_list_to_config(
        random_forest_config,
        [
            ModelDiff(name="1-vs-rest", diffs={"data_split_mode": "one_vs_many_split"}),
        ],
    )
)

print(f"Config 1/{len(config_list)}:")
print(config_list[0].model_dump_json(indent=2))

results = simulate_campaigns_with_config_checkpoints(
    eval_prefix=NAME,
    dms_ids=dms_ids,
    config_list=config_list,
    checkpoint_dir="notebooks/jacob/model_evals",
    round_size=16,
    number_of_simulations=10,
    activity_column="DMS_score",
    max_rounds=6,
    random_seed=42,
    num_workers=5,
)

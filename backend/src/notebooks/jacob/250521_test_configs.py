# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.data import get_available_proteingym_datasets
from folde.types import FolDEModelConfig, ModelDiff, ModelEvaluation
from folde.util import apply_diff_list_to_config

evolvepro_dms_ids = [
    "A0A140D2T1_ZIKV_Sourisseau_2019",
    "A0A2Z5U3Z0_9INFA_Doud_2016",
    "ADRB2_HUMAN_Jones_2020",
    "BLAT_ECOLX_Stiffler_2015",
    "C6KNH7_9INFA_Lee_2018",
    "IF1_ECOLI_Kelsic_2016",
    "MK01_HUMAN_Brenan_2016",
    "P53_HUMAN_Giacomelli_2018_Null_Etoposide",
    "PHOT_CHLRE_Chen_2023",
]

folde_train_dms_ids = [
    "ANCSZ_Hobbs_2022",
    "BLAT_ECOLX_Firnberg_2014",
    "CBS_HUMAN_Sun_2020",
    "HEM3_HUMAN_Loggerenberg_2023",
    "HSP82_YEAST_Flynn_2019",
    "HXK4_HUMAN_Gersing_2022_activity",
    "OXDA_RHOTO_Vanella_2023_activity",
    "PPM1D_HUMAN_Miller_2022",
    "SHOC2_HUMAN_Kwon_2022",
]

folde_test_dms_ids = [
    "ADRB2_HUMAN_Jones_2020",
    "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
    "MK01_HUMAN_Brenan_2016",
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    "CAS9_STRP1_Spencer_2017_positive",
    "SC6A4_HUMAN_Young_2021",
    # 'OXDA_RHOTO_Vanella_2023_expression',
    # 'HSP82_YEAST_Mishra_2016',
    "PTEN_HUMAN_Mighell_2018",
    "S22A1_HUMAN_Yee_2023_activity",
    "KKA2_KLEPN_Melnikov_2014",
    # Include some "easy to engineer" targets.
    "PPARG_HUMAN_Majithia_2016",
    # 'P53_HUMAN_Giacomelli_2018_Null_Etoposide',
    "MET_HUMAN_Estevam_2023",
    "MTHR_HUMAN_Weile_2021",
    "LGK_LIPST_Klesmith_2015",
    "AMIE_PSEAE_Wrenbeck_2017",
]

# Example configuration
NAME = "250711-folde-testset-wCL"

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

config_list = apply_diff_list_to_config(
    base_config,
    [
        ModelDiff(
            name="random",
            diffs={
                "zero_shot_model_name": "RandomZeroShotModel",
                "zero_shot_model_params": {},
                "few_shot_model_name": "RandomFewShotModel",
                "few_shot_model_params": {},
            },
        ),
        ModelDiff(
            name="Random-RandomForestCtrl",
            diffs={
                "zero_shot_model_name": "RandomZeroShotModel",
                "zero_shot_model_params": {},
                "few_shot_model_name": "RandomForestFewShotModel",
                "few_shot_model_params": {
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
            },
        ),
        ModelDiff(
            name="600m-embeddings",
            diffs={
                "embedding_model_id": "600m",
                "few_shot_model_params.embedding_dim": 1152,
            },
        ),
        ModelDiff(
            name="650m-embeddings",
            diffs={
                "embedding_model_id": "650m",
                "few_shot_model_params.embedding_dim": 1280,
            },
        ),
    ],
)


# # EMBEDDING_MODEL_ID = '300m_extras'
# EMBEDDING_MODEL_ID = "300m"
# NATURALNESS_MODEL_ID = "600m"

# print(f"Testing with embedding model: {EMBEDDING_MODEL_ID} and naturalness model: {NATURALNESS_MODEL_ID}")
# available_datasets = get_available_proteingym_datasets(EMBEDDING_MODEL_ID, NATURALNESS_MODEL_ID)

# assert not available_datasets.empty
# print(f"Found {len(available_datasets)} available datasets:")
# print(available_datasets)

print(f"Config 1/{len(config_list)}:")
print(config_list[0].model_dump_json(indent=2))

VIRUS_IDS = ["A0A140D2T1_ZIKV_Sourisseau_2019", "A0A2Z5U3Z0_9INFA_Doud_2016"]
results = simulate_campaigns_with_config_checkpoints(
    eval_prefix=NAME,
    dms_ids=folde_test_dms_ids,  # ['SPG1_STRSG_Olson_2014'],# [v for v in available_datasets['DMS_id'].values if v not in VIRUS_IDS],
    config_list=config_list,
    checkpoint_dir="notebooks/jacob/model_evals",
    round_size=16,
    number_of_simulations=10,
    activity_column="DMS_score",
    max_rounds=6,
    random_seed=42,
    num_workers=10,
)

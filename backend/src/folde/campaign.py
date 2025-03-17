"""
Campaign simulation functions for protein engineering prediction tasks.

This module provides functions for simulating protein engineering campaigns
and evaluating different model configurations.
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import logging
import random
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from scipy.stats import spearmanr
from pydantic import BaseModel, Field

from folde.data import get_proteingym_dataset
from folde.few_shot_models import get_few_shot_model
from folde.zero_shot_models import get_zero_shot_model

logger = logging.getLogger(__name__)


class FolDEModelConfig(BaseModel):
    naturalness_model_id: str
    embedding_model_id: str
    zeroshot_model_name: str
    zeroshot_model_params: Dict[str, Any]
    fewshot_model_name: str
    fewshot_model_params: Dict[str, Any]


class SimulationResult(BaseModel):
    rounds: int
    best_variant_per_round: List[str]
    best_activity_per_round: List[float]
    cumulative_best_activity: List[float]
    cumulative_best_percentile: List[float]
    tested_variants: List[List[str]]
    tested_variant_percentiles: List[List[float]]
    round_metrics: List[Dict[str, Any]]


class CampaignResults(BaseModel):
    dms_id: str
    round_size: int
    number_of_simulations: int
    activity_column: str
    max_rounds: int
    random_seed: int
    simulation_results: List[Tuple[FolDEModelConfig, List[SimulationResult]]]


def _evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for predictions.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = {}

    # Regression metrics
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    if len(np.unique(y_pred)) > 1:
        metrics["pearson"] = np.corrcoef(y_true, y_pred)[0, 1]
        metrics["spearman"] = pd.Series(y_true).corr(
            pd.Series(y_pred), method="spearman"
        )
    else:
        logging.warning(f"The predicted activities were degenerate: {y_pred}")
        metrics["pearson"] = None
        metrics["spearman"] = None

    return metrics


class CampaignWorldState:
    def __init__(
        self,
        golden_activity_df: pd.DataFrame,
        naturalness_df: pd.DataFrame,
        embedding_df: pd.DataFrame,
    ):
        assert golden_activity_df.index.name == "seq_id"
        assert naturalness_df.index.name == "seq_id"
        assert embedding_df.index.name == "seq_id"
        assert golden_activity_df.index.equals(naturalness_df.index)
        assert golden_activity_df.index.equals(embedding_df.index)
        self.golden_activity_df = golden_activity_df.copy()
        self.naturalness_df = naturalness_df.copy()
        self.embedding_df = embedding_df.copy()
        self.measured_seq_ids = []

    def measure_variant_activities(self, seq_ids: List[str]):
        """Adds seq ids to the collection of measured samples."""
        self.measured_seq_ids.extend(seq_ids)

    def get_unmeasured_variants_activity_df(self) -> pd.DataFrame:
        return self.golden_activity_df[
            ~self.golden_activity_df["seq_id"].isin(self.measured_seq_ids)
        ]

    def get_unmeasured_naturalness_df(self) -> pd.DataFrame:
        return self.naturalness_df[
            ~self.naturalness_df["seq_id"].isin(self.measured_seq_ids)
        ]

    def get_unmeasured_embeddings_df(self) -> pd.DataFrame:
        return self.embedding_df[
            ~self.embedding_df["seq_id"].isin(self.measured_seq_ids)
        ]

    def get_measured_activity_df(self) -> pd.DataFrame:
        return self.golden_activity_df.loc[self.measured_seq_ids]

    def get_measured_naturalness_df(self) -> pd.DataFrame:
        return self.naturalness_df.loc[self.measured_seq_ids]

    def get_measured_embeddings_df(self) -> pd.DataFrame:
        return self.embedding_df.loc[self.measured_seq_ids]


def _run_single_simulation(
    golden_activity_df: pd.DataFrame,
    naturalness_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    round_size: int,
    config: FolDEModelConfig,
    activity_column: str = "DMS_score",
    max_rounds: int = 10,
) -> SimulationResult:
    """Run a single campaign simulation.

    Args:
        golden_activity_df: DataFrame with ground truth activity data
        naturalness_df: DataFrame with naturalness scores
        embedding_df: DataFrame with embeddings
        round_size: Number of variants to test in each round
        config: Model configuration
        activity_column: Column in golden_activity_df containing activity values
        max_rounds: Maximum number of rounds to simulate

    Returns:
        Dictionary with simulation results
    """
    # Initialize results tracking
    results = SimulationResult(
        config=config,
        rounds=0,
        best_variant_per_round=[],
        best_activity_per_round=[],
        cumulative_best_activity=[],
        cumulative_best_percentile=[],
        tested_variants=[],
        tested_variant_percentiles=[],
        round_metrics=[],
    )

    # Make sure we have appropriate columns
    if "seq_id" not in golden_activity_df.columns:
        raise ValueError("golden_activity_df must contain 'seq_id' column")

    if activity_column not in golden_activity_df.columns:
        raise ValueError(f"golden_activity_df must contain '{activity_column}' column")

    world_state = CampaignWorldState(golden_activity_df, naturalness_df, embedding_df)

    best_activity_so_far = None

    # Run the simulation for the specified number of rounds
    for round_num in range(1, max_rounds + 1):
        logger.info(f"Running round {round_num}")

        if world_state.get_unmeasured_variants_activity_df().shape[0] == 0:
            logger.info("No more unmeasured variants, ending simulation")
            break

        # We're getting a topN to synthesize, and a predicted activity for every variant.
        top_seq_ids = None
        all_predicted_activities = None

        # First round: always use zero-shot model
        if round_num == 1:
            # Get the zero-shot model
            zeroshot_model = get_zero_shot_model(
                config.zeroshot_model_name, **config.zeroshot_model_params
            )
            # Get top variants using zero-shot model's get_top_n method
            top_seq_ids = zeroshot_model.get_top_n(
                round_size,
                world_state.get_unmeasured_naturalness_df(),
                world_state.get_unmeasured_embeddings_df(),
            )
            all_predicted_activities = zeroshot_model.predict(
                naturalness_df, embedding_df
            )

        # Subsequent rounds: use few-shot model if specified
        else:

            # Get few-shot model
            fewshot_model = get_few_shot_model(
                config.fewshot_model_name, **config.fewshot_model_params
            )

            # Convert list of embeddings to numpy array
            train_naturalness = world_state.get_measured_naturalness_df()
            train_embeddings = world_state.get_measured_embeddings_df()
            train_embeddings_array = np.array(
                [np.array(emb) for emb in train_embeddings.embedding.values]
            )

            fewshot_model.fit(
                train_embeddings_array,
                world_state.get_measured_activity_df()[activity_column].to_numpy(),
            )

            # Use the get_top_n method from FewShotModel
            top_seq_ids = fewshot_model.get_top_n(
                round_size,
                world_state.get_unmeasured_naturalness_df(),
                world_state.get_unmeasured_embeddings_df(),
            )
            all_embeddings_array = np.array(
                [np.array(emb) for emb in embedding_df.embedding.values]
            )
            all_predicted_activities = fewshot_model.predict(all_embeddings_array)

        assert (
            len(top_seq_ids) == round_size
        ), f"Must choose {round_size} variants per rounds, only chose {len(top_seq_ids)}"
        logging.info(
            f"In Round {round_num}: elected {len(top_seq_ids)} variants: {','.join(top_seq_ids)}"
        )
        world_state.measure_variant_activities(top_seq_ids)

        # Track results for this round
        selected_activity_df = world_state.get_measured_activity_df().loc[top_seq_ids]
        best_in_round = selected_activity_df.loc[
            selected_activity_df[activity_column].idxmax()
        ]
        best_in_round_activity = best_in_round[activity_column]

        all_percentiles = golden_activity_df[activity_column].rank(pct=True)
        top_seq_id_percentiles = [all_percentiles.loc[seq_id] for seq_id in top_seq_ids]

        if (
            best_activity_so_far == None
            or best_in_round_activity > best_activity_so_far
        ):
            best_activity_so_far = best_in_round_activity

        # Compute metrics for this round's predictions
        # TOOD(jacob): Compute metrics for every round, eg validation or test correlation.
        whole_dataset_spearman = spearmanr(
            golden_activity_df[activity_column].values, all_predicted_activities
        )[0]
        round_metrics = {
            "whole_dataset_spearman": whole_dataset_spearman,
        }

        # Store round results
        results.rounds = round_num
        results.best_variant_per_round.append(
            best_in_round["seq_id"] if best_in_round is not None else None
        )
        results.best_activity_per_round.append(best_in_round_activity)
        results.cumulative_best_activity.append(best_activity_so_far)
        results.cumulative_best_percentile.append(
            np.mean(golden_activity_df[activity_column].values <= best_activity_so_far)
        )
        results.tested_variants.append(top_seq_ids)
        results.tested_variant_percentiles.append(top_seq_id_percentiles)
        results.round_metrics.append(round_metrics)

    return results


def simulate_campaign(
    dms_id: str,
    round_size: int,
    number_of_simulations: int,
    config_list: List[FolDEModelConfig],
    activity_column: str = "DMS_score",
    max_rounds: int = 10,
    random_seed: int = 42,
) -> CampaignResults:
    """Simulate protein engineering campaigns with different model configurations.

    Args:
        dms_id: Identifier for the DMS dataset to use
        round_size: Number of variants to evaluate in each round
        number_of_simulations: Number of times to run the simulation
        config_list: List of model configurations to evaluate
        activity_column: Column in the dataset containing activity values
        max_rounds: Maximum number of rounds to simulate
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing simulation results for each configuration
    """

    # Initialize results
    campaign_results = CampaignResults(
        dms_id=dms_id,
        round_size=round_size,
        number_of_simulations=number_of_simulations,
        activity_column=activity_column,
        max_rounds=max_rounds,
        random_seed=random_seed,
        configurations=[],
        simulation_results=[],
    )

    # Run simulations for each configuration
    for config_idx, model_config in enumerate(config_list):
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info(
            f"Running simulations for configuration {config_idx+1}/{len(config_list)}"
        )
        logger.info(f"Config: {model_config}")

        single_model_campaign_results = []

        # Load dataset for this configuration
        try:
            naturalness_df, embedding_df, activity_df = get_proteingym_dataset(
                dms_id,
                model_config.embedding_model_id,
                model_config.naturalness_model_id,
            )
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            import traceback

            print(traceback.format_exc(), flush=True)
            raise e

        # Check that the activity column exists
        if activity_column not in activity_df.columns:
            activity_column = next(
                (col for col in activity_df.columns if "score" in col.lower()), None
            )
            if activity_column is None:
                logger.error(
                    f"Activity column not found in dataset: {activity_df.columns}"
                )
                continue
            logger.info(f"Using detected activity column: {activity_column}")

        # Run multiple simulations
        for sim_idx in range(number_of_simulations):
            logger.info(f"Running simulation {sim_idx+1}/{number_of_simulations}")

            bootstrapped_seq_ids = np.random.choice(
                activity_df.index.values,
                size=int(len(activity_df) * 0.5),
                replace=False,
            )

            sim_result = _run_single_simulation(
                activity_df.loc[bootstrapped_seq_ids],
                naturalness_df.loc[bootstrapped_seq_ids],
                embedding_df.loc[bootstrapped_seq_ids],
                round_size,
                model_config,
                activity_column,
                max_rounds,
            )
            single_model_campaign_results.append(sim_result)

        campaign_results.simulation_results.append(
            (model_config, single_model_campaign_results)
        )

    return campaign_results

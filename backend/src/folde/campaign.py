"""
Campaign simulation functions for protein engineering prediction tasks.

This module provides functions for simulating protein engineering campaigns
and evaluating different model configurations.
"""

import logging
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from folde.data import get_proteingym_dataset
from folde.few_shot_models import get_few_shot_model
from folde.types import (
    CampaignResult,
    FolDEModelConfig,
    ModelEvaluation,
    MutantMetrics,
    RoundMetrics,
    SimulationResult,
    SingleConfigCampaignResult,
)
from folde.zero_shot_models import get_zero_shot_model
from pydantic import BaseModel, Field
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score

logger = logging.getLogger(__name__)


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
        metrics["spearman"] = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
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
        self.measured_seq_ids: List[str] = []

    def measure_variant_activities(self, seq_ids: List[str]):
        """Adds seq ids to the collection of measured samples."""
        assert len(set(seq_ids)) == len(seq_ids), f"seq_ids must be unique, got {seq_ids}"
        for seq_id in seq_ids:
            assert type(seq_id) == str, f"seq_id must be a string, got {type(seq_id)} ({seq_id})"
            assert seq_id not in self.measured_seq_ids, f"seq_id {seq_id} already measured"
        self.measured_seq_ids.extend(seq_ids)

    def get_unmeasured_variants_activity_df(self) -> pd.DataFrame:
        return self.golden_activity_df[
            ~self.golden_activity_df["seq_id"].isin(self.measured_seq_ids)
        ]

    def get_unmeasured_naturalness_df(self) -> pd.DataFrame:
        return self.naturalness_df[~self.naturalness_df["seq_id"].isin(self.measured_seq_ids)]

    def get_unmeasured_embeddings_df(self) -> pd.DataFrame:
        return self.embedding_df[~self.embedding_df["seq_id"].isin(self.measured_seq_ids)]

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
        variant_pool_size=len(golden_activity_df),
        mutant_metrics=[],
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
        logger.debug(f"Running round {round_num}")

        if world_state.get_unmeasured_variants_activity_df().shape[0] == 0:
            logger.info("No more unmeasured variants, ending simulation")
            break

        # We're getting a topN to synthesize, and a predicted activity for every variant.
        top_seq_ids = None
        predicted_activity_df = None

        # First round: always use zero-shot model
        if round_num == 1:
            # Get the zero-shot model
            zero_shot_model = get_zero_shot_model(
                config.zero_shot_model_name, **config.zero_shot_model_params
            )
            # Get top variants using zero-shot model's get_top_n method
            top_seq_ids, predicted_activity_series = zero_shot_model.get_top_n(
                round_size,
                world_state.get_unmeasured_naturalness_df(),
                world_state.get_unmeasured_embeddings_df(),
            )

        # Subsequent rounds: use few-shot model if specified
        else:

            # Get few-shot model
            few_shot_model = get_few_shot_model(
                config.few_shot_model_name, **config.few_shot_model_params
            )

            # Convert list of embeddings to numpy array
            train_naturalness_df = world_state.get_measured_naturalness_df()
            train_embedding_df = world_state.get_measured_embeddings_df()
            train_activity_df = world_state.get_measured_activity_df()

            train_embeddings_array = np.array(
                [np.array(emb) for emb in train_embedding_df.embedding.values]
            )
            few_shot_model.fit(
                train_embeddings_array,
                train_activity_df[activity_column].to_numpy(),
            )

            # Use the get_top_n method from FewShotModel
            top_seq_ids, predicted_activity_series = few_shot_model.get_top_n(
                round_size,
                world_state.get_unmeasured_naturalness_df(),
                world_state.get_unmeasured_embeddings_df(),
            )

        assert type(top_seq_ids) == list, f"top_seq_ids must be a list, got {type(top_seq_ids)}"
        assert (
            len(top_seq_ids) == round_size
        ), f"Must choose {round_size} variants per rounds, only chose {len(top_seq_ids)}"
        logging.debug(
            f"In Round {round_num}: elected {len(top_seq_ids)} variants: {','.join(top_seq_ids)}"
        )
        world_state.measure_variant_activities(top_seq_ids)

        all_percentiles = golden_activity_df[activity_column].rank(pct=True)

        mutant_metrics_list = []
        for top_seq_id in top_seq_ids:
            # Get the activity from the dataframe and convert to float if needed
            golden_activity = world_state.get_measured_activity_df().loc[top_seq_id][
                activity_column
            ]
            assert (
                type(golden_activity) == float or type(golden_activity) == np.float64
            ), f"golden_activity must be a float, got {type(golden_activity)}"
            percentile = all_percentiles.loc[top_seq_id]
            mutant_metrics_list.append(
                MutantMetrics(
                    seq_id=top_seq_id,
                    round_found=round_num,
                    activity=golden_activity,
                    predicted_activity=predicted_activity_series.loc[top_seq_id],
                    percentile=percentile,
                    relevant_mutants=[],  # TODO(jacob): Compute relevant mutants
                )
            )

        # Compute metrics for this round's predictions
        # TOOD(jacob): Compute metrics for every round, eg validation or test correlation.
        whole_dataset_spearman = spearmanr(
            golden_activity_df.loc[predicted_activity_series.index][activity_column].values,
            predicted_activity_series.values,
        )[0]
        round_metrics = RoundMetrics(
            round_num=round_num,
            model_spearman=whole_dataset_spearman,
            misc={},
        )

        results.rounds = round_num
        results.round_metrics.append(round_metrics)
        results.mutant_metrics.extend(mutant_metrics_list)

    return results


# Run multiple simulations
def run_single_sim_parallel(
    sim_idx,
    activity_df_subset,
    naturalness_df_subset,
    embedding_df_subset,
    random_seed,
    **kwargs,
):
    logger.info(f"Running simulation {sim_idx+1}")
    random.seed(random_seed + sim_idx)
    np.random.seed(random_seed + sim_idx)

    sim_result = _run_single_simulation(
        activity_df_subset,
        naturalness_df_subset,
        embedding_df_subset,
        **kwargs,
    )
    return sim_result


def simulate_campaign(
    dms_id: str,
    round_size: int,
    number_of_simulations: int,
    config_list: List[FolDEModelConfig],
    activity_column: str = "DMS_score",
    max_rounds: int = 10,
    random_seed: int = 42,
) -> CampaignResult:
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
    campaign_result = CampaignResult(
        dms_id=dms_id,
        round_size=round_size,
        number_of_simulations=number_of_simulations,
        activity_column=activity_column,
        max_rounds=max_rounds,
        random_seed=random_seed,
        config_results=[],
    )

    df_cache = {}

    # Run simulations for each configuration
    for config_idx, model_config in enumerate(config_list):
        # Set random seed for reproducibility
        logger.info(f"Running simulations for configuration {config_idx+1}/{len(config_list)}")
        logger.info(f"Config: {model_config}")

        # Load dataset for this configuration
        cache_key = (model_config.embedding_model_id, model_config.naturalness_model_id)
        if cache_key not in df_cache:
            try:
                df_cache[cache_key] = get_proteingym_dataset(
                    dms_id,
                    model_config.embedding_model_id,
                    model_config.naturalness_model_id,
                )
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                import traceback

                print(traceback.format_exc(), flush=True)
                raise e
        naturalness_df, embedding_df, activity_df = df_cache[cache_key]

        # Check that the activity column exists
        if activity_column not in activity_df.columns:
            raise ValueError(
                f"Activity column {activity_column} not found in dataset: {activity_df.columns}"
            )

        single_model_campaign_results = None
        with ProcessPoolExecutor() as executor:
            futures = []
            for sim_idx in range(number_of_simulations):
                bootstrapped_seq_ids = np.random.choice(
                    activity_df.index.values,
                    size=int(len(activity_df) * 0.5),
                    replace=False,
                )
                futures.append(
                    executor.submit(
                        run_single_sim_parallel,
                        sim_idx,
                        activity_df.loc[bootstrapped_seq_ids],
                        naturalness_df.loc[bootstrapped_seq_ids],
                        embedding_df.loc[bootstrapped_seq_ids],
                        random_seed=random_seed,
                        round_size=round_size,
                        config=model_config,
                        activity_column=activity_column,
                        max_rounds=max_rounds,
                    )
                )
            single_model_campaign_results = [f.result() for f in futures]

        campaign_result.config_results.append(
            SingleConfigCampaignResult(
                config=model_config,
                simulation_results=single_model_campaign_results,
            )
        )

    return campaign_result


def simulate_campaigns(name: str, dms_ids: List[str], **kwargs) -> ModelEvaluation:
    """Run a list of campaigns."""
    results = ModelEvaluation(name=name, campaign_results=[])
    for dms_id in dms_ids:
        results.campaign_results.append(simulate_campaign(dms_id, **kwargs))
    return results

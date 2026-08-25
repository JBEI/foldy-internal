"""
Campaign simulation functions for protein engineering prediction tasks.

This module provides functions for simulating protein engineering campaigns
and evaluating different model configurations.
"""

import json
import logging
import multiprocessing
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score

from app.helpers.sequence_util import get_loci_set, is_homolog_seq_id
from folde.benchmarks.multimutant_oracle import ProteinGymFitnessOracle
from folde.candidate_generation.base import CandidatePoolStrategy, GeneratorContext
from folde.candidate_generation.strategy import derive_component_seed
from folde.data import get_proteingym_dataset
from folde.few_shot_models import get_consensus_scores, get_few_shot_model
from folde.rust_mutant_pool import get_mutant_pool as get_accelerated_mutant_pool
from folde.types import (
    CampaignResult,
    FolDEModelConfig,
    ModelEvaluation,
    MutantMetrics,
    RoundMetrics,
    SimulationResult,
    SingleConfigCampaignResult,
)
from folde.util import (
    get_consensus_scores,
    get_top_percentile_recall_score,
    get_top_percentile_recall_score_slate,
    top_k_mask,
)
from folde.zero_shot_models import get_zero_shot_model

logger = logging.getLogger(__name__)

_WORKER_ACTIVITY_SERIES: pd.Series | None = None
_WORKER_NATURALNESS_DF: pd.DataFrame | None = None
_WORKER_EMBEDDING_SERIES: pd.Series | None = None
_WORKER_PRETRAIN_NATURALNESS_DF: pd.DataFrame | None = None
_WORKER_FULL_SEQ_IDS: list[str] | None = None


def _initialize_simulation_worker(
    activity_series: pd.Series,
    naturalness_df: pd.DataFrame,
    embedding_series: pd.Series,
    pretrain_naturalness_df: pd.DataFrame,
    full_seq_ids: list[str],
    torch_threads: int,
) -> None:
    """Initialize immutable dataset state once in each spawned worker."""
    global _WORKER_ACTIVITY_SERIES
    global _WORKER_NATURALNESS_DF
    global _WORKER_EMBEDDING_SERIES
    global _WORKER_PRETRAIN_NATURALNESS_DF
    global _WORKER_FULL_SEQ_IDS

    _WORKER_ACTIVITY_SERIES = activity_series
    _WORKER_NATURALNESS_DF = naturalness_df
    _WORKER_EMBEDDING_SERIES = embedding_series
    _WORKER_PRETRAIN_NATURALNESS_DF = pretrain_naturalness_df
    _WORKER_FULL_SEQ_IDS = full_seq_ids
    torch.set_num_threads(torch_threads)


def _evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """Calculate the legacy regression metrics used by campaign tests and reports."""
    mse = float(mean_squared_error(y_true, y_pred))
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        pearson = None
        spearman = None
    else:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "pearson": pearson,
        "spearman": spearman,
    }


class CampaignWorldState:
    def __init__(
        self,
        golden_activity_series: pd.Series,
        naturalness_df: pd.DataFrame,
        embedding_series: pd.Series,
        one_mutation_at_a_time: bool = False,
    ):
        assert golden_activity_series.index.equals(naturalness_df.index)
        assert golden_activity_series.index.equals(embedding_series.index)
        # Removing these copies shaves ~25% time off the faster runs, and more when
        # my laptop is memory constrained...
        self.golden_activity_series = golden_activity_series  # .copy()
        self.naturalness_df = naturalness_df  # .copy()
        self.naturalness_series = naturalness_df  # Backward-compatible attribute name.
        self.embedding_series = embedding_series  # .copy()
        self.measured_seq_ids: List[str] = []
        self.measured_seq_id_set: set[str] = set()
        self.one_mutation_at_a_time = one_mutation_at_a_time

    def measure_variant_activities(self, seq_ids: List[str]):
        """Adds seq ids to the collection of measured samples."""
        assert len(set(seq_ids)) == len(seq_ids), f"seq_ids must be unique, got {seq_ids}"
        for seq_id in seq_ids:
            assert type(seq_id) == str, f"seq_id must be a string, got {type(seq_id)} ({seq_id})"
            assert seq_id not in self.measured_seq_id_set, f"seq_id {seq_id} already measured"
        self.measured_seq_ids.extend(seq_ids)
        self.measured_seq_id_set.update(seq_ids)

    def get_mutant_pool(self) -> List[str]:
        if self.one_mutation_at_a_time:
            return get_accelerated_mutant_pool(
                self.golden_activity_series.index.tolist(),
                self.measured_seq_ids,
            )
        else:
            return [
                s
                for s in self.golden_activity_series.index.tolist()
                if s not in self.measured_seq_id_set
            ]

    def get_unmeasured_activity_series(self) -> pd.Series:
        return self.golden_activity_series.loc[self.get_mutant_pool()]

    def get_unmeasured_variants_activity_df(self) -> pd.Series:
        """Backward-compatible alias for the unmeasured activity series."""
        return self.get_unmeasured_activity_series()

    def get_unmeasured_naturalness_df(self) -> pd.DataFrame:
        return self.naturalness_df.loc[pd.Index(self.get_mutant_pool())]

    def get_unmeasured_naturalness_series(self) -> pd.Series | pd.DataFrame:
        """Backward-compatible alias retained for older campaign callers."""
        return self.get_unmeasured_naturalness_df()

    def get_unmeasured_embeddings_series(self) -> pd.Series:
        return self.embedding_series.loc[self.get_mutant_pool()]

    def get_measured_activity_series(self) -> pd.Series:
        return self.golden_activity_series.loc[self.measured_seq_ids]

    def get_measured_naturalness_df(self) -> pd.DataFrame:
        return self.naturalness_df.loc[pd.Index(self.measured_seq_ids)]

    def get_measured_naturalness_series(self) -> pd.Series | pd.DataFrame:
        """Backward-compatible alias retained for older campaign callers."""
        return self.get_measured_naturalness_df()

    def get_measured_embeddings_series(self) -> pd.Series:
        return self.embedding_series.loc[self.measured_seq_ids]


def _run_single_simulation(
    available_seq_ids: List[str],
    entire_activity_series: pd.Series,
    entire_naturalness_df: pd.DataFrame,
    entire_embedding_series: pd.Series,
    round_size: int,
    config: FolDEModelConfig,
    random_seed: int,
    wt_aa_seq: str = "",
    pretrain_naturalness_df: pd.DataFrame | None = None,
    max_rounds: int = 10,
    candidate_pool_strategy: CandidatePoolStrategy | None = None,
    proposal_budget: int | None = None,
    candidate_min_mutation_depth: int = 0,
    candidate_max_mutation_depth: int | None = None,
) -> SimulationResult:
    """Run a single campaign simulation.

    Args:
        golden_activity_series: Series with ground truth activity data (some of which may be NAN)
        naturalness_df: DataFrame with naturalness scores (some of which may be NAN)
        embedding_series: Series with embeddings
        round_size: Number of variants to test in each round
        config: Model configuration
        max_rounds: Maximum number of rounds to simulate

    Returns:
        Dictionary with simulation results
    """
    # Initialize results tracking
    results = SimulationResult(
        rounds=0,
        variant_pool_size=len(available_seq_ids),
        mutant_metrics=[],
        round_metrics=[],
    )

    if isinstance(entire_naturalness_df, pd.Series):
        entire_naturalness_df = entire_naturalness_df.to_frame(
            name=entire_naturalness_df.name or "naturalness"
        )
    assert entire_naturalness_df.index.equals(entire_activity_series.index)
    assert entire_naturalness_df.index.equals(entire_embedding_series.index)
    if pretrain_naturalness_df is None:
        pretrain_naturalness_df = entire_naturalness_df
    assert pretrain_naturalness_df.index.equals(entire_embedding_series.index)

    whole_world_activity_series = entire_activity_series.loc[available_seq_ids]
    whole_world_naturalness_df = entire_naturalness_df.loc[pd.Index(available_seq_ids)]
    whole_world_embedding_series = entire_embedding_series.loc[available_seq_ids]
    assert (
        not whole_world_activity_series.isna().any()
    ), f'{whole_world_activity_series.isna().sum()} activity values in the "whole world" set are NAN'
    world_state = CampaignWorldState(
        whole_world_activity_series,
        whole_world_naturalness_df,
        whole_world_embedding_series,
        config.one_mutation_at_a_time,
    )
    all_percentiles = whole_world_activity_series.rank(pct=True)
    oracle = (
        ProteinGymFitnessOracle(
            wt_aa_seq,
            whole_world_activity_series,
            eligible_seq_ids=available_seq_ids,
        )
        if candidate_pool_strategy is not None
        else None
    )
    candidate_positions: frozenset[int] = frozenset()
    candidate_alphabet: frozenset[str] = frozenset()
    inferred_max_depth = 0
    if candidate_pool_strategy is not None:
        candidate_positions = frozenset(
            locus
            for seq_id in available_seq_ids
            if seq_id != "WT" and not is_homolog_seq_id(seq_id)
            for locus in get_loci_set(seq_id)
        )
        candidate_alphabet = frozenset(
            allele[-1]
            for seq_id in available_seq_ids
            if seq_id != "WT" and not is_homolog_seq_id(seq_id)
            for allele in seq_id.split("_")
        )
        inferred_max_depth = max(
            (
                len(get_loci_set(seq_id))
                for seq_id in available_seq_ids
                if seq_id != "WT" and not is_homolog_seq_id(seq_id)
            ),
            default=0,
        )

    def is_single_mutant_id(seq_id: str) -> bool:
        if seq_id == "WT" or is_homolog_seq_id(seq_id):
            return False
        try:
            return len(get_loci_set(seq_id)) == 1
        except (TypeError, ValueError):
            # Preserve support for legacy/custom datasets with opaque identifiers.
            return True

    single_mutant_seq_ids = [
        seq_id for seq_id in pretrain_naturalness_df.index if is_single_mutant_id(seq_id)
    ]

    pretraining_naturalness_df = pretrain_naturalness_df.loc[pd.Index(single_mutant_seq_ids)]
    pretraining_embedding_series = entire_embedding_series.loc[single_mutant_seq_ids]
    assert (
        not pretraining_naturalness_df.isna()
        .any()
        .any()  # Make sure SOME naturalness values are not NAN,
    ), f'{pretraining_naturalness_df.isna().sum()}/{len(pretraining_naturalness_df)} naturalness values in the "pretraining" set are NAN'

    held_out_series = (
        ~entire_activity_series.index.isin(available_seq_ids) & ~entire_activity_series.isna()
    )
    held_out_activity_series = entire_activity_series.loc[held_out_series]
    held_out_naturalness_df = entire_naturalness_df.loc[pd.Index(held_out_series)]
    held_out_embedding_series = entire_embedding_series.loc[held_out_series]
    has_held_out_slate = len(held_out_activity_series) >= round_size

    # Get the zero-shot model
    zero_shot_model = get_zero_shot_model(
        config.zero_shot_model_name, **config.zero_shot_model_params
    )

    # Get few-shot model
    few_shot_model = None
    if max_rounds > 1:
        if wt_aa_seq:
            few_shot_model = get_few_shot_model(
                config.few_shot_model_name,
                random_state=random_seed,
                wt_aa_seq=wt_aa_seq,
                **config.few_shot_model_params,
            )
        else:
            few_shot_model = get_few_shot_model(
                config.few_shot_model_name, **config.few_shot_model_params
            )

    zero_shot_model.pretrain(
        pretraining_naturalness_df,
        pretraining_embedding_series,
    )
    if few_shot_model is not None:
        few_shot_model.pretrain(
            pretraining_naturalness_df,
            pretraining_embedding_series,
        )

    # Run the simulation for the specified number of rounds
    for round_num in range(1, max_rounds + 1):
        logger.debug(f"Running round {round_num}")

        round_proposals = None
        if candidate_pool_strategy is None:
            proposal_seq_ids = world_state.get_mutant_pool()
        else:
            assert oracle is not None
            context = GeneratorContext(
                reference_sequence=wt_aa_seq,
                measured_variants=oracle.measured_variants,
                allowed_positions=candidate_positions,
                allowed_alphabet=candidate_alphabet,
                min_mutation_depth=candidate_min_mutation_depth,
                max_mutation_depth=(
                    inferred_max_depth
                    if candidate_max_mutation_depth is None
                    else candidate_max_mutation_depth
                ),
                proposal_budget=(
                    len(available_seq_ids) if proposal_budget is None else proposal_budget
                ),
                round_number=round_num,
                random_seed=derive_component_seed(
                    random_seed, config.name, round_num, candidate_pool_strategy.name
                ),
            )
            round_proposals = list(candidate_pool_strategy.build_pool(context))
            proposal_seq_ids = [proposal.identity.seq_id for proposal in round_proposals]
            unknown_proposals = set(proposal_seq_ids) - set(available_seq_ids)
            if unknown_proposals:
                raise ValueError(
                    "candidate strategy proposed variants outside the simulated world: "
                    f"{sorted(unknown_proposals)[:5]}"
                )

        if len(proposal_seq_ids) == 0:
            logger.info("No more unmeasured variants, ending simulation")
            break
        if len(proposal_seq_ids) < round_size:
            raise ValueError(
                f"Candidate pool has {len(proposal_seq_ids)} variants, fewer than round_size "
                f"{round_size} in round {round_num}"
            )

        proposal_naturalness_df = whole_world_naturalness_df.loc[pd.Index(proposal_seq_ids)]
        proposal_embedding_series = whole_world_embedding_series.loc[proposal_seq_ids]

        # We're getting a topN to synthesize, and a predicted activity for every variant.
        top_seq_ids = None
        predicted_activity_ensemble: List[pd.Series] = []
        held_out_prediction_ensemble: List[pd.Series] = []

        # First round: always use zero-shot model
        if round_num == 1:
            # Get top variants using zero-shot model's get_top_n method
            top_seq_ids, predicted_activity_ensemble = zero_shot_model.get_top_n(
                round_size,
                proposal_naturalness_df,
                proposal_embedding_series,
            )

            if has_held_out_slate:
                held_out_batch_seq_ids, held_out_prediction_ensemble = zero_shot_model.get_top_n(
                    round_size,
                    held_out_naturalness_df,
                    held_out_embedding_series,
                )

        # Subsequent rounds: use few-shot model if specified
        else:
            assert few_shot_model is not None
            # Convert list of embeddings to numpy array
            train_activity_series = world_state.get_measured_activity_series()

            few_shot_model.fit(
                whole_world_naturalness_df,
                whole_world_embedding_series,
                train_activity_series,
                test_naturalness_df=held_out_naturalness_df,
                test_embedding_series=held_out_embedding_series,
                test_activity_series=held_out_activity_series,
            )

            # Use the get_top_n method from FewShotModel
            top_seq_ids, predicted_activity_ensemble = few_shot_model.get_top_n(
                round_size,
                proposal_naturalness_df,
                proposal_embedding_series,
                round_number=round_num - 1,
            )

            if has_held_out_slate:
                held_out_batch_seq_ids, held_out_prediction_ensemble = few_shot_model.get_top_n(
                    round_size,
                    held_out_naturalness_df,
                    held_out_embedding_series,
                    round_number=round_num - 1,
                )

        # Update world state.
        assert type(top_seq_ids) == list, f"top_seq_ids must be a list, got {type(top_seq_ids)}"
        assert (
            len(top_seq_ids) == round_size
        ), f"Must choose {round_size} variants per rounds, only chose {len(top_seq_ids)}"
        logging.debug(
            f"In Round {round_num}: elected {len(top_seq_ids)} variants: {','.join(top_seq_ids)}"
        )
        if oracle is not None:
            oracle.measure(top_seq_ids, round_number=round_num)
        world_state.measure_variant_activities(top_seq_ids)

        # Metric calculations.
        consensus_predicted_activity = get_consensus_scores(
            predicted_activity_ensemble, decision_mode="mean"
        )
        mutant_metrics_list = []
        for top_seq_id in top_seq_ids:
            # Get the activity from the dataframe and convert to float if needed
            golden_activity = whole_world_activity_series.loc[top_seq_id]
            assert (
                type(golden_activity) == float or type(golden_activity) == np.float64
            ), f"golden_activity must be a float, got {type(golden_activity)}"
            percentile = all_percentiles.loc[top_seq_id]
            predicted_activity_stddev = float(
                np.std([pa.loc[top_seq_id] for pa in predicted_activity_ensemble])
            )
            mutant_metrics_list.append(
                MutantMetrics(
                    seq_id=top_seq_id,
                    round_found=round_num,
                    activity=float(golden_activity),
                    predicted_activity=consensus_predicted_activity.loc[top_seq_id],
                    predicted_activity_stddev=predicted_activity_stddev,
                    percentile=percentile,
                    relevant_mutants=[],  # TODO(jacob): Compute relevant mutants
                )
            )

        # Compute metrics for this round's predictions
        whole_dataset_spearman = spearmanr(
            entire_activity_series.loc[consensus_predicted_activity.index].values,
            consensus_predicted_activity.values,
        )[0]

        held_out_metrics = {
            "held_out_activity_spearman": float("nan"),
            "held_out_batch_1pct": float("nan"),
            "held_out_1pct_recall": float("nan"),
            "held_out_1pct_recall_slate": float("nan"),
            "held_out_1pct_auc": float("nan"),
            "held_out_batch_10pct": float("nan"),
            "held_out_10pct_recall": float("nan"),
            "held_out_10pct_recall_slate": float("nan"),
            "held_out_10pct_auc": float("nan"),
        }
        if has_held_out_slate:
            consensus_held_out_predictions = get_consensus_scores(
                held_out_prediction_ensemble, decision_mode="mean"
            )
            assert held_out_activity_series.index.equals(consensus_held_out_predictions.index)
            assert not held_out_activity_series.isna().any()
            assert not consensus_held_out_predictions.isna().any()
            held_out_metrics["held_out_activity_spearman"] = float(
                spearmanr(
                    held_out_activity_series.values,
                    consensus_held_out_predictions.values,
                )[0]
            )
            batch_percentiles = held_out_activity_series.rank(pct=True)[held_out_batch_seq_ids]
            held_out_metrics["held_out_batch_1pct"] = float((batch_percentiles >= 0.99).mean())
            held_out_metrics["held_out_batch_10pct"] = float((batch_percentiles >= 0.90).mean())

            for percentile, label in ((1, "1pct"), (10, "10pct")):
                recall = get_top_percentile_recall_score(
                    held_out_activity_series.to_numpy(),
                    consensus_held_out_predictions.to_numpy(),
                    percentile,
                )
                recall_slate = get_top_percentile_recall_score_slate(
                    held_out_activity_series.to_numpy(),
                    consensus_held_out_predictions.to_numpy(),
                    percentile,
                    round_size,
                )
                auc = roc_auc_score(
                    top_k_mask(held_out_activity_series, percentile),
                    consensus_held_out_predictions,
                )
                held_out_metrics[f"held_out_{label}_recall"] = float(recall)
                held_out_metrics[f"held_out_{label}_recall_slate"] = float(recall_slate)
                held_out_metrics[f"held_out_{label}_auc"] = float(auc)

        round_metrics = RoundMetrics(
            round_num=round_num,
            model_spearman=float(whole_dataset_spearman),  # type: ignore
            misc=held_out_metrics,
        )

        if round_num == 1:
            round_metrics.misc["zero_shot_debug_info"] = zero_shot_model.get_debug_info()
        else:
            assert few_shot_model is not None
            round_metrics.misc["few_shot_debug_info"] = few_shot_model.get_debug_info()
        if round_proposals is not None:
            round_metrics.misc["candidate_pool_strategy"] = candidate_pool_strategy.name
            round_metrics.misc["proposal_pool"] = [
                proposal.model_dump(mode="json") for proposal in round_proposals
            ]
            round_metrics.misc["selected_seq_ids"] = top_seq_ids
            round_metrics.misc["oracle_lookup"] = list(oracle.lookup_calls[-1])

        results.rounds = round_num
        results.round_metrics.append(round_metrics)
        results.mutant_metrics.extend(mutant_metrics_list)

    return results


# Run multiple simulations
def run_single_sim_parallel(
    sim_idx,
    available_seq_ids: List[str],
    activity_series: pd.Series,
    naturalness_df: pd.DataFrame,
    embedding_series: pd.Series,
    sim_random_seed,
    **kwargs,
):
    started_at = time.perf_counter()
    logger.info(
        f"Running simulation {sim_idx+1} ({len(available_seq_ids)} / {len(activity_series)} mutants in sim)"
    )
    random.seed(sim_random_seed)
    np.random.seed(sim_random_seed)

    sim_result = _run_single_simulation(
        available_seq_ids,
        activity_series,
        naturalness_df,
        embedding_series,
        random_seed=sim_random_seed,
        **kwargs,
    )
    logger.info(
        f"Completed simulation {sim_idx+1} in {time.perf_counter() - started_at:.1f} seconds"
    )
    return sim_result


def run_single_sim_from_worker(
    sim_idx: int,
    sim_random_seed: int,
    bootstrap_random_seed: int,
    world_size: int,
    **kwargs,
) -> SimulationResult:
    """Run one simulation using dataset state initialized once per worker."""
    if (
        _WORKER_ACTIVITY_SERIES is None
        or _WORKER_NATURALNESS_DF is None
        or _WORKER_EMBEDDING_SERIES is None
        or _WORKER_PRETRAIN_NATURALNESS_DF is None
        or _WORKER_FULL_SEQ_IDS is None
    ):
        raise RuntimeError("Simulation worker dataset was not initialized")

    rng = np.random.RandomState(bootstrap_random_seed)
    available_seq_ids = rng.choice(
        _WORKER_FULL_SEQ_IDS,
        size=world_size,
        replace=False,
    ).tolist()
    return run_single_sim_parallel(
        sim_idx,
        available_seq_ids,
        _WORKER_ACTIVITY_SERIES,
        _WORKER_NATURALNESS_DF,
        _WORKER_EMBEDDING_SERIES,
        sim_random_seed=sim_random_seed,
        pretrain_naturalness_df=_WORKER_PRETRAIN_NATURALNESS_DF,
        **kwargs,
    )


def simulate_campaign(
    dms_id: str,
    round_size: int,
    number_of_simulations: int,
    config_list: List[FolDEModelConfig],
    activity_column: str = "DMS_score",
    max_rounds: int = 10,
    random_seed: int = 42,
    num_workers: int = 10,
    skip_embedding_loading: bool = False,
    candidate_pool_strategy: CandidatePoolStrategy | None = None,
    proposal_budget: int | None = None,
    candidate_min_mutation_depth: int = 0,
    candidate_max_mutation_depth: int | None = None,
    dataset_cache: (
        dict[
            tuple[str, str, bool],
            tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ]
        | None
    ) = None,
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
        num_workers: Number of workers for parallel simulation.
        skip_embedding_loading: Benchmark-only option to skip embedding loading for
            naturalness/random-only configs.
        dataset_cache: Optional cache shared by sequential calls for the same DMS.

    Returns:
        Dictionary containing simulation results for each configuration
    """

    assert len(set([config.name for config in config_list])) == len(
        config_list
    ), f"Some configs have duplicate names."

    # Initialize results
    campaign_result = CampaignResult(
        dms_id=dms_id,
        round_size=round_size,
        number_of_simulations=number_of_simulations,
        activity_column=activity_column,
        max_rounds=max_rounds,
        random_seed=random_seed,
        min_activity=0.0,
        median_activity=0.0,
        max_activity=0.0,
        config_results=[],
    )

    df_cache = {} if dataset_cache is None else dataset_cache

    def get_cached_dataset(
        embedding_model_id: str,
        naturalness_model_id: str,
        skip_embeddings: bool,
    ) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cache_key = (embedding_model_id, naturalness_model_id, skip_embeddings)
        if cache_key not in df_cache:
            try:
                df_cache[cache_key] = get_proteingym_dataset(
                    dms_id,
                    embedding_model_id,
                    naturalness_model_id,
                    skip_embedding_loading=skip_embeddings,
                )
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                import traceback

                print(traceback.format_exc(), flush=True)
                raise e
        return df_cache[cache_key]

    # Run simulations for each configuration
    for config_idx, model_config in enumerate(config_list):
        # Set random seed for reproducibility
        logger.info(f"Running simulations for configuration {config_idx+1}/{len(config_list)}")
        logger.info(f"Config: {model_config}")

        skip_embeddings_for_config = False
        if skip_embedding_loading:
            allowed_zero_shot = {"NaturalnessZeroShotModel", "RandomZeroShotModel"}
            allowed_few_shot = {"NaturalnessFewShotModel", "RandomFewShotModel"}
            if (
                model_config.zero_shot_model_name in allowed_zero_shot
                and model_config.few_shot_model_name in allowed_few_shot
            ):
                skip_embeddings_for_config = True
                logger.info(
                    "skip_embedding_loading is enabled; embeddings will be replaced with placeholders."
                )
            else:
                logger.info(
                    "skip_embedding_loading requested but config uses embedding-dependent "
                    "models; loading embeddings for this config."
                )

        loaded_dataset = get_cached_dataset(
            model_config.embedding_model_id,
            model_config.naturalness_model_id,
            skip_embeddings_for_config,
        )
        if len(loaded_dataset) == 3:
            wt_aa_seq = ""
            entire_naturalness_df, embedding_df, activity_df = loaded_dataset  # type: ignore[reportGeneralTypeIssues] # compatibility path for the older 3-tuple dataset shape
            category_df = pd.DataFrame(index=activity_df.index)
        else:
            wt_aa_seq, entire_naturalness_df, embedding_df, activity_df, category_df = (
                loaded_dataset
            )
        default_naturalness_column = (
            "log_wt_marginal"
            if "log_wt_marginal" in entire_naturalness_df.columns
            else "wt_marginal"
        )
        naturalness_ensemble_df = entire_naturalness_df[
            (
                [default_naturalness_column]
                if model_config.naturalness_columns is None
                else model_config.naturalness_columns
            )
        ]

        pretrain_naturalness_model_id = (
            model_config.few_shot_pretrain_naturalness_model_id or model_config.naturalness_model_id
        )
        pretrain_naturalness_columns = (
            model_config.few_shot_pretrain_naturalness_columns
            if model_config.few_shot_pretrain_naturalness_columns is not None
            else model_config.naturalness_columns
        )
        if pretrain_naturalness_columns is None:
            pretrain_naturalness_columns = [default_naturalness_column]

        if pretrain_naturalness_model_id == model_config.naturalness_model_id:
            pretrain_base_naturalness_df = entire_naturalness_df
        else:
            _, pretrain_base_naturalness_df, _, _, _ = get_cached_dataset(
                model_config.embedding_model_id,
                pretrain_naturalness_model_id,
                skip_embeddings_for_config,
            )

        pretrain_naturalness_ensemble_df = pretrain_base_naturalness_df[
            pretrain_naturalness_columns
        ]
        embedding_series = embedding_df[
            "embedding" if model_config.embedding_column is None else model_config.embedding_column
        ]
        activity_series = activity_df[activity_column]

        # Check that the activity column exists
        if activity_column not in activity_df.columns:
            raise ValueError(
                f"Activity column {activity_column} not found in dataset: {activity_df.columns}"
            )

        # Store some activity stats.
        campaign_result.min_activity = activity_df[activity_column].min(skipna=True)
        campaign_result.median_activity = activity_df[activity_column].median(skipna=True)
        campaign_result.max_activity = activity_df[activity_column].max(skipna=True)
        assert not np.isnan(campaign_result.min_activity)
        assert not np.isnan(campaign_result.median_activity)
        assert not np.isnan(campaign_result.max_activity)

        if model_config.data_split_mode:
            if model_config.data_split_mode not in category_df.columns:
                raise ValueError(
                    f"Data split mode {model_config.data_split_mode} not found in "
                    f"category_df.columns: {category_df.columns}"
                )
            full_seq_id_list = list(
                category_df[category_df[model_config.data_split_mode]].index.values
            )
        else:
            full_seq_id_list = list(activity_df[activity_df[activity_column].notna()].index.values)

        world_size = int(len(full_seq_id_list) * 0.5)
        if world_size < max_rounds * round_size:
            raise ValueError(
                f"World size {world_size} is less than max_rounds * round_size "
                f"{max_rounds * round_size}"
            )

        sim_inputs = [
            (sim_idx, random_seed + 1000 * sim_idx, random_seed + sim_idx)
            for sim_idx in range(number_of_simulations)
        ]

        if num_workers <= 1:
            logger.info(f"Running simulations in-process (num_workers={num_workers}).")
            single_model_campaign_results = [
                run_single_sim_parallel(
                    sim_idx,
                    np.random.RandomState(bootstrap_random_seed)
                    .choice(full_seq_id_list, size=world_size, replace=False)
                    .tolist(),
                    activity_series,
                    naturalness_ensemble_df,
                    embedding_series,
                    sim_random_seed=sim_random_seed,
                    round_size=round_size,
                    config=model_config,
                    max_rounds=max_rounds,
                    wt_aa_seq=wt_aa_seq,
                    pretrain_naturalness_df=pretrain_naturalness_ensemble_df,
                    candidate_pool_strategy=candidate_pool_strategy,
                    proposal_budget=proposal_budget,
                    candidate_min_mutation_depth=candidate_min_mutation_depth,
                    candidate_max_mutation_depth=candidate_max_mutation_depth,
                )
                for sim_idx, bootstrap_random_seed, sim_random_seed in sim_inputs
            ]
        else:
            logger.info(f"Running simulations with {num_workers} workers.")
            torch_threads = max(1, (os.cpu_count() or 1) // num_workers)
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_simulation_worker,
                initargs=(
                    activity_series,
                    naturalness_ensemble_df,
                    embedding_series,
                    pretrain_naturalness_ensemble_df,
                    full_seq_id_list,
                    torch_threads,
                ),
            ) as executor:
                futures = [
                    executor.submit(
                        run_single_sim_from_worker,
                        sim_idx,
                        sim_random_seed,
                        bootstrap_random_seed,
                        world_size,
                        round_size=round_size,
                        config=model_config,
                        max_rounds=max_rounds,
                        wt_aa_seq=wt_aa_seq,
                        candidate_pool_strategy=candidate_pool_strategy,
                        proposal_budget=proposal_budget,
                        candidate_min_mutation_depth=candidate_min_mutation_depth,
                        candidate_max_mutation_depth=candidate_max_mutation_depth,
                    )
                    for sim_idx, bootstrap_random_seed, sim_random_seed in sim_inputs
                ]
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


# --------------------------------------------------------------------------- #
# New, config‑centric checkpointing helper
# --------------------------------------------------------------------------- #
def simulate_campaigns_with_config_checkpoints(
    eval_prefix: str,
    dms_ids: List[str],
    config_list: List[FolDEModelConfig],
    checkpoint_dir: str,
    overwrite: bool = False,
    **kwargs,
) -> Dict[str, ModelEvaluation]:
    """Run campaigns with *per-config* checkpoint files.

    Each ``FolDEModelConfig`` gets its own ``ModelEvaluation`` JSON file named
    ``{eval_prefix}_{config_name}.json`` in ``checkpoint_dir``. Datasets are the
    outer loop so configurations sharing a representation stack reuse one load,
    while every completed config/dataset pair is still checkpointed immediately.

    Parameters
    ----------
    eval_prefix
        Prefix for checkpoint filenames.
    dms_ids
        List of DMS dataset identifiers to evaluate.
    config_list
        Ordered list of model configs to evaluate.
    checkpoint_dir
        Where to store ``*.json`` checkpoint files.
    overwrite
        If *True*, always start fresh for every config even if a checkpoint is
        present.
    **kwargs
        Passed straight through to :func:`simulate_campaign`.  Must include
        ``round_size`` and ``number_of_simulations`` at minimum.

    Returns
    -------
    Dict[str, ModelEvaluation]
        Mapping ``config.name -> ModelEvaluation`` with all results.
    """
    cp_dir = Path(checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)

    # Validate config names early
    for cfg in config_list:
        cfg_name = cfg.name
        if not re.match(r"^[A-Za-z0-9.\-]+$", cfg_name):
            raise ValueError(
                f"Config name '{cfg_name}' contains invalid characters. "
                "Allowed characters are A-Z, a-z, 0-9, hyphen (-) and period (.). "
                "Underscores and other characters are not permitted."
            )

    # ------------------------------------------------------------------ #
    # First pass – validate existing checkpoints so we fail fast on
    # conflicts *before* starting any expensive work.
    # ------------------------------------------------------------------ #
    for cfg in config_list:
        cfg_name = cfg.name
        cp_path = cp_dir / f"{eval_prefix}_{cfg_name}.json"

        if cp_path.exists() and not overwrite:
            with cp_path.open() as f:
                data = json.load(f)
            eval_obj = ModelEvaluation.model_validate(data)

            # Sanity‑check that the stored config matches exactly
            if not eval_obj.campaign_results:
                raise ValueError(f"Checkpoint {cp_path} exists but contains no campaign_results.")
            stored_cfg = eval_obj.campaign_results[0].config_results[0].config
            if stored_cfg.model_dump() != cfg.model_dump():
                # Get the model dumps for comparison
                stored_dump = stored_cfg.model_dump()
                current_dump = cfg.model_dump()

                # Find differences
                differences = []
                all_keys = set(stored_dump.keys()) | set(current_dump.keys())

                for key in sorted(all_keys):
                    if key not in stored_dump:
                        differences.append(
                            f"  - '{key}': missing in stored config, current value: {current_dump[key]}"
                        )
                    elif key not in current_dump:
                        differences.append(
                            f"  - '{key}': missing in current config, stored value: {stored_dump[key]}"
                        )
                    elif stored_dump[key] != current_dump[key]:
                        differences.append(
                            f"  - '{key}': stored={stored_dump[key]}, current={current_dump[key]}"
                        )

                diff_msg = (
                    "\n".join(differences)
                    if differences
                    else "No specific differences found (possibly nested object differences)"
                )

                raise ValueError(
                    f"Config mismatch for checkpoint {cp_path}.\n"
                    f"Differences found:\n{diff_msg}\n"
                    "Pass overwrite=True or pick a new prefix."
                )

            # Ensure stored DMS IDs are a subset of the requested list
            stored_dms = {cr.dms_id for cr in eval_obj.campaign_results}
            if not stored_dms.issubset(set(dms_ids)):
                raise ValueError(
                    f"Checkpoint {cp_path} contains DMS IDs not requested in this run "
                    f"({sorted(stored_dms - set(dms_ids))})."
                )

    # ------------------------------------------------------------------ #
    # Second pass – load checkpoints, then run DMS-first so configs can share data.
    # ------------------------------------------------------------------ #
    all_evals: Dict[str, ModelEvaluation] = {}
    completed_by_config: dict[str, set[str]] = {}
    checkpoint_paths: dict[str, Path] = {}
    for cfg in config_list:
        cfg_name = cfg.name
        cp_path = cp_dir / f"{eval_prefix}_{cfg_name}.json"

        if cp_path.exists() and not overwrite:
            with cp_path.open() as f:
                data = json.load(f)
            eval_obj = ModelEvaluation.model_validate(data)
            completed_dms = {cr.dms_id for cr in eval_obj.campaign_results}
            logger.info(
                f"Resuming config '{cfg_name}' with {len(completed_dms)} / "
                f"{len(dms_ids)} DMS datasets complete."
            )
        else:
            eval_obj = ModelEvaluation(name=f"{eval_prefix}_{cfg_name}", campaign_results=[])
            completed_dms: set[str] = set()

        all_evals[cfg_name] = eval_obj
        completed_by_config[cfg_name] = completed_dms
        checkpoint_paths[cfg_name] = cp_path

    for dms_id in dms_ids:
        dataset_cache: dict[
            tuple[str, str, bool],
            tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ] = {}
        for cfg in config_list:
            cfg_name = cfg.name
            completed_dms = completed_by_config[cfg_name]
            if dms_id in completed_dms:
                logger.info(f"[{cfg_name}] Skipping already‑completed DMS '{dms_id}'.")
                continue

            logger.info(f"[{cfg_name}] Simulating DMS '{dms_id}'.")
            started_at = time.perf_counter()
            eval_obj = all_evals[cfg_name]
            eval_obj.campaign_results.append(
                simulate_campaign(
                    dms_id,
                    config_list=[cfg],
                    dataset_cache=dataset_cache,
                    **kwargs,
                )
            )
            completed_dms.add(dms_id)

            # Write atomically so interruption cannot truncate a valid checkpoint.
            cp_path = checkpoint_paths[cfg_name]
            tmp_path = cp_path.with_suffix(f"{cp_path.suffix}.tmp")
            with tmp_path.open("w") as f:
                json.dump(eval_obj.model_dump(), f, indent=2)
            tmp_path.replace(cp_path)
            logger.info(
                f"[{cfg_name}] Checkpointed DMS '{dms_id}' in "
                f"{time.perf_counter() - started_at:.1f} seconds."
            )

    return all_evals

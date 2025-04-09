import logging
from typing import Any, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from folde.types import ModelEvaluation
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.special import softmax


def internal_sample_n_indices(
    scores: Union[List[float], NDArray[np.float64]],
    n: int,
    temperature: float = 0.0,
    epsilon: float = 0.0,
) -> List[int]:
    """Given a list of scores and a number of samples: draw n samples as indices.

    Arguments:
      scores: Array of scores for each item
      n: Number of items to sample
      temperature: if nonzero, do boltzmann sampling.
      epsilon: if nonzero, do epsilon sampling.
    """

    chosen_indices = []

    if epsilon > 0:
        # Draw some samples randomly.
        epsilon_choices = np.random.choice(len(scores), size=int(epsilon * n), replace=False)
        chosen_indices.extend(epsilon_choices.tolist())

    if temperature < 1e-6:
        # Deterministic top-N selection (argmax without replacement)
        for top_ranked_index in np.argsort(scores)[::-1]:
            if top_ranked_index not in chosen_indices:
                chosen_indices.append(top_ranked_index)
                if len(chosen_indices) == n:
                    break
    else:
        try:
            scores = np.array(scores)
            remaining_indices = [ii for ii in range(len(scores)) if ii not in chosen_indices]
            while len(chosen_indices) < n:
                # Compute softmax over remaining scores
                remaining_scores = scores[remaining_indices]
                probs = softmax(remaining_scores / temperature)

                # Sample one index
                choice = np.random.choice(len(remaining_indices), p=probs)
                chosen_indices.append(remaining_indices[choice])

                # Remove selected index
                del remaining_indices[choice]
        except Exception as e:
            logging.error(f"Error in boltzmann sampling. Returning top {n} variants: {e}")
            # Fallback to deterministic top-N selection
            for top_ranked_index in np.argsort(scores)[::-1]:
                if top_ranked_index not in chosen_indices:
                    chosen_indices.append(top_ranked_index)
                    if len(chosen_indices) == n:
                        break

    assert (
        len(chosen_indices) == n
    ), f"This code should always produce n ({n}) indices, but produced {len(chosen_indices)}"

    assert len(set(chosen_indices)) == len(
        chosen_indices
    ), f"chosen_indices must be unique, got {chosen_indices}"

    return chosen_indices


def convert_compaign_result_collection_to_df(
    model_evaluation: ModelEvaluation,
) -> Tuple[DataFrame, DataFrame]:
    """Convert a CampaignResultCollection to mutant and round metrics dataframes.

    Args:
        model_evaluation: The ModelEvaluation object containing campaign results

    Returns:
        A tuple containing:
            - DataFrame with mutant metrics
            - DataFrame with round metrics
    """
    mutant_metrics_df_list = []
    round_metrics_df_list = []
    for campaign_result in model_evaluation.campaign_results:
        dms_id = campaign_result.dms_id
        for result in campaign_result.config_results:
            config = result.config
            for sim_num, sim_result in enumerate(result.simulation_results):
                mutant_metric_list = []
                for mutant_metric in sim_result.mutant_metrics:
                    mutant_metric_list.append(
                        {
                            "dms_id": dms_id,
                            "config_name": config.name,
                            "sim_num": sim_num,
                            **mutant_metric.model_dump(),
                        }
                    )
                mutant_metric_df = pd.DataFrame(mutant_metric_list)
                mutant_metrics_df_list.append(mutant_metric_df)

                round_metrics_list = []
                for round_metrics in sim_result.round_metrics:
                    round_num = round_metrics.round_num
                    mutants_this_round = mutant_metric_df[mutant_metric_df.round_found == round_num]
                    mutants_so_far = mutant_metric_df[mutant_metric_df.round_found <= round_num]
                    round_metrics_list.append(
                        {
                            "dms_id": dms_id,
                            "config_name": config.name,
                            "sim_num": sim_num,
                            "variant_pool_size": sim_result.variant_pool_size,
                            "best_activity_this_round": mutants_this_round.activity.max(),
                            "best_percentile_this_round": mutants_this_round.activity.max(),
                            "best_activity_so_far": mutants_so_far.activity.max(),
                            "best_percentile_so_far": mutants_so_far.percentile.max(),
                            **round_metrics.model_dump(),
                        }
                    )
                round_metrics_df = pd.DataFrame(round_metrics_list)
                round_metrics_df_list.append(round_metrics_df)

    mutant_df = pd.concat(mutant_metrics_df_list)
    round_metrics_df = pd.concat(round_metrics_df_list)

    return mutant_df, round_metrics_df

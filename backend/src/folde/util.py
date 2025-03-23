import numpy as np
import pandas as pd
import logging
from scipy.special import softmax

from folde.types import ModelEvaluation


def boltzmann_sample_n(scores, temperature, n):
    """Given a list of scores, a temperature, and a number of samples: draw n samples."""

    if temperature == 0:
        # Deterministic top-N selection (argmax without replacement)
        return list(np.argsort(scores)[-n:][::-1])  # descending order

    try:
        scores = np.array(scores)
        indices = list(range(len(scores)))
        selected = []
        for _ in range(min(n, len(scores))):
            # Compute softmax over remaining scores
            remaining_scores = scores[indices]
            probs = softmax(remaining_scores / temperature)

            # Sample one index
            choice = np.random.choice(len(indices), p=probs)
            selected.append(indices[choice])

            # Remove selected index
            del indices[choice]
        return selected
    except Exception as e:
        logging.error(f"Error in boltzmann_sample_n. Returning top {n} variants: {e}")
        return list(np.argsort(scores)[-n:][::-1])  # descending order


def convert_compaign_result_collection_to_df(
    model_evaluation: ModelEvaluation,
) -> pd.DataFrame:
    """Convert a CampaignResultCollection to mutant and round metrics dataframes."""
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
                    mutants_this_round = mutant_metric_df[
                        mutant_metric_df.round_found == round_num
                    ]
                    mutants_so_far = mutant_metric_df[
                        mutant_metric_df.round_found <= round_num
                    ]
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

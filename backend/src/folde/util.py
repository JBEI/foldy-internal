import logging
from pandas.core.frame import DataFrame
from typing import Any, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from folde.types import ModelEvaluation, FolDEModelConfig, ModelDiff
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.special import softmax
from sklearn.metrics import recall_score
import torch

def get_consensus_scores(pred_list: List[pd.Series], decision_mode: str) -> pd.Series:
    """Get the prediction of an ensemble using deicision mode (often max or median)."""
    pred_arr = np.stack([preds.to_numpy() for preds in pred_list])
    if decision_mode == "max":
        consensus_score_arr = np.max(pred_arr, axis=0)
    elif decision_mode == "ucb":
        consensus_score_arr = np.mean(pred_arr, axis=0) + np.std(pred_arr, axis=0)
    elif decision_mode == "median":
        consensus_score_arr = np.median(pred_arr, axis=0)
    elif decision_mode == "mean":
        consensus_score_arr = np.mean(pred_arr, axis=0)
    else:
        raise ValueError(f"Invalid decision mode {decision_mode}")
    return pd.Series(consensus_score_arr, index=pred_list[0].index).astype(float)  # type: ignore


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
        chosen_indices.extend(epsilon_choices.tolist())  # type: ignore

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
                # Convert the list to an array for proper indexing
                remaining_indices_arr = np.array(remaining_indices)
                remaining_scores = scores[remaining_indices_arr]
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


def constant_liar_sample(
    ensemble_preds: np.ndarray,
    seq_ids: np.ndarray,
    q_slate_size: int,
    beta: float,
    tau2=1e-3,
    choice_of_baseline: str = 'min') -> list[str]:
    """
    The "Constant Liar” approximation to the parallel EI acquisition function.

    Args:
        ensemble_preds: np.ndarray, shape (S, N)
        seq_ids: np.ndarray, shape (N,)
        q_slate_size: int, number of samples to draw
        beta: float, beta parameter for UCB
        tau2: float, a model of sample variance, helps numerical stability
    """
    if ensemble_preds.ndim != 2:
        raise ValueError(f"ensemble_preds must be a 2D array, got shape {ensemble_preds.shape}")
    if ensemble_preds.shape[0] != len(seq_ids):
        raise ValueError(f"ensemble_preds must have the same number of rows as seq_ids, got {ensemble_preds.shape[1]} and {len(seq_ids)}")
    if ensemble_preds.shape[0] > 10000:
        raise ValueError(f"ensemble_preds must have at most 10000 rows, got {ensemble_preds.shape[1]}")
    if ensemble_preds.shape[0] < q_slate_size:
        raise ValueError(f"ensemble_preds must have at least q_slate_size rows, got {ensemble_preds.shape[1]} vs {q_slate_size}")
    if ensemble_preds.shape[1] < 3:
        raise ValueError(f'Calculating a good variance requires at least 3 models, got {ensemble_preds.shape[1]}')

    pred_tensor  = torch.tensor(ensemble_preds.T, dtype=torch.float32)  # (S, N)
    S, N = pred_tensor.shape

    # means, deviations, and full covariance 
    means = pred_tensor.mean(dim=0)          # (N,)
    devs  = pred_tensor - means              # (S, N)
    Cov   = (devs.T @ devs) / S              # (N, N)
    vars  = Cov.diag().clamp_min(tau2)       # (N,)
    sigmas = vars.sqrt()                     # (N,)

    # constant liar setup: choose L as the ****minimum**** prior mean 
    # very adversarial
    if choice_of_baseline == 'min':
        L = means.min().item()
    elif choice_of_baseline == 'mean':
        L = means.mean().item()
    else:
        raise ValueError(f"Invalid choice of baseline {choice_of_baseline}")

    # greedy Constant Liar batch‐UCB selection 
    selected = []

    for _ in range(q_slate_size):
        # 1) compute marginal UCB scores with current means and variances
        ucb = means + beta * sigmas
        ucb[selected] = -float("inf")  # mask out already‐picked

        # 2) select best candidate
        idx = int(ucb.argmax().item())
        selected.append(idx)
        print(f'Selecting {seq_ids[idx]} with UCB: {ucb[idx]} = {means[idx]} + {beta} * {sigmas[idx]}')

        # 3) "lie" by imagining a ***bad*** constant observation y = L at idx:
        cov_i = Cov[:, idx].clone()        # (N,)
        v_i   = vars[idx].item()           # Var at idx

        # 4) update posterior means conditional on fake bad y=L
        #     μ_new = μ + cov_i * (L - μ[idx]) / v_i
        means = means + cov_i * (L - means[idx]) / (v_i + tau2)
        # if any(means > 1000):
        #     problem_indices = torch.where(means > 100)[0]
        #     print(f'means: {means[problem_indices[0]]}')
        #     print(f'cov_i: {cov_i[problem_indices[0]]}')
        #     print(f'v_i: {v_i}')
        #     print(f'L: {L}')

        # 5) update covariance via Schur complement, same as KG believer
        Cov   = Cov - torch.outer(cov_i, cov_i) / (v_i + tau2)
        Cov    = 0.5 * (Cov + Cov.T)         # re-symmetrise
        vars  = Cov.diag().clamp_min(tau2)
        sigmas = vars.sqrt()                # update std devs

    # map back to sequence IDs 
    constant_liar_chosen_seq_ids = seq_ids[selected]     # length Q
    return constant_liar_chosen_seq_ids.tolist()


def top_k_mask(series: pd.Series, percentile: float) -> pd.Series:
    k = max(1, int(np.ceil(len(series) * percentile / 100)))
    top_idx = series.nlargest(k).index  # strict ranking
    out = pd.Series(False, index=series.index)
    out.loc[top_idx] = True
    return out


def get_top_percentile_recall_score(target: np.ndarray, pred: np.ndarray, pct: float) -> float:
    target = np.asarray(target).ravel()      # <-- makes it 1-D
    pred   = np.asarray(pred).ravel()
    assert target.size == pred.size, "arrays must be same length"

    n = target.size
    k = max(1, int(np.ceil(n * pct / 100)))
    assert k <= n, f'k must be less than or equal to n, got k={k} and n={n}. target shape {target.shape}, pred shape {pred.shape} pct {pct}'
        

    top_tgt = np.argpartition(target, n - k)[n - k:]
    top_prd = np.argpartition(pred,   n - k)[n - k:]

    # recall = |intersection| / k
    return np.intersect1d(top_tgt, top_prd).size / k

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
                    best_activity_so_far = mutants_so_far.activity.max()
                    normalized_best_activity_so_far = (
                        best_activity_so_far - campaign_result.min_activity
                    ) / (campaign_result.max_activity - campaign_result.min_activity)
                    round_metrics_list.append(
                        {
                            "dms_id": dms_id,
                            "config_name": config.name,
                            "sim_num": sim_num,
                            "variant_pool_size": sim_result.variant_pool_size,
                            "best_activity_this_round": mutants_this_round.activity.max(),
                            "best_percentile_this_round": mutants_this_round.activity.max(),
                            "best_activity_so_far": best_activity_so_far,
                            "normalized_best_activity_so_far": normalized_best_activity_so_far,
                            "best_percentile_so_far": mutants_so_far.percentile.max(),
                            **round_metrics.model_dump(),
                            **round_metrics.misc,
                        }
                    )
                round_metrics_df = pd.DataFrame(round_metrics_list)
                round_metrics_df_list.append(round_metrics_df)

    mutant_df = pd.concat(mutant_metrics_df_list)
    round_metrics_df = pd.concat(round_metrics_df_list)

    return mutant_df, round_metrics_df


def get_training_loss_df(results: ModelEvaluation, round_idx: int) -> pd.DataFrame:
    train_df = pd.DataFrame()
    for campaign_results in results.campaign_results:
        for config_results in campaign_results.config_results:
            for sim_idx, sim_result in enumerate(config_results.simulation_results):
                few_shot_info = sim_result.round_metrics[round_idx].misc['few_shot_debug_info']
                for model_idx in range(len(few_shot_info['pretrain_metrics'])):
                    pretrain_train_loss_list = few_shot_info['pretrain_metrics'][model_idx]['train_loss']
                    pretrain_val_loss_list = few_shot_info['pretrain_metrics'][model_idx]['val_loss']
                    finetune_train_loss_list = few_shot_info['finetune_metrics'][model_idx]['train_loss']
                    finetune_val_loss_list = few_shot_info['finetune_metrics'][model_idx]['val_loss']
                    finetune_test_recall_1pct_list = few_shot_info['finetune_metrics'][model_idx]['test_recall_1pct']
                    model_train_df = pd.concat([pd.DataFrame({  
                        'loss_type': 'pretrain_train',
                        'log_step': list(range(len(pretrain_train_loss_list))),
                        'loss': pretrain_train_loss_list,
                    }), pd.DataFrame({
                        'loss_type': 'pretrain_val',
                        'log_step': list(range(len(pretrain_val_loss_list))),
                        'loss': pretrain_val_loss_list,
                    }), pd.DataFrame({
                        'loss_type': 'finetune_train',
                        'log_step': list(range(len(finetune_train_loss_list))),
                        'loss': finetune_train_loss_list,   
                    }), pd.DataFrame({
                        'loss_type': 'finetune_val',
                        'log_step': list(range(len(finetune_val_loss_list))),
                        'loss': finetune_val_loss_list,
                    }), pd.DataFrame({
                        'loss_type': 'finetune_recall_1pct',
                        'log_step': list(range(len(finetune_test_recall_1pct_list))),
                        'loss': finetune_test_recall_1pct_list,
                    })], ignore_index=True)
                    
                    model_train_df['model_idx'] = model_idx
                    model_train_df['sim_idx'] = sim_idx
                    model_train_df['dms_id'] = campaign_results.dms_id
                    model_train_df['config_name'] = config_results.config.name
                    train_df = pd.concat([train_df, model_train_df])
    return train_df


def apply_diff_to_dict_recursive(
    config_dict: dict[str, Any],
    path_components: list[str],
    new_value: Any,
) -> None:
    assert isinstance(path_components, list)
    if len(path_components) == 1:
        config_dict[path_components[0]] = new_value
        return
    else:
        next_component = path_components[0]
        if next_component not in config_dict:
            raise ValueError(f"Component {next_component} not found in config_dict")
        if type(config_dict[next_component]) is not dict:
            raise ValueError(f"Component {next_component} is not a dict")
        apply_diff_to_dict_recursive(config_dict[next_component], path_components[1:], new_value)


def apply_diff_list_to_config(
    folde_model_config_base: FolDEModelConfig,
    model_diffs: List[ModelDiff],
) -> List[FolDEModelConfig]:
    original_config = folde_model_config_base.model_copy(deep=True, update={"name": folde_model_config_base.name + '-base'})
    config_list = [original_config]
    for model_diff in model_diffs:
        folde_model_config_dict = folde_model_config_base.model_dump()
        for param_path, new_value in model_diff.diffs.items():
            apply_diff_to_dict_recursive(
                folde_model_config_dict,
                param_path.split("."),
                new_value,
            )
        folde_model_config = FolDEModelConfig(**folde_model_config_dict)
        folde_model_config.name = f"{folde_model_config_base.name}-{model_diff.name}"
        config_list.append(folde_model_config)
    return config_list

import os

import numpy as np
import pandas as pd
import torch

from folde.few_shot_models import _TORCH_MLP_PRETRAIN_CACHE, TorchMLPFewShotModel
from folde.util import constant_liar_sample


def _dense_constant_liar_reference(
    ensemble_predictions: np.ndarray,
    seq_ids: np.ndarray,
    q_slate_size: int,
    lie_noise_stddev_multiplier: float,
    choice_of_baseline: str,
    ucb_beta: float,
) -> list[str]:
    predictions = torch.as_tensor(ensemble_predictions.T, dtype=torch.float64)
    ensemble_size, candidate_count = predictions.shape
    noise = (lie_noise_stddev_multiplier * predictions.std(dim=0).median().item()) ** 2
    mean = predictions.mean(dim=0)
    deviations = predictions - mean
    covariance = deviations.T @ deviations / ensemble_size
    covariance += noise * torch.eye(candidate_count, dtype=predictions.dtype)
    baseline = {
        "min": mean.min().item(),
        "mean": mean.mean().item(),
        "max": mean.max().item(),
    }[choice_of_baseline]
    variances = covariance.diag()
    selected: list[int] = []
    for _ in range(q_slate_size):
        acquisition = mean + ucb_beta * variances.sqrt()
        acquisition[selected] = -torch.inf
        selected_index = int(torch.argmax(acquisition))
        selected.append(selected_index)
        covariance_column = covariance[:, selected_index].clone()
        selected_variance = variances[selected_index].item()
        mean = mean + covariance_column * ((baseline - mean[selected_index]) / selected_variance)
        covariance = (
            covariance - torch.outer(covariance_column, covariance_column) / selected_variance
        )
        covariance = 0.5 * (covariance + covariance.T)
        variances = covariance.diag()
    return seq_ids[selected].tolist()


def test_torch_mlp_pretraining_cache_reuses_matching_pretrain() -> None:
    seq_ids = pd.Index([f"A{i}C" for i in range(24)])
    naturalness_df = pd.DataFrame(
        {
            "model_0": np.linspace(-1.0, 1.0, len(seq_ids)),
            "model_1": np.linspace(1.0, -1.0, len(seq_ids)),
        },
        index=seq_ids,
    )
    embedding_series = pd.Series(
        [np.full(8, i / 10.0, dtype=np.float32) for i in range(len(seq_ids))],
        index=seq_ids,
    )
    params = {
        "wt_aa_seq": "A" * 30,
        "random_state": 7,
        "pretrain": True,
        "pretrain_epochs": 1,
        "pretrain_patience": 1,
        "pretrain_val_frequency": 1,
        "ensemble_size": 2,
        "hidden_dims": [4],
        "dropout": 0.0,
        "device": "cpu",
    }

    _TORCH_MLP_PRETRAIN_CACHE.clear()
    first_model = TorchMLPFewShotModel(**params).pretrain(naturalness_df, embedding_series)
    cache_size_after_first_pretrain = len(_TORCH_MLP_PRETRAIN_CACHE)
    second_model = TorchMLPFewShotModel(**params).pretrain(naturalness_df, embedding_series)

    assert cache_size_after_first_pretrain == 1
    assert len(_TORCH_MLP_PRETRAIN_CACHE) == 1
    assert first_model.pretrain_metrics == second_model.pretrain_metrics
    assert len(second_model.pretrained_model_state_dicts) == params["ensemble_size"]


def test_constant_liar_respects_cpu_device_override() -> None:
    rng = np.random.default_rng(123)
    ensemble_predictions = rng.normal(size=(100, 5))
    seq_ids = np.array([f"A{i}C" for i in range(ensemble_predictions.shape[0])])

    old_device = os.environ.get("FOLDE_CONSTANT_LIAR_DEVICE")
    try:
        os.environ["FOLDE_CONSTANT_LIAR_DEVICE"] = "cpu"
        first_slate = constant_liar_sample(
            ensemble_predictions,
            seq_ids,
            q_slate_size=8,
            lie_noise_stddev_multiplier=6.0,
            ucb_beta=0.0,
        )
        second_slate = constant_liar_sample(
            ensemble_predictions,
            seq_ids,
            q_slate_size=8,
            lie_noise_stddev_multiplier=6.0,
            ucb_beta=0.0,
        )
    finally:
        if old_device is None:
            os.environ.pop("FOLDE_CONSTANT_LIAR_DEVICE", None)
        else:
            os.environ["FOLDE_CONSTANT_LIAR_DEVICE"] = old_device

    assert first_slate == second_slate
    assert len(first_slate) == 8
    assert len(set(first_slate)) == 8


def test_low_rank_constant_liar_matches_dense_reference() -> None:
    rng = np.random.default_rng(991)
    predictions = rng.normal(size=(160, 5))
    seq_ids = np.asarray([f"A{index}C" for index in range(len(predictions))])

    old_device = os.environ.get("FOLDE_CONSTANT_LIAR_DEVICE")
    try:
        os.environ["FOLDE_CONSTANT_LIAR_DEVICE"] = "cpu"
        for baseline in ("min", "mean", "max"):
            for multiplier in (6.0, 100.0):
                expected = _dense_constant_liar_reference(
                    predictions,
                    seq_ids,
                    q_slate_size=16,
                    lie_noise_stddev_multiplier=multiplier,
                    choice_of_baseline=baseline,
                    ucb_beta=2.0,
                )
                actual = constant_liar_sample(
                    predictions,
                    seq_ids,
                    q_slate_size=16,
                    lie_noise_stddev_multiplier=multiplier,
                    choice_of_baseline=baseline,
                    ucb_beta=2.0,
                )
                assert actual == expected
    finally:
        if old_device is None:
            os.environ.pop("FOLDE_CONSTANT_LIAR_DEVICE", None)
        else:
            os.environ["FOLDE_CONSTANT_LIAR_DEVICE"] = old_device

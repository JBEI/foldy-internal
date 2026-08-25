from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from app.helpers.preference_ranking import (
    PreferenceDataset,
    PreferenceTrainer,
    batch_bradley_terry_loss,
    batch_preference_loss,
    held_out_sample_bradley_terry_loss,
)
from folde.few_shot_models import TorchMLPFewShotModel


def test_activity_difference_weighted_bt_uses_label_gap_weights() -> None:
    scores = torch.tensor([1.5, 0.5, -0.5])
    labels = torch.tensor([4.0, 3.0, 0.0])

    actual = batch_bradley_terry_loss(
        scores,
        labels,
        pair_mask=None,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        activity_difference_weighting=True,
    )

    pair_logits = torch.tensor([1.0, 2.0, 1.0])
    pair_targets = torch.ones(3)
    pair_weights = torch.tensor([1.0, 4.0, 3.0])
    pair_weights = pair_weights / pair_weights.sum()
    expected = (
        pair_weights
        * F.binary_cross_entropy_with_logits(pair_logits, pair_targets, reduction="none")
    ).sum()

    torch.testing.assert_close(actual, expected)


def test_activity_difference_weighted_bt_returns_zero_for_all_ties() -> None:
    scores = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)
    labels = torch.ones(3)

    loss = batch_bradley_terry_loss(
        scores,
        labels,
        pair_mask=None,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        activity_difference_weighting=True,
    )

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


def test_soft_target_bt_calibrates_standardized_label_gaps() -> None:
    scores = torch.tensor([0.2, -0.1, 0.4])
    labels = torch.tensor([0.0, 0.1, 1.0])
    label_std = labels.std(unbiased=False)
    temperature = 0.5

    actual = batch_bradley_terry_loss(
        scores,
        labels,
        pair_mask=None,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        soft_target_temperature=temperature,
        soft_target_label_std=label_std,
    )

    pair_logits = torch.tensor([0.3, -0.2, -0.5])
    pair_label_gaps = torch.tensor([-0.1, -1.0, -0.9])
    soft_targets = torch.sigmoid(pair_label_gaps / label_std / temperature)
    expected = F.binary_cross_entropy_with_logits(pair_logits, soft_targets)

    torch.testing.assert_close(actual, expected)

    scaled = batch_bradley_terry_loss(
        scores,
        10.0 * labels,
        pair_mask=None,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        soft_target_temperature=temperature,
        soft_target_label_std=10.0 * label_std,
    )
    torch.testing.assert_close(scaled, actual)


def test_soft_target_bt_applies_bounded_confidence_weights() -> None:
    scores = torch.tensor([0.2, -0.1, 0.4])
    labels = torch.tensor([0.0, 0.1, 1.0])
    label_std = labels.std(unbiased=False)
    temperature = 0.5
    confidence_floor = 0.2

    actual = batch_bradley_terry_loss(
        scores,
        labels,
        pair_mask=None,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        soft_target_temperature=temperature,
        soft_target_label_std=label_std,
        soft_target_confidence_floor=confidence_floor,
    )

    pair_logits = torch.tensor([0.3, -0.2, -0.5])
    pair_label_gaps = torch.tensor([-0.1, -1.0, -0.9])
    soft_targets = torch.sigmoid(pair_label_gaps / label_std / temperature)
    weights = confidence_floor + (1.0 - confidence_floor) * 2.0 * (soft_targets - 0.5).abs()
    pair_losses = F.binary_cross_entropy_with_logits(pair_logits, soft_targets, reduction="none")
    expected = (weights * pair_losses).sum() / weights.sum()

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"soft_target_temperature": 0.0, "soft_target_label_std": 1.0}, "positive"),
        ({"soft_target_temperature": 0.5}, "label_std"),
        ({"soft_target_confidence_floor": 0.2}, "requires"),
        (
            {
                "activity_difference_weighting": True,
                "soft_target_temperature": 0.5,
                "soft_target_label_std": 1.0,
            },
            "cannot be combined",
        ),
    ],
)
def test_soft_target_bt_rejects_invalid_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        batch_bradley_terry_loss(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0]),
            pair_mask=None,
            importance_sampling_reweighting_strat=None,
            importance_sampling_temperature=None,
            **kwargs,
        )


def test_batch_preference_loss_mixes_bt_with_globally_standardized_mse() -> None:
    scores = torch.tensor([1.0, -1.0, 0.5])
    labels = torch.tensor([10.0, 20.0, 30.0])
    mse_weight = 0.25
    target_mean = labels.mean()
    target_std = labels.std(unbiased=False)

    actual = batch_preference_loss(
        scores,
        labels,
        pair_mask=None,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        standardized_mse_weight=mse_weight,
        standardized_target_mean=target_mean,
        standardized_target_std=target_std,
    )

    bt_loss = batch_bradley_terry_loss(
        scores,
        labels,
        pair_mask=None,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
    )
    standardized_labels = (labels - target_mean) / target_std
    mse_loss = F.mse_loss(scores, standardized_labels)
    expected = (1.0 - mse_weight) * bt_loss + mse_weight * mse_loss

    torch.testing.assert_close(actual, expected)


def test_pure_standardized_mse_skips_pairwise_loss() -> None:
    scores = torch.tensor([1.0, -1.0, 0.5])
    labels = torch.tensor([10.0, 20.0, 30.0])
    with patch(
        "app.helpers.preference_ranking.batch_bradley_terry_loss",
        side_effect=AssertionError("BT should not be evaluated"),
    ):
        actual = batch_preference_loss(
            scores,
            labels,
            pair_mask=None,
            importance_sampling_reweighting_strat=None,
            importance_sampling_temperature=None,
            standardized_mse_weight=1.0,
            standardized_target_mean=labels.mean(),
            standardized_target_std=labels.std(unbiased=False),
        )

    expected = F.mse_loss(scores, (labels - labels.mean()) / labels.std(unbiased=False))
    torch.testing.assert_close(actual, expected)


def test_torch_mlp_threads_new_loss_options_to_trainer() -> None:
    trainer = Mock()
    trainer.train.return_value = {"train_loss": [], "val_loss": [], "test_recall_1pct": []}
    model = TorchMLPFewShotModel(
        wt_aa_seq="AA",
        random_state=7,
        device="cpu",
        train_epochs=1,
        standardized_mse_weight=0.2,
        bt_activity_difference_weighting=True,
    )
    model._create_model_ensemble = Mock(return_value=[(torch.nn.Linear(2, 1), trainer)])
    index = pd.Index(["A1G", "A2G"])
    naturalness_df = pd.DataFrame({"naturalness": [0.1, 0.2]}, index=index)
    embedding_series = pd.Series([np.array([1.0, 0.0]), np.array([0.0, 1.0])], index=index)
    activity_series = pd.Series([1.0, 2.0], index=index)

    model.fit(naturalness_df, embedding_series, activity_series)

    train_kwargs = trainer.train.call_args.kwargs
    assert train_kwargs["standardized_mse_weight"] == pytest.approx(0.2)
    assert train_kwargs["activity_difference_weighting"] is True


def test_torch_mlp_threads_soft_target_options_to_trainer() -> None:
    trainer = Mock()
    trainer.train.return_value = {"train_loss": [], "val_loss": [], "test_recall_1pct": []}
    model = TorchMLPFewShotModel(
        wt_aa_seq="AA",
        random_state=7,
        device="cpu",
        train_epochs=1,
        bt_soft_target_temperature=0.5,
        bt_soft_target_confidence_floor=0.2,
    )
    model._create_model_ensemble = Mock(return_value=[(torch.nn.Linear(2, 1), trainer)])
    index = pd.Index(["A1G", "A2G"])
    naturalness_df = pd.DataFrame({"naturalness": [0.1, 0.2]}, index=index)
    embedding_series = pd.Series([np.array([1.0, 0.0]), np.array([0.0, 1.0])], index=index)
    activity_series = pd.Series([1.0, 2.0], index=index)

    model.fit(naturalness_df, embedding_series, activity_series)

    train_kwargs = trainer.train.call_args.kwargs
    assert train_kwargs["soft_target_temperature"] == pytest.approx(0.5)
    assert train_kwargs["soft_target_confidence_floor"] == pytest.approx(0.2)


def test_torch_mlp_can_skip_redundant_training_test_metrics() -> None:
    trainer = Mock()
    trainer.train.return_value = {"train_loss": [], "val_loss": [], "test_recall_1pct": []}
    model = TorchMLPFewShotModel(
        wt_aa_seq="AA",
        random_state=7,
        device="cpu",
        train_epochs=1,
        track_test_metrics_during_training=False,
    )
    model._create_model_ensemble = Mock(return_value=[(torch.nn.Linear(2, 1), trainer)])
    train_index = pd.Index(["A1G", "A2G"])
    test_index = pd.Index(["A3G", "A4G"])
    train_embeddings = pd.Series([np.array([1.0, 0.0]), np.array([0.0, 1.0])], index=train_index)
    test_embeddings = pd.Series([np.array([0.5, 0.5]), np.array([1.0, 1.0])], index=test_index)

    model.fit(
        pd.DataFrame({"naturalness": [0.1, 0.2]}, index=train_index),
        train_embeddings,
        pd.Series([1.0, 2.0], index=train_index),
        test_naturalness_df=pd.DataFrame({"naturalness": [0.3, 0.4]}, index=test_index),
        test_embedding_series=test_embeddings,
        test_activity_series=pd.Series([3.0, 4.0], index=test_index),
    )

    train_kwargs = trainer.train.call_args.kwargs
    assert train_kwargs["test_embeddings"] is None
    assert train_kwargs["test_activity_labels"] is None


def test_torch_mlp_rejects_invalid_standardized_mse_weight() -> None:
    with pytest.raises(ValueError, match="standardized_mse_weight"):
        TorchMLPFewShotModel(
            wt_aa_seq="AA",
            random_state=7,
            standardized_mse_weight=1.1,
        )


def test_torch_mlp_rejects_standardized_mse_with_pair_fraction_validation() -> None:
    with pytest.raises(ValueError, match="point-level holdout validation"):
        TorchMLPFewShotModel(
            wt_aa_seq="AA",
            random_state=7,
            standardized_mse_weight=0.2,
            do_validation_with_pair_fraction=0.2,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bt_soft_target_temperature": 0.0},
        {"bt_soft_target_confidence_floor": 0.2},
        {
            "bt_soft_target_temperature": 0.5,
            "bt_activity_difference_weighting": True,
        },
        {"bt_soft_target_temperature": 0.5, "use_mse_loss": True},
    ],
)
def test_torch_mlp_rejects_invalid_soft_target_options(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TorchMLPFewShotModel(wt_aa_seq="AA", random_state=7, **kwargs)


def test_preference_trainer_runs_soft_target_training_and_validation() -> None:
    trainer = PreferenceTrainer(torch.nn.Linear(1, 1, bias=False), random_state=3, device="cpu")

    metrics = trainer.train(
        train_embeddings=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        train_activity_labels=np.array([0.0, 0.2, 1.0, 2.0]),
        val_embeddings=np.array([[1.5], [2.5]], dtype=np.float32),
        val_activity_labels=np.array([0.6, 1.5]),
        batch_size=4,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=None,
        use_mse_loss=False,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        use_exponential_learning_rate_decay=False,
        use_plateau_learning_rate_decay=False,
        val_frequency=1,
        soft_target_temperature=0.5,
        soft_target_confidence_floor=0.2,
    )

    assert np.isfinite(metrics["train_loss"][-1])
    assert np.isfinite(metrics["val_loss"][-1])


class _FirstFeatureScore(torch.nn.Module):
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings[:, :1]


def test_validation_uses_hybrid_loss_on_point_level_holdout() -> None:
    trainer = PreferenceTrainer(_FirstFeatureScore(), random_state=3, device="cpu")
    train_dataset = PreferenceDataset(
        np.array([[0.0], [1.0]]),
        np.array([0.0, 2.0]),
    )
    val_dataset = PreferenceDataset(
        np.array([[2.0], [3.0]]),
        np.array([4.0, 6.0]),
    )
    mse_weight = 0.25

    actual, _ = trainer.evaluate(
        train_dataset,
        val_dataset,
        standardized_mse_weight=mse_weight,
        standardized_target_mean=1.0,
        standardized_target_std=1.0,
    )

    combined_scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    combined_labels = torch.tensor([0.0, 2.0, 4.0, 6.0])
    pair_mask = torch.ones((4, 4), dtype=torch.int).triu(diagonal=1)
    pair_mask[:2, :2] = 0
    bt_loss = batch_bradley_terry_loss(
        combined_scores,
        combined_labels,
        pair_mask,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
    )
    standardized_val_labels = torch.tensor([3.0, 5.0])
    mse_loss = F.mse_loss(torch.tensor([2.0, 3.0]), standardized_val_labels)
    expected = (1.0 - mse_weight) * bt_loss + mse_weight * mse_loss

    assert actual is not None
    assert actual == pytest.approx(expected.item())


@pytest.mark.parametrize("activity_difference_weighting", [False, True])
@pytest.mark.parametrize("reweighting_strategy", [None, "min", "max"])
def test_blockwise_holdout_bt_matches_global_unchunked_loss(
    activity_difference_weighting: bool,
    reweighting_strategy: str | None,
) -> None:
    scores = torch.tensor([-1.1, 0.4, 0.8, -0.2, 1.7, 0.1, 2.2])
    labels = torch.tensor([0.0, 0.3, 0.9, 0.2, 2.5, 0.4, 4.0])
    n_train = 3
    temperature = 0.7 if reweighting_strategy is not None else None
    pair_mask = torch.ones((scores.shape[0], scores.shape[0]), dtype=torch.bool).triu(diagonal=1)
    pair_mask[:n_train, :n_train] = False

    expected = batch_bradley_terry_loss(
        scores,
        labels,
        pair_mask,
        importance_sampling_reweighting_strat=reweighting_strategy,
        importance_sampling_temperature=temperature,
        activity_difference_weighting=activity_difference_weighting,
    )
    actual = held_out_sample_bradley_terry_loss(
        scores,
        labels,
        n_train=n_train,
        block_size=2,
        importance_sampling_reweighting_strat=reweighting_strategy,
        importance_sampling_temperature=temperature,
        activity_difference_weighting=activity_difference_weighting,
    )

    torch.testing.assert_close(actual, expected)


def test_blockwise_holdout_soft_target_bt_matches_global_unchunked_loss() -> None:
    scores = torch.tensor([-1.1, 0.4, 0.8, -0.2, 1.7, 0.1, 2.2])
    labels = torch.tensor([0.0, 0.3, 0.9, 0.2, 2.5, 0.4, 4.0])
    n_train = 3
    label_std = labels[:n_train].std(unbiased=False)
    pair_mask = torch.ones((scores.shape[0], scores.shape[0]), dtype=torch.bool).triu(diagonal=1)
    pair_mask[:n_train, :n_train] = False

    expected = batch_bradley_terry_loss(
        scores,
        labels,
        pair_mask,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        soft_target_temperature=0.5,
        soft_target_label_std=label_std,
        soft_target_confidence_floor=0.2,
    )
    actual = held_out_sample_bradley_terry_loss(
        scores,
        labels,
        n_train=n_train,
        block_size=2,
        importance_sampling_reweighting_strat=None,
        importance_sampling_temperature=None,
        soft_target_temperature=0.5,
        soft_target_label_std=label_std,
        soft_target_confidence_floor=0.2,
    )

    torch.testing.assert_close(actual, expected)

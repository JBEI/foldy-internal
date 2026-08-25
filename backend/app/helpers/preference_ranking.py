"""
Preference ranking module for protein embeddings using Bradley-Terry models.

This module implements preference learning models based on the Bradley-Terry
approach, which learns preferences from pairwise comparisons. The implementation
uses PyTorch to build MLP models that predict preference scores from embeddings.

The key idea is to train a model to predict scalar preference scores for each
protein variant. These scores can then be used to rank variants, with higher
scores indicating more preferred/active variants.

Within each mini-batch during training, we generate all possible pairwise
preferences based on the activity labels. This approach efficiently leverages
the structure in the dataset, as each embedding is only processed once per batch,
but contributes to multiple preference pairs.
"""

import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from folde.util import get_top_percentile_recall_score

logger = logging.getLogger(__name__)


def _create_grad_scaler(enabled: bool) -> Any:
    """Create a CUDA gradient scaler across supported PyTorch AMP APIs."""
    grad_scaler = getattr(torch.amp, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    # PyTorch 2.2 does not yet expose torch.amp.GradScaler.
    return torch.cuda.amp.GradScaler(enabled=enabled)


class PreferenceDataset(Dataset):
    """Dataset for batch-based preference learning.

    This dataset handles embedding samples and their corresponding activity measurements.
    The training loop will compute all valid preferences within each mini-batch.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        activity_labels: np.ndarray,
        device: str | None = None,
    ):
        """Initialize the preference dataset.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings for all samples
            activity_labels: Array of shape (n_samples,) with activity measurements for all samples
        """
        # Keep full datasets on CPU and move only active batches to the target
        # device. This avoids large persistent GPU allocations.
        _ = device
        self.embeddings: torch.Tensor = torch.as_tensor(embeddings, dtype=torch.float32)
        self.activity_labels: torch.Tensor = torch.as_tensor(activity_labels, dtype=torch.float32)
        self.n_samples: int = len(embeddings)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple containing (embedding, activity_label)
        """
        return self.embeddings[idx], self.activity_labels[idx]


class BradleyTerryMLP(nn.Module):
    """Bradley-Terry model for preference learning using an MLP.

    This model maps embeddings to scalar preference scores using a multi-layer
    perceptron. The Bradley-Terry model predicts the probability that item i is
    preferred over item j as sigmoid(score_i - score_j).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        activation: nn.Module = nn.ReLU(),
    ):
        """Initialize the Bradley-Terry MLP model.

        Args:
            embedding_dim: Dimension of the input embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function to use
        """
        super().__init__()

        # Build MLP layers
        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final layer to scalar preference score
        layers.append(nn.Linear(prev_dim, 1, bias=False))  # no bias)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute preference score.

        Args:
            x: Tensor of shape (batch_size, embedding_dim) with embeddings

        Returns:
            Tensor of shape (batch_size, 1) with preference scores
        """
        return self.mlp(x)


def _validate_soft_target_bt_options(
    temperature: float | None,
    label_std: float | torch.Tensor | None,
    confidence_floor: float | None,
    activity_difference_weighting: bool,
) -> None:
    """Validate the gap-calibrated Bradley-Terry loss options."""
    if temperature is None:
        if confidence_floor is not None:
            raise ValueError("soft_target_confidence_floor requires soft_target_temperature")
        return
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("soft_target_temperature must be finite and positive")
    if label_std is None:
        raise ValueError("soft_target_label_std is required for soft-target BT")
    if confidence_floor is not None and not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("soft_target_confidence_floor must be between 0 and 1")
    if activity_difference_weighting:
        raise ValueError("activity_difference_weighting cannot be combined with soft-target BT")


def batch_bradley_terry_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    pair_mask: torch.Tensor | None,
    importance_sampling_reweighting_strat: str | None,
    importance_sampling_temperature: float | None,
    activity_difference_weighting: bool = False,
    soft_target_temperature: float | None = None,
    soft_target_label_std: float | torch.Tensor | None = None,
    soft_target_confidence_floor: float | None = None,
) -> torch.Tensor:
    _validate_soft_target_bt_options(
        soft_target_temperature,
        soft_target_label_std,
        soft_target_confidence_floor,
        activity_difference_weighting,
    )
    scores = scores.view(-1)
    labels = labels.view(-1)
    if pair_mask is None:
        pair_mask = torch.ones(
            (scores.shape[0], scores.shape[0]), dtype=torch.bool, device=scores.device
        ).triu(diagonal=1)
    weighted_loss_sum, weight_sum = _bradley_terry_loss_components(
        scores,
        labels,
        scores,
        labels,
        pair_mask,
        importance_sampling_reweighting_strat,
        importance_sampling_temperature,
        activity_difference_weighting,
        soft_target_temperature,
        soft_target_label_std,
        soft_target_confidence_floor,
    )
    if weight_sum <= 0:
        return weighted_loss_sum
    return weighted_loss_sum / weight_sum


def _bradley_terry_loss_components(
    left_scores: torch.Tensor,
    left_labels: torch.Tensor,
    right_scores: torch.Tensor,
    right_labels: torch.Tensor,
    pair_mask: torch.Tensor,
    importance_sampling_reweighting_strat: str | None,
    importance_sampling_temperature: float | None,
    activity_difference_weighting: bool,
    soft_target_temperature: float | None,
    soft_target_label_std: float | torch.Tensor | None,
    soft_target_confidence_floor: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the weighted-loss numerator and weight denominator for a pair block."""
    left_scores = left_scores.view(-1)
    right_scores = right_scores.view(-1)
    left_labels = left_labels.view(-1)
    right_labels = right_labels.view(-1)
    score_diff = left_scores[:, None] - right_scores[None, :]
    label_diff = left_labels[:, None] - right_labels[None, :]
    if soft_target_temperature is None:
        targets = (label_diff > 0).float()
    else:
        assert soft_target_label_std is not None
        label_std = torch.as_tensor(
            soft_target_label_std,
            dtype=label_diff.dtype,
            device=label_diff.device,
        )
        safe_label_std = label_std.abs().clamp_min(torch.finfo(label_diff.dtype).eps)
        targets = torch.sigmoid(label_diff / safe_label_std / soft_target_temperature)

    if importance_sampling_reweighting_strat == "min":
        if importance_sampling_temperature is None:
            raise ValueError("importance_sampling_temperature is required for min reweighting")
        weights = torch.exp(
            torch.minimum(left_scores[:, None], right_scores[None, :])
            / importance_sampling_temperature
        )
    elif importance_sampling_reweighting_strat == "max":
        if importance_sampling_temperature is None:
            raise ValueError("importance_sampling_temperature is required for max reweighting")
        weights = torch.exp(
            torch.maximum(left_scores[:, None], right_scores[None, :])
            / importance_sampling_temperature
        )
    else:
        weights = torch.ones_like(score_diff)

    if activity_difference_weighting:
        weights = weights * label_diff.abs()
    if soft_target_confidence_floor is not None:
        target_confidence = 2.0 * (targets - 0.5).abs()
        weights = weights * (
            soft_target_confidence_floor + (1.0 - soft_target_confidence_floor) * target_confidence
        )

    pair_mask = pair_mask.bool()
    logits = score_diff[pair_mask]
    selected_targets = targets[pair_mask]
    selected_weights = weights[pair_mask].detach()
    if logits.numel() == 0:
        zero = (left_scores.sum() + right_scores.sum()) * 0.0
        return zero, zero.detach()

    losses = F.binary_cross_entropy_with_logits(logits, selected_targets, reduction="none")
    return (selected_weights * losses).sum(), selected_weights.sum()


def held_out_sample_bradley_terry_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_train: int,
    block_size: int,
    importance_sampling_reweighting_strat: str | None,
    importance_sampling_temperature: float | None,
    activity_difference_weighting: bool = False,
    soft_target_temperature: float | None = None,
    soft_target_label_std: float | torch.Tensor | None = None,
    soft_target_confidence_floor: float | None = None,
) -> torch.Tensor:
    """Evaluate all pairs touching held-out samples with global weight normalization."""
    _validate_soft_target_bt_options(
        soft_target_temperature,
        soft_target_label_std,
        soft_target_confidence_floor,
        activity_difference_weighting,
    )
    scores = scores.view(-1)
    labels = labels.view(-1)
    n_samples = scores.shape[0]
    if not 0 <= n_train <= n_samples:
        raise ValueError(f"n_train must be between 0 and {n_samples}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    weighted_loss_sum = scores.sum() * 0.0
    weight_sum = scores.new_zeros(())
    for row_start in range(0, n_samples, block_size):
        row_end = min(row_start + block_size, n_samples)
        row_indices = torch.arange(row_start, row_end, device=scores.device)
        for col_start in range(row_start, n_samples, block_size):
            col_end = min(col_start + block_size, n_samples)
            col_indices = torch.arange(col_start, col_end, device=scores.device)
            pair_mask = (row_indices[:, None] < col_indices[None, :]) & (
                (row_indices[:, None] >= n_train) | (col_indices[None, :] >= n_train)
            )
            block_loss_sum, block_weight_sum = _bradley_terry_loss_components(
                scores[row_start:row_end],
                labels[row_start:row_end],
                scores[col_start:col_end],
                labels[col_start:col_end],
                pair_mask,
                importance_sampling_reweighting_strat,
                importance_sampling_temperature,
                activity_difference_weighting,
                soft_target_temperature,
                soft_target_label_std,
                soft_target_confidence_floor,
            )
            weighted_loss_sum = weighted_loss_sum + block_loss_sum
            weight_sum = weight_sum + block_weight_sum

    if weight_sum <= 0:
        return weighted_loss_sum
    return weighted_loss_sum / weight_sum


def batch_preference_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    pair_mask: torch.Tensor | None,
    importance_sampling_reweighting_strat: str | None,
    importance_sampling_temperature: float | None,
    standardized_mse_weight: float = 0.0,
    standardized_target_mean: float | torch.Tensor | None = None,
    standardized_target_std: float | torch.Tensor | None = None,
    activity_difference_weighting: bool = False,
    soft_target_temperature: float | None = None,
    soft_target_confidence_floor: float | None = None,
) -> torch.Tensor:
    """Combine Bradley-Terry ranking with MSE against standardized activity labels.

    ``standardized_mse_weight`` is the pointwise share of a convex combination; zero
    preserves the original Bradley-Terry objective. Target statistics must come from
    the complete training split so that standardization does not change by mini-batch.
    """
    if not 0.0 <= standardized_mse_weight <= 1.0:
        raise ValueError("standardized_mse_weight must be between 0 and 1")

    if standardized_mse_weight > 0.0 and (
        standardized_target_mean is None or standardized_target_std is None
    ):
        raise ValueError(
            "standardized_target_mean and standardized_target_std are required when "
            "standardized MSE is enabled"
        )
    _validate_soft_target_bt_options(
        soft_target_temperature,
        standardized_target_std,
        soft_target_confidence_floor,
        activity_difference_weighting,
    )
    if standardized_mse_weight == 1.0:
        assert standardized_target_mean is not None
        assert standardized_target_std is not None
        return standardized_mse_loss(
            scores,
            labels,
            standardized_target_mean,
            standardized_target_std,
        )

    bt_loss = batch_bradley_terry_loss(
        scores,
        labels,
        pair_mask,
        importance_sampling_reweighting_strat,
        importance_sampling_temperature,
        activity_difference_weighting=activity_difference_weighting,
        soft_target_temperature=soft_target_temperature,
        soft_target_label_std=standardized_target_std,
        soft_target_confidence_floor=soft_target_confidence_floor,
    )
    if standardized_mse_weight == 0.0:
        return bt_loss

    assert standardized_target_mean is not None
    assert standardized_target_std is not None
    mse_loss = standardized_mse_loss(
        scores,
        labels,
        standardized_target_mean,
        standardized_target_std,
    )
    return (1.0 - standardized_mse_weight) * bt_loss + standardized_mse_weight * mse_loss


def standardized_mse_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    target_mean: float | torch.Tensor,
    target_std: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate MSE against labels standardized by training-split statistics."""
    target_mean_tensor = torch.as_tensor(target_mean, dtype=labels.dtype, device=labels.device)
    target_std_tensor = torch.as_tensor(target_std, dtype=labels.dtype, device=labels.device)
    safe_target_std = target_std_tensor.abs().clamp_min(torch.finfo(labels.dtype).eps)
    standardized_labels = (labels - target_mean_tensor) / safe_target_std
    return F.mse_loss(scores.view(-1), standardized_labels.view(-1))


def get_random_pair_split(
    B: int, labels: np.ndarray, rng_seed: int, val_fraction: float = 0.2, device=None
):
    """
    Split directed pairs into training vs validation such that validation
    masks only include pairs for which there is NO directed chain
    through any intermediate in the training graph. This ensures that
    the validation loss only contains nontrivial comparisons.

    Returns:
      train_mask, val_mask : (BxB) boolean masks
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    # sample an initial undirected training graph with probability 1 - val_fraction
    prob_train = 1 - val_fraction
    train_graph = torch.rand((B, B), device=device) < prob_train
    train_graph.fill_diagonal_(False)

    # build directed adjacency from training_graph + labels for BFS
    labels_np = labels.tolist()
    adj = [[] for _ in range(B)]
    for u in range(B):
        for v in range(B):
            if train_graph[u, v] and labels_np[u] > labels_np[v]:
                adj[u].append(v)

    # do BFS
    reach = [[False] * B for _ in range(B)]
    for u in range(B):
        visited = [False] * B
        stack = [u]
        visited[u] = True
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not visited[y]:
                    visited[y] = True
                    stack.append(y)
        reach[u] = visited

    # collect all non‐trivial directed pairs (i→j) with no path for validation loss
    nontrivial = [(i, j) for i in range(B) for j in range(B) if i != j and not reach[i][j]]

    # sample exactly M = val_fraction * B*(B-1) of them for validation, note that B*(B-1) gets rid of the diagonal
    M = int(val_fraction * B * (B - 1))
    M = min(M, len(nontrivial))
    logging.debug(f"Sampling {M} validation pairs from {len(nontrivial)} non-trivial pairs")
    selected = random.sample(nontrivial, M)

    # build boolean masks as before
    val_mask = torch.zeros((B, B), dtype=torch.bool, device=device)
    for i, j in selected:
        val_mask[i, j] = True
    train_mask = ~val_mask
    train_mask.fill_diagonal_(False)
    val_mask.fill_diagonal_(False)

    return train_mask, val_mask


class PreferenceTrainer:
    """Trainer for Bradley-Terry preference models.

    This class handles the training and evaluation of Bradley-Terry models,
    including data preparation, training loops, and evaluation metrics.
    """

    def __init__(
        self,
        model: BradleyTerryMLP,
        random_state: int,
        device: str | None = None,
    ):
        """Initialize the preference trainer.

        Args:
            model: Bradley-Terry model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to use for training ('cpu' or 'cuda')
                If None, will use CUDA if available, otherwise CPU
        """
        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.random_state = random_state

        self.model = model.to(self.device)
        self.use_amp = self.device.startswith("cuda") and os.environ.get("FOLDE_DISABLE_AMP") != "1"
        self.amp_device_type = torch.device(self.device).type
        self.scaler = _create_grad_scaler(enabled=self.use_amp)

    def train(
        self,
        train_embeddings: np.ndarray,
        train_activity_labels: np.ndarray,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        patience: int | None,
        use_mse_loss: bool,
        importance_sampling_reweighting_strat: str | None,
        importance_sampling_temperature: float | None,
        use_exponential_learning_rate_decay: bool,
        use_plateau_learning_rate_decay: bool,
        val_embeddings: np.ndarray | None = None,
        val_activity_labels: np.ndarray | None = None,
        do_validation_with_pair_fraction: float | None = None,
        val_frequency: int = 10,
        test_embeddings: np.ndarray | None = None,
        test_activity_labels: np.ndarray | None = None,
        batch_test_size: int = 5000,
        standardized_mse_weight: float = 0.0,
        activity_difference_weighting: bool = False,
        soft_target_temperature: float | None = None,
        soft_target_confidence_floor: float | None = None,
    ) -> dict[str, Any]:
        """Train the Bradley-Terry model using batch-based training.

        Args:
            train_embeddings: Array of shape (n_samples, embedding_dim) with embeddings for all samples
            train_activity_labels: Array of shape (n_samples,) with activity measurements for all samples
            batch_size: Number of samples to include in each batch
            epochs: Maximum number of epochs to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            patience: Number of epochs to wait for validation improvement before early stopping
            use_mse_loss: Whether to use MSE loss instead of Bradley-Terry loss
            standardized_mse_weight: Pointwise share of BT plus standardized-MSE loss
            activity_difference_weighting: Weight BT pairs by their absolute label difference
            soft_target_temperature: Temperature mapping standardized label gaps to BT targets
            soft_target_confidence_floor: Optional minimum weight for ambiguous comparisons
            do_importance_sampling: Whether to use importance sampling / reweighting in BT loss
            val_embeddings: embeddings for validation set or None
            val_activity_labels: activity labels for validation set or None
            val_frequency: Frequency of validation runs
        Returns:
            Dictionary with training metrics:
                'train_loss': List of training losses for each epoch
                'val_loss': List of validation losses for each epoch
        """

        torch.manual_seed(self.random_state)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        if use_mse_loss and standardized_mse_weight > 0.0:
            raise ValueError("use_mse_loss cannot be combined with standardized_mse_weight")
        if use_mse_loss and soft_target_temperature is not None:
            raise ValueError("use_mse_loss cannot be combined with soft-target BT")
        if not 0.0 <= standardized_mse_weight <= 1.0:
            raise ValueError("standardized_mse_weight must be between 0 and 1")
        if standardized_mse_weight > 0.0 and do_validation_with_pair_fraction is not None:
            raise ValueError(
                "standardized_mse_weight requires point-level holdout validation; "
                "it cannot be combined with do_validation_with_pair_fraction"
            )
        standardized_target_mean = float(np.mean(train_activity_labels))
        standardized_target_std = float(np.std(train_activity_labels))
        _validate_soft_target_bt_options(
            soft_target_temperature,
            standardized_target_std,
            soft_target_confidence_floor,
            activity_difference_weighting,
        )

        # Create datasets and dataloaders
        train_dataset = PreferenceDataset(
            train_embeddings, train_activity_labels, device=self.device
        )
        val_dataset = None
        if val_embeddings is not None and val_activity_labels is not None:
            val_dataset = PreferenceDataset(val_embeddings, val_activity_labels, device=self.device)
        test_dataset = None
        if test_embeddings is not None and test_activity_labels is not None:
            test_dataset = PreferenceDataset(
                test_embeddings, test_activity_labels, device=self.device
            )

        shuffle_train_batches = True
        train_mask, val_mask = None, None
        if do_validation_with_pair_fraction is not None:
            assert val_embeddings is None and val_activity_labels is None
            shuffle_train_batches = False
            if batch_size < train_activity_labels.shape[0]:
                raise ValueError(
                    f"Batch size {batch_size} is less than the number of training samples {train_activity_labels.shape[0]}."
                )
            train_mask, val_mask = get_random_pair_split(
                train_activity_labels.shape[0],
                train_activity_labels,
                self.random_state,
                do_validation_with_pair_fraction,
                device=self.device,
            )

        # Create datasets and dataloaders with fixed random seed
        g = torch.Generator()
        g.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train_batches,
            drop_last=False,
            generator=g,
            pin_memory=self.device.startswith("cuda"),
            # Optional tuning knobs kept off for deterministic behavior.
            # persistent_workers=True,
            # num_workers=4,
            # pin_memory=True,
        )

        metrics: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "test_recall_1pct": []}

        best_val_loss = float("inf")
        best_val_loss_epoch = 0
        best_model_state = None

        val_loss, test_recall_1pct = self.evaluate(
            train_dataset,
            val_dataset,
            test_dataset,
            val_mask,
            importance_sampling_reweighting_strat,
            importance_sampling_temperature,
            batch_test_size,
            activity_difference_weighting,
            standardized_mse_weight,
            standardized_target_mean,
            standardized_target_std,
            soft_target_temperature,
            soft_target_confidence_floor,
        )
        metrics["train_loss"].append(np.inf)
        metrics["val_loss"].append(val_loss if val_loss is not None else np.nan)
        metrics["test_recall_1pct"].append(
            test_recall_1pct if test_recall_1pct is not None else np.nan
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        exponential_lr_schedule = None
        plateau_lr_schedule = None
        if use_exponential_learning_rate_decay:
            exponential_lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif use_plateau_learning_rate_decay:
            plateau_lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10, verbose=True
            )

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_number, (batch_embeddings, batch_activity_labels) in enumerate(train_loader):
                batch_embeddings = batch_embeddings.to(self.device, non_blocking=True)
                batch_activity_labels = batch_activity_labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Autocast is the first part of automatic mixed precision training.
                with torch.autocast(
                    device_type=self.amp_device_type,
                    enabled=self.use_amp,
                ):
                    scores = self.model(batch_embeddings)

                    if use_mse_loss:
                        loss = F.mse_loss(scores.squeeze(-1), batch_activity_labels)
                    else:
                        loss = batch_preference_loss(
                            scores,
                            batch_activity_labels,
                            train_mask,
                            importance_sampling_reweighting_strat,
                            importance_sampling_temperature,
                            standardized_mse_weight=standardized_mse_weight,
                            standardized_target_mean=standardized_target_mean,
                            standardized_target_std=standardized_target_std,
                            activity_difference_weighting=activity_difference_weighting,
                            soft_target_temperature=soft_target_temperature,
                            soft_target_confidence_floor=soft_target_confidence_floor,
                        )

                    loss_value = loss.item()
                    if loss_value == 0:
                        # logger.warning(f'Zero loss encountered in batch {batch_number} in epoch {epoch} with {len(batch_embeddings)} members.')
                        logger.warning(
                            f"Zero loss encountered in batch {batch_number} in epoch {epoch} with {len(batch_embeddings)} members. "
                            f"Score variance: {scores.var().detach().cpu().item()}; "
                            f"activity level variance: {batch_activity_labels.var().detach().cpu().item()}; "
                            f"train mask occupancy: {'NA' if train_mask is None else (train_mask.sum() / train_mask.numel()).item()}"
                        )

                # "Scaler" is the second part of automatic mixed precision training.
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                train_loss += loss_value
                num_batches += 1

            if exponential_lr_schedule is not None:
                exponential_lr_schedule.step()

            # Average training loss
            if num_batches > 0:
                train_loss /= num_batches

            # Validation - only run every val_frequency epochs
            is_val_round = epoch == 0 or ((epoch + 1) % val_frequency == 0)
            if is_val_round:
                metrics["train_loss"].append(train_loss)

                val_loss, test_recall_1pct = self.evaluate(
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    val_mask,
                    importance_sampling_reweighting_strat,
                    importance_sampling_temperature,
                    batch_test_size,
                    activity_difference_weighting,
                    standardized_mse_weight,
                    standardized_target_mean,
                    standardized_target_std,
                    soft_target_temperature,
                    soft_target_confidence_floor,
                )
                metrics["val_loss"].append(val_loss if val_loss is not None else np.nan)
                metrics["test_recall_1pct"].append(
                    test_recall_1pct if test_recall_1pct is not None else np.nan
                )

                if val_loss is not None:
                    if plateau_lr_schedule is not None:
                        plateau_lr_schedule.step(val_loss, epoch=epoch)

                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_loss_epoch = epoch
                        best_model_state = {
                            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                        }

                    if patience is not None and epoch - best_val_loss_epoch >= patience:
                        # logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        if best_model_state is not None:
            with torch.no_grad():
                self.model.load_state_dict(
                    {k: v.to(self.device) for k, v in best_model_state.items()}
                )

        return metrics

    def evaluate(
        self,
        train_dataset: PreferenceDataset,
        val_dataset: PreferenceDataset | None = None,
        test_dataset: PreferenceDataset | None = None,
        val_mask: torch.Tensor | None = None,
        importance_sampling_reweighting_strat: str | None = None,
        importance_sampling_temperature: float | None = None,
        batch_test_size: int = 5000,
        activity_difference_weighting: bool = False,
        standardized_mse_weight: float = 0.0,
        standardized_target_mean: float | None = None,
        standardized_target_std: float | None = None,
        soft_target_temperature: float | None = None,
        soft_target_confidence_floor: float | None = None,
    ) -> tuple[float | None, float | None]:
        """Evaluate the model using batch-based evaluation without leaking GPU memory.

        Args:
            train_dataset: Dataset containing only the training samples
            val_and_train_dataset: Dataset containing the union of train + val samples (or None)
            test_dataset: Dataset containing held‑out test samples (or None)

        Returns:
            Tuple ``(val_loss, test_recall_1pct)``. Either element may be ``None`` if the
            corresponding dataset was not supplied.
        """
        self.model.eval()
        if standardized_mse_weight > 0.0:
            if standardized_target_mean is None or standardized_target_std is None:
                raise ValueError(
                    "Training-split statistics are required for standardized-MSE validation"
                )
        if soft_target_temperature is not None:
            _validate_soft_target_bt_options(
                soft_target_temperature,
                standardized_target_std,
                soft_target_confidence_floor,
                activity_difference_weighting,
            )
        if standardized_mse_weight > 0.0:
            if val_mask is not None:
                raise ValueError(
                    "standardized-MSE validation requires held-out samples, not held-out pairs"
                )

        # Ensure we never build a computation graph and aggressively free intermediates.
        with torch.no_grad():
            val_loss: float | None = None
            test_recall_1pct: float | None = None

            if val_dataset is not None:
                if val_mask is not None:
                    raise ValueError(
                        "Cannot specify both a validation dataset and a validation mask"
                    )

                train_embeddings = train_dataset.embeddings.to(self.device, non_blocking=True)
                val_embeddings = val_dataset.embeddings.to(self.device, non_blocking=True)
                train_and_val_embeddings = torch.cat([train_embeddings, val_embeddings], dim=0)
                train_and_val_scores = self.model(train_and_val_embeddings)

                activity_labels = torch.cat(
                    [train_dataset.activity_labels, val_dataset.activity_labels], dim=0
                ).to(self.device, non_blocking=True)

                bt_val_loss = 0.0
                if standardized_mse_weight < 1.0:
                    # Evaluate every train-val and val-val pair in blocks. Numerators and
                    # denominators are accumulated globally so block size cannot change loss.
                    bt_val_loss_tensor = held_out_sample_bradley_terry_loss(
                        train_and_val_scores,
                        activity_labels,
                        n_train=train_dataset.embeddings.shape[0],
                        block_size=batch_test_size,
                        importance_sampling_reweighting_strat=importance_sampling_reweighting_strat,
                        importance_sampling_temperature=importance_sampling_temperature,
                        activity_difference_weighting=activity_difference_weighting,
                        soft_target_temperature=soft_target_temperature,
                        soft_target_label_std=standardized_target_std,
                        soft_target_confidence_floor=soft_target_confidence_floor,
                    )
                    bt_val_loss = float(bt_val_loss_tensor.detach().cpu().item())
                    del bt_val_loss_tensor
                if standardized_mse_weight > 0.0:
                    assert standardized_target_mean is not None
                    assert standardized_target_std is not None
                    val_scores = train_and_val_scores[train_dataset.embeddings.shape[0] :]
                    val_labels = val_dataset.activity_labels.to(self.device, non_blocking=True)
                    mse_val_loss = standardized_mse_loss(
                        val_scores,
                        val_labels,
                        standardized_target_mean,
                        standardized_target_std,
                    )
                    val_loss = float(
                        (1.0 - standardized_mse_weight) * bt_val_loss
                        + standardized_mse_weight * mse_val_loss.detach().cpu().item()
                    )
                    del val_scores, val_labels, mse_val_loss
                else:
                    val_loss = bt_val_loss

                del (
                    train_embeddings,
                    val_embeddings,
                    train_and_val_embeddings,
                    train_and_val_scores,
                    activity_labels,
                )
            elif val_mask is not None:
                train_embeddings = train_dataset.embeddings.to(self.device, non_blocking=True)
                train_activity_labels = train_dataset.activity_labels.to(
                    self.device, non_blocking=True
                )
                val_mask = val_mask.to(self.device, non_blocking=True)
                train_scores = self.model(train_embeddings)
                val_loss_tensor = batch_bradley_terry_loss(
                    train_scores,
                    train_activity_labels,
                    val_mask,
                    importance_sampling_reweighting_strat=importance_sampling_reweighting_strat,
                    importance_sampling_temperature=importance_sampling_temperature,
                    activity_difference_weighting=activity_difference_weighting,
                    soft_target_temperature=soft_target_temperature,
                    soft_target_label_std=standardized_target_std,
                    soft_target_confidence_floor=soft_target_confidence_floor,
                )
                val_loss = float(val_loss_tensor.detach().cpu().item())
                del train_embeddings, train_activity_labels, train_scores, val_loss_tensor

            if test_dataset is not None:
                test_scores = self.predict_scores(
                    test_dataset.embeddings.numpy(),
                    batch_size=batch_test_size,
                )
                test_recall_1pct = get_top_percentile_recall_score(
                    test_dataset.activity_labels.numpy(),
                    test_scores,
                    1.0,
                )
                del test_scores

        return val_loss, test_recall_1pct

    def predict_scores(self, embeddings: np.ndarray, batch_size: int = 5000) -> np.ndarray:
        """Predict preference scores for embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings
            batch_size: Number of embeddings to score per forward pass

        Returns:
            Array of shape (n_samples,) with preference scores
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if embeddings.shape[0] == 0:
            return np.array([], dtype=np.float32)

        self.model.eval()

        score_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start_idx in range(0, embeddings.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, embeddings.shape[0])
                embeddings_tensor = torch.as_tensor(
                    embeddings[start_idx:end_idx], dtype=torch.float32, device=self.device
                )
                batch_scores = self.model(embeddings_tensor).squeeze(-1).detach().cpu().numpy()
                score_batches.append(batch_scores)

        return np.concatenate(score_batches, axis=0)


def create_preference_model(
    embedding_dim: int,
    hidden_dims: list[int],
    dropout: float = 0.2,
    device: str | None = None,
    random_state: int = 0,
) -> Tuple[BradleyTerryMLP, PreferenceTrainer]:
    """Create a preference model and trainer with standard parameters.

    This is a convenience function to quickly set up a model and trainer
    with reasonable default parameters.

    Args:
        embedding_dim: Dimension of the input embeddings
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to use for training ('cpu' or 'cuda')
            If None, will use CUDA if available, otherwise CPU

    Returns:
        Tuple containing (model, trainer)
    """
    # This... might be where we should set seeds for initialization of model weights.
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    model = BradleyTerryMLP(embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)
    # model = torch.compile(model)

    trainer = PreferenceTrainer(
        model=model,
        random_state=random_state,
        device=device,
    )

    return model, trainer

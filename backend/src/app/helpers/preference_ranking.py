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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for batch-based preference learning.

    This dataset handles embedding samples and their corresponding activity measurements.
    The training loop will compute all valid preferences within each mini-batch.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        activity_labels: np.ndarray,
    ):
        """Initialize the preference dataset.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings for all samples
            activity_labels: Array of shape (n_samples,) with activity measurements for all samples
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.activity_labels = torch.tensor(activity_labels, dtype=torch.float32)
        self.n_samples = len(embeddings)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def predict_preference(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        """Predict preference between two embeddings.

        Args:
            emb_i: Tensor of shape (batch_size, embedding_dim) for first item
            emb_j: Tensor of shape (batch_size, embedding_dim) for second item

        Returns:
            Tensor of shape (batch_size, 1) with preference probabilities
            Values close to 1 indicate i is preferred over j
            Values close to 0 indicate j is preferred over i
        """
        score_i = self.forward(emb_i)
        score_j = self.forward(emb_j)
        return torch.sigmoid(score_i - score_j)


def batch_bradley_terry_loss(
    scores,
    labels,
    exclude_pair_from_loss: Callable[[int, int], bool] = lambda i, j: False,
):
    # scores : (B,1)  labels : (B,)
    s = scores.view(-1, 1)  # (B,1)
    diff = s - s.T  # (B,B)
    y = (labels.view(-1, 1) > labels).float()
    mask = labels.view(-1, 1) != labels  # ignore ties
    if mask.sum() == 0:
        # add a tiny penalty to keep grads flowing
        return (scores**2).mean() * 1e-4
    loss = F.binary_cross_entropy_with_logits(diff[mask], y[mask])
    return loss


# def batch_bradley_terry_loss(
#     scores: torch.Tensor,  # (batch_size, 1)
#     activity_labels: torch.Tensor,
#     exclude_pair_from_loss: Callable[[int, int], bool] = lambda i, j: False,
# ) -> torch.Tensor:
#     batch_size = scores.shape[0]
#     device = scores.device

#     # Flatten
#     scores = scores.view(-1)

#     # i < j
#     i_idx, j_idx = torch.triu_indices(batch_size, batch_size, offset=1, device=device)

#     # Filter out pairs i,j with equal label or excluded
#     mask = []
#     for i, j in zip(i_idx.tolist(), j_idx.tolist()):
#         if exclude_pair_from_loss(i, j):
#             continue
#         if activity_labels[i] == activity_labels[j]:
#             continue
#         mask.append((i, j))
#     if not mask:
#         return torch.tensor(0.0, requires_grad=True, device=device)

#     i_valid = torch.tensor([p[0] for p in mask], device=device)
#     j_valid = torch.tensor([p[1] for p in mask], device=device)

#     # 1 if label[i] > label[j]
#     y_true = (activity_labels[i_valid] > activity_labels[j_valid]).float()
#     y_pred = scores[i_valid] - scores[j_valid]

#     return F.binary_cross_entropy_with_logits(y_pred, y_true)

# pred_prob = torch.sigmoid(y_pred)
# loss = -(y_true * torch.log(pred_prob + 1e-7) + (1 - y_true) * torch.log(1 - pred_prob + 1e-7))
# return loss.mean()


#     embeddings: torch.Tensor,
#     scores: torch.Tensor,
#     activity_labels: torch.Tensor,
#     exclude_pair_from_loss: Callable[[int, int], bool] = lambda i, j: False,
# ) -> torch.Tensor:
#     """Compute Bradley-Terry loss for batch-based preference learning.

#     Args:
#         embeddings: Tensor of shape (batch_size, embedding_dim) with embeddings (for reference only)
#         scores: Tensor of shape (batch_size, 1) with predicted scores for all samples in batch
#         activity_labels: Tensor of shape (batch_size,) with activity measurements for all samples in batch
#         exclude_pair_from_loss: Function that takes indices i, j and returns True if the pair
#             should be excluded from the loss computation

#     Returns:
#         Bradley-Terry loss computed over all valid pairs within the batch
#     """
#     batch_size = scores.shape[0]
#     losses = []  # Track individual pair losses

#     for i in range(batch_size):
#         for j in range(batch_size):
#             if i == j or exclude_pair_from_loss(i, j):
#                 continue

#             # Skip pairs with equal activity
#             if activity_labels[i] == activity_labels[j]:
#                 continue

#             true_preference = 1.0 if activity_labels[i] > activity_labels[j] else 0.0
#             score_diff = scores[i] - scores[j]

#             # Use binary cross entropy loss instead
#             pred_prob = torch.sigmoid(score_diff)
#             loss = -1 * (
#                 true_preference * torch.log(pred_prob + 1e-7)
#                 + (1 - true_preference) * torch.log(1 - pred_prob + 1e-7)
#             )
#             losses.append(loss)

#     # Return mean loss if we have valid pairs, otherwise return zero loss
#     if losses:
#         return torch.mean(torch.stack(losses))
#     return torch.tensor(0.0, device=scores.device, requires_grad=True)


class PreferenceTrainer:
    """Trainer for Bradley-Terry preference models.

    This class handles the training and evaluation of Bradley-Terry models,
    including data preparation, training loops, and evaluation metrics.
    """

    def __init__(
        self,
        model: BradleyTerryMLP,
        random_state: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.random_state = random_state
        torch.manual_seed(random_state)  # Set PyTorch random seed
        np.random.seed(random_state)  # Set NumPy random seed

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def train(
        self,
        embeddings: np.ndarray,
        activity_labels: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        val_ratio_or_indices: Union[float, List[int], np.ndarray] = 0.1,
        patience: int = 10,
        exclude_pair_from_loss: Callable[[int, int], bool] = lambda i, j: False,
        verbose: bool = True,
        use_mse_loss: bool = False,
        val_frequency: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the Bradley-Terry model using batch-based training.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings for all samples
            activity_labels: Array of shape (n_samples,) with activity measurements for all samples
            batch_size: Number of samples to include in each batch
            epochs: Maximum number of epochs to train
            val_ratio_or_indices: Proportion of data to use for validation, or list of indices to use for validation
            patience: Number of epochs to wait for validation improvement before early stopping
            exclude_pair_from_loss: Function that takes indices i, j and returns True if the pair
                should be excluded from the loss computation
            verbose: Whether to print progress during training
            use_mse_loss: Whether to use MSE loss instead of Bradley-Terry loss
            val_frequency: Frequency of validation runs
        Returns:
            Dictionary with training metrics:
                'train_loss': List of training losses for each epoch
                'val_loss': List of validation losses for each epoch
        """
        # Split data into train and validation sets
        train_indices: np.ndarray
        val_indices: np.ndarray
        if isinstance(val_ratio_or_indices, float):
            indices = np.arange(len(embeddings))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_ratio_or_indices, random_state=42
            )
        elif isinstance(val_ratio_or_indices, (list, np.ndarray)):
            # Ensure val_indices is a proper numpy array, never None
            val_indices = (
                np.array(val_ratio_or_indices)
                if val_ratio_or_indices is not None
                else np.array([], dtype=int)
            )
            all_indices = np.arange(len(embeddings))
            val_set = set(val_indices.tolist())
            train_indices = np.array([v for v in all_indices if v not in val_set])
        else:
            raise KeyError(
                f"val_ratio_or_indices should be float or list, got {type(val_ratio_or_indices)} {val_ratio_or_indices}"
            )

        # Create datasets
        train_embeddings = embeddings[train_indices]
        train_activity_labels = activity_labels[train_indices]
        val_embeddings = embeddings[val_indices] if len(val_indices) > 0 else None
        val_activity_labels = activity_labels[val_indices] if len(val_indices) > 0 else None

        # Create datasets and dataloaders
        train_dataset = PreferenceDataset(train_embeddings, train_activity_labels)

        # Create datasets and dataloaders with fixed random seed
        g = torch.Generator()
        g.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=g
        )

        metrics: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        no_improve_epochs = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_number, (batch_embeddings, batch_activity_labels) in enumerate(train_loader):
                batch_embeddings = batch_embeddings.to(self.device)
                batch_activity_labels = batch_activity_labels.to(self.device)

                self.optimizer.zero_grad()
                scores = self.model(batch_embeddings)

                # Add gradient debugging
                pre_loss_params = [p.clone() for p in self.model.parameters()]

                if use_mse_loss:
                    loss = F.mse_loss(scores.squeeze(-1), batch_activity_labels)
                else:
                    loss = batch_bradley_terry_loss(
                        scores,
                        batch_activity_labels,
                        exclude_pair_from_loss,
                    )

                if loss.item() == 0:
                    logger.warning(
                        f"Zero loss encountered in batch {batch_number} in epoch {epoch} with {len(batch_embeddings)} members"
                    )

                loss.backward()

                # Check if gradients are being computed
                has_grad = all(
                    p.grad is not None and torch.any(p.grad != 0) for p in self.model.parameters()
                )
                # if not has_grad:
                #     logger.warning(
                #         f"No gradients computed for batch {batch_number} in epoch {epoch} with {len(batch_embeddings)} members"
                #     )

                self.optimizer.step()

                # Verify parameters are actually changing
                params_changed = any(
                    not torch.equal(p1, p2)
                    for p1, p2 in zip(pre_loss_params, self.model.parameters())
                )
                if not params_changed:
                    logger.warning("Model parameters did not change after optimization step")

                train_loss += loss.item()
                num_batches += 1

            # Average training loss
            if num_batches > 0:
                train_loss /= num_batches
            metrics["train_loss"].append(train_loss)

            # Validation - only run every val_frequency epochs
            is_val_round = epoch == 0 or ((epoch + 1) % val_frequency == 0)
            if is_val_round and len(val_indices) > 0:
                val_loss = self.evaluate(
                    val_embeddings,
                    val_activity_labels,
                    batch_size,
                    exclude_pair_from_loss,
                )
                metrics["val_loss"].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += val_frequency  # Increment by val_frequency instead of 1

                if no_improve_epochs >= patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                metrics["val_loss"].append(np.nan)

            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                val_loss_str = (
                    f"Val Loss: {metrics['val_loss'][-1]:.4f}"
                    if (epoch + 1) % val_frequency == 0
                    else "Val Loss: N/A"
                )
                logger.debug(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"{val_loss_str}"
                )

        return metrics

    def evaluate(
        self,
        embeddings: np.ndarray,
        activity_labels: np.ndarray,
        batch_size: int = 32,
        exclude_pair_from_loss: Callable[[int, int], bool] = lambda i, j: False,
    ) -> float:
        """Evaluate the model using batch-based evaluation.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings
            activity_labels: Array of shape (n_samples,) with activity measurements
            batch_size: Number of samples to include in each batch
            exclude_pair_from_loss: Function that takes indices i, j and returns True if the pair
                should be excluded from the loss computation

        Returns:
            Average loss across all valid pairs
        """
        self.model.eval()

        # Create dataset and dataloader
        dataset = PreferenceDataset(embeddings, activity_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_embeddings, batch_activity_labels in dataloader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_activity_labels = batch_activity_labels.to(self.device)

                # Forward pass
                scores = self.model(batch_embeddings)

                # Compute loss across all pairs in batch
                loss = batch_bradley_terry_loss(
                    scores,
                    batch_activity_labels,
                    exclude_pair_from_loss,
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict preference scores for embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings

        Returns:
            Array of shape (n_samples,) with preference scores
        """
        self.model.eval()
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scores: np.ndarray = self.model(embeddings_tensor).squeeze(-1).cpu().numpy()

        return scores

    def evaluate_ranking(self, embeddings: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate ranking performance on embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings
            true_labels: Array of shape (n_samples,) with true scores/activities

        Returns:
            Dictionary with evaluation metrics:
                'spearman_corr': Spearman rank correlation between predicted and true scores
                'preference_accuracy': Accuracy of pairwise preference predictions
                'auc': ROC AUC for preference prediction
        """
        predicted_scores = self.predict_scores(embeddings)

        # Spearman correlation
        spearman_corr, _ = spearmanr(predicted_scores, true_labels)

        # Generate all possible preference pairs
        n_samples = len(true_labels)
        pairs = []
        y_true = []

        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    continue

                # Skip pairs with equal activity (no clear preference)
                if true_labels[i] == true_labels[j]:
                    continue

                pairs.append((i, j))
                y_true.append(1.0 if true_labels[i] > true_labels[j] else 0.0)

        if not pairs:  # No valid pairs
            return {
                "spearman_corr": spearman_corr,
                "preference_accuracy": 0.0,
                "auc": 0.5,  # Random classifier
            }

        # Convert to numpy arrays explicitly with type annotation
        pairs_array: np.ndarray = np.array(pairs)
        y_true_array = np.array(y_true)

        # Compute preference predictions
        y_pred: List[float] = []
        for i, j in pairs_array:
            pred_i = predicted_scores[int(i)]
            pred_j = predicted_scores[int(j)]
            y_pred.append(float(pred_i - pred_j))

        y_pred_array = np.array(y_pred)

        # Compute preference accuracy
        correct_prefs: int = int(np.sum((y_pred_array > 0) == (y_true_array > 0.5)))
        preference_accuracy = correct_prefs / len(pairs)

        # Compute ROC AUC for preference prediction
        auc = roc_auc_score(y_true_array, y_pred_array)

        return {
            "spearman_corr": spearman_corr,
            "preference_accuracy": preference_accuracy,
            "auc": auc,
        }

    def evaluate_ranking_vectorized(
        self, embeddings: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate ranking performance on embeddings in a more efficient, vectorized way.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings
            true_labels: Array of shape (n_samples,) with true scores/activities

        Returns:
            Dictionary with evaluation metrics:
                'spearman_corr': Spearman rank correlation between predicted and true scores
                'preference_accuracy': Accuracy of pairwise preference predictions
                'auc': ROC AUC for preference prediction
        """
        # 1. Predict scores
        predicted_scores = self.predict_scores(embeddings)

        # 2. Spearman correlation
        spearman_corr, _ = spearmanr(predicted_scores, true_labels)

        # 3. Gather all i < j pairs via np.triu_indices
        n_samples = len(true_labels)
        i_idx, j_idx = np.triu_indices(n_samples, k=1)

        # 4. Filter out pairs where labels are equal (no preference)
        unequal_mask = true_labels[i_idx] != true_labels[j_idx]
        if not np.any(unequal_mask):
            # No valid pairs: return defaults
            return {
                "spearman_corr": spearman_corr,
                "preference_accuracy": 0.0,
                "auc": 0.5,  # Random classifier
            }

        i_valid = i_idx[unequal_mask]
        j_valid = j_idx[unequal_mask]

        # 5. True preference = 1 if label[i] > label[j], else 0
        y_true = (true_labels[i_valid] > true_labels[j_valid]).astype(float)

        # 6. Predicted preference = predicted_scores[i] - predicted_scores[j]
        y_pred = predicted_scores[i_valid] - predicted_scores[j_valid]

        # 7. Compute preference accuracy
        correct_count: int = int(np.sum((y_pred > 0) == (y_true > 0.5)))
        preference_accuracy = correct_count / len(y_true)

        # 8. Compute ROC AUC for preference prediction
        auc_val = roc_auc_score(y_true, y_pred)

        # assert False, "MUST FINISH GETTING THIS THRESHOLDING FUNCTION FINISHED"
        # top16_binarization = true_labels.rank(descending=False) < 16
        # top16_auc = compute_auc(predicted_scores, top16_binarization)
        top16_auc = 0.0

        return {
            "spearman_corr": spearman_corr,
            "preference_accuracy": preference_accuracy,
            "auc": auc_val,
            "top16_auc": top16_auc,
        }


def create_preference_model(
    embedding_dim: int,
    hidden_dims: List[int] = [128, 64],
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
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
    model = BradleyTerryMLP(embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)

    trainer = PreferenceTrainer(
        model=model,
        random_state=random_state,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    return model, trainer

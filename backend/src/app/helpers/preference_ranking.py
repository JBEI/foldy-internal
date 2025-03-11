"""
Preference ranking module for protein embeddings using Bradley-Terry models.

This module implements preference learning models based on the Bradley-Terry
approach, which learns preferences from pairwise comparisons. The implementation
uses PyTorch to build MLP models that predict preference scores from embeddings.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for pairwise preference learning.
    
    This dataset handles pairs of examples where one is preferred over the other.
    Each example consists of two embeddings (i, j) and a binary label indicating
    whether i is preferred over j (1) or not (0).
    """
    
    def __init__(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray, 
        preference_pairs: np.ndarray
    ):
        """Initialize the preference dataset.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings for all samples
            labels: Array of shape (n_samples,) with scores/activities for all samples
            preference_pairs: Array of shape (n_pairs, 2) with indices of preferred pairs
                Each row (i, j) indicates that sample i is preferred over sample j
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.preference_pairs = torch.tensor(preference_pairs, dtype=torch.long)
        
    def __len__(self) -> int:
        """Return the number of preference pairs."""
        return len(self.preference_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single preference pair.
        
        Args:
            idx: Index of the preference pair to retrieve
            
        Returns:
            Tuple containing (embedding_i, embedding_j, preference_label)
            where preference_label is 1 if i is preferred over j, 0 otherwise
        """
        i, j = self.preference_pairs[idx]
        emb_i = self.embeddings[i]
        emb_j = self.embeddings[j]
        # Bradley-Terry model assumes i is preferred over j, so label is always 1
        label = torch.tensor(1.0, dtype=torch.float32)
        
        return emb_i, emb_j, label


def generate_preference_pairs(
    labels: np.ndarray, 
    threshold: Optional[float] = None,
    max_pairs: Optional[int] = None
) -> np.ndarray:
    """Generate preference pairs from score/activity labels.
    
    Args:
        labels: Array of shape (n_samples,) with scores/activities for all samples
        threshold: Optional minimum difference in scores to consider a preference
            If None, any difference in scores will generate a preference pair
        max_pairs: Optional maximum number of pairs to generate
            If None, all possible pairs will be generated
            
    Returns:
        Array of shape (n_pairs, 2) with indices of preferred pairs
        Each row (i, j) indicates that sample i is preferred over sample j
    """
    n_samples = len(labels)
    pairs = []
    
    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                continue
                
            diff = labels[i] - labels[j]
            if threshold is not None and abs(diff) < threshold:
                continue
                
            if diff > 0:  # i is preferred over j
                pairs.append((i, j))
                
    pairs = np.array(pairs)
    
    if max_pairs is not None and len(pairs) > max_pairs:
        # Randomly sample max_pairs from all pairs
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = pairs[indices]
        
    return pairs


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
        activation: nn.Module = nn.ReLU()
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
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        # Final layer to scalar preference score
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute preference score.
        
        Args:
            x: Tensor of shape (batch_size, embedding_dim) with embeddings
            
        Returns:
            Tensor of shape (batch_size, 1) with preference scores
        """
        return self.mlp(x)
    
    def predict_preference(
        self, emb_i: torch.Tensor, emb_j: torch.Tensor
    ) -> torch.Tensor:
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


def bradley_terry_loss(
    score_i: torch.Tensor, 
    score_j: torch.Tensor, 
    preference: torch.Tensor
) -> torch.Tensor:
    """Compute Bradley-Terry loss for preference learning.
    
    Args:
        score_i: Tensor of shape (batch_size, 1) with scores for first item
        score_j: Tensor of shape (batch_size, 1) with scores for second item
        preference: Tensor of shape (batch_size, 1) with preference labels
            1 indicates i is preferred over j, 0 indicates j is preferred over i
            
    Returns:
        Binary cross-entropy loss based on Bradley-Terry model
    """
    log_prob = F.logsigmoid(score_i - score_j)
    loss = -torch.mean(preference * log_prob + (1 - preference) * F.logsigmoid(score_j - score_i))
    return loss


class PreferenceTrainer:
    """Trainer for Bradley-Terry preference models.
    
    This class handles the training and evaluation of Bradley-Terry models,
    including data preparation, training loops, and evaluation metrics.
    """
    
    def __init__(
        self,
        model: BradleyTerryMLP,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: Optional[str] = None
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
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
    def train(
        self,
        train_dataset: PreferenceDataset,
        val_dataset: Optional[PreferenceDataset] = None,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the Bradley-Terry model on preference data.
        
        Args:
            train_dataset: Training dataset of preference pairs
            val_dataset: Optional validation dataset of preference pairs
            batch_size: Batch size for training
            epochs: Maximum number of epochs to train
            patience: Number of epochs to wait for validation improvement before early stopping
            verbose: Whether to print progress during training
            
        Returns:
            Dictionary with training metrics:
                'train_loss': List of training losses for each epoch
                'val_loss': List of validation losses for each epoch (if val_dataset provided)
        """
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False
            )
            
        metrics = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        no_improve_epochs = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for emb_i, emb_j, label in train_loader:
                emb_i, emb_j, label = (
                    emb_i.to(self.device), 
                    emb_j.to(self.device), 
                    label.to(self.device)
                )
                
                self.optimizer.zero_grad()
                
                score_i = self.model(emb_i)
                score_j = self.model(emb_j)
                
                loss = bradley_terry_loss(score_i, score_j, label)
                loss.backward()
                
                self.optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            metrics['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                metrics['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    
                if no_improve_epochs >= patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}"
                    )
            elif verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}"
                )
                
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on a dataloader.
        
        Args:
            dataloader: DataLoader of preference pairs
            
        Returns:
            Average loss on the dataloader
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for emb_i, emb_j, label in dataloader:
                emb_i, emb_j, label = (
                    emb_i.to(self.device), 
                    emb_j.to(self.device), 
                    label.to(self.device)
                )
                
                score_i = self.model(emb_i)
                score_j = self.model(emb_j)
                
                loss = bradley_terry_loss(score_i, score_j, label)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
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
            scores = self.model(embeddings_tensor).squeeze(-1).cpu().numpy()
            
        return scores
    
    def evaluate_ranking(
        self, 
        embeddings: np.ndarray, 
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate ranking performance on embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings
            true_labels: Array of shape (n_samples,) with true scores/activities
            
        Returns:
            Dictionary with evaluation metrics:
                'spearman_corr': Spearman rank correlation between predicted and true scores
                'preference_accuracy': Accuracy of pairwise preference predictions
        """
        predicted_scores = self.predict_scores(embeddings)
        
        # Spearman correlation
        spearman_corr, _ = spearmanr(predicted_scores, true_labels)
        
        # Generate all possible pairwise preferences
        all_pairs = generate_preference_pairs(true_labels, threshold=0)
        n_pairs = len(all_pairs)
        
        # Compute preference accuracy
        correct_prefs = 0
        for i, j in all_pairs:
            pred_i = predicted_scores[i]
            pred_j = predicted_scores[j]
            if pred_i > pred_j:  # Predicted preference matches true preference
                correct_prefs += 1
                
        preference_accuracy = correct_prefs / n_pairs if n_pairs > 0 else 0.0
        
        # Compute ROC AUC for preference prediction
        y_true = np.ones(n_pairs)  # All pairs in all_pairs are positive preferences
        y_pred = np.array([predicted_scores[i] - predicted_scores[j] for i, j in all_pairs])
        auc = roc_auc_score(y_true, y_pred) if n_pairs > 0 else 0.0
        
        return {
            'spearman_corr': spearman_corr,
            'preference_accuracy': preference_accuracy,
            'auc': auc
        }
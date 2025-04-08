"""
Utilities for testing FolDE functionality.

This module provides shared functions and classes for testing the FolDE
protein engineering prediction system.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from folde.few_shot_models import FewShotModel, register_few_shot_model
from folde.zero_shot_models import ZeroShotModel, register_zeroshot_model


def create_simulated_protein_dataset(
    num_samples: int = 100, embedding_dim: int = 10, random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a simulated dataset for testing FolDE functionality.

    This creates three synchronized dataframes with the same seq_ids:
    1. An activity dataframe with DMS scores
    2. A naturalness dataframe with wt_marginal values
    3. An embedding dataframe with protein embeddings

    Args:
        num_samples: Number of protein samples to generate
        embedding_dim: Dimension of the protein embeddings
        random_seed: Optional random seed for reproducibility

    Returns:
        Tuple of (activity_df, naturalness_df, embedding_df)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create sequence IDs
    seq_ids = [f"seq_{i}" for i in range(num_samples)]

    # Create activity data
    activity_values = np.random.normal(0, 1, num_samples)
    activity_df = pd.DataFrame(
        {
            "seq_id": seq_ids,
            "DMS_score": activity_values,
            "mutant": [f"A{i}G" for i in range(num_samples)],
        }
    )
    activity_df = activity_df.set_index("seq_id", drop=False)

    # Create naturalness data
    naturalness_values = np.random.normal(-10, 2, num_samples)
    naturalness_df = pd.DataFrame(
        {"seq_id": seq_ids, "wt_marginal": naturalness_values}
    )
    naturalness_df = naturalness_df.set_index("seq_id", drop=False)

    # Create embedding data
    embeddings = [np.random.normal(0, 1, embedding_dim) for _ in range(num_samples)]
    embedding_df = pd.DataFrame({"seq_id": seq_ids, "embedding": embeddings})
    embedding_df = embedding_df.set_index("seq_id", drop=False)

    return activity_df, naturalness_df, embedding_df


@register_zeroshot_model
class MockZeroShotModel(ZeroShotModel):
    """Mock ZeroShotModel for testing campaign simulations."""

    def __init__(
        self, temperature=0.0, return_values=None, naturalness_col="wt_marginal"
    ):
        """Initialize mock model.

        Args:
            temperature: Temperature parameter for Boltzmann sampling
            return_values: Optional fixed values to return for predictions
            naturalness_col: Column in naturalness dataframe to use for predictions
        """
        super().__init__(temperature=temperature)
        self.return_values = return_values
        self.naturalness_col = naturalness_col
        self.predict_called = False
        self.predict_inputs = []

    def predict(self, naturalness_df, embedding_df=None):
        """Mock predict method."""
        self.predict_called = True
        self.predict_inputs.append((naturalness_df, embedding_df))

        if self.return_values is not None:
            return self.return_values

        # By default, return values proportional to naturalness scores
        return naturalness_df[self.naturalness_col].values


@register_few_shot_model
class MockFewShotModel(FewShotModel):
    """Mock FewShotModel for testing campaign simulations."""

    def __init__(self, return_values=None):
        """Initialize mock model.

        Args:
            return_values: Optional fixed values to return for predictions
        """
        self.return_values = return_values
        self.fit_called = False
        self.fit_inputs = []
        self.predict_called = False
        self.predict_inputs = []

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MockFewShotModel":
        """Mock fit method."""
        self.fit_called = True
        self.fit_inputs.append((X, y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock predict method."""
        self.predict_called = True
        self.predict_inputs.append(X)

        if self.return_values is not None:
            return self.return_values

        # By default, return random values with slight correlation to input
        random_values = np.random.rand(len(X))
        # Add a small correlation with the average of each embedding
        if X.ndim > 1:
            correlation = np.mean(X, axis=1) * 0.2
            return random_values + correlation
        return random_values

    def get_debug_info(self) -> Dict[str, Any]:
        """Mock debug info method."""
        return {
            "model_type": "MockFewShotModel",
            "fit_called": self.fit_called,
            "predict_called": self.predict_called,
        }

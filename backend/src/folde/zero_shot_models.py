"""
Implementation of zero-shot protein prediction models.

This module contains models that can predict protein properties
without training on labeled data. These models are useful for
low-N protein engineering campaigns where little training data
is available.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import numpy as np
import pandas as pd
from folde.util import internal_sample_n_indices
from numpy.typing import NDArray
from pandas import DataFrame, Series

# Registry of available zero-shot models
_ZERO_SHOT_MODELS = {}


# def register_zeroshot_model(name):
#     """Decorator to register a zero-shot model class."""

#     def decorator(cls):
#         _ZERO_SHOT_MODELS[name] = cls
#         return cls

#     return decorator


class ZeroShotModel(ABC):
    """Abstract base class for zero-shot prediction models.

    Zero-shot models can predict protein properties without
    requiring training on labeled fitness/activity data.
    """

    def __init__(self, temperature: float = 0.0, epsilon: float = 0.0):
        """Initialize the zero-shot model.

        Args:
            **kwargs: Model-specific parameters
        """
        self.temperature = temperature
        self.epsilon = epsilon

    @abstractmethod
    def predict(
        self, naturalness_df: DataFrame, embedding_df: Optional[DataFrame] = None
    ) -> NDArray[np.float64]:
        """Make predictions for protein variants.

        Args:
            naturalness_df: DataFrame containing naturalness scores
            embedding_df: Optional DataFrame containing protein embeddings

        Returns:
            Array of prediction scores for each variant
        """
        pass

    def get_top_n(
        self,
        n: int,
        naturalness_df: DataFrame,
        embedding_df: Optional[DataFrame] = None,
    ) -> Tuple[List[str], Series]:
        """Get the top N variants predicted by the model.

        This method predicts scores for all variants and returns the
        sequence IDs of the top N variants by predicted score.

        Args:
            n: Number of top variants to return
            naturalness_df: DataFrame containing naturalness scores
            embedding_df: Optional DataFrame containing protein embeddings

        Returns:
            Tuple of
              * List of sequence IDs for the top N variants.
              * Series of predictions for all input variants with seq_id as index.
        """
        if "seq_id" not in naturalness_df.columns:
            raise ValueError(
                f"naturalness_df must contain 'seq_id' column, found {naturalness_df.columns}"
            )

        # Get predictions
        predictions = self.predict(naturalness_df, embedding_df)

        # Create a DataFrame with sequence IDs and predictions
        results_df = pd.DataFrame({"seq_id": naturalness_df["seq_id"], "prediction": predictions})

        chosen_indices = internal_sample_n_indices(
            results_df.prediction.values,
            n,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )
        chosen_seqs = results_df.iloc[chosen_indices]

        return (
            chosen_seqs["seq_id"].tolist(),
            results_df.set_index("seq_id").prediction,
        )

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the model.

        Returns:
            Dictionary containing model parameters and other debug info
        """
        # Default implementation without model_params
        return {"model_type": self.__class__.__name__}


def register_zeroshot_model(model_class: Type[ZeroShotModel]) -> Type[ZeroShotModel]:
    _ZERO_SHOT_MODELS[model_class.__name__] = model_class
    return model_class


@register_zeroshot_model
class RandomZeroShotModel(ZeroShotModel):
    """Zero-shot prediction model based on sequence naturalness scores.

    This model uses sequence naturalness (log-likelihood) scores
    directly as the prediction, optionally with some transformation.
    """

    def predict(
        self, naturalness_df: DataFrame, embedding_df: Optional[DataFrame] = None
    ) -> NDArray[np.float64]:
        """Predict using naturalness scores.

        Args:
            naturalness_df: DataFrame containing 'wt_marginal' column
            embedding_df: Not used by this model, but included for API consistency

        Returns:
            Array of prediction scores based on naturalness
        """
        return np.random.rand(naturalness_df.shape[0])

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the model.

        Returns:
            Dictionary containing model parameters
        """
        return {}


@register_zeroshot_model
class NaturalnessZeroShotModel(ZeroShotModel):
    """Zero-shot prediction model based on sequence naturalness scores.

    This model uses sequence naturalness (log-likelihood) scores
    directly as the prediction, optionally with some transformation.
    """

    def __init__(self, naturalness_col: str = "wt_marginal", **kwargs):
        """Initialize the naturalness-based model.

        Args:
            transformation: Transformation to apply to naturalness scores.
                Options: "identity", "log", "exp", "neg"
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.naturalness_col = naturalness_col

    def predict(
        self, naturalness_df: DataFrame, embedding_df: Optional[DataFrame] = None
    ) -> NDArray[np.float64]:
        """Predict using naturalness scores.

        Args:
            naturalness_df: DataFrame containing 'wt_marginal' column
            embedding_df: Not used by this model, but included for API consistency

        Returns:
            Array of prediction scores based on naturalness
        """
        if self.naturalness_col not in naturalness_df.columns:
            raise ValueError(
                f"naturalness_df must contain '{self.naturalness_col}' column, got {naturalness_df.columns}"
            )

        scores = naturalness_df[self.naturalness_col].values
        return cast(NDArray[np.float64], scores)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the model.

        Returns:
            Dictionary containing model parameters
        """
        info = super().get_debug_info()
        info["naturalness_col"] = self.naturalness_col
        return info


def get_zero_shot_model(model_name: str, **kwargs) -> ZeroShotModel:
    """Get a zero-shot model instance by name.

    Args:
        model_name: Name of the zero-shot model to instantiate
        **kwargs: Parameters to pass to the model constructor

    Returns:
        Instantiated zero-shot model

    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name not in _ZERO_SHOT_MODELS:
        available_models = list(_ZERO_SHOT_MODELS.keys())
        raise ValueError(
            f"Unknown zero-shot model: {model_name}. Available models: {available_models}"
        )

    model_class = _ZERO_SHOT_MODELS[model_name]
    return model_class(**kwargs)

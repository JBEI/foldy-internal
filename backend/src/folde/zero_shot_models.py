"""
Implementation of zero-shot protein prediction models.

This module contains models that can predict protein properties
without training on labeled data. These models are useful for
low-N protein engineering campaigns where little training data
is available.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsRegressor

from app.helpers.sequence_util import is_homolog_seq_id
from folde.util import get_consensus_scores, internal_sample_n_indices

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

    def pretrain(
        self,
        naturalness_series: pd.Series,
        embedding_series: pd.Series,
    ) -> "ZeroShotModel":
        """Optional method to pretrain the model on naturalness and embedding data."""
        return self

    @abstractmethod
    def predict(
        self, naturalness_series: pd.Series, embedding_series: Optional[pd.Series] = None
    ) -> List[pd.Series]:
        """Make predictions for protein variants.

        Args:
            naturalness_series: Series containing naturalness scores
            embedding_series: Optional Series containing protein embeddings

        Returns:
            Array of prediction scores for each variant
        """
        pass

    def get_top_n(
        self,
        n: int,
        naturalness_series: pd.Series,
        embedding_series: Optional[pd.Series] = None,
    ) -> Tuple[List[str], List[pd.Series]]:
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
        # Get predictions
        ensemble_of_predictions = self.predict(naturalness_series, embedding_series)

        consensus_scores = get_consensus_scores(ensemble_of_predictions, decision_mode="median")

        # Create a DataFrame with sequence IDs and predictions
        # results_df = pd.DataFrame({"seq_id": naturalness_series.index, "prediction": predictions})

        chosen_indices = internal_sample_n_indices(
            consensus_scores.to_numpy(),
            n,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )

        return (
            naturalness_series.index[chosen_indices].tolist(),
            ensemble_of_predictions,
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
        self, naturalness_series: pd.Series, embedding_series: Optional[pd.Series] = None
    ) -> List[pd.Series]:
        """Predict using naturalness scores.

        Args:
            naturalness_df: DataFrame containing 'wt_marginal' column
            embedding_df: Not used by this model, but included for API consistency

        Returns:
            Array of prediction scores based on naturalness
        """
        return [
            pd.Series(np.random.rand(naturalness_series.shape[0]), index=naturalness_series.index)
        ]

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

    def __init__(self, **kwargs):
        """Initialize the naturalness-based model.

        Args:
            transformation: Transformation to apply to naturalness scores.
                Options: "identity", "log", "exp", "neg"
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.is_pretrained = False

    def pretrain(
        self,
        naturalness_series: pd.Series,
        embedding_series: pd.Series,
    ) -> "NaturalnessZeroShotModel":
        assert naturalness_series.index.equals(embedding_series.index)
        assert embedding_series.index.is_unique, "embedding_series contains duplicate indices"

        if self.is_pretrained:
            raise ValueError("Model is already pretrained.")

        has_naturalness_data = ~naturalness_series.isna()  # index.map(lambda sid: "_" not in sid)

        logging.info(
            f"Pretraining model with naturalness data with {sum(has_naturalness_data)} naturalness measurements."
        )

        embedding_series_with_data = embedding_series[has_naturalness_data]
        naturalness_series_with_data = naturalness_series[has_naturalness_data]

        X = np.array([np.array(emb) for emb in embedding_series_with_data.values])
        y = naturalness_series_with_data.values

        # Fit KNN regressor
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X, y)
        self.is_pretrained = True
        self.knn = knn
        return self

    def predict(
        self, naturalness_series: pd.Series, embedding_series: Optional[pd.Series] = None
    ) -> List[pd.Series]:
        """Predict using naturalness scores.

        Args:
            naturalness_series: Series containing naturalness scores, some of which may be NAN.
            embedding_series: Optional Series containing protein embeddings

        Returns:
            Array of prediction scores based on naturalness
        """

        def get_naturalness(seq_id, direct_naturalness) -> float:
            """Try computing naturalness for mutants even if none was provided by extrapolating for multimutants."""
            if direct_naturalness is not None and not pd.isna(direct_naturalness):
                return direct_naturalness

            if is_homolog_seq_id(seq_id):
                return np.nan

            # Break it down into single mutants.
            seq_id_parts = seq_id.split("_")

            # For multimutants, we compute naturalness as the product of the naturalness of the single mutants.
            computed_naturalness = naturalness_series.loc[seq_id_parts].prod()
            if pd.isna(computed_naturalness):
                raise ValueError(f"Computed naturalness is NAN for {seq_id} with parts {seq_id_parts}")
            return computed_naturalness

        computed_naturalness_series = naturalness_series.reset_index(name="wt_marginal").apply(
            lambda r: get_naturalness(r.seq_id, r.wt_marginal), axis=1
        )
        computed_naturalness_series.index = naturalness_series.index

        # Do KNN imputation to fill in NANs from homologs.
        if computed_naturalness_series.isna().any():
            if not self.is_pretrained:
                raise ValueError("Model is not pretrained, so cannot fill in NANs from homologs.")

            logging.info(f"Filling in NANs from homologs for {computed_naturalness_series.isna().sum()}/{len(computed_naturalness_series)} naturalness values.")
            assert embedding_series is not None
            embedding_array = np.array([np.array(emb) for emb in embedding_series.values])
            naturalness_array = computed_naturalness_series.values

            # Find indices for known and missing
            missing_mask = computed_naturalness_series.isna().to_numpy()
            X_missing = embedding_array[missing_mask]
            
            imputed_values = self.knn.predict(X_missing)

            # Fill in the missing values
            imputed_naturalness = naturalness_array.copy()
            imputed_naturalness[missing_mask] = imputed_values

            # Convert back to Series
            computed_naturalness_series = pd.Series(imputed_naturalness, index=naturalness_series.index)
        
        if computed_naturalness_series.isna().any():
            raise ValueError(f"Computed naturalness series still has NANs: {computed_naturalness_series.isna().sum()}/{len(computed_naturalness_series)}")

        return [computed_naturalness_series]

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the model.

        Returns:
            Dictionary containing model parameters
        """
        info = super().get_debug_info()
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

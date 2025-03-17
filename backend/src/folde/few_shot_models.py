"""
Machine learning models for protein engineering prediction tasks.

This module provides scikit-learn compatible models for predicting
protein properties from embeddings. Models are thin wrappers around
scikit-learn implementations with additional protein-specific functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Registry of available models
_FEW_SHOT_MODELS = {}


class FewShotModel(ABC):
    """Abstract base class for few-shot protein property prediction models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "FewShotModel":
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        pass

    def get_top_n(
        self, n: int, naturalness_df: pd.DataFrame, embedding_df: pd.DataFrame
    ) -> List[str]:
        """Get the top N variants predicted by the model.

        This method combines features from naturalness and embedding dataframes,
        makes predictions, and returns the sequence IDs of the top N variants.

        Args:
            n: Number of top variants to return
            naturalness_df: DataFrame containing naturalness scores
            embedding_df: DataFrame containing protein embeddings

        Returns:
            Tuple of
              * List of sequence IDs for the top N variants.
              * Series of predictions for all input variants with seq_id as index.

        Raises:
            ValueError: If the model is not fitted or if required columns are missing
        """

        # Convert list of embeddings to numpy array
        embeddings_array = np.array(
            [np.array(emb) for emb in embedding_df.embedding.values]
        )
        predictions = self.predict(embeddings_array)
        results_df = pd.DataFrame(
            {"seq_id": embedding_df["seq_id"], "prediction": predictions}
        )
        return (
            results_df.sort_values("prediction", ascending=False)
            .head(n)["seq_id"]
            .tolist(),
            results_df.set_index("seq_id").prediction,
        )

    @abstractmethod
    def get_debug_info(self) -> Dict[str, Any]:
        """Get model-specific debug information."""
        pass


def register_few_shot_model(model_class: Type[FewShotModel]):
    _FEW_SHOT_MODELS[model_class.__name__] = model_class
    return model_class


# def register_few_shot_model(name: str):
#     """Register a model in the global registry."""

#     def decorator(cls):
#         _FEW_SHOT_MODELS[name] = cls
#         return cls

#     return decorator


@register_few_shot_model
class MLPFewShotModel(FewShotModel):
    """Multi-layer Perceptron regressor for protein property prediction.

    Thin wrapper around sklearn's MLPRegressor. See scikit-learn documentation
    for full parameter details: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """

    def __init__(self, **kwargs):
        """Initialize the MLP regressor with any parameters supported by sklearn's MLPRegressor."""
        self.model = SklearnMLPRegressor(**kwargs)
        self.metrics_ = {}
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs,
    ) -> "MLPRegressor":
        """Train the MLP regressor.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            validation_data: Optional tuple (X_val, y_val) for validation
            **kwargs: Additional parameters passed to sklearn's fit method

        Returns:
            Self for method chaining
        """
        # Train the model
        self.model.fit(X, y, **kwargs)
        self._is_fitted = True

        # Calculate training metrics
        y_train_pred = self.model.predict(X)
        self.metrics_ = {
            "train_mse": mean_squared_error(y, y_train_pred),
            "train_r2": r2_score(y, y_train_pred),
            "train_mae": mean_absolute_error(y, y_train_pred),
        }

        # Calculate validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val_pred = self.model.predict(X_val)
            self.metrics_.update(
                {
                    "val_mse": mean_squared_error(y_val, y_val_pred),
                    "val_r2": r2_score(y_val, y_val_pred),
                    "val_mae": mean_absolute_error(y_val, y_val_pred),
                }
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained MLP."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict(X)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the MLP."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Basic debug info
        debug_info = {
            "metrics": self.metrics_,
            "n_layers": self.model.n_layers_,
            "n_outputs": self.model.n_outputs_,
            "n_iter": self.model.n_iter_,
            "loss_curve": (
                self.model.loss_curve_ if hasattr(self.model, "loss_curve_") else None
            ),
        }

        # Network structure
        network_structure = []
        for i, (coef, intercept) in enumerate(
            zip(self.model.coefs_, self.model.intercepts_)
        ):
            layer_info = {
                "layer": i + 1,
                "shape": coef.shape,
                "n_params": coef.size + intercept.size,
            }
            network_structure.append(layer_info)

        debug_info["network_structure"] = network_structure

        return debug_info


@register_few_shot_model
class RandomForestFewShotModel(FewShotModel):
    """Random Forest regressor for protein property prediction.

    Thin wrapper around sklearn's RandomForestRegressor. See scikit-learn documentation
    for full parameter details: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """

    def __init__(self, **kwargs):
        """Initialize the Random Forest regressor with any parameters supported by sklearn's RandomForestRegressor."""
        self.model = SklearnRandomForestRegressor(**kwargs)
        self.metrics_ = {}
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs,
    ) -> "RandomForestRegressor":
        """Train the Random Forest regressor.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            validation_data: Optional tuple (X_val, y_val) for validation
            **kwargs: Additional parameters passed to sklearn's fit method

        Returns:
            Self for method chaining
        """
        # Train the model
        self.model.fit(X, y, **kwargs)
        self._is_fitted = True

        # Calculate training metrics
        y_train_pred = self.model.predict(X)
        self.metrics_ = {
            "train_mse": mean_squared_error(y, y_train_pred),
            "train_r2": r2_score(y, y_train_pred),
            "train_mae": mean_absolute_error(y, y_train_pred),
        }

        # Calculate validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val_pred = self.model.predict(X_val)
            self.metrics_.update(
                {
                    "val_mse": mean_squared_error(y_val, y_val_pred),
                    "val_r2": r2_score(y_val, y_val_pred),
                    "val_mae": mean_absolute_error(y_val, y_val_pred),
                }
            )

        # Add OOB score if available
        if hasattr(self.model, "oob_score_"):
            self.metrics_["oob_score"] = self.model.oob_score_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained Random Forest."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict(X)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the Random Forest."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Get feature importances
        feature_importances = {
            "mean": self.model.feature_importances_.tolist(),
            "std": np.std(
                [tree.feature_importances_ for tree in self.model.estimators_], axis=0
            ).tolist(),
        }

        # Basic tree info (limit to first 5 trees to avoid excessive info)
        tree_info = []
        for i, tree in enumerate(self.model.estimators_[:5]):
            tree_info.append(
                {
                    "tree_idx": i,
                    "n_nodes": tree.tree_.node_count,
                    "max_depth": tree.tree_.max_depth,
                }
            )

        return {
            "metrics": self.metrics_,
            "n_estimators": len(self.model.estimators_),
            "feature_importances": feature_importances,
            "tree_info": tree_info,
        }


def get_few_shot_model(model_name: str, **kwargs) -> FewShotModel:
    """Get a few-shot model by name with the provided parameters.

    Args:
        model_name: Name of the model to create ('mlp' or 'random_forest')
        **kwargs: Parameters passed directly to the model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name not in _FEW_SHOT_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(_FEW_SHOT_MODELS.keys())}"
        )

    model_class = _FEW_SHOT_MODELS[model_name]
    return model_class(**kwargs)

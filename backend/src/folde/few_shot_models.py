"""
Machine learning models for protein engineering prediction tasks.

This module provides scikit-learn compatible models for predicting
protein properties from embeddings. Models are thin wrappers around
scikit-learn implementations with additional protein-specific functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
from app.helpers.preference_ranking import create_preference_model
from folde.util import internal_sample_n_indices
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor

# Registry of available models
_FEW_SHOT_MODELS = {}


class FewShotModel(ABC):
    """Abstract base class for few-shot protein property prediction models."""

    def __init__(self, temperature: float = 0.0, epsilon: float = 0.0):
        self.temperature = temperature
        self.epsilon = epsilon

    @abstractmethod
    def fit(
        self,
        naturalness_series: pd.Series,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        **kwargs,
    ) -> "FewShotModel":
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, naturalness_series: pd.Series, embedding_series: pd.Series) -> pd.Series:
        """Make predictions using the trained model."""
        pass

    def get_top_n(
        self, n: int, naturalness_series: pd.Series, embedding_series: pd.Series
    ) -> Tuple[List[str], pd.Series]:
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
        assert naturalness_series.index.equals(embedding_series.index)

        # Convert list of embeddings to numpy array
        predictions = self.predict(naturalness_series, embedding_series)

        chosen_indices = internal_sample_n_indices(
            predictions.values,
            n,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )

        return (
            naturalness_series.index[chosen_indices].tolist(),
            predictions,
        )

    @abstractmethod
    def get_debug_info(self) -> Dict[str, Any]:
        """Get model-specific debug information."""
        pass


def register_few_shot_model(model_class: Type[FewShotModel]) -> Type[FewShotModel]:
    _FEW_SHOT_MODELS[model_class.__name__] = model_class
    return model_class


# def register_few_shot_model(name: str):
#     """Register a model in the global registry."""

#     def decorator(cls):
#         _FEW_SHOT_MODELS[name] = cls
#         return cls

#     return decorator


def get_ensemble_prediction(
    model_list: List[Any], X: NDArray[np.float64], decision_mode: str
) -> NDArray[np.float64]:
    """Get the prediction of an ensemble using deicision mode (often max or median)."""
    pred_list = []
    for model in model_list:
        pred_list.append(model.predict(X))
    pred_arr = np.stack(pred_list)
    if decision_mode == "max":
        return np.max(pred_arr, axis=0)
    elif decision_mode == "ucb":
        return np.mean(pred_arr, axis=0) + np.std(pred_arr, axis=0)
    elif decision_mode == "median":
        return np.median(pred_arr, axis=0)
    elif decision_mode == "mean":
        return np.mean(pred_arr, axis=0)
    raise ValueError(f"Invalid decision mode {decision_mode}")


@register_few_shot_model
class RandomFewShotModel(FewShotModel):
    """Just guess random activity."""

    def fit(
        self,
        naturalness_series: pd.Series,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        **kwargs,
    ) -> "RandomFewShotModel":
        return self

    def predict(self, naturalness_series: pd.Series, embedding_series: pd.Series) -> pd.Series:
        return pd.Series(
            np.random.rand(naturalness_series.shape[0]), index=naturalness_series.index
        )

    def get_debug_info(self) -> Dict[str, Any]:
        return {}


# TODO(jacob): Implement GP Model
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html


@register_few_shot_model
class MLPFewShotModel(FewShotModel):
    """Multi-layer Perceptron regressor for protein property prediction.

    Thin wrapper around sklearn's MLPRegressor. See scikit-learn documentation
    for full parameter details: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """

    def __init__(self, ensemble_size: int = 1, **kwargs):
        """Initialize the MLP regressor with any parameters supported by sklearn's MLPRegressor."""
        super().__init__(
            temperature=kwargs.pop("temperature", 0.0),
            epsilon=kwargs.pop("epsilon", 0.0),
        )
        self.ensemble_size = ensemble_size
        base_random_state = kwargs.pop("random_state", 0)
        self.models: List[SklearnMLPRegressor] = [
            SklearnMLPRegressor(**kwargs, random_state=base_random_state + ii)
            for ii in range(ensemble_size)
        ]
        self.metrics_: Dict[str, float] = {}
        self._is_fitted: bool = False

    def fit(
        self,
        naturalness_series: pd.Series,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        validation_data: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
        **kwargs,
    ) -> "MLPFewShotModel":
        """Train the MLP regressor.

        Args:
            naturalness_df: Naturalness dataframe for all mutants indexed by seq_id
            embedding_df: Embedding dataframe for all mutants indexed by seq_id
            measured_activity_series: Series of activity measurements indexed by seq_id
            **kwargs: Additional parameters passed to sklearn's fit method

        Returns:
            Self for method chaining
        """
        assert naturalness_series.index.equals(embedding_series.index)

        measured_embedding_series = embedding_series.loc[measured_activity_series.index]
        X = np.array([np.array(emb) for emb in measured_embedding_series.values])
        y = measured_activity_series.to_numpy()
        # Train the model
        for model in self.models:
            model.fit(X, y, **kwargs)
        self._is_fitted = True

        # Calculate training metrics
        y_train_pred = get_ensemble_prediction(self.models, X, "median")
        self.metrics_ = {
            "train_mse": mean_squared_error(y, y_train_pred),
            "train_r2": r2_score(y, y_train_pred),
            "train_mae": mean_absolute_error(y, y_train_pred),
        }

        # Calculate validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val_pred = get_ensemble_prediction(self.models, X_val, "median")
            self.metrics_.update(
                {
                    "val_mse": mean_squared_error(y_val, y_val_pred),
                    "val_r2": r2_score(y_val, y_val_pred),
                    "val_mae": mean_absolute_error(y_val, y_val_pred),
                }
            )

        return self

    def predict(self, naturalness_series: pd.Series, embedding_series: pd.Series) -> pd.Series:
        """Make predictions using the trained MLP."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        X = np.array([np.array(emb) for emb in embedding_series.values])
        return pd.Series(
            get_ensemble_prediction(self.models, X, "ucb"), index=embedding_series.index
        )

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the MLP."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Basic debug info
        debug_info: Dict[str, Any] = {
            "metrics": self.metrics_,
        }

        # Get properties from the first model in the ensemble
        if len(self.models) > 0:
            first_model = self.models[0]

            if hasattr(first_model, "n_layers_"):
                debug_info["n_layers"] = first_model.n_layers_

            if hasattr(first_model, "n_outputs_"):
                debug_info["n_outputs"] = first_model.n_outputs_

            if hasattr(first_model, "n_iter_"):
                debug_info["n_iter"] = first_model.n_iter_

            if hasattr(first_model, "loss_curve_"):
                debug_info["loss_curve"] = first_model.loss_curve_

            # Network structure
            if hasattr(first_model, "coefs_") and hasattr(first_model, "intercepts_"):
                network_structure = []
                for i, (coef, intercept) in enumerate(
                    zip(first_model.coefs_, first_model.intercepts_)
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

    def __init__(self, ensemble_size: int = 1, **kwargs):
        """Initialize the Random Forest regressor with any parameters supported by sklearn's RandomForestRegressor."""
        super().__init__(
            temperature=kwargs.pop("temperature", 0.0),
            epsilon=kwargs.pop("epsilon", 0.0),
        )
        self.ensemble_size = ensemble_size
        base_random_state = kwargs.pop("random_state", 0)
        self.models: List[SklearnRandomForestRegressor] = [
            SklearnRandomForestRegressor(**kwargs, random_state=base_random_state + ii)
            for ii in range(ensemble_size)
        ]
        self.metrics_: Dict[str, float] = {}
        self._is_fitted: bool = False
        self.decision_mode = kwargs.pop("decision_mode", "median")

    def fit(
        self,
        naturalness_series: pd.Series,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        validation_data: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
        **kwargs,
    ) -> "RandomForestFewShotModel":
        """Train the Random Forest regressor.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            validation_data: Optional tuple (X_val, y_val) for validation
            **kwargs: Additional parameters passed to sklearn's fit method

        Returns:
            Self for method chaining
        """
        assert naturalness_series.index.equals(embedding_series.index)

        measured_embedding_series = embedding_series.loc[measured_activity_series.index]
        X = np.array([np.array(emb) for emb in measured_embedding_series.values])
        y = measured_activity_series.to_numpy()

        # Train the model
        for model in self.models:
            model.fit(X, y, **kwargs)
        self._is_fitted = True

        # Calculate training metrics
        y_train_pred = get_ensemble_prediction(self.models, X, "median")
        self.metrics_ = {
            "train_mse": mean_squared_error(y, y_train_pred),
            "train_r2": r2_score(y, y_train_pred),
            "train_mae": mean_absolute_error(y, y_train_pred),
        }

        # Calculate validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val_pred = get_ensemble_prediction(self.models, X_val, "median")
            self.metrics_.update(
                {
                    "val_mse": mean_squared_error(y_val, y_val_pred),
                    "val_r2": r2_score(y_val, y_val_pred),
                    "val_mae": mean_absolute_error(y_val, y_val_pred),
                }
            )

        # Add OOB score if available
        if hasattr(self.models, "oob_score_"):
            self.metrics_["oob_score"] = self.models[0].oob_score_

        return self

    def predict(self, naturalness_series: pd.Series, embedding_series: pd.Series) -> pd.Series:
        """Make predictions using the trained Random Forest."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        X = np.array([np.array(emb) for emb in embedding_series.values])
        return pd.Series(
            get_ensemble_prediction(self.models, X, self.decision_mode),
            index=embedding_series.index,
        )

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the Random Forest."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        debug_info: Dict[str, Any] = {
            "metrics": self.metrics_,
        }

        # Get properties from the first model in the ensemble
        if len(self.models) > 0:
            first_model = self.models[0]

            # Get feature importances if available
            if hasattr(first_model, "feature_importances_"):
                feature_importances = {"mean": first_model.feature_importances_.tolist()}

                # Add standard deviation if we have estimators
                if hasattr(first_model, "estimators_") and len(first_model.estimators_) > 0:
                    feature_importances["std"] = np.std(
                        [tree.feature_importances_ for tree in first_model.estimators_],
                        axis=0,
                    ).tolist()

                debug_info["feature_importances"] = feature_importances

                # Basic tree info (limit to first 5 trees to avoid excessive info)
                if hasattr(first_model, "estimators_"):
                    tree_info = []
                    for i, tree in enumerate(first_model.estimators_[:5]):
                        if hasattr(tree, "tree_"):
                            tree_info.append(
                                {
                                    "tree_idx": i,
                                    "n_nodes": tree.tree_.node_count,
                                    "max_depth": tree.tree_.max_depth,
                                }
                            )

                    if tree_info:
                        debug_info["tree_info"] = tree_info

                    debug_info["n_estimators"] = len(first_model.estimators_)

        return debug_info


@register_few_shot_model
class TorchMLPFewShotModel(FewShotModel):
    """Custom, torch-backed MLP model."""

    def __init__(self, **kwargs):
        """Initialize the Random Forest regressor with any parameters supported by sklearn's RandomForestRegressor."""
        super().__init__(
            temperature=kwargs.pop("temperature", 0.0),
            epsilon=kwargs.pop("epsilon", 0.0),
        )
        self.model, self.trainer = create_preference_model(**kwargs)

    def fit(
        self,
        naturalness_series: pd.Series,
        embedding_series: pd.Series,
        measured_activity_series: pd.Series,
        validation_data: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
        **kwargs,
    ) -> "TorchMLPFewShotModel":
        assert naturalness_series.index.equals(embedding_series.index)

        measured_embedding_series = embedding_series.loc[measured_activity_series.index]
        X = np.array([np.array(emb) for emb in measured_embedding_series.values])
        y = measured_activity_series.to_numpy()

        # assert False, "Double check this logic for validation data."
        validation_indices = []
        if validation_data:
            validation_indices = list(range(X.shape[0], X.shape[0] + validation_data[0].shape[0]))
            X = np.concatenate([X, validation_data[0]])
            y = np.concatenate([y, validation_data[1]])

        TRAIN_EPOCHS = 100 * int(256 / y.shape[0])
        PATIENCE = 200
        USE_MSE_LOSS = False
        VALIDATION_FREQUENCY = 10
        metrics = self.trainer.train(
            embeddings=X,
            activity_labels=y,
            val_ratio_or_indices=validation_indices,  # No validation data.
            batch_size=min(16, y.shape[0]),
            epochs=TRAIN_EPOCHS,
            patience=PATIENCE,
            verbose=True,
            use_mse_loss=USE_MSE_LOSS,
            val_frequency=VALIDATION_FREQUENCY,
        )

        self._is_fitted = True
        self._fitting_metrics = metrics

        return self

    def predict(self, naturalness_series: pd.Series, embedding_series: pd.Series) -> pd.Series:
        """Make predictions using the trained Random Forest."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        X = np.array([np.array(emb) for emb in embedding_series.values])
        pred = self.trainer.predict_scores(X)
        return pd.Series(pred, index=embedding_series.index)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for the Random Forest."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self._fitting_metrics


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

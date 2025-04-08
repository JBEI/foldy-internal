"""
Tests for the testing utilities.

This module contains tests for the test_utils module to ensure that
the testing utilities work correctly.
"""

import numpy as np
import pandas as pd
import pytest
from folde.tests.test_utils import (
    MockFewShotModel,
    MockZeroShotModel,
    create_simulated_protein_dataset,
)


def test_create_simulated_protein_dataset():
    """Test the simulated dataset creation."""
    num_samples = 50
    embedding_dim = 8
    random_seed = 42

    # Create the dataset
    activity_df, naturalness_df, embedding_df = create_simulated_protein_dataset(
        num_samples=num_samples, embedding_dim=embedding_dim, random_seed=random_seed
    )

    # Verify the datasets
    assert len(activity_df) == num_samples
    assert len(naturalness_df) == num_samples
    assert len(embedding_df) == num_samples

    # Verify the structure of each dataframe
    assert "seq_id" in activity_df.columns
    assert "DMS_score" in activity_df.columns
    assert "mutant" in activity_df.columns
    assert activity_df.index.name == "seq_id"

    assert "seq_id" in naturalness_df.columns
    assert "wt_marginal" in naturalness_df.columns
    assert naturalness_df.index.name == "seq_id"

    assert "seq_id" in embedding_df.columns
    assert "embedding" in embedding_df.columns
    assert embedding_df.index.name == "seq_id"

    # Check that embeddings have the correct dimension
    assert len(embedding_df.embedding.iloc[0]) == embedding_dim

    # Check that random seed works by creating datasets with different seeds
    activity_df2, _, _ = create_simulated_protein_dataset(
        num_samples=num_samples, random_seed=43
    )

    # The datasets should be different with different seeds
    assert not np.array_equal(
        activity_df.DMS_score.values, activity_df2.DMS_score.values
    )

    # The datasets should be the same with the same seed
    activity_df3, _, _ = create_simulated_protein_dataset(
        num_samples=num_samples, random_seed=42
    )
    assert np.array_equal(activity_df.DMS_score.values, activity_df3.DMS_score.values)


def test_mock_zero_shot_model():
    """Test the mock zero-shot model."""
    # Create a simulated dataset
    activity_df, naturalness_df, embedding_df = create_simulated_protein_dataset(
        random_seed=42
    )

    # Create a mock model
    model = MockZeroShotModel(temperature=0.1)

    # Test prediction
    predictions = model.predict(naturalness_df, embedding_df)
    assert len(predictions) == len(naturalness_df)
    assert np.array_equal(predictions, naturalness_df.wt_marginal.values)

    # Test get_top_n
    top_n = 5
    top_seq_ids, pred_series = model.get_top_n(top_n, naturalness_df, embedding_df)
    assert len(top_seq_ids) == top_n
    assert len(pred_series) == len(naturalness_df)

    # With temperature=0, top_n should be deterministic
    model.temperature = 0.0
    top_seq_ids_det, _ = model.get_top_n(top_n, naturalness_df, embedding_df)

    # Should get the highest wt_marginal values
    top_by_naturalness = naturalness_df.sort_values("wt_marginal", ascending=False)
    assert set(top_seq_ids_det) == set(top_by_naturalness.head(top_n).index)


def test_mock_few_shot_model():
    """Test the mock few-shot model."""
    # Create a simulated dataset
    activity_df, naturalness_df, embedding_df = create_simulated_protein_dataset(
        random_seed=42
    )

    # Create a mock model
    model = MockFewShotModel()

    # Test fit
    X = np.array([np.array(x) for x in embedding_df.embedding.values])
    y = activity_df.DMS_score.values
    model.fit(X, y)

    # Verify fit was called
    assert model.fit_called
    assert len(model.fit_inputs) == 1
    assert np.array_equal(model.fit_inputs[0][0], X)
    assert np.array_equal(model.fit_inputs[0][1], y)

    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(X)

    # Test get_top_n
    top_n = 5
    top_seq_ids, pred_series = model.get_top_n(top_n, naturalness_df, embedding_df)
    assert len(top_seq_ids) == top_n
    assert len(pred_series) == len(embedding_df)

    # Verify debug info
    debug_info = model.get_debug_info()
    assert debug_info["model_type"] == "MockFewShotModel"
    assert debug_info["fit_called"] is True
    assert debug_info["predict_called"] is True

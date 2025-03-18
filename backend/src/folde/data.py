"""
Data loading utilities for protein engineering prediction tasks.

This module provides functions for loading and processing protein engineering
datasets, including Deep Mutational Scanning (DMS) data from ProteinGym.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import os
import glob
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import json
import re

logger = logging.getLogger(__name__)

# Constants for data locations
MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = MODULE_DIR / "data"
DMS_DIR = DATA_DIR / "DMS_ProteinGym_substitutions"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
NATURALNESS_DIR = DATA_DIR / "naturalness"
DMS_METADATA_FILE = DATA_DIR / "DMS_substitutions.csv"


def maybe_modify_seq_id(dms_id: str, seq_id: str) -> str:
    """Modify the sequence ID if necessary."""
    if dms_id == "A0A140D2T1_ZIKV_Sourisseau_2019":
        m = re.match(r"^([A-Z])([0-9]+)([A-Z])$", seq_id)
        if not m:
            return seq_id
        return f"{m.group(1)}{int(m.group(2)) + 290}{m.group(3)}"
    return seq_id


def get_available_proteingym_datasets(
    embedding_model_id: str, naturalness_model_id: str
) -> pd.DataFrame:
    """Get metadata for available datasets with specific embedding and naturalness models.

    Args:
        embedding_model_id: The specific embedding model ID to search for
        naturalness_model_id: The specific naturalness model ID to search for

    Returns:
        DataFrame containing metadata for datasets with the specified models
    """
    # Check if metadata file exists
    if not os.path.exists(DMS_METADATA_FILE):
        logger.error(f"DMS metadata file not found at {DMS_METADATA_FILE}")
        return pd.DataFrame()

    # Load metadata
    try:
        dms_metadata = pd.read_csv(DMS_METADATA_FILE)
        logger.info(f"Loaded metadata for {len(dms_metadata)} DMS datasets")
    except Exception as e:
        logger.error(f"Error loading DMS metadata: {e}")
        return pd.DataFrame()

    # Find datasets that have both the specified embedding and naturalness files
    available_datasets = []

    for _, row in dms_metadata.iterrows():
        dms_id = row["DMS_id"]

        # Check if both required files exist
        embedding_file = os.path.join(
            EMBEDDINGS_DIR, f"{dms_id}_embedding_{embedding_model_id}.csv"
        )
        naturalness_file = os.path.join(
            NATURALNESS_DIR, f"{dms_id}_naturalness_{naturalness_model_id}.csv"
        )

        if os.path.exists(embedding_file) and os.path.exists(naturalness_file):
            available_datasets.append(dms_id)

    # Filter metadata to only include datasets with both required files
    filtered_metadata = dms_metadata[
        dms_metadata["DMS_id"].isin(available_datasets)
    ].copy()

    logger.info(
        f"Found {len(filtered_metadata)} datasets with embedding model '{embedding_model_id}' and naturalness model '{naturalness_model_id}'"
    )
    return filtered_metadata


def get_proteingym_dataset(
    dms_id: str, embedding_model_id: str, naturalness_model_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ProteinGym activity data, embeddings, and naturalness scores.

    Args:
        dms_id: Identifier for the DMS dataset (e.g., "BLAT_ECOLX_Stiffler_2015")
        embedding_model_id: Identifier for the embedding model (e.g., "300m")
        naturalness_model_id: Identifier for the naturalness model (e.g., "esm2")

    Returns:
        Tuple of (naturalness_df, embedding_df, activity_df)

    Raises:
        FileNotFoundError: If any required files are not found
    """
    # Check that the DMS data exists
    dms_file_path = os.path.join(DMS_DIR, f"{dms_id}.csv")
    if not os.path.exists(dms_file_path):
        raise FileNotFoundError(f"DMS data file not found: {dms_file_path}")

    # Check that the embedding file exists
    embedding_file_path = os.path.join(
        EMBEDDINGS_DIR, f"{dms_id}_embedding_{embedding_model_id}.csv"
    )
    if not os.path.exists(embedding_file_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_file_path}")

    # Check that the naturalness file exists
    naturalness_file_path = os.path.join(
        NATURALNESS_DIR, f"{dms_id}_naturalness_{naturalness_model_id}.csv"
    )
    if not os.path.exists(naturalness_file_path):
        raise FileNotFoundError(f"Naturalness file not found: {naturalness_file_path}")

    # Load DMS activity data
    activity_df = pd.read_csv(dms_file_path)
    logger.info(f"Loaded activity data for {dms_id} with {len(activity_df)} rows")

    # Convert 'mutant' column to 'seq_id' by replacing ':' with '_'
    activity_df["seq_id"] = activity_df["mutant"].str.replace(":", "_")
    activity_df = activity_df.set_index("seq_id", drop=False)

    # Load embeddings
    embedding_df = pd.read_csv(embedding_file_path)
    logger.info(f"Loaded embeddings for {dms_id} with {len(embedding_df)} rows")
    embedding_df["seq_id"] = embedding_df["seq_id"].apply(
        lambda x: maybe_modify_seq_id(dms_id, x)
    )
    embedding_df = embedding_df.set_index("seq_id", drop=False)

    # Load naturalness scores
    naturalness_df = pd.read_csv(naturalness_file_path)
    logger.info(
        f"Loaded naturalness scores for {dms_id} with {len(naturalness_df)} rows"
    )
    naturalness_df["seq_id"] = naturalness_df["seq_id"].apply(
        lambda x: maybe_modify_seq_id(dms_id, x)
    )
    naturalness_df = naturalness_df.set_index("seq_id", drop=False)

    # Ensure the naturalness file has the required column
    if "wt_marginal" not in naturalness_df.columns:
        raise ValueError(
            f"Naturalness file missing 'wt_marginal' column. Available columns: {naturalness_df.columns.tolist()}"
        )

    # Convert embedding column from string to numpy array if needed
    if isinstance(embedding_df["embedding"].iloc[0], str):
        # embedding_df["embedding"] = embedding_df["embedding"].apply(
        #     lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
        # )
        embedding_df["embedding"] = embedding_df["embedding"].apply(
            lambda x: np.array(json.loads(x)) if isinstance(x, str) else x
        )

    common_seq_ids = list(
        set(naturalness_df.seq_id) & set(embedding_df.seq_id) & set(activity_df.seq_id)
    )

    return (
        naturalness_df.loc[common_seq_ids],
        embedding_df.loc[common_seq_ids],
        activity_df.loc[common_seq_ids],
    )

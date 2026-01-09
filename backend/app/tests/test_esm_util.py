from typing import Optional
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from werkzeug.exceptions import BadRequest

from app.helpers.esm_util import (
    get_naturalness,
    normalize_msa_a3m_contents,
    validate_msa_a3m_contents,
)


@pytest.fixture
def mock_esm_setup():
    """Setup common mocks for ESM-related tests"""
    with (
        patch("torch.device") as mock_device,
        patch("esm.models.esmc.ESMC") as mock_ESMC,
        patch("esm.models.esm3.ESM3") as mock_ESM3,
    ):

        # Mock device setup
        mock_device.return_value = "cpu"

        # Create mock client
        mock_client = Mock()
        mock_ESMC.from_pretrained.return_value.to.return_value = mock_client

        mock_esm3_client = Mock()
        mock_ESM3.from_pretrained.return_value.to.return_value = mock_esm3_client

        # Create mock logits output
        # Shape: [batch=1, sequence_length=7, vocab_size=33]
        mock_logits = torch.randn(1, 7, 33)  # 5 amino acids + 2 special tokens
        mock_logits_output = Mock()
        mock_logits_output.logits.sequence = mock_logits.clone()

        # Setup client encode and logits methods
        mock_client.encode.return_value = Mock()
        mock_client.logits.return_value = mock_logits_output

        # Setup client encode and logits methods
        mock_esm3_client.encode.return_value = Mock()
        mock_esm3_client.logits.return_value = mock_logits_output

        yield {
            "device": mock_device,
            "ESMC": mock_ESMC,
            "ESM3": mock_ESM3,
            "client": mock_client,
            "logits": mock_logits,
        }


def test_get_naturalness_basic(mock_esm_setup):
    # Test sequence
    wt_aa_seq = "ABCDE"  # 5 amino acids

    # Call function
    logits_json, melted_df = get_naturalness(wt_aa_seq, "esmc_mock_model")

    # Verify basic calls
    mock_esm_setup["ESMC"].from_pretrained.assert_called_once_with("esmc_mock_model")

    # Verify shape of output DataFrame
    assert isinstance(melted_df, pd.DataFrame)
    assert set(melted_df.columns) == {
        "seq_id",
        "probability",
        "locus",
        "wt_probability",
        "wt_marginal",
    }

    # Each position should have entries for all possible amino acids
    expected_positions = len(wt_aa_seq)
    expected_rows = expected_positions * 33  # 33 amino acids in vocab per position
    assert len(melted_df) == expected_rows


def test_get_naturalness_wt_marginal_calculation(mock_esm_setup):
    wt_aa_seq = "ABC"

    # Create predictable logits for easier testing
    # Shape: [batch=1, sequence_length=5, vocab_size=33]
    mock_logits = torch.zeros(1, 5, 33)  # 3 amino acids + 2 special tokens
    # Set some known probabilities after softmax
    mock_logits[0, 1:4, 0] = 0  # probability for first amino acid in vocab
    mock_logits_output = Mock()
    mock_logits_output.logits.sequence = mock_logits
    mock_esm_setup["client"].logits.return_value = mock_logits_output

    # Call function
    _, melted_df = get_naturalness(wt_aa_seq, "esmc_mock_model")

    # Verify wt_marginal calculations
    # Filter for a specific position
    pos1_data = melted_df[melted_df.locus == 1]
    assert all(pos1_data.wt_marginal.notna())  # All wt_marginal values should be calculated


def test_get_naturalness_error_handling(mock_esm_setup):
    # Test with mismatched sequence length
    wt_aa_seq = "TOOLONG"  # 7 amino acids, but mock returns logits for 5

    with pytest.raises(IndexError, match="index 7 is out of bounds"):
        get_naturalness(wt_aa_seq, "esmc_mock_model")


def test_add_pdb_file_path_fails_for_esmc(mock_esm_setup):
    # Test with mismatched sequence length
    wt_aa_seq = "ABCDE"
    pdb_file_path = "app/tests/testdata/rubisco-boltz.pdb"

    with pytest.raises(ValueError, match="does not support PDB"):
        get_naturalness(wt_aa_seq, "esmc_mock_model", cif_file_path=pdb_file_path)


def test_add_pdb_file_path_works_for_esm3(mock_esm_setup):
    # Test with mismatched sequence length
    wt_aa_seq = "ABCDE"
    pdb_file_path = "app/tests/testdata/rubisco-boltz.pdb"

    logits_json, melted_df = get_naturalness(
        wt_aa_seq, "esm3_mock_model", cif_file_path=pdb_file_path
    )
    assert logits_json is not None
    assert melted_df is not None


def test_get_naturalness_depth_two_aggregates_base_sequences():
    mock_client = Mock()
    mock_client.get_logits.side_effect = [
        pd.DataFrame({"seq_id": ["A1A"], "probability": [0.1]}),
        pd.DataFrame({"seq_id": ["B1B"], "probability": [0.2]}),
    ]

    with (
        patch(
            "app.helpers.esm_util.get_seq_ids_for_deep_mutational_scan",
            return_value=["A1C", "B2D"],
        ),
        patch("app.helpers.esm_util.seq_id_to_seq", side_effect=["AC", "BD"]),
        patch("app.helpers.esm_client.FoldyPLMClient.get_client", return_value=mock_client),
    ):
        logits_json, melted_df = get_naturalness("AB", "esmc_mock_model", get_depth_two_logits=True)

    assert logits_json == ""
    assert "base_seq_id" in melted_df.columns
    assert set(melted_df["base_seq_id"]) == {"A1C", "B2D"}
    assert mock_client.get_logits.call_count == 2


def test_get_naturalness_passes_msa_context_args():
    mock_client = Mock()
    mock_client.get_logits.return_value = pd.DataFrame({"seq_id": ["A1A"], "probability": [0.9]})

    with patch("app.helpers.esm_client.FoldyPLMClient.get_client", return_value=mock_client):
        get_naturalness("A", "esmc_mock_model", use_msa_context=True, msa_a3m_path="msa.a3m")

    mock_client.get_logits.assert_called_once_with(
        "A", None, use_msa_context=True, msa_a3m_path="msa.a3m"
    )


def test_validate_msa_a3m_contents_accepts_matching_query():
    msa_a3m = ">query\nACD.\n>other\nACD\n"
    cleaned = validate_msa_a3m_contents(msa_a3m, "ACD")
    assert cleaned == "ACD"


def test_validate_msa_a3m_contents_rejects_mismatch():
    msa_a3m = ">query\nACE\n"
    with pytest.raises(Exception, match="does not match"):
        validate_msa_a3m_contents(msa_a3m, "ACD")


def test_validate_msa_a3m_contents_requires_header():
    with pytest.raises(Exception, match="FASTA"):
        validate_msa_a3m_contents("ACD", "ACD")


def test_validate_msa_a3m_contents_requires_query_sequence():
    msa_a3m = ">query\n>other\nACD\n"
    with pytest.raises(BadRequest, match="query sequence after the header"):
        validate_msa_a3m_contents(msa_a3m, "ACD")


def test_validate_msa_a3m_contents_rejects_empty_after_cleaning():
    msa_a3m = ">query\nacde.-\n"
    with pytest.raises(BadRequest, match="query sequence is empty"):
        validate_msa_a3m_contents(msa_a3m, "ACD")


def test_normalize_msa_a3m_contents_converts_boltz_csv():
    msa_csv = "key,sequence\n-1,ACD\n-1,ACd-\n"
    normalized = normalize_msa_a3m_contents(msa_csv, "ACD")
    assert normalized.startswith(">query\n")
    cleaned = validate_msa_a3m_contents(normalized, "ACD")
    assert cleaned == "ACD"


def test_normalize_msa_a3m_contents_rejects_boltz_csv_without_query():
    msa_csv = "key,sequence\n-1,ACE\n-1,ACd-\n"
    with pytest.raises(BadRequest, match="does not contain the WT query sequence"):
        normalize_msa_a3m_contents(msa_csv, "ACD")

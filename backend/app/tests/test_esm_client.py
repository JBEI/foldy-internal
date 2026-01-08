import json
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from app.helpers.esm_client import (
    FoldyE1Client,
    FoldyESM1and2Client,
    FoldyESM3Client,
    FoldyESMCClient,
    FoldyPLMClient,
    recursive_cleanup,
)

# Test sequences
TEST_SEQUENCE = "MGSSHHHHHHSSGLVPRGSHM"
TEST_PDB_PATH = "app/tests/testdata/rubisco-boltz.pdb"
ESM3_VOCAB_SIZE = 64
E1_VOCAB_SIZE = 32  # E1 uses a smaller vocabulary


@pytest.fixture
def mock_torch_device():
    with patch("torch.device") as mock_device, patch("torch.cuda.is_available", return_value=False):
        mock_device.return_value = "cpu"
        yield mock_device


@pytest.fixture
def mock_e1_client():
    """Mock fixture for Profluent E1 model."""
    with (
        patch("E1.modeling.E1ForMaskedLM") as MockE1Model,
        patch("E1.batch_preparer.E1BatchPreparer") as MockBatchPreparer,
    ):

        # Create mock model
        mock_model = Mock()
        MockE1Model.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        # Create mock batch preparer
        mock_batch_preparer = Mock()
        MockBatchPreparer.return_value = mock_batch_preparer

        # Mock get_batch_kwargs
        mock_batch = {
            "input_ids": torch.zeros((1, len(TEST_SEQUENCE) + 2), dtype=torch.long),
            "within_seq_position_ids": torch.zeros((1, len(TEST_SEQUENCE) + 2), dtype=torch.long),
            "global_position_ids": torch.zeros((1, len(TEST_SEQUENCE) + 2), dtype=torch.long),
            "sequence_ids": torch.zeros((1, len(TEST_SEQUENCE) + 2), dtype=torch.long),
        }
        mock_batch_preparer.get_batch_kwargs.return_value = mock_batch

        # Mock get_boundary_token_mask - returns True for first and last tokens (boundary)
        boundary_mask = torch.zeros((1, len(TEST_SEQUENCE) + 2), dtype=torch.bool)
        boundary_mask[0, 0] = True  # First token is boundary
        boundary_mask[0, -1] = True  # Last token is boundary
        mock_batch_preparer.get_boundary_token_mask.return_value = boundary_mask

        # Mock tokenizer vocab
        mock_tokenizer = Mock()
        vocab = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        mock_tokenizer.get_vocab.return_value = vocab
        mock_batch_preparer.tokenizer = mock_tokenizer

        # Mock model forward pass
        mock_embeddings = torch.randn(1, len(TEST_SEQUENCE) + 2, 1280)
        mock_logits = torch.randn(1, len(TEST_SEQUENCE) + 2, E1_VOCAB_SIZE)
        mock_output = Mock(embeddings=mock_embeddings, logits=mock_logits)
        mock_model.return_value = mock_output

        yield mock_model, mock_batch_preparer


@pytest.fixture
def mock_esmc_client():
    with patch("esm.models.esmc.ESMC") as MockESMC:

        # Create mock client
        mock_client = Mock()
        MockESMC.from_pretrained.return_value = mock_client
        mock_client.to.return_value = mock_client

        # Mock encode method
        mock_client.encode.return_value = Mock()

        # Mock logits method with embeddings
        mock_embeddings = torch.randn(1, len(TEST_SEQUENCE), 1280)  # Example embedding size
        mock_logits_output = Mock(embeddings=mock_embeddings)
        mock_client.logits.return_value = mock_logits_output

        yield mock_client


@pytest.fixture
def mock_esm3_client():
    with patch("esm.models.esm3.ESM3") as MockESM3:

        # Create mock client
        mock_client = Mock()
        MockESM3.from_pretrained.return_value = mock_client
        mock_client.to.return_value = mock_client

        # Mock encode method
        mock_client.encode.return_value = Mock()

        # Mock logits method with embeddings
        mock_embeddings = torch.randn(1, len(TEST_SEQUENCE), 1280)  # Example embedding size
        mock_logits_output = Mock(
            embeddings=mock_embeddings, hidden_states=torch.randn(33, len(TEST_SEQUENCE), 1280)
        )
        mock_client.logits.return_value = mock_logits_output

        yield mock_client


@pytest.fixture
def mock_esm2_hub():
    with patch("torch.hub.load") as mock_hub:
        # Create mock model and alphabet
        mock_model = Mock()
        mock_alphabet = Mock()
        mock_converter = Mock()

        # Setup returns
        mock_hub.return_value = (mock_model, mock_alphabet)
        mock_alphabet.get_batch_converter.return_value = mock_converter
        mock_converter.return_value = (
            None,
            None,
            torch.zeros((1, len(TEST_SEQUENCE) + 2)),
        )

        # Mock model forward pass
        mock_representations = {33: torch.randn(1, len(TEST_SEQUENCE) + 2, 1280)}
        mock_logits = torch.randn(1, len(TEST_SEQUENCE) + 2, 20)  # 20 amino acids
        mock_model.return_value = {
            "representations": mock_representations,
            "logits": mock_logits,
        }

        # Setup alphabet tokens
        mock_alphabet.all_toks = list("ACDEFGHIKLMNPQRSTVWY")
        mock_alphabet.standard_toks = list("ACDEFGHIKLMNPQRSTVWY")

        yield mock_model, mock_alphabet


def test_get_client_invalid():
    with pytest.raises(ValueError):
        FoldyPLMClient.get_client("invalid_model")


def test_recursive_cleanup_removes_tensors():
    data = {
        "tensor": torch.tensor([1.0]),
        "nested": {"tensor": torch.tensor([2.0]), "keep": "value"},
        "items": [torch.tensor([3.0]), "ok"],
    }

    cleaned = recursive_cleanup(data)

    assert "tensor" not in cleaned
    assert "tensor" not in cleaned["nested"]
    assert cleaned["nested"]["keep"] == "value"
    assert cleaned["items"] == ["ok"]


def test_esmc_embed(mock_torch_device, mock_esmc_client):
    client = FoldyPLMClient.get_client("esmc_t36_3B_UR50D")
    embedding = client.embed(TEST_SEQUENCE)

    assert isinstance(embedding, list)
    assert len(embedding) == 1
    assert len(embedding[0]) == 1280  # Expected embedding dimension


def test_esmc_embed_with_pdb_fails(mock_torch_device, mock_esmc_client):
    client = FoldyPLMClient.get_client("esmc_t36_3B_UR50D")
    with pytest.raises(ValueError, match="ESM-C does not support PDB-based embeddings"):
        client.embed(TEST_SEQUENCE, TEST_PDB_PATH)


def test_esmc_embed_with_complex_and_domain_boundaries(mock_torch_device, mock_esmc_client):
    client = FoldyPLMClient.get_client("esmc_t36_3B_UR50D")
    client._get_esm_protein_tensor_for_complex = Mock(return_value=Mock())

    embedding = client.embed([("A", "ACD"), ("B", "EFG")], domain_boundaries=[2])

    client._get_esm_protein_tensor_for_complex.assert_called_once()
    assert len(embedding) == 1
    assert len(embedding[0]) == 1280 * 2


def test_esm3_embed_with_pdb_succeeds(mock_torch_device, mock_esm3_client):
    client = FoldyPLMClient.get_client("esm3_t36_3B_UR50D")
    embedding = client.embed(TEST_SEQUENCE, TEST_PDB_PATH)

    assert isinstance(embedding, list)
    assert len(embedding) == 1
    assert len(embedding[0]) == 1280


def test_esm2_embed_with_pdb_fails(mock_torch_device, mock_esm2_hub):
    client = FoldyPLMClient.get_client("esm2_t33_650M_UR50D")
    with pytest.raises(ValueError, match="do not support PDB-based embeddings"):
        client.embed(TEST_SEQUENCE, TEST_PDB_PATH)


def test_esmc_get_logits(mock_torch_device, mock_esmc_client):
    # Mock the logits output
    sequence_logits = torch.randn(
        1, len(TEST_SEQUENCE) + 2, ESM3_VOCAB_SIZE
    )  # batch, seq_len + special tokens, vocab_size
    mock_esmc_client.logits.return_value = Mock(logits=Mock(sequence=sequence_logits))

    client = FoldyPLMClient.get_client("esmc_t36_3B_UR50D")
    df = client.get_logits(TEST_SEQUENCE)

    assert isinstance(df, pd.DataFrame)
    assert "seq_id" in df.columns
    assert "probability" in df.columns
    assert len(df) > 0


def test_esmc_get_logits_with_complex_input(mock_torch_device, mock_esmc_client):
    sequence_logits = torch.randn(1, 4, 2)
    mock_esmc_client.logits.return_value = Mock(logits=Mock(sequence=sequence_logits))

    client = FoldyPLMClient.get_client("esmc_t36_3B_UR50D")
    client._get_esm_protein_tensor_for_complex = Mock(return_value=Mock())

    with patch("esm.utils.constants.esm3.SEQUENCE_VOCAB", ["A", "B"]):
        df = client.get_logits([("A", "ACD"), ("B", "EFG")])

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"seq_id", "probability"}
    assert len(df) == 4 * 2
    assert "0A" in df["seq_id"].values


def test_esm2_embed(mock_torch_device, mock_esm2_hub):
    client = FoldyPLMClient.get_client("esm2_t33_650M_UR50D")
    embedding = client.embed(TEST_SEQUENCE)

    assert isinstance(embedding, list)
    assert len(embedding) == 1
    assert len(embedding[0]) == 1280


def test_esm2_embed_with_pdb(mock_torch_device, mock_esm2_hub):
    client = FoldyPLMClient.get_client("esm2_t33_650M_UR50D")
    with pytest.raises(ValueError):
        client.embed(TEST_SEQUENCE, TEST_PDB_PATH)


def test_esm2_get_logits(mock_torch_device, mock_esm2_hub):
    client = FoldyPLMClient.get_client("esm2_t33_650M_UR50D")
    df = client.get_logits(TEST_SEQUENCE)

    assert isinstance(df, pd.DataFrame)
    assert "seq_id" in df.columns
    assert "probability" in df.columns
    assert len(df) > 0


def test_esm2_get_logits_with_pdb(mock_torch_device, mock_esm2_hub):
    client = FoldyPLMClient.get_client("esm2_t33_650M_UR50D")
    with pytest.raises(ValueError):
        client.get_logits(TEST_SEQUENCE, TEST_PDB_PATH)


def test_esm3_embed_with_extra_layers(mock_torch_device, mock_esm3_client):
    client = FoldyPLMClient.get_client("esm3_t36_3B_UR50D")
    embedding = client.embed(TEST_SEQUENCE, extra_layers=[1, 2, 3])

    assert isinstance(embedding, list)
    assert len(embedding) == 4
    assert len(embedding[0]) == 1280
    assert len(embedding[1]) == 1280
    assert len(embedding[2]) == 1280
    assert len(embedding[3]) == 1280

    assert type(json.dumps(embedding[0])) == str
    assert type(json.dumps(embedding[1])) == str
    assert type(json.dumps(embedding[2])) == str
    assert type(json.dumps(embedding[3])) == str


# Helper function to verify DataFrame structure
def verify_logits_df_structure(df: pd.DataFrame, sequence: str):
    assert all(col in df.columns for col in ["seq_id", "probability"])
    assert len(df) > 0
    # Verify seq_id format (e.g., "M1A" for mutation of M at position 1 to A)
    assert all(len(seq_id) >= 3 for seq_id in df["seq_id"])
    assert all(0 <= prob <= 1 for prob in df["probability"])


# ============== Profluent E1 Tests ==============


def test_get_client_e1_routing(mock_torch_device, mock_e1_client):
    """Test that E1 model names are routed to FoldyE1Client."""
    client = FoldyPLMClient.get_client("e1_300m")
    assert isinstance(client, FoldyE1Client)


def test_get_client_e1_all_sizes(mock_torch_device, mock_e1_client):
    """Test that all E1 model sizes can be instantiated."""
    for model_name in ["e1_150m", "e1_300m", "e1_600m"]:
        client = FoldyPLMClient.get_client(model_name)
        assert isinstance(client, FoldyE1Client)


def test_e1_embed(mock_torch_device, mock_e1_client):
    """Test E1 embedding generation."""
    client = FoldyPLMClient.get_client("e1_300m")
    embedding = client.embed(TEST_SEQUENCE)

    assert isinstance(embedding, list)
    assert len(embedding) == 1
    assert len(embedding[0]) == 1280  # Expected embedding dimension


def test_e1_embed_with_pdb_fails(mock_torch_device, mock_e1_client):
    """Test that E1 raises error when CIF/PDB file is provided."""
    client = FoldyPLMClient.get_client("e1_300m")
    with pytest.raises(ValueError, match="E1 does not support CIF or PDB-based embeddings"):
        client.embed(TEST_SEQUENCE, TEST_PDB_PATH)


def test_e1_embed_with_complex_fails(mock_torch_device, mock_e1_client):
    """Test that E1 raises error for protein complexes."""
    client = FoldyPLMClient.get_client("e1_300m")
    complex_input = [("A", "MGSSH"), ("B", "HHHHH")]
    with pytest.raises(ValueError, match="E1 does not support protein complexes"):
        client.embed(complex_input)


def test_e1_embed_with_extra_layers_fails(mock_torch_device, mock_e1_client):
    """Test that E1 raises error when extra_layers is requested."""
    client = FoldyPLMClient.get_client("e1_300m")
    with pytest.raises(ValueError, match="E1 does not support extra layers"):
        client.embed(TEST_SEQUENCE, extra_layers=[1, 2, 3])


def test_e1_embed_with_msa_context_requires_path(mock_torch_device, mock_e1_client):
    client = FoldyPLMClient.get_client("e1_300m")
    with pytest.raises(ValueError, match="msa_a3m_path is required"):
        client.embed(TEST_SEQUENCE, use_msa_context=True)


def test_e1_get_logits(mock_torch_device, mock_e1_client):
    """Test E1 logits generation."""
    client = FoldyPLMClient.get_client("e1_300m")
    df = client.get_logits(TEST_SEQUENCE)

    assert isinstance(df, pd.DataFrame)
    assert "seq_id" in df.columns
    assert "probability" in df.columns
    assert len(df) > 0


def test_e1_get_logits_with_pdb_fails(mock_torch_device, mock_e1_client):
    """Test that E1 raises error for CIF/PDB-based logits."""
    client = FoldyPLMClient.get_client("e1_300m")
    with pytest.raises(ValueError, match="E1 does not support CIF or PDB-based logits"):
        client.get_logits(TEST_SEQUENCE, TEST_PDB_PATH)


def test_e1_get_logits_with_msa_context_requires_path(mock_torch_device, mock_e1_client):
    client = FoldyPLMClient.get_client("e1_300m")
    with pytest.raises(ValueError, match="msa_a3m_path is required"):
        client.get_logits(TEST_SEQUENCE, use_msa_context=True)


def test_e1_get_logits_with_complex_fails(mock_torch_device, mock_e1_client):
    """Test that E1 raises error for complex logits."""
    client = FoldyPLMClient.get_client("e1_300m")
    complex_input = [("A", "MGSSH"), ("B", "HHHHH")]
    with pytest.raises(ValueError, match="E1 does not support protein complexes"):
        client.get_logits(complex_input)


def test_e1_get_logits_with_msa_context_filters_to_query(mock_torch_device, mock_e1_client):
    """Test that E1 MSA context returns logits for the query sequence only."""
    mock_model, mock_batch_preparer = mock_e1_client
    client = FoldyPLMClient.get_client("e1_300m")
    client._get_msa_context = Mock(return_value="AAA")

    query_sequence = "ACDE"
    seq_len = 9
    mock_batch_preparer.get_batch_kwargs.return_value = {
        "input_ids": torch.zeros((1, seq_len), dtype=torch.long),
        "within_seq_position_ids": torch.zeros((1, seq_len), dtype=torch.long),
        "global_position_ids": torch.zeros((1, seq_len), dtype=torch.long),
        "sequence_ids": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
    }
    boundary_mask = torch.zeros((1, seq_len), dtype=torch.bool)
    boundary_mask[0, 0] = True
    boundary_mask[0, 2] = True
    boundary_mask[0, 3] = True
    boundary_mask[0, 8] = True
    mock_batch_preparer.get_boundary_token_mask.return_value = boundary_mask

    mock_embeddings = torch.randn(1, seq_len, 1280)
    mock_logits = torch.randn(1, seq_len, E1_VOCAB_SIZE)
    mock_model.return_value = Mock(embeddings=mock_embeddings, logits=mock_logits)

    df = client.get_logits(query_sequence, use_msa_context=True, msa_a3m_path="dummy.a3m")
    assert len(df) == len(query_sequence) * len(client.STANDARD_AAS)


def test_e1_embed_with_domain_boundaries(mock_torch_device, mock_e1_client):
    """Test E1 embedding with domain pooling."""
    client = FoldyPLMClient.get_client("e1_300m")
    # Domain boundary at position 10 (splits sequence into two domains)
    embedding = client.embed(TEST_SEQUENCE, domain_boundaries=[10])

    assert isinstance(embedding, list)
    assert len(embedding) == 1
    # With domain pooling, embedding should be concatenation of 2 domain embeddings
    assert len(embedding[0]) == 1280 * 2


def test_non_e1_embed_with_msa_context_fails(mock_torch_device, mock_esm2_hub):
    client = FoldyPLMClient.get_client("esm2_t33_650M_UR50D")
    with pytest.raises(ValueError, match="MSA context is only supported for E1 models"):
        client.embed(TEST_SEQUENCE, use_msa_context=True, msa_a3m_path="dummy.a3m")


def test_non_e1_get_logits_with_msa_context_fails(mock_torch_device, mock_esm2_hub):
    client = FoldyPLMClient.get_client("esm2_t33_650M_UR50D")
    with pytest.raises(ValueError, match="MSA context is only supported for E1 models"):
        client.get_logits(TEST_SEQUENCE, use_msa_context=True, msa_a3m_path="dummy.a3m")

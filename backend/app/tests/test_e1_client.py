import pytest
import torch
from unittest.mock import patch, Mock

from app.helpers.e1_client import FoldyE1Client, FoldyESMClient

TEST_SEQUENCE = "ACDEFG"

@pytest.fixture
def mock_device():
    with patch("app.helpers.e1_client.get_device", return_value="cpu"):
        with patch("app.helpers.e1_client.torch") as mock_torch:
            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_model.hidden_states = [Mock()]
            mock_model.hidden_states[0].shape = (1, len(TEST_SEQUENCE) + 2, 768)
            yield mock_model

@pytest.mark.skipif(torch.cuda.is_available(), reason="GPU segfault - use CPU-only")
def test_get_client_e1_300m_returns_foldy_e1_client(mock_device):
    """Verify FoldyESMClient.get_client('e1-300m') returns FoldyE1Client"""
    client = FoldyESMClient.get_client('e1-300m')
    assert isinstance(client, FoldyE1Client)

@pytest.mark.skipif(torch.cuda.is_available(), reason="GPU segfault - use CPU-only")
def test_e1_client_embed_returns_valid_embeddings(mock_device):
    """Verify client.embed returns valid embeddings (mocked model load)"""
    with patch("app.helpers.e1_client.AutoTokenizer.from_pretrained") as mock_tokenizer:
        with patch("app.helpers.e1_client.E1ForMaskedLM.from_pretrained") as mock_model:
            mock_model.return_value = mock_device
            client = FoldyE1Client('e1-300m')
            embedding = client.embed(TEST_SEQUENCE)
            
            assert isinstance(embedding, list)
            assert len(embedding) == 1
            assert len(embedding[0]) > 0  # Valid embedding dimension
            assert all(isinstance(x, float) for x in embedding[0])

@pytest.mark.skipif(torch.cuda.is_available(), reason="GPU segfault - use CPU-only")
def test_e1_client_model_loads_without_errors(mock_device):
    """Verify E1 model instantiation succeeds without errors"""
    with patch("app.helpers.e1_client.AutoTokenizer.from_pretrained"):
        with patch("app.helpers.e1_client.E1ForMaskedLM.from_pretrained") as mock_model:
            mock_model.return_value = mock_device
            client = FoldyE1Client('e1-300m')
            assert client.model is not None
            assert hasattr(client, 'tokenizer')
import json
import logging
import random
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd

# Type definitions for complex inputs
SequenceType = str
ComplexType = List[Tuple[str, str]]
SequenceOrComplexType = Union[SequenceType, ComplexType]


# Import for type checking only
if TYPE_CHECKING:
    import torch
    from esm.sdk.api import ESMProtein


def recursive_cleanup(obj):
    """Recursively detach and delete tensors in a complex nested object."""
    import torch

    if isinstance(obj, torch.Tensor):
        # Detach and move to CPU before deletion
        if obj.is_cuda:
            obj = obj.detach().cpu()
        return None  # Return None to indicate this should be deleted

    elif hasattr(obj, "__dict__"):
        # For objects with attributes
        for attr_name, attr_value in list(obj.__dict__.items()):
            result = recursive_cleanup(attr_value)
            if result is None:
                delattr(obj, attr_name)
            else:
                setattr(obj, attr_name, result)

    elif isinstance(obj, dict):
        # For dictionaries
        for key in list(obj.keys()):
            result = recursive_cleanup(obj[key])
            if result is None:
                del obj[key]
            else:
                obj[key] = result

    elif isinstance(obj, (list, tuple)):
        # For lists/tuples
        cleaned_items = []
        for item in obj:
            cleaned = recursive_cleanup(item)
            if cleaned is not None:
                cleaned_items.append(cleaned)
        new_obj = type(obj)(cleaned_items)
        return new_obj if new_obj else None

    return obj  # Return object with cleaned up components


class FoldyPLMClient(ABC):
    """
    Interface for Protein Language Model clients that provide embedding and logit functionality.

    This abstract base class defines the interface that all PLM client
    implementations must follow. Supports ESM, ESM-C, ESM-3, Profluent E1, and other models.
    """

    @classmethod
    def get_client(cls, model_name: str) -> "FoldyPLMClient":
        """
        Factory method to create appropriate PLM client based on model name.

        Args:
            model_name: Name of the protein language model to use

        Returns:
            An instance of the appropriate FoldyPLMClient subclass

        Raises:
            ValueError: If model_name does not match any known model type
        """
        if model_name.startswith("e1_"):
            return FoldyE1Client(model_name)
        elif model_name.startswith("esmc"):
            return FoldyESMCClient(model_name)
        elif model_name.startswith("esm3"):
            return FoldyESM3Client(model_name)
        elif model_name.startswith("esm1") or model_name.startswith("esm2"):
            return FoldyESM1and2Client(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    @abstractmethod
    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Get embedding for a protein sequence or complex.

        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            cif_file_path: Optional path to a CIF file for structure-aware models
            extra_layers: List of layer indices to return embeddings for
            domain_boundaries: List of domain boundary positions for domain pooling
            use_msa_context: Whether to prepend E1 MSA context (E1-only)
            msa_a3m_path: Path to a local .a3m file to sample context from

        Returns:
            A list of list of floats representing embedding vectors for extra_layers, and the final layer.
        """
        pass

    @abstractmethod
    def get_logits(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence or complex.

        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            cif_file_path: Optional path to a CIF file for structure-aware models
            use_msa_context: Whether to prepend E1 MSA context (E1-only)
            msa_a3m_path: Path to a local .a3m file to sample context from

        Returns:
            A pandas DataFrame with sequence logits in melted format with
            columns 'seq_id' and 'probability'
        """
        pass

    def supports_batch_embedding(self) -> bool:
        """Return True if the client can embed multiple sequences in one forward pass."""
        return False

    def embed_batch(
        self,
        sequences: List[str],
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[List[float]]]:
        """Embed a list of sequences, defaulting to sequential single-embed calls."""
        return [
            self.embed(
                sequence,
                cif_file_path=cif_file_path,
                extra_layers=extra_layers,
                domain_boundaries=domain_boundaries,
                use_msa_context=use_msa_context,
                msa_a3m_path=msa_a3m_path,
            )
            for sequence in sequences
        ]


class FoldyESMCClient(FoldyPLMClient):
    """
    ESM-C model client implementation for the foldy platform.

    Handles ESM-C specific operations including protein tensor creation,
    embedding extraction, and logit computation.
    """

    def _pool_by_domains(
        self, hidden_states: "torch.Tensor", domain_boundaries: List[int]
    ) -> "torch.Tensor":
        """
        Pool hidden states by domains and concatenate the results.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim]
            domain_boundaries: List of domain boundary positions (0-indexed)

        Returns:
            Concatenated tensor of domain embeddings
        """
        import torch

        if not domain_boundaries:
            return hidden_states.mean(dim=-2)

        # Convert to sorted list and add 0 at the start and seq_len at the end
        boundaries = [0] + sorted(domain_boundaries) + [hidden_states.shape[1]]
        domain_embeddings = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            domain_embedding = hidden_states[:, start_idx:end_idx, :].mean(dim=1)
            domain_embeddings.append(domain_embedding)

        return torch.cat(domain_embeddings, dim=-1)

    def __init__(self, model_name: str) -> None:
        """
        Initialize the ESM-C client with the specified model.

        Args:
            model_name: Name of the ESM-C model to load
        """
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ESMC.from_pretrained(model_name).to(device)

        # --- Add this block: ensure dtype compatibility ---
        if device.type == "cuda":
            major, _ = torch.cuda.get_device_capability()
            if major < 8:  # Turing (RTX 20xx) or older → no bfloat16 support
                model = model.to(torch.float16)
        # ---------------------------------------------------

        print(f"[ESMC] Loaded {model_name} on {device} with dtype {next(model.parameters()).dtype}")
        self.client = model
        self.device = device

    def supports_batch_embedding(self) -> bool:
        return True

    def _get_esm_protein_tensor_for_sequence(
        self, sequence: str, cif_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a sequence.

        Args:
            sequence: Protein sequence string
            cif_file_path: Not supported for ESM-C

        Returns:
            Tensor representation of the protein

        Raises:
            ValueError: If cif_file_path is provided (not supported)
        """
        from esm.sdk.api import ESMProtein, LogitsConfig

        if cif_file_path:
            raise ValueError("ESM-C does not support CIF or PDB-based embeddings")
        protein = ESMProtein(sequence=sequence)

        return self.client.encode(protein)

    def _get_esm_protein_tensor_for_complex(
        self, complex_input: ComplexType, cif_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a protein complex.

        Args:
            complex_input: List of (chain_id, sequence) tuples
            cif_file_path: Not supported for ESM-C

        Returns:
            Tensor representation of the protein complex

        Raises:
            ValueError: If cif_file_path is provided (not supported)
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_chain import ProteinChain
        from esm.utils.structure.protein_complex import ProteinComplex

        if cif_file_path:
            raise ValueError("ESM-C does not support CIF or PDB-based embeddings")

        chains = [
            ProteinChain(chain_id=chain_id, sequence=sequence)
            for chain_id, sequence in complex_input
        ]
        protein_complex = ProteinComplex.from_chains(chains)
        protein = ESMProtein.from_protein_complex(protein_complex)

        return self.client.encode(protein)

    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Get embedding for a protein sequence or complex.

        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            cif_file_path: Optional path to a CIF file (not supported for ESM-C)
            extra_layers: List of layer indices to return embeddings for
            domain_boundaries: List of domain boundary positions for domain pooling

        Returns:
            A list of list of floats representing embedding vectors for extra_layers, and the final layer.
        """
        import gc

        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        if use_msa_context:
            raise ValueError("MSA context is only supported for E1 models")

        # Force CUDA synchronization before we start
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        embeddings = []
        protein_tensor = None
        logits_output = None

        try:
            if isinstance(sequence_or_complex, list):
                protein_tensor = self._get_esm_protein_tensor_for_complex(
                    sequence_or_complex, cif_file_path
                )
            else:
                protein_tensor = self._get_esm_protein_tensor_for_sequence(
                    sequence_or_complex, cif_file_path
                )

            with torch.no_grad():
                logits_output = self.client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=False,
                        return_embeddings=True,
                        return_hidden_states=True if len(extra_layers) > 0 else False,
                    ),
                )
            if extra_layers:
                hidden_states = logits_output.hidden_states
                for extra_layer_idx in extra_layers:
                    layer_embedding = self._pool_by_domains(
                        hidden_states[extra_layer_idx], domain_boundaries
                    ).squeeze()
                    embeddings.append(layer_embedding.cpu().tolist())
                del hidden_states

            final_embedding = self._pool_by_domains(
                logits_output.embeddings.detach().cpu(), domain_boundaries
            ).squeeze(0)
            embeddings.append(final_embedding.tolist())

        finally:
            # First recursively clean complex objects
            if "logits_output" in locals() and logits_output is not None:
                recursive_cleanup(logits_output)

            if "protein_tensor" in locals() and protein_tensor is not None:
                recursive_cleanup(protein_tensor)
            # Expanded cleanup
            for local_var in [
                "logits_output",
                "protein_tensor",
                "all_embeddings",
                "final_embedding",
            ]:
                if local_var in locals() and locals()[local_var] is not None:
                    del locals()[local_var]

        return embeddings

    def embed_batch(
        self,
        sequences: List[str],
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[List[float]]]:
        """
        Get embeddings for a batch of protein sequences.

        Returns:
            A list of embedding lists (one per input sequence).
        """
        import torch
        from esm.sdk.api import LogitsConfig
        from esm.utils.sampling import _BatchedESMProteinTensor

        if use_msa_context:
            raise ValueError("MSA context is only supported for E1 models")
        if cif_file_path:
            raise ValueError("ESM-C does not support CIF or PDB-based embeddings")
        if not sequences:
            return []

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_tokens = None
        logits_output = None
        protein_tensor = None
        embeddings: List[List[List[float]]] = []

        try:
            batch_tokens = self.client._tokenize(sequences)
            pad_token_id = self.client.tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("ESM-C tokenizer is missing pad_token_id")
            batch_lens = (batch_tokens != pad_token_id).sum(dim=1)

            protein_tensor = _BatchedESMProteinTensor(sequence=batch_tokens)
            with torch.no_grad():
                logits_output = self.client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=False,
                        return_embeddings=True,
                        return_hidden_states=True if len(extra_layers) > 0 else False,
                    ),
                )

            hidden_states = logits_output.hidden_states if extra_layers else None
            for idx, seq_len in enumerate(batch_lens):
                seq_len_int = int(seq_len.item())
                seq_embeddings: List[List[float]] = []
                if extra_layers and hidden_states is not None:
                    for extra_layer_idx in extra_layers:
                        layer_hidden = hidden_states[extra_layer_idx][
                            idx : idx + 1, :seq_len_int, :
                        ]
                        layer_embedding = self._pool_by_domains(
                            layer_hidden, domain_boundaries
                        ).squeeze(0)
                        seq_embeddings.append(layer_embedding.detach().cpu().tolist())

                final_hidden = logits_output.embeddings[idx : idx + 1, :seq_len_int, :]
                final_embedding = self._pool_by_domains(final_hidden, domain_boundaries).squeeze(0)
                seq_embeddings.append(final_embedding.detach().cpu().tolist())
                embeddings.append(seq_embeddings)

        finally:
            if "logits_output" in locals() and logits_output is not None:
                recursive_cleanup(logits_output)

            if "protein_tensor" in locals() and protein_tensor is not None:
                recursive_cleanup(protein_tensor)

            for local_var in ["logits_output", "protein_tensor", "batch_tokens"]:
                if local_var in locals() and locals()[local_var] is not None:
                    del locals()[local_var]

        return embeddings

    def get_logits(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence or complex.

        Args:
            sequence_or_complex: Either a protein sequence string or a list of
                                (chain_id, sequence) tuples for complexes
            cif_file_path: Optional path to a CIF file (not supported for ESM-C)

        Returns:
            A pandas DataFrame with sequence logits in melted format with
            columns 'seq_id' and 'probability'
        """
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.constants import esm3 as esm3_constants
        from esm.utils.structure.protein_complex import ProteinComplex

        if use_msa_context:
            raise ValueError("MSA context is only supported for E1 models")

        if isinstance(sequence_or_complex, list):
            protein_tensor = self._get_esm_protein_tensor_for_complex(
                sequence_or_complex, cif_file_path
            )
        else:
            protein_tensor = self._get_esm_protein_tensor_for_sequence(
                sequence_or_complex, cif_file_path
            )
        logits_output = self.client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=False)
        )

        sequence_probs = torch.softmax(logits_output.logits.sequence, dim=-1)
        melted_rows: List[Dict[str, Any]] = []

        if isinstance(sequence_or_complex, str):
            sequence = sequence_or_complex
            for pos in range(1, len(sequence) + 1):  # 1-based positions
                wt_aa = sequence[pos - 1]
                probs = sequence_probs[0, pos, :].tolist()

                for vocab_idx, vocab_char in enumerate(esm3_constants.SEQUENCE_VOCAB):
                    prob = probs[vocab_idx]
                    seq_id = f"{wt_aa}{pos}{vocab_char}"
                    melted_rows.append({"seq_id": seq_id, "probability": prob})
        else:
            # We don't know how this is formatted to start, so we just dump out the data.
            for idx in range(sequence_probs.shape[1]):
                probs = sequence_probs[0, idx, :].tolist()

                for vocab_idx, vocab_char in enumerate(esm3_constants.SEQUENCE_VOCAB):
                    prob = probs[vocab_idx]
                    seq_id = f"{idx}{vocab_char}"
                    melted_rows.append({"seq_id": seq_id, "probability": prob})

        return pd.DataFrame(melted_rows)


class FoldyESM3Client(FoldyPLMClient):
    """
    ESM-3 model client implementation for the foldy platform.

    Handles ESM-3 specific operations including protein tensor creation.
    Uses the same embedding and logit computation as ESM-C.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize the ESM-3 client with the specified model.

        Args:
            model_name: Name of the ESM-3 model to load
        """
        import torch
        from esm.models.esm3 import ESM3

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = ESM3.from_pretrained(model_name).to(device)
        self.device = device

    def _get_esm_protein_tensor_for_sequence(
        self, sequence: str, cif_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a sequence.

        Args:
            sequence: Protein sequence string
            cif_file_path: Optional path to a CIF file for structure-aware modeling

        Returns:
            Tensor representation of the protein
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_complex import ProteinComplex

        if cif_file_path:
            protein_complex = ProteinComplex.from_cif(path=cif_file_path)
            protein = ESMProtein.from_protein_complex(protein_complex)
        else:
            protein = ESMProtein(sequence=sequence)

        return self.client.encode(protein)

    def _get_esm_protein_tensor_for_complex(
        self, complex_input: ComplexType, cif_file_path: Optional[str] = None
    ) -> Any:  # -> torch.Tensor (use Any to avoid torch import)
        """
        Create an ESM protein tensor from a protein complex.

        Args:
            complex_input: List of (chain_id, sequence) tuples
            cif_file_path: Optional path to a CIF file for structure-aware modeling

        Returns:
            Tensor representation of the protein complex
        """
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.structure.protein_chain import ProteinChain
        from esm.utils.structure.protein_complex import ProteinComplex

        if cif_file_path:
            protein_complex = ProteinComplex.from_cif(path=cif_file_path)
            protein = ESMProtein.from_protein_complex(protein_complex)
        else:
            chains = [
                ProteinChain(chain_id=chain_id, sequence=sequence)
                for chain_id, sequence in complex_input
            ]
            protein_complex = ProteinComplex.from_chains(chains)
            protein = ESMProtein.from_protein_complex(protein_complex)

        return self.client.encode(protein)

    def supports_batch_embedding(self) -> bool:
        return True

    def embed_batch(
        self,
        sequences: List[str],
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[List[float]]]:
        """
        Get embeddings for a batch of protein sequences.

        Returns:
            A list of embedding lists (one per input sequence).
        """
        import torch
        from esm.sdk.api import LogitsConfig
        from esm.utils import encoding
        from esm.utils.misc import stack_variable_length_tensors
        from esm.utils.sampling import _BatchedESMProteinTensor

        if use_msa_context:
            raise ValueError("MSA context is only supported for E1 models")
        if cif_file_path:
            raise ValueError("ESM-3 batch embedding does not support CIF or PDB inputs")
        if not sequences:
            return []

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_tokens = None
        logits_output = None
        protein_tensor = None
        embeddings: List[List[List[float]]] = []

        try:
            sequence_tokenizer = self.client.tokenizers.sequence
            pad_token_id = sequence_tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("ESM-3 sequence tokenizer is missing pad_token_id")

            token_sequences = [
                encoding.tokenize_sequence(sequence, sequence_tokenizer, add_special_tokens=True)
                for sequence in sequences
            ]
            batch_tokens = stack_variable_length_tensors(
                token_sequences, constant_value=pad_token_id
            ).to(self.device)
            batch_lens = (batch_tokens != pad_token_id).sum(dim=1)

            protein_tensor = _BatchedESMProteinTensor(sequence=batch_tokens)
            with torch.no_grad():
                logits_output = self.client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=False,
                        return_embeddings=True,
                        return_hidden_states=True if len(extra_layers) > 0 else False,
                    ),
                )

            hidden_states = logits_output.hidden_states if extra_layers else None
            for idx, seq_len in enumerate(batch_lens):
                seq_len_int = int(seq_len.item())
                seq_embeddings: List[List[float]] = []
                if extra_layers and hidden_states is not None:
                    for extra_layer_idx in extra_layers:
                        layer_hidden = hidden_states[extra_layer_idx][
                            idx : idx + 1, :seq_len_int, :
                        ]
                        layer_embedding = self._pool_by_domains(
                            layer_hidden, domain_boundaries
                        ).squeeze(0)
                        seq_embeddings.append(layer_embedding.detach().cpu().tolist())

                final_hidden = logits_output.embeddings[idx : idx + 1, :seq_len_int, :]
                final_embedding = self._pool_by_domains(final_hidden, domain_boundaries).squeeze(0)
                seq_embeddings.append(final_embedding.detach().cpu().tolist())
                embeddings.append(seq_embeddings)

        finally:
            if "logits_output" in locals() and logits_output is not None:
                recursive_cleanup(logits_output)

            if "protein_tensor" in locals() and protein_tensor is not None:
                recursive_cleanup(protein_tensor)

            for local_var in ["logits_output", "protein_tensor", "batch_tokens"]:
                if local_var in locals() and locals()[local_var] is not None:
                    del locals()[local_var]

        return embeddings

    # Implementation similar to ESMCClient - reuse these methods
    embed = FoldyESMCClient.embed
    get_logits = FoldyESMCClient.get_logits


class FoldyESM1and2Client(FoldyPLMClient):
    """
    ESM-1 and ESM-2 model client implementation for the foldy platform.

    Handles the older ESM-1 and ESM-2 models which have a different API
    compared to ESM-3 and ESM-C.
    """

    MODEL_TO_NUM_LAYERS = {
        "esm2_t48_15B_UR50D": 48,
        "esm2_t36_3B_UR50D": 36,
        "esm2_t33_650M_UR50D": 33,
        "esm2_t30_150M_UR50D": 30,
        "esm2_t12_35M_UR50D": 12,
        "esm2_t6_8M_UR50D": 6,
        "esm1v_t33_650M_UR90S_1": 33,
        "esm1v_t33_650M_UR90S_2": 33,
        "esm1v_t33_650M_UR90S_3": 33,
        "esm1v_t33_650M_UR90S_4": 33,
        "esm1v_t33_650M_UR90S_5": 33,
        "esm_msa1b_t12_100M_UR50S": 12,
        "esm_msa1_t12_100M_UR50S": 12,
        "esm1b_t33_650M_UR50S": 33,
        "esm1_t34_670M_UR50S": 34,
        "esm1_t34_670M_UR50D": 34,
        "esm1_t34_670M_UR100": 34,
        "esm1_t12_85M_UR50S": 12,
        "esm1_t6_43M_UR50S": 6,
    }

    def _pool_by_domains(
        self, hidden_states: "torch.Tensor", domain_boundaries: List[int]
    ) -> "torch.Tensor":
        """
        Pool hidden states by domains and concatenate the results.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim]
            domain_boundaries: List of domain boundary positions (0-indexed)

        Returns:
            Concatenated tensor of domain embeddings
        """
        import torch

        if not domain_boundaries:
            return hidden_states.mean(0)

        # Convert to sorted list and add 0 at the start and seq_len at the end
        boundaries = [0] + sorted(domain_boundaries) + [hidden_states.shape[0]]
        domain_embeddings = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            domain_embedding = hidden_states[start_idx:end_idx, :].mean(0)
            domain_embeddings.append(domain_embedding)

        return torch.cat(domain_embeddings, dim=-1)

    def __init__(self, model_name: str) -> None:
        """
        Initialize the ESM-1/2 client with the specified model.

        Args:
            model_name: Name of the ESM-1 or ESM-2 model to load
        """
        import torch

        logging.info(
            f"Loading ESM-1/2 model: {model_name} (note: we have esm in sys.modules: {'esm' in sys.modules})"
        )
        if "esm" in sys.modules:
            logging.error(
                f"WE ARE REMOVING ESM FROM sys.modules... GOD HELP US. To run ESM2, we have to uninstall the esm package (which is only for ESMC/3). This is effectively uninstalling ESMC/ESM3 from the system. If they get used later, they will mysteriously fail."
            )
            sys.modules.pop("esm")

        self.model_name = model_name

        # Load model from torch hub
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", model_name)  # type: ignore[reportGeneralTypeIssues]
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Set to evaluation mode

        # For 15B model, use accelerate to split across GPU + CPU only on smaller GPUs
        if model_name == "esm2_t48_15B_UR50D" and torch.cuda.is_available():
            # Check available GPU memory
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            if total_memory_gb < 50:  # Only use CPU offloading on GPUs with <50GB
                try:
                    import accelerate
                    from accelerate import dispatch_model, infer_auto_device_map

                    # Calculate max memory (adjust for CUDA overhead ~2GB)
                    max_memory = {0: "42GiB", "cpu": "100GiB"}  # Adjust based on your system

                    device_map = infer_auto_device_map(
                        self.model,
                        max_memory=max_memory,
                        no_split_module_classes=[
                            "ESM1bLayerNorm",
                            "TransformerLayer",
                        ],  # Keep layers intact
                    )

                    self.model = dispatch_model(self.model, device_map=device_map)
                    self.device = torch.device("cuda")  # Primary device

                    logging.info(
                        f"Loaded {model_name} with device_map across GPU + CPU (GPU memory: {total_memory_gb:.1f}GB)"
                    )
                    logging.info(f"Device map: {device_map}")

                except ImportError:
                    logging.warning("accelerate not available, falling back to CPU for 15B model")
                    self.device = torch.device("cpu")
            else:
                # Large GPU (≥50GB) - load entirely on GPU for best performance
                self.model = self.model.cuda()
                self.device = torch.device("cuda")
                logging.info(
                    f"Loaded {model_name} entirely on GPU (GPU memory: {total_memory_gb:.1f}GB)"
                )
        elif torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def supports_batch_embedding(self) -> bool:
        return True

    def embed_batch(
        self,
        sequences: List[str],
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[List[float]]]:
        """
        Get embeddings for a batch of protein sequences.

        Returns:
            A list of embedding lists (one per input sequence).
        """
        import gc

        import torch

        if use_msa_context:
            raise ValueError("MSA context is only supported for E1 models")
        if cif_file_path:
            raise ValueError("ESM1 and 2 do not support CIF or PDB-based embeddings")
        if len(extra_layers) > 0:
            raise ValueError("ESM1 and 2 do not support extra layers")
        if not sequences:
            return []

        # Force CUDA synchronization before we start
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_tokens = None
        token_embeddings = None
        results = None
        embeddings: List[List[List[float]]] = []

        try:
            data = [("protein", sequence) for sequence in sequences]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(dim=1)

            with torch.no_grad():
                results = self.model(
                    batch_tokens, repr_layers=[self.MODEL_TO_NUM_LAYERS.get(self.model_name)]
                )
                layer_num = self.MODEL_TO_NUM_LAYERS.get(self.model_name)
                token_embeddings = results["representations"][layer_num]

                for idx, seq_len in enumerate(batch_lens):
                    seq_len_int = int(seq_len.item())
                    token_embeddings_no_special = token_embeddings[idx, 1 : seq_len_int - 1]
                    protein_embedding = (
                        self._pool_by_domains(token_embeddings_no_special, domain_boundaries)
                        .detach()
                        .cpu()
                    )
                    embeddings.append([protein_embedding.tolist()])

        finally:
            for local_var in ["batch_tokens", "token_embeddings", "results"]:
                if local_var in locals() and locals()[local_var] is not None:
                    del locals()[local_var]

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()

        return embeddings

    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Get embedding for a protein sequence.

        Args:
            sequence_or_complex: Protein sequence string (complex not supported)
            cif_file_path: Not supported for ESM-1/2
            extra_layers: Not supported for ESM-1/2
            domain_boundaries: List of domain boundary positions for domain pooling

        Returns:
            A list of list of floats representing embedding vectors for extra_layers, and the final layer.

        Raises:
            ValueError: If a complex, CIF/PDB file, or extra_layers are provided (not supported)
        """
        import gc

        import torch

        if use_msa_context:
            raise ValueError("MSA context is only supported for E1 models")

        # Force CUDA synchronization before we start
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sequence = sequence_or_complex
        if use_msa_context:
            if not msa_a3m_path:
                raise ValueError("msa_a3m_path is required when use_msa_context=true")
            context = self._get_msa_context(msa_a3m_path)
            sequence = f"{context},{sequence}"
        if cif_file_path:
            raise ValueError("ESM1 and 2 do not support CIF or PDB-based embeddings")
        if isinstance(sequence_or_complex, list):
            raise ValueError("ESM1 and 2 do not support protein complexes")
        if len(extra_layers) > 0:
            raise ValueError("ESM1 and 2 do not support extra layers")

        batch_tokens = None
        token_embeddings = None
        protein_embedding = None

        try:
            data = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(
                    batch_tokens, repr_layers=[self.MODEL_TO_NUM_LAYERS.get(self.model_name)]
                )
                layer_num = self.MODEL_TO_NUM_LAYERS.get(self.model_name)
                token_embeddings = results["representations"][layer_num]

                # Remove cls and eos tokens, then pool by domains within no_grad block
                token_embeddings_no_special = token_embeddings[0, 1:-1]  # Remove CLS and EOS tokens
                protein_embedding = (
                    self._pool_by_domains(token_embeddings_no_special, domain_boundaries)
                    .detach()
                    .cpu()
                )
                embeddings = [protein_embedding.tolist()]

        finally:
            # Expanded cleanup
            for local_var in ["batch_tokens", "token_embeddings", "protein_embedding", "results"]:
                if local_var in locals() and locals()[local_var] is not None:
                    del locals()[local_var]

            # Force garbage collection
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure GPU operations are complete
                torch.cuda.ipc_collect()  # Critical for multi-process environments

        return embeddings

    def get_logits(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence.

        Args:
            sequence_or_complex: Protein sequence string (complex not supported)
            cif_file_path: Not supported for ESM-1/2

        Returns:
            A pandas DataFrame with sequence logits in melted format with
            columns 'seq_id' and 'probability'

        Raises:
            ValueError: If a complex or CIF/PDB file is provided (not supported)
        """
        import torch

        if use_msa_context:
            raise ValueError("MSA context is only supported for E1 models")

        if isinstance(sequence_or_complex, list):
            raise ValueError("ESM1 and 2 do not support protein complexes")
        sequence = sequence_or_complex

        if cif_file_path:
            raise ValueError("ESM1 and 2 do not support CIF or PDB-based logits")

        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tokens, repr_layers=[33])["logits"]

        sequence_probs = torch.softmax(logits, dim=-1)
        melted_rows: List[Dict[str, Any]] = []

        for pos in range(1, len(sequence) + 1):  # 1-based positions
            wt_aa = sequence[pos - 1]
            probs = sequence_probs[0, pos, :].cpu().tolist()

            for vocab_idx, vocab_char in enumerate(self.alphabet.all_toks):
                if vocab_char in self.alphabet.standard_toks:  # Only include standard amino acids
                    prob = probs[vocab_idx]
                    seq_id = f"{wt_aa}{pos}{vocab_char}"
                    melted_rows.append({"seq_id": seq_id, "probability": prob})

        return pd.DataFrame(melted_rows)


class FoldyE1Client(FoldyPLMClient):
    """
    Profluent E1 model client implementation for the foldy platform.

    Handles Profluent E1 specific operations including embedding extraction
    and logit computation. E1 models are designed as drop-in replacements
    for ESM models with improved performance.
    """

    # Standard amino acids for logit output
    STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

    # Model name mapping from foldy names to HuggingFace names
    MODEL_NAME_MAP = {
        "e1_150m": "Profluent-Bio/E1-150m",
        "e1_300m": "Profluent-Bio/E1-300m",
        "e1_600m": "Profluent-Bio/E1-600m",
    }

    def _pool_by_domains(
        self, hidden_states: "torch.Tensor", domain_boundaries: List[int]
    ) -> "torch.Tensor":
        """
        Pool hidden states by domains and concatenate the results.

        Args:
            hidden_states: Tensor of shape [seq_len, hidden_dim]
            domain_boundaries: List of domain boundary positions (0-indexed)

        Returns:
            Concatenated tensor of domain embeddings
        """
        import torch

        if not domain_boundaries:
            return hidden_states.mean(dim=0)

        # Convert to sorted list and add 0 at the start and seq_len at the end
        boundaries = [0] + sorted(domain_boundaries) + [hidden_states.shape[0]]
        domain_embeddings = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            domain_embedding = hidden_states[start_idx:end_idx, :].mean(dim=0)
            domain_embeddings.append(domain_embedding)

        return torch.cat(domain_embeddings, dim=-1)

    def __init__(self, model_name: str) -> None:
        """
        Initialize the E1 client with the specified model.

        Args:
            model_name: Name of the E1 model to load (e1_150m, e1_300m, e1_600m)
        """
        import os

        import torch
        from E1.batch_preparer import E1BatchPreparer
        from E1.modeling import E1ForMaskedLM

        self.model_name = model_name
        hf_model_name = self.MODEL_NAME_MAP.get(model_name)
        if not hf_model_name:
            raise ValueError(
                f"Unknown E1 model: {model_name}. Available models: {list(self.MODEL_NAME_MAP.keys())}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = E1ForMaskedLM.from_pretrained(hf_model_name).to(self.device)
        self.model.eval()

        # Check for bfloat16 support
        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability()
            self.use_bf16 = major >= 8  # Ampere (RTX 30xx) or newer
        else:
            self.use_bf16 = False

        self.batch_preparer = E1BatchPreparer()
        self._msa_context_cache: Dict[str, str] = {}
        self._msa_context_seed: Optional[int] = None
        self._inductor_disabled = False
        if os.environ.get("FOLDY_E1_DISABLE_TORCH_COMPILE") == "1":
            self._disable_torch_compile("FOLDY_E1_DISABLE_TORCH_COMPILE=1")

        print(
            f"[E1] Loaded {model_name} ({hf_model_name}) on {self.device} "
            f"with bf16={'enabled' if self.use_bf16 else 'disabled'}"
        )

    def supports_batch_embedding(self) -> bool:
        return True

    def _disable_torch_compile(self, reason: str) -> None:
        if self._inductor_disabled:
            return

        disabled = False
        try:
            import torch._dynamo as torch_dynamo

            config = getattr(torch_dynamo, "config", None)
            if config is None:
                raise AttributeError("torch._dynamo.config is not available")
            config.disable = True
            config.suppress_errors = True
            disabled = True
        except Exception as exc:
            logging.warning("E1: failed to disable torch.compile: %s", exc)

        if disabled:
            logging.warning("E1: torch.compile disabled (%s)", reason)
        self._inductor_disabled = True

    def _should_disable_inductor(self, exc: Exception) -> bool:
        message = str(exc)
        return "No valid triton configs" in message or (
            "triton" in message and "out of resource" in message
        )

    def _run_model(self, batch: Dict[str, "torch.Tensor"]) -> Any:
        import torch

        def _forward():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                return self.model(
                    input_ids=batch["input_ids"],
                    within_seq_position_ids=batch["within_seq_position_ids"],
                    global_position_ids=batch["global_position_ids"],
                    sequence_ids=batch["sequence_ids"],
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )

        try:
            with torch.no_grad():
                return _forward()
        except Exception as exc:
            if not self._inductor_disabled and self._should_disable_inductor(exc):
                logging.warning(
                    "E1 attention kernel compile failed; retrying with torch.compile disabled: %s",
                    exc,
                )
                self._disable_torch_compile("triton config error")
                with torch.no_grad():
                    return _forward()
            raise

    def _get_msa_context(self, msa_a3m_path: str) -> str:
        if msa_a3m_path in self._msa_context_cache:
            return self._msa_context_cache[msa_a3m_path]

        try:
            with open(msa_a3m_path, "r") as msa_file:
                contents = msa_file.read()
            if ">" not in contents:
                context = contents.strip()
                if not context:
                    raise ValueError("Stored MSA context is empty")
                self._msa_context_cache[msa_a3m_path] = context
                logging.info(
                    "E1 MSA context loaded from stored context file (%d sequences, msa_a3m_path=%s)",
                    len(context.split(",")),
                    msa_a3m_path,
                )
                return context
        except OSError as exc:
            logging.warning("E1 MSA context read failed; falling back to sampling: %s", exc)

        if self._msa_context_seed is None:
            self._msa_context_seed = random.SystemRandom().randrange(0, 2**32 - 1)

        from E1.msa_sampling import ContextSpecification, sample_context

        context_spec = ContextSpecification()
        context, context_ids = sample_context(
            msa_a3m_path,
            max_num_samples=context_spec.max_num_samples,
            max_token_length=context_spec.max_token_length,
            max_query_similarity=context_spec.max_query_similarity,
            min_query_similarity=context_spec.min_query_similarity,
            neighbor_similarity_lower_bound=context_spec.neighbor_similarity_lower_bound,
            seed=self._msa_context_seed,
            device=self.device,
        )
        logging.info(
            "E1 MSA context sampled %d sequences (msa_a3m_path=%s)",
            len(context_ids),
            msa_a3m_path,
        )
        if not context:
            raise ValueError("MSA context sampling returned no context sequences")

        self._msa_context_cache[msa_a3m_path] = context
        return context

    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Get embedding for a protein sequence.

        Args:
            sequence_or_complex: Protein sequence string (complex not supported)
            cif_file_path: Not supported for E1
            extra_layers: Not supported for E1
            domain_boundaries: List of domain boundary positions for domain pooling

        Returns:
            A list containing the embedding vector(s)

        Raises:
            ValueError: If a complex, CIF file, or extra_layers are provided (not supported)
        """
        import gc

        import torch

        if cif_file_path:
            raise ValueError("E1 does not support CIF or PDB-based embeddings")
        if isinstance(sequence_or_complex, list):
            raise ValueError("E1 does not support protein complexes")
        if len(extra_layers) > 0:
            raise ValueError("E1 does not support extra layers")

        sequence = sequence_or_complex
        if use_msa_context:
            if not msa_a3m_path:
                raise ValueError("msa_a3m_path is required when use_msa_context=true")
            context = self._get_msa_context(msa_a3m_path)
            sequence = f"{context},{sequence}"

        # Force CUDA synchronization before we start
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        embeddings_result = None
        try:
            # Prepare batch using E1BatchPreparer
            batch = self.batch_preparer.get_batch_kwargs([sequence], device=self.device)

            outputs = self._run_model(batch)

            # Get embeddings for residues in the last sequence only (exclude boundary tokens)
            boundary_token_mask = self.batch_preparer.get_boundary_token_mask(batch["input_ids"])
            last_sequence_mask = (
                batch["sequence_ids"] == batch["sequence_ids"].max(dim=1).values[:, None]
            )
            residue_selector = last_sequence_mask & ~boundary_token_mask
            residue_embeddings = outputs.embeddings[0, residue_selector[0]].detach().cpu()

            # Pool by domains
            pooled_embedding = self._pool_by_domains(residue_embeddings, domain_boundaries)
            embeddings_result = [pooled_embedding.tolist()]

        finally:
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return embeddings_result

    def embed_batch(
        self,
        sequences: List[str],
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> List[List[List[float]]]:
        """
        Get embeddings for a batch of protein sequences.

        Returns:
            A list of embedding lists (one per input sequence).
        """
        import gc

        import torch

        if cif_file_path:
            raise ValueError("E1 does not support CIF or PDB-based embeddings")
        if len(extra_layers) > 0:
            raise ValueError("E1 does not support extra layers")
        if not sequences:
            return []
        if use_msa_context:
            if not msa_a3m_path:
                raise ValueError("msa_a3m_path is required when use_msa_context=true")
            context = self._get_msa_context(msa_a3m_path)
            sequences = [f"{context},{sequence}" for sequence in sequences]

        # Force CUDA synchronization before we start
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        embeddings_result: List[List[List[float]]] = []
        try:
            batch = self.batch_preparer.get_batch_kwargs(sequences, device=self.device)

            outputs = self._run_model(batch)

            boundary_token_mask = self.batch_preparer.get_boundary_token_mask(batch["input_ids"])
            last_sequence_mask = (
                batch["sequence_ids"] == batch["sequence_ids"].max(dim=1).values[:, None]
            )
            residue_selector = last_sequence_mask & ~boundary_token_mask

            for idx in range(batch["input_ids"].shape[0]):
                residue_embeddings = outputs.embeddings[idx, residue_selector[idx]].detach().cpu()
                pooled_embedding = self._pool_by_domains(residue_embeddings, domain_boundaries)
                embeddings_result.append([pooled_embedding.tolist()])

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return embeddings_result

    def get_logits(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        use_msa_context: bool = False,
        msa_a3m_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence.

        Args:
            sequence_or_complex: Protein sequence string (complex not supported)
            cif_file_path: Not supported for E1

        Returns:
            A pandas DataFrame with sequence logits in melted format with
            columns 'seq_id' and 'probability'

        Raises:
            ValueError: If a complex or CIF file is provided (not supported)
        """
        import torch

        if isinstance(sequence_or_complex, list):
            raise ValueError("E1 does not support protein complexes")
        if cif_file_path:
            raise ValueError("E1 does not support CIF or PDB-based logits")

        sequence = sequence_or_complex
        query_sequence = sequence
        if use_msa_context:
            if not msa_a3m_path:
                raise ValueError("msa_a3m_path is required when use_msa_context=true")
            context = self._get_msa_context(msa_a3m_path)
            sequence = f"{context},{sequence}"

        # Prepare batch
        batch = self.batch_preparer.get_batch_kwargs([sequence], device=self.device)

        outputs = self._run_model(batch)

        # Get logits for residues in the last sequence only (exclude boundary tokens)
        boundary_token_mask = self.batch_preparer.get_boundary_token_mask(batch["input_ids"])
        last_sequence_mask = (
            batch["sequence_ids"] == batch["sequence_ids"].max(dim=1).values[:, None]
        )
        residue_selector = last_sequence_mask & ~boundary_token_mask
        residue_logits = outputs.logits[0, residue_selector[0]]

        # Convert to probabilities
        sequence_probs = torch.softmax(residue_logits, dim=-1).cpu()

        # Get the vocabulary from the batch preparer/tokenizer
        # E1 uses a similar vocabulary structure to ESM
        vocab = self.batch_preparer.tokenizer.get_vocab()
        # Create reverse mapping: token_id -> token_str
        id_to_token = {v: k for k, v in vocab.items()}

        melted_rows: List[Dict[str, Any]] = []

        for pos in range(len(query_sequence)):  # 0-indexed positions in the filtered output
            wt_aa = query_sequence[pos]
            probs = sequence_probs[pos, :].tolist()

            for aa in self.STANDARD_AAS:
                if aa in vocab:
                    token_id = vocab[aa]
                    prob = probs[token_id]
                    seq_id = f"{wt_aa}{pos + 1}{aa}"  # 1-based position for seq_id
                    melted_rows.append({"seq_id": seq_id, "probability": prob})

        return pd.DataFrame(melted_rows)

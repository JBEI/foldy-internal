import gc
import json
import logging
import sys
from abc import ABC
from typing import TYPE_CHECKING, List, Optional, Union

import pandas as pd
import torch
from app.helpers.esm_client import FoldyESMClient

if TYPE_CHECKING:
    from E1.batch_preparer import E1BatchPreparer
    from E1.modeling import E1ForMaskedLM

SequenceType = str
SequenceOrComplexType = Union[str, List[str]]


class FoldyE1Client(FoldyESMClient):
    """
    E1 model client implementation mirroring ESM client pattern.
    Supports E1-150M, E1-300M, E1-600M models from Profluent-Bio.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize E1 client with specified model size.

        Args:
            model_name: 'e1-150m', 'e1-300m', or 'e1-600m'
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional["E1ForMaskedLM"] = None
        self._preparer: Optional["E1BatchPreparer"] = None
        self._available = False
        self.dtype = torch.float32

        if self.device.type == "cuda":
            import multiprocessing

            try:
                multiprocessing.set_start_method("spawn", force=True)
                logging.info(
                    f"E1 client {self.model_name}: set multiprocessing start method to 'spawn'"
                )
            except RuntimeError:
                logging.info(
                    f"E1 client {self.model_name}: multiprocessing start method already set"
                )

    def _lazy_load(self) -> None:
        if self._model is None:
            logging.info(f"E1 {self.model_name} _lazy_load: starting on {self.device}")
            try:
                logging.info(f"E1 {self.model_name}: attempting E1 package imports...")
                from E1.batch_preparer import E1BatchPreparer
                from E1.modeling import E1ForMaskedLM

                logging.info(f"E1 {self.model_name}: E1 imports successful")

                size_map = {"e1-150m": "150", "e1-300m": "300", "e1-600m": "600"}
                size = size_map[self.model_name]
                import multiprocessing

                logging.info(
                    f"E1 {self.model_name} _lazy_load: current mp start method '{multiprocessing.get_start_method()}'"
                )
                logging.info(f"Loading E1-{size}m from Profluent-Bio/E1-{size}m on {self.device}")

                self._preparer = E1BatchPreparer()
                logging.info(f"E1 {self.model_name}: E1BatchPreparer initialized")

                temp_model = E1ForMaskedLM.from_pretrained(f"Profluent-Bio/E1-{size}m")
                logging.info(f"E1 {self.model_name}: model downloaded/instantiated")

                self._model = temp_model.to(self.device, dtype=self.dtype).eval()
                logging.info(f"E1 {self.model_name}: model moved to {self.device} and set to eval")

                if (
                    self.device.type == "cuda"
                    and torch.cuda.is_bf16_supported()
                    and self._model is not None
                ):
                    self._model = self._model.to(torch.bfloat16)
                    self.dtype = torch.bfloat16
                    logging.info("E1: Using bfloat16 precision")
                else:
                    self.dtype = torch.float32
                    logging.info("E1: Using float32 precision")

                self._available = True
                logging.info(f"E1 {self.model_name}: LOADED SUCCESSFULLY - available=True")

            except ImportError as ie:
                logging.error(
                    f"E1 {self.model_name}: IMPORT FAILED - {ie}. Install with: pip install E1"
                )
                self._available = False
            except Exception as e:
                logging.error(f"E1 {self.model_name} _lazy_load FAILED: {type(e).__name__}: {e}")
                logging.error(f"E1 {self.model_name}: Full traceback: {sys.exc_info()}")
                self._available = False

    def embed(
        self,
        sequence_or_complex: SequenceOrComplexType,
        cif_file_path: Optional[str] = None,
        extra_layers: List[int] = [],
        domain_boundaries: List[int] = [],
    ) -> List[List[float]]:
        """
        Get E1 embeddings for protein sequence (pooled mean over residues excl. specials).

        Args:
            sequence_or_complex: Protein sequence string (complexes not supported)
            cif_file_path: Not supported for E1
            extra_layers: Not supported for E1 (returns final layer only)
            domain_boundaries: Not supported for E1 (returns full sequence mean pool)

        Returns:
            List of embedding vectors [[pooled_embedding]]
        """
        self._lazy_load()
        if not self._available:
            logging.warning("E1 client not available, returning empty embedding.")
            return [[]]

        if cif_file_path:
            raise ValueError("E1 does not support CIF/PDB structure input")
        if isinstance(sequence_or_complex, list):
            raise ValueError("E1 does not support protein complexes")
        if extra_layers:
            raise ValueError("E1 does not support extra layers")
        if domain_boundaries:
            raise ValueError("E1 does not support domain boundaries")

        sequence = sequence_or_complex
        if not sequence:
            raise ValueError("Empty sequence provided")

        seqs = [sequence]
        if self._preparer is not None:
            batch = self._preparer.get_batch_kwargs(seqs, device=self.device)
        else:
            raise ValueError("E1 preparer not initialized")

        try:
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    outputs = self._model(
                        input_ids=batch["input_ids"],
                        within_seq_position_ids=batch["within_seq_position_ids"],
                        global_position_ids=batch["global_position_ids"],
                        sequence_ids=batch["sequence_ids"],
                    )

            boundary_mask = self._preparer.get_boundary_token_mask(batch["input_ids"])
            residue_mask = ~boundary_mask[0]
            residue_embs = outputs.embeddings[0][residue_mask]
            embedding = residue_embs.mean(dim=0).cpu().tolist()
            embedding_list = [embedding]
        finally:
            # Comprehensive cleanup
            del batch
            del outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return embedding_list

    def get_logits(
        self, sequence_or_complex: SequenceOrComplexType, cif_file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get logits for a protein sequence (melted probabilities over standard AAs).

        Args:
            sequence_or_complex: Protein sequence string (complexes not supported)
            cif_file_path: Not supported for E1

        Returns:
            pd.DataFrame with 'seq_id' (e.g. 'A1C') and 'probability' columns
        """
        self._lazy_load()
        if self._preparer is None or self._model is None or not self._available:
            logging.warning("E1 client not available, returning empty DataFrame for logits.")
            return pd.DataFrame(columns=["seq_id", "probability"])

        if cif_file_path:
            raise ValueError("E1 does not support CIF/PDB structure input")
        if isinstance(sequence_or_complex, list):
            raise ValueError("E1 does not support protein complexes")

        sequence = sequence_or_complex
        if not sequence:
            raise ValueError("Empty sequence provided")

        seqs = [sequence]
        if self._preparer is not None:
            batch = self._preparer.get_batch_kwargs(seqs, device=self.device)
        else:
            raise ValueError("E1 preparer not initialized")

        try:
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    outputs = self._model(
                        input_ids=batch["input_ids"],
                        within_seq_position_ids=batch["within_seq_position_ids"],
                        global_position_ids=batch["global_position_ids"],
                        sequence_ids=batch["sequence_ids"],
                    )

            logits = outputs.logits[0]
            boundary_mask = self._preparer.get_boundary_token_mask(batch["input_ids"])
            residue_mask = ~boundary_mask[0]
            residue_logits = logits[residue_mask]
            probs = torch.softmax(residue_logits, dim=-1)

            L = len(sequence)
            assert (
                residue_logits.shape[0] == L
            ), f"Expected {L} residues, got {residue_logits.shape[0]}"

            standard_aas = [
                "A",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "K",
                "L",
                "M",
                "N",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "V",
                "W",
                "Y",
            ]
            vocab = self._preparer.tokenizer.get_vocab()
            melted_rows = []
            for i in range(L):
                wt_aa = sequence[i]
                pos_probs = probs[i]
                for aa in standard_aas:
                    vocab_idx = vocab[aa]
                    prob = pos_probs[vocab_idx].item()
                    seq_id = f"{wt_aa}{i+1}{aa}"
                    melted_rows.append({"seq_id": seq_id, "probability": prob})

            return pd.DataFrame(melted_rows)
        finally:
            del batch
            del outputs
            del logits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


# Patch FoldyESMClient factory to support E1 models
def get_e1_client(model_name: str) -> FoldyE1Client:
    """Factory method for E1 clients."""
    if model_name not in ["e1-150m", "e1-300m", "e1-600m"]:
        raise ValueError(f"Unknown E1 model: {model_name}")
    return FoldyE1Client(model_name)


def new_get_client(cls, model_name):
    """New get_client with E1 support."""
    if model_name.startswith("e1-"):
        return get_e1_client(model_name)
    return original_get_client(model_name)


original_get_client = FoldyESMClient.get_client
FoldyESMClient.get_client = classmethod(new_get_client)

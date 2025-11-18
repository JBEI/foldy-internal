#!/usr/bin/env python3
"""
Standalone E1-150m inference script for protein sequence logits and embeddings.
Based on E1-copy/cookbook/basic.ipynb patterns.
"""

import os
import torch
import numpy as np
from pathlib import Path
import subprocess
from typing import Tuple, Dict, Any

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Install E1 if not present
try:
    from E1 import E1ForMaskedLM, E1BatchPreparer
except ImportError:
    print("Installing E1 from git+https://github.com/Profluent-AI/E1.git")
    subprocess.check_call(["pip", "install", "git+https://github.com/Profluent-AI/E1.git"])
    from E1 import E1ForMaskedLM, E1BatchPreparer

def setup_device_and_flash_attn() -> torch.device:
    """Setup CUDA device and optionally install flash-attn."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")

        # Install flash-attn if not available and CUDA supports it
        if not FLASH_ATTN_AVAILABLE:
            sm_version = torch.cuda.get_device_capability()
            if sm_version >= (8, 0):
                print("Installing flash-attn for CUDA...")
                subprocess.check_call(["pip", "install", "flash-attn", "--no-build-isolation"])
                print("Flash-attn installed successfully.")
            else:
                print(f"CUDA capability {sm_version} too low for flash-attn (requires >=8.0)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def aa_to_id(seq: str) -> torch.Tensor:
    """Convert amino acid sequence to input_ids using E1 alphabet."""
    # E1 uses standard 20 AA + special tokens
    aa_map = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8,
              'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16,
              'V':17, 'W':18, 'Y':19}
    return torch.tensor([aa_map.get(c, 0) for c in seq], dtype=torch.long)

def compute_e1_logits_embeddings(
    sequence: str,
    model_name: str = "Profluent-Bio/E1-150m",
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Compute logits and embeddings for a protein sequence using E1-150m.

    Args:
        sequence: Protein sequence (e.g., "ACDEFGHIKLMNPQRSTVWY")
        model_name: HuggingFace model name
        device: torch device

    Returns:
        Dict containing logits, embeddings, and sequence embedding
    """
    if device is None:
        device = setup_device_and_flash_attn()

    # Tokenize sequence
    input_ids = aa_to_id(sequence).unsqueeze(0).to(device)  # [1, seq_len]
    seq_len = len(sequence)

    print(f"Sequence: {sequence}")
    print(f"Input shape: {input_ids.shape}")

    # Setup batch preparer and model
    batch_preparer = E1BatchPreparer(alphabet_size=20)
    model = E1ForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    # Prepare inputs matching notebook patterns
    with torch.no_grad():
        # Single sequence: within_seq_position_ids = 0, global_position_ids = 0, sequence_ids = 0
        within_seq_position_ids = torch.zeros_like(input_ids)
        global_position_ids = torch.zeros_like(input_ids)
        sequence_ids = torch.zeros_like(input_ids)

        # Forward pass with autocast
        with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
            outputs = model(
                input_ids=input_ids,
                within_seq_position_ids=within_seq_position_ids,
                global_position_ids=global_position_ids,
                sequence_ids=sequence_ids
            )

        logits = outputs.logits  # [1, seq_len, 20]
        hidden_states = outputs.last_hidden_state  # [1, seq_len, embed_dim]

    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {hidden_states.shape}")

    # Convert to CPU numpy for analysis
    logits = logits.detach().cpu().numpy().squeeze()  # [seq_len, 20]
    embeddings = hidden_states.detach().cpu().numpy().squeeze()  # [seq_len, embed_dim]

    # Overall sequence embedding (mean pool)
    seq_embedding = embeddings.mean(axis=0)

    return {
        'sequence': sequence,
        'logits': logits,
        'embeddings': embeddings,
        'sequence_embedding': seq_embedding,
        'sequence_length': seq_len
    }

def visualize_top_predictions(logits: np.ndarray, sequence: str) -> None:
    """Print top-5 log probabilities per position."""
    print("\n" + "="*80)
    print("TOP-5 LOG PROBABILITIES PER POSITION")
    print("="*80)

    aa_names = 'ACDEFGHIKLMNPQRSTVWY'
    log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=-1).numpy()

    for pos in range(len(sequence)):
        top5 = np.argsort(log_probs[pos])[-5:][::-1]
        print(f"Position {pos+1} ({sequence[pos]}):")
        for rank, aa_idx in enumerate(top5, 1):
            prob = np.exp(log_probs[pos, aa_idx])
            print(f"  {rank}. {aa_names[aa_idx]}: {prob:.3f}")

def print_embedding_stats(embeddings: np.ndarray, seq_embedding: np.ndarray) -> None:
    """Print embedding statistics."""
    print("\n" + "="*80)
    print("EMBEDDING STATISTICS")
    print("="*80)
    print(f"Per-residue embeddings shape: {embeddings.shape}")
    print(f"Sequence embedding shape: {seq_embedding.shape}")
    print(f"Per-residue embedding norms: min={np.linalg.norm(embeddings, axis=1).min():.3f}, "
          f"max={np.linalg.norm(embeddings, axis=1).max():.3f}, mean={np.linalg.norm(embeddings, axis=1).mean():.3f}")
    print(f"Sequence embedding norm: {np.linalg.norm(seq_embedding):.3f}")

def main():
    sequence = "ACDEFGHIKLMNPQRSTVWY"

    results = compute_e1_logits_embeddings(sequence)

    # Visualize results
    visualize_top_predictions(results['logits'], results['sequence'])
    print_embedding_stats(results['embeddings'], results['sequence_embedding'])

    print("\n" + "="*80)
    print("E1-150m INFERENCE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

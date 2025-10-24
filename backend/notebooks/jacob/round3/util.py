"""Utilities for analyzing mutations in GenBank plasmid sequences."""

from dataclasses import dataclass
from typing import List, Optional

import Levenshtein
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


@dataclass
class Mutation:
    """Represents a mutation in a plasmid sequence.

    Attributes:
        position: 0-indexed genomic position of the mutation
        mutation_type: 'codon' for CDS codon changes, or 'substitution'/'insertion'/'deletion' for non-CDS
        reference_base: Base(s) in reference sequence (genomic orientation)
        variant_base: Base(s) in variant sequence (genomic orientation)
        annotations: List of feature labels that overlap this position
        amino_acid_change: For CDS mutations, the AA change (e.g., 'G47Y')
        end_position: For codon mutations, the end genomic position
    """

    position: int
    mutation_type: str
    reference_base: str
    variant_base: str
    annotations: List[str]
    amino_acid_change: Optional[str] = None
    end_position: Optional[int] = None


def reindex_circular_genbank(reference: SeqRecord, query: SeqRecord) -> str:
    """Reindex a circular plasmid to match the reference starting position.

    Uses Levenshtein distance on subsequences to find the best alignment offset.

    Args:
        reference: Reference GenBank SeqRecord
        query: Query GenBank SeqRecord to reindex

    Returns:
        Reindexed sequence string
    """
    ref_seq = str(reference.seq).upper()
    query_seq = str(query.seq).upper()

    if ref_seq == query_seq:
        return query_seq

    rev_query_seq = str(Seq(query_seq).reverse_complement())
    if ref_seq == rev_query_seq:
        return rev_query_seq

    window_size = min(500, len(ref_seq))
    ref_window = ref_seq[:window_size]

    best_result = {
        "distance": float("inf"),
        "offset": 0,
        "sequence": query_seq,
    }

    for orientation_seq in (query_seq, rev_query_seq):
        doubled_query = orientation_seq + orientation_seq
        best_offset = 0
        best_distance = float("inf")

        for offset in range(len(query_seq)):
            query_window = doubled_query[offset : offset + window_size]
            if len(query_window) != window_size:
                continue

            distance = Levenshtein.distance(ref_window, query_window)

            if distance < best_distance:
                best_distance = distance
                best_offset = offset

            if distance < window_size * 0.01:
                break

        if best_distance < best_result["distance"]:
            best_result["distance"] = best_distance
            best_result["offset"] = best_offset
            best_result["sequence"] = orientation_seq

    doubled_best = best_result["sequence"] + best_result["sequence"]
    start = best_result["offset"]
    end = start + len(query_seq)
    return doubled_best[start:end]


def find_mutations(reference: SeqRecord, query_seq: str) -> List[Mutation]:
    """Find mutations between reference and query sequences.

    Uses codon-by-codon comparison for CDS features, base-level for non-CDS regions.

    Args:
        reference: Reference SeqRecord with annotations
        query_seq: Query sequence string (already reindexed if circular)

    Returns:
        List of Mutation objects
    """
    ref_seq = str(reference.seq)

    # Get base-level mutations from Levenshtein
    ops = Levenshtein.editops(ref_seq, query_seq)

    base_mutations = []
    for op, ref_pos, query_pos in ops:
        if op == "insert":
            base_mutations.append(
                Mutation(
                    position=ref_pos,
                    mutation_type="insertion",
                    reference_base="",
                    variant_base=query_seq[query_pos],
                    annotations=[],
                    amino_acid_change=None,
                )
            )
        elif op == "delete" or ref_pos >= len(query_seq):
            base_mutations.append(
                Mutation(
                    position=ref_pos,
                    mutation_type="deletion",
                    reference_base=ref_seq[ref_pos],
                    variant_base="",
                    annotations=[],
                    amino_acid_change=None,
                )
            )
        elif op == "replace":
            base_mutations.append(
                Mutation(
                    position=ref_pos,
                    mutation_type="substitution",
                    reference_base=ref_seq[ref_pos],
                    variant_base=query_seq[ref_pos],
                    annotations=[],
                    amino_acid_change=None,
                )
            )
        else:
            raise ValueError(f"Unknown operation: {op}")

    # Get CDS features and their positions
    cds_features = [f for f in reference.features if f.type == "CDS"]
    cds_positions = set()
    for feature in cds_features:
        for pos in feature.location:
            cds_positions.add(pos)

    # Keep only non-CDS mutations
    non_cds_mutations = [m for m in base_mutations if m.position not in cds_positions]

    # Add annotations for non-CDS mutations
    for mut in non_cds_mutations:
        mut.annotations = _get_annotations_at_position(reference, mut.position)

    # Analyze CDS features codon-by-codon
    cds_mutations = []
    for feature in cds_features:
        cds_muts = _analyze_cds_codons(reference, query_seq, feature)
        cds_mutations.extend(cds_muts)

    # Combine and sort
    all_mutations = non_cds_mutations + cds_mutations
    all_mutations.sort(key=lambda m: m.position)

    return all_mutations


def _analyze_cds_codons(reference: SeqRecord, query_seq: str, feature) -> List[Mutation]:
    """Analyze a CDS feature codon-by-codon.

    Args:
        reference: Reference SeqRecord
        query_seq: Query sequence
        feature: The CDS feature to analyze

    Returns:
        List of codon-level Mutation objects
    """
    # Extract reference CDS sequence (gene orientation)
    ref_cds_seq = str(feature.location.extract(reference.seq))

    # Find the CDS in the query by searching for it
    # This handles cases where indels outside the CDS shift the positions
    window_size = min(100, len(ref_cds_seq))
    ref_window = ref_cds_seq[:window_size]

    # Search for the CDS start in the query
    best_score = float("inf")
    best_pos = None

    # For forward strand features, search forward strand
    # For reverse strand features, search reverse strand
    if feature.location.strand == 1:
        search_seq = query_seq
    else:
        search_seq = str(Seq(query_seq).reverse_complement())

    for i in range(len(search_seq) - len(ref_cds_seq) + 1):
        query_window = search_seq[i : i + window_size]
        score = Levenshtein.distance(ref_window, query_window)
        if score < best_score:
            best_score = score
            best_pos = i
        if score == 0:
            break

    # Check if the whole thing is deleted.
    if best_pos is None:
        query_cds_seq = "N" * len(ref_cds_seq)
    else:
        # Extract query CDS from found position
        query_cds_seq = search_seq[best_pos : best_pos + len(ref_cds_seq)]

    # Get feature label
    label = _get_feature_label(feature)

    mutations = []

    # Compare codon by codon
    num_codons = len(ref_cds_seq) // 3

    for codon_idx in range(num_codons):
        start = codon_idx * 3
        end = start + 3

        ref_codon = ref_cds_seq[start:end]
        query_codon = (
            query_cds_seq[start:end]
            if end <= len(query_cds_seq)
            else query_cds_seq[start:] + "N" * (end - len(query_cds_seq))
        )

        if ref_codon != query_codon:
            # Get genomic positions for this codon from reference
            positions = list(feature.location)
            codon_genomic_positions = sorted(positions[start:end])

            # Get genomic bases from reference
            ref_genomic = "".join([str(reference.seq[p]) for p in codon_genomic_positions])

            # For query genomic bases, we need to figure out where this codon is in the query
            # Use the found CDS position and map back to genomic coords
            # For simplicity, just show the query codon in gene orientation
            query_genomic = (
                query_codon
                if feature.location.strand == 1
                else str(Seq(query_codon).reverse_complement())
            )

            # Translate
            ref_aa = Seq(ref_codon).translate() if len(ref_codon) == 3 else "?"
            query_aa = Seq(query_codon).translate() if len(query_codon) == 3 else "?"
            aa_pos = codon_idx + 1
            aa_change = f"{ref_aa}{aa_pos}{query_aa}"

            mutations.append(
                Mutation(
                    position=codon_genomic_positions[0],
                    end_position=codon_genomic_positions[-1],
                    mutation_type="codon",
                    reference_base=ref_genomic,
                    variant_base=query_genomic,
                    annotations=[label],
                    amino_acid_change=aa_change,
                )
            )

    return mutations


def _get_annotations_at_position(record: SeqRecord, position: int) -> List[str]:
    """Get all feature labels that overlap a given position."""
    annotations = []
    for feature in record.features:
        if position in feature.location:
            label = _get_feature_label(feature)
            annotations.append(label)
    return annotations


def _get_feature_label(feature) -> str:
    """Get a label for a feature."""
    if "label" in feature.qualifiers:
        label = feature.qualifiers["label"]
        return label[0] if isinstance(label, list) else label
    elif "gene" in feature.qualifiers:
        gene = feature.qualifiers["gene"]
        return gene[0] if isinstance(gene, list) else gene
    else:
        return feature.type

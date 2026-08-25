"""Dataset auditing and reproducible manifests for multimutant benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from folde.data import DMS_DIR, DMS_METADATA_FILE, EMBEDDINGS_DIR, NATURALNESS_DIR

ALLELE_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")
MANIFEST_SCHEMA_VERSION = "1.0"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_mutant(mutant: str) -> list[tuple[str, int, str]] | None:
    if mutant == "WT":
        return []
    substitutions: list[tuple[str, int, str]] = []
    for allele in mutant.split(":"):
        match = ALLELE_PATTERN.fullmatch(allele)
        if match is None:
            return None
        substitutions.append((match.group(1), int(match.group(2)), match.group(3)))
    if len({position for _, position, _ in substitutions}) != len(substitutions):
        return None
    return substitutions


def _possible_shell_size(position_count: int, depth: int) -> int:
    if depth > position_count:
        return 0
    return math.comb(position_count, depth) * (19**depth)


def audit_dataset(path: Path, reference_sequence: str, chunk_size: int = 100_000) -> dict[str, Any]:
    """Audit one ProteinGym activity CSV using bounded memory."""
    depth_counts: Counter[int] = Counter()
    positions: set[int] = set()
    observed_single_alleles: set[str] = set()
    required_component_alleles: set[str] = set()
    row_count = 0
    finite_score_count = 0
    invalid_mutant_count = 0
    reference_mismatch_count = 0
    unique_mutants: set[str] = set()

    for chunk in pd.read_csv(path, usecols=["mutant", "DMS_score"], chunksize=chunk_size):
        row_count += len(chunk)
        finite_score_count += int(
            np.isfinite(pd.to_numeric(chunk["DMS_score"], errors="coerce")).sum()
        )
        for raw_mutant in chunk["mutant"]:
            mutant = str(raw_mutant)
            unique_mutants.add(mutant)
            substitutions = _parse_mutant(mutant)
            if substitutions is None:
                invalid_mutant_count += 1
                continue
            depth = len(substitutions)
            depth_counts[depth] += 1
            alleles = {
                f"{reference}{position}{alternate}"
                for reference, position, alternate in substitutions
            }
            positions.update(position for _, position, _ in substitutions)
            if depth == 1:
                observed_single_alleles.update(alleles)
            elif depth > 1:
                required_component_alleles.update(alleles)
            for reference, position, _ in substitutions:
                index = position - 1
                if (
                    index < 0
                    or index >= len(reference_sequence)
                    or reference_sequence[index] != reference
                ):
                    reference_mismatch_count += 1

    position_count = len(positions)
    shell_counts = {str(depth): count for depth, count in sorted(depth_counts.items())}
    shell_possible = {
        str(depth): _possible_shell_size(position_count, depth) for depth in sorted(depth_counts)
    }
    shell_coverage = {
        depth: (shell_counts[depth] / possible if possible else None)
        for depth, possible in shell_possible.items()
    }
    component_count = len(required_component_alleles)
    return {
        "dms_id": path.stem,
        "activity_file": path.name,
        "activity_file_bytes": path.stat().st_size,
        "activity_sha256": sha256_file(path),
        "row_count": row_count,
        "unique_mutant_count": len(unique_mutants),
        "finite_score_count": finite_score_count,
        "invalid_mutant_count": invalid_mutant_count,
        "reference_mismatch_count": reference_mismatch_count,
        "mutation_depth_counts": shell_counts,
        "maximum_mutation_depth": max(depth_counts, default=0),
        "mutated_position_count": position_count,
        "mutated_positions": sorted(positions),
        "standard_shell_possible_counts": shell_possible,
        "standard_shell_coverage": shell_coverage,
        "multimutant_component_allele_count": component_count,
        "measured_component_allele_count": len(
            required_component_alleles & observed_single_alleles
        ),
        "multimutant_component_coverage": (
            len(required_component_alleles & observed_single_alleles) / component_count
            if component_count
            else None
        ),
    }


def _feature_kind(path: Path) -> str:
    return "embedding" if "embedding" in path.name else "naturalness"


def _matching_dms_id(path: Path, dms_ids: Iterable[str]) -> str | None:
    matches = [dms_id for dms_id in dms_ids if path.name.startswith(f"{dms_id}_")]
    return max(matches, key=len) if matches else None


def inventory_features(dms_ids: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Hash locally available embedding and naturalness feature files."""
    known_ids = tuple(dms_ids)
    records: list[dict[str, Any]] = []
    unmatched: list[str] = []
    paths = sorted(
        {
            path.resolve()
            for directory in (EMBEDDINGS_DIR, NATURALNESS_DIR)
            if directory.exists()
            for path in directory.iterdir()
            if path.is_file()
        }
    )
    for path in paths:
        dms_id = _matching_dms_id(path, known_ids)
        relative_path = path.relative_to(DMS_DIR.parent.parent).as_posix()
        if dms_id is None:
            unmatched.append(relative_path)
            continue
        records.append(
            {
                "dms_id": dms_id,
                "kind": _feature_kind(path),
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records, unmatched


def build_dataset_manifest(include_feature_hashes: bool = True) -> dict[str, Any]:
    """Build a deterministic-content manifest for every local ProteinGym assay."""
    metadata = pd.read_csv(DMS_METADATA_FILE).set_index("DMS_id", drop=False)
    csv_paths = sorted(DMS_DIR.glob("*.csv"))
    datasets: list[dict[str, Any]] = []
    for path in csv_paths:
        if path.stem not in metadata.index:
            raise ValueError(f"Activity file has no metadata row: {path.name}")
        row = metadata.loc[path.stem]
        datasets.append(audit_dataset(path, str(row["target_seq"])))

    metadata_ids = set(metadata.index.astype(str))
    activity_ids = {path.stem for path in csv_paths}
    feature_files: list[dict[str, Any]] = []
    unmatched_features: list[str] = []
    if include_feature_hashes:
        feature_files, unmatched_features = inventory_features(sorted(metadata_ids))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata_file": DMS_METADATA_FILE.name,
        "metadata_sha256": sha256_file(DMS_METADATA_FILE),
        "metadata_dataset_count": len(metadata_ids),
        "activity_dataset_count": len(activity_ids),
        "metadata_without_activity": sorted(metadata_ids - activity_ids),
        "activity_without_metadata": sorted(activity_ids - metadata_ids),
        "datasets": datasets,
        "feature_files": feature_files,
        "unmatched_feature_files": unmatched_features,
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Write a manifest atomically as stable, human-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(output_path)

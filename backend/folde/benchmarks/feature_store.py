"""Indexed memory-mapped storage for large benchmark embedding tables."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from folde.benchmarks.multimutant_data import sha256_file

STORE_SCHEMA_VERSION = "1.0"


class MemmapFeatureStore:
    """Read embedding rows without materializing the full matrix in host memory."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata.get("schema_version") != STORE_SCHEMA_VERSION:
            raise ValueError("unsupported feature-store schema version")
        self.seq_ids = tuple((directory / "seq_ids.txt").read_text().splitlines())
        if len(self.seq_ids) != self.metadata["row_count"]:
            raise ValueError("feature-store identifier count does not match metadata")
        if len(self.seq_ids) != len(set(self.seq_ids)):
            raise ValueError("feature-store identifiers are not unique")
        self._row_by_seq_id = {seq_id: row for row, seq_id in enumerate(self.seq_ids)}
        self._matrix = np.load(directory / "embeddings.npy", mmap_mode="r")
        expected_shape = (self.metadata["row_count"], self.metadata["embedding_dim"])
        if self._matrix.shape != expected_shape:
            raise ValueError("feature-store matrix shape does not match metadata")

    @property
    def embedding_dim(self) -> int:
        return int(self.metadata["embedding_dim"])

    def contains(self, seq_id: str) -> bool:
        return seq_id in self._row_by_seq_id

    def get_array(self, seq_ids: Sequence[str]) -> np.ndarray:
        """Return requested rows in caller order, failing on missing identifiers."""
        missing = [seq_id for seq_id in seq_ids if seq_id not in self._row_by_seq_id]
        if missing:
            raise KeyError(f"feature store lacks {len(missing)} identifiers: {missing[:5]}")
        rows = np.fromiter(
            (self._row_by_seq_id[seq_id] for seq_id in seq_ids),
            dtype=np.int64,
            count=len(seq_ids),
        )
        return np.asarray(self._matrix[rows])

    def get_series(self, seq_ids: Sequence[str]) -> pd.Series:
        matrix = self.get_array(seq_ids)
        return pd.Series([row for row in matrix], index=pd.Index(seq_ids), dtype=object)


def _parse_embedding(value: str) -> np.ndarray:
    stripped = value.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise ValueError("embedding must be encoded as a JSON array")
    parsed = np.fromstring(stripped[1:-1], dtype=np.float32, sep=",")
    if parsed.ndim != 1 or not np.isfinite(parsed).all():
        raise ValueError("embedding must be a finite one-dimensional vector")
    return parsed


def prepare_memmap_feature_store(
    source_csv: Path,
    output_dir: Path,
    *,
    chunk_size: int = 2_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a CSV embedding table once into an indexed ``.npy`` memmap.

    The output is written via sibling temporary files and promoted only after every
    source row has been parsed and a sampled equality check succeeds.
    """
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not overwrite:
        return json.loads(metadata_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    first = pd.read_csv(source_csv, usecols=["seq_id", "embedding"], nrows=1)
    if len(first) != 1:
        raise ValueError("embedding source is empty")
    first_vector = _parse_embedding(str(first.iloc[0]["embedding"]))
    row_count = sum(
        len(chunk) for chunk in pd.read_csv(source_csv, usecols=["seq_id"], chunksize=100_000)
    )
    temporary_matrix = output_dir / "embeddings.npy.tmp"
    temporary_ids = output_dir / "seq_ids.txt.tmp"
    matrix = np.lib.format.open_memmap(
        temporary_matrix,
        mode="w+",
        dtype=np.float32,
        shape=(row_count, len(first_vector)),
    )
    sample_rows: dict[int, np.ndarray] = {0: first_vector}
    row_offset = 0
    with temporary_ids.open("w") as id_handle:
        for chunk in pd.read_csv(
            source_csv,
            usecols=["seq_id", "embedding"],
            chunksize=chunk_size,
        ):
            vectors = np.stack([_parse_embedding(str(value)) for value in chunk["embedding"]])
            if vectors.shape[1] != len(first_vector):
                raise ValueError("embedding dimensions are inconsistent")
            end = row_offset + len(chunk)
            matrix[row_offset:end] = vectors
            id_handle.writelines(f"{seq_id}\n" for seq_id in chunk["seq_id"].astype(str))
            for local_row in (0, len(chunk) // 2, len(chunk) - 1):
                if local_row >= 0:
                    sample_rows[row_offset + local_row] = vectors[local_row].copy()
            row_offset = end
    if row_offset != row_count:
        raise ValueError(f"parsed {row_offset} rows but expected {row_count}")
    matrix.flush()
    del matrix
    reloaded = np.load(temporary_matrix, mmap_mode="r")
    for row, expected in sample_rows.items():
        if not np.array_equal(reloaded[row], expected):
            raise ValueError(f"embedding equality check failed at row {row}")
    del reloaded
    matrix_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "seq_ids.txt"
    temporary_matrix.replace(matrix_path)
    temporary_ids.replace(ids_path)
    metadata: dict[str, Any] = {
        "schema_version": STORE_SCHEMA_VERSION,
        "source_path": str(source_csv),
        "source_bytes": source_csv.stat().st_size,
        "source_sha256": sha256_file(source_csv),
        "row_count": row_count,
        "embedding_dim": len(first_vector),
        "dtype": "float32",
        "sampled_equality_rows": sorted(sample_rows),
        "matrix_sha256": sha256_file(matrix_path),
        "seq_ids_sha256": sha256_file(ids_path),
    }
    temporary_metadata = output_dir / "metadata.json.tmp"
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary_metadata.replace(metadata_path)
    return metadata

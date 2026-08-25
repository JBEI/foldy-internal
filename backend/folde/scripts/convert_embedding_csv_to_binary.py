"""Convert JSON-in-CSV embeddings into a memory-mapped float32 cache.

The cache lives beside the source CSV as ``*.float32.npy`` plus
``*.seq_ids.txt``. ``folde.data`` uses it automatically while it is newer than
the source CSV. Conversion is streaming, so even the 12 GB Olson file does not
need to fit in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from folde.data import get_binary_embedding_paths


def _count_data_rows(path: Path) -> int:
    newline_count = 0
    last_byte = b"\n"
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    line_count = newline_count + int(last_byte != b"\n")
    return max(0, line_count - 1)


def convert_embedding_csv(path: Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Stream one embedding CSV into an atomic float32 binary cache."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    matrix_path, seq_ids_path = get_binary_embedding_paths(path)
    cache_is_fresh = (
        matrix_path.exists()
        and seq_ids_path.exists()
        and matrix_path.stat().st_mtime >= path.stat().st_mtime
        and seq_ids_path.stat().st_mtime >= path.stat().st_mtime
    )
    if cache_is_fresh and not overwrite:
        return matrix_path, seq_ids_path

    csv.field_size_limit(sys.maxsize)
    row_count = _count_data_rows(path)
    if row_count == 0:
        raise ValueError(f"Embedding CSV has no data rows: {path}")

    with path.open(newline="") as source:
        first_row = next(csv.DictReader(source))
    first_embedding = np.asarray(json.loads(first_row["embedding"]), dtype=np.float32)
    if first_embedding.ndim != 1:
        raise ValueError(f"Expected one-dimensional embeddings in {path}")

    matrix_tmp = matrix_path.with_name(f".{matrix_path.name}.tmp")
    seq_ids_tmp = seq_ids_path.with_name(f".{seq_ids_path.name}.tmp")
    try:
        matrix = open_memmap(
            matrix_tmp,
            mode="w+",
            dtype=np.float32,
            shape=(row_count, first_embedding.shape[0]),
        )
        with path.open(newline="") as source, seq_ids_tmp.open("w") as id_output:
            reader = csv.DictReader(source)
            written_rows = 0
            for row_index, row in enumerate(reader):
                embedding = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                if embedding.shape != (matrix.shape[1],):
                    raise ValueError(
                        f"Row {row_index + 2} has shape {embedding.shape}; "
                        f"expected {(matrix.shape[1],)}"
                    )
                matrix[row_index] = embedding
                id_output.write(f"{row['seq_id']}\n")
                written_rows = row_index + 1
        if written_rows != row_count:
            raise ValueError(
                f"Counted {row_count} rows in {path}, but parsed {written_rows}; "
                "blank or malformed CSV records may be present"
            )
        matrix.flush()
        del matrix
        os.replace(matrix_tmp, matrix_path)
        os.replace(seq_ids_tmp, seq_ids_path)
    except BaseException:
        matrix_tmp.unlink(missing_ok=True)
        seq_ids_tmp.unlink(missing_ok=True)
        raise
    return matrix_path, seq_ids_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for source_path in args.paths:
        matrix_path, seq_ids_path = convert_embedding_csv(
            source_path,
            overwrite=args.overwrite,
        )
        print(f"{source_path}: {matrix_path} + {seq_ids_path}")


if __name__ == "__main__":
    main()

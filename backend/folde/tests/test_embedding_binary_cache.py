import json
import os
from pathlib import Path

import numpy as np

from folde.data import try_load_binary_embedding_cache
from folde.scripts.convert_embedding_csv_to_binary import convert_embedding_csv


def _write_embedding_csv(path: Path) -> None:
    rows = [
        ("A1G", [1.25, -2.5, 3.75]),
        ("C2D", [4.5, 5.25, -6.0]),
    ]
    path.write_text(
        "seq_id,embedding\n"
        + "".join(
            f'{seq_id},"{json.dumps(values).replace(chr(34), chr(34) * 2)}"\n'
            for seq_id, values in rows
        )
    )


def test_binary_embedding_cache_round_trip_and_staleness(tmp_path: Path) -> None:
    csv_path = tmp_path / "embeddings.csv"
    _write_embedding_csv(csv_path)

    matrix_path, seq_ids_path = convert_embedding_csv(csv_path)
    loaded = try_load_binary_embedding_cache(csv_path)

    assert loaded is not None
    assert seq_ids_path.read_text().splitlines() == ["A1G", "C2D"]
    assert loaded["seq_id"].tolist() == ["A1G", "C2D"]
    assert loaded["embedding"].iloc[0].dtype == np.float32
    np.testing.assert_array_equal(loaded["embedding"].iloc[1], [4.5, 5.25, -6.0])

    newer_time = max(matrix_path.stat().st_mtime, seq_ids_path.stat().st_mtime) + 1
    os.utime(csv_path, (newer_time, newer_time))
    assert try_load_binary_embedding_cache(csv_path) is None

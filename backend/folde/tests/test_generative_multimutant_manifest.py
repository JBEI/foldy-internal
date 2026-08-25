"""Tests for Phase 0 dataset auditing and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from folde.benchmarks.multimutant_data import audit_dataset, sha256_file, write_manifest
from folde.benchmarks.schemas import GenerativeMultimutantBenchmarkConfig

FOLDE_DIR = Path(__file__).resolve().parents[1]


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"FolDE phase zero\n")

    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_dataset_reports_depth_coverage_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "mutant": ["A1C", "A1D", "C2A", "A1C:C2A", "A1D:C2A", "not-a-mutant"],
            "DMS_score": [0.1, 0.2, 0.3, 1.0, float("nan"), 2.0],
        }
    ).to_csv(path, index=False)

    result = audit_dataset(path, "AC")

    assert result["row_count"] == 6
    assert result["finite_score_count"] == 5
    assert result["invalid_mutant_count"] == 1
    assert result["reference_mismatch_count"] == 0
    assert result["mutation_depth_counts"] == {"1": 3, "2": 2}
    assert result["maximum_mutation_depth"] == 2
    assert result["mutated_positions"] == [1, 2]
    assert result["multimutant_component_coverage"] == 1.0
    assert result["standard_shell_possible_counts"] == {"1": 38, "2": 361}


def test_write_manifest_is_stable_and_replaces_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    write_manifest({"schema_version": "1.0", "datasets": []}, output)
    first = output.read_text()
    write_manifest({"schema_version": "1.0", "datasets": []}, output)

    assert output.read_text() == first
    assert not output.with_suffix(".json.tmp").exists()


def test_checked_in_phase_zero_artifacts_are_complete() -> None:
    manifest_path = FOLDE_DIR / "model_evals/260811-generative-multimutant-manifest.json"
    schema_path = FOLDE_DIR / "benchmarks/generative_multimutant_config.schema.json"
    manifest = json.loads(manifest_path.read_text())
    schema = json.loads(schema_path.read_text())

    assert manifest["schema_version"] == "1.0"
    assert manifest["metadata_dataset_count"] == 217
    assert manifest["activity_dataset_count"] == 217
    assert len(manifest["datasets"]) == 217
    assert manifest["metadata_without_activity"] == []
    assert manifest["activity_without_metadata"] == []
    assert manifest["unmatched_feature_files"] == []
    assert all(len(dataset["activity_sha256"]) == 64 for dataset in manifest["datasets"])
    assert all(len(feature["sha256"]) == 64 for feature in manifest["feature_files"])
    assert schema["title"] == "GenerativeMultimutantBenchmarkConfig"
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    current_schema = json.loads(
        json.dumps(GenerativeMultimutantBenchmarkConfig.model_json_schema())
    )
    assert schema == current_schema

"""Tests for the Phase 0 generative benchmark configuration contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from folde.benchmarks.schemas import GenerativeMultimutantBenchmarkConfig


def _valid_config() -> dict:
    return {
        "name": "generative-folde-v1",
        "seed": 42,
        "datasets": [
            {
                "dms_id": "SPG1_STRSG_Olson_2014",
                "protocol": "single_to_double",
                "coverage_policy": "exhaustive",
                "allowed_positions": list(range(228, 283)),
                "available_depths": [1, 2],
                "target_depths": [2],
            }
        ],
        "campaign": {"proposal_budget": 10_000, "round_size": 16},
        "features": {
            "embedding_model": "esmc_300m",
            "embedding_revision": "esmc-300m-2024-12",
            "naturalness_model": "esmc_600m",
            "naturalness_revision": "esmc-600m-2024-12",
        },
        "generator": {
            "name": "esmc_iterative_mask",
            "model": "esmc_300m",
            "revision": "esmc-300m-2024-12",
            "raw_sampling_cap": 100_000,
        },
        "folde": {
            "zero_shot_model_name": "NaturalnessZeroShotModel",
            "few_shot_model_name": "TorchMLPFewShotModel",
            "acquisition": "constantliar",
        },
        "selection": {"diversity_metric": "mutation_jaccard"},
        "output": {"checkpoint_dir": "folde/model_evals/generative"},
    }


def test_valid_configuration_round_trips() -> None:
    config = GenerativeMultimutantBenchmarkConfig.model_validate(_valid_config())

    assert config.schema_version == "1.0"
    assert config.campaign.proposal_budget == 10_000
    assert config.datasets[0].target_depths == {2}


def test_proposal_budget_must_cover_round_size() -> None:
    raw = _valid_config()
    raw["campaign"] = {"proposal_budget": 15, "round_size": 16}

    with pytest.raises(ValidationError, match="proposal_budget must be at least round_size"):
        GenerativeMultimutantBenchmarkConfig.model_validate(raw)


def test_requested_shell_must_exist() -> None:
    raw = _valid_config()
    raw["datasets"][0]["target_depths"] = [3]

    with pytest.raises(ValidationError, match="requested mutation depths are absent"):
        GenerativeMultimutantBenchmarkConfig.model_validate(raw)


def test_release_run_rejects_unpinned_feature_revision() -> None:
    raw = _valid_config()
    raw["release_run"] = True
    raw["features"]["embedding_revision"] = "PIN_REQUIRED"

    with pytest.raises(ValidationError, match="release runs require pinned revisions"):
        GenerativeMultimutantBenchmarkConfig.model_validate(raw)


def test_sparse_replication_requires_library_constrained_policy() -> None:
    raw = _valid_config()
    raw["datasets"][0].update(
        {
            "protocol": "double_mutant_replication",
            "coverage_policy": "rejection_sampling",
        }
    )

    with pytest.raises(ValidationError, match="requires library_constrained"):
        GenerativeMultimutantBenchmarkConfig.model_validate(raw)


def test_nonstandard_generator_alphabet_is_rejected() -> None:
    raw = _valid_config()
    raw["generator"]["allowed_alphabet"] = ["A", "B"]

    with pytest.raises(ValidationError, match="nonstandard residues"):
        GenerativeMultimutantBenchmarkConfig.model_validate(raw)


def test_proposal_mix_requires_a_positive_weight() -> None:
    raw = _valid_config()
    raw["generator"]["proposal_mix"] = {"plm": 0.0, "adjacent": 0.0}

    with pytest.raises(ValidationError, match="at least one proposal_mix weight"):
        GenerativeMultimutantBenchmarkConfig.model_validate(raw)

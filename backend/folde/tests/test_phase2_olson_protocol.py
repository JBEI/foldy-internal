"""Phase 2 closed-world Olson benchmark tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from folde.benchmarks.feature_store import MemmapFeatureStore, prepare_memmap_feature_store
from folde.benchmarks.multimutant_metrics import campaign_metrics, paired_statistical_report
from folde.benchmarks.olson_protocol import (
    ARM_NAMES,
    OlsonProtocolConfig,
    run_olson_arm,
)
from folde.candidate_generation import GeneratorContext, LibraryConstrainedPLMGenerator


def _write_feature_csv(path: Path, seq_ids: list[str]) -> None:
    pd.DataFrame(
        {
            "seq_id": seq_ids,
            "embedding": [json.dumps([float(index), 1.0]) for index in range(len(seq_ids))],
        }
    ).to_csv(path, index=False)


def test_memmap_feature_store_preserves_identifiers_vectors_and_order(tmp_path: Path) -> None:
    source = tmp_path / "features.csv"
    _write_feature_csv(source, ["A1C", "A2C", "A1C_A2C"])
    metadata = prepare_memmap_feature_store(source, tmp_path / "store", chunk_size=2)
    store = MemmapFeatureStore(tmp_path / "store")

    assert metadata["row_count"] == 3
    assert metadata["embedding_dim"] == 2
    np.testing.assert_array_equal(
        store.get_array(["A1C_A2C", "A1C"]),
        np.asarray([[2.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_library_constrained_plm_generator_ranks_without_activity() -> None:
    generator = LibraryConstrainedPLMGenerator(
        ["A1C_A2C", "A1C_A3C", "A2C_A3C"],
        {"A1C_A2C": -3.0, "A1C_A3C": -1.0, "A2C_A3C": -2.0},
        model_name="fake-plm",
        model_revision="revision-1",
    )
    proposals = generator.generate(
        GeneratorContext(
            reference_sequence="AAA",
            allowed_positions=frozenset({1, 2, 3}),
            allowed_alphabet=frozenset({"C"}),
            min_mutation_depth=2,
            max_mutation_depth=2,
            proposal_budget=2,
            round_number=1,
            random_seed=7,
        )
    )

    assert [proposal.identity.seq_id for proposal in proposals] == ["A1C_A3C", "A2C_A3C"]
    assert all(
        proposal.metadata["coverage_policy"] == "library_constrained" for proposal in proposals
    )


def test_campaign_and_paired_metrics_match_hand_computation() -> None:
    metrics = campaign_metrics(
        [0.0, 1.0, 0.5, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        initial_measurement_count=2,
        measurement_batch_sizes=[2, 1, 1],
    )
    assert metrics["best_dms_score"] == 3.0
    assert metrics["simple_regret"] == 0.0
    assert metrics["area_under_best_found_curve"] == 1.5
    report = paired_statistical_report(
        {"plm_plus_folde": {1: 3.0, 2: 5.0}, "adjacent_folde": {1: 2.0, 2: 4.0}},
        bootstrap_samples=100,
        seed=11,
    )
    assert report["median_paired_difference"] == 1.0
    assert report["wins"] == 2
    assert report["ties"] == 0
    assert report["losses"] == 0


def test_protocol_a_runs_all_six_arms_and_resumes(tmp_path: Path) -> None:
    singles = ["A1C", "A2C", "A3C"]
    doubles = ["A1C_A2C", "A1C_A3C", "A2C_A3C"]
    seq_ids = [*singles, *doubles]
    source = tmp_path / "features.csv"
    _write_feature_csv(source, seq_ids)
    prepare_memmap_feature_store(source, tmp_path / "store", chunk_size=2)
    store = MemmapFeatureStore(tmp_path / "store")
    activity = pd.Series([1.0, 2.0, 1.5, 10.0, 4.0, 5.0], index=seq_ids, dtype=float)
    naturalness = {
        "A1C": -1.0,
        "A2C": -2.0,
        "A3C": -3.0,
        "A1C_A2C": -3.0,
        "A1C_A3C": -4.0,
        "A2C_A3C": -5.0,
    }
    config = OlsonProtocolConfig(
        simulation_seeds=(0,),
        initial_singles=2,
        round_size=1,
        rounds=1,
        proposal_budget=2,
    )
    results = {}
    for arm in ARM_NAMES:
        results[arm] = run_olson_arm(
            arm=arm,
            simulation_seed=0,
            config=config,
            reference_sequence="AAA",
            activity=activity,
            singles=singles,
            doubles=doubles,
            naturalness=naturalness,
            feature_store=store,
            output_dir=tmp_path / "results",
        )
        assert len(results[arm].rounds) == 1
        assert len(results[arm].measured_variants) == 3
        assert results[arm].rounds[0].proposal_pool_sha256

    pool_dir = tmp_path / "results" / "proposal_pools"
    with (
        np.load(pool_dir / "plm_only-seed-0-round-1.npz") as plm_only,
        np.load(pool_dir / "plm_plus_folde-seed-0-round-1.npz") as plm_plus_folde,
    ):
        assert plm_only.files == plm_plus_folde.files
        for column in plm_only.files:
            np.testing.assert_array_equal(plm_only[column], plm_plus_folde[column])

    resumed = run_olson_arm(
        arm="plm_plus_folde",
        simulation_seed=0,
        config=config,
        reference_sequence="AAA",
        activity=activity,
        singles=singles,
        doubles=doubles,
        naturalness=naturalness,
        feature_store=store,
        output_dir=tmp_path / "results",
    )
    assert resumed == results["plm_plus_folde"]

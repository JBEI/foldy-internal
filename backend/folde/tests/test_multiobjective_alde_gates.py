"""Tests for the multi-objective ALDE proposal and selection gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from folde.benchmarks.multiobjective_alde import (
    ALDE_ARMS,
    ALDEFeatureSpace,
    MultiObjectiveALDEConfig,
    MultiObjectiveOracle,
    _fast_nondominated_mask_2d,
    _naturalness_percentiles,
    analyze_hybrid_gate_campaigns,
    analyze_mlp_replication_campaigns,
    build_proposal_pool,
    predict_objective_ensemble,
    run_multiobjective_campaign,
    select_candidates,
)
from folde.multiobjective_data import MultiObjectiveDataset


def _dataset(n: int = 30) -> MultiObjectiveDataset:
    seq_ids = pd.Index([f"variant_{index:02d}" for index in range(n)], name="seq_id")
    x = np.linspace(0.0, 1.0, n)
    activity = pd.DataFrame(
        {
            "objective_a": x + 0.1 * np.sin(9 * x),
            "objective_b": 1.0 - x + 0.1 * np.cos(7 * x),
        },
        index=seq_ids,
    )
    embeddings = pd.Series(
        [np.asarray([xv, xv**2, np.sin(xv), np.cos(xv)], dtype=float) for xv in x],
        index=seq_ids,
    )
    naturalness = pd.DataFrame({"log_wt_marginal": -np.abs(x - 0.65)}, index=seq_ids)
    return MultiObjectiveDataset(
        activity_df=activity,
        embedding_series=embeddings,
        naturalness_df=naturalness,
        objectives=[],
        wt_sequence="AAAA",
        protein="SYNTHETIC",
    )


def _config() -> MultiObjectiveALDEConfig:
    return MultiObjectiveALDEConfig(
        simulation_seeds=(0,),
        initial_size=6,
        batch_size=2,
        rounds=2,
        proposal_budget=12,
        projection_dim=3,
        ensemble_size=4,
    )


def test_fast_pareto_mask_handles_dominance_and_duplicates() -> None:
    points = np.asarray([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [1.5, 1.5]])
    assert _fast_nondominated_mask_2d(points).tolist() == [False, True, True, True, True]


def test_multiobjective_oracle_reveals_only_requested_rows() -> None:
    dataset = _dataset()
    oracle = MultiObjectiveOracle(dataset.activity_df)
    revealed = oracle.measure(["variant_00", "variant_01"])
    assert list(revealed.index) == ["variant_00", "variant_01"]
    assert oracle.measured_ids == ("variant_00", "variant_01")
    assert oracle.calls == (("variant_00", "variant_01"),)
    try:
        oracle.measure(["variant_00"])
    except ValueError as error:
        assert "already measured" in str(error)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("duplicate measurement was accepted")


def test_mixed_proposal_pool_is_deterministic_and_budgeted() -> None:
    dataset = _dataset()
    config = _config()
    features = ALDEFeatureSpace(dataset, config)
    measured = dataset.activity_df.iloc[:6]
    kwargs = {
        "kind": "mixed",
        "all_seq_ids": features.seq_ids,
        "measured_values": measured,
        "feature_space": features,
        "budget": 12,
        "random_seed": 19,
        "proposal_mix": config.proposal_mix,
    }
    first = build_proposal_pool(**kwargs)
    second = build_proposal_pool(**kwargs)
    assert first == second
    assert len(first[0]) == 12
    assert len(first[0]) == len(set(first[0]))
    assert not set(first[0]) & set(measured.index)


def test_hybrid_selectors_are_deterministic_and_veto_low_naturalness() -> None:
    dataset = _dataset()
    config = _config()
    features = ALDEFeatureSpace(dataset, config)
    candidate_ids = list(features.seq_ids)
    rng = np.random.default_rng(22)
    predictions = rng.normal(size=(len(candidate_ids), config.ensemble_size, 2))
    kwargs = {
        "predictions": predictions,
        "candidate_ids": candidate_ids,
        "feature_space": features,
        "batch_size": config.batch_size,
        "random_seed": 44,
    }
    first, _ = select_candidates(selector="hybrid_soft", **kwargs)
    second, _ = select_candidates(selector="hybrid_soft", **kwargs)
    assert first == second

    vetoed, _ = select_candidates(selector="hybrid_veto", hybrid_veto_quantile=0.25, **kwargs)
    percentiles = dict(zip(candidate_ids, _naturalness_percentiles(candidate_ids, features)))
    assert all(percentiles[seq_id] >= 0.25 for seq_id in vetoed)


def test_torch_mlp_predictor_retrains_on_revealed_measurements() -> None:
    dataset = _dataset()
    config = MultiObjectiveALDEConfig(
        simulation_seeds=(0,),
        initial_size=6,
        batch_size=2,
        rounds=1,
        proposal_budget=12,
        projection_dim=3,
        ensemble_size=3,
        ranker_type="torch_mlp",
        mlp_hidden_dims=(4,),
        mlp_dropout=0.0,
        mlp_pretrain_epochs=1,
        mlp_train_epochs=1,
        mlp_train_patience=1,
    )
    features = ALDEFeatureSpace(dataset, config)
    predictions = predict_objective_ensemble(
        measured_values=dataset.activity_df.iloc[:6],
        candidate_ids=list(dataset.activity_df.index[6:18]),
        feature_space=features,
        config=config,
        random_seed=17,
        dataset=dataset,
    )
    assert predictions.shape == (12, 3, 2)
    assert np.isfinite(predictions).all()


def test_all_gate_arms_checkpoint_resume_and_share_initial_mixed_pool(tmp_path: Path) -> None:
    dataset = _dataset()
    config = _config()
    features = ALDEFeatureSpace(dataset, config)
    results = []
    for arm in ALDE_ARMS:
        result = run_multiobjective_campaign(
            dataset=dataset,
            arm=arm,
            simulation_seed=0,
            config=config,
            feature_space=features,
            output_dir=tmp_path / "run",
        )
        results.append(result)
        assert len(result.rounds) == 2
        assert len(result.measured_seq_ids) == 10
        assert all(len(record.selected_seq_ids) == 2 for record in result.rounds)
        assert all(record.proposal_pool_sha256 for record in result.rounds)

    mixed_paths = [
        tmp_path / "run" / "proposal_pools" / f"SYNTHETIC-{arm}-seed-0-round-1.npz"
        for arm in (
            "mixed_parego",
            "mixed_fixed",
            "mixed_plm_only",
            "mixed_random",
            "mixed_hybrid_soft25",
            "mixed_hybrid_veto25",
        )
    ]
    with np.load(mixed_paths[0]) as reference:
        for path in mixed_paths[1:]:
            with np.load(path) as candidate:
                assert reference.files == candidate.files
                for column in reference.files:
                    np.testing.assert_array_equal(reference[column], candidate[column])

    resumed = run_multiobjective_campaign(
        dataset=dataset,
        arm="mixed_parego",
        simulation_seed=0,
        config=config,
        feature_space=features,
        output_dir=tmp_path / "run",
    )
    assert resumed == results[0]
    report = analyze_hybrid_gate_campaigns(results)
    assert set(report["gates"]) >= {
        "proposal_gate",
        "selector_gate",
        "end_to_end_gate",
        "full_pool_noninferiority_gate",
        "front_coverage_gate",
        "hybrid_selector_gate",
        "hybrid_end_to_end_gate",
        "hybrid_parego_retention_gate",
        "hybrid_full_pool_noninferiority_gate",
    }
    mlp_report = analyze_mlp_replication_campaigns(
        [
            result
            for result in results
            if result.arm
            in {"mixed_parego", "mixed_hybrid_veto25", "mixed_plm_only", "mixed_random"}
        ]
    )
    assert set(mlp_report["gates"]) >= {
        "veto_increment_gate",
        "learned_selector_gate",
        "mlp_pipeline_authorized",
    }


def test_fresh_rerun_reproduces_selections_and_pool_records(tmp_path: Path) -> None:
    dataset = _dataset()
    config = _config()
    features = ALDEFeatureSpace(dataset, config)
    first = run_multiobjective_campaign(
        dataset=dataset,
        arm="mixed_parego",
        simulation_seed=0,
        config=config,
        feature_space=features,
        output_dir=tmp_path / "first",
        resume=False,
    )
    second = run_multiobjective_campaign(
        dataset=dataset,
        arm="mixed_parego",
        simulation_seed=0,
        config=config,
        feature_space=features,
        output_dir=tmp_path / "second",
        resume=False,
    )
    assert [record.selected_seq_ids for record in first.rounds] == [
        record.selected_seq_ids for record in second.rounds
    ]
    for round_number in (1, 2):
        first_path = (
            tmp_path
            / "first"
            / "proposal_pools"
            / f"SYNTHETIC-mixed_parego-seed-0-round-{round_number}.npz"
        )
        second_path = (
            tmp_path
            / "second"
            / "proposal_pools"
            / f"SYNTHETIC-mixed_parego-seed-0-round-{round_number}.npz"
        )
        with np.load(first_path) as left, np.load(second_path) as right:
            for column in left.files:
                np.testing.assert_array_equal(left[column], right[column])

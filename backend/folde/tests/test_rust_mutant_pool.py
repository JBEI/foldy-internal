import itertools

import numpy as np
import pandas as pd

from folde.campaign import CampaignWorldState, _run_single_simulation
from folde.rust_mutant_pool import (
    get_mutant_pool_native,
    get_mutant_pool_python,
    native_available,
)
from folde.types import FolDEModelConfig


def _make_world_state(
    seq_ids: list[str], one_mutation_at_a_time: bool = True
) -> CampaignWorldState:
    activity_series = pd.Series(np.arange(len(seq_ids), dtype=float), index=seq_ids)
    naturalness_df = pd.DataFrame({"log_wt_marginal": np.arange(len(seq_ids))}, index=seq_ids)
    embedding_series = pd.Series([np.array([float(i)]) for i in range(len(seq_ids))], index=seq_ids)
    return CampaignWorldState(
        activity_series,
        naturalness_df,
        embedding_series,
        one_mutation_at_a_time=one_mutation_at_a_time,
    )


def _generate_seq_ids() -> list[str]:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_ids = ["WT", "HOM-alpha-1"]
    for locus in range(1, 9):
        for amino_acid in amino_acids[:8]:
            if amino_acid != "A":
                seq_ids.append(f"A{locus}{amino_acid}")
    for left, right in itertools.combinations(seq_ids[2:20], 2):
        left_locus = int(left[1:-1])
        right_locus = int(right[1:-1])
        if left_locus != right_locus:
            seq_ids.append("_".join(sorted([left, right], key=lambda allele: int(allele[1:-1]))))
    return list(dict.fromkeys(seq_ids))


def test_native_mutant_pool_matches_python_edge_cases():
    assert native_available(), "Run backend/folde/rust_ext/build.sh before native tests"

    seq_ids = [
        "WT",
        "A1C",
        "A1D",
        "A2C",
        "A1C_A2C",
        "A1D_A2C",
        "A2C_A3D",
        "A1C_A2C_A3D",
        "HOM-example",
    ]
    measured_seq_ids = ["A1C", "A1C_A2C"]

    assert get_mutant_pool_native(seq_ids, measured_seq_ids) == get_mutant_pool_python(
        seq_ids, measured_seq_ids
    )


def test_native_mutant_pool_matches_python_generated_inputs():
    assert native_available(), "Run backend/folde/rust_ext/build.sh before native tests"

    seq_ids = _generate_seq_ids()
    measured_sets = [
        [],
        ["A1C"],
        ["WT"],
        ["A1C", "A2D", "A1C_A3E"],
        seq_ids[3:18:3],
    ]

    for measured_seq_ids in measured_sets:
        assert get_mutant_pool_native(seq_ids, measured_seq_ids) == get_mutant_pool_python(
            seq_ids, measured_seq_ids
        )


def test_campaign_world_state_uses_identical_mutant_pool_output():
    assert native_available(), "Run backend/folde/rust_ext/build.sh before native tests"

    seq_ids = _generate_seq_ids()
    measured_seq_ids = ["A1C", "A2D", "A1C_A3E"]
    world_state = _make_world_state(seq_ids, one_mutation_at_a_time=True)
    world_state.measure_variant_activities(measured_seq_ids)

    assert world_state.get_mutant_pool() == get_mutant_pool_python(seq_ids, measured_seq_ids)


def test_campaign_world_state_unrestricted_mode_is_unchanged():
    seq_ids = _generate_seq_ids()
    measured_seq_ids = ["A1C", "A2D", "A1C_A3E"]
    world_state = _make_world_state(seq_ids, one_mutation_at_a_time=False)
    world_state.measure_variant_activities(measured_seq_ids)

    assert world_state.get_mutant_pool() == [
        seq_id for seq_id in seq_ids if seq_id not in measured_seq_ids
    ]


def test_single_simulation_runs_with_native_one_mutation_pool():
    assert native_available(), "Run backend/folde/rust_ext/build.sh before native tests"

    seq_ids = [
        "A1C",
        "A1D",
        "A2C",
        "A2D",
        "A3C",
        "A3D",
        "A4C",
        "A4D",
        "A1C_A2C",
        "A1D_A2D",
        "A2C_A3C",
        "A2D_A3D",
        "A3C_A4C",
        "A3D_A4D",
        "A1C_A4C",
        "A1D_A4D",
    ]
    activity_series = pd.Series(np.linspace(0.0, 1.0, len(seq_ids)), index=seq_ids)
    naturalness_df = pd.DataFrame(
        {"log_wt_marginal": np.linspace(-2.0, 2.0, len(seq_ids))},
        index=seq_ids,
    )
    embedding_series = pd.Series(
        [np.array([float(i), float(i + 1)]) for i in range(len(seq_ids))],
        index=seq_ids,
    )
    config = FolDEModelConfig(
        name="native-mutant-pool-smoke",
        one_mutation_at_a_time=True,
        naturalness_model_id="mock-naturalness",
        embedding_model_id="mock-embedding",
        zero_shot_model_name="RandomZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="RandomFewShotModel",
        few_shot_model_params={},
    )

    result = _run_single_simulation(
        available_seq_ids=seq_ids[:10],
        entire_activity_series=activity_series,
        entire_naturalness_df=naturalness_df,
        entire_embedding_series=embedding_series,
        round_size=1,
        config=config,
        random_seed=42,
        wt_aa_seq="AAAA",
        max_rounds=2,
    )

    assert result.rounds == 2
    assert result.variant_pool_size == 10
    assert len(result.round_metrics) == 2
    assert len(result.mutant_metrics) == 2

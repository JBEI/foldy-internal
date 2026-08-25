"""Integration test for candidate-pool injection into the production campaign loop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from folde.campaign import _run_single_simulation
from folde.candidate_generation import MixedCandidatePoolStrategy, UniformShellGenerator
from folde.types import FolDEModelConfig


def test_campaign_scores_only_injected_pool_and_saves_provenance() -> None:
    seq_ids = ["A1C", "A2C", "A3C", "A1C_A2C", "A1C_A3C", "A2C_A3C"]
    activity = pd.Series(
        [1.0, 2.0, 1.5, 10.0, 4.0, 5.0],
        index=seq_ids,
        dtype=float,
    )
    naturalness = pd.DataFrame({"log_wt_marginal": np.arange(len(seq_ids))}, index=seq_ids)
    embeddings = pd.Series(
        [np.asarray([index, 1.0], dtype=float) for index in range(len(seq_ids))],
        index=seq_ids,
    )
    generator = UniformShellGenerator(seq_ids)
    strategy = MixedCandidatePoolStrategy({generator.name: generator}, {generator.name: 1.0})
    config = FolDEModelConfig(
        name="injected-pool",
        naturalness_model_id="test",
        embedding_model_id="test",
        zero_shot_model_name="NaturalnessZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="NaturalnessFewShotModel",
        few_shot_model_params={},
    )

    result = _run_single_simulation(
        seq_ids,
        activity,
        naturalness,
        embeddings,
        round_size=1,
        config=config,
        random_seed=17,
        wt_aa_seq="AAA",
        max_rounds=1,
        candidate_pool_strategy=strategy,
        proposal_budget=2,
        candidate_min_mutation_depth=2,
        candidate_max_mutation_depth=2,
    )

    provenance = result.round_metrics[0].misc
    proposal_ids = {proposal["identity"]["seq_id"] for proposal in provenance["proposal_pool"]}
    assert len(proposal_ids) == 2
    assert all(seq_id.count("_") == 1 for seq_id in proposal_ids)
    assert set(provenance["selected_seq_ids"]) <= proposal_ids
    assert provenance["oracle_lookup"] == provenance["selected_seq_ids"]
    assert provenance["candidate_pool_strategy"] == "mixed"

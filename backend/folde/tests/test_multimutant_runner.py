"""Synthetic integration tests for the Phase 1 campaign boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from folde.benchmarks.multimutant_oracle import ProteinGymFitnessOracle
from folde.benchmarks.synthetic_multimutant import (
    ProposalScorer,
    ProposalScoreScorer,
    run_synthetic_campaign,
)
from folde.candidate_generation import (
    AdjacentGenerator,
    CandidateGenerator,
    CandidateProposal,
    MeasuredVariant,
    MixedCandidatePoolStrategy,
    UniformShellGenerator,
)

REFERENCE = "AAA"
ACTIVITIES = pd.Series(
    {
        "WT": 0.0,
        "A1C": 1.0,
        "A2C": 2.0,
        "A3C": 1.5,
        "A1C_A2C": 10.0,
        "A1C_A3C": 4.0,
        "A2C_A3C": 5.0,
    }
)


def _oracle() -> ProteinGymFitnessOracle:
    return ProteinGymFitnessOracle(REFERENCE, ACTIVITIES)


def _strategy(generator: CandidateGenerator) -> MixedCandidatePoolStrategy:
    return MixedCandidatePoolStrategy({generator.name: generator}, {generator.name: 1.0})


class RecordingScorer(ProposalScorer):
    def __init__(self) -> None:
        self.seen_proposal_ids: list[set[str]] = []
        self.seen_measured_ids: list[set[str]] = []

    def score(
        self,
        proposals: Sequence[CandidateProposal],
        measured_variants: Sequence[MeasuredVariant],
    ) -> Mapping[str, float]:
        self.seen_proposal_ids.append({proposal.identity.seq_id for proposal in proposals})
        self.seen_measured_ids.append(
            {measurement.identity.seq_id for measurement in measured_variants}
        )
        return {
            proposal.identity.seq_id: float(proposal.identity.seq_id == "A1C_A2C")
            for proposal in proposals
        }


def test_long_jump_is_immediate_and_only_selected_activity_is_revealed() -> None:
    oracle = _oracle()
    scorer = RecordingScorer()
    generator = UniformShellGenerator(ACTIVITIES.index.tolist())

    result = run_synthetic_campaign(
        oracle=oracle,
        strategy=_strategy(generator),
        scorer=scorer,
        reference_sequence=REFERENCE,
        allowed_positions=frozenset({1, 2, 3}),
        allowed_alphabet=frozenset({"C"}),
        min_mutation_depth=2,
        max_mutation_depth=2,
        proposal_budget=3,
        round_size=1,
        rounds=1,
        seed=7,
    )

    record = result.rounds[0]
    assert "A1C_A2C" in scorer.seen_proposal_ids[0]
    assert scorer.seen_measured_ids[0] == {"WT"}
    assert record.selected_seq_ids == ("A1C_A2C",)
    assert {measurement.identity.seq_id for measurement in result.measured_variants} == {
        "WT",
        "A1C_A2C",
    }
    assert oracle.lookup_calls == (("WT",), ("A1C_A2C",))
    assert len(record.proposals) == 3


def test_adjacent_generator_cannot_reach_double_from_wt() -> None:
    generator = AdjacentGenerator(ACTIVITIES.index.tolist())
    oracle = _oracle()
    scorer = RecordingScorer()

    result = run_synthetic_campaign(
        oracle=oracle,
        strategy=_strategy(generator),
        scorer=scorer,
        reference_sequence=REFERENCE,
        allowed_positions=frozenset({1, 2, 3}),
        allowed_alphabet=frozenset({"C"}),
        min_mutation_depth=1,
        max_mutation_depth=2,
        proposal_budget=3,
        round_size=1,
        rounds=1,
        seed=7,
    )

    assert all(proposal.identity.mutation_depth == 1 for proposal in result.rounds[0].proposals)


def test_checkpoint_resume_matches_uninterrupted_campaign(tmp_path: Path) -> None:
    strategy = _strategy(UniformShellGenerator(ACTIVITIES.index.tolist()))
    common = {
        "strategy": strategy,
        "scorer": ProposalScoreScorer(),
        "reference_sequence": REFERENCE,
        "allowed_positions": frozenset({1, 2, 3}),
        "allowed_alphabet": frozenset({"C"}),
        "min_mutation_depth": 2,
        "max_mutation_depth": 2,
        "proposal_budget": 3,
        "round_size": 1,
        "seed": 19,
    }
    uninterrupted = run_synthetic_campaign(oracle=_oracle(), rounds=2, **common)
    checkpoint_path = tmp_path / "campaign.json"
    run_synthetic_campaign(oracle=_oracle(), rounds=1, checkpoint_path=checkpoint_path, **common)
    resumed = run_synthetic_campaign(
        oracle=_oracle(),
        rounds=2,
        checkpoint_path=checkpoint_path,
        resume=True,
        **common,
    )

    assert resumed == uninterrupted
    assert len({variant.identity.seq_id for variant in resumed.measured_variants}) == 3

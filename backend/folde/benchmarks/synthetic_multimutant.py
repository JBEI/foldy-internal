"""Small deterministic campaign harness for testing generator/oracle boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from folde.benchmarks.multimutant_oracle import ProteinGymFitnessOracle
from folde.candidate_generation.base import (
    CandidatePoolStrategy,
    CandidateProposal,
    GeneratorContext,
    MeasuredVariant,
)
from folde.candidate_generation.strategy import derive_component_seed


class ProposalScorer(Protocol):
    def score(
        self,
        proposals: Sequence[CandidateProposal],
        measured_variants: Sequence[MeasuredVariant],
    ) -> Mapping[str, float]: ...


class SyntheticRoundRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_number: int = Field(gt=0)
    proposals: tuple[CandidateProposal, ...]
    proposal_scores: dict[str, float]
    selected_seq_ids: tuple[str, ...]
    revealed_measurements: tuple[MeasuredVariant, ...]


class SyntheticCampaignCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    seed: int
    measured_variants: tuple[MeasuredVariant, ...] = ()
    rounds: tuple[SyntheticRoundRecord, ...] = ()


class ProposalScoreScorer:
    """Test scorer that uses proposal scores and never receives an oracle."""

    def score(
        self,
        proposals: Sequence[CandidateProposal],
        measured_variants: Sequence[MeasuredVariant],
    ) -> Mapping[str, float]:
        del measured_variants
        return {
            proposal.identity.seq_id: (
                proposal.proposal_score if proposal.proposal_score is not None else 0.0
            )
            for proposal in proposals
        }


def _write_checkpoint(checkpoint: SyntheticCampaignCheckpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(checkpoint.model_dump_json(indent=2) + "\n")
    temporary.replace(path)


def run_synthetic_campaign(
    *,
    oracle: ProteinGymFitnessOracle,
    strategy: CandidatePoolStrategy,
    scorer: ProposalScorer,
    reference_sequence: str,
    allowed_positions: frozenset[int],
    allowed_alphabet: frozenset[str],
    min_mutation_depth: int,
    max_mutation_depth: int,
    proposal_budget: int,
    round_size: int,
    rounds: int,
    seed: int,
    initial_seq_ids: Sequence[str] = ("WT",),
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> SyntheticCampaignCheckpoint:
    """Run a compact campaign whose only activity reads cross ``oracle.measure``."""
    if proposal_budget < round_size:
        raise ValueError("proposal_budget must be at least round_size")
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = SyntheticCampaignCheckpoint.model_validate_json(checkpoint_path.read_text())
        if checkpoint.seed != seed:
            raise ValueError("checkpoint seed does not match requested seed")
        if len(checkpoint.rounds) > rounds:
            raise ValueError("checkpoint contains more rounds than requested")
        oracle.restore(checkpoint.measured_variants)
    else:
        oracle.measure(initial_seq_ids, round_number=0)
        checkpoint = SyntheticCampaignCheckpoint(
            seed=seed,
            measured_variants=oracle.measured_variants,
        )

    records = list(checkpoint.rounds)
    for round_number in range(len(records) + 1, rounds + 1):
        context = GeneratorContext(
            reference_sequence=reference_sequence,
            measured_variants=oracle.measured_variants,
            allowed_positions=allowed_positions,
            allowed_alphabet=allowed_alphabet,
            min_mutation_depth=min_mutation_depth,
            max_mutation_depth=max_mutation_depth,
            proposal_budget=proposal_budget,
            round_number=round_number,
            random_seed=derive_component_seed(seed, "synthetic", round_number),
        )
        proposals = tuple(strategy.build_pool(context))
        if len(proposals) < round_size:
            raise ValueError(
                f"proposal pool contains {len(proposals)} variants, fewer than round_size"
            )
        scores = dict(scorer.score(proposals, oracle.measured_variants))
        proposal_ids = {proposal.identity.seq_id for proposal in proposals}
        if set(scores) != proposal_ids:
            raise ValueError("scorer must return exactly one score for every proposed variant")
        selected = tuple(sorted(scores, key=lambda seq_id: (-scores[seq_id], seq_id))[:round_size])
        before_measurement_count = len(oracle.measured_variants)
        oracle.measure(selected, round_number=round_number)
        revealed = oracle.measured_variants[before_measurement_count:]
        records.append(
            SyntheticRoundRecord(
                round_number=round_number,
                proposals=proposals,
                proposal_scores=scores,
                selected_seq_ids=selected,
                revealed_measurements=revealed,
            )
        )
        checkpoint = SyntheticCampaignCheckpoint(
            seed=seed,
            measured_variants=oracle.measured_variants,
            rounds=tuple(records),
        )
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint, checkpoint_path)
    return checkpoint

"""Closed-world candidate ranking by assay-independent PLM scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from folde.candidate_generation.base import (
    CandidateProposal,
    GeneratorContext,
    proposal_for_identity,
    variant_identity_from_seq_id,
)
from folde.candidate_generation.validation import validate_and_deduplicate_proposals


class LibraryConstrainedPLMGenerator:
    """Rank a measured library using precomputed, activity-independent PLM scores."""

    def __init__(
        self,
        eligible_seq_ids: Sequence[str],
        scores: Mapping[str, float],
        *,
        model_name: str,
        model_revision: str,
    ):
        self._eligible_seq_ids = tuple(eligible_seq_ids)
        self._scores = dict(scores)
        self._model_name = model_name
        self._model_revision = model_revision
        missing = set(self._eligible_seq_ids) - set(self._scores)
        if missing:
            raise ValueError(f"PLM scores missing for eligible variants: {sorted(missing)[:5]}")

    @property
    def name(self) -> str:
        return "library_constrained_plm"

    def generate(self, context: GeneratorContext) -> list[CandidateProposal]:
        measured_ids = {variant.identity.seq_id for variant in context.measured_variants}
        ranked = sorted(
            (seq_id for seq_id in self._eligible_seq_ids if seq_id not in measured_ids),
            key=lambda seq_id: (-self._scores[seq_id], seq_id),
        )
        proposals = [
            proposal_for_identity(
                variant_identity_from_seq_id(context.reference_sequence, seq_id),
                generator_name=self.name,
                generation_seed=context.random_seed,
                proposal_rank=rank,
                proposal_score=float(self._scores[seq_id]),
                metadata={
                    "coverage_policy": "library_constrained",
                    "score_semantics": "plm_log_naturalness",
                    "model_name": self._model_name,
                    "model_revision": self._model_revision,
                },
            )
            for rank, seq_id in enumerate(ranked[: context.proposal_budget], start=1)
        ]
        return validate_and_deduplicate_proposals(proposals, context)

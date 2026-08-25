"""Validation and canonical deduplication for generated proposal pools."""

from __future__ import annotations

from collections.abc import Sequence

from folde.candidate_generation.base import CandidateProposal, GeneratorContext


def validate_and_deduplicate_proposals(
    proposals: Sequence[CandidateProposal],
    context: GeneratorContext,
) -> list[CandidateProposal]:
    """Validate proposals without consulting activity values."""
    measured_ids = {variant.identity.seq_id for variant in context.measured_variants}
    seen_sequences: set[str] = set()
    accepted: list[CandidateProposal] = []
    for proposal in proposals:
        identity = proposal.identity
        if identity.seq_id in measured_ids or identity.sequence in seen_sequences:
            continue
        if not context.min_mutation_depth <= identity.mutation_depth <= context.max_mutation_depth:
            continue
        if not {sub.position for sub in identity.substitutions} <= context.allowed_positions:
            continue
        if not {sub.alternate_aa for sub in identity.substitutions} <= context.allowed_alphabet:
            continue
        seen_sequences.add(identity.sequence)
        accepted.append(proposal.model_copy(update={"proposal_rank": len(accepted) + 1}))
        if len(accepted) == context.proposal_budget:
            break
    return accepted

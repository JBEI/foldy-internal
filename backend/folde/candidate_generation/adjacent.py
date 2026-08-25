"""One-edit candidate generation compatible with the existing mutant pool."""

from __future__ import annotations

import numpy as np

from app.helpers.sequence_util import get_allele_set, is_homolog_seq_id
from folde.candidate_generation.base import (
    CandidateProposal,
    GeneratorContext,
    proposal_for_identity,
    variant_identity_from_seq_id,
)
from folde.candidate_generation.validation import validate_and_deduplicate_proposals
from folde.rust_mutant_pool import get_mutant_pool


class AdjacentGenerator:
    """Generate covered variants one allele edit from a measured parent."""

    def __init__(self, eligible_seq_ids: list[str]):
        self._eligible_seq_ids = tuple(eligible_seq_ids)

    @property
    def name(self) -> str:
        return "adjacent"

    def generate(self, context: GeneratorContext) -> list[CandidateProposal]:
        measured_ids = [variant.identity.seq_id for variant in context.measured_variants]
        candidates = get_mutant_pool(self._eligible_seq_ids, measured_ids)
        candidates = sorted(seq_id for seq_id in candidates if not is_homolog_seq_id(seq_id))
        if len(candidates) > context.proposal_budget:
            rng = np.random.default_rng(context.random_seed)
            chosen = rng.choice(len(candidates), size=context.proposal_budget, replace=False)
            candidates = [candidates[index] for index in sorted(chosen)]
        proposals = []
        for rank, seq_id in enumerate(candidates, start=1):
            alleles = get_allele_set(seq_id)
            parents = tuple(
                measured_id
                for measured_id in measured_ids
                if not is_homolog_seq_id(measured_id)
                and len(alleles ^ get_allele_set(measured_id)) == 1
            )
            proposals.append(
                proposal_for_identity(
                    variant_identity_from_seq_id(context.reference_sequence, seq_id),
                    generator_name=self.name,
                    generation_seed=context.random_seed,
                    proposal_rank=rank,
                    parent_seq_ids=parents,
                )
            )
        return validate_and_deduplicate_proposals(proposals, context)

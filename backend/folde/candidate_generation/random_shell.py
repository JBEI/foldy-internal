"""Uniform sampling from measured-library mutation shells."""

from __future__ import annotations

import numpy as np

from app.helpers.sequence_util import get_locus_from_allele_id
from folde.candidate_generation.base import (
    CandidateProposal,
    GeneratorContext,
    proposal_for_identity,
    variant_identity_from_seq_id,
)
from folde.candidate_generation.validation import validate_and_deduplicate_proposals


class UniformShellGenerator:
    def __init__(self, eligible_seq_ids: list[str]):
        self._eligible_seq_ids = tuple(eligible_seq_ids)

    @property
    def name(self) -> str:
        return "uniform_shell"

    def generate(self, context: GeneratorContext) -> list[CandidateProposal]:
        measured_ids = {variant.identity.seq_id for variant in context.measured_variants}
        eligible = []
        for seq_id in self._eligible_seq_ids:
            if seq_id in measured_ids:
                continue
            alleles = [] if seq_id == "WT" else seq_id.split("_")
            if not context.min_mutation_depth <= len(alleles) <= context.max_mutation_depth:
                continue
            if (
                not {get_locus_from_allele_id(allele) for allele in alleles}
                <= context.allowed_positions
            ):
                continue
            if not {allele[-1] for allele in alleles} <= context.allowed_alphabet:
                continue
            eligible.append(seq_id)
        rng = np.random.default_rng(context.random_seed)
        order = rng.permutation(len(eligible))[: context.proposal_budget]
        proposals = [
            proposal_for_identity(
                variant_identity_from_seq_id(context.reference_sequence, eligible[index]),
                generator_name=self.name,
                generation_seed=context.random_seed,
                proposal_rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        ]
        return validate_and_deduplicate_proposals(proposals, context)

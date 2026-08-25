"""Combinatorial baseline built from measured high-performing singles."""

from __future__ import annotations

from app.helpers.sequence_util import get_locus_from_allele_id
from folde.candidate_generation.base import (
    CandidateProposal,
    GeneratorContext,
    proposal_for_identity,
    variant_identity_from_seq_id,
)
from folde.candidate_generation.validation import validate_and_deduplicate_proposals


class TopSingleCombinationGenerator:
    """Rank covered multimutants by the sum of revealed single effects."""

    def __init__(
        self,
        eligible_seq_ids: list[str],
        naturalness_scores: dict[str, float] | None = None,
    ):
        self._eligible_seq_ids = tuple(eligible_seq_ids)
        self._naturalness_scores = naturalness_scores or {}

    @property
    def name(self) -> str:
        return "top_single_combination"

    def generate(self, context: GeneratorContext) -> list[CandidateProposal]:
        wt_activity = next(
            (
                measured.activity
                for measured in context.measured_variants
                if measured.identity.seq_id == "WT"
            ),
            0.0,
        )
        single_effects = {
            measured.identity.seq_id: measured.activity - wt_activity
            for measured in context.measured_variants
            if measured.identity.mutation_depth == 1
        }
        measured_ids = {measured.identity.seq_id for measured in context.measured_variants}
        ranked: list[tuple[float, float, str]] = []
        for seq_id in self._eligible_seq_ids:
            if seq_id in measured_ids:
                continue
            components = [] if seq_id == "WT" else seq_id.split("_")
            if not context.min_mutation_depth <= len(components) <= context.max_mutation_depth:
                continue
            if (
                not {get_locus_from_allele_id(component) for component in components}
                <= context.allowed_positions
            ):
                continue
            if not {component[-1] for component in components} <= context.allowed_alphabet:
                continue
            if not components or any(component not in single_effects for component in components):
                continue
            additive_score = sum(single_effects[component] for component in components)
            ranked.append(
                (
                    additive_score,
                    self._naturalness_scores.get(seq_id, float("-inf")),
                    seq_id,
                )
            )
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        proposals = []
        for rank, (score, _, seq_id) in enumerate(ranked[: context.proposal_budget], start=1):
            identity = variant_identity_from_seq_id(context.reference_sequence, seq_id)
            parents = tuple(
                f"{sub.reference_aa}{sub.position}{sub.alternate_aa}"
                for sub in identity.substitutions
            )
            proposals.append(
                proposal_for_identity(
                    identity,
                    generator_name=self.name,
                    generation_seed=context.random_seed,
                    proposal_rank=rank,
                    parent_seq_ids=parents,
                    proposal_score=score,
                    metadata={"score_semantics": "sum_of_measured_single_effects"},
                )
            )
        return validate_and_deduplicate_proposals(proposals, context)

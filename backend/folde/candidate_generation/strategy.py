"""Deterministic candidate-channel mixing and fallback handling."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from folde.candidate_generation.base import (
    CandidateGenerator,
    CandidateProposal,
    GeneratorContext,
)
from folde.candidate_generation.validation import validate_and_deduplicate_proposals


def derive_component_seed(base_seed: int, *components: object) -> int:
    """Derive a stable NumPy-compatible seed without Python's salted hash."""
    material = "\0".join([str(base_seed), *(str(component) for component in components)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def allocate_proposal_budget(weights: Mapping[str, float], budget: int) -> dict[str, int]:
    """Allocate integer counts using deterministic largest remainders."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not weights or any(weight < 0 for weight in weights.values()):
        raise ValueError("weights must be nonempty and nonnegative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    exact = {name: budget * weight / total for name, weight in weights.items()}
    allocation = {name: math.floor(value) for name, value in exact.items()}
    remainder = budget - sum(allocation.values())
    order = sorted(weights, key=lambda name: (-(exact[name] - allocation[name]), name))
    for name in order[:remainder]:
        allocation[name] += 1
    return allocation


class MixedCandidatePoolStrategy:
    """Mix generator channels while enforcing one global proposal budget."""

    def __init__(
        self,
        generators: Mapping[str, CandidateGenerator],
        weights: Mapping[str, float],
        fallback: CandidateGenerator | None = None,
    ):
        if set(generators) != set(weights):
            raise ValueError("generator and weight channel names must match")
        self._generators = dict(generators)
        self._weights = dict(weights)
        self._fallback = fallback

    @property
    def name(self) -> str:
        return "mixed"

    def build_pool(self, context: GeneratorContext) -> list[CandidateProposal]:
        allocations = allocate_proposal_budget(self._weights, context.proposal_budget)
        collected: list[CandidateProposal] = []
        for channel in self._generators:
            allocation = allocations[channel]
            if allocation == 0:
                continue
            child_context = context.model_copy(
                update={
                    "proposal_budget": allocation,
                    "random_seed": derive_component_seed(
                        context.random_seed, context.round_number, channel
                    ),
                }
            )
            for proposal in self._generators[channel].generate(child_context):
                metadata = dict(proposal.metadata)
                metadata["source_channel"] = channel
                collected.append(proposal.model_copy(update={"metadata": metadata}))

        accepted = validate_and_deduplicate_proposals(collected, context)
        shortfall = context.proposal_budget - len(accepted)
        if shortfall > 0 and self._fallback is not None:
            fallback_context = context.model_copy(
                update={
                    "proposal_budget": context.proposal_budget,
                    "random_seed": derive_component_seed(
                        context.random_seed, context.round_number, "fallback"
                    ),
                }
            )
            for proposal in self._fallback.generate(fallback_context):
                metadata = dict(proposal.metadata)
                metadata["source_channel"] = "fallback"
                collected.append(proposal.model_copy(update={"metadata": metadata}))
            accepted = validate_and_deduplicate_proposals(collected, context)
        return accepted

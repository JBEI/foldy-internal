"""Candidate generation interfaces and Version 1 baseline strategies."""

from folde.candidate_generation.adjacent import AdjacentGenerator
from folde.candidate_generation.base import (
    CandidateGenerator,
    CandidatePoolStrategy,
    CandidateProposal,
    GeneratorContext,
    MeasuredVariant,
    Substitution,
    VariantIdentity,
)
from folde.candidate_generation.library_scored import LibraryConstrainedPLMGenerator
from folde.candidate_generation.random_shell import UniformShellGenerator
from folde.candidate_generation.strategy import MixedCandidatePoolStrategy
from folde.candidate_generation.top_single import TopSingleCombinationGenerator

__all__ = [
    "AdjacentGenerator",
    "CandidateGenerator",
    "CandidatePoolStrategy",
    "CandidateProposal",
    "GeneratorContext",
    "LibraryConstrainedPLMGenerator",
    "MeasuredVariant",
    "MixedCandidatePoolStrategy",
    "Substitution",
    "TopSingleCombinationGenerator",
    "UniformShellGenerator",
    "VariantIdentity",
]

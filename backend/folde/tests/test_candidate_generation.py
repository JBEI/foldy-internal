"""Unit tests for Phase 1 candidate generators and pool strategies."""

from __future__ import annotations

import pytest

from folde.candidate_generation import (
    AdjacentGenerator,
    GeneratorContext,
    MeasuredVariant,
    MixedCandidatePoolStrategy,
    TopSingleCombinationGenerator,
    UniformShellGenerator,
)
from folde.candidate_generation.base import variant_identity_from_seq_id
from folde.candidate_generation.strategy import (
    allocate_proposal_budget,
    derive_component_seed,
)

REFERENCE = "AAA"
ELIGIBLE = [
    "A1C",
    "A1D",
    "A2C",
    "A3C",
    "A1C_A2C",
    "A1D_A2C",
    "A1C_A3C",
]


def _measurement(seq_id: str, activity: float, round_number: int = 0) -> MeasuredVariant:
    return MeasuredVariant(
        identity=variant_identity_from_seq_id(REFERENCE, seq_id),
        activity=activity,
        measured_round=round_number,
    )


def _context(
    *,
    measured: tuple[MeasuredVariant, ...] = (),
    minimum_depth: int = 1,
    maximum_depth: int = 2,
    budget: int = 10,
    seed: int = 42,
) -> GeneratorContext:
    return GeneratorContext(
        reference_sequence=REFERENCE,
        measured_variants=measured,
        allowed_positions=frozenset({1, 2, 3}),
        allowed_alphabet=frozenset({"C", "D"}),
        min_mutation_depth=minimum_depth,
        max_mutation_depth=maximum_depth,
        proposal_budget=budget,
        round_number=1,
        random_seed=seed,
    )


def test_variant_identity_is_canonical_and_reference_validated() -> None:
    identity = variant_identity_from_seq_id(REFERENCE, "A1C_A3C")

    assert identity.sequence == "CAC"
    assert identity.mutation_depth == 2
    assert [sub.position for sub in identity.substitutions] == [1, 3]
    assert len(identity.sequence_hash) == 64
    with pytest.raises(ValueError, match="Loci are not sorted"):
        variant_identity_from_seq_id(REFERENCE, "A3C_A1C")
    with pytest.raises(ValueError, match="does not correspond to WT"):
        variant_identity_from_seq_id(REFERENCE, "G1C")


def test_adjacent_requires_a_component_single_before_double() -> None:
    generator = AdjacentGenerator(ELIGIBLE)

    initial = generator.generate(_context(measured=(_measurement("WT", 0.0),)))
    assert {proposal.identity.seq_id for proposal in initial} == {
        "A1C",
        "A1D",
        "A2C",
        "A3C",
    }
    assert all(proposal.parent_seq_ids == ("WT",) for proposal in initial)

    after_single = generator.generate(
        _context(measured=(_measurement("WT", 0.0), _measurement("A1C", 1.0)))
    )
    assert "A1C_A2C" in {proposal.identity.seq_id for proposal in after_single}
    double = next(proposal for proposal in after_single if proposal.identity.seq_id == "A1C_A2C")
    assert double.parent_seq_ids == ("A1C",)


def test_uniform_shell_is_deterministic_and_enforces_budget() -> None:
    generator = UniformShellGenerator(ELIGIBLE)
    context = _context(minimum_depth=2, maximum_depth=2, budget=2, seed=9)

    first = generator.generate(context)
    second = generator.generate(context)

    assert first == second
    assert len(first) == 2
    assert all(proposal.identity.mutation_depth == 2 for proposal in first)
    assert len({proposal.identity.seq_id for proposal in first}) == 2


def test_top_single_combination_uses_only_revealed_single_effects() -> None:
    measured = (
        _measurement("WT", 1.0),
        _measurement("A1C", 4.0),
        _measurement("A1D", 4.0),
        _measurement("A2C", 3.0),
        _measurement("A3C", 2.0),
    )
    generator = TopSingleCombinationGenerator(
        ELIGIBLE,
        naturalness_scores={"A1C_A2C": 0.2, "A1D_A2C": 0.8},
    )

    proposals = generator.generate(
        _context(measured=measured, minimum_depth=2, maximum_depth=2, budget=3)
    )

    assert [proposal.identity.seq_id for proposal in proposals] == [
        "A1D_A2C",
        "A1C_A2C",
        "A1C_A3C",
    ]
    assert [proposal.proposal_score for proposal in proposals] == [5.0, 5.0, 4.0]
    assert all(
        proposal.metadata["score_semantics"] == "sum_of_measured_single_effects"
        for proposal in proposals
    )


def test_mixed_strategy_allocates_budget_and_records_source_channel() -> None:
    context = _context(budget=5)
    strategy = MixedCandidatePoolStrategy(
        generators={
            "adjacent": AdjacentGenerator(ELIGIBLE),
            "random": UniformShellGenerator(ELIGIBLE),
        },
        weights={"adjacent": 0.6, "random": 0.4},
        fallback=UniformShellGenerator(ELIGIBLE),
    )

    proposals = strategy.build_pool(context)

    assert len(proposals) == 5
    assert len({proposal.identity.sequence for proposal in proposals}) == 5
    assert all("source_channel" in proposal.metadata for proposal in proposals)


def test_budget_allocation_and_seed_derivation_are_stable() -> None:
    assert allocate_proposal_budget({"plm": 0.7, "adjacent": 0.2, "random": 0.1}, 11) == {
        "plm": 8,
        "adjacent": 2,
        "random": 1,
    }
    assert derive_component_seed(42, "dataset", 1, "generator") == derive_component_seed(
        42, "dataset", 1, "generator"
    )
    assert derive_component_seed(42, "dataset", 1, "generator") != derive_component_seed(
        42, "dataset", 1, "selector"
    )

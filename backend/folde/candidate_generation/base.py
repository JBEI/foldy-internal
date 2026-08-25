"""Shared types and protocols for candidate-pool construction."""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from app.helpers.sequence_util import (
    VALID_AMINO_ACIDS,
    allele_set_to_seq_id,
    get_locus_from_allele_id,
    maybe_get_seq_id_error_message,
    seq_id_to_seq,
)


class CandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Substitution(CandidateModel):
    reference_aa: str = Field(min_length=1, max_length=1)
    position: int = Field(gt=0)
    alternate_aa: str = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_residue_change(self) -> "Substitution":
        if self.reference_aa == self.alternate_aa:
            raise ValueError("a substitution must change the reference residue")
        return self


class VariantIdentity(CandidateModel):
    sequence: str = Field(min_length=1)
    sequence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seq_id: str = Field(min_length=1)
    substitutions: tuple[Substitution, ...]
    mutation_depth: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_identity(self) -> "VariantIdentity":
        if self.mutation_depth != len(self.substitutions):
            raise ValueError("mutation_depth must equal the number of substitutions")
        if hashlib.sha256(self.sequence.encode("ascii")).hexdigest() != self.sequence_hash:
            raise ValueError("sequence_hash does not match sequence")
        positions = [substitution.position for substitution in self.substitutions]
        if positions != sorted(positions) or len(positions) != len(set(positions)):
            raise ValueError("substitutions must use unique positions in canonical order")
        canonical_seq_id = allele_set_to_seq_id(
            {
                f"{substitution.reference_aa}{substitution.position}" f"{substitution.alternate_aa}"
                for substitution in self.substitutions
            }
        )
        if canonical_seq_id != self.seq_id:
            raise ValueError(f"seq_id is not canonical; expected {canonical_seq_id}")
        return self


class MeasuredVariant(CandidateModel):
    identity: VariantIdentity
    activity: float
    measured_round: int = Field(ge=0)


class CandidateProposal(CandidateModel):
    identity: VariantIdentity
    parent_seq_ids: tuple[str, ...] = ()
    generator_name: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    generator_checkpoint: str | None = None
    generation_seed: int
    proposal_rank: int = Field(gt=0)
    proposal_score: float | None = None
    proposal_log_probability: float | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class GeneratorContext(CandidateModel):
    reference_sequence: str = Field(min_length=1)
    measured_variants: tuple[MeasuredVariant, ...] = ()
    allowed_positions: frozenset[int]
    allowed_alphabet: frozenset[str]
    min_mutation_depth: int = Field(ge=0)
    max_mutation_depth: int = Field(ge=0)
    proposal_budget: int = Field(gt=0)
    round_number: int = Field(gt=0)
    random_seed: int

    @model_validator(mode="after")
    def validate_context(self) -> "GeneratorContext":
        if self.min_mutation_depth > self.max_mutation_depth:
            raise ValueError("min_mutation_depth must not exceed max_mutation_depth")
        if not self.allowed_positions:
            raise ValueError("allowed_positions must not be empty")
        if min(self.allowed_positions) < 1 or max(self.allowed_positions) > len(
            self.reference_sequence
        ):
            raise ValueError("allowed_positions must lie within the reference sequence")
        if not self.allowed_alphabet:
            raise ValueError("allowed_alphabet must not be empty")
        invalid_residues = self.allowed_alphabet - set(VALID_AMINO_ACIDS)
        if invalid_residues:
            raise ValueError(
                f"allowed_alphabet contains nonstandard residues: {sorted(invalid_residues)}"
            )
        return self


@runtime_checkable
class CandidateGenerator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, context: GeneratorContext) -> Sequence[CandidateProposal]: ...


@runtime_checkable
class CandidatePoolStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def build_pool(self, context: GeneratorContext) -> Sequence[CandidateProposal]: ...


def variant_identity_from_seq_id(reference_sequence: str, seq_id: str) -> VariantIdentity:
    """Build a canonical identity and validate every reference residue."""
    error = maybe_get_seq_id_error_message(reference_sequence, seq_id)
    if error is not None:
        raise ValueError(f"Invalid seq_id '{seq_id}': {error}")
    alleles = [] if seq_id == "WT" else seq_id.split("_")
    substitutions = tuple(
        Substitution(
            reference_aa=allele[0],
            position=get_locus_from_allele_id(allele),
            alternate_aa=allele[-1],
        )
        for allele in alleles
    )
    sequence = seq_id_to_seq(reference_sequence, seq_id)
    return VariantIdentity(
        sequence=sequence,
        sequence_hash=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        seq_id=seq_id,
        substitutions=substitutions,
        mutation_depth=len(substitutions),
    )


def proposal_for_identity(
    identity: VariantIdentity,
    *,
    generator_name: str,
    generation_seed: int,
    proposal_rank: int,
    parent_seq_ids: tuple[str, ...] = (),
    proposal_score: float | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> CandidateProposal:
    return CandidateProposal(
        identity=identity,
        parent_seq_ids=parent_seq_ids,
        generator_name=generator_name,
        generator_version="1.0",
        generation_seed=generation_seed,
        proposal_rank=proposal_rank,
        proposal_score=proposal_score,
        metadata={} if metadata is None else metadata,
    )

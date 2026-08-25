"""Versioned configuration models for the generative multimutant benchmark."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
UNPINNED_REVISIONS = frozenset({"", "PIN_REQUIRED", "latest", "main"})

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class StrictModel(BaseModel):
    """Base model that rejects misspelled or obsolete configuration fields."""

    model_config = ConfigDict(extra="forbid")


class CoveragePolicy(str, Enum):
    EXHAUSTIVE = "exhaustive"
    LIBRARY_CONSTRAINED = "library_constrained"
    REJECTION_SAMPLING = "rejection_sampling"


class BenchmarkProtocol(str, Enum):
    SINGLE_TO_DOUBLE = "single_to_double"
    SHELL_JUMP = "shell_jump"
    DOUBLE_MUTANT_REPLICATION = "double_mutant_replication"
    SPARSE_LANDSCAPE_STRESS = "sparse_landscape_stress"


class DatasetBenchmarkConfig(StrictModel):
    dms_id: str = Field(min_length=1)
    protocol: BenchmarkProtocol
    coverage_policy: CoveragePolicy
    allowed_positions: tuple[PositiveInt, ...]
    available_depths: frozenset[PositiveInt] | None = None
    target_depths: frozenset[PositiveInt] = frozenset({2})

    @model_validator(mode="after")
    def validate_dataset_protocol(self) -> "DatasetBenchmarkConfig":
        if len(set(self.allowed_positions)) != len(self.allowed_positions):
            raise ValueError("allowed_positions must be unique")
        if not self.target_depths:
            raise ValueError("target_depths must not be empty")
        if self.available_depths is not None and not self.target_depths <= self.available_depths:
            missing = sorted(self.target_depths - self.available_depths)
            raise ValueError(f"requested mutation depths are absent from the dataset: {missing}")
        if (
            self.protocol == BenchmarkProtocol.DOUBLE_MUTANT_REPLICATION
            and self.coverage_policy != CoveragePolicy.LIBRARY_CONSTRAINED
        ):
            raise ValueError("double-mutant replication requires library_constrained coverage")
        return self


class CampaignBenchmarkConfig(StrictModel):
    initial_measurements: PositiveInt = 32
    round_size: PositiveInt = 16
    rounds: PositiveInt = 5
    simulations: PositiveInt = 20
    proposal_budget: PositiveInt = 10_000

    @model_validator(mode="after")
    def validate_budgets(self) -> "CampaignBenchmarkConfig":
        if self.proposal_budget < self.round_size:
            raise ValueError("proposal_budget must be at least round_size")
        return self


class FeatureBenchmarkConfig(StrictModel):
    embedding_model: str = Field(min_length=1)
    embedding_revision: str = Field(min_length=1)
    embedding_column: str = "embedding"
    naturalness_model: str = Field(min_length=1)
    naturalness_revision: str = Field(min_length=1)
    naturalness_columns: tuple[str, ...] = ("log_wt_marginal",)


class GeneratorBenchmarkConfig(StrictModel):
    name: str = Field(min_length=1)
    model: str | None = None
    revision: str | None = None
    temperature: float = Field(default=1.0, gt=0)
    top_k: PositiveInt | None = None
    refinement_sweeps: NonNegativeInt = 0
    raw_sampling_cap: PositiveInt | None = None
    parent_policy: str = "mixed_elite"
    position_policy: str = "uniform"
    min_mutation_depth: PositiveInt = 2
    max_mutation_depth: PositiveInt = 2
    allowed_alphabet: frozenset[str] = Field(default_factory=lambda: STANDARD_AMINO_ACIDS)
    proposal_mix: dict[str, float] | None = None
    fallback_generator: str | None = None

    @model_validator(mode="after")
    def validate_generator(self) -> "GeneratorBenchmarkConfig":
        if self.min_mutation_depth > self.max_mutation_depth:
            raise ValueError("min_mutation_depth must not exceed max_mutation_depth")
        invalid = self.allowed_alphabet - STANDARD_AMINO_ACIDS
        if invalid or any(len(residue) != 1 for residue in self.allowed_alphabet):
            raise ValueError(f"allowed_alphabet contains nonstandard residues: {sorted(invalid)}")
        if self.name == "esmc_iterative_mask" and not self.revision:
            raise ValueError("esmc_iterative_mask requires a pinned model revision")
        if self.proposal_mix is not None:
            if not self.proposal_mix or any(weight < 0 for weight in self.proposal_mix.values()):
                raise ValueError("proposal_mix weights must be nonempty and nonnegative")
            if sum(self.proposal_mix.values()) <= 0:
                raise ValueError("at least one proposal_mix weight must be positive")
        return self


class FolDEBenchmarkConfig(StrictModel):
    zero_shot_model_name: str
    few_shot_model_name: str
    acquisition: Literal["mean", "ucb", "constantliar", "random"]
    ensemble_size: PositiveInt = 5


class SelectionBenchmarkConfig(StrictModel):
    diversity_metric: Literal["embedding", "mutation_jaccard", "hamming"]
    minimum_pairwise_distance: NonNegativeInt = 1
    max_per_parent: PositiveInt | None = None


class OutputBenchmarkConfig(StrictModel):
    checkpoint_dir: str = Field(min_length=1)
    save_proposal_pools: bool = True
    compression: Literal["zstd", "gzip", "none"] = "zstd"


class GenerativeMultimutantBenchmarkConfig(StrictModel):
    """Top-level, versioned configuration for benchmark execution."""

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1)
    seed: int
    release_run: bool = False
    datasets: tuple[DatasetBenchmarkConfig, ...]
    campaign: CampaignBenchmarkConfig
    features: FeatureBenchmarkConfig
    generator: GeneratorBenchmarkConfig
    folde: FolDEBenchmarkConfig
    selection: SelectionBenchmarkConfig
    output: OutputBenchmarkConfig

    @model_validator(mode="after")
    def validate_release_contract(self) -> "GenerativeMultimutantBenchmarkConfig":
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        dms_ids = [dataset.dms_id for dataset in self.datasets]
        if len(dms_ids) != len(set(dms_ids)):
            raise ValueError("dataset dms_id values must be unique")
        requested_depths = {depth for dataset in self.datasets for depth in dataset.target_depths}
        if self.generator.min_mutation_depth > min(requested_depths) or (
            self.generator.max_mutation_depth < max(requested_depths)
        ):
            raise ValueError("generator mutation-depth range does not cover all target depths")
        if self.generator.raw_sampling_cap is not None and (
            self.generator.raw_sampling_cap < self.campaign.proposal_budget
        ):
            raise ValueError("raw_sampling_cap must be at least proposal_budget")
        if self.release_run:
            revisions = {
                "embedding_revision": self.features.embedding_revision,
                "naturalness_revision": self.features.naturalness_revision,
            }
            if self.generator.model is not None:
                revisions["generator.revision"] = self.generator.revision or ""
            unpinned = [
                name for name, revision in revisions.items() if revision in UNPINNED_REVISIONS
            ]
            if unpinned:
                raise ValueError(f"release runs require pinned revisions: {', '.join(unpinned)}")
        return self

"""Closed-world Protocol A runner for the exhaustive Olson double-mutant landscape."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from app.helpers.sequence_util import allele_set_to_seq_id
from folde.benchmarks.feature_store import MemmapFeatureStore
from folde.benchmarks.multimutant_data import sha256_file
from folde.benchmarks.multimutant_metrics import campaign_metrics, paired_statistical_report
from folde.benchmarks.multimutant_oracle import ProteinGymFitnessOracle
from folde.candidate_generation import (
    AdjacentGenerator,
    CandidateGenerator,
    CandidateProposal,
    GeneratorContext,
    LibraryConstrainedPLMGenerator,
    MeasuredVariant,
    TopSingleCombinationGenerator,
    UniformShellGenerator,
)
from folde.candidate_generation.strategy import derive_component_seed
from folde.data import DMS_DIR, DMS_METADATA_FILE, NATURALNESS_DIR

OLSON_DMS_ID = "SPG1_STRSG_Olson_2014"
ARM_NAMES = (
    "random_folde",
    "adjacent_folde",
    "top_single_folde",
    "plm_only",
    "plm_plus_folde",
    "full_universe_folde",
)


class OlsonProtocolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    benchmark_seed: int = 42
    simulation_seeds: tuple[int, ...] = tuple(range(20))
    initial_singles: int = Field(default=32, gt=0)
    round_size: int = Field(default=16, gt=0)
    rounds: int = Field(default=5, gt=0)
    proposal_budget: int = Field(default=10_000, gt=0)
    model_name: str = "esm2_650m"
    model_revision: str = "ProteinGym-local-600m"
    ridge_alpha: float = Field(default=10.0, gt=0)


class OlsonRoundRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_number: int
    proposal_count: int
    proposal_generator: str
    proposal_pool_path: str | None
    proposal_pool_sha256: str | None
    selected_seq_ids: tuple[str, ...]
    selected_scores: dict[str, float]
    revealed_scores: dict[str, float]
    best_score_after_round: float
    top_1pct_pool_recall: float
    top_10pct_pool_recall: float
    top_1pct_selected_precision: float
    generator_wall_seconds: float
    selection_wall_seconds: float


class OlsonArmResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    config_sha256: str
    arm: str
    simulation_seed: int
    initial_seq_ids: tuple[str, ...]
    measured_variants: tuple[MeasuredVariant, ...]
    rounds: tuple[OlsonRoundRecord, ...]
    terminal_metrics: dict[str, float | int] = Field(default_factory=dict)


class PoolScorer(Protocol):
    def score(
        self,
        proposals: Sequence[CandidateProposal],
        measured_variants: Sequence[MeasuredVariant],
    ) -> Mapping[str, float]: ...


class ProposalScoreScorer:
    def score(
        self,
        proposals: Sequence[CandidateProposal],
        measured_variants: Sequence[MeasuredVariant],
    ) -> Mapping[str, float]:
        del measured_variants
        return {
            proposal.identity.seq_id: float(
                proposal.proposal_score if proposal.proposal_score is not None else float("-inf")
            )
            for proposal in proposals
        }


class FolDELinearScorer:
    """Leakage-safe FolDE surrogate using embeddings plus PLM naturalness.

    This lightweight scorer is intended for the closed-world benchmark sweep. It follows
    the FolDE contract (activity-independent precomputed features plus revealed labels)
    while avoiding repeated construction of a 500k-row object DataFrame.
    """

    def __init__(
        self,
        feature_store: MemmapFeatureStore,
        naturalness_scores: Mapping[str, float],
        *,
        alpha: float,
        prediction_chunk_size: int = 10_000,
    ):
        self._feature_store = feature_store
        self._naturalness_scores = naturalness_scores
        self._alpha = alpha
        self._prediction_chunk_size = prediction_chunk_size

    def _features(self, seq_ids: Sequence[str]) -> np.ndarray:
        embeddings = self._feature_store.get_array(seq_ids)
        naturalness = np.asarray(
            [self._naturalness_scores[seq_id] for seq_id in seq_ids], dtype=np.float32
        )[:, None]
        return np.concatenate([embeddings, naturalness], axis=1)

    def score(
        self,
        proposals: Sequence[CandidateProposal],
        measured_variants: Sequence[MeasuredVariant],
    ) -> Mapping[str, float]:
        return self.score_seq_ids(
            [proposal.identity.seq_id for proposal in proposals], measured_variants
        )

    def score_seq_ids(
        self,
        proposal_ids: Sequence[str],
        measured_variants: Sequence[MeasuredVariant],
    ) -> dict[str, float]:
        measured_ids = [variant.identity.seq_id for variant in measured_variants]
        scaler = StandardScaler()
        train_features = scaler.fit_transform(self._features(measured_ids))
        model = Ridge(alpha=self._alpha)
        model.fit(train_features, [variant.activity for variant in measured_variants])
        predictions: dict[str, float] = {}
        for offset in range(0, len(proposal_ids), self._prediction_chunk_size):
            chunk_ids = proposal_ids[offset : offset + self._prediction_chunk_size]
            proposal_features = scaler.transform(self._features(chunk_ids))
            predictions.update(zip(chunk_ids, model.predict(proposal_features), strict=True))
        return predictions


class FullUniverseGenerator:
    def __init__(self, eligible_seq_ids: Sequence[str]):
        self._eligible_seq_ids = tuple(eligible_seq_ids)

    @property
    def name(self) -> str:
        return "full_universe_upper_bound"

    def generate(self, context: GeneratorContext) -> list[CandidateProposal]:
        return UniformShellGenerator(list(self._eligible_seq_ids)).generate(
            context.model_copy(update={"proposal_budget": len(self._eligible_seq_ids)})
        )


def load_olson_activity() -> tuple[str, pd.Series, list[str], list[str]]:
    metadata = pd.read_csv(DMS_METADATA_FILE, usecols=["DMS_id", "target_seq"])
    reference = str(metadata.loc[metadata["DMS_id"] == OLSON_DMS_ID, "target_seq"].item())
    activity = pd.read_csv(DMS_DIR / f"{OLSON_DMS_ID}.csv", usecols=["mutant", "DMS_score"])
    activity["seq_id"] = [
        allele_set_to_seq_id(set(str(mutant).split(":"))) for mutant in activity["mutant"]
    ]
    scores = pd.Series(
        pd.to_numeric(activity["DMS_score"], errors="coerce").to_numpy(),
        index=activity["seq_id"],
        dtype=float,
    )
    scores = scores[np.isfinite(scores)]
    if not scores.index.is_unique:
        raise ValueError("Olson activity identifiers are not unique")
    singles = sorted(seq_id for seq_id in scores.index if "_" not in seq_id)
    doubles = sorted(seq_id for seq_id in scores.index if seq_id.count("_") == 1)
    return reference, scores, singles, doubles


def load_additive_naturalness_scores(eligible_seq_ids: Sequence[str]) -> dict[str, float]:
    table = pd.read_csv(
        NATURALNESS_DIR / f"{OLSON_DMS_ID}_naturalness_600m.csv",
        usecols=["seq_id", "wt_marginal"],
    )
    single_scores = {
        str(seq_id): float(np.log(max(float(value), 1e-20)))
        for seq_id, value in zip(table["seq_id"], table["wt_marginal"], strict=True)
        if len(str(seq_id)) >= 3 and str(seq_id)[0].isalpha() and str(seq_id)[-1].isalpha()
    }
    scores: dict[str, float] = {}
    for seq_id in eligible_seq_ids:
        components = seq_id.split("_")
        if any(component not in single_scores for component in components):
            raise ValueError(f"naturalness table lacks a component of {seq_id}")
        scores[seq_id] = sum(single_scores[component] for component in components)
    return scores


def _write_proposal_pool(proposals: Sequence[CandidateProposal], path: Path) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    np.savez_compressed(
        temporary,
        seq_id=np.asarray([proposal.identity.seq_id for proposal in proposals]),
        proposal_score=np.asarray(
            [
                np.nan if proposal.proposal_score is None else proposal.proposal_score
                for proposal in proposals
            ],
            dtype=np.float64,
        ),
        proposal_rank=np.asarray(
            [proposal.proposal_rank for proposal in proposals], dtype=np.int64
        ),
        generator_name=np.asarray([proposal.generator_name for proposal in proposals]),
        generator_version=np.asarray([proposal.generator_version for proposal in proposals]),
        generator_checkpoint=np.asarray(
            [proposal.generator_checkpoint or "" for proposal in proposals]
        ),
        generation_seed=np.asarray(
            [proposal.generation_seed for proposal in proposals], dtype=np.int64
        ),
        mutation_depth=np.asarray(
            [proposal.identity.mutation_depth for proposal in proposals], dtype=np.int16
        ),
        parent_seq_ids=np.asarray([json.dumps(proposal.parent_seq_ids) for proposal in proposals]),
        metadata=np.asarray(
            [json.dumps(proposal.metadata, sort_keys=True) for proposal in proposals]
        ),
    )
    generated = temporary.with_suffix(f"{temporary.suffix}.npz")
    generated.replace(path)
    return path.name, sha256_file(path)


def _write_full_universe_pool(
    seq_ids: Sequence[str], path: Path, generation_seed: int
) -> tuple[str, str]:
    """Write compact columnar provenance without allocating proposal models."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    np.savez_compressed(
        temporary,
        seq_id=np.asarray(seq_ids),
        proposal_score=np.full(len(seq_ids), np.nan, dtype=np.float64),
        proposal_rank=np.arange(1, len(seq_ids) + 1, dtype=np.int64),
        generator_name=np.full(len(seq_ids), "full_universe_upper_bound"),
        generator_version=np.full(len(seq_ids), "1.0"),
        generator_checkpoint=np.full(len(seq_ids), ""),
        generation_seed=np.full(len(seq_ids), generation_seed, dtype=np.int64),
        mutation_depth=np.full(len(seq_ids), 2, dtype=np.int16),
        parent_seq_ids=np.full(len(seq_ids), "[]"),
        metadata=np.full(
            len(seq_ids),
            json.dumps({"experiment_role": "upper_bound_selector"}, sort_keys=True),
        ),
    )
    generated = temporary.with_suffix(f"{temporary.suffix}.npz")
    generated.replace(path)
    return path.name, sha256_file(path)


def _write_checkpoint(result: OlsonArmResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(result.model_dump_json(indent=2) + "\n")
    temporary.replace(path)


def _select_initial_singles(
    singles: Sequence[str], config: OlsonProtocolConfig, seed: int
) -> tuple[str, ...]:
    if config.initial_singles > len(singles):
        raise ValueError("initial single-mutant budget exceeds available singles")
    rng = np.random.default_rng(
        derive_component_seed(config.benchmark_seed, OLSON_DMS_ID, seed, "initial")
    )
    indices = rng.choice(len(singles), size=config.initial_singles, replace=False)
    return tuple(sorted(singles[index] for index in indices))


def _generator_for_arm(
    arm: str,
    doubles: Sequence[str],
    naturalness: Mapping[str, float],
    config: OlsonProtocolConfig,
) -> CandidateGenerator:
    if arm == "random_folde":
        return UniformShellGenerator(list(doubles))
    if arm == "adjacent_folde":
        return AdjacentGenerator(list(doubles))
    if arm == "top_single_folde":
        return TopSingleCombinationGenerator(list(doubles), dict(naturalness))
    if arm in {"plm_only", "plm_plus_folde"}:
        return LibraryConstrainedPLMGenerator(
            doubles,
            naturalness,
            model_name=config.model_name,
            model_revision=config.model_revision,
        )
    if arm == "full_universe_folde":
        return FullUniverseGenerator(doubles)
    raise ValueError(f"unknown Protocol A arm: {arm}")


def run_olson_arm(
    *,
    arm: str,
    simulation_seed: int,
    config: OlsonProtocolConfig,
    reference_sequence: str,
    activity: pd.Series,
    singles: Sequence[str],
    doubles: Sequence[str],
    naturalness: Mapping[str, float],
    feature_store: MemmapFeatureStore,
    output_dir: Path,
    resume: bool = True,
) -> OlsonArmResult:
    """Run one arm/seed pair with a round-level atomic checkpoint."""
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown Protocol A arm: {arm}")
    checkpoint_path = output_dir / f"{arm}-seed-{simulation_seed}.json"
    config_sha256 = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
    initial_ids = _select_initial_singles(singles, config, simulation_seed)
    oracle = ProteinGymFitnessOracle(
        reference_sequence, activity, eligible_seq_ids=[*singles, *doubles]
    )
    if resume and checkpoint_path.exists():
        result = OlsonArmResult.model_validate_json(checkpoint_path.read_text())
        if (
            result.arm != arm
            or result.simulation_seed != simulation_seed
            or result.initial_seq_ids != initial_ids
            or result.config_sha256 != config_sha256
        ):
            raise ValueError("checkpoint does not match requested configuration/arm/seed")
        oracle.restore(result.measured_variants)
    else:
        oracle.measure(initial_ids, round_number=0)
        result = OlsonArmResult(
            config_sha256=config_sha256,
            arm=arm,
            simulation_seed=simulation_seed,
            initial_seq_ids=initial_ids,
            measured_variants=oracle.measured_variants,
            rounds=(),
        )
    generator = _generator_for_arm(arm, doubles, naturalness, config)
    scorer: PoolScorer = (
        ProposalScoreScorer()
        if arm == "plm_only"
        else FolDELinearScorer(feature_store, naturalness, alpha=config.ridge_alpha)
    )
    records = list(result.rounds)
    positions = frozenset(
        int(component[1:-1]) for seq_id in doubles for component in seq_id.split("_")
    )
    alphabet = frozenset(component[-1] for seq_id in doubles for component in seq_id.split("_"))
    eligible_double_scores = activity.loc[list(doubles)].to_numpy(dtype=float)
    top_1_cutoff = float(np.quantile(eligible_double_scores, 0.99))
    top_10_cutoff = float(np.quantile(eligible_double_scores, 0.90))
    for round_number in range(len(records) + 1, config.rounds + 1):
        proposal_budget = len(doubles) if arm == "full_universe_folde" else config.proposal_budget
        generator_arm = "plm_shared" if arm in {"plm_only", "plm_plus_folde"} else arm
        generation_seed = derive_component_seed(
            config.benchmark_seed,
            OLSON_DMS_ID,
            simulation_seed,
            generator_arm,
            round_number,
            "generator",
        )
        context = GeneratorContext(
            reference_sequence=reference_sequence,
            measured_variants=oracle.measured_variants,
            allowed_positions=positions,
            allowed_alphabet=alphabet,
            min_mutation_depth=2,
            max_mutation_depth=2,
            proposal_budget=proposal_budget,
            round_number=round_number,
            random_seed=generation_seed,
        )
        started = time.perf_counter()
        if arm == "full_universe_folde":
            measured_ids = {variant.identity.seq_id for variant in oracle.measured_variants}
            pool_ids = [seq_id for seq_id in doubles if seq_id not in measured_ids]
            proposals: tuple[CandidateProposal, ...] = ()
        else:
            proposals = tuple(generator.generate(context))
            pool_ids = [proposal.identity.seq_id for proposal in proposals]
        generator_seconds = time.perf_counter() - started
        if len(pool_ids) < config.round_size:
            raise ValueError("proposal pool is smaller than the experimental slate")
        started = time.perf_counter()
        if arm == "full_universe_folde":
            assert isinstance(scorer, FolDELinearScorer)
            predicted = scorer.score_seq_ids(pool_ids, oracle.measured_variants)
        else:
            predicted = dict(scorer.score(proposals, oracle.measured_variants))
        selected = tuple(
            sorted(predicted, key=lambda seq_id: (-predicted[seq_id], seq_id))[: config.round_size]
        )
        selection_seconds = time.perf_counter() - started
        pool_scores = activity.loc[pool_ids].to_numpy(dtype=float)
        revealed = oracle.measure(selected, round_number=round_number)
        pool_path = (
            output_dir / "proposal_pools" / f"{arm}-seed-{simulation_seed}-round-{round_number}.npz"
        )
        pool_file, pool_hash = (
            _write_full_universe_pool(pool_ids, pool_path, generation_seed)
            if arm == "full_universe_folde"
            else _write_proposal_pool(proposals, pool_path)
        )
        records.append(
            OlsonRoundRecord(
                round_number=round_number,
                proposal_count=len(pool_ids),
                proposal_generator=generator.name,
                proposal_pool_path=f"proposal_pools/{pool_file}",
                proposal_pool_sha256=pool_hash,
                selected_seq_ids=selected,
                selected_scores={seq_id: float(predicted[seq_id]) for seq_id in selected},
                revealed_scores={seq_id: float(revealed.loc[seq_id]) for seq_id in selected},
                best_score_after_round=max(
                    variant.activity for variant in oracle.measured_variants
                ),
                top_1pct_pool_recall=float(
                    np.sum(pool_scores >= top_1_cutoff)
                    / np.sum(eligible_double_scores >= top_1_cutoff)
                ),
                top_10pct_pool_recall=float(
                    np.sum(pool_scores >= top_10_cutoff)
                    / np.sum(eligible_double_scores >= top_10_cutoff)
                ),
                top_1pct_selected_precision=float(np.mean(revealed.to_numpy() >= top_1_cutoff)),
                generator_wall_seconds=generator_seconds,
                selection_wall_seconds=selection_seconds,
            )
        )
        result = OlsonArmResult(
            config_sha256=config_sha256,
            arm=arm,
            simulation_seed=simulation_seed,
            initial_seq_ids=initial_ids,
            measured_variants=oracle.measured_variants,
            rounds=tuple(records),
        )
        _write_checkpoint(result, checkpoint_path)
    terminal = campaign_metrics(
        [variant.activity for variant in oracle.measured_variants],
        activity.loc[[*singles, *doubles]].to_numpy(dtype=float),
        initial_measurement_count=config.initial_singles,
        measurement_batch_sizes=[config.initial_singles] + [config.round_size] * len(records),
    )
    result.terminal_metrics = terminal
    _write_checkpoint(result, checkpoint_path)
    return result


def write_paired_report(results: Sequence[OlsonArmResult], output_path: Path) -> dict[str, object]:
    endpoint = {
        arm: {
            result.simulation_seed: float(result.terminal_metrics["area_under_best_found_curve"])
            for result in results
            if result.arm == arm
        }
        for arm in ARM_NAMES
    }
    report = paired_statistical_report(endpoint)
    report["endpoint"] = "area_under_best_found_curve"
    report["dataset"] = OLSON_DMS_ID
    report["endpoint_by_arm_seed"] = endpoint
    report["comparisons"] = {
        arm: paired_statistical_report(
            endpoint,
            reference_arm="plm_plus_folde",
            comparison_arm=arm,
        )
        for arm in ARM_NAMES
        if arm != "plm_plus_folde" and endpoint[arm]
    }
    active_arms = [arm for arm in ARM_NAMES if endpoint[arm]]
    common_seeds = sorted(set.intersection(*(set(endpoint[arm]) for arm in active_arms)))
    ranks: dict[str, list[float]] = {arm: [] for arm in active_arms}
    for simulation_seed in common_seeds:
        values = pd.Series(
            {arm: endpoint[arm][simulation_seed] for arm in active_arms}, dtype=float
        )
        for arm, rank in values.rank(ascending=False, method="average").items():
            ranks[str(arm)].append(float(rank))
    report["mean_normalized_rank"] = {
        arm: float(np.mean(arm_ranks) / len(active_arms))
        for arm, arm_ranks in ranks.items()
        if arm_ranks
    }
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return report

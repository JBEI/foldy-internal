"""Paired proposal/selection gates for multi-objective ALDE campaigns."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge

from folde.benchmarks.multimutant_data import sha256_file
from folde.candidate_generation.strategy import allocate_proposal_budget, derive_component_seed
from folde.multiobjective_data import MultiObjectiveDataset
from folde.multiobjective_metrics import fit_quantile_reference
from folde.parego import chebyshev_scalarize, parego_select, quantile_normalize

ALDE_GATE_SCHEMA_VERSION = "1.0"
ALDE_BASELINE_ARMS = (
    "mixed_parego",
    "mixed_fixed",
    "mixed_plm_only",
    "mixed_random",
    "random_pool_parego",
    "naturalness_pool_parego",
    "full_parego",
)
ALDE_HYBRID_ARMS = ("mixed_hybrid_soft25", "mixed_hybrid_veto25")
ALDE_ARMS = ALDE_BASELINE_ARMS + ALDE_HYBRID_ARMS
POOL_KIND_BY_ARM = {
    "mixed_parego": "mixed",
    "mixed_fixed": "mixed",
    "mixed_plm_only": "mixed",
    "mixed_random": "mixed",
    "random_pool_parego": "random",
    "naturalness_pool_parego": "naturalness",
    "full_parego": "full",
    "mixed_hybrid_soft25": "mixed",
    "mixed_hybrid_veto25": "mixed",
}
SELECTOR_BY_ARM = {
    "mixed_parego": "parego",
    "mixed_fixed": "fixed",
    "mixed_plm_only": "plm_only",
    "mixed_random": "random",
    "random_pool_parego": "parego",
    "naturalness_pool_parego": "parego",
    "full_parego": "parego",
    "mixed_hybrid_soft25": "hybrid_soft",
    "mixed_hybrid_veto25": "hybrid_veto",
}


class MultiObjectiveALDEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ALDE_GATE_SCHEMA_VERSION
    benchmark_seed: int = 81_811
    simulation_seeds: tuple[int, ...] = tuple(range(10))
    initial_size: int = Field(default=16, gt=2)
    batch_size: int = Field(default=16, gt=0)
    rounds: int = Field(default=5, gt=0)
    proposal_budget: int = Field(default=512, gt=0)
    projection_dim: int = Field(default=32, gt=1)
    ensemble_size: int = Field(default=8, ge=3)
    ridge_alpha: float = Field(default=10.0, gt=0)
    ranker_type: str = Field(default="ridge", pattern="^(ridge|torch_mlp)$")
    mlp_hidden_dims: tuple[int, ...] = (100, 50)
    mlp_dropout: float = Field(default=0.2, ge=0, lt=1)
    mlp_learning_rate: float = Field(default=3e-4, gt=0)
    mlp_weight_decay: float = Field(default=1e-5, ge=0)
    mlp_pretrain_epochs: int = Field(default=10, gt=0)
    mlp_train_epochs: int = Field(default=200, gt=0)
    mlp_train_patience: int = Field(default=40, gt=0)
    epsilon: float = Field(default=0.05, ge=0, le=1)
    hybrid_prior_weight: float = Field(default=0.25, gt=0, lt=1)
    hybrid_veto_quantile: float = Field(default=0.25, gt=0, lt=1)
    proposal_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "plm_naturalness": 0.40,
            "elite_neighbor": 0.30,
            "uniform": 0.30,
        }
    )

    @model_validator(mode="after")
    def validate_budget(self) -> "MultiObjectiveALDEConfig":
        if self.proposal_budget < self.batch_size:
            raise ValueError("proposal_budget must be at least batch_size")
        allocate_proposal_budget(self.proposal_mix, self.proposal_budget)
        return self


class MultiObjectiveRoundResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_number: int
    proposal_pool_path: str
    proposal_pool_sha256: str
    proposal_pool_size: int
    source_counts: dict[str, int]
    selected_seq_ids: tuple[str, ...]
    revealed_values: dict[str, dict[str, float]]
    measured_hypervolume: float
    hypervolume_regret: float
    exact_front_recall: float
    epsilon_front_coverage: float
    measured_front_spread: float
    selected_embedding_diversity: float
    proposal_attainable_hypervolume: float
    proposal_front_recall: float
    proposal_epsilon_front_coverage: float
    model_wall_seconds: float
    selection_wall_seconds: float


class MultiObjectiveCampaignResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ALDE_GATE_SCHEMA_VERSION
    config_sha256: str
    dataset: str
    arm: str
    simulation_seed: int
    initial_seq_ids: tuple[str, ...]
    measured_seq_ids: tuple[str, ...]
    rounds: tuple[MultiObjectiveRoundResult, ...]


class MultiObjectiveOracle:
    """Reveal complete objective rows only for explicitly selected variants."""

    def __init__(self, activity_df: pd.DataFrame):
        if not activity_df.index.is_unique or activity_df.isna().any().any():
            raise ValueError("oracle activity table must be complete with a unique index")
        self._activity_df = activity_df.copy()
        self._measured_ids: list[str] = []
        self._calls: list[tuple[str, ...]] = []

    @property
    def measured_ids(self) -> tuple[str, ...]:
        return tuple(self._measured_ids)

    @property
    def calls(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._calls)

    def measured_values(self) -> pd.DataFrame:
        return self._activity_df.loc[self._measured_ids].copy()

    def measure(self, seq_ids: Sequence[str]) -> pd.DataFrame:
        requested = tuple(seq_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("measurement slate must be nonempty and unique")
        already_measured = set(requested) & set(self._measured_ids)
        if already_measured:
            raise ValueError(f"variants already measured: {sorted(already_measured)[:5]}")
        missing = set(requested) - set(self._activity_df.index)
        if missing:
            raise ValueError(f"variants outside oracle universe: {sorted(missing)[:5]}")
        revealed = self._activity_df.loc[list(requested)]
        if not np.isfinite(revealed.to_numpy(dtype=float)).all():
            raise ValueError("oracle can reveal only finite complete objective rows")
        self._calls.append(requested)
        self._measured_ids.extend(requested)
        return revealed.copy()

    def restore(
        self, initial_ids: Sequence[str], rounds: Sequence[MultiObjectiveRoundResult]
    ) -> None:
        if self._measured_ids:
            raise ValueError("restore requires a fresh oracle")
        self.measure(initial_ids)
        for record in rounds:
            revealed = self.measure(record.selected_seq_ids)
            for seq_id, values in record.revealed_values.items():
                for objective, expected in values.items():
                    if float(revealed.loc[seq_id, objective]) != expected:
                        raise ValueError("checkpoint objective value does not match oracle")


def _fast_nondominated_mask_2d(points: np.ndarray) -> np.ndarray:
    """O(N log N) nondominated mask for two maximize objectives."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("fast Pareto mask requires an (N, 2) array")
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    order = np.lexsort((-unique_points[:, 1], -unique_points[:, 0]))
    keep_unique = np.zeros(len(unique_points), dtype=bool)
    best_y = float("-inf")
    for index in order:
        y = float(unique_points[index, 1])
        if y > best_y:
            keep_unique[index] = True
            best_y = y
    return keep_unique[inverse]


def _front_ids(values: pd.DataFrame) -> tuple[str, ...]:
    mask = _fast_nondominated_mask_2d(values.to_numpy(dtype=float))
    return tuple(str(seq_id) for seq_id in values.index[mask])


def _epsilon_front_coverage(
    measured_points: np.ndarray,
    true_front_points: np.ndarray,
    epsilon: float,
) -> float:
    if len(measured_points) == 0 or len(true_front_points) == 0:
        return 0.0
    covered = [
        np.any(np.all(measured_points >= front_point[None, :] - epsilon, axis=1))
        for front_point in true_front_points
    ]
    return float(np.mean(covered))


def _front_spread(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    front = points[_fast_nondominated_mask_2d(points)]
    return float(np.mean(np.ptp(front, axis=0))) if len(front) >= 2 else 0.0


def _mean_pairwise_distance(features: np.ndarray) -> float:
    if len(features) < 2:
        return 0.0
    differences = features[:, None, :] - features[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    upper = distances[np.triu_indices(len(features), k=1)]
    return float(np.mean(upper))


class ALDEFeatureSpace:
    """Shared low-dimensional screening features and PLM scores for one dataset."""

    def __init__(
        self,
        dataset: MultiObjectiveDataset,
        config: MultiObjectiveALDEConfig,
    ):
        self.seq_ids = tuple(str(seq_id) for seq_id in dataset.activity_df.index)
        self.row_by_id = {seq_id: index for index, seq_id in enumerate(self.seq_ids)}
        embeddings = np.stack(
            [np.asarray(value, dtype=np.float32) for value in dataset.embedding_series.values]
        )
        rng = np.random.default_rng(
            derive_component_seed(config.benchmark_seed, dataset.protein, "projection")
        )
        projection = rng.normal(
            0.0,
            1.0 / np.sqrt(config.projection_dim),
            size=(embeddings.shape[1], config.projection_dim),
        ).astype(np.float32)
        projected = embeddings @ projection
        means = projected.mean(axis=0)
        scales = projected.std(axis=0)
        scales[scales == 0] = 1.0
        self.projected = ((projected - means) / scales).astype(np.float32)
        naturalness = dataset.naturalness_df.mean(axis=1, skipna=True).to_numpy(dtype=float)
        finite = np.isfinite(naturalness)
        fill = float(np.median(naturalness[finite])) if finite.any() else 0.0
        self.naturalness = np.where(finite, naturalness, fill)

    def rows(self, seq_ids: Sequence[str]) -> np.ndarray:
        return np.fromiter(
            (self.row_by_id[seq_id] for seq_id in seq_ids),
            dtype=np.int64,
            count=len(seq_ids),
        )


def _initial_ids(
    seq_ids: Sequence[str], dataset: str, simulation_seed: int, config: MultiObjectiveALDEConfig
) -> tuple[str, ...]:
    rng = np.random.default_rng(
        derive_component_seed(
            config.benchmark_seed, dataset, simulation_seed, "initial_measurements"
        )
    )
    chosen = rng.choice(len(seq_ids), size=config.initial_size, replace=False)
    return tuple(sorted(seq_ids[index] for index in chosen))


def _rank_naturalness(available_ids: Sequence[str], feature_space: ALDEFeatureSpace) -> list[str]:
    return sorted(
        available_ids,
        key=lambda seq_id: (
            -feature_space.naturalness[feature_space.row_by_id[seq_id]],
            seq_id,
        ),
    )


def _elite_neighbor_order(
    available_ids: Sequence[str],
    measured_values: pd.DataFrame,
    feature_space: ALDEFeatureSpace,
) -> list[str]:
    elite_ids = _front_ids(measured_values)
    available_rows = feature_space.rows(available_ids)
    elite_rows = feature_space.rows(elite_ids)
    available_features = feature_space.projected[available_rows]
    elite_features = feature_space.projected[elite_rows]
    min_distances = np.full(len(available_ids), np.inf, dtype=float)
    for elite in elite_features:
        distances = np.sum((available_features - elite) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
    return [
        available_ids[index]
        for index in sorted(
            range(len(available_ids)),
            key=lambda index: (min_distances[index], available_ids[index]),
        )
    ]


def build_proposal_pool(
    *,
    kind: str,
    all_seq_ids: Sequence[str],
    measured_values: pd.DataFrame,
    feature_space: ALDEFeatureSpace,
    budget: int,
    random_seed: int,
    proposal_mix: Mapping[str, float],
) -> tuple[list[str], list[str]]:
    """Build a proposal pool without access to unmeasured objective values."""
    measured_ids = set(str(seq_id) for seq_id in measured_values.index)
    available_ids = [seq_id for seq_id in all_seq_ids if seq_id not in measured_ids]
    target = min(budget, len(available_ids))
    if target == 0:
        return [], []
    rng = np.random.default_rng(random_seed)
    if kind == "full":
        return available_ids, ["full_universe"] * len(available_ids)
    if kind == "random":
        order = rng.permutation(len(available_ids))[:target]
        return [available_ids[index] for index in order], ["uniform"] * target
    if kind == "naturalness":
        selected = _rank_naturalness(available_ids, feature_space)[:target]
        return selected, ["plm_naturalness"] * len(selected)
    if kind != "mixed":
        raise ValueError(f"unknown proposal-pool kind: {kind}")

    allocation = allocate_proposal_budget(proposal_mix, target)
    channel_orders = {
        "plm_naturalness": _rank_naturalness(available_ids, feature_space),
        "elite_neighbor": _elite_neighbor_order(available_ids, measured_values, feature_space),
        "uniform": [available_ids[index] for index in rng.permutation(len(available_ids))],
    }
    selected: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()
    for channel in proposal_mix:
        if allocation[channel] == 0:
            continue
        for seq_id in channel_orders[channel]:
            if seq_id in seen:
                continue
            selected.append(seq_id)
            sources.append(channel)
            seen.add(seq_id)
            if sources.count(channel) == allocation[channel]:
                break
    if len(selected) < target:
        for seq_id in channel_orders["uniform"]:
            if seq_id not in seen:
                selected.append(seq_id)
                sources.append("fallback_uniform")
                seen.add(seq_id)
                if len(selected) == target:
                    break
    return selected, sources


def predict_objective_ensemble(
    *,
    measured_values: pd.DataFrame,
    candidate_ids: Sequence[str],
    feature_space: ALDEFeatureSpace,
    config: MultiObjectiveALDEConfig,
    random_seed: int,
    dataset: MultiObjectiveDataset | None = None,
) -> np.ndarray:
    """Fit independent bootstrap-ridge objective ensembles on revealed labels only."""
    if config.ranker_type == "torch_mlp":
        if dataset is None:
            raise ValueError("Torch-MLP prediction requires the complete feature dataset")
        return _predict_torch_mlp_ensemble(
            measured_values=measured_values,
            candidate_ids=candidate_ids,
            dataset=dataset,
            config=config,
            random_seed=random_seed,
        )
    train_features = feature_space.projected[feature_space.rows(list(measured_values.index))]
    candidate_features = feature_space.projected[feature_space.rows(candidate_ids)]
    predictions = np.empty(
        (len(candidate_ids), config.ensemble_size, measured_values.shape[1]), dtype=np.float64
    )
    for objective_index, objective in enumerate(measured_values.columns):
        labels = measured_values[objective].to_numpy(dtype=float)
        for member in range(config.ensemble_size):
            rng = np.random.default_rng(derive_component_seed(random_seed, objective, member))
            bootstrap = rng.choice(len(labels), size=len(labels), replace=True)
            model = Ridge(alpha=config.ridge_alpha)
            model.fit(train_features[bootstrap], labels[bootstrap])
            predictions[:, member, objective_index] = model.predict(candidate_features)
    return predictions


def _predict_torch_mlp_ensemble(
    *,
    measured_values: pd.DataFrame,
    candidate_ids: Sequence[str],
    dataset: MultiObjectiveDataset,
    config: MultiObjectiveALDEConfig,
    random_seed: int,
) -> np.ndarray:
    """Retrain one production Torch-MLP ensemble per objective on revealed labels."""
    from folde.few_shot_models import TorchMLPFewShotModel

    candidate_index = list(candidate_ids)
    naturalness = dataset.naturalness_df.loc[candidate_index]
    embeddings = dataset.embedding_series.loc[candidate_index]
    per_objective: list[np.ndarray] = []
    for objective_index, objective in enumerate(measured_values.columns):
        model = TorchMLPFewShotModel(
            wt_aa_seq=dataset.wt_sequence,
            random_state=derive_component_seed(random_seed, objective_index),
            hidden_dims=list(config.mlp_hidden_dims),
            dropout=config.mlp_dropout,
            learning_rate=config.mlp_learning_rate,
            weight_decay=config.mlp_weight_decay,
            ensemble_size=config.ensemble_size,
            pretrain=True,
            pretrain_epochs=config.mlp_pretrain_epochs,
            train_epochs=config.mlp_train_epochs,
            train_patience=config.mlp_train_patience,
            val_frequency=10,
            do_validation_with_pair_fraction=0.2,
            device="cpu",
        )
        logging.disable(logging.CRITICAL)
        model.pretrain(dataset.naturalness_df, dataset.embedding_series)
        model.fit(
            dataset.naturalness_df,
            dataset.embedding_series,
            measured_values[objective],
        )
        predictions = model.predict(naturalness, embeddings)
        per_objective.append(
            np.stack([prediction.to_numpy(dtype=float) for prediction in predictions], axis=1)
        )
    return np.stack(per_objective, axis=2)


def _fixed_weight_select(
    predictions: np.ndarray,
    candidate_ids: Sequence[str],
    batch_size: int,
    random_seed: int,
) -> tuple[list[str], list[dict[str, object]]]:
    normalized = quantile_normalize(predictions, np.ones(predictions.shape[2]))
    weights = np.full(predictions.shape[2], 1.0 / predictions.shape[2])
    scalar = chebyshev_scalarize(normalized, weights, rho=0.05)
    result = parego_select(
        scalar[:, :, None],
        np.asarray(candidate_ids),
        np.ones(1),
        batch_size,
        lie_noise_stddev_multiplier=3.0,
        ucb_beta=2.0,
        random_state=random_seed,
    )
    return result.selected_seq_ids, result.records


def _naturalness_percentiles(
    candidate_ids: Sequence[str], feature_space: ALDEFeatureSpace
) -> np.ndarray:
    """Return deterministic empirical-CDF naturalness values, where one is best."""
    values = feature_space.naturalness[feature_space.rows(candidate_ids)]
    sorted_values = np.sort(values)
    return np.searchsorted(sorted_values, values, side="right") / len(values)


def select_candidates(
    *,
    selector: str,
    predictions: np.ndarray | None,
    candidate_ids: Sequence[str],
    feature_space: ALDEFeatureSpace,
    batch_size: int,
    random_seed: int,
    hybrid_prior_weight: float = 0.25,
    hybrid_veto_quantile: float = 0.25,
) -> tuple[list[str], list[dict[str, object]]]:
    if len(candidate_ids) < batch_size:
        raise ValueError("proposal pool is smaller than batch_size")
    if selector == "random":
        rng = np.random.default_rng(random_seed)
        chosen = rng.choice(len(candidate_ids), size=batch_size, replace=False)
        return [candidate_ids[int(index)] for index in chosen], []
    if selector == "plm_only":
        return _rank_naturalness(candidate_ids, feature_space)[:batch_size], []
    if predictions is None:
        raise ValueError(f"selector {selector} requires objective predictions")
    if selector == "fixed":
        return _fixed_weight_select(predictions, candidate_ids, batch_size, random_seed)
    if selector == "parego":
        result = parego_select(
            predictions,
            np.asarray(candidate_ids),
            np.ones(predictions.shape[2]),
            batch_size,
            lie_noise_stddev_multiplier=3.0,
            ucb_beta=2.0,
            rho=0.05,
            random_state=random_seed,
        )
        return result.selected_seq_ids, result.records
    if selector == "hybrid_soft":
        normalized = quantile_normalize(predictions, np.ones(predictions.shape[2]))
        naturalness = _naturalness_percentiles(candidate_ids, feature_space)
        hybrid_predictions = (
            1.0 - hybrid_prior_weight
        ) * normalized + hybrid_prior_weight * naturalness[:, None, None]
        result = parego_select(
            hybrid_predictions,
            np.asarray(candidate_ids),
            np.ones(predictions.shape[2]),
            batch_size,
            lie_noise_stddev_multiplier=3.0,
            ucb_beta=2.0,
            rho=0.05,
            random_state=random_seed,
        )
        return result.selected_seq_ids, result.records
    if selector == "hybrid_veto":
        naturalness = _naturalness_percentiles(candidate_ids, feature_space)
        result = parego_select(
            predictions,
            np.asarray(candidate_ids),
            np.ones(predictions.shape[2]),
            batch_size,
            lie_noise_stddev_multiplier=3.0,
            ucb_beta=2.0,
            rho=0.05,
            random_state=random_seed,
            feasibility=naturalness,
            min_feasibility=hybrid_veto_quantile,
        )
        return result.selected_seq_ids, result.records
    raise ValueError(f"unknown selector: {selector}")


def _campaign_metrics(
    seq_ids: Sequence[str],
    normalized_activity: pd.DataFrame,
    true_front_ids: Sequence[str],
    true_hypervolume: float,
    epsilon: float,
) -> dict[str, float]:
    values = normalized_activity.loc[list(seq_ids)]
    # Values are already on the pinned shared-reference scale. Re-fitting an empirical
    # CDF here would silently make hypervolume incomparable across arms.
    from folde.parego import dominated_hypervolume

    hypervolume = dominated_hypervolume(values.to_numpy(dtype=float), np.zeros(2))
    true_front_points = normalized_activity.loc[list(true_front_ids)].to_numpy(dtype=float)
    measured_points = values.to_numpy(dtype=float)
    return {
        "hypervolume": hypervolume,
        "hypervolume_regret": true_hypervolume - hypervolume,
        "exact_front_recall": len(set(seq_ids) & set(true_front_ids)) / len(true_front_ids),
        "epsilon_front_coverage": _epsilon_front_coverage(
            measured_points, true_front_points, epsilon
        ),
        "front_spread": _front_spread(measured_points),
    }


def _write_pool_artifact(
    *,
    path: Path,
    candidate_ids: Sequence[str],
    sources: Sequence[str],
    feature_space: ALDEFeatureSpace,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    np.savez_compressed(
        temporary,
        seq_id=np.asarray(candidate_ids),
        source_channel=np.asarray(sources),
        naturalness=feature_space.naturalness[feature_space.rows(candidate_ids)],
    )
    generated = temporary.with_suffix(f"{temporary.suffix}.npz")
    generated.replace(path)
    return sha256_file(path)


def _write_checkpoint(result: MultiObjectiveCampaignResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(result.model_dump_json(indent=2) + "\n")
    temporary.replace(path)


def run_multiobjective_campaign(
    *,
    dataset: MultiObjectiveDataset,
    arm: str,
    simulation_seed: int,
    config: MultiObjectiveALDEConfig,
    feature_space: ALDEFeatureSpace,
    output_dir: Path,
    resume: bool = True,
) -> MultiObjectiveCampaignResult:
    """Run one paired multi-objective arm with round checkpointing."""
    if arm not in ALDE_ARMS:
        raise ValueError(f"unknown ALDE arm: {arm}")
    config_sha256 = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
    checkpoint = output_dir / f"{dataset.protein}-{arm}-seed-{simulation_seed}.json"
    initial_ids = _initial_ids(feature_space.seq_ids, dataset.protein, simulation_seed, config)
    oracle = MultiObjectiveOracle(dataset.activity_df)
    if resume and checkpoint.exists():
        result = MultiObjectiveCampaignResult.model_validate_json(checkpoint.read_text())
        if (
            result.config_sha256 != config_sha256
            or result.dataset != dataset.protein
            or result.arm != arm
            or result.simulation_seed != simulation_seed
            or result.initial_seq_ids != initial_ids
        ):
            raise ValueError("checkpoint does not match requested campaign")
        oracle.restore(initial_ids, result.rounds)
    else:
        oracle.measure(initial_ids)
        result = MultiObjectiveCampaignResult(
            config_sha256=config_sha256,
            dataset=dataset.protein,
            arm=arm,
            simulation_seed=simulation_seed,
            initial_seq_ids=initial_ids,
            measured_seq_ids=oracle.measured_ids,
            rounds=(),
        )

    reference = fit_quantile_reference(dataset.activity_df)
    normalized_activity = reference.transform(dataset.activity_df)
    true_front_ids = _front_ids(normalized_activity)
    from folde.parego import dominated_hypervolume

    true_hypervolume = dominated_hypervolume(normalized_activity.to_numpy(dtype=float), np.zeros(2))
    records = list(result.rounds)
    for round_number in range(len(records) + 1, config.rounds + 1):
        measured_values = oracle.measured_values()
        pool_kind = POOL_KIND_BY_ARM[arm]
        shared_pool_arm = "mixed_shared" if pool_kind == "mixed" else arm
        pool_seed = derive_component_seed(
            config.benchmark_seed,
            dataset.protein,
            simulation_seed,
            round_number,
            shared_pool_arm,
            "proposal",
        )
        candidate_ids, sources = build_proposal_pool(
            kind=pool_kind,
            all_seq_ids=feature_space.seq_ids,
            measured_values=measured_values,
            feature_space=feature_space,
            budget=(len(feature_space.seq_ids) if pool_kind == "full" else config.proposal_budget),
            random_seed=pool_seed,
            proposal_mix=config.proposal_mix,
        )
        selector = SELECTOR_BY_ARM[arm]
        model_started = time.perf_counter()
        predictions = None
        if selector in {"parego", "fixed", "hybrid_soft", "hybrid_veto"}:
            model_seed_components: tuple[object, ...] = (
                (dataset.protein, "objective_models")
                if config.ranker_type == "torch_mlp"
                else (dataset.protein, simulation_seed, round_number, "objective_models")
            )
            predictions = predict_objective_ensemble(
                measured_values=measured_values,
                candidate_ids=candidate_ids,
                feature_space=feature_space,
                config=config,
                random_seed=derive_component_seed(config.benchmark_seed, *model_seed_components),
                dataset=dataset,
            )
        model_seconds = time.perf_counter() - model_started
        selection_started = time.perf_counter()
        selector_seed_arm = (
            "mixed_activity_shared"
            if arm in {"mixed_parego", "mixed_hybrid_soft25", "mixed_hybrid_veto25"}
            else arm
        )
        selected_ids, _ = select_candidates(
            selector=selector,
            predictions=predictions,
            candidate_ids=candidate_ids,
            feature_space=feature_space,
            batch_size=config.batch_size,
            random_seed=derive_component_seed(
                config.benchmark_seed,
                dataset.protein,
                simulation_seed,
                round_number,
                selector_seed_arm,
                "selector",
            ),
            hybrid_prior_weight=config.hybrid_prior_weight,
            hybrid_veto_quantile=config.hybrid_veto_quantile,
        )
        selection_seconds = time.perf_counter() - selection_started
        pool_path = (
            output_dir
            / "proposal_pools"
            / f"{dataset.protein}-{arm}-seed-{simulation_seed}-round-{round_number}.npz"
        )
        pool_hash = _write_pool_artifact(
            path=pool_path,
            candidate_ids=candidate_ids,
            sources=sources,
            feature_space=feature_space,
        )
        # Post-hoc proposal analysis starts only after the pool and selected slate freeze.
        pool_metrics = _campaign_metrics(
            candidate_ids,
            normalized_activity,
            true_front_ids,
            true_hypervolume,
            config.epsilon,
        )
        revealed = oracle.measure(selected_ids)
        metrics = _campaign_metrics(
            oracle.measured_ids,
            normalized_activity,
            true_front_ids,
            true_hypervolume,
            config.epsilon,
        )
        selected_features = feature_space.projected[feature_space.rows(selected_ids)]
        source_counts = {source: sources.count(source) for source in sorted(set(sources))}
        records.append(
            MultiObjectiveRoundResult(
                round_number=round_number,
                proposal_pool_path=f"proposal_pools/{pool_path.name}",
                proposal_pool_sha256=pool_hash,
                proposal_pool_size=len(candidate_ids),
                source_counts=source_counts,
                selected_seq_ids=tuple(selected_ids),
                revealed_values={
                    seq_id: {
                        objective: float(revealed.loc[seq_id, objective])
                        for objective in revealed.columns
                    }
                    for seq_id in selected_ids
                },
                measured_hypervolume=metrics["hypervolume"],
                hypervolume_regret=metrics["hypervolume_regret"],
                exact_front_recall=metrics["exact_front_recall"],
                epsilon_front_coverage=metrics["epsilon_front_coverage"],
                measured_front_spread=metrics["front_spread"],
                selected_embedding_diversity=_mean_pairwise_distance(selected_features),
                proposal_attainable_hypervolume=pool_metrics["hypervolume"],
                proposal_front_recall=pool_metrics["exact_front_recall"],
                proposal_epsilon_front_coverage=pool_metrics["epsilon_front_coverage"],
                model_wall_seconds=model_seconds,
                selection_wall_seconds=selection_seconds,
            )
        )
        result = MultiObjectiveCampaignResult(
            config_sha256=config_sha256,
            dataset=dataset.protein,
            arm=arm,
            simulation_seed=simulation_seed,
            initial_seq_ids=initial_ids,
            measured_seq_ids=oracle.measured_ids,
            rounds=tuple(records),
        )
        _write_checkpoint(result, checkpoint)
        if config.ranker_type == "torch_mlp":
            print(
                f"    round={round_number}/{config.rounds} measured={len(oracle.measured_ids)} "
                f"regret={metrics['hypervolume_regret']:.5f} "
                f"train={model_seconds:.1f}s select={selection_seconds:.1f}s",
                flush=True,
            )
    return result


def paired_dataset_report(
    differences_by_dataset: Mapping[str, float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Analyze six dataset-level paired differences; positive favors the target arm."""
    datasets = sorted(differences_by_dataset)
    differences = np.asarray([differences_by_dataset[name] for name in datasets], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
    bootstrap_medians = np.median(samples, axis=1)
    if np.all(differences == 0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(differences, alternative="two-sided", method="exact").pvalue)
    return {
        "datasets": datasets,
        "differences": differences.tolist(),
        "median_difference": float(np.median(differences)),
        "mean_difference": float(np.mean(differences)),
        "bootstrap_95pct_ci": [
            float(np.quantile(bootstrap_medians, 0.025)),
            float(np.quantile(bootstrap_medians, 0.975)),
        ],
        "wilcoxon_exact_p": p_value,
        "wins": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "losses": int(np.sum(differences < 0)),
    }


def analyze_gate_campaigns(
    campaigns: Sequence[MultiObjectiveCampaignResult],
) -> dict[str, Any]:
    """Aggregate campaign results into the preregistered dataset-level gates."""
    if not campaigns:
        raise ValueError("no campaigns supplied for gate analysis")
    datasets = sorted({campaign.dataset for campaign in campaigns})
    arms = sorted({campaign.arm for campaign in campaigns})
    per_dataset_arm: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        per_dataset_arm[dataset] = {}
        for arm in arms:
            matching = [
                campaign
                for campaign in campaigns
                if campaign.dataset == dataset and campaign.arm == arm
            ]
            if not matching:
                continue
            per_dataset_arm[dataset][arm] = {
                metric: float(
                    np.mean([getattr(campaign.rounds[-1], metric) for campaign in matching])
                )
                for metric in (
                    "hypervolume_regret",
                    "exact_front_recall",
                    "epsilon_front_coverage",
                    "measured_front_spread",
                    "selected_embedding_diversity",
                )
            }
            per_dataset_arm[dataset][arm].update(
                {
                    "round1_proposal_attainable_hypervolume": float(
                        np.mean(
                            [
                                campaign.rounds[0].proposal_attainable_hypervolume
                                for campaign in matching
                            ]
                        )
                    ),
                    "round1_proposal_front_recall": float(
                        np.mean([campaign.rounds[0].proposal_front_recall for campaign in matching])
                    ),
                }
            )

    def differences(
        target_arm: str,
        comparison_arm: str,
        metric: str,
        *,
        lower_is_better: bool,
    ) -> dict[str, float]:
        values = {}
        for dataset in datasets:
            target = per_dataset_arm[dataset][target_arm][metric]
            comparison = per_dataset_arm[dataset][comparison_arm][metric]
            values[dataset] = comparison - target if lower_is_better else target - comparison
        return values

    comparisons = {
        "proposal_mixed_vs_random": paired_dataset_report(
            differences(
                "mixed_parego",
                "random_pool_parego",
                "round1_proposal_attainable_hypervolume",
                lower_is_better=False,
            ),
            seed=101,
        ),
        "proposal_mixed_vs_naturalness": paired_dataset_report(
            differences(
                "mixed_parego",
                "naturalness_pool_parego",
                "round1_proposal_attainable_hypervolume",
                lower_is_better=False,
            ),
            seed=102,
        ),
        "selector_parego_vs_plm_only": paired_dataset_report(
            differences(
                "mixed_parego",
                "mixed_plm_only",
                "hypervolume_regret",
                lower_is_better=True,
            ),
            seed=103,
        ),
        "selector_parego_vs_random": paired_dataset_report(
            differences(
                "mixed_parego",
                "mixed_random",
                "hypervolume_regret",
                lower_is_better=True,
            ),
            seed=104,
        ),
        "selector_parego_vs_fixed": paired_dataset_report(
            differences(
                "mixed_parego",
                "mixed_fixed",
                "hypervolume_regret",
                lower_is_better=True,
            ),
            seed=105,
        ),
        "end_to_end_mixed_vs_random_pool": paired_dataset_report(
            differences(
                "mixed_parego",
                "random_pool_parego",
                "hypervolume_regret",
                lower_is_better=True,
            ),
            seed=106,
        ),
        "full_pool_regret_delta_mixed_minus_full": paired_dataset_report(
            {
                dataset: per_dataset_arm[dataset]["mixed_parego"]["hypervolume_regret"]
                - per_dataset_arm[dataset]["full_parego"]["hypervolume_regret"]
                for dataset in datasets
            },
            seed=107,
        ),
        "front_coverage_parego_vs_fixed": paired_dataset_report(
            differences(
                "mixed_parego",
                "mixed_fixed",
                "epsilon_front_coverage",
                lower_is_better=False,
            ),
            seed=108,
        ),
    }

    def positive_ci_pass(name: str) -> bool:
        report = comparisons[name]
        assert isinstance(report, dict)
        interval = report["bootstrap_95pct_ci"]
        assert isinstance(interval, list)
        return bool(report["median_difference"] > 0 and interval[0] > 0)

    proposal_pass = positive_ci_pass("proposal_mixed_vs_random")
    selector_pass = positive_ci_pass("selector_parego_vs_plm_only") and positive_ci_pass(
        "selector_parego_vs_random"
    )
    end_to_end_pass = positive_ci_pass("end_to_end_mixed_vs_random_pool")
    noninferiority_report = comparisons["full_pool_regret_delta_mixed_minus_full"]
    noninferiority_pass = bool(noninferiority_report["bootstrap_95pct_ci"][1] <= 0.01)
    front_coverage_pass = positive_ci_pass("front_coverage_parego_vs_fixed")
    return {
        "schema_version": ALDE_GATE_SCHEMA_VERSION,
        "datasets": datasets,
        "arms": arms,
        "per_dataset_arm_means": per_dataset_arm,
        "comparisons": comparisons,
        "gates": {
            "proposal_gate": proposal_pass,
            "selector_gate": selector_pass,
            "end_to_end_gate": end_to_end_pass,
            "full_pool_noninferiority_gate": noninferiority_pass,
            "front_coverage_gate": front_coverage_pass,
            "production_replication_authorized": all(
                (
                    proposal_pass,
                    selector_pass,
                    end_to_end_pass,
                    noninferiority_pass,
                    front_coverage_pass,
                )
            ),
        },
    }


def analyze_hybrid_gate_campaigns(
    campaigns: Sequence[MultiObjectiveCampaignResult],
) -> dict[str, Any]:
    """Add the preregistered naturalness-aware hybrid comparisons and gates."""
    report = analyze_gate_campaigns(campaigns)
    per_dataset_arm = report["per_dataset_arm_means"]
    datasets = report["datasets"]

    def regret_improvement(target_arm: str, comparison_arm: str) -> dict[str, float]:
        return {
            dataset: per_dataset_arm[dataset][comparison_arm]["hypervolume_regret"]
            - per_dataset_arm[dataset][target_arm]["hypervolume_regret"]
            for dataset in datasets
        }

    comparisons = report["comparisons"]
    comparisons.update(
        {
            "hybrid_soft_vs_parego": paired_dataset_report(
                regret_improvement("mixed_hybrid_soft25", "mixed_parego"), seed=201
            ),
            "hybrid_soft_vs_plm_only": paired_dataset_report(
                regret_improvement("mixed_hybrid_soft25", "mixed_plm_only"), seed=202
            ),
            "hybrid_soft_vs_random_pool": paired_dataset_report(
                regret_improvement("mixed_hybrid_soft25", "random_pool_parego"), seed=203
            ),
            "hybrid_veto_vs_parego": paired_dataset_report(
                regret_improvement("mixed_hybrid_veto25", "mixed_parego"), seed=204
            ),
            "hybrid_soft_regret_delta_vs_parego": paired_dataset_report(
                {
                    dataset: -difference
                    for dataset, difference in regret_improvement(
                        "mixed_hybrid_soft25", "mixed_parego"
                    ).items()
                },
                seed=205,
            ),
            "hybrid_soft_regret_delta_vs_full": paired_dataset_report(
                {
                    dataset: per_dataset_arm[dataset]["mixed_hybrid_soft25"]["hypervolume_regret"]
                    - per_dataset_arm[dataset]["full_parego"]["hypervolume_regret"]
                    for dataset in datasets
                },
                seed=206,
            ),
        }
    )

    def positive_ci_pass(name: str) -> bool:
        comparison = comparisons[name]
        return bool(comparison["median_difference"] > 0 and comparison["bootstrap_95pct_ci"][0] > 0)

    hybrid_selector_pass = positive_ci_pass("hybrid_soft_vs_parego") and positive_ci_pass(
        "hybrid_soft_vs_plm_only"
    )
    hybrid_end_to_end_pass = positive_ci_pass("hybrid_soft_vs_random_pool")
    retention_pass = bool(
        comparisons["hybrid_soft_regret_delta_vs_parego"]["bootstrap_95pct_ci"][1] <= 0.005
    )
    hybrid_full_pass = bool(
        comparisons["hybrid_soft_regret_delta_vs_full"]["bootstrap_95pct_ci"][1] <= 0.01
    )
    report["gates"].update(
        {
            "hybrid_selector_gate": hybrid_selector_pass,
            "hybrid_end_to_end_gate": hybrid_end_to_end_pass,
            "hybrid_parego_retention_gate": retention_pass,
            "hybrid_full_pool_noninferiority_gate": hybrid_full_pass,
            "hybrid_production_replication_authorized": all(
                (
                    hybrid_selector_pass,
                    hybrid_end_to_end_pass,
                    retention_pass,
                    hybrid_full_pass,
                )
            ),
        }
    )
    return report


def analyze_mlp_replication_campaigns(
    campaigns: Sequence[MultiObjectiveCampaignResult],
) -> dict[str, Any]:
    """Analyze the preregistered four-arm sequential Torch-MLP replication."""
    if not campaigns:
        raise ValueError("no campaigns supplied for MLP analysis")
    required_arms = {
        "mixed_parego",
        "mixed_hybrid_veto25",
        "mixed_plm_only",
        "mixed_random",
    }
    datasets = sorted({campaign.dataset for campaign in campaigns})
    arms = {campaign.arm for campaign in campaigns}
    if not required_arms <= arms:
        raise ValueError(f"MLP analysis is missing arms: {sorted(required_arms - arms)}")

    per_dataset_arm: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        per_dataset_arm[dataset] = {}
        for arm in sorted(required_arms):
            matching = [
                campaign
                for campaign in campaigns
                if campaign.dataset == dataset and campaign.arm == arm
            ]
            if not matching:
                raise ValueError(f"MLP analysis is missing {dataset}/{arm}")
            per_dataset_arm[dataset][arm] = {
                "hypervolume_regret": float(
                    np.mean([campaign.rounds[-1].hypervolume_regret for campaign in matching])
                ),
                "epsilon_front_coverage": float(
                    np.mean([campaign.rounds[-1].epsilon_front_coverage for campaign in matching])
                ),
                "exact_front_recall": float(
                    np.mean([campaign.rounds[-1].exact_front_recall for campaign in matching])
                ),
            }

    def improvement(target: str, comparison: str) -> dict[str, float]:
        return {
            dataset: per_dataset_arm[dataset][comparison]["hypervolume_regret"]
            - per_dataset_arm[dataset][target]["hypervolume_regret"]
            for dataset in datasets
        }

    comparisons = {
        "veto_vs_parego": paired_dataset_report(
            improvement("mixed_hybrid_veto25", "mixed_parego"), seed=301
        ),
        "veto_vs_plm_only": paired_dataset_report(
            improvement("mixed_hybrid_veto25", "mixed_plm_only"), seed=302
        ),
        "veto_vs_random": paired_dataset_report(
            improvement("mixed_hybrid_veto25", "mixed_random"), seed=303
        ),
        "parego_vs_plm_only": paired_dataset_report(
            improvement("mixed_parego", "mixed_plm_only"), seed=304
        ),
        "parego_vs_random": paired_dataset_report(
            improvement("mixed_parego", "mixed_random"), seed=305
        ),
    }

    def positive_ci_pass(name: str) -> bool:
        comparison = comparisons[name]
        return bool(comparison["median_difference"] > 0 and comparison["bootstrap_95pct_ci"][0] > 0)

    veto_increment_pass = positive_ci_pass("veto_vs_parego")
    learned_selector_pass = positive_ci_pass("veto_vs_plm_only") and positive_ci_pass(
        "veto_vs_random"
    )
    return {
        "schema_version": ALDE_GATE_SCHEMA_VERSION,
        "ranker_type": "torch_mlp",
        "datasets": datasets,
        "arms": sorted(required_arms),
        "per_dataset_arm_means": per_dataset_arm,
        "comparisons": comparisons,
        "gates": {
            "veto_increment_gate": veto_increment_pass,
            "learned_selector_gate": learned_selector_pass,
            "plain_parego_vs_plm_only": positive_ci_pass("parego_vs_plm_only"),
            "plain_parego_vs_random": positive_ci_pass("parego_vs_random"),
            "mlp_pipeline_authorized": veto_increment_pass and learned_selector_pass,
        },
    }

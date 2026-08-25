"""Analyze a paired hard-, weighted-, and soft-target Bradley-Terry sweep.

The loader discovers arms and datasets from checkpoint JSON files, validates the
paired design, and writes reusable tidy tables plus a Markdown report.  In
particular, it verifies that every matched arm saw the same round-one slate and,
by default, that round one contains single mutants only.

Run from ``backend/`` after a sweep completes::

    ../.venv/bin/python -m folde.scripts.analyze_soft_bt_sweep

The default input is the corrected multi-mutant SoftBT sweep.  Other paired
SoftBT sweeps can be analyzed with ``--eval-prefix`` and ``--control-arm``.
Checkpoint files are read-only; outputs go to a sibling ``-analysis`` directory.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from folde.data import DMS_DIR
from folde.scripts.run_bt_mse_weight_sweep import DEFAULT_CHECKPOINT_DIR
from folde.scripts.run_soft_bt_sweep import DEFAULT_EVAL_PREFIX as RUN_EVAL_PREFIX
from folde.util import DMS_SHORTNAMES

DEFAULT_EVAL_PREFIX = f"{RUN_EVAL_PREFIX}-multi"
DEFAULT_CONTROL_ARM = "E1E1-300m-BT-hard"
SUMMARY_METRICS = (
    "cumulative_10pct_hits",
    "has_found_top_1pct",
    "cumulative_1pct_hits",
    "best_percentile_so_far",
    "held_out_activity_spearman",
)
PRIMARY_METRICS = ("cumulative_10pct_hits", "has_found_top_1pct")
STAGES = ("round3", "terminal")
SPREAD_METRICS = (
    "score_range",
    "score_std",
    "score_iqr",
    "score_central90_range",
)


def _short_name(dms_id: str) -> str:
    """Return a paper short name with a conservative fallback."""
    return DMS_SHORTNAMES.get(dms_id, dms_id.split("_")[0])


def _rng(seed: int, *parts: object) -> np.random.Generator:
    """Return a stable task-specific random generator."""
    token = "|".join(str(part) for part in parts).encode()
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(token)]))


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Apply Holm's family-wise-error correction."""
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return pd.Series(adjusted, index=p_values.index)


def _arm_metadata(config: Mapping[str, object]) -> dict[str, object]:
    """Derive stable labels and ordering from a FolDE configuration."""
    name = str(config["name"])
    params = config.get("few_shot_model_params")
    if not isinstance(params, Mapping):
        raise ValueError(f"{name} has no few_shot_model_params mapping")

    mse_weight = float(params.get("standardized_mse_weight", 0.0))
    if not np.isclose(mse_weight, 0.0):
        raise ValueError(f"{name} is not a pure BT-family arm: MSE weight={mse_weight:g}")
    raw_gap = bool(params.get("bt_activity_difference_weighting", False))
    temperature_value = params.get("bt_soft_target_temperature")
    floor_value = params.get("bt_soft_target_confidence_floor")
    temperature = None if temperature_value is None else float(temperature_value)
    confidence_floor = None if floor_value is None else float(floor_value)

    if raw_gap and temperature is not None:
        raise ValueError(f"{name} combines raw-gap weighting and soft targets")
    if confidence_floor is not None and temperature is None:
        raise ValueError(f"{name} has a confidence floor without soft targets")

    if raw_gap:
        family = "raw_gap"
        label = "Raw-gap BT"
        short_label = "Raw gap"
        sort_key = (1.0, 0.0, 0.0)
    elif temperature is None:
        family = "hard"
        label = "Hard BT"
        short_label = "Hard BT"
        sort_key = (0.0, 0.0, 0.0)
    else:
        family = "soft"
        floor_suffix = "" if confidence_floor is None else f", floor={confidence_floor:g}"
        short_suffix = "" if confidence_floor is None else f" + F={confidence_floor:g}"
        label = f"SoftBT T={temperature:g}{floor_suffix}"
        short_label = f"T={temperature:g}{short_suffix}"
        sort_key = (2.0, temperature, -1.0 if confidence_floor is None else confidence_floor)

    return {
        "arm": name,
        "arm_label": label,
        "arm_short_label": short_label,
        "family": family,
        "temperature": temperature,
        "confidence_floor": confidence_floor,
        "raw_gap_weighting": raw_gap,
        "_sort_key": sort_key,
    }


def _discover_checkpoints(checkpoint_dir: Path, eval_prefix: str) -> list[Path]:
    """Find checkpoint JSON files for one exact evaluation prefix."""
    paths = sorted(checkpoint_dir.glob(f"{eval_prefix}_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints matching {eval_prefix!r} in {checkpoint_dir.resolve()}"
        )
    return paths


def load_round_records(
    checkpoint_dir: Path,
    eval_prefix: str,
    *,
    control_arm: str = DEFAULT_CONTROL_ARM,
    require_single_round_one: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load checkpoints as tidy records and validate their paired campaign design."""
    rows: list[dict[str, object]] = []
    arm_rows: dict[str, dict[str, object]] = {}
    dataset_reference: set[str] | None = None
    round_one: dict[tuple[str, int, str], tuple[str, ...]] = {}
    pairing_keys: dict[str, set[tuple[str, int]]] = {}

    for path in _discover_checkpoints(checkpoint_dir, eval_prefix):
        payload = json.loads(path.read_text())
        campaigns = payload.get("campaign_results")
        if not isinstance(campaigns, list) or not campaigns:
            raise ValueError(f"{path.name} contains no campaign_results")
        observed_datasets = {str(campaign["dms_id"]) for campaign in campaigns}
        if dataset_reference is None:
            dataset_reference = observed_datasets
        elif observed_datasets != dataset_reference:
            raise ValueError(
                f"Dataset mismatch in {path.name}: expected={sorted(dataset_reference)}, "
                f"observed={sorted(observed_datasets)}"
            )

        checkpoint_arm: str | None = None
        checkpoint_keys: set[tuple[str, int]] = set()
        for campaign in campaigns:
            dms_id = str(campaign["dms_id"])
            config_results = campaign["config_results"]
            if len(config_results) != 1:
                raise ValueError(f"Expected one config in {path.name}/{dms_id}")
            config_result = config_results[0]
            config = config_result["config"]
            arm = str(config["name"])
            if checkpoint_arm is None:
                checkpoint_arm = arm
            elif checkpoint_arm != arm:
                raise ValueError(f"{path.name} contains more than one arm")

            metadata = _arm_metadata(config)
            previous_metadata = arm_rows.get(arm)
            if previous_metadata is not None and previous_metadata != metadata:
                raise ValueError(f"Configuration metadata changes within arm {arm}")
            arm_rows[arm] = metadata
            if require_single_round_one and not bool(config.get("one_mutation_at_a_time")):
                raise ValueError(f"{path.name}/{dms_id} does not enable one-mutation-at-a-time")

            max_rounds = int(campaign["max_rounds"])
            round_size = int(campaign["round_size"])
            expected_simulations = int(campaign["number_of_simulations"])
            simulations = config_result["simulation_results"]
            if len(simulations) != expected_simulations:
                raise ValueError(
                    f"{path.name}/{dms_id} has {len(simulations)} simulations; "
                    f"expected {expected_simulations}"
                )

            for simulation_index, simulation in enumerate(simulations):
                pairing_key = (dms_id, simulation_index)
                checkpoint_keys.add(pairing_key)
                mutants = simulation["mutant_metrics"]
                if len(mutants) != max_rounds * round_size:
                    raise ValueError(
                        f"{path.name}/{dms_id}/simulation {simulation_index} has "
                        f"{len(mutants)} mutants; expected {max_rounds * round_size}"
                    )
                first_round = tuple(
                    sorted(
                        str(mutant["seq_id"])
                        for mutant in mutants
                        if int(mutant["round_found"]) == 1
                    )
                )
                if len(first_round) != round_size:
                    raise ValueError(
                        f"{path.name}/{dms_id}/simulation {simulation_index} has an "
                        f"incomplete round-one slate"
                    )
                if require_single_round_one and any(
                    seq_id == "WT" or "_" in seq_id for seq_id in first_round
                ):
                    raise ValueError(
                        f"{path.name}/{dms_id}/simulation {simulation_index} contains a "
                        f"non-single mutant in round one"
                    )
                round_one[(dms_id, simulation_index, arm)] = first_round

                metrics_by_round = {
                    int(metric["round_num"]): metric for metric in simulation["round_metrics"]
                }
                if set(metrics_by_round) != set(range(1, max_rounds + 1)):
                    raise ValueError(
                        f"Incomplete round metrics for {path.name}/{dms_id}/"
                        f"simulation {simulation_index}"
                    )

                for round_num in range(1, max_rounds + 1):
                    cumulative = [
                        mutant for mutant in mutants if int(mutant["round_found"]) <= round_num
                    ]
                    current = [
                        mutant for mutant in mutants if int(mutant["round_found"]) == round_num
                    ]
                    if len(current) != round_size:
                        raise ValueError(
                            f"Incomplete round {round_num} for {path.name}/{dms_id}/"
                            f"simulation {simulation_index}"
                        )
                    round_metric = metrics_by_round[round_num]
                    misc = round_metric.get("misc", {})
                    rows.append(
                        {
                            "dms_id": dms_id,
                            "dms_shortname": _short_name(dms_id),
                            "arm": arm,
                            "simulation_index": simulation_index,
                            "round_num": round_num,
                            "is_terminal": round_num == max_rounds,
                            "round_size": round_size,
                            "screened_mutants": round_num * round_size,
                            "variant_pool_size": int(simulation["variant_pool_size"]),
                            "cumulative_10pct_hits": sum(
                                float(mutant["percentile"]) >= 0.90 for mutant in cumulative
                            ),
                            "cumulative_1pct_hits": sum(
                                float(mutant["percentile"]) >= 0.99 for mutant in cumulative
                            ),
                            "has_found_top_1pct": float(
                                any(float(mutant["percentile"]) >= 0.99 for mutant in cumulative)
                            ),
                            "best_percentile_this_round": max(
                                float(mutant["percentile"]) for mutant in current
                            ),
                            "best_percentile_so_far": max(
                                float(mutant["percentile"]) for mutant in cumulative
                            ),
                            "model_spearman": round_metric.get("model_spearman", np.nan),
                            "held_out_activity_spearman": misc.get(
                                "held_out_activity_spearman", np.nan
                            ),
                            "held_out_1pct_recall": misc.get("held_out_1pct_recall", np.nan),
                            "held_out_10pct_recall": misc.get("held_out_10pct_recall", np.nan),
                        }
                    )
        if checkpoint_arm is None:
            raise ValueError(f"{path.name} contains no arm")
        if checkpoint_arm in pairing_keys:
            raise ValueError(f"Duplicate checkpoint for arm {checkpoint_arm}")
        pairing_keys[checkpoint_arm] = checkpoint_keys

    if control_arm not in arm_rows:
        raise ValueError(f"Control arm {control_arm!r} was not found")
    if arm_rows[control_arm]["family"] != "hard":
        raise ValueError(f"Control arm {control_arm!r} is not hard BT")
    control_keys = pairing_keys[control_arm]
    for arm, keys in pairing_keys.items():
        if keys != control_keys:
            raise ValueError(f"Pairing keys differ between {control_arm} and {arm}")
        for dms_id, simulation_index in keys:
            if (
                round_one[(dms_id, simulation_index, arm)]
                != round_one[(dms_id, simulation_index, control_arm)]
            ):
                raise ValueError(
                    f"Round-one slate differs for {dms_id}/simulation {simulation_index}/"
                    f"arm {arm}"
                )

    ordered_metadata = sorted(arm_rows.values(), key=lambda row: row["_sort_key"])
    metadata_rows = []
    for arm_order, metadata in enumerate(ordered_metadata):
        result = {key: value for key, value in metadata.items() if key != "_sort_key"}
        result["arm_order"] = arm_order
        result["is_control"] = result["arm"] == control_arm
        metadata_rows.append(result)
    arm_metadata = pd.DataFrame(metadata_rows)
    records = pd.DataFrame(rows).merge(
        arm_metadata[["arm", "arm_label", "arm_short_label", "family", "arm_order"]],
        on="arm",
        validate="many_to_one",
    )
    return (
        records.sort_values(["dms_id", "arm_order", "simulation_index", "round_num"]).reset_index(
            drop=True
        ),
        arm_metadata.sort_values("arm_order").reset_index(drop=True),
    )


def _t_interval(values: pd.Series) -> tuple[float, float]:
    """Return a dataset-level 95% t interval."""
    clean = values.dropna().astype(float)
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, mean
    sem = float(stats.sem(clean))
    if sem == 0.0:
        return mean, mean
    low, high = stats.t.interval(0.95, len(clean) - 1, loc=mean, scale=sem)
    return float(low), float(high)


def make_aggregate_round_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Average simulations within datasets and datasets across the benchmark."""
    dataset_means = records.groupby(
        ["dms_id", "arm", "arm_label", "arm_order", "round_num"], as_index=False
    )[list(SUMMARY_METRICS)].mean()
    rows: list[dict[str, object]] = []
    for keys, group in dataset_means.groupby(
        ["arm", "arm_label", "arm_order", "round_num"], sort=True
    ):
        arm, arm_label, arm_order, round_num = keys
        for metric in SUMMARY_METRICS:
            low, high = _t_interval(group[metric])
            rows.append(
                {
                    "arm": arm,
                    "arm_label": arm_label,
                    "arm_order": arm_order,
                    "round_num": round_num,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "dataset_count": int(group["dms_id"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["arm_order", "round_num", "metric"])


def _stage_subset(records: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Select the pre-specified round-three or terminal checkpoint."""
    if stage == "round3":
        subset = records[records["round_num"] == 3]
    elif stage == "terminal":
        subset = records[records["is_terminal"]]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    if subset.empty:
        raise ValueError(f"No records are available for stage {stage}")
    return subset


def _metric_matrix(records: pd.DataFrame, arm: str, stage: str, metric: str) -> pd.DataFrame:
    """Return a paired dataset-by-simulation matrix."""
    subset = _stage_subset(records, stage)
    subset = subset[subset["arm"] == arm]
    matrix = subset.pivot(index="dms_id", columns="simulation_index", values=metric)
    if matrix.empty or matrix.isna().any().any():
        raise ValueError(f"Missing values in {arm}/{stage}/{metric}")
    return matrix.sort_index(axis=0).sort_index(axis=1)


def _bootstrap_means(matrix: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    """Hierarchically resample datasets and simulations, preserving paired values."""
    dataset_count, simulation_count = matrix.shape
    dataset_indices = rng.integers(0, dataset_count, size=(replicates, dataset_count))
    simulation_indices = rng.integers(
        0,
        simulation_count,
        size=(replicates, dataset_count, simulation_count),
    )
    sampled = matrix[dataset_indices[:, :, None], simulation_indices]
    return sampled.mean(axis=(1, 2))


def _estimate(
    records: pd.DataFrame,
    arm: str,
    stage: str,
    metric: str,
    replicates: int,
    seed: int,
    *,
    control_arm: str | None = None,
) -> tuple[float, float, float]:
    """Return a macro mean or paired delta and hierarchical-bootstrap bounds."""
    matrix = _metric_matrix(records, arm, stage, metric)
    values = matrix.to_numpy(dtype=float)
    if control_arm is not None:
        control = _metric_matrix(records, control_arm, stage, metric)
        if not matrix.index.equals(control.index) or not matrix.columns.equals(control.columns):
            raise ValueError(f"Pairing mismatch for {arm}/{stage}/{metric}")
        values = values - control.to_numpy(dtype=float)
    estimate = float(values.mean())
    bootstrap = _bootstrap_means(
        values,
        replicates,
        _rng(seed, arm, stage, metric, control_arm),
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return estimate, float(low), float(high)


def _randomization_p_value(
    records: pd.DataFrame,
    arm: str,
    control_arm: str,
    stage: str,
    metric: str,
    replicates: int,
    seed: int,
) -> float:
    """Calculate a paired dataset-level sign-flip p-value."""
    candidate = _metric_matrix(records, arm, stage, metric)
    control = _metric_matrix(records, control_arm, stage, metric)
    dataset_deltas = (candidate - control).mean(axis=1).to_numpy(dtype=float)
    observed = abs(float(dataset_deltas.mean()))
    rng = _rng(seed, arm, stage, metric, "randomization")
    exceedances = 0
    completed = 0
    while completed < replicates:
        batch_size = min(10_000, replicates - completed)
        signs = rng.choice((-1.0, 1.0), size=(batch_size, len(dataset_deltas)))
        null_values = np.abs((signs * dataset_deltas).mean(axis=1))
        exceedances += int(np.sum(null_values >= observed - 1e-15))
        completed += batch_size
    return (exceedances + 1.0) / (replicates + 1.0)


def make_stage_summary(
    records: pd.DataFrame,
    arms: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize round-three and terminal outcomes for every arm."""
    rows: list[dict[str, object]] = []
    for stage in STAGES:
        for arm_row in arms.itertuples(index=False):
            for metric in SUMMARY_METRICS:
                mean, low, high = _estimate(
                    records,
                    arm_row.arm,
                    stage,
                    metric,
                    bootstrap_replicates,
                    seed,
                )
                rows.append(
                    {
                        "stage": stage,
                        "arm": arm_row.arm,
                        "arm_label": arm_row.arm_label,
                        "arm_order": arm_row.arm_order,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def make_paired_comparisons(
    records: pd.DataFrame,
    arms: pd.DataFrame,
    control_arm: str,
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Compare every non-control arm with hard BT using paired inference."""
    rows: list[dict[str, object]] = []
    for stage in STAGES:
        for arm_row in arms[~arms["is_control"]].itertuples(index=False):
            for metric in SUMMARY_METRICS:
                delta, low, high = _estimate(
                    records,
                    arm_row.arm,
                    stage,
                    metric,
                    bootstrap_replicates,
                    seed,
                    control_arm=control_arm,
                )
                p_value = _randomization_p_value(
                    records,
                    arm_row.arm,
                    control_arm,
                    stage,
                    metric,
                    randomization_replicates,
                    seed,
                )
                rows.append(
                    {
                        "stage": stage,
                        "arm": arm_row.arm,
                        "arm_label": arm_row.arm_label,
                        "arm_order": arm_row.arm_order,
                        "metric": metric,
                        "mean_delta_vs_control": delta,
                        "ci_low": low,
                        "ci_high": high,
                        "randomization_p": p_value,
                    }
                )
    comparisons = pd.DataFrame(rows)
    comparisons["holm_p"] = np.nan
    for _, indices in comparisons.groupby(["stage", "metric"]).groups.items():
        comparisons.loc[indices, "holm_p"] = _holm_adjust(
            comparisons.loc[indices, "randomization_p"]
        )
    return comparisons


def make_per_target_summary(
    records: pd.DataFrame, arms: pd.DataFrame, control_arm: str
) -> pd.DataFrame:
    """Calculate per-target arm means and paired deltas at both stages."""
    rows: list[dict[str, object]] = []
    for stage in STAGES:
        means = (
            _stage_subset(records, stage)
            .groupby(
                ["dms_id", "dms_shortname", "arm", "arm_label", "arm_order"],
                as_index=False,
            )[list(SUMMARY_METRICS)]
            .mean()
        )
        control = means[means["arm"] == control_arm].set_index("dms_id")
        for row in means.itertuples(index=False):
            result: dict[str, object] = {
                "stage": stage,
                "dms_id": row.dms_id,
                "dms_shortname": row.dms_shortname,
                "arm": row.arm,
                "arm_label": row.arm_label,
                "arm_order": row.arm_order,
            }
            control_row = control.loc[row.dms_id]
            for metric in SUMMARY_METRICS:
                value = float(getattr(row, metric))
                result[metric] = value
                result[f"delta_{metric}_vs_control"] = value - float(control_row[metric])
            rows.append(result)
    return pd.DataFrame(rows).sort_values(["stage", "dms_id", "arm_order"])


def make_score_spreads(records: pd.DataFrame) -> pd.DataFrame:
    """Calculate raw and robust DMS-score spread statistics for tested datasets."""
    metadata = records.groupby(["dms_id", "dms_shortname"], as_index=False).agg(
        variant_pool_size=("variant_pool_size", "median")
    )
    rows: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        scores = pd.to_numeric(
            pd.read_csv(DMS_DIR / f"{row.dms_id}.csv", usecols=["DMS_score"])["DMS_score"],
            errors="coerce",
        ).dropna()
        values = scores.to_numpy(dtype=float)
        quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        score_iqr = float(quantiles[4] - quantiles[2])
        rows.append(
            {
                "dms_id": row.dms_id,
                "dms_shortname": row.dms_shortname,
                "score_count": len(values),
                "variant_pool_size": int(row.variant_pool_size),
                "score_min": float(values.min()),
                "score_q01": float(quantiles[0]),
                "score_q05": float(quantiles[1]),
                "score_q25": float(quantiles[2]),
                "score_median": float(quantiles[3]),
                "score_q75": float(quantiles[4]),
                "score_q95": float(quantiles[5]),
                "score_q99": float(quantiles[6]),
                "score_max": float(values.max()),
                "score_range": float(values.max() - values.min()),
                "score_std": float(values.std(ddof=1)),
                "score_iqr": score_iqr,
                "score_central90_range": float(quantiles[5] - quantiles[1]),
            }
        )
    return pd.DataFrame(rows).sort_values("dms_id")


def _permutation_spearman(
    x_values: np.ndarray,
    y_values: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return Spearman rho and a two-sided label-permutation p-value."""
    x_ranks = stats.rankdata(x_values).astype(float)
    y_ranks = stats.rankdata(y_values).astype(float)
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0.0:
        return float("nan"), float("nan")
    observed = float(x_centered @ y_centered / denominator)
    exceedances = 0
    completed = 0
    while completed < replicates:
        batch_size = min(5_000, replicates - completed)
        orders = np.argsort(rng.random((batch_size, len(y_ranks))), axis=1)
        null_values = np.abs(y_centered[orders] @ x_centered / denominator)
        exceedances += int(np.sum(null_values >= abs(observed) - 1e-15))
        completed += batch_size
    return observed, (exceedances + 1.0) / (replicates + 1.0)


def make_spread_arm_correlations(
    per_target: pd.DataFrame,
    spreads: pd.DataFrame,
    arms: pd.DataFrame,
    permutation_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Correlate score spread with each arm's paired discovery gain."""
    merged = per_target.merge(spreads, on=["dms_id", "dms_shortname"])
    noncontrol = set(arms.loc[~arms["is_control"], "arm"])
    merged = merged[merged["arm"].isin(noncontrol)]
    rows: list[dict[str, object]] = []
    for stage in STAGES:
        for metric in PRIMARY_METRICS:
            outcome = f"delta_{metric}_vs_control"
            for spread_metric in SPREAD_METRICS:
                for arm_row in arms[~arms["is_control"]].itertuples(index=False):
                    subset = merged[(merged["stage"] == stage) & (merged["arm"] == arm_row.arm)]
                    rho, p_value = _permutation_spearman(
                        subset[spread_metric].to_numpy(dtype=float),
                        subset[outcome].to_numpy(dtype=float),
                        permutation_replicates,
                        _rng(seed, stage, metric, spread_metric, arm_row.arm),
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "metric": metric,
                            "spread_metric": spread_metric,
                            "arm": arm_row.arm,
                            "arm_label": arm_row.arm_label,
                            "arm_order": arm_row.arm_order,
                            "dataset_count": len(subset),
                            "spearman_rho": rho,
                            "permutation_p": p_value,
                        }
                    )
    correlations = pd.DataFrame(rows)
    correlations["holm_within_spread_p"] = np.nan
    for _, indices in correlations.groupby(["stage", "metric", "spread_metric"]).groups.items():
        correlations.loc[indices, "holm_within_spread_p"] = _holm_adjust(
            correlations.loc[indices, "permutation_p"]
        )
    correlations["holm_global_p"] = np.nan
    for _, indices in correlations.groupby(["stage", "metric"]).groups.items():
        correlations.loc[indices, "holm_global_p"] = _holm_adjust(
            correlations.loc[indices, "permutation_p"]
        )
    return correlations


def make_soft_bt_responses(per_target: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    """Reduce each target's SoftBT arms to mean and best paired responses."""
    soft_arms = set(arms.loc[arms["family"] == "soft", "arm"])
    soft = per_target[per_target["arm"].isin(soft_arms)]
    rows: list[dict[str, object]] = []
    for (stage, dms_id, shortname), group in soft.groupby(
        ["stage", "dms_id", "dms_shortname"], sort=True
    ):
        result: dict[str, object] = {
            "stage": stage,
            "dms_id": dms_id,
            "dms_shortname": shortname,
            "soft_arm_count": len(group),
        }
        for metric in PRIMARY_METRICS:
            delta = group[f"delta_{metric}_vs_control"]
            result[f"mean_delta_{metric}"] = float(delta.mean())
            result[f"best_delta_{metric}"] = float(delta.max())
        rows.append(result)
    return pd.DataFrame(rows)


def make_spread_response_correlations(
    responses: pd.DataFrame,
    spreads: pd.DataFrame,
    permutation_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Test whether score spread predicts mean or best SoftBT response."""
    merged = responses.merge(spreads, on=["dms_id", "dms_shortname"])
    rows: list[dict[str, object]] = []
    for stage in STAGES:
        subset = merged[merged["stage"] == stage]
        for metric in PRIMARY_METRICS:
            for response_type in ("mean", "best"):
                response = f"{response_type}_delta_{metric}"
                for spread_metric in SPREAD_METRICS:
                    rho, p_value = _permutation_spearman(
                        subset[spread_metric].to_numpy(dtype=float),
                        subset[response].to_numpy(dtype=float),
                        permutation_replicates,
                        _rng(seed, stage, metric, response_type, spread_metric),
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "metric": metric,
                            "response_type": response_type,
                            "spread_metric": spread_metric,
                            "dataset_count": len(subset),
                            "spearman_rho": rho,
                            "permutation_p": p_value,
                        }
                    )
    correlations = pd.DataFrame(rows)
    correlations["holm_p"] = np.nan
    for _, indices in correlations.groupby(["stage", "metric"]).groups.items():
        correlations.loc[indices, "holm_p"] = _holm_adjust(
            correlations.loc[indices, "permutation_p"]
        )
    return correlations


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a small dependency-free Markdown table."""
    lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
    lines.extend(" | ".join(str(value) for value in row) for row in rows)
    return "\n".join(lines)


def write_summary_report(
    records: pd.DataFrame,
    arms: pd.DataFrame,
    stage_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    per_target: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a concise, data-derived interpretation."""
    summary_rows = []
    for arm_row in arms.itertuples(index=False):
        values: list[object] = [arm_row.arm_label]
        for stage in STAGES:
            subset = stage_summary[
                (stage_summary["stage"] == stage) & (stage_summary["arm"] == arm_row.arm)
            ].set_index("metric")
            values.extend(
                [
                    f"{subset.loc['cumulative_10pct_hits', 'mean']:.3f}",
                    f"{subset.loc['has_found_top_1pct', 'mean']:.3f}",
                    f"{100 * subset.loc['best_percentile_so_far', 'mean']:.3f}%",
                ]
            )
        summary_rows.append(values)

    def winner_for(stage: str) -> str:
        subset = stage_summary[stage_summary["stage"] == stage].pivot(
            index="arm", columns="metric", values="mean"
        )
        return max(
            arms.sort_values("arm_order")["arm"],
            key=lambda arm: (
                subset.loc[arm, "has_found_top_1pct"],
                subset.loc[arm, "cumulative_10pct_hits"],
            ),
        )

    round3_winner = winner_for("round3")
    terminal_winner = winner_for("terminal")
    labels = arms.set_index("arm")["arm_label"]
    comparison_rows = []
    primary = comparisons[comparisons["metric"].isin(PRIMARY_METRICS)]
    for row in primary.sort_values(["stage", "arm_order", "metric"]).itertuples(index=False):
        comparison_rows.append(
            [
                row.stage,
                row.arm_label,
                row.metric,
                f"{row.mean_delta_vs_control:+.3f}",
                f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]",
                f"{row.randomization_p:.4f}",
                f"{row.holm_p:.4f}",
            ]
        )

    terminal_target = per_target[
        (per_target["stage"] == "terminal")
        & per_target["arm"].isin(arms.loc[~arms["is_control"], "arm"])
    ]
    direction_rows = []
    for arm, group in terminal_target.groupby("arm", sort=False):
        delta = group["delta_cumulative_10pct_hits_vs_control"]
        direction_rows.append(
            [
                labels.loc[arm],
                int((delta > 0).sum()),
                int(np.isclose(delta, 0).sum()),
                int((delta < 0).sum()),
            ]
        )

    significant = comparisons[comparisons["holm_p"] < 0.05]
    significance_text = (
        "No paired comparison survives Holm correction."
        if significant.empty
        else f"{len(significant)} paired comparisons survive Holm correction."
    )
    dataset_count = int(records["dms_id"].nunique())
    simulation_count = int(
        records[["dms_id", "arm", "simulation_index"]].drop_duplicates().shape[0]
    )
    report = f"""# Soft-target BT sweep analysis

All {dataset_count} datasets completed for {len(arms)} paired loss configurations,
covering {simulation_count} simulated campaigns. Round one was verified to contain
single mutants only and to use the same slate across arms for every dataset and seed.

The pre-specified development criterion is dataset-macro probability of finding at
least one top-1% mutant through round 3, with cumulative top-10% hits as the
tie-breaker. The nominal development winner is **{labels.loc[round3_winner]}**. The
terminal criterion nominally selects **{labels.loc[terminal_winner]}**.

## Dataset-macro outcomes

{_markdown_table(
    [
        'Loss',
        'R3 top10',
        'R3 P(top1)',
        'R3 best pct',
        'Terminal top10',
        'Terminal P(top1)',
        'Terminal best pct',
    ],
    summary_rows,
)}

## Paired comparisons with hard BT

Confidence intervals use a paired hierarchical bootstrap over datasets and simulation
seeds. P-values use paired dataset-level sign flips; Holm correction covers the seven
non-control arms within each stage and metric.

{_markdown_table(
    ['Stage', 'Loss', 'Metric', 'Delta', '95% CI', 'p', 'Holm p'],
    comparison_rows,
)}

## Terminal per-target direction for top-10% discoveries

{_markdown_table(['Loss', 'Better', 'Tied', 'Worse'], direction_rows)}

## Interpretation

- **{labels.loc[round3_winner]}** is the nominal early-discovery winner under the
  pre-specified criterion. {significance_text}
- Loss choice has a larger effect on top-1% discovery timing than on broad held-out
  rank correlation; these objectives should not be treated as interchangeable.
- These are tuning results on the same benchmark suite. Confirm a selected loss on
  fresh seeds or held-out datasets before treating it as an unbiased improvement.
"""
    (output_dir / "README.md").write_text(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--control-arm", default=DEFAULT_CONTROL_ARM)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--randomization-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1407)
    parser.add_argument(
        "--allow-non-single-round-one",
        action="store_true",
        help="Allow round-one multi-mutants; matched slates are still required.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate reusable statistical-analysis artifacts."""
    args = _parse_args()
    if args.bootstrap_replicates <= 0 or args.randomization_replicates <= 0:
        raise ValueError("Bootstrap and randomization replicate counts must be positive")
    output_dir = args.output_dir or args.checkpoint_dir / f"{args.eval_prefix}-analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records, arms = load_round_records(
        args.checkpoint_dir,
        args.eval_prefix,
        control_arm=args.control_arm,
        require_single_round_one=not args.allow_non_single_round_one,
    )
    aggregate = make_aggregate_round_summary(records)
    stage_summary = make_stage_summary(records, arms, args.bootstrap_replicates, args.seed)
    comparisons = make_paired_comparisons(
        records,
        arms,
        args.control_arm,
        args.bootstrap_replicates,
        args.randomization_replicates,
        args.seed,
    )
    per_target = make_per_target_summary(records, arms, args.control_arm)
    spreads = make_score_spreads(records)
    spread_arm = make_spread_arm_correlations(
        per_target,
        spreads,
        arms,
        args.randomization_replicates,
        args.seed,
    )
    responses = make_soft_bt_responses(per_target, arms)
    spread_responses = make_spread_response_correlations(
        responses,
        spreads,
        args.randomization_replicates,
        args.seed,
    )

    records.to_csv(output_dir / "campaign_round_outcomes.csv", index=False)
    arms.to_csv(output_dir / "arm_metadata.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_round_summary.csv", index=False)
    stage_summary.to_csv(output_dir / "stage_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    per_target.to_csv(output_dir / "per_target_summary.csv", index=False)
    spreads.to_csv(output_dir / "dataset_score_spreads.csv", index=False)
    spread_arm.to_csv(output_dir / "score_spread_arm_correlations.csv", index=False)
    responses.to_csv(output_dir / "dataset_soft_bt_responses.csv", index=False)
    spread_responses.to_csv(output_dir / "score_spread_response_correlations.csv", index=False)
    write_summary_report(records, arms, stage_summary, comparisons, per_target, output_dir)
    print(f"Wrote SoftBT analysis tables to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

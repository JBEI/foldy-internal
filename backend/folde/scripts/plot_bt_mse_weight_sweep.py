"""Generate publication-style analysis artifacts for the BT/MSE weight sweep.

The figures follow the visual conventions used in ``notebooks/jacob/paperplots``:
Okabe-Ito colors, cumulative top-10% discovery counts, top-1% discovery
probability, and separate single- and multi-mutation benchmark panels.

Run from ``backend/`` after all sweep checkpoints have completed::

    MPLCONFIGDIR=/tmp/folde-mpl ../.venv/bin/python \
        -m folde.scripts.plot_bt_mse_weight_sweep

The script never modifies checkpoint JSON files. It writes CSV summaries, a
Markdown report, and PNG/SVG figures to a sibling analysis directory.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from scipy import stats

from folde.scripts.run_bt_mse_weight_sweep import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_EVAL_PREFIX,
    DEFAULT_MSE_WEIGHTS,
    MULTI_MUTANT_DMS_IDS,
    SINGLE_MUTANT_DMS_IDS,
)
from folde.util import DMS_SHORTNAMES

WEIGHTS = tuple(float(weight) for weight in DEFAULT_MSE_WEIGHTS)
WEIGHT_LABELS = {
    0.0: "BT",
    0.05: "5% MSE",
    0.10: "10% MSE",
    0.20: "20% MSE",
    0.40: "40% MSE",
    0.70: "70% MSE",
    1.0: "MSE",
}
WEIGHT_COLORS = {
    0.0: "#000000",
    0.05: "#56B4E9",
    0.10: "#0072B2",
    0.20: "#009E73",
    0.40: "#E69F00",
    0.70: "#D55E00",
    1.0: "#CC79A7",
}
BENCHMARK_TITLES = {
    "single": "Single-Mutation Benchmark",
    "multi": "Multi-Mutation Benchmark",
}
STAGES: dict[str, dict[str, int]] = {
    "round3": {"single": 3, "multi": 3},
    "terminal": {"single": 6, "multi": 5},
}
STAGE_TITLES = {
    "round3": "Through Round 3",
    "terminal": "Terminal (Single R6; Multi R5)",
}
SUMMARY_METRICS = (
    "cumulative_10pct_hits",
    "has_found_top_1pct",
    "cumulative_1pct_hits",
    "best_percentile_so_far",
    "held_out_activity_spearman",
)
PRIMARY_METRICS = ("cumulative_10pct_hits", "has_found_top_1pct")


def _short_name(dms_id: str) -> str:
    """Return the paper short name, with a conservative fallback."""
    return DMS_SHORTNAMES.get(dms_id, dms_id.split("_")[0])


def _rng(seed: int, *parts: object) -> np.random.Generator:
    """Return a stable task-specific random generator."""
    token = "|".join(str(part) for part in parts).encode()
    checksum = zlib.crc32(token)
    return np.random.default_rng(np.random.SeedSequence([seed, checksum]))


def _weight_from_checkpoint(payload: Mapping[str, object]) -> float:
    """Extract and validate the sole MSE weight in a checkpoint."""
    campaign_results = payload.get("campaign_results")
    if not isinstance(campaign_results, list) or not campaign_results:
        raise ValueError("Checkpoint contains no campaign_results")

    observed: set[float] = set()
    for campaign in campaign_results:
        config_results = campaign["config_results"]
        if len(config_results) != 1:
            raise ValueError("Expected exactly one configuration per checkpoint")
        params = config_results[0]["config"]["few_shot_model_params"]
        observed.add(float(params["standardized_mse_weight"]))
    if len(observed) != 1:
        raise ValueError(f"Checkpoint contains multiple MSE weights: {sorted(observed)}")
    return observed.pop()


def load_round_records(checkpoint_dir: Path, eval_prefix: str) -> pd.DataFrame:
    """Load and validate every paired simulation as a tidy round-level table."""
    expected_datasets = {
        "single": set(SINGLE_MUTANT_DMS_IDS),
        "multi": set(MULTI_MUTANT_DMS_IDS),
    }
    rows: list[dict[str, object]] = []
    round_one_selections: dict[tuple[str, str, int, float], tuple[str, ...]] = {}

    for benchmark in ("single", "multi"):
        paths = sorted(checkpoint_dir.glob(f"{eval_prefix}-{benchmark}_*.json"))
        if len(paths) != len(WEIGHTS):
            raise ValueError(f"Expected {len(WEIGHTS)} {benchmark} checkpoints; found {len(paths)}")

        seen_weights: set[float] = set()
        for path in paths:
            payload = json.loads(path.read_text())
            weight = _weight_from_checkpoint(payload)
            if weight in seen_weights:
                raise ValueError(f"Duplicate {benchmark} checkpoint for w={weight:g}")
            seen_weights.add(weight)

            campaigns = payload["campaign_results"]
            observed_datasets = {campaign["dms_id"] for campaign in campaigns}
            if observed_datasets != expected_datasets[benchmark]:
                missing = sorted(expected_datasets[benchmark] - observed_datasets)
                unexpected = sorted(observed_datasets - expected_datasets[benchmark])
                raise ValueError(
                    f"Dataset mismatch in {path.name}: missing={missing}, "
                    f"unexpected={unexpected}"
                )

            for campaign in campaigns:
                dms_id = str(campaign["dms_id"])
                max_rounds = int(campaign["max_rounds"])
                round_size = int(campaign["round_size"])
                simulations = campaign["config_results"][0]["simulation_results"]
                expected_simulations = int(campaign["number_of_simulations"])
                if len(simulations) != expected_simulations:
                    raise ValueError(
                        f"{path.name}/{dms_id} has {len(simulations)} simulations; "
                        f"expected {expected_simulations}"
                    )

                for simulation_index, simulation in enumerate(simulations):
                    mutants = simulation["mutant_metrics"]
                    expected_mutants = max_rounds * round_size
                    if len(mutants) != expected_mutants:
                        raise ValueError(
                            f"{path.name}/{dms_id}/simulation {simulation_index} has "
                            f"{len(mutants)} selected mutants; expected {expected_mutants}"
                        )
                    round_one_selections[(benchmark, dms_id, simulation_index, weight)] = tuple(
                        mutant["seq_id"] for mutant in mutants if int(mutant["round_found"]) == 1
                    )
                    metrics_by_round = {
                        int(metric["round_num"]): metric for metric in simulation["round_metrics"]
                    }
                    if set(metrics_by_round) != set(range(1, max_rounds + 1)):
                        raise ValueError(
                            f"Round metrics are incomplete for {path.name}/{dms_id}/"
                            f"simulation {simulation_index}"
                        )

                    for round_num in range(1, max_rounds + 1):
                        cumulative = [
                            mutant for mutant in mutants if int(mutant["round_found"]) <= round_num
                        ]
                        current = [
                            mutant for mutant in mutants if int(mutant["round_found"]) == round_num
                        ]
                        round_metric = metrics_by_round[round_num]
                        misc = round_metric.get("misc", {})
                        rows.append(
                            {
                                "benchmark": benchmark,
                                "dms_id": dms_id,
                                "dms_shortname": _short_name(dms_id),
                                "mse_weight": weight,
                                "weight_label": WEIGHT_LABELS[weight],
                                "simulation_index": simulation_index,
                                "round_num": round_num,
                                "round_size": round_size,
                                "screened_mutants": round_num * round_size,
                                "variant_pool_size": int(simulation["variant_pool_size"]),
                                "cumulative_10pct_hits": sum(
                                    mutant["percentile"] >= 0.90 for mutant in cumulative
                                ),
                                "cumulative_1pct_hits": sum(
                                    mutant["percentile"] >= 0.99 for mutant in cumulative
                                ),
                                "has_found_top_1pct": float(
                                    any(mutant["percentile"] >= 0.99 for mutant in cumulative)
                                ),
                                "best_percentile_this_round": max(
                                    mutant["percentile"] for mutant in current
                                ),
                                "best_percentile_so_far": max(
                                    mutant["percentile"] for mutant in cumulative
                                ),
                                "model_spearman": round_metric.get("model_spearman", np.nan),
                                "held_out_activity_spearman": misc.get(
                                    "held_out_activity_spearman", np.nan
                                ),
                                "held_out_1pct_recall": misc.get("held_out_1pct_recall", np.nan),
                                "held_out_10pct_recall": misc.get("held_out_10pct_recall", np.nan),
                            }
                        )

        if seen_weights != set(WEIGHTS):
            raise ValueError(
                f"{benchmark} weights differ from plan: observed={sorted(seen_weights)}"
            )

    for benchmark, datasets in expected_datasets.items():
        for dms_id in datasets:
            for simulation_index in range(10):
                control = round_one_selections[(benchmark, dms_id, simulation_index, 0.0)]
                for weight in WEIGHTS[1:]:
                    candidate = round_one_selections[(benchmark, dms_id, simulation_index, weight)]
                    if candidate != control:
                        raise ValueError(
                            f"Round-one pairing differs for {benchmark}/{dms_id}/"
                            f"simulation {simulation_index}/w={weight:g}"
                        )

    records = pd.DataFrame(rows)
    return records.sort_values(
        ["benchmark", "dms_id", "mse_weight", "simulation_index", "round_num"]
    ).reset_index(drop=True)


def _t_interval(values: pd.Series) -> tuple[float, float]:
    """Return a dataset-level 95% t interval, or a point interval for one dataset."""
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
    """Average simulations within datasets and datasets within benchmarks."""
    dataset_means = (
        records.groupby(["benchmark", "dms_id", "mse_weight", "round_num"], as_index=False)[
            list(SUMMARY_METRICS)
        ]
        .mean()
        .sort_values(["benchmark", "mse_weight", "round_num", "dms_id"])
    )

    rows: list[dict[str, object]] = []
    group_columns = ["benchmark", "mse_weight", "round_num"]
    for keys, group in dataset_means.groupby(group_columns, sort=True):
        benchmark, weight, round_num = keys
        for metric in SUMMARY_METRICS:
            low, high = _t_interval(group[metric])
            rows.append(
                {
                    "benchmark": benchmark,
                    "mse_weight": weight,
                    "round_num": round_num,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "dataset_count": int(group["dms_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _metric_matrix(
    records: pd.DataFrame,
    benchmark: str,
    weight: float,
    round_num: int,
    metric: str,
) -> pd.DataFrame:
    """Return a paired dataset-by-simulation matrix for one arm."""
    subset = records[
        (records["benchmark"] == benchmark)
        & np.isclose(records["mse_weight"], weight)
        & (records["round_num"] == round_num)
    ]
    matrix = subset.pivot(index="dms_id", columns="simulation_index", values=metric)
    if matrix.isna().any().any():
        raise ValueError(f"Missing values in {benchmark}/w={weight:g}/round={round_num}/{metric}")
    return matrix.sort_index(axis=0).sort_index(axis=1)


def _bootstrap_benchmark(
    matrix: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Hierarchically resample paired datasets and simulations."""
    dataset_count, simulation_count = matrix.shape
    dataset_indices = rng.integers(0, dataset_count, size=(replicates, dataset_count))
    simulation_indices = rng.integers(0, simulation_count, size=(replicates, dataset_count))
    return matrix[dataset_indices, simulation_indices].mean(axis=1)


def _scope_estimate(
    records: pd.DataFrame,
    weight: float,
    rounds: Mapping[str, int],
    metric: str,
    scope: str,
    replicates: int,
    seed: int,
    *,
    paired_delta: bool,
) -> tuple[float, float, float]:
    """Estimate an arm mean or paired delta with hierarchical bootstrap bounds."""
    benchmark_estimates: dict[str, float] = {}
    benchmark_replicates: dict[str, np.ndarray] = {}
    for benchmark in ("single", "multi"):
        matrix = _metric_matrix(records, benchmark, weight, rounds[benchmark], metric)
        values = matrix.to_numpy(dtype=float)
        if paired_delta:
            control = _metric_matrix(records, benchmark, 0.0, rounds[benchmark], metric)
            if not matrix.index.equals(control.index) or not matrix.columns.equals(control.columns):
                raise ValueError(f"Pairing mismatch for {benchmark}/w={weight:g}/{metric}")
            values = values - control.to_numpy(dtype=float)
        benchmark_estimates[benchmark] = float(values.mean())
        benchmark_replicates[benchmark] = _bootstrap_benchmark(
            values,
            replicates,
            _rng(seed, weight, metric, scope, benchmark, paired_delta),
        )

    if scope in benchmark_estimates:
        estimate = benchmark_estimates[scope]
        bootstrap = benchmark_replicates[scope]
    elif scope == "macro":
        estimate = 0.5 * (benchmark_estimates["single"] + benchmark_estimates["multi"])
        bootstrap = 0.5 * (benchmark_replicates["single"] + benchmark_replicates["multi"])
    else:
        raise ValueError(f"Unknown scope: {scope}")

    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return estimate, float(low), float(high)


def _randomization_p_value(
    records: pd.DataFrame,
    weight: float,
    rounds: Mapping[str, int],
    metric: str,
    scope: str,
    replicates: int,
    seed: int,
) -> float:
    """Calculate a paired dataset-level sign-flip p-value."""
    values: list[float] = []
    coefficients: list[float] = []
    selected_benchmarks = (scope,) if scope in ("single", "multi") else ("single", "multi")
    for benchmark in selected_benchmarks:
        arm = _metric_matrix(records, benchmark, weight, rounds[benchmark], metric)
        control = _metric_matrix(records, benchmark, 0.0, rounds[benchmark], metric)
        dataset_deltas = (arm - control).mean(axis=1).to_numpy(dtype=float)
        stratum_weight = 1.0 if scope != "macro" else 0.5
        values.extend(dataset_deltas.tolist())
        coefficients.extend([stratum_weight / len(dataset_deltas)] * len(dataset_deltas))

    delta_values = np.asarray(values)
    coefficient_values = np.asarray(coefficients)
    observed = abs(float(delta_values @ coefficient_values))
    rng = _rng(seed, weight, metric, scope, "randomization")
    exceedances = 0
    completed = 0
    while completed < replicates:
        batch_size = min(10_000, replicates - completed)
        signs = rng.choice((-1.0, 1.0), size=(batch_size, len(delta_values)))
        null_values = np.abs((signs * delta_values) @ coefficient_values)
        exceedances += int(np.sum(null_values >= observed - 1e-15))
        completed += batch_size
    return (exceedances + 1.0) / (replicates + 1.0)


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Apply Holm's family-wise error correction."""
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return pd.Series(adjusted, index=p_values.index)


def make_stage_summary(records: pd.DataFrame, bootstrap_replicates: int, seed: int) -> pd.DataFrame:
    """Summarize round-three and terminal outcomes for each benchmark and macro-average."""
    rows: list[dict[str, object]] = []
    for stage, rounds in STAGES.items():
        for scope in ("single", "multi", "macro"):
            for weight in WEIGHTS:
                for metric in SUMMARY_METRICS:
                    mean, low, high = _scope_estimate(
                        records,
                        weight,
                        rounds,
                        metric,
                        scope,
                        bootstrap_replicates,
                        seed,
                        paired_delta=False,
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "scope": scope,
                            "mse_weight": weight,
                            "metric": metric,
                            "mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
    return pd.DataFrame(rows)


def make_paired_comparisons(
    records: pd.DataFrame,
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Compare every non-control weight with BT using paired hierarchical inference."""
    rows: list[dict[str, object]] = []
    for stage, rounds in STAGES.items():
        for scope in ("single", "multi", "macro"):
            for weight in WEIGHTS[1:]:
                for metric in SUMMARY_METRICS:
                    delta, low, high = _scope_estimate(
                        records,
                        weight,
                        rounds,
                        metric,
                        scope,
                        bootstrap_replicates,
                        seed,
                        paired_delta=True,
                    )
                    p_value = _randomization_p_value(
                        records,
                        weight,
                        rounds,
                        metric,
                        scope,
                        randomization_replicates,
                        seed,
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "scope": scope,
                            "mse_weight": weight,
                            "metric": metric,
                            "mean_delta_vs_bt": delta,
                            "ci_low": low,
                            "ci_high": high,
                            "randomization_p": p_value,
                        }
                    )
    comparisons = pd.DataFrame(rows)
    comparisons["holm_p"] = np.nan
    group_columns = ["stage", "scope", "metric"]
    for _, indices in comparisons.groupby(group_columns).groups.items():
        comparisons.loc[indices, "holm_p"] = _holm_adjust(
            comparisons.loc[indices, "randomization_p"]
        )
    return comparisons


def make_per_target_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-target arm means and paired deltas at both decision stages."""
    rows: list[dict[str, object]] = []
    metrics = list(SUMMARY_METRICS)
    for stage, rounds in STAGES.items():
        stage_parts = []
        for benchmark, round_num in rounds.items():
            subset = records[
                (records["benchmark"] == benchmark) & (records["round_num"] == round_num)
            ]
            means = (
                subset.groupby(
                    ["benchmark", "dms_id", "dms_shortname", "mse_weight"],
                    as_index=False,
                )[metrics]
                .mean()
                .copy()
            )
            stage_parts.append(means)
        stage_means = pd.concat(stage_parts, ignore_index=True)
        control = stage_means[np.isclose(stage_means["mse_weight"], 0.0)].set_index(
            ["benchmark", "dms_id"]
        )
        for row in stage_means.itertuples(index=False):
            result: dict[str, object] = {
                "stage": stage,
                "benchmark": row.benchmark,
                "dms_id": row.dms_id,
                "dms_shortname": row.dms_shortname,
                "mse_weight": row.mse_weight,
            }
            control_row = control.loc[(row.benchmark, row.dms_id)]
            for metric in metrics:
                value = float(getattr(row, metric))
                result[metric] = value
                result[f"delta_{metric}_vs_bt"] = value - float(control_row[metric])
            rows.append(result)
    return pd.DataFrame(rows)


def _save_figure(fig: Figure, output_dir: Path, stem: str) -> None:
    """Save a figure in editable SVG and high-resolution PNG formats."""
    fig.savefig(output_dir / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax: Axes, label: str) -> None:
    """Add a Jacob-style panel label."""
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def plot_campaign_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot the two discovery metrics across rounds for both benchmark strata."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), dpi=300, sharex="col")
    plot_metrics = (
        ("cumulative_10pct_hits", "Top 10% Mutant\nCount"),
        ("has_found_top_1pct", "Top 1% Mutant\nProbability"),
    )
    for column, benchmark in enumerate(("single", "multi")):
        for row_index, (metric, ylabel) in enumerate(plot_metrics):
            ax = axes[row_index, column]
            subset = summary[(summary["benchmark"] == benchmark) & (summary["metric"] == metric)]
            max_round = int(subset["round_num"].max())
            x_values = np.arange(0, max_round + 1) * 16
            if metric == "cumulative_10pct_hits":
                random_baseline = 0.10 * x_values
            else:
                random_baseline = 1.0 - np.power(0.99, x_values)
            ax.plot(
                x_values,
                random_baseline,
                color="#999999",
                linestyle="--",
                linewidth=1.0,
                label="random expectation",
                zorder=1,
            )

            for weight in WEIGHTS:
                arm = subset[np.isclose(subset["mse_weight"], weight)].sort_values("round_num")
                x = np.concatenate(([0.0], 16.0 * arm["round_num"].to_numpy()))
                y = np.concatenate(([0.0], arm["mean"].to_numpy()))
                ax.plot(
                    x,
                    y,
                    color=WEIGHT_COLORS[weight],
                    label=WEIGHT_LABELS[weight],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.6 if weight in (0.0, 1.0) else 1.2,
                    zorder=2,
                )

            if row_index == 0:
                ax.set_title(BENCHMARK_TITLES[benchmark], fontsize=10, fontweight="bold")
            if column == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
            if row_index == 1:
                ax.set_xlabel("Number of Screened Mutants")
            else:
                ax.set_xlabel("")
            if metric == "has_found_top_1pct":
                ax.set_ylim(0.0, 1.03)
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)

    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        fontsize=7.5,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.subplots_adjust(top=0.82, hspace=0.24, wspace=0.22)
    _save_figure(fig, output_dir, "campaign_discovery_curves")


def plot_weight_summary(stage_summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot benchmark and benchmark-macro outcomes as a function of MSE weight."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), dpi=300, sharex=True)
    weight_positions = np.arange(len(WEIGHTS))
    weight_tick_labels = ("0", ".05", ".1", ".2", ".4", ".7", "1")
    scope_setup = (
        ("single", "single", "#0072B2"),
        ("multi", "multi", "#D55E00"),
        ("macro", "benchmark macro", "#000000"),
    )
    metric_setup = (
        ("cumulative_10pct_hits", "Top 10% Mutant Count"),
        ("has_found_top_1pct", "Top 1% Mutant Probability"),
    )
    for column, stage in enumerate(("round3", "terminal")):
        for row_index, (metric, ylabel) in enumerate(metric_setup):
            ax = axes[row_index, column]
            for scope, label, color in scope_setup:
                subset = stage_summary[
                    (stage_summary["stage"] == stage)
                    & (stage_summary["scope"] == scope)
                    & (stage_summary["metric"] == metric)
                ].sort_values("mse_weight")
                yerr = np.vstack(
                    (
                        subset["mean"].to_numpy() - subset["ci_low"].to_numpy(),
                        subset["ci_high"].to_numpy() - subset["mean"].to_numpy(),
                    )
                )
                ax.errorbar(
                    weight_positions,
                    subset["mean"],
                    yerr=yerr,
                    label=label,
                    color=color,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.3,
                    capsize=2,
                    elinewidth=0.7,
                )
            if row_index == 0:
                ax.set_title(STAGE_TITLES[stage], fontsize=10, fontweight="bold")
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index == 1:
                ax.set_xlabel("Standardized-MSE Loss Share")
            if metric == "has_found_top_1pct":
                ax.set_ylim(0.0, 1.04)
            ax.set_xticks(weight_positions, labels=weight_tick_labels)
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3)
    fig.subplots_adjust(top=0.84, hspace=0.25, wspace=0.22)
    _save_figure(fig, output_dir, "weight_sweep_summary")


def plot_paired_deltas(comparisons: pd.DataFrame, output_dir: Path) -> None:
    """Plot benchmark-macro paired effects relative to pure BT."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.7), dpi=300, sharey=True)
    metric_setup = (
        ("cumulative_10pct_hits", "Δ Top 10% Mutant Count"),
        ("has_found_top_1pct", "Δ Top 1% Mutant Probability"),
    )
    weights = list(WEIGHTS[1:])
    y_positions = np.arange(len(weights))
    for column, stage in enumerate(("round3", "terminal")):
        for row_index, (metric, xlabel) in enumerate(metric_setup):
            ax = axes[row_index, column]
            subset = comparisons[
                (comparisons["stage"] == stage)
                & (comparisons["scope"] == "macro")
                & (comparisons["metric"] == metric)
            ].set_index("mse_weight")
            for y_position, weight in enumerate(weights):
                row = subset.loc[weight]
                ax.errorbar(
                    row["mean_delta_vs_bt"],
                    y_position,
                    xerr=np.array(
                        [
                            [row["mean_delta_vs_bt"] - row["ci_low"]],
                            [row["ci_high"] - row["mean_delta_vs_bt"]],
                        ]
                    ),
                    color=WEIGHT_COLORS[weight],
                    marker="o",
                    markersize=4,
                    capsize=2,
                    elinewidth=1,
                )
            ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([WEIGHT_LABELS[weight] for weight in weights])
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            if row_index == 0:
                ax.set_title(STAGE_TITLES[stage], fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    fig.suptitle("Paired benchmark-macro differences from pure BT", fontsize=10, y=1.02)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    _save_figure(fig, output_dir, "paired_deltas_vs_bt")


def plot_per_target_heatmap(per_target: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-target paired outcome differences from BT."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 10.5), dpi=300)
    dataset_order = [_short_name(dms_id) for dms_id in SINGLE_MUTANT_DMS_IDS + MULTI_MUTANT_DMS_IDS]
    heatmap_setup = (
        ("delta_cumulative_10pct_hits_vs_bt", "Δ Top 10% Count", ".1f"),
        ("delta_has_found_top_1pct_vs_bt", "Δ Top 1% Probability", ".2f"),
    )
    nonzero_weights = list(WEIGHTS[1:])
    for row_index, (metric, title, annotation_format) in enumerate(heatmap_setup):
        metric_max = float(per_target[metric].abs().max())
        for column, stage in enumerate(("round3", "terminal")):
            ax = axes[row_index, column]
            subset = per_target[
                (per_target["stage"] == stage) & per_target["mse_weight"].isin(nonzero_weights)
            ]
            pivot = subset.pivot(
                index="dms_shortname", columns="mse_weight", values=metric
            ).reindex(index=dataset_order, columns=nonzero_weights)
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="RdBu_r",
                center=0.0,
                vmin=-metric_max,
                vmax=metric_max,
                annot=True,
                fmt=annotation_format,
                annot_kws={"fontsize": 5.5},
                linewidths=0.25,
                linecolor="white",
                cbar_kws={"label": title, "shrink": 0.7},
            )
            ax.axhline(len(SINGLE_MUTANT_DMS_IDS), color="black", linewidth=1.2)
            if row_index == 0:
                ax.set_title(STAGE_TITLES[stage], fontsize=10, fontweight="bold")
            ax.set_xlabel("Standardized-MSE Loss Share")
            ax.set_ylabel("")
            if column == 1:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelrotation=0)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    fig.suptitle("Per-target mean paired differences from pure BT", fontsize=11, y=1.01)
    fig.subplots_adjust(hspace=0.18, wspace=0.10)
    _save_figure(fig, output_dir, "per_target_paired_heatmap")


def _plot_round_metric_curves(
    summary: pd.DataFrame,
    output_dir: Path,
    metric: str,
    ylabel: str,
    stem: str,
    *,
    percent_axis: bool = False,
) -> None:
    """Plot one round-level metric for both benchmark strata."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=300, sharey=percent_axis)
    for ax, benchmark in zip(axes, ("single", "multi"), strict=True):
        subset = summary[(summary["benchmark"] == benchmark) & (summary["metric"] == metric)]
        for weight in WEIGHTS:
            arm = subset[np.isclose(subset["mse_weight"], weight)].sort_values("round_num")
            ax.plot(
                arm["round_num"],
                arm["mean"],
                color=WEIGHT_COLORS[weight],
                label=WEIGHT_LABELS[weight],
                marker="o",
                markersize=3,
                linewidth=1.5 if weight in (0.0, 1.0) else 1.1,
            )
        ax.set_title(BENCHMARK_TITLES[benchmark], fontsize=10, fontweight="bold")
        ax.set_xlabel("Round")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        if percent_axis:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    if percent_axis:
        metric_values = summary.loc[summary["metric"] == metric, "mean"]
        axes[0].set_ylim(max(0.0, float(metric_values.min()) - 0.002), 1.0002)
        axes[1].tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel(ylabel)
    for label, ax in zip("ab", axes, strict=True):
        _panel_label(ax, label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        fontsize=7.5,
    )
    fig.subplots_adjust(top=0.76, wspace=0.12 if percent_axis else 0.18)
    _save_figure(fig, output_dir, stem)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a small dependency-free Markdown table."""
    lines = [
        " | ".join(headers),
        " | ".join("---" for _ in headers),
    ]
    lines.extend(" | ".join(str(value) for value in row) for row in rows)
    return "\n".join(lines)


def write_summary_report(
    stage_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    per_target: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a concise interpretation alongside the machine-readable tables."""
    macro = stage_summary[stage_summary["scope"] == "macro"]

    summary_rows = []
    for weight in WEIGHTS:
        values: list[object] = [f"{weight:g}"]
        for stage in ("round3", "terminal"):
            stage_weight = macro[
                (macro["stage"] == stage) & np.isclose(macro["mse_weight"], weight)
            ].set_index("metric")
            values.extend(
                [
                    f"{stage_weight.loc['cumulative_10pct_hits', 'mean']:.3f}",
                    f"{stage_weight.loc['has_found_top_1pct', 'mean']:.3f}",
                    f"{100 * stage_weight.loc['best_percentile_so_far', 'mean']:.3f}%",
                ]
            )
        summary_rows.append(values)

    comparison_rows = []
    macro_primary = comparisons[
        (comparisons["scope"] == "macro") & comparisons["metric"].isin(PRIMARY_METRICS)
    ]
    for row in macro_primary.itertuples(index=False):
        comparison_rows.append(
            [
                row.stage,
                f"{row.mse_weight:g}",
                row.metric,
                f"{row.mean_delta_vs_bt:+.3f}",
                f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]",
                f"{row.randomization_p:.4f}",
                f"{row.holm_p:.4f}",
            ]
        )

    round_three = macro[
        (macro["stage"] == "round3") & (macro["metric"].isin(PRIMARY_METRICS))
    ].pivot(index="mse_weight", columns="metric", values="mean")
    round_three_winner = max(
        WEIGHTS,
        key=lambda weight: (
            round_three.loc[weight, "has_found_top_1pct"],
            round_three.loc[weight, "cumulative_10pct_hits"],
        ),
    )
    terminal = macro[
        (macro["stage"] == "terminal") & (macro["metric"].isin(PRIMARY_METRICS))
    ].pivot(index="mse_weight", columns="metric", values="mean")
    terminal_winner = max(
        WEIGHTS,
        key=lambda weight: (
            terminal.loc[weight, "has_found_top_1pct"],
            terminal.loc[weight, "cumulative_10pct_hits"],
        ),
    )

    terminal_target = per_target[
        (per_target["stage"] == "terminal") & (per_target["mse_weight"] > 0)
    ]
    target_direction_rows = []
    for weight, group in terminal_target.groupby("mse_weight"):
        delta = group["delta_cumulative_10pct_hits_vs_bt"]
        target_direction_rows.append(
            [
                f"{weight:g}",
                int((delta > 0).sum()),
                int(np.isclose(delta, 0).sum()),
                int((delta < 0).sum()),
            ]
        )

    report = f"""# BT/MSE weight-sweep analysis

All 17 single-mutation and 3 multi-mutation datasets completed for seven paired loss
weights and ten simulation seeds per dataset. Round one was verified to be identical
across weights for every paired campaign.

The pre-specified development criterion is benchmark-macro probability of finding at
least one top-1% mutant through round 3, with cumulative top-10% hits as the tie-breaker.
Under that criterion the selected arm is **w={round_three_winner:g}**. The terminal
criterion nominally selects **w={terminal_winner:g}**, but terminal selection is
exploratory and its paired improvement is not statistically resolved.

## Benchmark-macro outcomes

{_markdown_table(
    [
        'MSE weight',
        'R3 top10',
        'R3 P(top1)',
        'R3 best pct',
        'Terminal top10',
        'Terminal P(top1)',
        'Terminal best pct',
    ],
    summary_rows,
)}

## Paired benchmark-macro comparisons with pure BT

Confidence intervals use a paired hierarchical bootstrap over datasets and simulation
seeds. P-values use paired dataset-level sign flips; Holm correction covers the six
non-control weights within each stage, scope, and metric.

{_markdown_table(
    ['Stage', 'Weight', 'Metric', 'Delta', '95% CI', 'p', 'Holm p'],
    comparison_rows,
)}

## Terminal per-target direction for top-10% discoveries

{_markdown_table(['Weight', 'Better', 'Tied', 'Worse'], target_direction_rows)}

## Interpretation

- Pure BT remains the development winner at round 3. No MSE mixture improves the
  primary top-1% discovery probability, and no round-3 top-10% delta survives
  multiplicity correction.
- At the terminal checkpoint, 40% MSE has a nominally higher macro top-1% discovery
  probability, while 5% MSE has the largest top-10% count. Both effects are small,
  their confidence intervals include zero, and no paired comparison survives Holm
  correction.
- Maximum percentile and held-out Spearman are effectively flat across the grid.
  Broad predictive ranking and discovery-policy performance should therefore not be
  treated as interchangeable.
- These are tuning results on the same benchmark suite. Confirm any selected weight on
  fresh seeds or held-out datasets before treating it as an unbiased improvement.
"""
    (output_dir / "README.md").write_text(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--randomization-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=991)
    return parser.parse_args()


def main() -> None:
    """Generate all sweep analysis artifacts."""
    args = _parse_args()
    if args.bootstrap_replicates <= 0 or args.randomization_replicates <= 0:
        raise ValueError("Bootstrap and randomization replicate counts must be positive")
    output_dir = args.output_dir or args.checkpoint_dir / f"{args.eval_prefix}-analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
        }
    )

    records = load_round_records(args.checkpoint_dir, args.eval_prefix)
    aggregate_summary = make_aggregate_round_summary(records)
    stage_summary = make_stage_summary(records, args.bootstrap_replicates, args.seed)
    comparisons = make_paired_comparisons(
        records,
        args.bootstrap_replicates,
        args.randomization_replicates,
        args.seed,
    )
    per_target = make_per_target_summary(records)

    records.to_csv(output_dir / "campaign_round_outcomes.csv", index=False)
    aggregate_summary.to_csv(output_dir / "aggregate_round_summary.csv", index=False)
    stage_summary.to_csv(output_dir / "stage_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    per_target.to_csv(output_dir / "per_target_summary.csv", index=False)

    plot_campaign_curves(aggregate_summary, output_dir)
    plot_weight_summary(stage_summary, output_dir)
    plot_paired_deltas(comparisons, output_dir)
    plot_per_target_heatmap(per_target, output_dir)
    _plot_round_metric_curves(
        aggregate_summary,
        output_dir,
        "best_percentile_so_far",
        "Mean Best Percentile Found",
        "maximum_percentile_curves",
        percent_axis=True,
    )
    _plot_round_metric_curves(
        aggregate_summary,
        output_dir,
        "held_out_activity_spearman",
        "Held-Out Activity Spearman",
        "heldout_spearman_curves",
    )
    write_summary_report(stage_summary, comparisons, per_target, output_dir)

    print(f"Wrote BT/MSE analysis artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

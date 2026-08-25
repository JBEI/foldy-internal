"""Relate DMS score spread to gains from adding standardized MSE loss.

This analysis uses the completed paired BT/MSE sweep and the raw ``DMS_score``
distribution for every tested dataset. It asks two complementary questions:

* At each MSE weight, do wider score distributions improve more over pure BT?
* Averaged across all nonzero weights, do wider datasets benefit more from MSE?

Spearman correlations are tested with dataset-label permutations. The primary
``score_range`` family is corrected over the six non-control weights; an additional
global correction covers all four spread definitions and six weights. The analysis
is repeated on all 20 datasets and on the 17 single-mutant datasets alone.

Run from ``backend/`` after the main sweep analysis::

    MPLCONFIGDIR=/tmp/folde-mpl ../.venv/bin/python \
        -m folde.scripts.analyze_bt_mse_score_spread
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from folde.data import DMS_DIR
from folde.scripts.plot_bt_mse_weight_sweep import (
    WEIGHTS,
    _panel_label,
    _save_figure,
    _short_name,
)
from folde.scripts.run_bt_mse_weight_sweep import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_EVAL_PREFIX,
    MULTI_MUTANT_DMS_IDS,
    SINGLE_MUTANT_DMS_IDS,
)

NONCONTROL_WEIGHTS = WEIGHTS[1:]
OUTCOMES = {
    "delta_cumulative_10pct_hits_vs_bt": "Top 10% Mutant Count",
    "delta_has_found_top_1pct_vs_bt": "Top 1% Mutant Probability",
}
SPREAD_METRICS = {
    "score_range": "Total range",
    "score_std": "Standard deviation",
    "score_iqr": "Interquartile range",
    "score_central90_range": "P95 - P05",
}
SCOPE_LABELS = {
    "all": "all 20 datasets",
    "single": "17 single-mutant datasets",
}
STAGE_LABELS = {
    "round3": "Through Round 3",
    "terminal": "Terminal (Single R6; Multi R5)",
}
RESPONSE_METRICS = {
    "mean_delta_across_mse_weights": "Mean gain across MSE weights",
    "linear_weight_response_slope": "Linear gain per unit MSE weight",
}


def _rng(seed: int, *parts: object) -> np.random.Generator:
    """Return a stable task-specific random number generator."""
    token = "|".join(str(part) for part in parts).encode()
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(token)]))


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Apply Holm's family-wise error correction."""
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return pd.Series(adjusted, index=p_values.index)


def _checkpoint_metadata(checkpoint_dir: Path, eval_prefix: str) -> pd.DataFrame:
    """Load dataset-level score and candidate-pool metadata from checkpoints."""
    rows: list[dict[str, object]] = []
    for benchmark in ("single", "multi"):
        paths = sorted(checkpoint_dir.glob(f"{eval_prefix}-{benchmark}_*.json"))
        if not paths:
            raise FileNotFoundError(f"No {benchmark} checkpoints found for {eval_prefix}")
        payload = json.loads(paths[0].read_text())
        for campaign in payload["campaign_results"]:
            simulations = campaign["config_results"][0]["simulation_results"]
            pool_sizes = {int(simulation["variant_pool_size"]) for simulation in simulations}
            if len(pool_sizes) != 1:
                raise ValueError(f"Variant-pool sizes vary for {campaign['dms_id']}")
            rows.append(
                {
                    "benchmark": benchmark,
                    "dms_id": str(campaign["dms_id"]),
                    "checkpoint_min": float(campaign["min_activity"]),
                    "checkpoint_median": float(campaign["median_activity"]),
                    "checkpoint_max": float(campaign["max_activity"]),
                    "variant_pool_size": pool_sizes.pop(),
                }
            )
    return pd.DataFrame(rows)


def make_score_spreads(checkpoint_dir: Path, eval_prefix: str) -> pd.DataFrame:
    """Calculate raw and robust DMS-score spread statistics for every dataset."""
    metadata = _checkpoint_metadata(checkpoint_dir, eval_prefix).set_index("dms_id")
    expected_ids = SINGLE_MUTANT_DMS_IDS + MULTI_MUTANT_DMS_IDS
    if set(metadata.index) != set(expected_ids):
        raise ValueError("Checkpoint datasets do not match the configured sweep datasets")

    rows: list[dict[str, object]] = []
    for dms_id in expected_ids:
        path = DMS_DIR / f"{dms_id}.csv"
        scores = pd.to_numeric(
            pd.read_csv(path, usecols=["DMS_score"])["DMS_score"], errors="coerce"
        ).dropna()
        values = scores.to_numpy(dtype=float)
        quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        checkpoint = metadata.loc[dms_id]
        raw_extrema = (float(values.min()), float(np.median(values)), float(values.max()))
        checkpoint_extrema = (
            float(checkpoint["checkpoint_min"]),
            float(checkpoint["checkpoint_median"]),
            float(checkpoint["checkpoint_max"]),
        )
        if not np.allclose(raw_extrema, checkpoint_extrema):
            raise ValueError(
                f"Raw score distribution does not match checkpoint metadata for {dms_id}"
            )

        score_iqr = float(quantiles[4] - quantiles[2])
        rows.append(
            {
                "benchmark": str(checkpoint["benchmark"]),
                "dms_id": dms_id,
                "dms_shortname": _short_name(dms_id),
                "score_count": len(values),
                "variant_pool_size": int(checkpoint["variant_pool_size"]),
                "score_min": raw_extrema[0],
                "score_q01": float(quantiles[0]),
                "score_q05": float(quantiles[1]),
                "score_q25": float(quantiles[2]),
                "score_median": raw_extrema[1],
                "score_q75": float(quantiles[4]),
                "score_q95": float(quantiles[5]),
                "score_q99": float(quantiles[6]),
                "score_max": raw_extrema[2],
                "score_range": raw_extrema[2] - raw_extrema[0],
                "score_std": float(values.std(ddof=1)),
                "score_iqr": score_iqr,
                "score_central90_range": float(quantiles[5] - quantiles[1]),
                "score_central98_range": float(quantiles[6] - quantiles[0]),
                "range_over_iqr": (raw_extrema[2] - raw_extrema[0]) / score_iqr,
                "upper_tail_over_iqr": float((quantiles[6] - quantiles[3]) / score_iqr),
            }
        )
    return pd.DataFrame(rows)


def _permutation_spearman(
    x_values: np.ndarray,
    y_values: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return Spearman rho and a two-sided dataset-label permutation p-value."""
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
        permuted = y_centered[orders]
        null_correlations = np.abs(permuted @ x_centered / denominator)
        exceedances += int(np.sum(null_correlations >= abs(observed) - 1e-15))
        completed += batch_size
    p_value = (exceedances + 1.0) / (replicates + 1.0)
    return observed, p_value


def _scope_filter(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Select all datasets or the single-mutant sensitivity subset."""
    if scope == "all":
        return frame
    if scope == "single":
        return frame[frame["benchmark"] == "single"]
    raise ValueError(f"Unknown scope: {scope}")


def make_weight_correlations(
    per_target: pd.DataFrame,
    spreads: pd.DataFrame,
    permutation_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Correlate spread with paired improvement separately at every loss weight."""
    merged = per_target.merge(spreads, on=["benchmark", "dms_id", "dms_shortname"])
    merged = merged[merged["mse_weight"] > 0].copy()
    rows: list[dict[str, object]] = []
    for scope in SCOPE_LABELS:
        scope_data = _scope_filter(merged, scope)
        for stage in STAGE_LABELS:
            stage_data = scope_data[scope_data["stage"] == stage]
            for outcome in OUTCOMES:
                for spread_metric in SPREAD_METRICS:
                    for weight in NONCONTROL_WEIGHTS:
                        arm = stage_data[np.isclose(stage_data["mse_weight"], weight)]
                        rho, p_value = _permutation_spearman(
                            arm[spread_metric].to_numpy(dtype=float),
                            arm[outcome].to_numpy(dtype=float),
                            permutation_replicates,
                            _rng(seed, scope, stage, outcome, spread_metric, weight),
                        )
                        rows.append(
                            {
                                "scope": scope,
                                "stage": stage,
                                "outcome": outcome,
                                "spread_metric": spread_metric,
                                "mse_weight": weight,
                                "dataset_count": len(arm),
                                "spearman_rho": rho,
                                "permutation_p": p_value,
                            }
                        )

    correlations = pd.DataFrame(rows)
    correlations["holm_within_spread_p"] = np.nan
    within_columns = ["scope", "stage", "outcome", "spread_metric"]
    for _, indices in correlations.groupby(within_columns).groups.items():
        correlations.loc[indices, "holm_within_spread_p"] = _holm_adjust(
            correlations.loc[indices, "permutation_p"]
        )

    correlations["holm_across_spreads_and_weights_p"] = np.nan
    global_columns = ["scope", "stage", "outcome"]
    for _, indices in correlations.groupby(global_columns).groups.items():
        correlations.loc[indices, "holm_across_spreads_and_weights_p"] = _holm_adjust(
            correlations.loc[indices, "permutation_p"]
        )
    return correlations


def make_dataset_weight_responses(per_target: pd.DataFrame) -> pd.DataFrame:
    """Reduce each dataset's six MSE arms to average and linear weight responses."""
    rows: list[dict[str, object]] = []
    for (stage, benchmark, dms_id, shortname), group in per_target.groupby(
        ["stage", "benchmark", "dms_id", "dms_shortname"], sort=True
    ):
        ordered = group.sort_values("mse_weight")
        if not np.allclose(ordered["mse_weight"].to_numpy(dtype=float), WEIGHTS):
            raise ValueError(f"Incomplete MSE grid for {stage}/{dms_id}")
        result: dict[str, object] = {
            "stage": stage,
            "benchmark": benchmark,
            "dms_id": dms_id,
            "dms_shortname": shortname,
        }
        weights = ordered["mse_weight"].to_numpy(dtype=float)
        for outcome in OUTCOMES:
            deltas = ordered[outcome].to_numpy(dtype=float)
            result[f"{outcome}__mean_delta_across_mse_weights"] = float(deltas[1:].mean())
            result[f"{outcome}__linear_weight_response_slope"] = float(
                np.polyfit(weights, deltas, 1)[0]
            )
        rows.append(result)
    return pd.DataFrame(rows)


def make_response_correlations(
    responses: pd.DataFrame,
    spreads: pd.DataFrame,
    permutation_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Test whether score spread predicts a dataset's overall MSE response."""
    merged = responses.merge(spreads, on=["benchmark", "dms_id", "dms_shortname"])
    rows: list[dict[str, object]] = []
    for scope in SCOPE_LABELS:
        scope_data = _scope_filter(merged, scope)
        for stage in STAGE_LABELS:
            stage_data = scope_data[scope_data["stage"] == stage]
            for outcome in OUTCOMES:
                for response_metric in RESPONSE_METRICS:
                    response_column = f"{outcome}__{response_metric}"
                    for spread_metric in SPREAD_METRICS:
                        rho, p_value = _permutation_spearman(
                            stage_data[spread_metric].to_numpy(dtype=float),
                            stage_data[response_column].to_numpy(dtype=float),
                            permutation_replicates,
                            _rng(seed, scope, stage, outcome, response_metric, spread_metric),
                        )
                        rows.append(
                            {
                                "scope": scope,
                                "stage": stage,
                                "outcome": outcome,
                                "response_metric": response_metric,
                                "spread_metric": spread_metric,
                                "dataset_count": len(stage_data),
                                "spearman_rho": rho,
                                "permutation_p": p_value,
                            }
                        )
    correlations = pd.DataFrame(rows)
    correlations["holm_p"] = np.nan
    family_columns = ["scope", "stage", "outcome"]
    for _, indices in correlations.groupby(family_columns).groups.items():
        correlations.loc[indices, "holm_p"] = _holm_adjust(
            correlations.loc[indices, "permutation_p"]
        )
    return correlations


def plot_weight_correlation_heatmap(correlations: pd.DataFrame, output_dir: Path) -> None:
    """Plot spread-improvement correlations across MSE weights."""
    all_data = correlations[correlations["scope"] == "all"]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.5), dpi=300)
    cbar_ax = fig.add_axes([0.92, 0.23, 0.018, 0.52])
    for row_index, (outcome, outcome_label) in enumerate(OUTCOMES.items()):
        for column, stage in enumerate(STAGE_LABELS):
            ax = axes[row_index, column]
            subset = all_data[(all_data["stage"] == stage) & (all_data["outcome"] == outcome)]
            rho = subset.pivot(
                index="spread_metric", columns="mse_weight", values="spearman_rho"
            ).reindex(index=SPREAD_METRICS, columns=NONCONTROL_WEIGHTS)
            adjusted = subset.pivot(
                index="spread_metric",
                columns="mse_weight",
                values="holm_across_spreads_and_weights_p",
            ).reindex(index=SPREAD_METRICS, columns=NONCONTROL_WEIGHTS)
            annotations = rho.copy().astype(object)
            for spread_metric in SPREAD_METRICS:
                for weight in NONCONTROL_WEIGHTS:
                    suffix = "*" if adjusted.loc[spread_metric, weight] < 0.05 else ""
                    annotations.loc[spread_metric, weight] = (
                        f"{rho.loc[spread_metric, weight]:+.2f}{suffix}"
                    )
            show_colorbar = row_index == 0 and column == 1
            sns.heatmap(
                rho,
                ax=ax,
                vmin=-1.0,
                vmax=1.0,
                center=0.0,
                cmap="RdBu_r",
                annot=annotations,
                fmt="",
                linewidths=0.4,
                linecolor="white",
                cbar=show_colorbar,
                cbar_ax=cbar_ax if show_colorbar else None,
                cbar_kws={"label": "Spearman rho"},
                annot_kws={"fontsize": 7},
            )
            if column == 0:
                ax.set_yticklabels(list(SPREAD_METRICS.values()), rotation=0)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
            ax.set_xticklabels([f"{weight:g}" for weight in NONCONTROL_WEIGHTS], rotation=0)
            ax.set_xlabel("Standardized-MSE Loss Share")
            ax.set_ylabel("")
            if row_index == 0:
                ax.set_title(STAGE_LABELS[stage], fontsize=10, fontweight="bold")
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    fig.suptitle(
        "Dataset score spread vs. improvement over pure BT\n"
        "* Holm-adjusted p < 0.05 across 24 spread-by-weight tests",
        fontsize=10,
        y=1.01,
    )
    fig.text(
        0.025,
        0.63,
        "Gain in Top-10% Count",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.text(
        0.025,
        0.24,
        "Gain in P(Top-1%)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.23, top=0.82, right=0.89, hspace=0.42, wspace=0.18)
    _save_figure(fig, output_dir, "score_spread_weight_correlations")


def plot_range_vs_average_gain(
    responses: pd.DataFrame,
    spreads: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot total score range against average gain across MSE-containing arms."""
    merged = responses.merge(spreads, on=["benchmark", "dms_id", "dms_shortname"])
    correlation_lookup = correlations[
        (correlations["scope"] == "all")
        & (correlations["spread_metric"] == "score_range")
        & (correlations["response_metric"] == "mean_delta_across_mse_weights")
    ].set_index(["stage", "outcome"])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), dpi=300, sharex=True)
    colors = {"single": "#0072B2", "multi": "#D55E00"}
    labels = {"single": "single", "multi": "multi"}
    for row_index, (outcome, outcome_label) in enumerate(OUTCOMES.items()):
        response_column = f"{outcome}__mean_delta_across_mse_weights"
        for column, stage in enumerate(STAGE_LABELS):
            ax = axes[row_index, column]
            subset = merged[merged["stage"] == stage]
            for benchmark in ("single", "multi"):
                group = subset[subset["benchmark"] == benchmark]
                ax.scatter(
                    group["score_range"],
                    group[response_column],
                    color=colors[benchmark],
                    edgecolor="white",
                    linewidth=0.4,
                    s=25,
                    label=labels[benchmark],
                    zorder=3,
                )
            coefficients = np.polyfit(subset["score_range"], subset[response_column], 1)
            x_line = np.linspace(subset["score_range"].min(), subset["score_range"].max(), 100)
            ax.plot(x_line, np.polyval(coefficients, x_line), color="black", linestyle="--")
            ax.axhline(0.0, color="#888888", linewidth=0.8, zorder=1)

            result = correlation_lookup.loc[(stage, outcome)]
            ax.text(
                0.03,
                0.96,
                f"rho={result['spearman_rho']:+.2f}; p={result['permutation_p']:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
            )
            if row_index == 0:
                ax.set_title(STAGE_LABELS[stage], fontsize=10, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"Mean gain in\n{outcome_label}")
            if row_index == 1:
                ax.set_xlabel("DMS Score Range (max - min)")
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    fig.subplots_adjust(top=0.84, hspace=0.35, wspace=0.24)
    _save_figure(fig, output_dir, "score_range_vs_average_mse_gain")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a compact Markdown table."""
    lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
    lines.extend(" | ".join(str(value) for value in row) for row in rows)
    return "\n".join(lines)


def write_report(
    spreads: pd.DataFrame,
    weight_correlations: pd.DataFrame,
    response_correlations: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the statistical results and their interpretation."""
    spread_rows = []
    for row in spreads.sort_values("score_range").itertuples(index=False):
        spread_rows.append(
            [
                row.dms_shortname,
                row.benchmark,
                f"{row.score_range:.3f}",
                f"{row.score_std:.3f}",
                f"{row.score_iqr:.3f}",
                f"{row.score_central90_range:.3f}",
            ]
        )

    overall_rows = []
    overall = response_correlations[
        (response_correlations["spread_metric"] == "score_range")
        & (response_correlations["response_metric"] == "mean_delta_across_mse_weights")
    ]
    for row in overall.itertuples(index=False):
        overall_rows.append(
            [
                SCOPE_LABELS[row.scope],
                row.stage,
                OUTCOMES[row.outcome],
                f"{row.spearman_rho:+.3f}",
                f"{row.permutation_p:.4f}",
                f"{row.holm_p:.4f}",
            ]
        )

    weight_rows = []
    range_correlations = weight_correlations[
        (weight_correlations["scope"] == "all")
        & (weight_correlations["spread_metric"] == "score_range")
    ]
    for row in range_correlations.itertuples(index=False):
        weight_rows.append(
            [
                row.stage,
                OUTCOMES[row.outcome],
                f"{row.mse_weight:g}",
                f"{row.spearman_rho:+.3f}",
                f"{row.permutation_p:.4f}",
                f"{row.holm_within_spread_p:.4f}",
                f"{row.holm_across_spreads_and_weights_p:.4f}",
            ]
        )

    notable = range_correlations[
        (range_correlations["stage"] == "terminal")
        & (range_correlations["outcome"] == "delta_has_found_top_1pct_vs_bt")
        & np.isclose(range_correlations["mse_weight"], 0.1)
    ].iloc[0]
    notable_single = weight_correlations[
        (weight_correlations["scope"] == "single")
        & (weight_correlations["stage"] == "terminal")
        & (weight_correlations["outcome"] == "delta_has_found_top_1pct_vs_bt")
        & (weight_correlations["spread_metric"] == "score_range")
        & np.isclose(weight_correlations["mse_weight"], 0.1)
    ].iloc[0]

    report = f"""# DMS score spread vs. BT/MSE sweep response

This analysis relates each tested dataset's raw DMS-score distribution to its paired
improvement over pure BT. It covers all 20 datasets and repeats the inference on the
17 single-mutant datasets so the three multi-mutant benchmarks cannot drive the result.
The raw minimum, median, and maximum match the values stored in the completed campaign
checkpoints for all 20 datasets.

## Main result

There is **no evidence that wider score ranges generally benefit more from adding
MSE**. When improvement is averaged across all six nonzero MSE weights, total score
range has a small and inconsistent association with discovery performance; none of
the associations below is statistically resolved.

{_markdown_table(
    ['Scope', 'Stage', 'Outcome', 'Spearman rho', 'Permutation p', 'Holm p'],
    overall_rows,
)}

The only multiplicity-resolved weight-specific association is at terminal 10% MSE for
top-1% discovery probability: rho={notable['spearman_rho']:+.3f}, permutation
p={notable['permutation_p']:.4f}, and 24-test Holm
p={notable['holm_across_spreads_and_weights_p']:.4f}. The sign is **negative**, the
opposite of the proposed hypothesis. The single-mutant sensitivity analysis is also
negative (rho={notable_single['spearman_rho']:+.3f},
p={notable_single['permutation_p']:.4f}). Because neighboring weights do not reproduce
the association, it should be treated as a localized exploratory signal rather than a
monotonic range-by-MSE effect.

## Total-range correlation at each MSE weight

`Weight Holm p` corrects the six loss weights for the displayed spread definition.
`Global Holm p` corrects all four spread definitions by six weights within each stage
and outcome.

{_markdown_table(
    [
        'Stage',
        'Outcome',
        'MSE weight',
        'rho',
        'p',
        'Weight Holm p',
        'Global Holm p',
    ],
    weight_rows,
)}

## Dataset score spreads

{_markdown_table(
    ['Dataset', 'Benchmark', 'Range', 'SD', 'IQR', 'P95-P05'],
    spread_rows,
)}

## Interpretation and limitations

- The MSE term in this sweep uses standardized activity labels, so absolute assay scale
  is removed before MSE is calculated. A direct benefit from a larger raw range was
  therefore not expected mechanistically.
- Total range is outlier-sensitive. Standard deviation, IQR, and P95-P05 give the same
  qualitative answer: no consistent positive spread-by-weight relationship.
- ProteinGym scores are directionality-normalized but not placed on one universal
  assay scale. Correlations with raw range can reflect assay construction, noise, or
  normalization rather than exploitable biological signal.
- There are only 20 datasets and top-1% probability is quantized in tenths from ten
  simulations. A fresh-dataset test would be needed before using score spread to choose
  a loss weight.
"""
    (output_dir / "score_spread_analysis.md").write_text(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2203)
    return parser.parse_args()


def main() -> None:
    """Run the score-spread association analysis and write all artifacts."""
    args = _parse_args()
    if args.permutation_replicates <= 0:
        raise ValueError("Permutation replicate count must be positive")
    output_dir = args.output_dir or args.checkpoint_dir / f"{args.eval_prefix}-analysis"
    per_target_path = output_dir / "per_target_summary.csv"
    if not per_target_path.exists():
        raise FileNotFoundError(f"Run plot_bt_mse_weight_sweep first; missing {per_target_path}")
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

    per_target = pd.read_csv(per_target_path)
    spreads = make_score_spreads(args.checkpoint_dir, args.eval_prefix)
    weight_correlations = make_weight_correlations(
        per_target, spreads, args.permutation_replicates, args.seed
    )
    responses = make_dataset_weight_responses(per_target)
    response_correlations = make_response_correlations(
        responses, spreads, args.permutation_replicates, args.seed
    )

    spreads.to_csv(output_dir / "dataset_score_spreads.csv", index=False)
    weight_correlations.to_csv(output_dir / "score_spread_weight_correlations.csv", index=False)
    responses.to_csv(output_dir / "dataset_mse_weight_responses.csv", index=False)
    response_correlations.to_csv(output_dir / "score_spread_response_correlations.csv", index=False)
    plot_weight_correlation_heatmap(weight_correlations, output_dir)
    plot_range_vs_average_gain(responses, spreads, response_correlations, output_dir)
    write_report(spreads, weight_correlations, response_correlations, output_dir)
    print(f"Wrote score-spread analysis artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

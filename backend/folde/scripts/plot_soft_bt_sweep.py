"""Plot publication-style figures from ``analyze_soft_bt_sweep`` outputs.

The figure set mirrors the BT/MSE sweep report: campaign discovery curves, a
loss-sweep summary, paired deltas from hard BT, a per-target heatmap, maximum
percentile and held-out-Spearman curves, and two score-spread response figures.

Run the statistical analysis first, then run from ``backend/``::

    MPLCONFIGDIR=/tmp/folde-mpl ../.venv/bin/python \
        -m folde.scripts.analyze_soft_bt_sweep
    MPLCONFIGDIR=/tmp/folde-mpl ../.venv/bin/python \
        -m folde.scripts.plot_soft_bt_sweep

Use ``--analysis-dir`` to reuse the plotter with a different SoftBT sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from folde.scripts.analyze_soft_bt_sweep import (
    DEFAULT_EVAL_PREFIX,
    PRIMARY_METRICS,
    SPREAD_METRICS,
    STAGES,
)
from folde.scripts.run_bt_mse_weight_sweep import DEFAULT_CHECKPOINT_DIR

OKABE_ITO_EXTENDED = (
    "#000000",
    "#999999",
    "#56B4E9",
    "#0072B2",
    "#009E73",
    "#F0E442",
    "#E69F00",
    "#D55E00",
    "#CC79A7",
)
STAGE_TITLES = {"round3": "Through Round 3", "terminal": "Terminal"}
METRIC_LABELS = {
    "cumulative_10pct_hits": "Top 10% Mutant Count",
    "has_found_top_1pct": "Top 1% Mutant Probability",
    "best_percentile_so_far": "Mean Best Percentile Found",
    "held_out_activity_spearman": "Held-Out Activity Spearman",
}
SPREAD_LABELS = {
    "score_range": "Total range",
    "score_std": "Standard deviation",
    "score_iqr": "Interquartile range",
    "score_central90_range": "P95 - P05",
}


def _read_table(analysis_dir: Path, name: str) -> pd.DataFrame:
    """Read one required analysis table with a useful missing-file error."""
    path = analysis_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run folde.scripts.analyze_soft_bt_sweep first")
    return pd.read_csv(path)


def load_analysis_tables(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all tables needed by the complete figure set."""
    return {
        "arms": _read_table(analysis_dir, "arm_metadata.csv").sort_values("arm_order"),
        "outcomes": _read_table(analysis_dir, "campaign_round_outcomes.csv"),
        "rounds": _read_table(analysis_dir, "aggregate_round_summary.csv"),
        "stages": _read_table(analysis_dir, "stage_summary.csv"),
        "comparisons": _read_table(analysis_dir, "paired_comparisons.csv"),
        "targets": _read_table(analysis_dir, "per_target_summary.csv"),
        "spreads": _read_table(analysis_dir, "dataset_score_spreads.csv"),
        "spread_arms": _read_table(analysis_dir, "score_spread_arm_correlations.csv"),
        "responses": _read_table(analysis_dir, "dataset_soft_bt_responses.csv"),
        "spread_responses": _read_table(analysis_dir, "score_spread_response_correlations.csv"),
    }


def _arm_order(arms: pd.DataFrame) -> list[str]:
    """Return arm identifiers in their validated analysis order."""
    return arms.sort_values("arm_order")["arm"].tolist()


def _arm_labels(arms: pd.DataFrame) -> Mapping[str, str]:
    """Map arm identifiers to compact plot labels."""
    return arms.set_index("arm")["arm_short_label"].to_dict()


def _arm_colors(arms: pd.DataFrame) -> Mapping[str, str]:
    """Assign stable colorblind-safe colors by arm order."""
    order = _arm_order(arms)
    if len(order) > len(OKABE_ITO_EXTENDED):
        colors = sns.color_palette("colorblind", n_colors=len(order)).as_hex()
    else:
        colors = list(OKABE_ITO_EXTENDED[: len(order)])
    return dict(zip(order, colors, strict=True))


def _target_labels(targets: pd.DataFrame) -> Mapping[str, str]:
    """Return unique display labels even when paper short names collide."""
    unique = targets[["dms_id", "dms_shortname"]].drop_duplicates()
    counts = unique["dms_shortname"].value_counts()
    labels: dict[str, str] = {}
    for row in unique.itertuples(index=False):
        label = str(row.dms_shortname)
        if counts[label] > 1:
            label = f"{label} · {str(row.dms_id).rsplit('_', 1)[-1]}"
        labels[str(row.dms_id)] = label
    return labels


def _save_figure(fig: Figure, output_dir: Path, stem: str) -> None:
    """Save editable SVG and high-resolution PNG copies."""
    fig.savefig(output_dir / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax: Axes, label: str) -> None:
    """Add a compact Jacob-style panel label."""
    ax.text(
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def plot_campaign_curves(
    summary: pd.DataFrame,
    arms: pd.DataFrame,
    output_dir: Path,
    round_size: int,
) -> None:
    """Plot cumulative enrichment and top-1% discovery probability by round."""
    order = _arm_order(arms)
    labels = _arm_labels(arms)
    colors = _arm_colors(arms)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=300)
    setup = (
        ("cumulative_10pct_hits", "Top 10% Mutant Count"),
        ("has_found_top_1pct", "Top 1% Mutant Probability"),
    )
    for ax, (metric, ylabel) in zip(axes, setup, strict=True):
        subset = summary[summary["metric"] == metric]
        max_round = int(subset["round_num"].max())
        x_values = np.arange(max_round + 1) * round_size
        random_baseline = (
            0.10 * x_values if metric == "cumulative_10pct_hits" else 1.0 - np.power(0.99, x_values)
        )
        ax.plot(
            x_values,
            random_baseline,
            color="#BBBBBB",
            linestyle="--",
            linewidth=1.0,
            label="random expectation",
        )
        for arm in order:
            arm_data = subset[subset["arm"] == arm].sort_values("round_num")
            x = np.concatenate(([0.0], round_size * arm_data["round_num"].to_numpy()))
            y = np.concatenate(([0.0], arm_data["mean"].to_numpy()))
            ax.plot(
                x,
                y,
                color=colors[arm],
                label=labels[arm],
                marker="o",
                markersize=3.2,
                linewidth=1.6 if bool(arms.set_index("arm").loc[arm, "is_control"]) else 1.2,
            )
        ax.set_xlabel("Number of Screened Mutants")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        if metric == "has_found_top_1pct":
            ax.set_ylim(0.0, 1.03)
    for label, ax in zip("ab", axes, strict=True):
        _panel_label(ax, label)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=3,
        fontsize=7.2,
    )
    fig.subplots_adjust(top=0.72, wspace=0.28)
    _save_figure(fig, output_dir, "campaign_discovery_curves")


def plot_loss_summary(stage_summary: pd.DataFrame, arms: pd.DataFrame, output_dir: Path) -> None:
    """Plot early and terminal outcomes for every loss formulation."""
    order = _arm_order(arms)
    labels = _arm_labels(arms)
    colors = _arm_colors(arms)
    positions = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.0), dpi=300, sharex=True)
    for column, stage in enumerate(STAGES):
        for row_index, metric in enumerate(PRIMARY_METRICS):
            ax = axes[row_index, column]
            subset = (
                stage_summary[
                    (stage_summary["stage"] == stage) & (stage_summary["metric"] == metric)
                ]
                .set_index("arm")
                .loc[order]
            )
            means = subset["mean"].to_numpy()
            yerr = np.vstack(
                (means - subset["ci_low"].to_numpy(), subset["ci_high"].to_numpy() - means)
            )
            for position, arm in enumerate(order):
                ax.errorbar(
                    position,
                    means[position],
                    yerr=yerr[:, position : position + 1],
                    color=colors[arm],
                    marker="o",
                    markersize=4,
                    capsize=2,
                    elinewidth=0.8,
                )
            if row_index == 0:
                ax.set_title(STAGE_TITLES[stage], fontsize=10, fontweight="bold")
            if column == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            if row_index == 1:
                ax.set_xticks(positions, [labels[arm] for arm in order], rotation=38, ha="right")
            if metric == "has_found_top_1pct":
                ax.set_ylim(0.0, 1.03)
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    fig.subplots_adjust(hspace=0.24, wspace=0.22)
    _save_figure(fig, output_dir, "loss_sweep_summary")


def plot_paired_deltas(comparisons: pd.DataFrame, arms: pd.DataFrame, output_dir: Path) -> None:
    """Plot paired effects relative to hard BT."""
    noncontrol = arms[~arms["is_control"]].sort_values("arm_order")
    order = noncontrol["arm"].tolist()
    labels = _arm_labels(arms)
    colors = _arm_colors(arms)
    positions = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2), dpi=300, sharey=True)
    for column, stage in enumerate(STAGES):
        for row_index, metric in enumerate(PRIMARY_METRICS):
            ax = axes[row_index, column]
            subset = comparisons[
                (comparisons["stage"] == stage) & (comparisons["metric"] == metric)
            ].set_index("arm")
            for position, arm in enumerate(order):
                row = subset.loc[arm]
                ax.errorbar(
                    row["mean_delta_vs_control"],
                    position,
                    xerr=np.array(
                        [
                            [row["mean_delta_vs_control"] - row["ci_low"]],
                            [row["ci_high"] - row["mean_delta_vs_control"]],
                        ]
                    ),
                    color=colors[arm],
                    marker="o",
                    markersize=4,
                    capsize=2,
                    elinewidth=1,
                )
            ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
            ax.set_yticks(positions, [labels[arm] for arm in order])
            ax.invert_yaxis()
            ax.set_xlabel(f"Δ {METRIC_LABELS[metric]}")
            if row_index == 0:
                ax.set_title(STAGE_TITLES[stage], fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    fig.suptitle("Paired differences from hard BT", fontsize=10, y=1.01)
    fig.subplots_adjust(hspace=0.36, wspace=0.20)
    _save_figure(fig, output_dir, "paired_deltas_vs_hard_bt")


def plot_per_target_heatmap(per_target: pd.DataFrame, arms: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-target paired outcome differences from hard BT."""
    noncontrol = arms[~arms["is_control"]].sort_values("arm_order")
    order = noncontrol["arm"].tolist()
    labels = _arm_labels(arms)
    target_labels = _target_labels(per_target)
    dataset_order = sorted(target_labels, key=lambda dms_id: target_labels[dms_id])
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), dpi=300)
    for row_index, metric in enumerate(PRIMARY_METRICS):
        column_name = f"delta_{metric}_vs_control"
        metric_max = max(float(per_target[column_name].abs().max()), 1e-12)
        for column, stage in enumerate(STAGES):
            ax = axes[row_index, column]
            subset = per_target[
                (per_target["stage"] == stage) & per_target["arm"].isin(order)
            ].copy()
            subset["arm_plot_label"] = subset["arm"].map(labels)
            pivot = subset.pivot(
                index="dms_id", columns="arm_plot_label", values=column_name
            ).reindex(index=dataset_order, columns=[labels[arm] for arm in order])
            pivot.index = [target_labels[dms_id] for dms_id in pivot.index]
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="RdBu_r",
                center=0.0,
                vmin=-metric_max,
                vmax=metric_max,
                annot=True,
                fmt=".2f" if metric == "has_found_top_1pct" else ".1f",
                annot_kws={"fontsize": 6},
                linewidths=0.25,
                linecolor="white",
                cbar_kws={"label": f"Δ {METRIC_LABELS[metric]}", "shrink": 0.75},
            )
            if row_index == 0:
                ax.set_title(STAGE_TITLES[stage], fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            if column == 1:
                ax.set_yticklabels([])
            ax.tick_params(axis="x", labelrotation=38, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    fig.suptitle("Per-target mean paired differences from hard BT", fontsize=11, y=1.01)
    fig.subplots_adjust(hspace=0.22, wspace=0.10)
    _save_figure(fig, output_dir, "per_target_paired_heatmap")


def plot_round_metric(
    summary: pd.DataFrame,
    arms: pd.DataFrame,
    output_dir: Path,
    metric: str,
    stem: str,
    *,
    percent_axis: bool = False,
) -> None:
    """Plot one round-level metric for all arms."""
    order = _arm_order(arms)
    labels = _arm_labels(arms)
    colors = _arm_colors(arms)
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=300)
    subset = summary[summary["metric"] == metric]
    for arm in order:
        arm_data = subset[subset["arm"] == arm].sort_values("round_num")
        ax.plot(
            arm_data["round_num"],
            arm_data["mean"],
            color=colors[arm],
            label=labels[arm],
            marker="o",
            markersize=3,
            linewidth=1.6 if bool(arms.set_index("arm").loc[arm, "is_control"]) else 1.1,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    if percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        values = subset["mean"]
        ax.set_ylim(max(0.0, float(values.min()) - 0.003), min(1.0002, float(values.max()) + 0.001))
    _panel_label(ax, "a")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=3,
        fontsize=7.2,
    )
    fig.subplots_adjust(top=0.72)
    _save_figure(fig, output_dir, stem)


def plot_spread_arm_correlations(
    correlations: pd.DataFrame, arms: pd.DataFrame, output_dir: Path
) -> None:
    """Plot score-spread correlations with paired gain for each arm."""
    noncontrol = arms[~arms["is_control"]].sort_values("arm_order")
    order = noncontrol["arm"].tolist()
    labels = _arm_labels(arms)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2), dpi=300)
    cbar_ax = fig.add_axes([0.92, 0.22, 0.018, 0.55])
    for row_index, metric in enumerate(PRIMARY_METRICS):
        for column, stage in enumerate(STAGES):
            ax = axes[row_index, column]
            subset = correlations[
                (correlations["stage"] == stage) & (correlations["metric"] == metric)
            ].copy()
            rho = subset.pivot(index="spread_metric", columns="arm", values="spearman_rho").reindex(
                index=list(SPREAD_METRICS), columns=order
            )
            adjusted = subset.pivot(
                index="spread_metric", columns="arm", values="holm_global_p"
            ).reindex(index=list(SPREAD_METRICS), columns=order)
            annotations = rho.copy().astype(object)
            for spread_metric in SPREAD_METRICS:
                for arm in order:
                    suffix = "*" if adjusted.loc[spread_metric, arm] < 0.05 else ""
                    annotations.loc[spread_metric, arm] = (
                        f"{rho.loc[spread_metric, arm]:.2f}{suffix}"
                    )
            sns.heatmap(
                rho,
                ax=ax,
                cmap="vlag",
                center=0.0,
                vmin=-1.0,
                vmax=1.0,
                annot=annotations,
                fmt="",
                annot_kws={"fontsize": 6},
                xticklabels=[labels[arm] for arm in order],
                yticklabels=[SPREAD_LABELS[name] for name in SPREAD_METRICS],
                cbar=column == 1 and row_index == 0,
                cbar_ax=cbar_ax if column == 1 and row_index == 0 else None,
                linewidths=0.25,
                linecolor="white",
            )
            if row_index == 0:
                ax.set_title(STAGE_TITLES[stage], fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(METRIC_LABELS[metric] if column == 0 else "")
            if column == 1:
                ax.set_yticklabels([])
            ax.tick_params(axis="x", labelrotation=38, labelsize=6.5)
            ax.tick_params(axis="y", labelsize=7)
    cbar_ax.set_ylabel("Spearman ρ", rotation=270, labelpad=12)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    fig.suptitle("Dataset score spread vs. paired loss-arm gain", fontsize=10, y=1.01)
    fig.subplots_adjust(right=0.89, hspace=0.28, wspace=0.12)
    _save_figure(fig, output_dir, "score_spread_arm_correlations")


def plot_score_range_response(
    responses: pd.DataFrame,
    spreads: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot score range against average gain across SoftBT configurations."""
    merged = responses.merge(spreads, on=["dms_id", "dms_shortname"])
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), dpi=300)
    for row_index, metric in enumerate(PRIMARY_METRICS):
        response_column = f"mean_delta_{metric}"
        for column, stage in enumerate(STAGES):
            ax = axes[row_index, column]
            subset = merged[merged["stage"] == stage]
            ax.scatter(
                subset["score_range"],
                subset[response_column],
                color="#0072B2",
                edgecolor="white",
                linewidth=0.5,
                s=28,
            )
            if subset["score_range"].nunique() > 1:
                coefficients = np.polyfit(subset["score_range"], subset[response_column], deg=1)
                x_values = np.linspace(subset["score_range"].min(), subset["score_range"].max())
                ax.plot(x_values, np.polyval(coefficients, x_values), color="#D55E00", lw=1)
            correlation = correlations[
                (correlations["stage"] == stage)
                & (correlations["metric"] == metric)
                & (correlations["response_type"] == "mean")
                & (correlations["spread_metric"] == "score_range")
            ].iloc[0]
            ax.text(
                0.03,
                0.96,
                f"ρ={correlation.spearman_rho:.2f}; Holm p={correlation.holm_p:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
            )
            ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
            if row_index == 0:
                ax.set_title(STAGE_TITLES[stage], fontsize=10, fontweight="bold")
            if row_index == 1:
                ax.set_xlabel("DMS Score Range")
            if column == 0:
                ax.set_ylabel(f"Mean SoftBT Δ\n{METRIC_LABELS[metric]}")
            ax.grid(alpha=0.15, linewidth=0.5)
    for label, ax in zip("abcd", axes.flatten(), strict=True):
        _panel_label(ax, label)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    _save_figure(fig, output_dir, "score_range_vs_average_soft_bt_gain")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    return parser.parse_args()


def main() -> None:
    """Generate the complete SoftBT figure set from reusable CSV tables."""
    args = _parse_args()
    analysis_dir = args.analysis_dir or args.checkpoint_dir / f"{args.eval_prefix}-analysis"
    output_dir = args.output_dir or analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_analysis_tables(analysis_dir)
    round_sizes = tables["outcomes"]["round_size"].drop_duplicates()
    terminal_rounds = (
        tables["outcomes"].loc[tables["outcomes"]["is_terminal"], "round_num"].drop_duplicates()
    )
    if len(round_sizes) != 1 or len(terminal_rounds) != 1:
        raise ValueError("The figure set requires one shared round size and terminal round")
    round_size = int(round_sizes.iloc[0])
    STAGE_TITLES["terminal"] = f"Terminal (Round {int(terminal_rounds.iloc[0])})"

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

    plot_campaign_curves(tables["rounds"], tables["arms"], output_dir, round_size)
    plot_loss_summary(tables["stages"], tables["arms"], output_dir)
    plot_paired_deltas(tables["comparisons"], tables["arms"], output_dir)
    plot_per_target_heatmap(tables["targets"], tables["arms"], output_dir)
    plot_round_metric(
        tables["rounds"],
        tables["arms"],
        output_dir,
        "best_percentile_so_far",
        "maximum_percentile_curves",
        percent_axis=True,
    )
    plot_round_metric(
        tables["rounds"],
        tables["arms"],
        output_dir,
        "held_out_activity_spearman",
        "heldout_spearman_curves",
    )
    plot_spread_arm_correlations(tables["spread_arms"], tables["arms"], output_dir)
    plot_score_range_response(
        tables["responses"],
        tables["spreads"],
        tables["spread_responses"],
        output_dir,
    )
    print(f"Wrote SoftBT figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

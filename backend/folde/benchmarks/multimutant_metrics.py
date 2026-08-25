"""Campaign and paired-comparison metrics for multimutant benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def campaign_metrics(
    measured_scores: Sequence[float],
    eligible_scores: Sequence[float],
    *,
    initial_measurement_count: int,
    measurement_batch_sizes: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Compute terminal metrics from measurements in chronological order."""
    measured = np.asarray(measured_scores, dtype=float)
    eligible = np.asarray(eligible_scores, dtype=float)
    if len(measured) < initial_measurement_count or initial_measurement_count <= 0:
        raise ValueError("initial_measurement_count is inconsistent with measured scores")
    if not np.isfinite(measured).all() or not np.isfinite(eligible).all() or len(eligible) == 0:
        raise ValueError("metric inputs must be finite and nonempty")
    if measurement_batch_sizes is None:
        best_curve = np.maximum.accumulate(measured)
    else:
        if any(size <= 0 for size in measurement_batch_sizes) or sum(
            measurement_batch_sizes
        ) != len(measured):
            raise ValueError("measurement_batch_sizes must partition measured scores")
        best_so_far = float("-inf")
        best_values: list[float] = []
        offset = 0
        for size in measurement_batch_sizes:
            best_so_far = max(best_so_far, float(np.max(measured[offset : offset + size])))
            best_values.extend([best_so_far] * size)
            offset += size
        best_curve = np.asarray(best_values)
    lower, upper = float(np.min(eligible)), float(np.max(eligible))
    scale = upper - lower
    normalized_best = 1.0 if scale == 0 else (float(best_curve[-1]) - lower) / scale
    percentile = float(np.mean(eligible <= best_curve[-1]))
    top_1_cutoff = float(np.quantile(eligible, 0.99))
    top_10_cutoff = float(np.quantile(eligible, 0.90))
    return {
        "best_dms_score": float(best_curve[-1]),
        "normalized_best_found": normalized_best,
        "best_percentile": percentile,
        "cumulative_top_1pct_hits": int(np.sum(measured >= top_1_cutoff)),
        "cumulative_top_10pct_hits": int(np.sum(measured >= top_10_cutoff)),
        "simple_regret": upper - float(best_curve[-1]),
        "area_under_best_found_curve": float(np.mean(best_curve)),
        "post_initial_area_under_best_found_curve": float(
            np.mean(best_curve[initial_measurement_count:])
            if len(best_curve) > initial_measurement_count
            else best_curve[-1]
        ),
    }


def paired_statistical_report(
    endpoint_by_arm_seed: Mapping[str, Mapping[int, float]],
    *,
    reference_arm: str = "plm_plus_folde",
    comparison_arm: str = "adjacent_folde",
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Return paired summaries and a deterministic percentile bootstrap CI."""
    if reference_arm not in endpoint_by_arm_seed or comparison_arm not in endpoint_by_arm_seed:
        raise ValueError("paired report arms are absent")
    reference = endpoint_by_arm_seed[reference_arm]
    comparison = endpoint_by_arm_seed[comparison_arm]
    common_seeds = sorted(set(reference) & set(comparison))
    if not common_seeds:
        raise ValueError("paired report has no common simulation seeds")
    differences = np.asarray(
        [reference[sim_seed] - comparison[sim_seed] for sim_seed in common_seeds],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True)
    bootstrap_medians = np.median(samples, axis=1)
    return {
        "reference_arm": reference_arm,
        "comparison_arm": comparison_arm,
        "common_seeds": common_seeds,
        "paired_differences": differences.tolist(),
        "median_paired_difference": float(np.median(differences)),
        "mean_paired_difference": float(np.mean(differences)),
        "bootstrap_statistic": "median_paired_difference",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_95pct_ci": [
            float(np.quantile(bootstrap_medians, 0.025)),
            float(np.quantile(bootstrap_medians, 0.975)),
        ],
        "wins": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "losses": int(np.sum(differences < 0)),
    }

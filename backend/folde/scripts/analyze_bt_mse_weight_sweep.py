"""Summarize paired campaign outcomes from the BT/MSE weight sweep.

The primary development criterion is the benchmark-macro probability of finding
at least one top-1% mutant within the first three rounds. Single- and
multi-mutant benchmark means receive equal weight. Cumulative top-10% hits are
reported as the secondary criterion, along with paired deltas from the fresh
``w=0`` control.

Run from ``backend/`` after the sweep completes::

    ../.venv/bin/python -m folde.scripts.analyze_bt_mse_weight_sweep
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable

from folde.scripts.run_bt_mse_weight_sweep import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_EVAL_PREFIX,
    MULTI_MUTANT_DMS_IDS,
    SINGLE_MUTANT_DMS_IDS,
)
from folde.types import ModelEvaluation


@dataclass(frozen=True)
class CampaignOutcome:
    """Terminal discovery metrics for one matched simulated campaign."""

    benchmark: str
    mse_weight: float
    dms_id: str
    simulation_index: int
    top_10pct_hits: int
    found_top_1pct: int

    @property
    def pair_key(self) -> tuple[str, int]:
        return self.dms_id, self.simulation_index


@dataclass(frozen=True)
class BenchmarkSummary:
    """Dataset-macro campaign outcomes for one loss weight and benchmark."""

    benchmark: str
    mse_weight: float
    dataset_count: int
    campaign_count: int
    top_10pct_hits: float
    found_top_1pct: float


def outcomes_from_evaluation(
    evaluation: ModelEvaluation,
    *,
    benchmark: str,
    through_round: int,
) -> list[CampaignOutcome]:
    """Extract paired discovery outcomes from a single-config checkpoint."""
    outcomes: list[CampaignOutcome] = []
    observed_weight: float | None = None
    for campaign in evaluation.campaign_results:
        if len(campaign.config_results) != 1:
            raise ValueError(
                f"Expected one config in {evaluation.name}/{campaign.dms_id}; "
                f"found {len(campaign.config_results)}"
            )
        config_result = campaign.config_results[0]
        config = config_result.config
        weight_value = config.few_shot_model_params.get("standardized_mse_weight")
        if weight_value is None:
            raise ValueError(f"{evaluation.name} is missing standardized_mse_weight")
        weight = float(weight_value)
        if observed_weight is None:
            observed_weight = weight
        elif weight != observed_weight:
            raise ValueError(f"{evaluation.name} contains multiple MSE weights")

        for simulation_index, simulation in enumerate(config_result.simulation_results):
            measured = [
                mutant
                for mutant in simulation.mutant_metrics
                if mutant.round_found <= through_round
            ]
            outcomes.append(
                CampaignOutcome(
                    benchmark=benchmark,
                    mse_weight=weight,
                    dms_id=campaign.dms_id,
                    simulation_index=simulation_index,
                    top_10pct_hits=sum(mutant.percentile >= 0.90 for mutant in measured),
                    found_top_1pct=int(any(mutant.percentile >= 0.99 for mutant in measured)),
                )
            )
    return outcomes


def load_sweep_outcomes(
    checkpoint_dir: Path,
    eval_prefix: str,
    *,
    through_round: int,
    allow_partial: bool,
) -> list[CampaignOutcome]:
    """Load every matching single- and multi-mutant sweep checkpoint."""
    all_outcomes: list[CampaignOutcome] = []
    expected_datasets = {
        "single": set(SINGLE_MUTANT_DMS_IDS),
        "multi": set(MULTI_MUTANT_DMS_IDS),
    }
    for benchmark in ("single", "multi"):
        paths = sorted(checkpoint_dir.glob(f"{eval_prefix}-{benchmark}_*.json"))
        if not paths:
            raise FileNotFoundError(
                f"No {benchmark} checkpoints matching {eval_prefix!r} in {checkpoint_dir}"
            )
        seen_weights: set[float] = set()
        for path in paths:
            evaluation = ModelEvaluation.model_validate_json(path.read_text())
            observed_datasets = {campaign.dms_id for campaign in evaluation.campaign_results}
            if not allow_partial and observed_datasets != expected_datasets[benchmark]:
                missing = sorted(expected_datasets[benchmark] - observed_datasets)
                unexpected = sorted(observed_datasets - expected_datasets[benchmark])
                raise ValueError(
                    f"Incomplete or unexpected datasets in {path}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            outcomes = outcomes_from_evaluation(
                evaluation,
                benchmark=benchmark,
                through_round=through_round,
            )
            if not outcomes:
                raise ValueError(f"Checkpoint has no simulation outcomes: {path}")
            weight = outcomes[0].mse_weight
            if weight in seen_weights:
                raise ValueError(f"Duplicate {benchmark} checkpoint for MSE weight {weight}")
            seen_weights.add(weight)
            all_outcomes.extend(outcomes)
    return all_outcomes


def summarize_outcomes(outcomes: Iterable[CampaignOutcome]) -> list[BenchmarkSummary]:
    """Average simulations within datasets, then datasets within each benchmark."""
    grouped: dict[tuple[str, float, str], list[CampaignOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[(outcome.benchmark, outcome.mse_weight, outcome.dms_id)].append(outcome)

    dataset_means: dict[tuple[str, float], list[tuple[float, float, int]]] = defaultdict(list)
    for (benchmark, weight, _), values in grouped.items():
        dataset_means[(benchmark, weight)].append(
            (
                fmean(value.top_10pct_hits for value in values),
                fmean(value.found_top_1pct for value in values),
                len(values),
            )
        )

    summaries = []
    for (benchmark, weight), values in dataset_means.items():
        summaries.append(
            BenchmarkSummary(
                benchmark=benchmark,
                mse_weight=weight,
                dataset_count=len(values),
                campaign_count=sum(value[2] for value in values),
                top_10pct_hits=fmean(value[0] for value in values),
                found_top_1pct=fmean(value[1] for value in values),
            )
        )
    return sorted(summaries, key=lambda summary: (summary.mse_weight, summary.benchmark))


def paired_deltas_from_control(
    outcomes: Iterable[CampaignOutcome], control_weight: float = 0.0
) -> dict[tuple[str, float], tuple[float, float]]:
    """Calculate within-dataset/seed mean deltas from a matched control."""
    by_arm: dict[tuple[str, float], dict[tuple[str, int], CampaignOutcome]] = defaultdict(dict)
    for outcome in outcomes:
        by_arm[(outcome.benchmark, outcome.mse_weight)][outcome.pair_key] = outcome

    deltas: dict[tuple[str, float], tuple[float, float]] = {}
    benchmarks = {benchmark for benchmark, _ in by_arm}
    for benchmark in benchmarks:
        control = by_arm.get((benchmark, control_weight))
        if control is None:
            raise ValueError(f"Missing w={control_weight:g} control for {benchmark}")
        for arm_key, arm in by_arm.items():
            arm_benchmark, weight = arm_key
            if arm_benchmark != benchmark:
                continue
            if set(arm) != set(control):
                raise ValueError(
                    f"Campaign pairing differs between {benchmark} w={control_weight:g} "
                    f"and w={weight:g}"
                )
            deltas[arm_key] = (
                fmean(arm[key].top_10pct_hits - control[key].top_10pct_hits for key in arm),
                fmean(arm[key].found_top_1pct - control[key].found_top_1pct for key in arm),
            )
    return deltas


def _macro_by_weight(
    summaries: Iterable[BenchmarkSummary],
) -> dict[float, tuple[float, float]]:
    by_weight: dict[float, list[BenchmarkSummary]] = defaultdict(list)
    for summary in summaries:
        by_weight[summary.mse_weight].append(summary)
    macro: dict[float, tuple[float, float]] = {}
    for weight, values in by_weight.items():
        if {value.benchmark for value in values} != {"single", "multi"}:
            raise ValueError(f"Weight {weight:g} does not have both benchmark strata")
        macro[weight] = (
            fmean(value.top_10pct_hits for value in values),
            fmean(value.found_top_1pct for value in values),
        )
    return macro


def render_report(
    outcomes: list[CampaignOutcome], summaries: list[BenchmarkSummary], through_round: int
) -> str:
    """Render the sweep table and development-winner decision."""
    by_key = {(summary.benchmark, summary.mse_weight): summary for summary in summaries}
    macro = _macro_by_weight(summaries)
    paired_deltas = paired_deltas_from_control(outcomes)
    weights = sorted(macro)

    lines = [
        f"BT/MSE sweep through round {through_round}",
        "",
        (
            "MSE weight | single top10 | single P(top1) | multi top10 | "
            "multi P(top1) | macro top10 | macro P(top1) | paired ΔP(top1)"
        ),
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for weight in weights:
        single = by_key[("single", weight)]
        multi = by_key[("multi", weight)]
        macro_top10, macro_top1 = macro[weight]
        macro_delta_top1 = fmean(
            paired_deltas[(benchmark, weight)][1] for benchmark in ("single", "multi")
        )
        lines.append(
            f"{weight:.3f} | {single.top_10pct_hits:.3f} | {single.found_top_1pct:.3f} | "
            f"{multi.top_10pct_hits:.3f} | {multi.found_top_1pct:.3f} | "
            f"{macro_top10:.3f} | {macro_top1:.3f} | {macro_delta_top1:+.3f}"
        )

    # The discovery-probability objective is primary; top-10% count breaks ties.
    winner = max(weights, key=lambda weight: (macro[weight][1], macro[weight][0]))
    lines.extend(
        [
            "",
            (
                f"Development winner: w={winner:g} "
                f"(macro P(top1)={macro[winner][1]:.3f}, "
                f"macro top10={macro[winner][0]:.3f})."
            ),
            (
                "This is a tuning result on the benchmark suite, not an unbiased final-test "
                "estimate; confirm the selected weight on fresh seeds or held-out datasets."
            ),
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    parser.add_argument("--through-round", type=int, default=3)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Analyze incomplete dataset checkpoints; winner selection may then be biased.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    if args.through_round <= 0:
        raise ValueError("--through-round must be positive")
    outcomes = load_sweep_outcomes(
        args.checkpoint_dir,
        args.eval_prefix,
        through_round=args.through_round,
        allow_partial=args.allow_partial,
    )
    summaries = summarize_outcomes(outcomes)
    print(render_report(outcomes, summaries, args.through_round))


if __name__ == "__main__":
    main()

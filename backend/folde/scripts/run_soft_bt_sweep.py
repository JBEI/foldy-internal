"""Run a paired gap-calibrated soft-target Bradley-Terry sweep.

The representation and campaign stack matches the completed BT/MSE sweep:

* E1-600m naturalness for round-one selection and MLP pretraining;
* ESMC-300M embeddings for the MLP inputs; and
* ten paired simulations per dataset with point-level holdout validation.

Multi-mutant campaigns use Jacob's one-mutation-at-a-time candidate pool: round
one ranks single mutants only, then later rounds can advance to adjacent
multi-mutants.

The soft-target arms map standardized activity gaps to preference probabilities::

    target(i > j) = sigmoid((activity_i - activity_j) / train_std / temperature)

Optional confidence arms multiply each pair by a bounded weight between ``floor``
and one. The default plan includes hard BT and raw-gap-weighted BT controls, three
soft-target temperatures, and the same temperatures with a 0.2 confidence floor.

Run from ``backend/``. Checkpoints are written after every completed dataset and an
interrupted run resumes automatically::

    ../.venv/bin/python -m folde.scripts.run_soft_bt_sweep --benchmark all

Use ``--dry-run`` to inspect the exact configurations without loading data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from folde.scripts.run_bt_mse_weight_sweep import (
    DEFAULT_CHECKPOINT_DIR,
    SINGLE_MUTANT_DMS_IDS,
    build_sweep_config,
    run_benchmark,
)
from folde.types import FolDEModelConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_TEMPERATURES = (0.25, 0.5, 1.0)
DEFAULT_CONFIDENCE_FLOORS = (0.2,)
DEFAULT_EVAL_PREFIX = "260823-soft-bt-sweep-v2-one-away"

# Every locally runnable assay dataset whose variant pool is majority multi-mutant
# and that has both ESMC-300M embeddings and E1-600M naturalness scores. Keep the
# three historical benchmarks first so partial results remain directly comparable
# while the broader long-range/higher-order suite runs.
LOCAL_MULTI_MUTANT_DMS_IDS = [
    "SPG1_STRSG_Olson_2014",
    "GRB2_HUMAN_Faure_2021",
    "PABP_YEAST_Melamed_2013",
    "PHOT_CHLRE_Chen_2023",
    "SPG1_STRSG_Wu_2016",
    "GFP_AEQVI_Sarkisyan_2016",
    "Q8WTC7_9CNID_Somermeyer_2022",
    "Q6WV12_9MAXI_Somermeyer_2022",
    "D7PM05_CLYGR_Somermeyer_2022",
    "RASK_HUMAN_Weng_2022_abundance",
    "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
]


def _float_token(value: float) -> str:
    """Return a checkpoint-safe, lossless token for a finite float."""
    if not np.isfinite(value):
        raise ValueError(f"Values must be finite; received {value}")
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def validate_temperatures(temperatures: Sequence[float]) -> list[float]:
    """Validate and de-duplicate positive temperatures while preserving order."""
    validated: list[float] = []
    for raw_temperature in temperatures:
        temperature = float(raw_temperature)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"Temperatures must be finite and positive; received {temperature}")
        if temperature in validated:
            raise ValueError(f"Duplicate soft-target temperature: {temperature}")
        validated.append(temperature)
    if not validated:
        raise ValueError("At least one soft-target temperature is required")
    return validated


def validate_confidence_floors(floors: Sequence[float]) -> list[float]:
    """Validate and de-duplicate confidence floors while preserving order."""
    validated: list[float] = []
    for raw_floor in floors:
        floor = float(raw_floor)
        if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(f"Confidence floors must be between 0 and 1; received {floor}")
        if floor in validated:
            raise ValueError(f"Duplicate confidence floor: {floor}")
        validated.append(floor)
    return validated


def _config_with_loss(
    *,
    name: str,
    device: str,
    activity_difference_weighting: bool = False,
    soft_target_temperature: float | None = None,
    soft_target_confidence_floor: float | None = None,
) -> FolDEModelConfig:
    """Clone the established representation stack with one BT loss formulation."""
    payload = build_sweep_config(0.0, device=device).model_dump()
    payload["name"] = name
    params = dict(payload["few_shot_model_params"])
    params["standardized_mse_weight"] = 0.0
    params["bt_activity_difference_weighting"] = activity_difference_weighting
    if soft_target_temperature is not None:
        params["bt_soft_target_temperature"] = soft_target_temperature
    if soft_target_confidence_floor is not None:
        params["bt_soft_target_confidence_floor"] = soft_target_confidence_floor
    payload["few_shot_model_params"] = params
    return FolDEModelConfig.model_validate(payload)


def build_soft_bt_configs(
    temperatures: Sequence[float],
    confidence_floors: Sequence[float],
    *,
    device: str = "cuda",
    include_hard_control: bool = True,
    include_raw_gap_control: bool = True,
) -> list[FolDEModelConfig]:
    """Build ordered control, soft-target, and bounded-confidence sweep arms."""
    validated_temperatures = validate_temperatures(temperatures)
    validated_floors = validate_confidence_floors(confidence_floors)
    configs: list[FolDEModelConfig] = []
    if include_hard_control:
        configs.append(_config_with_loss(name="E1E1-300m-BT-hard", device=device))
    if include_raw_gap_control:
        configs.append(
            _config_with_loss(
                name="E1E1-300m-BT-rawgap",
                device=device,
                activity_difference_weighting=True,
            )
        )

    for temperature in validated_temperatures:
        temperature_token = _float_token(temperature)
        configs.append(
            _config_with_loss(
                name=f"E1E1-300m-SoftBT-t{temperature_token}",
                device=device,
                soft_target_temperature=temperature,
            )
        )
        for floor in validated_floors:
            configs.append(
                _config_with_loss(
                    name=(f"E1E1-300m-SoftBT-t{temperature_token}-" f"f{_float_token(floor)}"),
                    device=device,
                    soft_target_temperature=temperature,
                    soft_target_confidence_floor=floor,
                )
            )
    if not configs:
        raise ValueError("The sweep plan contains no configurations")
    return configs


def configs_for_benchmark(
    configs: Sequence[FolDEModelConfig], benchmark: str
) -> list[FolDEModelConfig]:
    """Apply the one-away constraint to multi-mutant campaigns only."""
    if benchmark not in {"single", "multi"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    one_mutation_at_a_time = benchmark == "multi"
    return [
        config.model_copy(
            update={"one_mutation_at_a_time": one_mutation_at_a_time},
            deep=True,
        )
        for config in configs
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("single", "multi", "all"),
        default="all",
        help="Benchmark stratum to run (default: all).",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEMPERATURES),
        help="Soft-target temperatures in training-label standard-deviation units.",
    )
    parser.add_argument(
        "--confidence-floors",
        nargs="*",
        type=float,
        default=list(DEFAULT_CONFIDENCE_FLOORS),
        help="Bounded confidence floors; pass with no values to omit confidence arms.",
    )
    parser.add_argument("--no-hard-control", action="store_true")
    parser.add_argument("--no-raw-gap-control", action="store_true")
    parser.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--number-of-simulations", type=int, default=10)
    parser.add_argument("--round-size", type=int, default=16)
    parser.add_argument("--single-rounds", type=int, default=6)
    parser.add_argument("--multi-rounds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--dms-ids",
        nargs="+",
        default=None,
        help="Run only these IDs. Requires --benchmark single or multi.",
    )
    parser.add_argument("--single-workers", type=int, default=2)
    parser.add_argument("--multi-workers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_positive_args(args: argparse.Namespace) -> None:
    for name in (
        "number_of_simulations",
        "round_size",
        "single_rounds",
        "multi_rounds",
        "single_workers",
        "multi_workers",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers is not None and args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.dms_ids is not None and args.benchmark == "all":
        raise ValueError("--dms-ids requires --benchmark single or --benchmark multi")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    _validate_positive_args(args)
    configs = build_soft_bt_configs(
        args.temperatures,
        args.confidence_floors,
        device=args.device,
        include_hard_control=not args.no_hard_control,
        include_raw_gap_control=not args.no_raw_gap_control,
    )
    benchmarks = ("single", "multi") if args.benchmark == "all" else (args.benchmark,)
    plan = {
        "eval_prefix": args.eval_prefix,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "benchmarks": list(benchmarks),
        "dataset_counts": {
            "single": (
                len(args.dms_ids)
                if args.dms_ids is not None and args.benchmark == "single"
                else len(SINGLE_MUTANT_DMS_IDS)
            ),
            "multi": (
                len(args.dms_ids)
                if args.dms_ids is not None and args.benchmark == "multi"
                else len(LOCAL_MULTI_MUTANT_DMS_IDS)
            ),
        },
        "rounds": {"single": args.single_rounds, "multi": args.multi_rounds},
        "round_size": args.round_size,
        "number_of_simulations": args.number_of_simulations,
        "random_seed": args.random_seed,
        "workers": {
            "single": args.num_workers or args.single_workers,
            "multi": args.num_workers or args.multi_workers,
        },
        "configs": {
            benchmark: [
                config.model_dump(mode="json")
                for config in configs_for_benchmark(configs, benchmark)
            ]
            for benchmark in benchmarks
        },
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for benchmark in benchmarks:
        benchmark_configs = configs_for_benchmark(configs, benchmark)
        dms_ids_override = args.dms_ids
        if dms_ids_override is None and benchmark == "multi":
            dms_ids_override = LOCAL_MULTI_MUTANT_DMS_IDS
        run_benchmark(
            benchmark=benchmark,
            configs=benchmark_configs,
            eval_prefix=args.eval_prefix,
            checkpoint_dir=args.checkpoint_dir,
            number_of_simulations=args.number_of_simulations,
            round_size=args.round_size,
            max_rounds=args.single_rounds if benchmark == "single" else args.multi_rounds,
            random_seed=args.random_seed,
            num_workers=(
                args.num_workers
                or (args.single_workers if benchmark == "single" else args.multi_workers)
            ),
            overwrite=args.overwrite,
            dms_ids_override=dms_ids_override,
        )
        LOGGER.info(f"Completed {benchmark} soft-target BT sweep")


if __name__ == "__main__":
    main()

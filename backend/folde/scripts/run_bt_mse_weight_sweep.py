"""Run a paired Bradley-Terry/standardized-MSE FolDE weight sweep.

The sweep holds the previously selected representation stack fixed:

* E1-600m naturalness for round-one selection;
* E1-600m naturalness as the MLP warm-start target; and
* ESMC-300M embeddings for the MLP inputs.

Only ``standardized_mse_weight`` changes between arms. The mixed objective is
``(1 - w) * BT + w * standardized_MSE`` and is used for both naturalness
pretraining and activity fine-tuning. All arms use point-level holdout
validation, including the pure-BT control, so the validation protocol is not a
confound in the weight comparison.

Run from ``backend/``. Checkpoints are written after every completed dataset
and an interrupted run resumes automatically::

    ../.venv/bin/python -m folde.scripts.run_bt_mse_weight_sweep --benchmark all

Use ``--dry-run`` to print the exact configurations without loading data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import torch

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.data import EMBEDDINGS_DIR
from folde.scripts.convert_embedding_csv_to_binary import convert_embedding_csv
from folde.types import FolDEModelConfig

LOGGER = logging.getLogger(__name__)

SINGLE_MUTANT_DMS_IDS = [
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    "SC6A4_HUMAN_Young_2021",
    "PTEN_HUMAN_Mighell_2018",
    "S22A1_HUMAN_Yee_2023_activity",
    "KKA2_KLEPN_Melnikov_2014",
    "PPARG_HUMAN_Majithia_2016",
    "MET_HUMAN_Estevam_2023",
    "MTHR_HUMAN_Weile_2021",
    "LGK_LIPST_Klesmith_2015",
    "AMIE_PSEAE_Wrenbeck_2017",
    "PAI1_HUMAN_Huttinger_2021",
    "A4GRB6_PSEAI_Chen_2020",
    "MSH2_HUMAN_Jia_2020",
    "MLAC_ECOLI_MacRae_2023",
    "RNC_ECOLI_Weeks_2023",
    "HMDH_HUMAN_Jiang_2019",
    "CAS9_STRP1_Spencer_2017_positive",
]

MULTI_MUTANT_DMS_IDS = [
    "SPG1_STRSG_Olson_2014",
    "GRB2_HUMAN_Faure_2021",
    "PABP_YEAST_Melamed_2013",
]

DEFAULT_MSE_WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_EVAL_PREFIX = "260819-bt-mse-sweep-v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_DIR = REPOSITORY_ROOT / "folde" / "model_evals"


def _weight_token(weight: float) -> str:
    """Return a checkpoint-safe, lossless token for a weight."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"MSE weights must be between 0 and 1; received {weight}")
    text = f"{weight:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def validate_weights(weights: Sequence[float]) -> list[float]:
    """Validate and de-duplicate a weight grid while preserving order."""
    validated: list[float] = []
    for raw_weight in weights:
        weight = float(raw_weight)
        _weight_token(weight)
        if weight in validated:
            raise ValueError(f"Duplicate MSE weight: {weight}")
        validated.append(weight)
    if not validated:
        raise ValueError("At least one MSE weight is required")
    return validated


def build_sweep_config(mse_weight: float, device: str = "cuda") -> FolDEModelConfig:
    """Build one arm while holding every parameter except loss weight fixed."""
    weight = float(mse_weight)
    token = _weight_token(weight)
    few_shot_params: dict[str, object] = {
        "pretrain": True,
        "pretrain_epochs": 50,
        "ensemble_size": 5,
        "embedding_dim": 960,
        "hidden_dims": [100, 50],
        "dropout": 0.2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "train_epochs": 200,
        "train_patience": 40,
        "val_frequency": 10,
        "use_mse_loss": False,
        "standardized_mse_weight": weight,
        "bt_activity_difference_weighting": False,
        # Final per-round held-out metrics are still computed by the campaign.
        # Re-scoring the held-out set at every optimizer validation step is redundant.
        "track_test_metrics_during_training": False,
        # Mixed pointwise/pairwise validation must hold out complete samples.
        # Keep this protocol for w=0 too, so weight is the only changed factor.
        "do_holdout_validation": True,
        "do_validation_with_pair_fraction": None,
        "decision_mode": "constantliar",
        "lie_noise_stddev_multiplier_schedule": [6.0] * 2 + [100.0] * 8,
    }
    if device != "auto":
        few_shot_params["device"] = device

    return FolDEModelConfig(
        name=f"E1E1-300m-BTMSE-mse{token}",
        naturalness_model_id="E1-600m_melted",
        few_shot_pretrain_naturalness_model_id="E1-600m_melted",
        embedding_model_id="300m",
        zero_shot_model_name="NaturalnessZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="TorchMLPFewShotModel",
        few_shot_model_params=few_shot_params,
    )


def build_sweep_configs(weights: Sequence[float], device: str = "cuda") -> list[FolDEModelConfig]:
    """Build the ordered set of paired sweep arms."""
    return [build_sweep_config(weight, device=device) for weight in validate_weights(weights)]


def run_benchmark(
    *,
    benchmark: str,
    configs: list[FolDEModelConfig],
    eval_prefix: str,
    checkpoint_dir: Path,
    number_of_simulations: int,
    round_size: int,
    max_rounds: int,
    random_seed: int,
    num_workers: int,
    overwrite: bool,
    dms_ids_override: Sequence[str] | None = None,
) -> None:
    """Run one benchmark stratum with per-config, per-dataset checkpoints."""
    if dms_ids_override is not None:
        dms_ids = list(dms_ids_override)
    elif benchmark == "single":
        dms_ids = SINGLE_MUTANT_DMS_IDS
    elif benchmark == "multi":
        dms_ids = MULTI_MUTANT_DMS_IDS
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    simulate_campaigns_with_config_checkpoints(
        eval_prefix=f"{eval_prefix}-{benchmark}",
        dms_ids=dms_ids,
        config_list=configs,
        checkpoint_dir=str(checkpoint_dir),
        overwrite=overwrite,
        round_size=round_size,
        number_of_simulations=number_of_simulations,
        activity_column="DMS_score",
        max_rounds=max_rounds,
        random_seed=random_seed,
        num_workers=num_workers,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("single", "multi", "all"),
        default="all",
        help="Benchmark stratum to run (default: all).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_MSE_WEIGHTS),
        help="Standardized-MSE shares to sweep.",
    )
    parser.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--number-of-simulations", type=int, default=10)
    parser.add_argument("--round-size", type=int, default=16)
    parser.add_argument(
        "--single-rounds",
        type=int,
        default=6,
        help="Rounds for the 17 single-mutant datasets (default: 6).",
    )
    parser.add_argument(
        "--multi-rounds",
        type=int,
        default=5,
        help="Rounds for the 3 multi-mutant datasets, matching recent runs (default: 5).",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--dms-ids",
        nargs="+",
        default=None,
        help="Run only these dataset IDs. Requires --benchmark single or multi.",
    )
    parser.add_argument("--single-workers", type=int, default=2)
    parser.add_argument("--multi-workers", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override both worker counts (primarily for equivalence tests).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="Training device. CUDA is explicit by default to avoid accidental CPU sweeps.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prepare-embedding-caches",
        action="store_true",
        help="Build fresh memory-mapped float32 caches before starting the sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs and planned workload without loading datasets.",
    )
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
    configs = build_sweep_configs(args.weights, device=args.device)
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
                else len(MULTI_MUTANT_DMS_IDS)
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
        "prepare_embedding_caches": args.prepare_embedding_caches,
        "configs": [config.model_dump(mode="json") for config in configs],
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_embedding_caches:
        selected_dms_ids: list[str] = []
        for benchmark in benchmarks:
            if args.dms_ids is not None:
                benchmark_dms_ids = args.dms_ids
            elif benchmark == "single":
                benchmark_dms_ids = SINGLE_MUTANT_DMS_IDS
            else:
                benchmark_dms_ids = MULTI_MUTANT_DMS_IDS
            for dms_id in benchmark_dms_ids:
                if dms_id not in selected_dms_ids:
                    selected_dms_ids.append(dms_id)

        for dms_id in selected_dms_ids:
            embedding_path = EMBEDDINGS_DIR / f"{dms_id}_embedding_300m.csv"
            LOGGER.info(f"Preparing binary embedding cache for {dms_id}")
            convert_embedding_csv(embedding_path)

    for benchmark in benchmarks:
        run_benchmark(
            benchmark=benchmark,
            configs=configs,
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
            dms_ids_override=args.dms_ids,
        )
        LOGGER.info(f"Completed {benchmark} BT/MSE sweep")


if __name__ == "__main__":
    main()

"""Run the preregistered sequential Torch-MLP multi-objective ALDE replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from folde.benchmarks.multimutant_data import sha256_file
from folde.benchmarks.multiobjective_alde import (
    ALDEFeatureSpace,
    MultiObjectiveALDEConfig,
    MultiObjectiveCampaignResult,
    analyze_mlp_replication_campaigns,
    run_multiobjective_campaign,
)
from folde.multiobjective_data import MULTIOBJECTIVE_DATASETS, load_multiobjective_dataset

DEFAULT_DATASETS = ("KCNJ2", "PTEN", "S22A1", "KCNE1", "OXDA", "RASK")
MLP_ARMS = (
    "mixed_parego",
    "mixed_hybrid_veto25",
    "mixed_plm_only",
    "mixed_random",
)


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _write_manifest(
    config: MultiObjectiveALDEConfig, datasets: tuple[str, ...], output_dir: Path
) -> None:
    package_config = Path(__file__).resolve().parents[2] / "pyproject.toml"
    _write_json(
        {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config.model_dump(mode="json"),
            "config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest(),
            "datasets": list(datasets),
            "arms": list(MLP_ARMS),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "dirty_worktree": bool(_git_value("status", "--porcelain")),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_config_sha256": sha256_file(package_config),
            "ranker": "TorchMLPFewShotModel",
        },
        output_dir / "manifest.json",
    )


def _load_campaigns(output_dir: Path) -> list[MultiObjectiveCampaignResult]:
    return [
        MultiObjectiveCampaignResult.model_validate_json(path.read_text())
        for path in sorted(output_dir.glob("*-seed-*.json"))
    ]


def _audit(
    campaigns: list[MultiObjectiveCampaignResult],
    output_dir: Path,
    datasets: tuple[str, ...],
    config: MultiObjectiveALDEConfig,
) -> dict[str, object]:
    errors: list[str] = []
    expected = len(datasets) * len(MLP_ARMS) * len(config.simulation_seeds)
    if len(campaigns) != expected:
        errors.append(f"expected {expected} campaigns, found {len(campaigns)}")
    for campaign in campaigns:
        if len(campaign.rounds) != config.rounds:
            errors.append(f"incomplete rounds: {campaign.dataset}/{campaign.arm}")
        for record in campaign.rounds:
            path = output_dir / record.proposal_pool_path
            if not path.exists():
                errors.append(f"missing {path.name}")
            elif sha256_file(path) != record.proposal_pool_sha256:
                errors.append(f"hash mismatch {path.name}")

    for dataset in datasets:
        for simulation_seed in config.simulation_seeds:
            paths = [
                output_dir
                / "proposal_pools"
                / f"{dataset}-{arm}-seed-{simulation_seed}-round-1.npz"
                for arm in MLP_ARMS
            ]
            if not all(path.exists() for path in paths):
                continue
            with np.load(paths[0]) as reference:
                for path in paths[1:]:
                    with np.load(path) as candidate:
                        if reference.files != candidate.files or not all(
                            np.array_equal(reference[column], candidate[column])
                            for column in reference.files
                        ):
                            errors.append(
                                f"round-1 proposal mismatch: {dataset}/seed-{simulation_seed}"
                            )
                            break
    return {
        "campaign_count": len(campaigns),
        "round_pool_count": sum(len(campaign.rounds) for campaign in campaigns),
        "errors": errors,
        "passed": not errors,
    }


def _write_results(report: dict[str, Any], output_dir: Path) -> None:
    comparisons = report["comparisons"]
    rows = []
    for name, comparison in comparisons.items():
        interval = comparison["bootstrap_95pct_ci"]
        rows.append(
            f"| {name} | {comparison['median_difference']:+.5f} | "
            f"[{interval[0]:+.5f}, {interval[1]:+.5f}] | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} | "
            f"{comparison['wilcoxon_exact_p']:.4f} |"
        )
    dataset_rows = []
    for dataset in report["datasets"]:
        arms = report["per_dataset_arm_means"][dataset]
        dataset_rows.append(
            f"| {dataset} | {arms['mixed_parego']['hypervolume_regret']:.5f} | "
            f"{arms['mixed_hybrid_veto25']['hypervolume_regret']:.5f} | "
            f"{arms['mixed_plm_only']['hypervolume_regret']:.5f} | "
            f"{arms['mixed_random']['hypervolume_regret']:.5f} |"
        )
    gate_lines = [
        f"- **{name}:** {'PASS' if passed else 'FAIL'}" for name, passed in report["gates"].items()
    ]
    content = "\n".join(
        [
            "# Sequential Torch-MLP multi-objective ALDE results",
            "",
            "Six datasets, ten common seeds, 16 initial measurements, five rounds of 16, "
            "and a 512-candidate heterogeneous pool. Activity-driven arms retrain two "
            "eight-member Torch-MLP ensembles after every round.",
            "",
            "| Comparison | Median improvement | 95% bootstrap CI | W/T/L | Exact p |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Gates",
            "",
            *gate_lines,
            "",
            "## Mean final hypervolume regret",
            "",
            "| Dataset | ParEGO | Veto25 + ParEGO | PLM-only | Random |",
            "|---|---:|---:|---:|---:|",
            *dataset_rows,
            "",
            "Lower regret is better. Positive paired improvements favor the first named "
            "target in each comparison.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS)
    )
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--initial-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--proposal-budget", type=int, default=512)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=81_811)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("folde/model_evals/260811-multiobjective-alde-mlp"),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = tuple(args.datasets)
    config = MultiObjectiveALDEConfig(
        benchmark_seed=args.seed,
        simulation_seeds=tuple(range(args.simulations)),
        initial_size=args.initial_size,
        batch_size=args.batch_size,
        rounds=args.rounds,
        proposal_budget=args.proposal_budget,
        ensemble_size=args.ensemble_size,
        ranker_type="torch_mlp",
        mlp_pretrain_epochs=args.pretrain_epochs,
        mlp_train_epochs=args.train_epochs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(config, datasets, args.output_dir)
    if not args.analyze_only:
        for dataset_name in datasets:
            print(f"Loading {dataset_name}", flush=True)
            dataset = load_multiobjective_dataset(MULTIOBJECTIVE_DATASETS[dataset_name])
            feature_space = ALDEFeatureSpace(dataset, config)
            for simulation_seed in config.simulation_seeds:
                for arm in MLP_ARMS:
                    print(f"Running {dataset_name} seed={simulation_seed} arm={arm}", flush=True)
                    run_multiobjective_campaign(
                        dataset=dataset,
                        arm=arm,
                        simulation_seed=simulation_seed,
                        config=config,
                        feature_space=feature_space,
                        output_dir=args.output_dir,
                        resume=not args.no_resume,
                    )
    campaigns = _load_campaigns(args.output_dir)
    audit = _audit(campaigns, args.output_dir, datasets, config)
    report = analyze_mlp_replication_campaigns(campaigns)
    report["engineering_audit"] = audit
    report["gates"]["engineering_gate"] = audit["passed"]
    report["gates"]["mlp_pipeline_authorized"] = bool(
        report["gates"]["mlp_pipeline_authorized"] and audit["passed"]
    )
    _write_json(report, args.output_dir / "gate-report.json")
    _write_results(report, args.output_dir)
    print(json.dumps({"gates": report["gates"], "audit": audit}, indent=2))


if __name__ == "__main__":
    main()

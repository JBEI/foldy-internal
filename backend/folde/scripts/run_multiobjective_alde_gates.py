"""Run and analyze the preregistered multi-objective ALDE gate campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from folde.benchmarks.multimutant_data import sha256_file
from folde.benchmarks.multiobjective_alde import (
    ALDE_ARMS,
    ALDEFeatureSpace,
    MultiObjectiveALDEConfig,
    MultiObjectiveCampaignResult,
    analyze_hybrid_gate_campaigns,
    run_multiobjective_campaign,
)
from folde.multiobjective_data import MULTIOBJECTIVE_DATASETS, load_multiobjective_dataset

DEFAULT_DATASETS = ("KCNJ2", "PTEN", "S22A1", "KCNE1", "OXDA", "RASK")


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_manifest(
    config: MultiObjectiveALDEConfig,
    datasets: tuple[str, ...],
    output_dir: Path,
) -> None:
    package_config = Path(__file__).resolve().parents[2] / "pyproject.toml"
    config_json = config.model_dump_json()
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(mode="json"),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "datasets": list(datasets),
        "arms": list(ALDE_ARMS),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(_git_value("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_config_sha256": sha256_file(package_config),
        "ranker": {
            "name": "random_projection_bootstrap_ridge_screening",
            "projection_dim": config.projection_dim,
            "ensemble_size": config.ensemble_size,
            "ridge_alpha": config.ridge_alpha,
        },
    }
    _write_json(manifest, output_dir / "manifest.json")


def _load_results(output_dir: Path) -> list[MultiObjectiveCampaignResult]:
    return [
        MultiObjectiveCampaignResult.model_validate_json(path.read_text())
        for path in sorted(output_dir.glob("*-seed-*.json"))
    ]


def _audit_artifacts(
    campaigns: list[MultiObjectiveCampaignResult],
    output_dir: Path,
    expected_count: int,
    config: MultiObjectiveALDEConfig,
) -> dict[str, object]:
    errors: list[str] = []
    if len(campaigns) != expected_count:
        errors.append(f"expected {expected_count} campaigns, found {len(campaigns)}")
    for campaign in campaigns:
        if len(campaign.rounds) != config.rounds:
            errors.append(f"{campaign.dataset}/{campaign.arm}/{campaign.simulation_seed}: rounds")
        expected_measured = config.initial_size + config.rounds * config.batch_size
        if len(campaign.measured_seq_ids) != expected_measured:
            errors.append(f"{campaign.dataset}/{campaign.arm}/{campaign.simulation_seed}: measured")
        for record in campaign.rounds:
            pool_path = output_dir / record.proposal_pool_path
            if not pool_path.exists():
                errors.append(f"missing {pool_path.name}")
            elif sha256_file(pool_path) != record.proposal_pool_sha256:
                errors.append(f"hash mismatch {pool_path.name}")

    # Shared initial state must produce exactly the same mixed proposal records.
    import numpy as np

    for dataset in sorted({campaign.dataset for campaign in campaigns}):
        for simulation_seed in config.simulation_seeds:
            paths = [
                output_dir
                / "proposal_pools"
                / f"{dataset}-{arm}-seed-{simulation_seed}-round-1.npz"
                for arm in (
                    "mixed_parego",
                    "mixed_fixed",
                    "mixed_plm_only",
                    "mixed_random",
                    "mixed_hybrid_soft25",
                    "mixed_hybrid_veto25",
                )
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
                                f"mixed round-1 pool mismatch: {dataset}/seed-{simulation_seed}"
                            )
                            break
    return {
        "campaign_count": len(campaigns),
        "round_pool_count": sum(len(campaign.rounds) for campaign in campaigns),
        "errors": errors,
        "passed": not errors,
    }


def _format_comparison(name: str, report: dict[str, Any]) -> str:
    interval = report["bootstrap_95pct_ci"]
    return (
        f"| {name} | {report['median_difference']:+.5f} | "
        f"[{interval[0]:+.5f}, {interval[1]:+.5f}] | "
        f"{report['wins']}/{report['ties']}/{report['losses']} | "
        f"{report['wilcoxon_exact_p']:.4f} |"
    )


def _write_markdown(report: dict[str, Any], output_dir: Path) -> None:
    comparisons = report["comparisons"]
    gates = report["gates"]
    rows = [_format_comparison(name, comparison) for name, comparison in comparisons.items()]
    gate_lines = [f"- **{name}:** {'PASS' if passed else 'FAIL'}" for name, passed in gates.items()]
    per_dataset = report["per_dataset_arm_means"]
    dataset_rows = []
    for dataset in report["datasets"]:
        arms = per_dataset[dataset]
        proposal_delta = (
            arms["mixed_parego"]["round1_proposal_attainable_hypervolume"]
            - arms["random_pool_parego"]["round1_proposal_attainable_hypervolume"]
        )
        selector_delta = (
            arms["mixed_plm_only"]["hypervolume_regret"]
            - arms["mixed_parego"]["hypervolume_regret"]
        )
        end_to_end_delta = (
            arms["random_pool_parego"]["hypervolume_regret"]
            - arms["mixed_parego"]["hypervolume_regret"]
        )
        coverage_delta = (
            arms["mixed_parego"]["epsilon_front_coverage"]
            - arms["mixed_fixed"]["epsilon_front_coverage"]
        )
        dataset_rows.append(
            f"| {dataset} | {proposal_delta:+.5f} | {selector_delta:+.5f} | "
            f"{end_to_end_delta:+.5f} | {coverage_delta:+.5f} |"
        )
    hybrid_rows = []
    for dataset in report["datasets"]:
        arms = per_dataset[dataset]
        soft_vs_parego = (
            arms["mixed_parego"]["hypervolume_regret"]
            - arms["mixed_hybrid_soft25"]["hypervolume_regret"]
        )
        soft_vs_plm = (
            arms["mixed_plm_only"]["hypervolume_regret"]
            - arms["mixed_hybrid_soft25"]["hypervolume_regret"]
        )
        veto_vs_parego = (
            arms["mixed_parego"]["hypervolume_regret"]
            - arms["mixed_hybrid_veto25"]["hypervolume_regret"]
        )
        hybrid_rows.append(
            f"| {dataset} | {soft_vs_parego:+.5f} | {soft_vs_plm:+.5f} | "
            f"{veto_vs_parego:+.5f} |"
        )
    content = "\n".join(
        [
            "# Naturalness-aware multi-objective ALDE hybrid results",
            "",
            "Six datasets, ten common seeds, 16 initial measurements, five 16-variant "
            "rounds, and a 512-candidate proposal budget.",
            "",
            "Positive comparison differences favor the first named target: "
            "`mixed_parego` for baseline comparisons and the named hybrid for hybrid "
            "comparisons. Fields containing `regret_delta` are explicitly target regret "
            "minus comparator regret, where lower is better.",
            "",
            "| Comparison | Median difference | 95% bootstrap CI | W/T/L | Exact p |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Gates",
            "",
            *gate_lines,
            "",
            "## Dataset-level localization",
            "",
            "All differences below are oriented so positive values favor `mixed_parego`.",
            "",
            "| Dataset | Mixed vs random proposal HV | ParEGO vs PLM-only regret | "
            "Mixed vs random-pool regret | ParEGO vs fixed coverage |",
            "|---|---:|---:|---:|---:|",
            *dataset_rows,
            "",
            "## Hybrid selector localization",
            "",
            "Positive differences reduce regret relative to the named comparator.",
            "",
            "| Dataset | Soft hybrid vs ParEGO | Soft hybrid vs PLM-only | "
            "Veto hybrid vs ParEGO |",
            "|---|---:|---:|---:|",
            *hybrid_rows,
            "",
            "## Interpretation",
            "",
            "The 25% soft prior is effectively neutral in aggregate: it splits three wins "
            "and three losses against both plain ParEGO and PLM-only selection, with both "
            "intervals crossing zero. It helps strongly on KCNE1 but hurts PTEN and RASK. "
            "The diagnostic bottom-quartile veto is more consistent (five wins, one loss "
            "against ParEGO), but its interval also crosses zero. The soft hybrid therefore "
            "does not authorize production replication. If this branch continues, the "
            "hard veto is the better hypothesis to validate; a soft 25% prior is not.",
            "",
            "This is the preregistered random-projection/bootstrap-ridge screening tier. "
            "A pass authorizes production Torch-MLP replication; it is not itself a "
            "wet-lab deployment claim.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS)
    )
    parser.add_argument("--arms", nargs="+", choices=ALDE_ARMS, default=list(ALDE_ARMS))
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--initial-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--proposal-budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=81_811)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("folde/model_evals/260811-multiobjective-alde-hybrid"),
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
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(config, datasets, args.output_dir)
    if not args.analyze_only:
        for dataset_name in datasets:
            print(f"Loading {dataset_name}", flush=True)
            dataset = load_multiobjective_dataset(MULTIOBJECTIVE_DATASETS[dataset_name])
            feature_space = ALDEFeatureSpace(dataset, config)
            for simulation_seed in config.simulation_seeds:
                for arm in args.arms:
                    print(
                        f"Running {dataset_name} seed={simulation_seed} arm={arm}",
                        flush=True,
                    )
                    run_multiobjective_campaign(
                        dataset=dataset,
                        arm=arm,
                        simulation_seed=simulation_seed,
                        config=config,
                        feature_space=feature_space,
                        output_dir=args.output_dir,
                        resume=not args.no_resume,
                    )
    campaigns = _load_results(args.output_dir)
    expected_count = len(datasets) * len(args.arms) * len(config.simulation_seeds)
    audit = _audit_artifacts(campaigns, args.output_dir, expected_count, config)
    report = analyze_hybrid_gate_campaigns(campaigns)
    report["engineering_audit"] = audit
    report["gates"]["engineering_gate"] = audit["passed"]
    report["gates"]["production_replication_authorized"] = bool(
        report["gates"]["production_replication_authorized"] and audit["passed"]
    )
    report["gates"]["hybrid_production_replication_authorized"] = bool(
        report["gates"]["hybrid_production_replication_authorized"] and audit["passed"]
    )
    _write_json(report, args.output_dir / "gate-report.json")
    _write_markdown(report, args.output_dir)
    print(json.dumps({"gates": report["gates"], "engineering_audit": audit}, indent=2))


if __name__ == "__main__":
    main()

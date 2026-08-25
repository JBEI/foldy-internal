"""Prepare and run Phase 2's closed-world Olson Protocol A benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from folde.benchmarks.feature_store import MemmapFeatureStore, prepare_memmap_feature_store
from folde.benchmarks.multimutant_data import sha256_file
from folde.benchmarks.olson_protocol import (
    ARM_NAMES,
    OLSON_DMS_ID,
    OlsonProtocolConfig,
    load_additive_naturalness_scores,
    load_olson_activity,
    run_olson_arm,
    write_paired_report,
)
from folde.data import DMS_DIR, EMBEDDINGS_DIR, NATURALNESS_DIR


def _git_value(*args: str) -> str:
    completed = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    return completed.stdout.strip()


def _write_manifest(config: OlsonProtocolConfig, output_dir: Path, feature_store_dir: Path) -> None:
    config_json = config.model_dump_json()
    activity_path = DMS_DIR / f"{OLSON_DMS_ID}.csv"
    naturalness_path = NATURALNESS_DIR / f"{OLSON_DMS_ID}_naturalness_600m.csv"
    package_config = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        cuda = {"available": False, "device_count": 0, "devices": []}
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(mode="json"),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(_git_value("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": cuda,
        "activity_sha256": sha256_file(activity_path),
        "naturalness_sha256": sha256_file(naturalness_path),
        "feature_store_metadata_sha256": sha256_file(feature_store_dir / "metadata.json"),
        "package_config_sha256": sha256_file(package_config),
    }
    path = output_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=EMBEDDINGS_DIR / f"{OLSON_DMS_ID}_embedding_300m.memmap",
    )
    parser.add_argument("--prepare-feature-store", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("folde/model_evals/generative/olson-protocol-a"),
    )
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(ARM_NAMES))
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--initial-singles", type=int, default=32)
    parser.add_argument("--round-size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--proposal-budget", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = EMBEDDINGS_DIR / f"{OLSON_DMS_ID}_embedding_300m.csv"
    if args.prepare_feature_store:
        metadata = prepare_memmap_feature_store(source, args.feature_store)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    if not (args.feature_store / "metadata.json").exists():
        raise FileNotFoundError(
            f"Feature store is absent at {args.feature_store}; rerun with --prepare-feature-store"
        )
    config = OlsonProtocolConfig(
        benchmark_seed=args.seed,
        simulation_seeds=tuple(range(args.simulations)),
        initial_singles=args.initial_singles,
        round_size=args.round_size,
        rounds=args.rounds,
        proposal_budget=args.proposal_budget,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(config, args.output_dir, args.feature_store)
    reference, activity, singles, doubles = load_olson_activity()
    naturalness = load_additive_naturalness_scores([*singles, *doubles])
    feature_store = MemmapFeatureStore(args.feature_store)
    missing_features = [
        seq_id for seq_id in [*singles, *doubles] if not feature_store.contains(seq_id)
    ]
    if missing_features:
        raise ValueError(f"Olson feature store lacks activity variants: {missing_features[:5]}")
    results = []
    for simulation_seed in config.simulation_seeds:
        for arm in args.arms:
            print(f"Running {arm}, simulation seed {simulation_seed}", flush=True)
            results.append(
                run_olson_arm(
                    arm=arm,
                    simulation_seed=simulation_seed,
                    config=config,
                    reference_sequence=reference,
                    activity=activity,
                    singles=singles,
                    doubles=doubles,
                    naturalness=naturalness,
                    feature_store=feature_store,
                    output_dir=args.output_dir,
                    resume=not args.no_resume,
                )
            )
    if {"plm_plus_folde", "adjacent_folde"} <= set(args.arms):
        report = write_paired_report(results, args.output_dir / "paired-report.json")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

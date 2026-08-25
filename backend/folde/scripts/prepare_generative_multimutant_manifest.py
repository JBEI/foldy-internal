"""Generate the Phase 0 dataset and feature manifest.

Run from ``backend/``::

    python -m folde.scripts.prepare_generative_multimutant_manifest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from folde.benchmarks.schemas import GenerativeMultimutantBenchmarkConfig

DEFAULT_OUTPUT = Path("folde/model_evals/260811-generative-multimutant-manifest.json")
DEFAULT_SCHEMA_OUTPUT = Path("folde/benchmarks/generative_multimutant_config.schema.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schema-output", type=Path, default=DEFAULT_SCHEMA_OUTPUT)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Regenerate the JSON configuration schema without auditing datasets.",
    )
    parser.add_argument(
        "--skip-feature-hashes",
        action="store_true",
        help="Audit activity datasets without reading large local feature files.",
    )
    args = parser.parse_args()
    args.schema_output.parent.mkdir(parents=True, exist_ok=True)
    args.schema_output.write_text(
        json.dumps(GenerativeMultimutantBenchmarkConfig.model_json_schema(), indent=2) + "\n"
    )
    if args.schema_only:
        print(f"Wrote configuration schema to {args.schema_output}")
        return 0

    from folde.benchmarks.multimutant_data import build_dataset_manifest, write_manifest

    manifest = build_dataset_manifest(include_feature_hashes=not args.skip_feature_hashes)
    write_manifest(manifest, args.output)
    print(
        f"Wrote {len(manifest['datasets'])} datasets and "
        f"{len(manifest['feature_files'])} feature hashes to {args.output}; "
        f"configuration schema to {args.schema_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

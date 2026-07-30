"""Copy worker embedding output into the explicit paths the FolDE loaders require.

The ESM worker writes to `foldydata/<06d fold_id>/embed/<06d fold_id>_embeddings_
<model>_<name>.csv`. `folde.multiobjective_data._load_embedding_df` deliberately
refuses to go looking for that file -- it requires an exact local path
`folde/data/embeddings/<dms_id>_embedding_<model_id>.csv` -- because the fuzzy
finder in `folde.data` matches on a bare substring like "300m", which collides
between ESMC-300M (960-dim) and Profluent E1-300m (1024-dim). This script does
that copy explicitly, one destination per dms_id.

Naturalness needs no copy: `_find_foldydata_naturalness_file` resolves it from
the foldydata fold directory, which works as long as the fold's name equals the
dms_id (see 260730_submit_new_dataset_jobs.py).

Verifies before writing that the embedding dimension is what the dataset spec
expects and that coverage of the activity seq_ids is complete enough to be
usable, then reports the aligned dataset size.

Run:
  cd backend && ../.venv/bin/python -u folde/scripts/260730_consolidate_new_embeddings.py
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

os.environ.setdefault("FOLDE_CONSTANT_LIAR_DEVICE", "cpu")

from app.helpers.sequence_util import allele_set_to_seq_id  # noqa: E402
from folde.data import DMS_DIR  # noqa: E402
from folde.multiobjective_data import EMBEDDINGS_DIR  # noqa: E402

FOLDYDATA = Path("foldydata")
EXPECTED_DIM = 960  # ESMC-300M

# fold_id -> (worker embed job name, [dms_ids that should receive a copy])
SOURCES: Dict[int, Tuple[str, List[str]]] = {
    179: (
        "esmc-300m",
        ["KCNE1_HUMAN_Muhammad_2023_function", "KCNE1_HUMAN_Muhammad_2023_expression"],
    ),
    180: (
        "esmc-300m",
        ["RASK_HUMAN_Weng_2022_abundance", "RASK_HUMAN_Weng_2022_binding-DARPin_K55"],
    ),
    181: (
        "esmc-300m",
        ["OXDA_RHOTO_Vanella_2023_activity", "OXDA_RHOTO_Vanella_2023_expression"],
    ),
    21: ("wu2016-esmc-300m", ["SPG1_STRSG_Wu_2016"]),
}


def activity_seq_ids(dms_id: str) -> set:
    df = pd.read_csv(DMS_DIR / f"{dms_id}.csv")
    return {allele_set_to_seq_id(set(m.split(":"))) for m in df["mutant"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.99,
        help="Refuse to copy if the embedding covers less than this fraction of activity seq_ids.",
    )
    ap.add_argument(
        "--force", action="store_true", help="Copy even if coverage is below threshold."
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated fold_ids to process (default: all). Avoids re-copying "
        "multi-GB sources that have already landed.",
    )
    args = ap.parse_args()

    only = {int(s) for s in args.only.split(",") if s.strip()}

    exit_code = 0
    for fold_id, (embed_name, dms_ids) in SOURCES.items():
        if only and fold_id not in only:
            continue
        src = (
            FOLDYDATA
            / f"{fold_id:06d}"
            / "embed"
            / (f"{fold_id:06d}_embeddings_esmc_300m_{embed_name}.csv")
        )
        print(f"\nfold {fold_id:06d}  ({embed_name})")
        if not src.exists():
            print(f"  SKIP: worker output not present yet: {src}")
            exit_code = 1
            continue

        # Read only what the checks need. The SPG1_Wu source is 3.3 GB (149k rows
        # x ~19 KB of embedding text each); a full read would materialize ~3 GB of
        # Python strings to verify one dimension and one key set. `usecols` keeps
        # this at a few MB regardless of source size.
        header = pd.read_csv(src, nrows=1)
        if "seq_id" not in header.columns or "embedding" not in header.columns:
            print(f"  ERROR: missing seq_id/embedding columns: {header.columns.tolist()}")
            exit_code = 1
            continue

        seq_id_col = pd.read_csv(src, usecols=["seq_id"])["seq_id"]
        dim = len(str(header["embedding"].iloc[0]).split(","))
        print(f"  rows={len(seq_id_col)}  dim={dim}")
        if dim != EXPECTED_DIM:
            print(
                f"  ERROR: dimension {dim} != expected {EXPECTED_DIM}. This is the ESMC/E1 "
                f"collision the loader guards against -- refusing to copy."
            )
            exit_code = 1
            continue

        embedded = set(seq_id_col)
        for dms_id in dms_ids:
            wanted = activity_seq_ids(dms_id)
            covered = len(wanted & embedded) / max(len(wanted), 1)
            dest = EMBEDDINGS_DIR / f"{dms_id}_embedding_300m.csv"
            status = "ok" if covered >= args.min_coverage else "LOW"
            print(f"  {dms_id:45s} coverage={covered:6.2%} [{status}] -> {dest.name}")
            if covered < args.min_coverage and not args.force:
                print("    skipped (use --force to copy anyway)")
                exit_code = 1
                continue
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    if exit_code:
        print("\nSome sources were missing or incomplete; re-run once the esm queue drains.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

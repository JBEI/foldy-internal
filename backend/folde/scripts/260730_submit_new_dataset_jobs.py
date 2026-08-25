"""Submit fold / embedding / naturalness jobs for the datasets needed to reach
the spec's dataset floor for acceptance testing.

Adds three new multi-objective proteins (KCNE1, RASK, OXDA) and the higher-order
SPG1_STRSG_Wu_2016 single-objective set. For each new protein this creates a
Fold row (no structure prediction -- embeddings and naturalness do not need a
structure), then enqueues:

  * one ESMC-300M embedding job covering the UNION of seq_ids across the
    protein's objectives, plus WT;
  * one Profluent E1-600M naturalness job.

SPG1_STRSG_Wu_2016 shares its wild-type sequence with SPG1_STRSG_Olson_2014, so
it reuses that existing fold and needs embeddings only -- naturalness is a
per-single-substitution table (L x 20) that higher-order variants score against
additively, and Olson's E1-600m table is already present.

Two naming constraints are load-bearing and are asserted rather than assumed:

  * the fold's `name` must equal the `embedding_dms_id`, because
    `folde.data._get_foldydata_fold_dir` resolves a dataset's foldydata
    directory by fold name;
  * the naturalness job's `name` must contain the model token, because
    `_find_foldydata_naturalness_file` globs `*_melted.csv` and filters on
    `naturalness_model_id.lower() in path.name.lower()`.

seq_ids are generated through `allele_set_to_seq_id`, which sorts alleles by
position. This is NOT the same as the `mutant.replace(":", "_")` used by the
older notebooks/jacob/*_seqids.txt files: for multi-mutant datasets like RASK
the DMS csv's allele order is arbitrary, so the two disagree and embeddings
keyed the old way would silently fail to join against the activity frame.

Run (dry run first -- it prints counts and validates every seq_id without
enqueueing anything):
  cd backend && ../.venv/bin/python -u folde/scripts/260730_submit_new_dataset_jobs.py
  cd backend && ../.venv/bin/python -u folde/scripts/260730_submit_new_dataset_jobs.py --submit
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import pandas as pd
import requests

os.environ.setdefault("FOLDE_CONSTANT_LIAR_DEVICE", "cpu")

from app.helpers.sequence_util import (  # noqa: E402
    allele_set_to_seq_id,
    maybe_get_seq_id_error_message,
)
from folde.data import DMS_DIR, get_dms_metadata  # noqa: E402

API = "http://localhost:8080/api"
TOKEN_SCRIPT = os.path.expanduser("~/.claude/skills/foldy-dev-docker/scripts/foldy-token.sh")

EMBEDDING_MODEL = "esmc_300m"
NATURALNESS_MODEL = "e1_600m"
# Must contain the token that `_find_foldydata_naturalness_file` greps for, and
# must match the existing on-disk convention (naturalness_E1-600m_melted.csv).
NATURALNESS_NAME = "E1-600m"
EMBEDDING_NAME = "esmc-300m"


# Each entry: fold name (== embedding_dms_id) -> the dms_ids whose seq_ids the
# embedding must cover. A protein's objectives share a wild-type sequence, which
# is asserted below before anything is submitted.
JOBS: Dict[str, Dict] = {
    "KCNE1_HUMAN_Muhammad_2023_function": {
        "dms_ids": [
            "KCNE1_HUMAN_Muhammad_2023_function",
            "KCNE1_HUMAN_Muhammad_2023_expression",
        ],
        "tags": ["folde", "multiobjective", "kcne1"],
        "needs_naturalness": True,
    },
    "RASK_HUMAN_Weng_2022_abundance": {
        "dms_ids": [
            "RASK_HUMAN_Weng_2022_abundance",
            "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        ],
        "tags": ["folde", "multiobjective", "rask"],
        "needs_naturalness": True,
    },
    "OXDA_RHOTO_Vanella_2023_activity": {
        "dms_ids": [
            "OXDA_RHOTO_Vanella_2023_activity",
            "OXDA_RHOTO_Vanella_2023_expression",
        ],
        "tags": ["folde", "multiobjective", "oxda"],
        "needs_naturalness": True,
    },
    # Higher-order (up to quadruple) mutants on the GB1 domain. Reuses the
    # existing SPG1_STRSG_Olson_2014 fold: same wild-type, and its E1-600m
    # naturalness table already covers every single substitution.
    "SPG1_STRSG_Olson_2014": {
        "dms_ids": ["SPG1_STRSG_Wu_2016"],
        "tags": ["folde", "higher-order", "spg1"],
        "needs_naturalness": False,
        "embedding_name": "wu2016-esmc-300m",
    },
    # ── Long-range multi-mutant sets (see 260730_survey_multimutant_datasets.py) ──
    #
    # SPG1_Wu is a 4-site combinatorial library: 121k quadruples confined to four
    # mutually contacting positions. It measures local epistasis and cannot show
    # whether a model generalizes to mutations that are far apart. The sets below
    # were selected on distinct-mutated-positions AND absolute sequence
    # separation, which is the pair of statistics that distinguishes a
    # distributed library from a fixed-site or segment-tiled one. (Ratio alone is
    # misleading: SPG1_Wu tops span/window at 0.94 because its window is 16
    # residues wide.)
    #
    # Sarkisyan is the canonical distributed case -- random mutagenesis, so 233 of
    # 238 positions are hit and co-mutants sit a median 130 residues apart.
    "GFP_AEQVI_Sarkisyan_2016": {
        "dms_ids": ["GFP_AEQVI_Sarkisyan_2016"],
        "tags": ["folde", "multimutant", "long-range", "gfp"],
        "needs_naturalness": True,
    },
    # Three GFP homologs at 222-238 residues, each ~100% of positions mutated.
    # Their value beyond spread is cross-homolog transfer: train on one, test on
    # another, which measures long-range generalization ACROSS sequence identity.
    # Q8WTC7 reaches mutation order 43.
    "Q8WTC7_9CNID_Somermeyer_2022": {
        "dms_ids": ["Q8WTC7_9CNID_Somermeyer_2022"],
        "tags": ["folde", "multimutant", "long-range", "gfp-homolog"],
        "needs_naturalness": True,
    },
    "D7PM05_CLYGR_Somermeyer_2022": {
        "dms_ids": ["D7PM05_CLYGR_Somermeyer_2022"],
        "tags": ["folde", "multimutant", "long-range", "gfp-homolog"],
        "needs_naturalness": True,
    },
    "Q6WV12_9MAXI_Somermeyer_2022": {
        "dms_ids": ["Q6WV12_9MAXI_Somermeyer_2022"],
        "tags": ["folde", "multimutant", "long-range", "gfp-homolog"],
        "needs_naturalness": True,
    },
    # Distance control: only 15 mutated positions, but spans of ~106 residues in a
    # 118-residue LOV domain. Isolates "long-range" from "many-site" -- a model
    # that handles Sarkisyan but fails here is limited by separation, not by the
    # number of distinct sites.
    "PHOT_CHLRE_Chen_2023": {
        "dms_ids": ["PHOT_CHLRE_Chen_2023"],
        "tags": ["folde", "multimutant", "long-range", "distance-control"],
        "needs_naturalness": True,
    },
}


def get_token() -> str:
    return subprocess.check_output(["bash", TOKEN_SCRIPT], text=True).strip()


def seq_ids_for(dms_id: str) -> List[str]:
    """Canonical seq_ids for one DMS file, in the same form the loaders use."""
    df = pd.read_csv(DMS_DIR / f"{dms_id}.csv")
    return [allele_set_to_seq_id(set(m.split(":"))) for m in df["mutant"]]


def wt_sequence_for(dms_id: str) -> str:
    metadata = get_dms_metadata()
    row = metadata[metadata["DMS_id"] == dms_id]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one metadata row for {dms_id}, found {len(row)}")
    return str(row["target_seq"].iloc[0])


def build_yaml_config(sequence: str) -> str:
    return f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {sequence}\n"


def find_fold(name: str) -> Optional[int]:
    """Resolve a fold id by name straight from Postgres.

    There is no list-folds REST endpoint that filters by exact name, and the
    Fold model lives in the legacy `roles` table (see app/models.py), so this
    queries the dev container directly rather than guessing an API shape.
    """
    out = subprocess.check_output(
        [
            "docker",
            "exec",
            "foldy-internal-db-1",
            "psql",
            "-U",
            "user",
            "-d",
            "postgres",
            "-t",
            "-A",
            "-c",
            f"select id from roles where name = '{name}';",
        ],
        text=True,
    ).strip()
    return int(out) if out else None


def create_fold(session: requests.Session, name: str, sequence: str, tags: List[str]) -> None:
    payload = {
        "folds_data": [
            {
                "name": name,
                "sequence": sequence,
                "yaml_config": build_yaml_config(sequence),
                "tags": tags,
                "public": False,
                # Required by make_new_folds' bracket access; a dry run does NOT
                # catch its absence, so it is always sent explicitly.
                "disable_relaxation": True,
                "diffusion_samples": 1,
            }
        ],
        # No structure is needed for embeddings or naturalness, and the boltz
        # queue would otherwise burn GPU time we want for the ESM jobs.
        "start_fold_job": False,
        "email_on_completion": False,
        "skip_duplicate_entries": True,
        "is_dry_run": False,
    }
    resp = session.post(f"{API}/fold", json=payload)
    resp.raise_for_status()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--submit",
        action="store_true",
        help="Actually create folds and enqueue jobs (default is a dry run).",
    )
    ap.add_argument("--only", default="", help="Comma-separated subset of fold names to process.")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    token = get_token()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})

    plan = []
    for fold_name, cfg in JOBS.items():
        if only and fold_name not in only:
            continue

        wt_sequences = {d: wt_sequence_for(d) for d in cfg["dms_ids"]}
        # The fold itself is keyed on fold_name, which for the reused SPG1 fold
        # is not one of cfg["dms_ids"], so pull its wild-type separately.
        wt_sequences[fold_name] = wt_sequence_for(fold_name)
        distinct = set(wt_sequences.values())
        if len(distinct) != 1:
            raise ValueError(
                f"{fold_name}: objectives do not share a wild-type sequence: "
                f"{ {k: len(v) for k, v in wt_sequences.items()} }"
            )
        wt_sequence = distinct.pop()

        union: set = set()
        per_dms = {}
        for dms_id in cfg["dms_ids"]:
            ids = seq_ids_for(dms_id)
            per_dms[dms_id] = len(set(ids))
            union |= set(ids)
        union.discard("WT")

        invalid = [s for s in union if maybe_get_seq_id_error_message(wt_sequence, s)]
        if invalid:
            raise ValueError(
                f"{fold_name}: {len(invalid)} seq_ids fail validation against the "
                f"wild-type (e.g. {sorted(invalid)[:5]}). Refusing to submit."
            )

        seq_ids = ["WT"] + sorted(union)
        plan.append((fold_name, cfg, wt_sequence, seq_ids, per_dms))

        print(f"{fold_name}")
        print(f"  wt_len={len(wt_sequence)}  union_seq_ids={len(seq_ids)}  per_dms={per_dms}")
        print(
            f"  embedding={EMBEDDING_MODEL}  naturalness="
            f"{NATURALNESS_MODEL if cfg['needs_naturalness'] else 'SKIP (reuses existing)'}"
        )

    if not args.submit:
        print("\nDry run only. Re-run with --submit to create folds and enqueue jobs.")
        return 0

    for fold_name, cfg, wt_sequence, seq_ids, _ in plan:
        fold_id = find_fold(fold_name)
        if fold_id is None:
            print(f"\n{fold_name}: creating fold...")
            create_fold(session, fold_name, wt_sequence, cfg["tags"])
            fold_id = find_fold(fold_name)
            if fold_id is None:
                raise RuntimeError(f"{fold_name}: fold creation reported success but no row found")
            print(f"  created fold_id={fold_id}")
        else:
            print(f"\n{fold_name}: reusing existing fold_id={fold_id}")

        if cfg["needs_naturalness"]:
            resp = session.post(
                f"{API}/startnaturalness/{fold_id}",
                json={
                    "name": NATURALNESS_NAME,
                    "logit_model": NATURALNESS_MODEL,
                    "use_structure": False,
                    "get_depth_two_logits": False,
                    "use_msa_context": False,
                },
            )
            if resp.status_code >= 400:
                print(f"  naturalness FAILED {resp.status_code}: {resp.text[:300]}")
            else:
                print(f"  naturalness enqueued ({NATURALNESS_MODEL} as {NATURALNESS_NAME!r})")

        embed_name = cfg.get("embedding_name", EMBEDDING_NAME)
        resp = session.post(
            f"{API}/embeddings",
            json={
                "fold_id": fold_id,
                "name": embed_name,
                "embedding_model": EMBEDDING_MODEL,
                "extra_seq_ids": ",".join(seq_ids),
                "dms_starting_seq_ids": "",
                "extra_layers": "",
                "domain_boundaries": "",
                "use_msa_context": False,
            },
        )
        if resp.status_code >= 400:
            print(f"  embedding FAILED {resp.status_code}: {resp.text[:300]}")
        else:
            print(f"  embedding enqueued: {len(seq_ids)} seq_ids as {embed_name!r}")
            print(
                f"  -> output: foldydata/{fold_id:06d}/embed/"
                f"{fold_id:06d}_embeddings_{EMBEDDING_MODEL}_{embed_name}.csv"
            )

    print("\nSubmitted. Watch the esm queue:")
    print("  docker exec foldy-internal-redis-1 redis-cli llen rq:queue:esm")
    print("  docker logs -f foldy-internal-worker_esm-1")
    return 0


if __name__ == "__main__":
    sys.exit(main())

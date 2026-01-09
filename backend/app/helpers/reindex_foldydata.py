"""Recreate database records from existing foldydata folders."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml
from sqlalchemy import text

from app.extensions import db
from app.factory import create_app
from app.models import Embedding, Fold, Invokation, Naturalness, User
from app.views.esm_views import ALLOWED_ESM_MODELS, ALLOWED_NATURALNESS_MODELS


@dataclass(frozen=True)
class FoldInputs:
    """Derived inputs needed to create a Fold record."""

    fold_id: int
    name: str
    yaml_config: Optional[str]
    sequence: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args for foldydata reindexing."""
    parser = argparse.ArgumentParser(
        description="Recreate Fold (and optional run) records from foldydata folders."
    )
    parser.add_argument(
        "--foldydata-dir",
        type=str,
        default=None,
        help="Root foldydata directory (defaults to FOLDY_LOCAL_STORAGE_DIR).",
    )
    parser.add_argument(
        "--user-email",
        type=str,
        default=None,
        help="User email to assign as owner (defaults to existing user or tester@test.edu).",
    )
    parser.add_argument(
        "--name-template",
        type=str,
        default="fold_{id}",
        help="Template for fold names. Available fields: {id}, {padded}.",
    )
    parser.add_argument(
        "--include-runs",
        action="store_true",
        help="Also recreate embedding/naturalness records from output files.",
    )
    parser.add_argument(
        "--default-logit-model",
        type=str,
        default="esmc_600m",
        help="Fallback logit model when it cannot be inferred.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be created without modifying the database.",
    )
    return parser.parse_args(argv)


def list_fold_dirs(root: Path) -> List[Path]:
    """Return numeric fold directories under the given root."""
    fold_dirs: List[Path] = []
    if not root.exists():
        return fold_dirs
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            fold_dirs.append(entry)
    return sorted(fold_dirs, key=lambda p: int(p.name))


def read_yaml_config(fold_dir: Path) -> Optional[str]:
    """Read a Boltz YAML config from known locations."""
    candidates = [
        fold_dir / "boltz_input.yaml",
        fold_dir / "boltz" / "input.yml",
        fold_dir / "boltz" / "input.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text()
    return None


def get_protein_sequences(yaml_str: str) -> List[str]:
    """Extract protein sequences from a Boltz YAML string."""
    data = yaml.safe_load(yaml_str)
    if not isinstance(data, dict):
        return []
    sequences = data.get("sequences", [])
    if not isinstance(sequences, list):
        return []
    protein_sequences: List[str] = []
    for entry in sequences:
        if not isinstance(entry, dict):
            continue
        protein = entry.get("protein")
        if not isinstance(protein, dict):
            continue
        seq = protein.get("sequence")
        if isinstance(seq, str) and seq:
            protein_sequences.append(seq)
    return protein_sequences


def read_first_fasta_sequence(fold_dir: Path, padded_fold_id: str) -> Optional[str]:
    """Read the first protein sequence from a FASTA file."""
    preferred = fold_dir / f"{padded_fold_id}.fasta"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(sorted(fold_dir.glob("*.fasta")))
    for candidate in candidates:
        if "_dna" in candidate.name:
            continue
        lines = [line.strip() for line in candidate.read_text().splitlines()]
        seq = "".join(line for line in lines if line and not line.startswith(">"))
        if seq:
            return seq
    return None


def ensure_fold_inputs(fold_dir: Path, name_template: str) -> Optional[FoldInputs]:
    """Build FoldInputs from a fold directory."""
    fold_id = int(fold_dir.name)
    padded = f"{fold_id:06d}"
    yaml_config = read_yaml_config(fold_dir)
    sequence = None
    if yaml_config:
        protein_sequences = get_protein_sequences(yaml_config)
        if len(protein_sequences) > 1:
            logging.warning(
                f"Fold {fold_id} has {len(protein_sequences)} protein sequences; using the first."
            )
        sequence = protein_sequences[0] if protein_sequences else None
    if not sequence:
        sequence = read_first_fasta_sequence(fold_dir, padded)
    if not sequence:
        logging.error(f"Fold {fold_id} has no detectable protein sequence; skipping.")
        return None
    name = name_template.format(id=fold_id, padded=padded)
    return FoldInputs(fold_id=fold_id, name=name, yaml_config=yaml_config, sequence=sequence)


def resolve_user(user_email: Optional[str]) -> User:
    """Resolve or create the user that owns imported folds."""
    if user_email:
        user = db.session.query(User).filter_by(email=user_email).first()
        if user:
            return user
        logging.info(f"Creating user {user_email} for import.")
        return User.create(email=user_email, access_type="editor")

    existing = db.session.query(User).order_by(User.id.asc()).first()
    if existing:
        return existing

    default_email = "tester@test.edu"
    logging.info(f"No users found; creating default user {default_email}.")
    return User.create(email=default_email, access_type="editor")


def reset_postgres_sequence(table: str, column: str = "id") -> None:
    """Reset a Postgres sequence to match the current max id."""
    engine = db.engine
    if engine.dialect.name != "postgresql":
        return
    sql = text(
        """
        SELECT setval(
            pg_get_serial_sequence(:table, :column),
            COALESCE((SELECT MAX(id) FROM {table}), 1),
            (SELECT MAX(id) FROM {table}) IS NOT NULL
        )
        """.format(
            table=table
        )
    )
    db.session.execute(sql, {"table": table, "column": column})


def parse_embedding_from_filename(
    file_path: Path, padded_fold_id: str
) -> Optional[Tuple[str, str]]:
    """Parse embedding model and name from an embedding output filename."""
    filename = file_path.name
    prefix = f"{padded_fold_id}_embeddings_"
    if not (filename.startswith(prefix) and filename.endswith(".csv")):
        return None
    remainder = filename[len(prefix) : -len(".csv")]
    for model in sorted(ALLOWED_ESM_MODELS, key=len, reverse=True):
        if remainder.startswith(f"{model}_"):
            return model, remainder[len(model) + 1 :]
    logging.warning(f"Could not parse embedding model from {filename}; skipping.")
    return None


def infer_logit_model(name: str, default_model: str) -> str:
    """Infer naturalness logit model from the run name."""
    if name in ALLOWED_NATURALNESS_MODELS:
        return name
    for model in sorted(ALLOWED_NATURALNESS_MODELS, key=len, reverse=True):
        if model in name:
            return model
    lowered = name.lower()
    if "150m" in lowered:
        return "e1_150m"
    if "300m" in lowered:
        return "e1_300m"
    if "600m" in lowered:
        return "e1_600m"
    return default_model


def mtime_as_datetime(path: Path) -> datetime:
    """Return file mtime as UTC datetime."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def create_invokation(fold_id: int, run_type: str, run_time: datetime) -> Invokation:
    """Create a finished invokation for a fold run."""
    invokation = Invokation(fold_id=fold_id, type=run_type, state="finished")
    invokation.starttime = run_time
    db.session.add(invokation)
    db.session.flush()
    return invokation


def create_embedding_records(fold: Fold, fold_dir: Path, dry_run: bool) -> Tuple[int, int]:
    """Create embedding records from output CSVs."""
    created = 0
    skipped = 0
    embed_dir = fold_dir / "embed"
    if not embed_dir.exists():
        return created, skipped
    padded = f"{fold.id:06d}"
    for csv_path in sorted(embed_dir.glob("*.csv")):
        parsed = parse_embedding_from_filename(csv_path, padded)
        if not parsed:
            skipped += 1
            continue
        embedding_model, name = parsed
        output_fpath = f"embed/{csv_path.name}"
        existing = Embedding.query.filter_by(fold_id=fold.id, output_fpath=output_fpath).first()
        if existing:
            skipped += 1
            continue
        run_time = mtime_as_datetime(csv_path)
        if dry_run:
            logging.info(
                f"[dry-run] Would create embedding {name} ({embedding_model}) for fold {fold.id}."
            )
            created += 1
            continue
        invokation = create_invokation(fold.id, f"embed_{name}", run_time)
        use_msa_context = "msa" in name.lower()
        embedding = Embedding(
            name=name,
            fold_id=fold.id,
            embedding_model=embedding_model,
            use_msa_context=use_msa_context,
            output_fpath=output_fpath,
            date_created=run_time,
            invokation_id=invokation.id,
        )
        db.session.add(embedding)
        created += 1
    return created, skipped


def create_naturalness_records(
    fold: Fold, fold_dir: Path, default_logit_model: str, dry_run: bool
) -> Tuple[int, int]:
    """Create naturalness records from output CSVs."""
    created = 0
    skipped = 0
    naturalness_dir = fold_dir / "naturalness"
    if not naturalness_dir.exists():
        return created, skipped
    for csv_path in sorted(naturalness_dir.glob("naturalness_*_melted.csv")):
        name = csv_path.name[len("naturalness_") : -len("_melted.csv")]
        output_fpath = f"naturalness/{csv_path.name}"
        existing = Naturalness.query.filter_by(fold_id=fold.id, output_fpath=output_fpath).first()
        if existing:
            skipped += 1
            continue
        logit_model = infer_logit_model(name, default_logit_model)
        use_msa_context = "msa" in name.lower() and logit_model.startswith("e1_")
        run_time = mtime_as_datetime(csv_path)
        if dry_run:
            logging.info(
                f"[dry-run] Would create naturalness {name} ({logit_model}) for fold {fold.id}."
            )
            created += 1
            continue
        invokation = create_invokation(fold.id, f"naturalness_{name}", run_time)
        naturalness = Naturalness(
            name=name,
            fold_id=fold.id,
            logit_model=logit_model,
            use_msa_context=use_msa_context,
            output_fpath=output_fpath,
            date_created=run_time,
            invokation_id=invokation.id,
        )
        db.session.add(naturalness)
        created += 1
    return created, skipped


def reindex_foldydata(
    foldydata_dir: Path,
    user_email: Optional[str],
    name_template: str,
    include_runs: bool,
    default_logit_model: str,
    dry_run: bool,
) -> None:
    """Reindex foldydata into database records."""
    user = resolve_user(user_email)
    fold_dirs = list_fold_dirs(foldydata_dir)
    logging.info(f"Found {len(fold_dirs)} fold directories under {foldydata_dir}.")

    for fold_dir in fold_dirs:
        fold_inputs = ensure_fold_inputs(fold_dir, name_template)
        if not fold_inputs:
            continue
        existing = Fold.get_by_id(fold_inputs.fold_id)
        if existing:
            logging.info(f"Fold {fold_inputs.fold_id} already exists; skipping.")
            continue

        name = fold_inputs.name
        if db.session.query(Fold).filter_by(name=name).first():
            name = f"{name}_{fold_inputs.fold_id}"
            logging.warning(f"Name collision; using {name} instead.")

        if dry_run:
            logging.info(
                f"[dry-run] Would create fold {fold_inputs.fold_id} ({name}) for {user.email}."
            )
            if include_runs:
                create_embedding_records(Fold(id=fold_inputs.fold_id), fold_dir, dry_run=True)
                create_naturalness_records(
                    Fold(id=fold_inputs.fold_id),
                    fold_dir,
                    default_logit_model,
                    dry_run=True,
                )
            continue

        fold = Fold(
            id=fold_inputs.fold_id,
            name=name,
            user_id=user.id,
            tagstring="",
            yaml_config=fold_inputs.yaml_config,
            sequence=fold_inputs.sequence,
            af2_model_preset="boltz",
            disable_relaxation=False,
        )
        db.session.add(fold)
        db.session.flush()

        if include_runs:
            created_embeddings, skipped_embeddings = create_embedding_records(
                fold, fold_dir, dry_run=False
            )
            created_naturalness, skipped_naturalness = create_naturalness_records(
                fold, fold_dir, default_logit_model, dry_run=False
            )
            logging.info(
                f"Fold {fold.id}: embeddings created={created_embeddings}, "
                f"skipped={skipped_embeddings}; naturalness created={created_naturalness}, "
                f"skipped={skipped_naturalness}."
            )

        db.session.commit()

    if not dry_run:
        reset_postgres_sequence("roles")
        db.session.commit()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the foldydata reindexer."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    app = create_app()
    with app.app_context():
        foldydata_dir = (
            Path(args.foldydata_dir)
            if args.foldydata_dir
            else Path(app.config["FOLDY_LOCAL_STORAGE_DIR"])
        )
        reindex_foldydata(
            foldydata_dir=foldydata_dir,
            user_email=args.user_email,
            name_template=args.name_template,
            include_runs=args.include_runs,
            default_logit_model=args.default_logit_model,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

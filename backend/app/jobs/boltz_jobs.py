import io
import logging
import re
import string
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml
from Bio.PDB.MMCIFParser import (
    MMCIFParser,  # type: ignore[reportPrivateImportUsage] # Bio.PDB module structure quirk
)
from Bio.PDB.PDBIO import (
    PDBIO,  # type: ignore[reportPrivateImportUsage] # Bio.PDB module structure quirk
)
from werkzeug.exceptions import BadRequest

from app.helpers.boltz_msa import rewrite_msa_query_sequence
from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.jobs_util import (
    LoggingRecorder,
    get_torch_cuda_is_available_and_add_logs,
)
from app.models import Fold, Invokation

# Maximum time to wait for MSA server jobs stuck in PENDING state before failing.
# The ColabFold MSA server can drop or deprioritize jobs, causing them to stay
# PENDING forever. Failing fast allows RQ retry to resubmit with a fresh job ID.
# See: https://github.com/sokrypton/ColabFold/issues/606
MSA_PENDING_TIMEOUT_SECONDS = 10 * 60  # 10 minutes
FOLDY_MSA_PREFIX = "foldy://"


def materialize_foldy_msa_paths(
    yaml_config: str,
    fold_id: int,
    fsm: FoldStorageManager,
    temp_dir: str,
) -> str:
    """Download Foldy-managed MSAs and replace their YAML URIs with local paths."""
    document: Any = yaml.safe_load(yaml_config)
    if not isinstance(document, dict):
        raise BadRequest("Boltz YAML must be a dictionary.")
    if fsm.storage_manager is None:
        raise BadRequest("Storage manager not initialized.")

    for index, sequence_entry in enumerate(document.get("sequences", [])):
        if not isinstance(sequence_entry, dict) or "protein" not in sequence_entry:
            continue
        protein = sequence_entry["protein"]
        msa_uri = protein.get("msa") if isinstance(protein, dict) else None
        if not isinstance(msa_uri, str) or not msa_uri.startswith(FOLDY_MSA_PREFIX):
            continue
        relative_path = msa_uri.removeprefix(FOLDY_MSA_PREFIX).lstrip("/")
        if not relative_path or ".." in Path(relative_path).parts:
            raise BadRequest(f"Invalid Foldy MSA path: {msa_uri}")
        path_parts = relative_path.split("/", 1)
        msa_fold_id = fold_id
        if len(path_parts) == 2 and path_parts[0].isdigit():
            msa_fold_id = int(path_parts[0])
            relative_path = path_parts[1]
        msa_bytes = fsm.storage_manager.get_binary(msa_fold_id, relative_path)
        protein_sequence = protein.get("sequence")
        if not isinstance(protein_sequence, str):
            raise BadRequest("Protein sequence is required when materializing an MSA.")
        msa_bytes = rewrite_msa_query_sequence(msa_bytes, protein_sequence)
        local_path = Path(temp_dir) / f"protein_{index}_msa.csv"
        local_path.write_bytes(msa_bytes)
        protein["msa"] = str(local_path)
    return yaml.safe_dump(document, sort_keys=False)


def try_check_smiles_string_validity(smiles_string):
    """Try to check if a smiles string is valid."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            logging.error(f"Invalid SMILES: {smiles_string}")
    except Exception as e:
        logging.error(f"Error checking SMILES: {smiles_string} {e}")


def run_boltz(fold_id, invokation_id):
    """Run boltz workflow."""
    fold = Fold.get_by_id(fold_id)
    if not fold:
        raise BadRequest(f"Fold {fold_id} not found")
    invokation = Invokation.get_by_id(invokation_id)
    if not invokation:
        raise BadRequest(f"Invokation {invokation_id} not found")

    with LoggingRecorder(invokation):
        logging.info(
            "Starting Boltz execution...",
        )

        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)

        for ligand in boltz_yaml_helper.get_ligands():
            if "smiles" in ligand:
                try_check_smiles_string_validity(ligand["smiles"])

        # Create a foldstoragemanager.
        padded_fold_id = "%06d" % fold_id
        # fasta_relative_path = f"{padded_fold_id}.fasta"

        # Make a temporary directory for running Boltz.
        with TemporaryDirectory() as temp_dir:
            logging.info(f"Got temp directory at {temp_dir}")

            # Download the fasta file to the temporary directory.
            fsm = FoldStorageManager()
            fsm.setup()
            # binary_fasta_str = fsm.storage_manager.get_binary(
            #     fold_id, fasta_relative_path
            # )
            # fasta_file_path = Path(temp_dir) / fasta_relative_path
            # fasta_file_path.write_bytes(binary_fasta_str)
            yaml_file_str = materialize_foldy_msa_paths(fold.yaml_config, fold_id, fsm, temp_dir)
            yaml_file_path = Path(temp_dir) / "input.yml"
            yaml_file_path.write_text(yaml_file_str)
            fsm.storage_manager.write_file(fold_id, "boltz_input.yaml", yaml_file_str)
            logging.info(f"YAML file contents: {yaml_file_str}")

            diffusion_samples = fold.diffusion_samples or 1

            # Run Boltz.
            #
            # Note that we keep running out of shared memory (shm) when running Boltz
            # on A100s on Google Cloud.
            #
            # We increased shared memory to 20Gi but it didn't help.
            #
            # Based on the comments in this issue, it seems like we can improve
            # performance by reducing the number of dataworkers.
            # https://github.com/pytorch/pytorch/issues/5040#issuecomment-439590544
            #
            # Boltz API: https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
            gpu_available = get_torch_cuda_is_available_and_add_logs(logging.info)
            accelerator = "gpu" if gpu_available else "cpu"
            boltz_command = [
                "/opt/conda/envs/boltzenv/bin/boltz",
                "predict",
                str(yaml_file_path),
                "--out_dir",
                str(temp_dir),
                "--use_msa_server",
                "--diffusion_samples",
                str(diffusion_samples),
                "--accelerator",
                accelerator,
                "--cache",
                "/hf-cache/",
                "--num_workers",
                "0",  # Should this be 1 or 0? 1 seems to work ok, but zero doesnt spin up any workers (a behavior which seems to cause a "pin memory" issue for foldy-in-a-box).
                "--use_potentials",
                "--write_full_pae",
                "--write_full_pde",
            ]
            logging.info(f"Running boltz with command: {boltz_command}")

            def upload_msa_outputs() -> None:
                msa_dirs = sorted(Path(temp_dir).glob("boltz_results*/msa"))
                if not msa_dirs:
                    logging.info("No MSA outputs found to upload.")
                    return
                for msa_dir in msa_dirs:
                    relative_path = f"boltz/{msa_dir.parent.name}/msa"
                    logging.info(f"Uploading MSA outputs from {msa_dir} to {relative_path}")
                    fsm.storage_manager.upload_folder(fold_id, str(msa_dir), relative_path)
                input_path = Path(temp_dir) / "input.yml"
                if input_path.exists():
                    fsm.storage_manager.write_file(
                        fold_id,
                        "boltz/input.yml",
                        input_path.read_text(),
                    )

            try:
                process = subprocess.Popen(
                    boltz_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                # Track MSA PENDING state to detect stuck jobs
                pending_start_time = None
                pending_pattern = re.compile(r"PENDING.*\d+/\d+")

                for line in iter(process.stdout.readline, ""):
                    logging.info(line.strip())

                    # Check if this line indicates MSA jobs stuck in PENDING
                    if pending_pattern.search(line):
                        if pending_start_time is None:
                            pending_start_time = time.time()
                            logging.info(
                                f"MSA server jobs entered PENDING state. "
                                f"Will timeout after {MSA_PENDING_TIMEOUT_SECONDS // 60} minutes."
                            )
                        elif time.time() - pending_start_time > MSA_PENDING_TIMEOUT_SECONDS:
                            process.kill()
                            process.wait()
                            raise TimeoutError(
                                f"MSA server jobs stuck in PENDING state for over "
                                f"{MSA_PENDING_TIMEOUT_SECONDS // 60} minutes. "
                                f"The ColabFold server may have dropped this job. "
                                f"See: https://github.com/sokrypton/ColabFold/issues/606"
                            )
                    elif "RUNNING" in line or "COMPLETE" in line:
                        # Reset timer if jobs start making progress
                        if pending_start_time is not None:
                            logging.info("MSA server jobs progressing, resetting PENDING timer.")
                            pending_start_time = None

                process.stdout.close()
                process.wait()

                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
            except Exception as e:
                logging.error(f"Boltz failed; attempting to upload MSA outputs: {e}")
                try:
                    upload_msa_outputs()
                except Exception as upload_error:
                    logging.error(
                        f"Failed to upload MSA outputs after boltz failure: {upload_error}"
                    )
                raise

            logging.info(f'Uploading files {list(Path(temp_dir).glob("*"))}')
            fsm.storage_manager.upload_folder(fold_id, temp_dir, "boltz")
            logging.info(f"Now converting mmCIF to PDB")

            # Use glob to find all files matching the pattern
            cif_files = list(Path(temp_dir).glob("boltz_results*/predictions/*/*_model_0.cif"))
            logging.info(f"Found {len(cif_files)} cif files: {cif_files}")
            if len(cif_files) == 0:
                logging.error(f"No CIF files found in {temp_dir}")
                raise BadRequest(f"No CIF files found in {temp_dir}")

            cif_file = cif_files[0]
            logging.info(f"Copying {cif_file} to ranked_0.cif")

            try:
                fsm.storage_manager.write_file(fold_id, "ranked_0.cif", cif_file.read_text())
            except Exception as e:
                logging.error(f"Error writing CIF to cif: {e}")
                raise e

            logging.info("We no longer convert CIF to PDB. In this case, CIF format is superior!!!")
            logging.info(f"Finished!")

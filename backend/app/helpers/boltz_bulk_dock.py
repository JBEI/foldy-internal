"""Build, submit, and grade Boltz variant-ligand docking matrices."""

import io
import json
import logging
import re
from datetime import UTC, datetime
from hashlib import sha1
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from Bio.PDB.MMCIFParser import MMCIFParser  # type: ignore[reportPrivateImportUsage]
from Bio.PDB.Superimposer import Superimposer
from flask import current_app
from scipy.stats import spearmanr
from werkzeug.exceptions import BadRequest

from app.extensions import db
from app.helpers.boltz_reaction_mapping import (
    compile_comparison_mapping,
    heavy_atom_count,
    molecule_from_smiles,
)
from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.sequence_util import seq_id_to_seq, validate_aa_sequence
from app.models import (
    BoltzDockBatch,
    BoltzDockResult,
    CampaignRound,
    Fold,
    User,
)
from app.util import start_stage

FOLDY_MSA_PREFIX = "foldy://"
DEFAULT_MAX_BATCH_JOBS = 500


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"{field_name} must be a non-empty string.")
    return value.strip()


def _validate_chain_id(value: Any, field_name: str) -> str:
    chain_id = _require_string(value, field_name)
    if not re.fullmatch(r"[A-Za-z0-9]+", chain_id):
        raise BadRequest(f"{field_name} must contain only letters and numbers.")
    return chain_id


def get_source_sequence(source_fold: Fold, protein_chain_id: str) -> str:
    """Return the selected protein chain sequence from a source Fold."""
    if source_fold.yaml_config:
        proteins = dict(BoltzYamlHelper(source_fold.yaml_config).get_protein_sequences())
        if protein_chain_id in proteins:
            return proteins[protein_chain_id]
        if len(proteins) == 1:
            return next(iter(proteins.values()))
        raise BadRequest(f"Source fold {source_fold.id} has no protein chain {protein_chain_id}.")
    if source_fold.sequence:
        return source_fold.sequence
    raise BadRequest(f"Source fold {source_fold.id} has no protein sequence.")


def _normalize_variants(raw_variants: Any, source_sequence: str) -> List[Dict[str, str]]:
    if not isinstance(raw_variants, list) or not raw_variants:
        raise BadRequest("variants must be a non-empty list of seq_ids or variant objects.")

    variants: List[Dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw_variant in enumerate(raw_variants):
        if isinstance(raw_variant, str):
            seq_id = _require_string(raw_variant, f"variants[{index}]")
            try:
                sequence = seq_id_to_seq(source_sequence, seq_id)
            except (AssertionError, ValueError, IndexError) as error:
                raise BadRequest(f"Invalid variant {seq_id}: {error}") from error
        elif isinstance(raw_variant, dict):
            seq_id = _require_string(
                raw_variant.get("seq_id") or raw_variant.get("id"),
                f"variants[{index}].seq_id",
            )
            raw_sequence = raw_variant.get("sequence")
            if raw_sequence:
                sequence = _require_string(raw_sequence, f"variants[{index}].sequence").upper()
            else:
                try:
                    sequence = seq_id_to_seq(source_sequence, seq_id)
                except (AssertionError, ValueError, IndexError) as error:
                    raise BadRequest(f"Invalid variant {seq_id}: {error}") from error
        else:
            raise BadRequest(f"variants[{index}] must be a seq_id string or object.")

        validate_aa_sequence(seq_id, sequence, "boltz")
        if len(sequence) != len(source_sequence):
            raise BadRequest(
                f"Variant {seq_id} has length {len(sequence)}; expected {len(source_sequence)}."
            )
        if seq_id in seen_ids:
            raise BadRequest(f"Duplicate variant seq_id: {seq_id}.")
        seen_ids.add(seq_id)
        variants.append({"seq_id": seq_id, "sequence": sequence})
    return variants


def _normalize_ligands(raw_ligands: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_ligands, list) or not raw_ligands:
        raise BadRequest("ligands must be a non-empty list.")

    ligands: List[Dict[str, str]] = []
    seen_names: set[str] = set()
    for index, raw_ligand in enumerate(raw_ligands):
        if not isinstance(raw_ligand, dict):
            raise BadRequest(f"ligands[{index}] must be an object.")
        name = _require_string(raw_ligand.get("name"), f"ligands[{index}].name")
        smiles = _require_string(raw_ligand.get("smiles"), f"ligands[{index}].smiles")
        try:
            molecule_from_smiles(smiles, f"ligands[{index}].smiles")
        except ValueError as error:
            raise BadRequest(str(error)) from error
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise BadRequest(
                f"Ligand name {name} must contain only letters, numbers, hyphens, or underscores."
            )
        if name in seen_names:
            raise BadRequest(f"Duplicate ligand name: {name}.")
        seen_names.add(name)
        ligands.append({"name": name, "smiles": smiles})
    return ligands


def _next_component_chain_id(preferred: str, reserved_chains: set[str]) -> str:
    candidates = [preferred, *"CDEFGHIJKLMNOPQRSTUVWXYZ", *"0123456789"]
    for chain_id in dict.fromkeys(candidates):
        if chain_id not in reserved_chains:
            return chain_id
    raise BadRequest("No chain IDs remain for ligand components.")


def _normalize_states(
    raw_states: Any,
    raw_ligands: Any,
    ligand_chain_id: str,
    reserved_chains: set[str],
) -> List[Dict[str, Any]]:
    if raw_states is None:
        ligands = _normalize_ligands(raw_ligands)
        return [
            {
                "name": ligand["name"],
                "role": "ligand",
                "components": [
                    {
                        **ligand,
                        "chain_id": ligand_chain_id,
                        "heavy_atom_count": heavy_atom_count(ligand["smiles"]),
                    }
                ],
            }
            for ligand in ligands
        ]
    if raw_ligands is not None:
        raise BadRequest("Provide states or ligands, not both.")
    if not isinstance(raw_states, list) or not raw_states:
        raise BadRequest("states must be a non-empty list.")

    states: List[Dict[str, Any]] = []
    seen_state_names: set[str] = set()
    for state_index, raw_state in enumerate(raw_states):
        if not isinstance(raw_state, dict):
            raise BadRequest(f"states[{state_index}] must be an object.")
        state_name = _require_string(raw_state.get("name"), f"states[{state_index}].name")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", state_name):
            raise BadRequest(
                f"State name {state_name} must contain only letters, numbers, hyphens, or "
                "underscores."
            )
        if state_name in seen_state_names:
            raise BadRequest(f"Duplicate state name: {state_name}.")
        seen_state_names.add(state_name)
        role = str(raw_state.get("role", "ligand"))
        if role not in {"pre", "post", "ligand"}:
            raise BadRequest(f"states[{state_index}].role must be pre, post, or ligand.")
        raw_components = raw_state.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise BadRequest(f"states[{state_index}].components must be a non-empty list.")

        components: List[Dict[str, Any]] = []
        used_chains = set(reserved_chains)
        seen_component_names: set[str] = set()
        for component_index, raw_component in enumerate(raw_components):
            field_prefix = f"states[{state_index}].components[{component_index}]"
            if not isinstance(raw_component, dict):
                raise BadRequest(f"{field_prefix} must be an object.")
            component_name = _require_string(raw_component.get("name"), f"{field_prefix}.name")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", component_name):
                raise BadRequest(
                    f"Component name {component_name} must contain only letters, numbers, "
                    "hyphens, or underscores."
                )
            if component_name in seen_component_names:
                raise BadRequest(f"Duplicate component name in {state_name}: {component_name}.")
            seen_component_names.add(component_name)
            smiles = _require_string(raw_component.get("smiles"), f"{field_prefix}.smiles")
            try:
                molecule_from_smiles(smiles, f"{field_prefix}.smiles")
            except ValueError as error:
                raise BadRequest(str(error)) from error
            raw_chain_id = raw_component.get("chain_id")
            if raw_chain_id is None:
                chain_id = _next_component_chain_id(ligand_chain_id, used_chains)
            else:
                chain_id = _validate_chain_id(raw_chain_id, f"{field_prefix}.chain_id")
                if chain_id in used_chains:
                    raise BadRequest(f"Chain ID {chain_id} is used more than once in {state_name}.")
            used_chains.add(chain_id)
            components.append(
                {
                    "name": component_name,
                    "smiles": smiles,
                    "chain_id": chain_id,
                    "heavy_atom_count": heavy_atom_count(smiles),
                }
            )
        states.append({"name": state_name, "role": role, "components": components})
    return states


def _normalize_comparisons(
    raw_comparisons: Any, states: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if raw_comparisons is None:
        return []
    if not isinstance(raw_comparisons, list):
        raise BadRequest("comparisons must be a list.")
    states_by_name = {state["name"]: state for state in states}
    comparisons: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw_comparison in enumerate(raw_comparisons):
        if not isinstance(raw_comparison, dict):
            raise BadRequest(f"comparisons[{index}] must be an object.")
        name = _require_string(raw_comparison.get("name"), f"comparisons[{index}].name")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise BadRequest(
                f"Comparison name {name} must contain only letters, numbers, hyphens, or "
                "underscores."
            )
        if name in seen_names:
            raise BadRequest(f"Duplicate comparison name: {name}.")
        seen_names.add(name)
        pre_state = _require_string(
            raw_comparison.get("pre_state"), f"comparisons[{index}].pre_state"
        )
        post_state = _require_string(
            raw_comparison.get("post_state"), f"comparisons[{index}].post_state"
        )
        if pre_state not in states_by_name or post_state not in states_by_name:
            raise BadRequest(f"Comparison {name} references an unknown state.")
        if pre_state == post_state:
            raise BadRequest(f"Comparison {name} must use two different states.")
        try:
            comparisons.append(
                compile_comparison_mapping(
                    {"name": name, "pre_state": pre_state, "post_state": post_state},
                    states_by_name,
                )
            )
        except ValueError as error:
            raise BadRequest(str(error)) from error
    return comparisons


def _normalize_cofactors(raw_cofactors: Any, reserved_chains: set[str]) -> List[Dict[str, str]]:
    if raw_cofactors is None:
        return []
    if not isinstance(raw_cofactors, list):
        raise BadRequest("cofactors must be a list.")

    cofactors: List[Dict[str, str]] = []
    used_chains = set(reserved_chains)
    for index, raw_cofactor in enumerate(raw_cofactors):
        if not isinstance(raw_cofactor, dict):
            raise BadRequest(f"cofactors[{index}] must be an object.")
        chain_id = _validate_chain_id(
            raw_cofactor.get("chain_id") or raw_cofactor.get("id"),
            f"cofactors[{index}].chain_id",
        )
        ccd = _require_string(raw_cofactor.get("ccd"), f"cofactors[{index}].ccd").upper()
        if chain_id in used_chains:
            raise BadRequest(f"Chain ID {chain_id} is used more than once.")
        used_chains.add(chain_id)
        cofactors.append({"chain_id": chain_id, "ccd": ccd})
    return cofactors


def _normalize_bonds(
    raw_bonds: Any, sequence_length: int, protein_chain_id: str
) -> List[Dict[str, List[Any]]]:
    if raw_bonds is None:
        return []
    if not isinstance(raw_bonds, list):
        raise BadRequest("bonds must be a list.")

    bonds: List[Dict[str, List[Any]]] = []
    for index, raw_bond in enumerate(raw_bonds):
        if not isinstance(raw_bond, dict):
            raise BadRequest(f"bonds[{index}] must be an object.")
        atom1 = raw_bond.get("atom1")
        atom2 = raw_bond.get("atom2")
        if not isinstance(atom1, list) or len(atom1) != 3:
            raise BadRequest(f"bonds[{index}].atom1 must be [chain, residue, atom].")
        if not isinstance(atom2, list) or len(atom2) != 3:
            raise BadRequest(f"bonds[{index}].atom2 must be [chain, residue, atom].")
        for atom_name, atom_value in (("atom1", atom1), ("atom2", atom2)):
            _validate_chain_id(atom_value[0], f"bonds[{index}].{atom_name}[0]")
            if not isinstance(atom_value[1], int) or atom_value[1] < 1:
                raise BadRequest(f"bonds[{index}].{atom_name}[1] must be a positive integer.")
            _require_string(atom_value[2], f"bonds[{index}].{atom_name}[2]")
        if atom1[1] > sequence_length and atom1[0] == protein_chain_id:
            raise BadRequest(f"Bond residue {atom1[1]} exceeds protein length {sequence_length}.")
        if atom2[1] > sequence_length and atom2[0] == protein_chain_id:
            raise BadRequest(f"Bond residue {atom2[1]} exceeds protein length {sequence_length}.")
        bonds.append({"atom1": atom1, "atom2": atom2})
    return bonds


def _normalize_pocket(raw_pocket: Any) -> Optional[Dict[str, Any]]:
    if raw_pocket is None:
        return None
    if not isinstance(raw_pocket, dict):
        raise BadRequest("pocket must be an object.")
    contacts = raw_pocket.get("contacts")
    if not isinstance(contacts, list) or not contacts:
        raise BadRequest("pocket.contacts must be a non-empty list of [chain, residue-or-atom].")
    normalized_contacts: List[List[Any]] = []
    for index, contact in enumerate(contacts):
        if not isinstance(contact, list) or len(contact) != 2:
            raise BadRequest(f"pocket.contacts[{index}] must be [chain, residue-or-atom].")
        normalized_contacts.append(
            [
                _validate_chain_id(contact[0], f"pocket.contacts[{index}][0]"),
                contact[1],
            ]
        )
    max_distance = float(raw_pocket.get("max_distance", 6.0))
    if max_distance <= 0:
        raise BadRequest("pocket.max_distance must be positive.")
    return {
        "contacts": normalized_contacts,
        "max_distance": max_distance,
        "force": bool(raw_pocket.get("force", True)),
    }


def prepare_batch_plan(payload: Dict[str, Any], source_fold: Fold) -> Dict[str, Any]:
    """Validate a request and return normalized config plus expanded variants."""
    name = _require_string(payload.get("name"), "name")
    if len(name) > 120:
        raise BadRequest("name must be at most 120 characters.")
    protein_chain_id = _validate_chain_id(payload.get("protein_chain_id", "A"), "protein_chain_id")
    ligand_chain_id = _validate_chain_id(payload.get("ligand_chain_id", "C"), "ligand_chain_id")
    if ligand_chain_id == protein_chain_id:
        raise BadRequest("ligand_chain_id must differ from protein_chain_id.")

    source_sequence = get_source_sequence(source_fold, protein_chain_id)
    variants = _normalize_variants(payload.get("variants"), source_sequence)
    cofactors = _normalize_cofactors(payload.get("cofactors"), {protein_chain_id, ligand_chain_id})
    states = _normalize_states(
        payload.get("states"),
        payload.get("ligands"),
        ligand_chain_id,
        {protein_chain_id, *[cofactor["chain_id"] for cofactor in cofactors]},
    )
    comparisons = _normalize_comparisons(payload.get("comparisons"), states)
    bonds = _normalize_bonds(payload.get("bonds"), len(source_sequence), protein_chain_id)
    pocket = _normalize_pocket(payload.get("pocket"))

    allowed_chains = {
        protein_chain_id,
        *[cofactor["chain_id"] for cofactor in cofactors],
        *[component["chain_id"] for state in states for component in state["components"]],
    }
    for index, bond in enumerate(bonds):
        for atom_name in ("atom1", "atom2"):
            if bond[atom_name][0] not in allowed_chains:
                raise BadRequest(
                    f"bonds[{index}].{atom_name} references undeclared chain "
                    f"{bond[atom_name][0]}."
                )
    if pocket:
        for index, contact in enumerate(pocket["contacts"]):
            if contact[0] not in allowed_chains:
                raise BadRequest(
                    f"pocket.contacts[{index}] references undeclared chain {contact[0]}."
                )

    diffusion_samples = int(payload.get("diffusion_samples", 3))
    if diffusion_samples < 1 or diffusion_samples > 10:
        raise BadRequest("diffusion_samples must be between 1 and 10.")
    msa_mode = payload.get("msa_mode", "server")
    if msa_mode not in {"server", "reuse_source"}:
        raise BadRequest("msa_mode must be 'server' or 'reuse_source'.")

    job_count = len(variants) * len(states)
    max_jobs = int(current_app.config.get("BOLTZ_DOCK_MAX_BATCH_JOBS", DEFAULT_MAX_BATCH_JOBS))
    if job_count > max_jobs:
        raise BadRequest(f"Batch expands to {job_count} jobs; the configured limit is {max_jobs}.")

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        raise BadRequest("tags must be a list.")
    for tag in tags:
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            raise BadRequest(f"Invalid tag: {tag}.")

    activities: Dict[str, float] = {}
    raw_activities = payload.get("activities", [])
    if raw_activities is not None:
        if not isinstance(raw_activities, list):
            raise BadRequest("activities must be a list.")
        for row in raw_activities:
            if not isinstance(row, dict) or "seq_id" not in row or "activity" not in row:
                raise BadRequest("Each activity must contain seq_id and activity.")
            activities[str(row["seq_id"])] = float(row["activity"])

    config = {
        "protein_chain_id": protein_chain_id,
        "ligand_chain_id": ligand_chain_id,
        "diffusion_samples": diffusion_samples,
        "msa_mode": msa_mode,
        "cofactors": cofactors,
        "bonds": bonds,
        "pocket": pocket,
        "tags": tags,
        "activities": activities,
        "variant_ids": [variant["seq_id"] for variant in variants],
        "states": states,
        "comparisons": comparisons,
    }
    return {
        "name": name,
        "source_sequence": source_sequence,
        "variants": variants,
        "states": states,
        # Kept as an alias while clients migrate from single ligands to docking states.
        "ligands": states,
        "job_count": job_count,
        "config": config,
    }


def build_boltz_dock_yaml(sequence: str, state: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Build one Boltz YAML document for a variant-state matrix cell."""
    protein: Dict[str, Any] = {
        "id": config["protein_chain_id"],
        "sequence": sequence,
    }
    if config["msa_mode"] == "reuse_source":
        source_msa = config.get("source_msa")
        if not source_msa:
            raise BadRequest("Source MSA reference was not resolved.")
        protein["msa"] = f"{FOLDY_MSA_PREFIX}{source_msa['fold_id']}/{source_msa['path']}"

    sequences: List[Dict[str, Any]] = [{"protein": protein}]
    for cofactor in config["cofactors"]:
        sequences.append({"ligand": {"id": cofactor["chain_id"], "ccd": cofactor["ccd"]}})
    for component in state["components"]:
        sequences.append(
            {
                "ligand": {
                    "id": component["chain_id"],
                    "smiles": component["smiles"],
                }
            }
        )

    document: Dict[str, Any] = {"version": 1, "sequences": sequences}
    constraints: List[Dict[str, Any]] = [
        {"bond": {"atom1": bond["atom1"], "atom2": bond["atom2"]}} for bond in config["bonds"]
    ]
    if config["pocket"]:
        for component in state["components"]:
            constraints.append(
                {
                    "pocket": {
                        "binder": component["chain_id"],
                        **config["pocket"],
                    }
                }
            )
    if constraints:
        document["constraints"] = constraints
    return yaml.safe_dump(document, sort_keys=False)


def get_source_msa_reference(fsm: FoldStorageManager, source_fold_id: int) -> Dict[str, Any]:
    """Find a Boltz CSV MSA saved under a source fold."""
    if fsm.storage_manager is None:
        raise BadRequest("Storage manager not initialized.")
    files = fsm.storage_manager.list_files(source_fold_id, subfolder="boltz")
    candidates = [
        str(file_info["key"]).lstrip("/")
        for file_info in files
        if str(file_info["key"]).endswith("/msa/input_0.csv")
    ]
    if not candidates:
        raise BadRequest(
            f"Source fold {source_fold_id} has no saved Boltz input_0.csv MSA; use msa_mode=server."
        )
    candidates.sort(key=lambda path: ("boltz_results_input" not in path, len(path), path))
    return {"fold_id": source_fold_id, "path": f"boltz/{candidates[0]}"}


def _child_fold_name(batch_id: int, seq_id: str, ligand_name: str) -> str:
    raw_name = f"BoltzDock_{batch_id}_{ligand_name}_{seq_id}"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_name)
    if len(safe_name) <= 80:
        return safe_name
    digest = sha1(safe_name.encode("utf-8")).hexdigest()[:10]
    return f"{safe_name[:69]}_{digest}"


def create_batch(
    payload: Dict[str, Any], user_email: str, fsm: FoldStorageManager
) -> BoltzDockBatch:
    """Create child folds for a request, persist provenance, and optionally queue Boltz."""
    source_fold_id = payload.get("source_fold_id")
    if not isinstance(source_fold_id, int):
        raise BadRequest("source_fold_id must be an integer.")
    source_fold = Fold.get_by_id(source_fold_id)
    if not source_fold:
        raise BadRequest(f"Source fold {source_fold_id} not found.")
    user = db.session.query(User).filter_by(email=user_email).first()
    if not user:
        raise BadRequest(f"User {user_email} not found.")

    campaign_round_id = payload.get("campaign_round_id")
    campaign_round: Optional[CampaignRound] = None
    if campaign_round_id is not None:
        campaign_round = CampaignRound.get_by_id(campaign_round_id)
        if not campaign_round:
            raise BadRequest(f"Campaign round {campaign_round_id} not found.")
        if campaign_round.campaign.fold_id != source_fold_id:
            raise BadRequest("Campaign round and source fold do not match.")

    plan = prepare_batch_plan(payload, source_fold)
    config = plan["config"]
    if config["msa_mode"] == "reuse_source":
        config["source_msa"] = get_source_msa_reference(fsm, source_fold_id)

    batch = BoltzDockBatch(
        name=plan["name"],
        source_fold_id=source_fold_id,
        user_id=user.id,
        config=config,
        campaign_round_id=campaign_round.id if campaign_round else None,
    )
    db.session.add(batch)
    db.session.flush()

    base_tags = ["BoltzDock", f"BoltzDockBatch-{batch.id}", *config["tags"]]
    tagstring = ",".join(dict.fromkeys(base_tags))
    entries: List[BoltzDockResult] = []
    for variant in plan["variants"]:
        for state in plan["states"]:
            yaml_config = build_boltz_dock_yaml(variant["sequence"], state, config)
            fold = Fold(
                name=_child_fold_name(batch.id, variant["seq_id"], state["name"]),
                user_id=user.id,
                tagstring=tagstring,
                yaml_config=yaml_config,
                diffusion_samples=config["diffusion_samples"],
                sequence=variant["sequence"],
                af2_model_preset="boltz",
                disable_relaxation=False,
                public=source_fold.public,
            )
            db.session.add(fold)
            db.session.flush()
            entry = BoltzDockResult(
                batch_id=batch.id,
                fold_id=fold.id,
                seq_id=variant["seq_id"],
                sequence=variant["sequence"],
                ligand_name=state["name"],
                ligand_smiles=".".join(component["smiles"] for component in state["components"]),
                state_data=state,
            )
            db.session.add(entry)
            entries.append(entry)
    db.session.commit()

    start_jobs = bool(payload.get("start_jobs", True))
    assert fsm.storage_manager is not None
    for entry in entries:
        try:
            fsm.write_fastas(entry.fold_id, entry.fold.yaml_config)
            if start_jobs:
                start_stage(entry.fold_id, "boltz", False)
        except Exception as error:
            logging.error(f"Failed to prepare Boltz dock fold {entry.fold_id}: {error}")
            entry.setup_error = str(error)
    db.session.commit()
    return batch


def _get_boltz_state(entry: BoltzDockResult) -> str:
    if entry.setup_error:
        return "failed"
    boltz_jobs = [job for job in entry.fold.jobs if job.type == "boltz"]
    if not boltz_jobs:
        return "not_started"
    return boltz_jobs[-1].state or "unknown"


def _find_atom(structure: Any, chain_id: str, residue_number: Any, atom_name: str) -> Any:
    try:
        chain = next(structure.get_models())[chain_id]
    except (KeyError, StopIteration) as error:
        raise ValueError(f"Chain {chain_id} not found") from error
    for residue in chain.get_residues():
        if residue_number is not None and residue.id[1] != int(residue_number):
            continue
        if atom_name in residue:
            return residue[atom_name]
    residue_description = f" residue {residue_number}" if residue_number is not None else ""
    raise ValueError(f"Atom {chain_id}{residue_description}:{atom_name} not found")


def _ligand_atoms(structure: Any, chain_id: str) -> Dict[str, Any]:
    try:
        chain = next(structure.get_models())[chain_id]
    except (KeyError, StopIteration) as error:
        raise ValueError(f"Ligand chain {chain_id} not found") from error
    atoms = {
        atom.get_name(): atom for atom in chain.get_atoms() if (atom.element or "").upper() != "H"
    }
    if not atoms:
        raise ValueError(f"Ligand chain {chain_id} has no heavy atoms")
    return atoms


def _ligand_atom_list(structure: Any, chain_id: str) -> List[Any]:
    try:
        chain = next(structure.get_models())[chain_id]
    except (KeyError, StopIteration) as error:
        raise ValueError(f"Ligand chain {chain_id} not found") from error
    atoms = [atom for atom in chain.get_atoms() if (atom.element or "").upper() != "H"]
    if not atoms:
        raise ValueError(f"Ligand chain {chain_id} has no heavy atoms")
    return atoms


def _entry_state(entry: BoltzDockResult) -> Dict[str, Any]:
    if entry.state_data:
        return entry.state_data
    chain_id = entry.batch.config.get("ligand_chain_id", "C")
    return {
        "name": entry.ligand_name,
        "role": "ligand",
        "components": [
            {
                "name": entry.ligand_name,
                "smiles": entry.ligand_smiles,
                "chain_id": chain_id,
                "heavy_atom_count": heavy_atom_count(entry.ligand_smiles),
            }
        ],
    }


def _parse_structure(cif_bytes: bytes, label: str) -> Any:
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(label, io.StringIO(cif_bytes.decode("utf-8")))


def _protein_superimposer(reference: Any, mobile: Any, protein_chain_id: str) -> Superimposer:
    reference_model = next(reference.get_models())
    mobile_model = next(mobile.get_models())
    reference_cas: Dict[int, Any] = {
        residue.id[1]: residue["CA"]
        for residue in reference_model[protein_chain_id].get_residues()
        if residue.id[0] == " " and "CA" in residue
    }
    mobile_cas: Dict[int, Any] = {
        residue.id[1]: residue["CA"]
        for residue in mobile_model[protein_chain_id].get_residues()
        if residue.id[0] == " " and "CA" in residue
    }
    shared_residues = sorted(reference_cas.keys() & mobile_cas.keys())
    if len(shared_residues) < 3:
        raise ValueError("Fewer than three shared protein CA atoms")
    superimposer = Superimposer()
    superimposer.set_atoms(
        [reference_cas[index] for index in shared_residues],
        [mobile_cas[index] for index in shared_residues],
    )
    return superimposer


def _pose_rmsd(
    reference: Any,
    mobile: Any,
    component_chain_ids: List[str],
    protein_chain_id: str,
) -> float:
    superimposer = _protein_superimposer(reference, mobile, protein_chain_id)
    rotation, translation = superimposer.rotran  # type: ignore[reportGeneralTypeIssues] # Superimposer.rotran is populated by run()
    squared_distances: List[float] = []
    for chain_id in component_chain_ids:
        reference_atoms = _ligand_atom_list(reference, chain_id)
        mobile_atoms = _ligand_atom_list(mobile, chain_id)
        if len(reference_atoms) != len(mobile_atoms):
            raise ValueError(
                f"Ligand chain {chain_id} atom counts differ between diffusion samples"
            )
        for reference_atom, mobile_atom in zip(reference_atoms, mobile_atoms):
            mobile_coord = np.dot(mobile_atom.coord, rotation) + translation
            delta = reference_atom.coord - mobile_coord
            squared_distances.append(float(np.dot(delta, delta)))
    if not squared_distances:
        raise ValueError("Predicted state poses have no shared heavy atoms")
    return float(np.sqrt(np.mean(squared_distances)))


def _find_boltz_output_paths(
    entry: BoltzDockResult, fsm: FoldStorageManager
) -> tuple[Dict[int, str], Dict[int, str]]:
    assert fsm.storage_manager is not None
    files = fsm.storage_manager.list_files(entry.fold_id, subfolder="boltz")
    confidence_paths: Dict[int, str] = {}
    cif_paths: Dict[int, str] = {}
    for file_info in files:
        key = str(file_info["key"]).lstrip("/")
        confidence_match = re.search(r"confidence_.+_model_(\d+)\.json$", key)
        cif_match = re.search(r"/(?:.+)_model_(\d+)\.cif$", key)
        if confidence_match:
            confidence_paths[int(confidence_match.group(1))] = f"boltz/{key}"
        if cif_match and "/predictions/" in key:
            cif_paths[int(cif_match.group(1))] = f"boltz/{key}"
    return confidence_paths, cif_paths


def grade_entry(entry: BoltzDockResult, fsm: FoldStorageManager) -> Optional[Dict[str, Any]]:
    """Read completed Boltz outputs and calculate confidence and geometry metrics."""
    if fsm.storage_manager is None:
        raise BadRequest("Storage manager not initialized.")
    confidence_paths, cif_paths = _find_boltz_output_paths(entry, fsm)
    model_numbers = sorted(confidence_paths.keys() & cif_paths.keys())
    if not model_numbers:
        return None

    config = entry.batch.config
    state = _entry_state(entry)
    components = state["components"]
    component_chain_ids = [component["chain_id"] for component in components]
    anchor_bond = config["bonds"][0] if config.get("bonds") else None
    pocket_contact = (config.get("pocket") or {}).get("contacts", [None])[0]
    model_scores: List[Dict[str, Any]] = []
    structures: Dict[int, Any] = {}
    for model_number in model_numbers:
        confidence = json.loads(
            fsm.storage_manager.get_binary(entry.fold_id, confidence_paths[model_number]).decode(
                "utf-8"
            )
        )
        structure = _parse_structure(
            fsm.storage_manager.get_binary(entry.fold_id, cif_paths[model_number]),
            f"dock_{entry.id}_{model_number}",
        )
        structures[model_number] = structure
        score: Dict[str, Any] = {
            "model": model_number,
            "confidence_score": confidence.get("confidence_score"),
            "ptm": confidence.get("ptm"),
            "iptm": confidence.get("iptm"),
            "ligand_iptm": confidence.get("ligand_iptm"),
            "complex_plddt": confidence.get("complex_plddt"),
            "complex_iplddt": confidence.get("complex_iplddt"),
        }
        warnings: List[str] = []
        component_scores: List[Dict[str, Any]] = []
        all_component_atoms: List[Any] = []
        for component in components:
            component_score: Dict[str, Any] = {
                "name": component["name"],
                "chain_id": component["chain_id"],
            }
            try:
                component_atoms = _ligand_atom_list(structure, component["chain_id"])
                all_component_atoms.extend(component_atoms)
                component_score["plddt"] = float(
                    np.mean([atom.get_bfactor() / 100.0 for atom in component_atoms])
                )
                if pocket_contact:
                    target_atom = _find_atom(
                        structure, pocket_contact[0], None, str(pocket_contact[1])
                    )
                    component_score["target_distance"] = min(
                        float(np.linalg.norm(atom.coord - target_atom.coord))
                        for atom in component_atoms
                    )
            except ValueError as error:
                warnings.append(str(error))
            component_scores.append(component_score)
        if all_component_atoms:
            score["ligand_plddt"] = float(
                np.mean([atom.get_bfactor() / 100.0 for atom in all_component_atoms])
            )
        target_distances = [
            float(component_score["target_distance"])
            for component_score in component_scores
            if component_score.get("target_distance") is not None
        ]
        if target_distances:
            # A joint pre-state is only catalytically placed when every substrate is nearby.
            score["target_distance"] = max(target_distances)
            score["closest_target_distance"] = min(target_distances)
        score["components"] = component_scores
        if anchor_bond:
            try:
                atom1 = _find_atom(structure, *anchor_bond["atom1"])
                atom2 = _find_atom(structure, *anchor_bond["atom2"])
                score["anchor_distance"] = float(np.linalg.norm(atom1.coord - atom2.coord))
            except ValueError as error:
                warnings.append(str(error))
        if warnings:
            score["warnings"] = warnings
        model_scores.append(score)

    best_score = max(
        model_scores,
        key=lambda score: float(score.get("confidence_score") or float("-inf")),
    )
    reference_number = int(best_score["model"])
    pose_rmsds: List[float] = []
    pose_warnings: List[str] = []
    for model_number in model_numbers:
        if model_number == reference_number:
            continue
        try:
            pose_rmsds.append(
                _pose_rmsd(
                    structures[reference_number],
                    structures[model_number],
                    component_chain_ids,
                    config["protein_chain_id"],
                )
            )
        except ValueError as error:
            pose_warnings.append(str(error))

    result = dict(best_score)
    result.update(
        {
            "best_model": reference_number,
            "model_count": len(model_numbers),
            "pose_rmsd": float(np.mean(pose_rmsds)) if pose_rmsds else None,
            "models": model_scores,
        }
    )
    if pose_warnings:
        result["pose_warnings"] = pose_warnings
    return result


def _load_scored_structure(entry: BoltzDockResult, fsm: FoldStorageManager) -> tuple[Any, int]:
    if not entry.score_data or entry.score_data.get("best_model") is None:
        raise ValueError(f"State {entry.ligand_name} has not been structurally graded")
    model_number = int(entry.score_data["best_model"])
    _, cif_paths = _find_boltz_output_paths(entry, fsm)
    if model_number not in cif_paths:
        raise ValueError(f"Best model {model_number} is missing for state {entry.ligand_name}")
    assert fsm.storage_manager is not None
    structure = _parse_structure(
        fsm.storage_manager.get_binary(entry.fold_id, cif_paths[model_number]),
        f"comparison_{entry.id}_{model_number}",
    )
    return structure, model_number


def _grade_structural_comparison(
    definition: Dict[str, Any],
    pre_entry: BoltzDockResult,
    post_entry: BoltzDockResult,
    fsm: FoldStorageManager,
) -> Dict[str, Any]:
    post_structure, post_model = _load_scored_structure(post_entry, fsm)
    pre_structure, pre_model = _load_scored_structure(pre_entry, fsm)
    superimposer = _protein_superimposer(
        post_structure,
        pre_structure,
        pre_entry.batch.config["protein_chain_id"],
    )
    rotation, translation = superimposer.rotran  # type: ignore[reportGeneralTypeIssues] # Superimposer.rotran is populated by run()

    best_rmsd: Optional[float] = None
    best_distances: List[float] = []
    best_mapping_index: Optional[int] = None
    for mapping_index, mapping_option in enumerate(definition["mapping_options"]):
        distances: List[float] = []
        for component_mapping in mapping_option:
            pre_atoms = _ligand_atom_list(pre_structure, component_mapping["pre_chain_id"])
            post_atoms = _ligand_atom_list(post_structure, component_mapping["post_chain_id"])
            for pre_atom_index, post_atom_index in component_mapping["atom_pairs"]:
                if pre_atom_index >= len(pre_atoms) or post_atom_index >= len(post_atoms):
                    raise ValueError(
                        "Predicted ligand atom order does not match the submitted SMILES"
                    )
                pre_coord = np.dot(pre_atoms[pre_atom_index].coord, rotation) + translation
                distances.append(
                    float(np.linalg.norm(post_atoms[post_atom_index].coord - pre_coord))
                )
        if not distances:
            continue
        rmsd = float(np.sqrt(np.mean(np.square(distances))))
        if best_rmsd is None or rmsd < best_rmsd:
            best_rmsd = rmsd
            best_distances = distances
            best_mapping_index = mapping_index
    if best_rmsd is None:
        raise ValueError(f"Comparison {definition['name']} has no maintained atom coordinates")

    pre_heavy_atom_count = sum(
        component["pre_heavy_atom_count"] for component in definition["pre_components"]
    )
    lost_atoms = [
        {
            "component": component["pre_component"],
            "atoms": component["lost_atoms"],
        }
        for component in definition["pre_components"]
        if component["lost_atoms"]
    ]
    return {
        "comparison_name": definition["name"],
        "seq_id": pre_entry.seq_id,
        "pre_state": definition["pre_state"],
        "post_state": definition["post_state"],
        "pre_fold_id": pre_entry.fold_id,
        "post_fold_id": post_entry.fold_id,
        "pre_model": pre_model,
        "post_model": post_model,
        "maintained_atom_rmsd": best_rmsd,
        "mean_maintained_atom_displacement": float(np.mean(best_distances)),
        "max_maintained_atom_displacement": float(np.max(best_distances)),
        "mapped_atom_count": definition["mapped_atom_count"],
        "pre_heavy_atom_count": pre_heavy_atom_count,
        "post_heavy_atom_count": definition["post_heavy_atom_count"],
        "maintained_atom_fraction": definition["mapped_atom_count"] / pre_heavy_atom_count,
        "lost_atom_count": sum(len(item["atoms"]) for item in lost_atoms),
        "lost_atoms": lost_atoms,
        "post_unmapped_atoms": definition["post_unmapped_atoms"],
        "symmetry_mapping_index": best_mapping_index,
    }


def _grade_batch_comparisons(batch: BoltzDockBatch, fsm: FoldStorageManager) -> None:
    definitions = batch.config.get("comparisons", [])
    if not definitions:
        batch.comparison_data = None
        return
    entries_by_cell = {(entry.seq_id, entry.ligand_name): entry for entry in batch.entries}  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
    results: List[Dict[str, Any]] = []
    for definition in definitions:
        for seq_id in batch.config.get("variant_ids", []):
            pre_entry = entries_by_cell.get((seq_id, definition["pre_state"]))
            post_entry = entries_by_cell.get((seq_id, definition["post_state"]))
            if (
                not pre_entry
                or not post_entry
                or not pre_entry.score_data
                or not post_entry.score_data
            ):
                continue
            try:
                result = _grade_structural_comparison(definition, pre_entry, post_entry, fsm)
            except Exception as error:
                logging.error(
                    f"Failed Boltz state comparison {definition['name']} for {seq_id}: {error}"
                )
                result = {
                    "comparison_name": definition["name"],
                    "seq_id": seq_id,
                    "pre_state": definition["pre_state"],
                    "post_state": definition["post_state"],
                    "comparison_error": str(error),
                }
            results.append(result)
    batch.comparison_data = {
        "definitions": [
            {key: value for key, value in definition.items() if key != "mapping_options"}
            for definition in definitions
        ],
        "results": results,
    }


def grade_batch(batch: BoltzDockBatch, fsm: FoldStorageManager) -> int:
    """Grade all available, ungraded outputs in a batch and return the update count."""
    updated = 0
    for entry in batch.entries:  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        if _get_boltz_state(entry) != "finished":
            continue
        try:
            score_data = grade_entry(entry, fsm)
        except Exception as error:
            logging.error(f"Failed grading Boltz dock result {entry.id}: {error}")
            entry.score_data = {"grading_error": str(error)}
            continue
        if score_data is not None:
            entry.score_data = score_data
            entry.graded_at = datetime.now(UTC)
            updated += 1
    _grade_batch_comparisons(batch, fsm)
    db.session.commit()
    return updated


def serialize_batch(batch: BoltzDockBatch, include_entries: bool = True) -> Dict[str, Any]:
    """Serialize a batch with derived job states, pose-quality ranks, and WT deltas."""
    states = [_get_boltz_state(entry) for entry in batch.entries]  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
    state_counts = {state: states.count(state) for state in sorted(set(states))}
    public_config = dict(batch.config)
    public_config["comparisons"] = [
        {key: value for key, value in comparison.items() if key != "mapping_options"}
        for comparison in batch.config.get("comparisons", [])
    ]
    serialized: Dict[str, Any] = {
        "id": batch.id,
        "name": batch.name,
        "source_fold_id": batch.source_fold_id,
        "campaign_round_id": batch.campaign_round_id,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "config": public_config,
        "entry_count": len(batch.entries),
        "state_counts": state_counts,
        "comparison_result_count": len((batch.comparison_data or {}).get("results", [])),
    }
    if not include_entries:
        return serialized
    serialized["comparison_data"] = batch.comparison_data

    activities = batch.config.get("activities", {})
    wt_scores = {
        entry.ligand_name: entry.score_data
        for entry in batch.entries  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        if entry.seq_id == "WT" and entry.score_data
    }
    ranks: Dict[int, int] = {}
    state_summaries: Dict[str, Any] = {}
    for state_name in {entry.ligand_name for entry in batch.entries}:  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        graded_entries = [
            entry
            for entry in batch.entries  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
            if entry.ligand_name == state_name
            and entry.score_data
            and entry.score_data.get("ligand_iptm") is not None
        ]
        graded_entries.sort(key=lambda entry: float(entry.score_data["ligand_iptm"]), reverse=True)
        ranks.update({entry.id: index + 1 for index, entry in enumerate(graded_entries)})
        metric_correlations: Dict[str, Any] = {}
        for metric in ("ligand_iptm", "ligand_plddt", "target_distance", "pose_rmsd"):
            pairs = [
                (float(activities[entry.seq_id]), float(entry.score_data[metric]))
                for entry in graded_entries
                if entry.seq_id in activities and entry.score_data.get(metric) is not None
            ]
            correlation: Optional[float] = None
            if (
                len(pairs) >= 3
                and len({activity for activity, _ in pairs}) > 1
                and len({value for _, value in pairs}) > 1
            ):
                statistic = spearmanr(
                    [activity for activity, _ in pairs],
                    [value for _, value in pairs],
                ).statistic
                if not np.isnan(statistic):
                    correlation = float(statistic)
            metric_correlations[metric] = {"spearman": correlation, "n": len(pairs)}
        state_summaries[state_name] = {
            "graded_count": len(graded_entries),
            "metric_correlations": metric_correlations,
        }

    entries = []
    for entry in batch.entries:  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        score_data = dict(entry.score_data) if entry.score_data else None
        if score_data and entry.ligand_name in wt_scores:
            wt_score = wt_scores[entry.ligand_name]
            if (
                score_data.get("ligand_iptm") is not None
                and wt_score.get("ligand_iptm") is not None
            ):
                score_data["delta_ligand_iptm_vs_wt"] = float(
                    score_data["ligand_iptm"] - wt_score["ligand_iptm"]
                )
        entries.append(
            {
                "id": entry.id,
                "fold_id": entry.fold_id,
                "fold_name": entry.fold.name,
                "seq_id": entry.seq_id,
                "ligand_name": entry.ligand_name,
                "ligand_smiles": entry.ligand_smiles,
                "state_data": _entry_state(entry),
                "state": _get_boltz_state(entry),
                "setup_error": entry.setup_error,
                "graded_at": entry.graded_at.isoformat() if entry.graded_at else None,
                "activity": activities.get(entry.seq_id),
                "pose_quality_rank": ranks.get(entry.id),
                "score_data": score_data,
            }
        )
    serialized["entries"] = entries
    serialized["state_summaries"] = state_summaries
    # Backward-compatible response key for existing single-ligand clients.
    serialized["ligand_summaries"] = state_summaries

    comparison_summaries: Dict[str, Any] = {}
    for comparison in batch.config.get("comparisons", []):
        comparison_results = [
            result
            for result in (batch.comparison_data or {}).get("results", [])
            if result["comparison_name"] == comparison["name"]
            and result.get("maintained_atom_rmsd") is not None
        ]
        pairs = [
            (float(activities[result["seq_id"]]), float(result["maintained_atom_rmsd"]))
            for result in comparison_results
            if result["seq_id"] in activities
        ]
        correlation: Optional[float] = None
        if (
            len(pairs) >= 3
            and len({activity for activity, _ in pairs}) > 1
            and len({value for _, value in pairs}) > 1
        ):
            statistic = spearmanr(
                [activity for activity, _ in pairs],
                [value for _, value in pairs],
            ).statistic
            if not np.isnan(statistic):
                correlation = float(statistic)
        comparison_summaries[comparison["name"]] = {
            "graded_count": len(comparison_results),
            "maintained_atom_rmsd_activity_spearman": correlation,
            "activity_pair_count": len(pairs),
        }
    serialized["comparison_summaries"] = comparison_summaries
    return serialized

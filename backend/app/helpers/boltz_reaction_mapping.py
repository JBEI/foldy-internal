"""Chemical atom mapping for pre/post Boltz docking-state comparisons."""

from itertools import product
from typing import Any, Dict, List

from rdkit import Chem
from rdkit.Chem import rdFMCS

MAX_COMPONENT_MATCHES = 32
MAX_COMBINED_MAPPINGS = 256


def molecule_from_smiles(smiles: str, field_name: str) -> Any:
    """Parse a SMILES string or raise a field-specific validation error."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"{field_name} is not a valid SMILES string.")
    return molecule


def heavy_atom_count(smiles: str) -> int:
    """Return the number of non-hydrogen atoms in a SMILES string."""
    molecule = molecule_from_smiles(smiles, "smiles")
    return sum(atom.GetAtomicNum() != 1 for atom in molecule.GetAtoms())


def _heavy_atom_ordinals(molecule: Any) -> Dict[int, int]:
    return {
        atom.GetIdx(): ordinal
        for ordinal, atom in enumerate(
            atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
        )
    }


def _lost_atoms(molecule: Any, mapped_atom_indices: set[int]) -> List[Dict[str, Any]]:
    ordinals = _heavy_atom_ordinals(molecule)
    return [
        {
            "atom_index": atom.GetIdx(),
            "heavy_atom_index": ordinals[atom.GetIdx()],
            "element": atom.GetSymbol(),
        }
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() != 1 and atom.GetIdx() not in mapped_atom_indices
    ]


def _component_mapping_options(
    pre_component: Dict[str, Any], post_component: Dict[str, Any]
) -> Dict[str, Any]:
    pre_molecule = molecule_from_smiles(pre_component["smiles"], pre_component["name"])
    post_molecule = molecule_from_smiles(post_component["smiles"], post_component["name"])
    mcs = rdFMCS.FindMCS(
        [pre_molecule, post_molecule],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    if mcs.canceled or not mcs.smartsString:
        raise ValueError(
            f"Could not determine a conserved atom mapping from {pre_component['name']} "
            f"to {post_component['name']}."
        )
    query = Chem.MolFromSmarts(mcs.smartsString)
    if query is None:
        raise ValueError(f"Could not parse the conserved mapping for {pre_component['name']}.")

    pre_matches = pre_molecule.GetSubstructMatches(
        query, uniquify=False, maxMatches=MAX_COMPONENT_MATCHES
    )
    post_matches = post_molecule.GetSubstructMatches(
        query, uniquify=False, maxMatches=MAX_COMPONENT_MATCHES
    )
    if not pre_matches or not post_matches:
        raise ValueError(f"No conserved atoms were found for {pre_component['name']}.")

    pre_ordinals = _heavy_atom_ordinals(pre_molecule)
    post_ordinals = _heavy_atom_ordinals(post_molecule)
    options: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[tuple[int, int], ...]] = set()
    for pre_match in pre_matches:
        for post_match in post_matches:
            atom_pairs = tuple(
                (pre_ordinals[pre_index], post_ordinals[post_index])
                for pre_index, post_index in zip(pre_match, post_match)
                if pre_index in pre_ordinals and post_index in post_ordinals
            )
            if not atom_pairs or atom_pairs in seen_pairs:
                continue
            seen_pairs.add(atom_pairs)
            options.append(
                {
                    "pre_component": pre_component["name"],
                    "pre_chain_id": pre_component["chain_id"],
                    "post_component": post_component["name"],
                    "post_chain_id": post_component["chain_id"],
                    "atom_pairs": [list(pair) for pair in atom_pairs],
                    "mapped_atom_count": len(atom_pairs),
                }
            )

    if not options:
        raise ValueError(f"No heavy-atom mapping was found for {pre_component['name']}.")
    first_pre_match = set(pre_matches[0])
    return {
        "pre_component": pre_component["name"],
        "pre_chain_id": pre_component["chain_id"],
        "pre_heavy_atom_count": len(pre_ordinals),
        "mapped_atom_count": len(options[0]["atom_pairs"]),
        "lost_atoms": _lost_atoms(pre_molecule, first_pre_match),
        "options": options,
    }


def compile_comparison_mapping(
    comparison: Dict[str, Any], states_by_name: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compile symmetry-aware, non-overlapping conserved-heavy-atom mappings."""
    pre_state = states_by_name[comparison["pre_state"]]
    post_state = states_by_name[comparison["post_state"]]
    if len(post_state["components"]) != 1:
        raise ValueError(
            f"Comparison {comparison['name']} requires exactly one post-state component."
        )
    post_component = post_state["components"][0]
    component_mappings = [
        _component_mapping_options(component, post_component)
        for component in pre_state["components"]
    ]

    combined_options: List[List[Dict[str, Any]]] = []
    for option_group in product(*(mapping["options"] for mapping in component_mappings)):
        post_atom_indices = [pair[1] for option in option_group for pair in option["atom_pairs"]]
        if len(post_atom_indices) != len(set(post_atom_indices)):
            continue
        combined_options.append(list(option_group))
        if len(combined_options) >= MAX_COMBINED_MAPPINGS:
            break
    if not combined_options:
        raise ValueError(
            f"Comparison {comparison['name']} has no non-overlapping substrate-to-product "
            "atom mapping."
        )

    post_molecule = molecule_from_smiles(post_component["smiles"], post_component["name"])
    mapped_post_atoms = {pair[1] for option in combined_options[0] for pair in option["atom_pairs"]}
    compiled = dict(comparison)
    compiled.update(
        {
            "pre_components": [
                {
                    key: mapping[key]
                    for key in (
                        "pre_component",
                        "pre_chain_id",
                        "pre_heavy_atom_count",
                        "mapped_atom_count",
                        "lost_atoms",
                    )
                }
                for mapping in component_mappings
            ],
            "post_component": post_component["name"],
            "post_chain_id": post_component["chain_id"],
            "post_heavy_atom_count": sum(
                atom.GetAtomicNum() != 1 for atom in post_molecule.GetAtoms()
            ),
            "post_unmapped_atoms": _lost_atoms(post_molecule, mapped_post_atoms),
            "mapped_atom_count": sum(
                mapping["mapped_atom_count"] for mapping in component_mappings
            ),
            "mapping_options": combined_options,
        }
    )
    return compiled

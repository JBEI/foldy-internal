#!/usr/bin/env python
"""Build a custom Boltz CCD component (.pkl) from a molecule file.

Boltz resolves `ccd: CODE` entries by loading `<moldir>/<CODE>.pkl` — a pickled
RDKit Mol carrying per-atom `name`/`leaving_atom` props, a 3D conformer, and a
hex-encoded `symmetries` prop (geometry tensors like `pb_edge_index` are
optional; see boltz/data/mol.py:get_symmetries).

Using a custom component lets us supply our OWN conformer, which bypasses Boltz's
ETKDG embedding (schema.py:compute_3d_conformer) — essential for organometallics
that RDKit cannot embed.

Usage (inside the boltz worker env):
    /opt/conda/envs/boltzenv/bin/python make_custom_ccd_component.py \
        ligand.mol MYLIG --out /hf-cache/mols/MYLIG.pkl
"""

import argparse
import pickle
import sys

import numpy as np
from rdkit import Chem


def prepare_organometallic(mol: Chem.Mol) -> Chem.Mol:
    """Coerce a raw, unsanitized organometallic Mol into a state RDKit will
    accept and pickle, WITHOUT destroying the coordination geometry.

    `mol` is read with sanitize=False / removeHs=False, so it still has your
    Avogadro coordinates in conformer 0, but RDKit has not yet validated
    valences, perceived aromaticity, or assigned stereochemistry. A raw metal
    center will typically fail default Chem.SanitizeMol() on valence.

    Return a Mol that (a) survives pickle.dumps + pickle.loads and (b) keeps the
    metal-donor connectivity intact.
    """
    # TODO(human): implement the metal-safe sanitization.
    # Consider:
    #   - Which bonds are truly covalent vs. dative? RDKit models coordinate
    #     bonds as Chem.BondType.DATIVE (metal is the END atom: donor->metal),
    #     which exempts the metal from donor valence accounting.
    #   - Formal charge / oxidation state on the metal so valence balances.
    #   - Partial sanitization: Chem.SanitizeMol(mol, sanitizeOps=...) with
    #     SANITIZE_ALL ^ SANITIZE_PROPERTIES (skip valence) or similar flags.
    #   - Whether to set noImplicit=True on metal-bonded atoms.
    raise NotImplementedError("prepare_organometallic: implement metal handling")


def assign_atom_metadata(mol: Chem.Mol) -> None:
    """Give every atom a unique <=4-char PDB-style name and leaving_atom=0."""
    seen: dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol().upper()
        seen[sym] = seen.get(sym, 0) + 1
        name = f"{sym}{seen[sym]}"
        if len(name) > 4:
            raise ValueError(f"Atom name '{name}' exceeds 4 characters")
        atom.SetProp("name", name)
        atom.SetProp("alt_name", name)
        atom.SetProp("leaving_atom", "0")


def identity_symmetries(mol: Chem.Mol) -> str:
    """Minimal symmetry entry: the identity permutation, hex-encoded pickle.

    Conservative (treats all atoms as distinguishable); Boltz's symmetry-aware
    loss simply won't swap atoms for this ligand. Safe for pose prediction.
    """
    n = mol.GetNumAtoms()
    perms = np.arange(n, dtype=np.int64).reshape(n, 1)
    return pickle.dumps(perms).hex()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mol_file", help="input .mol/.sdf with 3D coordinates")
    ap.add_argument("code", help="CCD code to register (referenced as ccd: CODE)")
    ap.add_argument("--out", required=True, help="output <CODE>.pkl path")
    args = ap.parse_args()

    raw = Chem.MolFromMolFile(args.mol_file, sanitize=False, removeHs=False)
    if raw is None:
        print(f"RDKit could not read {args.mol_file}", file=sys.stderr)
        return 1
    if raw.GetNumConformers() == 0:
        print("Input has no 3D conformer", file=sys.stderr)
        return 1

    mol = prepare_organometallic(raw)
    assign_atom_metadata(mol)
    mol.SetProp("MOL_NAME", args.code)
    mol.SetProp("symmetries", identity_symmetries(mol))

    # Round-trip guard: the pickle must reload cleanly the way Boltz loads it.
    reloaded = pickle.loads(pickle.dumps(mol))
    assert reloaded.GetNumConformers() >= 1

    with open(args.out, "wb") as f:
        pickle.dump(mol, f)
    print(
        f"Wrote {args.out} ({mol.GetNumAtoms()} atoms, " f"{mol.GetNumConformers()} conformer(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

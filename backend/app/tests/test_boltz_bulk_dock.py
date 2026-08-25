from unittest.mock import MagicMock

import pytest
import yaml
from werkzeug.exceptions import BadRequest

from app.extensions import db
from app.helpers.boltz_bulk_dock import (
    build_boltz_dock_yaml,
    create_batch,
    prepare_batch_plan,
    serialize_batch,
)
from app.helpers.boltz_msa import rewrite_msa_query_sequence


def _payload() -> dict:
    return {
        "name": "test matrix",
        "source_fold_id": 1,
        "variants": ["WT", "A1G"],
        "ligands": [{"name": "acetate", "smiles": "CC(=O)O"}],
        "protein_chain_id": "A",
        "ligand_chain_id": "C",
        "diffusion_samples": 3,
        "msa_mode": "server",
        "cofactors": [{"chain_id": "B", "ccd": "HEM"}],
        "bonds": [{"atom1": ["A", 2, "SG"], "atom2": ["B", 1, "FE"]}],
        "pocket": {"contacts": [["B", "FE"]], "max_distance": 6, "force": True},
        "start_jobs": False,
    }


def _reaction_payload(product_name: str, product_smiles: str) -> dict:
    payload = _payload()
    payload.pop("ligands")
    payload["states"] = [
        {
            "name": "pre_substrates",
            "role": "pre",
            "components": [
                {"name": "substrate_A", "smiles": "OC1=CC=C(CC)C=C1"},
                {"name": "substrate_B", "smiles": "NC1=CC=C(C(O)=O)C=C1"},
            ],
        },
        {
            "name": "product",
            "role": "post",
            "components": [{"name": product_name, "smiles": product_smiles}],
        },
    ]
    payload["comparisons"] = [
        {
            "name": "pre_to_product",
            "pre_state": "pre_substrates",
            "post_state": "product",
        }
    ]
    return payload


def test_prepare_batch_plan_expands_mutant_matrix(app, test_fold):
    with app.app_context():
        payload = _payload()
        payload["bonds"][0]["atom1"] = ["A", 2, "CA"]
        plan = prepare_batch_plan(payload, test_fold)

        assert plan["job_count"] == 2
        assert plan["variants"] == [
            {"seq_id": "WT", "sequence": "ACD"},
            {"seq_id": "A1G", "sequence": "GCD"},
        ]
        assert plan["config"]["cofactors"] == [{"chain_id": "B", "ccd": "HEM"}]


def test_prepare_batch_plan_rejects_bad_mutation(app, test_fold):
    with app.app_context():
        payload = _payload()
        payload["variants"] = ["V1G"]
        with pytest.raises(BadRequest, match="Invalid variant V1G"):
            prepare_batch_plan(payload, test_fold)


def test_build_boltz_yaml_includes_generic_cofactor_constraints(app, test_fold):
    with app.app_context():
        plan = prepare_batch_plan(_payload(), test_fold)
        document = yaml.safe_load(build_boltz_dock_yaml("ACD", plan["ligands"][0], plan["config"]))

        assert document["sequences"][1] == {"ligand": {"id": "B", "ccd": "HEM"}}
        assert document["sequences"][2]["ligand"]["id"] == "C"
        assert document["constraints"][0]["bond"]["atom2"] == ["B", 1, "FE"]
        assert document["constraints"][1]["pocket"]["binder"] == "C"


def test_reaction_state_docks_substrates_together_and_targets_each_to_heme(app, test_fold):
    with app.app_context():
        payload = _reaction_payload("OrthoCC", "OC1=CC=C(CC)C=C1C2=C(N)C=CC(C(O)=O)=C2")
        plan = prepare_batch_plan(payload, test_fold)
        pre_state = plan["states"][0]
        document = yaml.safe_load(build_boltz_dock_yaml("ACD", pre_state, plan["config"]))

        assert plan["job_count"] == 4
        assert [component["chain_id"] for component in pre_state["components"]] == ["C", "D"]
        assert [sequence["ligand"]["id"] for sequence in document["sequences"][1:]] == [
            "B",
            "C",
            "D",
        ]
        assert [
            constraint["pocket"]["binder"]
            for constraint in document["constraints"]
            if "pocket" in constraint
        ] == ["C", "D"]
        assert not any(
            "bond" in constraint and constraint["bond"]["atom1"][0] in {"C", "D"}
            for constraint in document["constraints"]
        )


@pytest.mark.parametrize(
    ("product_name", "product_smiles", "mapped_atoms", "lost_elements"),
    [
        (
            "OrthoCC",
            "OC1=CC=C(CC)C=C1C2=C(N)C=CC(C(O)=O)=C2",
            19,
            [],
        ),
        (
            "Ortho",
            "OC1=CC=C(CC)C=C1NC2=CC=C(C(O)=O)C=C2",
            19,
            [],
        ),
        ("DC", "OC1=CC=C(CC)C=C1C2=CC=C(N)C=C2", 16, ["C", "O", "O"]),
    ],
)
def test_reaction_mapping_only_scores_maintained_atoms(
    app,
    test_fold,
    product_name,
    product_smiles,
    mapped_atoms,
    lost_elements,
):
    with app.app_context():
        plan = prepare_batch_plan(_reaction_payload(product_name, product_smiles), test_fold)
        comparison = plan["config"]["comparisons"][0]

        assert comparison["mapped_atom_count"] == mapped_atoms
        assert [
            atom["element"]
            for component in comparison["pre_components"]
            for atom in component["lost_atoms"]
        ] == lost_elements


def test_rewrite_msa_query_sequence_only_changes_first_data_row():
    msa = b"key,sequence\n-1,ACD\n-1,A-D\n"
    rewritten = rewrite_msa_query_sequence(msa, "GCD").decode("utf-8")
    assert rewritten == "key,sequence\n-1,GCD\n-1,A-D\n"


def test_create_batch_persists_child_folds_without_queueing(app, test_fold, test_user):
    with app.app_context():
        fsm = MagicMock()
        fsm.storage_manager = MagicMock()
        batch = create_batch(_payload(), test_user.email, fsm)

        assert batch.id is not None
        assert len(batch.entries) == 2
        assert {entry.seq_id for entry in batch.entries} == {"WT", "A1G"}  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        assert all(entry.fold.diffusion_samples == 3 for entry in batch.entries)  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        assert all("ccd: HEM" in entry.fold.yaml_config for entry in batch.entries)  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        assert fsm.write_fastas.call_count == 2

        batch.config["activities"] = {"WT": 1.0, "A1G": 1.5}
        for index, entry in enumerate(batch.entries):
            entry.score_data = {"ligand_iptm": 0.8 + index * 0.1}

        serialized = serialize_batch(batch)
        assert serialized["entry_count"] == 2
        assert serialized["state_counts"] == {"not_started": 2}
        correlation = serialized["ligand_summaries"]["acetate"]["metric_correlations"][
            "ligand_iptm"
        ]
        assert correlation == {"spearman": None, "n": 2}

        db.session.rollback()


def test_create_reaction_batch_persists_joint_pre_and_post_states(app, test_fold, test_user):
    with app.app_context():
        fsm = MagicMock()
        fsm.storage_manager = MagicMock()
        payload = _reaction_payload("DC", "OC1=CC=C(CC)C=C1C2=CC=C(N)C=C2")
        batch = create_batch(payload, test_user.email, fsm)

        assert len(batch.entries) == 4
        assert {(entry.seq_id, entry.ligand_name) for entry in batch.entries} == {  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
            ("WT", "pre_substrates"),
            ("WT", "product"),
            ("A1G", "pre_substrates"),
            ("A1G", "product"),
        }
        pre_entries = [entry for entry in batch.entries if entry.ligand_name == "pre_substrates"]  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        assert all(len(entry.state_data["components"]) == 2 for entry in pre_entries)
        assert all("id: D" in entry.fold.yaml_config for entry in pre_entries)
        assert fsm.write_fastas.call_count == 4

        db.session.rollback()

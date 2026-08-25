"""Slow contracts for the two primary ProteinGym benchmark landscapes."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from folde.data import DMS_DIR


def _load_mutation_contract(dms_id: str) -> tuple[Counter[int], set[int], set[str], set[str]]:
    mutants = pd.read_csv(DMS_DIR / f"{dms_id}.csv", usecols=["mutant"])["mutant"].astype(str)
    depth_counts: Counter[int] = Counter()
    positions: set[int] = set()
    singles: set[str] = set()
    multimutant_components: set[str] = set()
    for mutant in mutants:
        alleles = mutant.split(":")
        depth_counts[len(alleles)] += 1
        positions.update(int(allele[1:-1]) for allele in alleles)
        if len(alleles) == 1:
            singles.add(alleles[0])
        else:
            multimutant_components.update(alleles)
    return depth_counts, positions, singles, multimutant_components


@pytest.mark.slow
def test_spg1_olson_dataset_contract() -> None:
    depths, positions, singles, multimutant_components = _load_mutation_contract(
        "SPG1_STRSG_Olson_2014"
    )

    assert depths == {1: 1_045, 2: 535_917}
    assert len(positions) == 55
    assert max(depths) == 2
    assert multimutant_components <= singles


@pytest.mark.slow
def test_spg1_wu_dataset_contract() -> None:
    depths, positions, _, _ = _load_mutation_contract("SPG1_STRSG_Wu_2016")

    assert depths == {1: 76, 2: 2_091, 3: 26_019, 4: 121_174}
    assert positions == {265, 266, 267, 280}
    assert max(depths) == 4

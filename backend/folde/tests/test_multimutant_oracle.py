"""Tests for the Phase 1 oracle boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from folde.benchmarks.multimutant_oracle import ProteinGymFitnessOracle


def _oracle() -> ProteinGymFitnessOracle:
    return ProteinGymFitnessOracle(
        "AA",
        pd.Series({"WT": 0.0, "A1C": 1.0, "A2C": 2.0, "A1C_A2C": 10.0}),
        eligible_seq_ids=["WT", "A1C", "A2C", "A1C_A2C"],
    )


def test_oracle_reveals_only_requested_variants_and_records_calls() -> None:
    oracle = _oracle()

    revealed = oracle.measure(["WT", "A1C"], round_number=0)

    assert revealed.to_dict() == {"WT": 0.0, "A1C": 1.0}
    assert [variant.identity.seq_id for variant in oracle.measured_variants] == ["WT", "A1C"]
    assert oracle.lookup_calls == (("WT", "A1C"),)


def test_oracle_rejects_duplicate_ineligible_and_nonfinite_measurements() -> None:
    oracle = _oracle()
    oracle.measure(["WT"], round_number=0)
    with pytest.raises(ValueError, match="already measured"):
        oracle.measure(["WT"], round_number=1)
    with pytest.raises(ValueError, match="outside the eligible"):
        oracle.measure(["A1D"], round_number=1)

    nonfinite = ProteinGymFitnessOracle("A", pd.Series({"A1C": float("nan")}))
    with pytest.raises(ValueError, match="finite"):
        nonfinite.measure(["A1C"], round_number=1)


def test_oracle_restore_revalidates_checkpoint_values() -> None:
    source = _oracle()
    source.measure(["WT"], round_number=0)
    source.measure(["A2C"], round_number=1)
    restored = _oracle()

    restored.restore(source.measured_variants)

    assert restored.measured_variants == source.measured_variants
    assert restored.lookup_calls == (("WT",), ("A2C",))

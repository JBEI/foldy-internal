"""Guarded fitness oracle for finite measured benchmark landscapes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from folde.candidate_generation.base import MeasuredVariant, variant_identity_from_seq_id


class ProteinGymFitnessOracle:
    """Reveal finite activity values once, only for explicitly requested variants."""

    def __init__(
        self,
        reference_sequence: str,
        activity_series: pd.Series,
        eligible_seq_ids: Sequence[str] | None = None,
    ):
        if not activity_series.index.is_unique:
            raise ValueError("activity index must be unique")
        self._reference_sequence = reference_sequence
        self._activity_series = activity_series.copy()
        self._eligible_seq_ids = frozenset(
            activity_series.index if eligible_seq_ids is None else eligible_seq_ids
        )
        unknown = self._eligible_seq_ids - set(activity_series.index)
        if unknown:
            raise ValueError(f"eligible variants lack activity records: {sorted(unknown)[:5]}")
        self._measurements: list[MeasuredVariant] = []
        self._lookup_calls: list[tuple[str, ...]] = []

    @property
    def measured_variants(self) -> tuple[MeasuredVariant, ...]:
        return tuple(self._measurements)

    @property
    def lookup_calls(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._lookup_calls)

    def measure(self, seq_ids: Sequence[str], round_number: int) -> pd.Series:
        """Reveal a slate and record the exact lookup boundary."""
        requested = tuple(seq_ids)
        if not requested:
            raise ValueError("measurement slate must not be empty")
        if len(requested) != len(set(requested)):
            raise ValueError("measurement slate contains duplicate variants")
        measured_ids = {measurement.identity.seq_id for measurement in self._measurements}
        duplicate = set(requested) & measured_ids
        if duplicate:
            raise ValueError(f"variants were already measured: {sorted(duplicate)}")
        ineligible = set(requested) - self._eligible_seq_ids
        if ineligible:
            raise ValueError(f"variants are outside the eligible universe: {sorted(ineligible)}")
        values = self._activity_series.loc[list(requested)]
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("all requested variants must have finite activity")
        self._lookup_calls.append(requested)
        self._measurements.extend(
            MeasuredVariant(
                identity=variant_identity_from_seq_id(self._reference_sequence, seq_id),
                activity=float(values.loc[seq_id]),
                measured_round=round_number,
            )
            for seq_id in requested
        )
        return values.copy()

    def restore(self, measurements: Sequence[MeasuredVariant]) -> None:
        """Restore a checkpoint into a fresh oracle after validating it against ground truth."""
        if self._measurements or self._lookup_calls:
            raise ValueError("restore requires a fresh oracle")
        by_round: dict[int, list[MeasuredVariant]] = {}
        for measurement in measurements:
            by_round.setdefault(measurement.measured_round, []).append(measurement)
        for round_number in sorted(by_round):
            expected = by_round[round_number]
            revealed = self.measure(
                [measurement.identity.seq_id for measurement in expected], round_number
            )
            for measurement in expected:
                if float(revealed.loc[measurement.identity.seq_id]) != measurement.activity:
                    raise ValueError("checkpoint measurement does not match oracle activity")

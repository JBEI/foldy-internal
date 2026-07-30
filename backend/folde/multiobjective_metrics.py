"""Multi-objective campaign evaluation metrics.

This module extends the single-objective conventions of ``folde/types.py`` and
``folde/campaign.py`` (``MutantMetrics.activity``, ``RoundMetrics.model_spearman``)
to the multi-objective setting described in
``2026-07-21-moce-folde-hybrid-spec.md`` section 10.

The central constraint (spec 10.3) is that Bradley-Terry ranker ``mean_score``
values are ordinal on a per-training-run latent scale: the reference architecture
uses bias-free linear layers throughout, so neither the zero point nor the scale
of a predicted score is identified, and both can drift after every retrain.
Hypervolume is *not* scale-invariant, so:

- Hypervolume over quantile-normalized **measured** assay values, against a
  reference distribution and reference point pinned once per campaign/dataset,
  is a legitimate metric for cross-round and cross-arm comparison
  (``measured_hypervolume``).
- Hypervolume over predicted scores, even quantile-normalized, only describes
  model belief at a particular retrain and must never be treated as a campaign
  outcome or compared across retrains/arms (``diagnostic_predicted_hypervolume``).

To make it hard to accidentally conflate these two uses, there is no generic
``hypervolume(...)`` entry point -- callers must pick one of the two named
functions, and the diagnostic one carries an explicit warning in its docstring
and return type name.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import spearmanr

__all__ = [
    "nondominated_mask",
    "pareto_front",
    "QuantileReference",
    "fit_quantile_reference",
    "measured_hypervolume",
    "diagnostic_predicted_hypervolume",
    "MultiObjectiveRoundMetrics",
    "compute_round_metrics",
    "hypervolume_trajectory",
]


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the nondominated rows of ``points``.

    ``points`` is an (N, M) array using the maximize convention (higher is
    better on every objective/column). A point ``p`` is dominated by point
    ``q`` if ``q`` is at least as good as ``p`` on every objective and
    strictly better on at least one. Duplicate points do not dominate one
    another (neither is strictly better), so exact duplicates are all marked
    nondominated together.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"points must be 2-D (N, M), got shape {points.shape}")
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        pi = points[i]
        # q dominates p iff q >= p elementwise and q > p somewhere.
        ge_all = np.all(points >= pi, axis=1)
        gt_any = np.any(points > pi, axis=1)
        dominated_by_others = ge_all & gt_any
        dominated_by_others[i] = False
        if dominated_by_others.any():
            mask[i] = False
    return mask


def pareto_front(points: np.ndarray, ids: List[str]) -> tuple[np.ndarray, List[str]]:
    """Return the nondominated subset of ``points`` and their ``ids``.

    ``ids`` must have the same length as ``points`` (one id per row).
    """
    points = np.asarray(points, dtype=float)
    if len(ids) != points.shape[0]:
        raise ValueError(f"len(ids)={len(ids)} does not match points.shape[0]={points.shape[0]}")
    mask = nondominated_mask(points)
    front_points = points[mask]
    front_ids = [seq_id for seq_id, keep in zip(ids, mask) if keep]
    return front_points, front_ids


# ---------------------------------------------------------------------------
# Quantile normalization (the shared reference set of spec 10.3 item 3)
# ---------------------------------------------------------------------------


class QuantileReference(BaseModel):
    """A per-objective empirical CDF fit from a fixed reference distribution.

    Two arms (or two rounds) being compared must use the exact same
    ``QuantileReference`` instance/values, per spec 10.3 item 3 ("the same
    normalization statistics ... so that 'one unit of hypervolume' means the
    same thing in both arms being compared"). This model is a plain pydantic
    model (sorted reference samples per objective) so it can be pinned in a
    round manifest per spec 7.6.
    """

    objectives: List[str]
    # sorted reference sample values per objective, used for CDF lookup
    sorted_reference_values: Dict[str, List[float]]

    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        """Quantile-normalize ``values`` against the fitted reference.

        ``values`` must have one column per objective in ``self.objectives``
        (extra columns are ignored, missing columns raise). Output is a
        DataFrame of the same shape/index with entries in [0, 1], using the
        empirical CDF of the reference distribution (fraction of reference
        samples <= value), clipped to [0, 1]. NaNs pass through as NaN.
        """
        missing = [obj for obj in self.objectives if obj not in values.columns]
        if missing:
            raise ValueError(f"values is missing objective columns: {missing}")

        out = {}
        for obj in self.objectives:
            ref = np.asarray(self.sorted_reference_values[obj], dtype=float)
            col = values[obj].to_numpy(dtype=float)
            if len(ref) == 0:
                raise ValueError(f"reference distribution for objective {obj!r} is empty")
            # empirical CDF via searchsorted: fraction of ref <= x
            ranks = np.searchsorted(ref, col, side="right")
            quantiles = ranks / len(ref)
            quantiles = np.clip(quantiles, 0.0, 1.0)
            quantiles = np.where(np.isnan(col), np.nan, quantiles)
            out[obj] = quantiles
        return pd.DataFrame(out, index=values.index)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "QuantileReference":
        return cls.model_validate(data)


def fit_quantile_reference(measured_values: pd.DataFrame) -> QuantileReference:
    """Fit a :class:`QuantileReference` from the full measured distribution.

    ``measured_values`` has one column per objective; NaNs are dropped
    per-column before sorting. This is the "shared reference set" of spec
    10.3 item 3 (e.g. the full DMS activity distribution for a dataset) --
    fit it once, up front, and reuse the same object for every round/arm you
    intend to compare.
    """
    objectives = list(measured_values.columns)
    sorted_reference_values: Dict[str, List[float]] = {}
    for obj in objectives:
        col = measured_values[obj].to_numpy(dtype=float)
        col = col[~np.isnan(col)]
        if len(col) == 0:
            raise ValueError(f"objective {obj!r} has no non-NaN measured values to fit a reference")
        sorted_reference_values[obj] = sorted(float(v) for v in col)
    return QuantileReference(objectives=objectives, sorted_reference_values=sorted_reference_values)


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


def _hypervolume_2d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Exact dominated hypervolume (area) for M=2, maximize convention.

    ``points`` need not be pre-filtered to the Pareto front or sorted; both
    are handled here. Points that do not dominate the reference point on
    every axis (i.e. contribute zero volume) are dropped.
    """
    rx, ry = reference_point
    pts = points[(points[:, 0] > rx) & (points[:, 1] > ry)]
    if len(pts) == 0:
        return 0.0

    mask = nondominated_mask(pts)
    front = pts[mask]

    # Sweep left to right (x ascending). The strip contributed at width
    # [x_{i-1}, x_i] has height equal to the best (max) y among all points
    # with x' >= x_i, since a nondominated point with larger x remains valid
    # (dominates the reference rectangle) across that whole width. Computing
    # a suffix-max of y over x-ascending order gives exactly that height for
    # each point's strip.
    order_asc = np.argsort(front[:, 0])
    front_asc = front[order_asc]
    ys = front_asc[:, 1]
    suffix_max_y = np.maximum.accumulate(ys[::-1])[::-1]
    xs = front_asc[:, 0]
    area = 0.0
    prev_x = rx
    for i in range(len(xs)):
        width = xs[i] - prev_x
        height = suffix_max_y[i] - ry
        area += width * height
        prev_x = xs[i]
    return float(area)


def _hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Dispatch to an exact hypervolume algorithm based on dimensionality."""
    if points.shape[0] == 0:
        return 0.0
    m = points.shape[1]
    if m == 2:
        return _hypervolume_2d(points, reference_point)
    raise NotImplementedError(
        f"Exact hypervolume is only implemented for M=2 objectives (got M={m}). "
        "Implement a correct M>=3 algorithm (e.g. WFG or Beume et al. slicing) "
        "before calling this on higher-dimensional objective sets -- do not "
        "silently approximate."
    )


def _prepare_hypervolume_points(
    values: pd.DataFrame,
    reference: QuantileReference,
    require_all_objectives: bool,
) -> np.ndarray:
    normalized = reference.transform(values[reference.objectives])
    if require_all_objectives:
        normalized = normalized.dropna(axis=0, how="any")
    else:
        # A row missing even one objective cannot contribute to a
        # multi-objective hypervolume point; drop rows with any NaN either way,
        # this flag only documents/allows future policies to diverge.
        normalized = normalized.dropna(axis=0, how="any")
    return normalized.to_numpy(dtype=float)


def measured_hypervolume(
    measured_values: pd.DataFrame,
    reference: QuantileReference,
    reference_point: List[float],
    require_all_objectives: bool = True,
) -> float:
    """Dominated hypervolume of quantile-normalized MEASURED assay values.

    This is the legitimate campaign metric (spec 10.3 item 2): it is anchored
    to ground-truth assay measurements, not to a training-run-specific
    predicted-score scale, so it may be compared across rounds, retrains,
    configurations, and arms -- **provided** all comparisons use the same
    ``reference`` (spec 10.3 item 3) and the same ``reference_point``.

    Args:
        measured_values: index seq_id, one column per objective in
            ``reference.objectives``. NaN allowed (see NaN policy below).
        reference: a :class:`QuantileReference` fit once and shared across
            every round/arm being compared (e.g. via
            :func:`fit_quantile_reference` on the full DMS activity
            distribution for a dataset).
        reference_point: the hypervolume reference point in the *normalized*
            [0, 1] space, one value per objective in ``reference.objectives``
            order. Pin this once in campaign configuration (spec 10.3 item 1)
            and reuse it for the life of the campaign.
        require_all_objectives: NaN policy. If True (default), a variant is
            excluded from the hypervolume computation unless every objective
            in ``reference.objectives`` has a non-NaN measurement for it (a
            variant contributes to a Pareto point only when it is fully
            characterized). If False, rows with any missing objective are
            still dropped in this implementation (there is no well-defined
            partial-objective hypervolume point), but the parameter is kept
            so callers can document/opt into a future imputation policy
            without changing this function's signature.
    """
    reference_point_arr = np.asarray(reference_point, dtype=float)
    if len(reference_point_arr) != len(reference.objectives):
        raise ValueError(
            f"reference_point has {len(reference_point_arr)} entries, "
            f"expected {len(reference.objectives)} (one per objective)"
        )
    points = _prepare_hypervolume_points(measured_values, reference, require_all_objectives)
    return _hypervolume(points, reference_point_arr)


def diagnostic_predicted_hypervolume(
    predicted_values: pd.DataFrame,
    reference: QuantileReference,
    reference_point: List[float],
    require_all_objectives: bool = True,
) -> float:
    """Dominated hypervolume of quantile-normalized PREDICTED ranker scores.

    *** DIAGNOSTIC ONLY. DO NOT REPORT THIS AS CAMPAIGN SUCCESS. ***

    Bradley-Terry ranker ``mean_score`` values are ordinal on a per-training-
    run latent scale (bias-free linear layers throughout, including the
    output layer, so neither the zero point nor the scale is identified, and
    both can shift after every retrain -- spec 10.3). Quantile-normalizing
    against a fixed reference makes this number reproducible for a *single*
    model snapshot, but it still describes model belief, not outcome, and it
    is NOT comparable across rounds (each round may retrain the ranker),
    across configurations, or across arms. Use :func:`measured_hypervolume`
    for any cross-round/cross-arm/campaign-success claim.

    Signature intentionally mirrors :func:`measured_hypervolume` so the only
    difference a caller has to reason about is "predicted vs. measured" --
    but the two are kept as separate functions (not one function with a
    ``kind=`` flag) specifically so a diagnostic call cannot be silently
    substituted for the real metric at a call site.
    """
    reference_point_arr = np.asarray(reference_point, dtype=float)
    if len(reference_point_arr) != len(reference.objectives):
        raise ValueError(
            f"reference_point has {len(reference_point_arr)} entries, "
            f"expected {len(reference.objectives)} (one per objective)"
        )
    points = _prepare_hypervolume_points(predicted_values, reference, require_all_objectives)
    return _hypervolume(points, reference_point_arr)


# ---------------------------------------------------------------------------
# Multi-objective round metrics
# ---------------------------------------------------------------------------


class MultiObjectiveRoundMetrics(BaseModel):
    """Multi-objective analogue of ``folde.types.RoundMetrics``.

    Mirrors the single-objective conventions (``round_num``, a ``misc`` bag
    for auxiliary/exploratory stats) but replaces the single
    ``model_spearman: float`` with a per-objective dict, and reports the
    measured (not predicted) Pareto/hypervolume state of the round.
    """

    round_num: int
    model_spearman: Dict[str, float]
    measured_hypervolume: float
    n_nondominated: int
    pareto_front_ids: List[str]
    misc: Dict[str, object] = {}


def compute_round_metrics(
    measured_df: pd.DataFrame,
    predictions_by_objective: Dict[str, pd.Series],
    reference: QuantileReference,
    reference_point: List[float],
    round_num: int,
    require_all_objectives: bool = True,
) -> MultiObjectiveRoundMetrics:
    """Compute :class:`MultiObjectiveRoundMetrics` for one round.

    Args:
        measured_df: index seq_id, one column per objective in
            ``reference.objectives``. NaN allowed for unmeasured
            objectives/variants.
        predictions_by_objective: objective -> pd.Series of predicted mean
            scores indexed by seq_id. Used ONLY to compute per-objective
            Spearman correlation against measured values (a legitimate use of
            ordinal ranker scores, since Spearman only depends on rank order,
            not on the latent scale/zero-point spec 10.3 warns about).
        reference: shared :class:`QuantileReference`, see
            :func:`measured_hypervolume`.
        reference_point: pinned reference point in normalized space, see
            :func:`measured_hypervolume`.
        round_num: the round number, recorded as-is.
        require_all_objectives: NaN policy forwarded to
            :func:`measured_hypervolume` / the Pareto-front computation. A
            variant is included in the Pareto/hypervolume computation only if
            every objective has a non-NaN measurement (default policy).

    NaN policy: for Spearman, each objective's correlation is computed only
    over seq_ids where both the measured value and the prediction are
    non-NaN for that objective (pairwise-complete, matching
    ``folde.campaign``'s existing use of ``spearmanr`` over fully-observed
    series). If fewer than 2 such points exist, that objective's Spearman is
    recorded as NaN rather than raising.
    """
    missing = [obj for obj in reference.objectives if obj not in measured_df.columns]
    if missing:
        raise ValueError(f"measured_df is missing objective columns: {missing}")

    model_spearman: Dict[str, float] = {}
    for obj in reference.objectives:
        measured_col = measured_df[obj]
        pred = predictions_by_objective.get(obj)
        if pred is None:
            model_spearman[obj] = float("nan")
            continue
        common_ids = measured_col.index.intersection(pred.index)
        measured_vals = measured_col.loc[common_ids]
        pred_vals = pred.loc[common_ids]
        valid_mask = measured_vals.notna() & pred_vals.notna()
        if valid_mask.sum() < 2:
            model_spearman[obj] = float("nan")
            continue
        corr = spearmanr(measured_vals[valid_mask].to_numpy(), pred_vals[valid_mask].to_numpy())[0]
        model_spearman[obj] = float(corr)

    complete_mask = measured_df[reference.objectives].notna().all(axis=1)
    complete_df = measured_df.loc[complete_mask, reference.objectives]

    hv = measured_hypervolume(
        complete_df, reference, reference_point, require_all_objectives=require_all_objectives
    )

    if len(complete_df) == 0:
        n_nondominated = 0
        front_ids: List[str] = []
    else:
        front_points, front_ids = pareto_front(
            complete_df.to_numpy(dtype=float), list(complete_df.index)
        )
        n_nondominated = len(front_ids)

    return MultiObjectiveRoundMetrics(
        round_num=round_num,
        model_spearman=model_spearman,
        measured_hypervolume=hv,
        n_nondominated=n_nondominated,
        pareto_front_ids=front_ids,
        misc={},
    )


# ---------------------------------------------------------------------------
# Cumulative campaign trajectory
# ---------------------------------------------------------------------------


def hypervolume_trajectory(rounds: List[MultiObjectiveRoundMetrics]) -> List[float]:
    """Cumulative measured hypervolume after each round.

    This is the primary Tier-1 metric (spec section 11 criterion 10): the
    hypervolume of the set of all variants measured up to and including a
    given round. Because it is computed over a monotonically growing measured
    set with a fixed reference/reference-point, it is mathematically
    guaranteed to be non-decreasing (adding points to a set can only grow or
    preserve its dominated hypervolume against a fixed reference point). This
    function trusts the per-round ``measured_hypervolume`` values already
    computed cumulatively by the caller (i.e. each round's
    ``measured_hypervolume`` in ``rounds`` must itself already be computed
    over the full measured-so-far set, not just that round's new
    measurements -- see :func:`compute_round_metrics`) and asserts the
    invariant; a violation indicates a bug in the normalization or reference
    point, not a legitimate campaign outcome, and is raised rather than
    silently accepted.
    """
    trajectory: List[float] = []
    prev: Optional[float] = None
    for round_metrics in rounds:
        hv = round_metrics.measured_hypervolume
        if prev is not None and hv < prev - 1e-9:
            raise ValueError(
                f"hypervolume_trajectory is non-monotonic at round {round_metrics.round_num}: "
                f"{hv} < previous {prev}. This indicates a bug in the normalization or "
                "reference point (hypervolume over a growing measured set must be "
                "non-decreasing), not a legitimate campaign outcome."
            )
        trajectory.append(hv)
        prev = hv
    return trajectory

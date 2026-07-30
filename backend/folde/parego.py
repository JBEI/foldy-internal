"""ParEGO multi-objective batch acquisition.

Implements the design in section 6.6 of
`2026-07-21-moce-folde-hybrid-spec.md` (subsections 6.6.2-6.6.6): quantile
normalization of per-ensemble-member predictions, randomized augmented
Chebyshev scalarization, and a batch selector that replays the existing
single-objective "constant liar" rank-1 covariance downdate once per slate
slot, on a freshly-scalarized covariance each time.

This module intentionally does not modify `folde/util.py`. The covariance /
downdate math mirrors `constant_liar_sample` in that module (cited inline
below where reused) but is reimplemented locally with numpy so that this
module has no side effects on, and no shared mutable state with, the
single-objective selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────
# 6.6.2 Quantile normalization
# ─────────────────────────────────────────────────────────────────────────


def quantile_normalize(
    ensemble_predictions: NDArray[np.floating],
    directions: NDArray[np.floating],
    reference: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Quantile-normalize ensemble predictions to [0, 1], per objective AND per member.

    Args:
        ensemble_predictions: (N, S, M) array of candidates x ensemble members x objectives.
        directions: length-M array of +1 (maximize) / -1 (minimize). Applied first so
            everything downstream is in maximize convention.
        reference: optional (R, S, M) array giving the reference distribution used to fit
            each per-(member, objective) empirical CDF. Defaults to `ensemble_predictions`
            itself (the "pool-anchored" map of spec 6.6.2, `Q_pool`). Pass a fixed external
            array to get the "measured-anchored" map, `Q_measured`; in that case values
            falling outside the reference range are clipped to [0, 1].

    Returns:
        (N, S, M) array in [0, 1].

    Critical property (spec 6.6.2): the empirical-CDF map is fit independently for every
    (ensemble member s, objective j) pair, using ONLY that member's own predictions (from
    `reference`, or from `ensemble_predictions` itself when `reference` is None) as the
    reference distribution. Pooling across ensemble members before fitting the map would
    compress ensemble disagreement and destroy the uncertainty signal that
    `parego_select` depends on -- do not do that.
    """
    ensemble_predictions = np.asarray(ensemble_predictions, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if ensemble_predictions.ndim != 3:
        raise ValueError(
            f"ensemble_predictions must be (N, S, M), got shape {ensemble_predictions.shape}"
        )
    N, S, M = ensemble_predictions.shape
    if directions.shape != (M,):
        raise ValueError(f"directions must have shape ({M},), got {directions.shape}")

    directed = ensemble_predictions * directions[None, None, :]

    clip_output = reference is not None
    if reference is None:
        ref = directed
    else:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 3 or reference.shape[1:] != (S, M):
            raise ValueError(f"reference must have shape (R, {S}, {M}), got {reference.shape}")
        ref = reference * directions[None, None, :]

    out = np.empty_like(directed)
    for s in range(S):
        for j in range(M):
            sorted_ref = np.sort(ref[:, s, j])
            R = len(sorted_ref)
            # Empirical CDF, right-continuous: fraction of reference values <= x.
            # rank-based, so invariant to any monotone rescaling of the raw scores.
            ranks = np.searchsorted(sorted_ref, directed[:, s, j], side="right")
            out[:, s, j] = ranks / R

    if clip_output:
        out = np.clip(out, 0.0, 1.0)
    return out


# ─────────────────────────────────────────────────────────────────────────
# 6.6.3 Weight sampling + augmented Chebyshev scalarization
# ─────────────────────────────────────────────────────────────────────────


def sample_simplex_weights(
    n_weights: int,
    n_objectives: int,
    random_state: int,
    method: str = "low_discrepancy",
) -> NDArray[np.floating]:
    """Draw `n_weights` weight vectors on the (M-1)-simplex, reproducible from `random_state`.

    Returns an (n_weights, n_objectives) array; every row is non-negative and sums to 1.
    """
    if n_objectives < 1:
        raise ValueError(f"n_objectives must be >= 1, got {n_objectives}")
    if n_objectives == 1:
        # Degenerate simplex: every weight vector is [1.0].
        return np.ones((n_weights, 1), dtype=np.float64)

    rng = np.random.default_rng(random_state)

    if method == "uniform":
        return rng.dirichlet(np.ones(n_objectives), size=n_weights)

    if method == "low_discrepancy":
        if n_objectives == 2:
            # Deterministic even spread over the 1-simplex; the seed only controls the
            # order in which spread points are assigned to slate slots (which matters
            # downstream, since each slot uses a different weight vector).
            t = (np.arange(n_weights) + 0.5) / n_weights
            rng.shuffle(t)
            return np.stack([t, 1.0 - t], axis=1)

        from scipy.stats import qmc

        sampler = qmc.Sobol(d=n_objectives, scramble=True, seed=rng)
        u = sampler.random(n_weights)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        # Map low-discrepancy uniforms to the simplex via the standard exponential
        # (Dirichlet(1,...,1)) construction: normalize -log(u).
        e = -np.log(u)
        return e / e.sum(axis=1, keepdims=True)

    raise ValueError(f"Unknown method {method!r}")


def chebyshev_scalarize(
    normalized: NDArray[np.floating],
    weights_row: NDArray[np.floating],
    rho: float = 0.05,
) -> NDArray[np.floating]:
    """Augmented Chebyshev scalarization, applied independently per ensemble member.

    Args:
        normalized: (N, S, M) quantile-normalized predictions in [0, 1].
        weights_row: (M,) weight vector for this slate slot, non-negative, summing to 1.
        rho: augmentation coefficient (spec default 0.05).

    Returns:
        (N, S) array -- same shape `constant_liar_sample` already consumes, so the batch
        selector's scalar core does not need to know M.
    """
    normalized = np.asarray(normalized, dtype=np.float64)
    weights_row = np.asarray(weights_row, dtype=np.float64)
    if normalized.ndim != 3:
        raise ValueError(f"normalized must be (N, S, M), got shape {normalized.shape}")
    if weights_row.shape != (normalized.shape[2],):
        raise ValueError(
            f"weights_row must have shape ({normalized.shape[2]},), got {weights_row.shape}"
        )

    weighted = normalized * weights_row[None, None, :]  # (N, S, M)
    return weighted.min(axis=-1) + rho * weighted.sum(axis=-1)


# ─────────────────────────────────────────────────────────────────────────
# Shared covariance / constant-liar downdate helpers.
#
# Reimplemented here (not imported) from `constant_liar_sample` in
# `folde/util.py` (~line 145), per the task constraint that util.py must not
# be modified. Numerically equivalent in spirit to that function (float64,
# ddof=1 (unbiased) std for the lie-noise nugget, matching torch's default
# `.std()`), but represented in *factored* form so the O(q^2 N^2) dense-(N,N)
# bottleneck of a naive reimplementation never materializes.
#
# ── Why factored, and what "factored" means here ───────────────────────────
#
# For a given slate slot, the covariance built from the (S, N) scalarized
# deviations matrix D (rows = ensemble members, columns = candidates) is
#
#     Cov = D^T D / S + sigma2 * I                                      (*)
#
# i.e. a rank-S positive-semidefinite term plus a scaled identity. Every rank-1
# constant-liar downdate
#
#     Cov <- Cov - k k^T / v,   k = Cov[:, idx],   v = Cov[idx, idx]
#
# adds exactly one more rank-1 term to the non-identity part -- it does not
# break the "scaled identity + sum of rank-1 terms" structure, it just grows
# it by one term per downdate. So we never build the dense (N, N) matrix at
# all: we maintain
#
#     Cov = sigma2 * I + sum_r coeff_r * outer(w_r, w_r)
#
# where the first S terms are coeff_r = 1/S, w_r = D[r, :] (exactly
# reproducing (*) termwise, not just up to rounding), and each downdate
# appends one more term with coeff = -1/v, w = k. Note coeff_r for a downdate
# term is allowed to be negative (and v itself could in principle be
# negative under FP cancellation) -- using a signed *coefficient* rather than
# trying to normalize into a unit vector (which would require sqrt(v) and
# breaks for v <= 0) keeps this well-defined in exactly the same cases the
# original dense code was well-defined (division by v, no sqrt of v
# anywhere).
#
# With this representation, `diag(Cov)` and any single column `Cov[:, idx]`
# cost O(rank * N) instead of O(N) trivial-slice-but-O(N^2)-to-produce, where
# rank <= S + q_slate_size (>= S initial terms, plus at most q_slate_size - 1
# downdate terms replayed within a slot). For S=8, q=24 that's rank <= 32,
# vs. N possibly in the tens of thousands -- an O(N / rank) factor
# reduction, and critically no (N, N) allocation ever happens.
#
# The original code re-symmetrizes (`cov = 0.5 * (cov + cov.T)`) after every
# downdate to kill FP drift away from symmetry. In factored form `Cov` is
# symmetric by construction (every term `outer(w_r, w_r)` is symmetric), so
# there is no drift to correct and the resymmetrization step is unnecessary.
# What *can* drift under FP cancellation is a single diagonal entry (a sum of
# positive and negative rank-1 contributions), which could in principle come
# out slightly negative even though the true variance is >= 0. The original
# code already guards exactly this case before the final `sqrt` (see
# `np.clip(variances, 0.0, None)` below, unchanged from the original) -- we
# rely on that same clamp and do not clamp anywhere else, so behavior when a
# variance is well away from zero is untouched.
# ─────────────────────────────────────────────────────────────────────────


class _LowRankCov:
    """`sigma2 * I + sum_r coeff_r * outer(w_r, w_r)`, built incrementally.

    `capacity` rows are preallocated (S initial terms + up to q_slate_size - 1
    downdate terms) so that appending a term during the replay loop never
    reallocates.
    """

    __slots__ = ("sigma2", "N", "W", "coeffs", "rank")

    def __init__(self, sigma2: float, N: int, capacity: int):
        self.sigma2 = sigma2
        self.N = N
        self.W = np.empty((capacity, N), dtype=np.float64)
        self.coeffs = np.empty(capacity, dtype=np.float64)
        self.rank = 0

    def set_initial_terms(self, w_rows: NDArray[np.floating], coeff: float) -> None:
        """Bulk-install the S initial rank-1 terms (all sharing the same coeff = 1/S)."""
        r = w_rows.shape[0]
        self.W[:r] = w_rows
        self.coeffs[:r] = coeff
        self.rank = r

    def diag(self) -> NDArray[np.floating]:
        active_w = self.W[: self.rank]  # (rank, N)
        active_c = self.coeffs[: self.rank]  # (rank,)
        return self.sigma2 + (active_c[:, None] * active_w * active_w).sum(axis=0)

    def column_and_value(self, idx: int):
        """Return (Cov[:, idx], Cov[idx, idx]) in O(rank * N)."""
        active_w = self.W[: self.rank]  # (rank, N)
        active_c = self.coeffs[: self.rank]  # (rank,)
        weighted = active_c * active_w[:, idx]  # (rank,)
        col = active_w.T @ weighted  # (N,)
        col[idx] += self.sigma2
        v = col[idx]
        return col, v

    def append_term(self, w: NDArray[np.floating], coeff: float) -> None:
        self.W[self.rank] = w
        self.coeffs[self.rank] = coeff
        self.rank += 1


def _prior_mean_and_cov(
    scalarized: NDArray[np.floating], lie_noise_stddev_multiplier: float, capacity: int
):
    """Build the empirical prior mean/factored-covariance from an (N, S) scalarized matrix.

    Mirrors the "compute empirical prior mean/covariance" block of `constant_liar_sample`,
    but returns a `_LowRankCov` instead of a dense (N, N) array (see module-level comment
    above for why that's exact, not approximate).
    """
    mat = scalarized.T.astype(np.float64)  # (S, N), matches pred_tensor in util.py
    S, N = mat.shape
    lie_noise_variance = (lie_noise_stddev_multiplier * np.median(mat.std(axis=0, ddof=1))) ** 2

    prior_mean = mat.mean(axis=0)  # (N,)
    devs = mat - prior_mean  # (S, N)

    cov = _LowRankCov(sigma2=lie_noise_variance, N=N, capacity=capacity)
    cov.set_initial_terms(devs, 1.0 / S)
    return prior_mean, cov


def _baseline(prior_mean: NDArray[np.floating], choice_of_baseline: str) -> float:
    if choice_of_baseline == "min":
        return float(prior_mean.min())
    if choice_of_baseline == "mean":
        return float(prior_mean.mean())
    if choice_of_baseline == "max":
        return float(prior_mean.max())
    raise ValueError(f"Invalid choice of baseline {choice_of_baseline}")


def _downdate(
    prior_mean: NDArray[np.floating],
    cov: _LowRankCov,
    idx: int,
    baseline: float,
):
    """Single-point rank-1 "constant liar" downdate at `idx`, observation = `baseline`.

    Mirrors the "single-point GP update with fake observation" block of
    `constant_liar_sample`, operating on the factored `cov` in O(rank * N) instead of
    materializing/mutating a dense (N, N) matrix.
    """
    k_i, v_i = cov.column_and_value(idx)

    delta = (baseline - prior_mean[idx]) / v_i
    prior_mean = prior_mean + k_i * delta

    cov.append_term(k_i, -1.0 / v_i)
    return prior_mean, cov


# ─────────────────────────────────────────────────────────────────────────
# 6.6.4 Batch selection
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ParegoSelectionResult:
    """Result of `parego_select`, with enough provenance for spec 7.5's ranked-candidate record."""

    selected_seq_ids: List[str]
    slot_weights: NDArray[np.floating]  # (q, M), weight vector used at each slate slot
    slot_scalarized_scores: List[float]  # augmented-Chebyshev value of the winner, per slot
    records: List[dict] = field(default_factory=list)
    # Each record: {"seq_id", "slate_slot", "scalarization_weights", "scalarized_score"}


def parego_select(
    ensemble_predictions: NDArray[np.floating],
    seq_ids: NDArray,
    directions: NDArray[np.floating],
    q_slate_size: int,
    lie_noise_stddev_multiplier: float,
    ucb_beta: float = 2.0,
    rho: float = 0.05,
    random_state: int = 42,
    choice_of_baseline: str = "min",
    feasibility: Optional[NDArray[np.floating]] = None,
    min_feasibility: float = 0.0,
    redundancy_penalty_fn: Optional[Callable[[List[int], int], float]] = None,
    weight_method: str = "low_discrepancy",
) -> ParegoSelectionResult:
    """ParEGO batch selection (spec 6.6.4): a different weight vector per slate slot.

    Args:
        ensemble_predictions: (N, S, M) candidates x ensemble members x objectives.
        seq_ids: (N,) candidate identifiers.
        directions: (M,) array of +1 (maximize) / -1 (minimize) per objective.
        q_slate_size: number of candidates to select.
        lie_noise_stddev_multiplier: multiplier for the lie-noise nugget (single
            exploration knob, per spec 6.6.4).
        ucb_beta: UCB exploration coefficient.
        rho: augmented-Chebyshev augmentation coefficient.
        random_state: seed controlling the drawn weight vectors (and low-discrepancy
            slot assignment); reproducible.
        choice_of_baseline: 'min', 'mean', or 'max' -- baseline fake observation for the
            constant-liar downdate.
        feasibility: optional (N,) array in [0, 1]. Candidates with feasibility below
            `min_feasibility` are removed before scalarization (spec 6.6.5).
        min_feasibility: feasibility threshold.
        redundancy_penalty_fn: optional callable(selected_indices, candidate_index) ->
            float penalty, subtracted from the UCB score before argmax at each slot.
            Indices are into the (post-feasibility-filter) candidate pool actually being
            selected over, in the same order as the filtered `seq_ids`.
        weight_method: passed through to `sample_simplex_weights`.

    Returns:
        ParegoSelectionResult with the selected seq_ids in slot order, the per-slot weight
        vectors, and provenance records for spec 7.5.
    """
    ensemble_predictions = np.asarray(ensemble_predictions, dtype=np.float64)
    seq_ids = np.asarray(seq_ids)
    directions = np.asarray(directions, dtype=np.float64)

    if ensemble_predictions.ndim != 3:
        raise ValueError(
            f"ensemble_predictions must be (N, S, M), got shape {ensemble_predictions.shape}"
        )
    N, S, M = ensemble_predictions.shape
    if ensemble_predictions.shape[0] != len(seq_ids):
        raise ValueError(
            f"ensemble_predictions must have the same number of rows as seq_ids, got "
            f"{ensemble_predictions.shape[0]} and {len(seq_ids)}"
        )
    if S < 3:
        raise ValueError(f"Calculating a good variance requires at least 3 models, got {S}")

    # ── 6.6.5 feasibility pre-filter ──────────────────────────────────────
    if feasibility is not None:
        feasibility = np.asarray(feasibility, dtype=np.float64)
        if feasibility.shape != (N,):
            raise ValueError(f"feasibility must have shape ({N},), got {feasibility.shape}")
        keep_mask = feasibility >= min_feasibility
    else:
        keep_mask = np.ones(N, dtype=bool)

    pool_preds = ensemble_predictions[keep_mask]
    pool_ids = seq_ids[keep_mask]
    N2 = pool_preds.shape[0]

    if N2 < q_slate_size:
        raise ValueError(
            f"After feasibility filtering, {N2} candidates remain, fewer than "
            f"q_slate_size ({q_slate_size})"
        )

    # ── 6.6.3 weights + 6.6.2 pool-anchored quantile normalization (once) ──
    weights = sample_simplex_weights(q_slate_size, M, random_state, method=weight_method)
    normalized = quantile_normalize(pool_preds, directions, reference=None)  # (N2, S, M)

    selected: List[int] = []
    slot_scalarized_scores: List[float] = []
    records: List[dict] = []

    # Rank of the factored covariance never exceeds S initial terms plus one downdate
    # term per already-selected candidate (at most q_slate_size - 1 within a slot).
    cov_capacity = S + q_slate_size

    for k in range(q_slate_size):
        w = weights[k]
        g = chebyshev_scalarize(normalized, w, rho=rho)  # (N2, S)

        # Covariance must be rebuilt every slot, since the scalarization changed.
        prior_mean, cov = _prior_mean_and_cov(g, lie_noise_stddev_multiplier, cov_capacity)
        baseline = _baseline(prior_mean, choice_of_baseline)

        # Replay the liar downdate for every candidate already selected, in order.
        for prev_idx in selected:
            prior_mean, cov = _downdate(prior_mean, cov, prev_idx, baseline)

        variances = cov.diag()
        sigmas = np.sqrt(np.clip(variances, 0.0, None))

        ucb = prior_mean + ucb_beta * sigmas
        ucb = ucb.copy()
        if selected:
            ucb[selected] = -np.inf

        if redundancy_penalty_fn is not None:
            penalties = np.array(
                [redundancy_penalty_fn(list(selected), i) for i in range(N2)], dtype=np.float64
            )
            ucb = ucb - penalties
            if selected:
                ucb[selected] = -np.inf

        idx = int(np.argmax(ucb))
        selected.append(idx)

        scalarized_score = float(g[idx].mean())
        slot_scalarized_scores.append(scalarized_score)
        records.append(
            {
                "seq_id": pool_ids[idx],
                "slate_slot": k,
                "scalarization_weights": w.tolist(),
                "scalarized_score": scalarized_score,
            }
        )

    return ParegoSelectionResult(
        selected_seq_ids=[str(pool_ids[i]) for i in selected],
        slot_weights=weights,
        slot_scalarized_scores=slot_scalarized_scores,
        records=records,
    )


# ─────────────────────────────────────────────────────────────────────────
# Evaluation helper (spec 10.3): hypervolume, MEASURED values only.
# ─────────────────────────────────────────────────────────────────────────


def dominated_hypervolume(
    points: NDArray[np.floating], reference_point: NDArray[np.floating]
) -> float:
    """Hypervolume dominated by `points`, bounded below by `reference_point`.

    Maximize convention throughout: a point contributes only the region between it and
    `reference_point` on each axis, and only points that dominate the reference point
    (every coordinate >= the reference) count at all.

    Exact for M=2 objectives via a sweep over the non-dominated front. Raises
    NotImplementedError above M=2 -- extending to M=3 requires a more elaborate sweep
    (e.g. the HSO / Beume et al. algorithm) that isn't needed by anything in this module
    yet.

    Per spec 10.3: this must be computed only on quantile-normalized, MEASURED objective
    values against a reference point pinned once in campaign configuration. It must never
    be applied to raw Bradley-Terry ensemble scores -- those carry an arbitrary,
    per-training-run latent scale (unidentified zero point and scale, since the reference
    ranker architecture is bias-free throughout including the output layer), so a
    hypervolume computed on them is not comparable across rounds, configs, or arms and is
    at best a diagnostic on model belief, never a campaign-success metric.
    """
    points = np.asarray(points, dtype=np.float64)
    reference_point = np.asarray(reference_point, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"points must be 2D (N, M), got shape {points.shape}")
    M = points.shape[1]
    if reference_point.shape != (M,):
        raise ValueError(f"reference_point must have shape ({M},), got {reference_point.shape}")
    if M != 2:
        raise NotImplementedError(
            f"dominated_hypervolume only supports M=2 objectives (exact sweep), got M={M}"
        )

    mask = np.all(points >= reference_point, axis=1)
    pts = points[mask]
    if pts.shape[0] == 0:
        return 0.0

    order = np.argsort(-pts[:, 0])
    pts = pts[order]

    hv = 0.0
    max_y = reference_point[1]
    for x, y in pts:
        if y > max_y:
            hv += (x - reference_point[0]) * (y - max_y)
            max_y = y
    return float(hv)

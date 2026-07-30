"""Equivalence + performance regression tests for the ParEGO batch selector.

`folde/parego.py` was optimized to avoid materializing a dense (N, N) covariance
matrix per slate slot (see the module-level comment above `_LowRankCov` in
`folde/parego.py` for the factored representation). This file pins a small,
self-contained *dense* reference implementation -- a direct copy of the
pre-optimization algorithm -- and asserts that the optimized `parego_select`
reproduces its selections exactly across a range of N/q/S/M/seed combinations.

This is intentionally standalone (no scratchpad dependency, no import of private
helpers from `folde/parego.py`) so it runs in CI on its own.
"""

import os
import time
from typing import List, Optional

import numpy as np
import pytest
from numpy.typing import NDArray

os.environ.setdefault("FOLDE_CONSTANT_LIAR_DEVICE", "cpu")

from folde.parego import (
    chebyshev_scalarize,
    parego_select,
    quantile_normalize,
    sample_simplex_weights,
)

# ─────────────────────────────────────────────────────────────────────────
# Pinned dense reference implementation (pre-optimization algorithm).
#
# This is a verbatim reimplementation of the O(q^2 N^2) selector that
# `parego_select` used to run: full (N, N) covariance rebuilt per slot,
# rank-1 downdates replayed as dense outer-product subtractions with
# re-symmetrization. Kept here only as a correctness oracle; it must stay a
# straight-line copy of "the slow way to do it" so it remains trustworthy as
# a reference independent of any cleverness in the optimized version.
# ─────────────────────────────────────────────────────────────────────────


def _dense_prior_mean_and_cov(scalarized: NDArray[np.floating], lie_noise_stddev_multiplier: float):
    mat = scalarized.T.astype(np.float64)  # (S, N)
    S, N = mat.shape
    lie_noise_variance = (lie_noise_stddev_multiplier * np.median(mat.std(axis=0, ddof=1))) ** 2
    prior_mean = mat.mean(axis=0)
    devs = mat - prior_mean
    cov = (devs.T @ devs) / S
    cov = cov + lie_noise_variance * np.eye(N, dtype=np.float64)
    return prior_mean, cov


def _dense_baseline(prior_mean: NDArray[np.floating], choice_of_baseline: str) -> float:
    if choice_of_baseline == "min":
        return float(prior_mean.min())
    if choice_of_baseline == "mean":
        return float(prior_mean.mean())
    if choice_of_baseline == "max":
        return float(prior_mean.max())
    raise ValueError(f"Invalid choice of baseline {choice_of_baseline}")


def _dense_downdate(prior_mean, cov, idx, baseline):
    v_i = cov[idx, idx]
    k_i = cov[:, idx].copy()
    delta = (baseline - prior_mean[idx]) / v_i
    prior_mean = prior_mean + k_i * delta
    cov = cov - np.outer(k_i, k_i) / v_i
    cov = 0.5 * (cov + cov.T)
    return prior_mean, cov


def dense_parego_select(
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
    weight_method: str = "low_discrepancy",
) -> List[str]:
    ensemble_predictions = np.asarray(ensemble_predictions, dtype=np.float64)
    seq_ids = np.asarray(seq_ids)
    directions = np.asarray(directions, dtype=np.float64)
    N, S, M = ensemble_predictions.shape

    if feasibility is not None:
        feasibility = np.asarray(feasibility, dtype=np.float64)
        keep_mask = feasibility >= min_feasibility
    else:
        keep_mask = np.ones(N, dtype=bool)

    pool_preds = ensemble_predictions[keep_mask]
    pool_ids = seq_ids[keep_mask]

    weights = sample_simplex_weights(q_slate_size, M, random_state, method=weight_method)
    normalized = quantile_normalize(pool_preds, directions, reference=None)

    selected: List[int] = []
    for k in range(q_slate_size):
        w = weights[k]
        g = chebyshev_scalarize(normalized, w, rho=rho)

        prior_mean, cov = _dense_prior_mean_and_cov(g, lie_noise_stddev_multiplier)
        baseline = _dense_baseline(prior_mean, choice_of_baseline)

        for prev_idx in selected:
            prior_mean, cov = _dense_downdate(prior_mean, cov, prev_idx, baseline)

        variances = np.diag(cov)
        sigmas = np.sqrt(np.clip(variances, 0.0, None))
        ucb = prior_mean + ucb_beta * sigmas
        ucb = ucb.copy()
        if selected:
            ucb[selected] = -np.inf

        idx = int(np.argmax(ucb))
        selected.append(idx)

    return [str(pool_ids[i]) for i in selected]


# ─────────────────────────────────────────────────────────────────────────
# Equivalence tests: optimized parego_select vs. the pinned dense reference.
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "N,q,S,M,seed",
    [
        (50, 5, 5, 1, 0),
        (50, 5, 5, 2, 0),
        (120, 12, 8, 2, 1),
        (120, 12, 8, 3, 2),
        (200, 20, 6, 2, 3),
        (200, 24, 8, 1, 4),
        (75, 10, 5, 3, 5),
    ],
)
def test_optimized_matches_dense_reference(N, q, S, M, seed):
    rng = np.random.default_rng(seed)
    preds = rng.normal(size=(N, S, M))
    seq_ids = np.array([f"seq_{i}" for i in range(N)])
    directions = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(M)])
    kwargs = dict(
        lie_noise_stddev_multiplier=0.3,
        ucb_beta=2.0,
        rho=0.05,
        random_state=seed,
        choice_of_baseline="min",
    )

    expected = dense_parego_select(preds, seq_ids, directions, q, **kwargs)
    result = parego_select(preds, seq_ids, directions, q, **kwargs)

    assert result.selected_seq_ids == expected


@pytest.mark.parametrize("choice_of_baseline", ["min", "mean", "max"])
def test_optimized_matches_dense_reference_all_baselines(choice_of_baseline):
    rng = np.random.default_rng(7)
    N, S, M, q = 100, 6, 2, 10
    preds = rng.normal(size=(N, S, M))
    seq_ids = np.array([f"seq_{i}" for i in range(N)])
    directions = np.array([1.0, -1.0])
    kwargs = dict(
        lie_noise_stddev_multiplier=0.4,
        ucb_beta=1.5,
        rho=0.1,
        random_state=11,
        choice_of_baseline=choice_of_baseline,
    )

    expected = dense_parego_select(preds, seq_ids, directions, q, **kwargs)
    result = parego_select(preds, seq_ids, directions, q, **kwargs)

    assert result.selected_seq_ids == expected


def test_optimized_matches_dense_reference_with_feasibility_filter():
    rng = np.random.default_rng(9)
    N, S, M, q = 150, 5, 2, 15
    preds = rng.normal(size=(N, S, M))
    seq_ids = np.array([f"seq_{i}" for i in range(N)])
    directions = np.array([1.0, 1.0])
    feasibility = rng.uniform(size=N)
    kwargs = dict(
        lie_noise_stddev_multiplier=0.3,
        random_state=13,
        feasibility=feasibility,
        min_feasibility=0.3,
    )

    expected = dense_parego_select(preds, seq_ids, directions, q, **kwargs)
    result = parego_select(preds, seq_ids, directions, q, **kwargs)

    assert result.selected_seq_ids == expected


# ─────────────────────────────────────────────────────────────────────────
# Benchmark (not run in the normal suite). Run explicitly with:
#   pytest folde/tests/test_parego_perf_equivalence.py -m slow -s
# or:
#   python folde/tests/test_parego_perf_equivalence.py
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_benchmark_scaling():
    _run_benchmark()


def _run_benchmark():
    print()
    for N in (1000, 4000, 8000, 20000):
        rng = np.random.default_rng(0)
        S, M, q = 8, 2, 24
        preds = rng.normal(size=(N, S, M))
        seq_ids = np.array([f"seq_{i}" for i in range(N)])
        directions = np.array([1.0, -1.0])
        t0 = time.time()
        parego_select(
            preds,
            seq_ids,
            directions,
            q,
            lie_noise_stddev_multiplier=0.3,
            random_state=1,
        )
        dt = time.time() - t0
        print(f"N={N:6d} q={q} S={S} M={M}: {dt:.3f}s")


if __name__ == "__main__":
    _run_benchmark()

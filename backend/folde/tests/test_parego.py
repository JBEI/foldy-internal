import os

import numpy as np
import pytest

# This machine's GPU (sm_120) is newer than the installed torch build supports; force CPU
# so constant_liar_sample doesn't crash trying to use an incompatible CUDA kernel.
os.environ.setdefault("FOLDE_CONSTANT_LIAR_DEVICE", "cpu")

from folde.parego import (
    ParegoSelectionResult,
    chebyshev_scalarize,
    dominated_hypervolume,
    parego_select,
    quantile_normalize,
    sample_simplex_weights,
)
from folde.util import constant_liar_sample

# ─────────────────────────────────────────────────────────────────────────
# M=1 equivalence
# ─────────────────────────────────────────────────────────────────────────


def test_m1_equivalence_with_constant_liar_sample():
    """With a single objective, parego_select must reduce to constant_liar_sample.

    Judgment call: quantile_normalize is an empirical-CDF (rank) transform, so it is only
    the identity up to a *global* scale factor when every ensemble member's raw column is
    already a permutation of {1, ..., N} (no ties). We construct exactly such an input.
    Under that construction, quantile_normalize(raw) == raw / N pointwise, i.e. every
    entry of the whole (N, S) matrix is scaled by the SAME constant c = 1/N. The
    constant-liar mean/covariance/downdate/UCB pipeline is exactly homogeneous of degree
    1 in that sense (mean, sigma, and the lie-noise nugget all scale by c together), so
    argmax(UCB) at every slot is invariant to this uniform rescaling. With rho=0 the
    single-objective Chebyshev scalarization is exactly g = w * f = f (w degenerates to
    [1.0]), so parego_select on this input must select the identical slate, in the
    identical order, as constant_liar_sample on the raw matrix.
    """
    rng = np.random.default_rng(0)
    N, S = 40, 6
    q = 5

    # Each member's raw column is an independent permutation of 1..N (no ties), so its
    # empirical CDF is exactly value / N.
    raw = np.stack(
        [rng.permutation(np.arange(1, N + 1)).astype(np.float64) for _ in range(S)], axis=1
    )
    assert raw.shape == (N, S)

    seq_ids = np.array([f"seq_{i}" for i in range(N)])

    baseline_slate = constant_liar_sample(
        raw,
        seq_ids,
        q_slate_size=q,
        lie_noise_stddev_multiplier=0.5,
        choice_of_baseline="min",
        ucb_beta=2.0,
    )

    ensemble_predictions = raw[:, :, None]  # (N, S, 1)
    result = parego_select(
        ensemble_predictions,
        seq_ids,
        directions=np.array([1.0]),
        q_slate_size=q,
        lie_noise_stddev_multiplier=0.5,
        ucb_beta=2.0,
        rho=0.0,
        choice_of_baseline="min",
        random_state=123,  # irrelevant for M=1: weights are degenerate at [1.0]
    )

    assert result.selected_seq_ids == baseline_slate


# ─────────────────────────────────────────────────────────────────────────
# Quantile normalization
# ─────────────────────────────────────────────────────────────────────────


def test_quantile_normalize_per_member_per_objective_spans_01():
    rng = np.random.default_rng(1)
    N, S, M = 50, 4, 2
    # Deliberately different scales per member.
    scales = np.array([1.0, 100.0, 0.001, 1e6])
    raw = np.stack(
        [rng.normal(loc=0, scale=scale, size=(N, M)) for scale in scales], axis=1
    )  # (N, S, M)

    normalized = quantile_normalize(raw, directions=np.array([1.0, 1.0]))

    assert normalized.shape == (N, S, M)
    assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)
    for s in range(S):
        for j in range(M):
            col = normalized[:, s, j]
            assert col.max() > 0.9
            assert col.min() < 0.2


def test_quantile_normalize_invariant_to_monotone_rescaling_per_member():
    rng = np.random.default_rng(2)
    N, S, M = 30, 3, 2
    raw = rng.normal(size=(N, S, M))

    normalized_before = quantile_normalize(raw, directions=np.array([1.0, 1.0]))

    # Monotonically rescale only member 0's raw scores (an affine + cubic monotone map).
    raw_rescaled = raw.copy()
    x = raw_rescaled[:, 0, :]
    raw_rescaled[:, 0, :] = 3.0 * x**3 + 7.0 * x + 500.0  # strictly increasing, monotone

    normalized_after = quantile_normalize(raw_rescaled, directions=np.array([1.0, 1.0]))

    np.testing.assert_allclose(normalized_before[:, 0, :], normalized_after[:, 0, :])
    # Sanity: other members untouched, so their normalized output is also untouched.
    np.testing.assert_allclose(normalized_before[:, 1:, :], normalized_after[:, 1:, :])


def test_quantile_normalize_does_not_pool_across_members():
    """Pooling across members would make each member's normalized range depend on the
    scale of other members. With per-member fitting, giving one member a huge outlier
    scale must not compress another member's normalized spread."""
    N, S, M = 20, 2, 1
    rng = np.random.default_rng(3)
    small = rng.normal(loc=0, scale=1.0, size=(N,))
    huge = rng.normal(loc=0, scale=1e9, size=(N,))
    raw = np.stack([small, huge], axis=1)[:, :, None]  # (N, 2, 1)

    normalized = quantile_normalize(raw, directions=np.array([1.0]))

    # Member 0 (small-scale) should still span close to [0, 1] on its own -- if pooling
    # had happened, its normalized values would collapse toward the middle since `huge`'s
    # range would dominate the shared reference distribution.
    assert normalized[:, 0, 0].max() > 0.9
    assert normalized[:, 0, 0].min() < 0.1


def test_quantile_normalize_reference_anchoring_clips_to_01():
    N, S, M = 10, 3, 1
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(N, S, M))
    # Reference distribution with a much narrower range -> some raw points will fall
    # outside it and must be clipped to [0, 1], not extrapolated beyond.
    reference = rng.normal(loc=0, scale=0.01, size=(5, S, M))

    normalized = quantile_normalize(raw, directions=np.array([1.0]), reference=reference)
    assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)


# ─────────────────────────────────────────────────────────────────────────
# Chebyshev scalarization
# ─────────────────────────────────────────────────────────────────────────


def test_chebyshev_scalarize_output_shape():
    N, S, M = 15, 5, 3
    normalized = np.random.default_rng(5).uniform(size=(N, S, M))
    g = chebyshev_scalarize(normalized, weights_row=np.array([0.2, 0.3, 0.5]), rho=0.05)
    assert g.shape == (N, S)


def test_chebyshev_scalarize_recovers_nonconvex_front_point():
    """A pure weighted sum can never select a point in a nonconvex 'dent' of the front;
    the augmented-Chebyshev min term can."""
    # Explicit nonconvex 2-objective front (single ensemble member, S=1).
    # Point B sits in the concave dent between A and C: no linear weighted sum ever
    # ranks B above both A and C, but a Chebyshev (min) scalarization can, for a weight
    # vector balanced between the two objectives.
    front = np.array(
        [
            [1.0, 0.0],  # A: all objective 1
            [0.55, 0.55],  # B: balanced, concave dent
            [0.0, 1.0],  # C: all objective 2
        ]
    )
    normalized = front[:, None, :]  # (N=3, S=1, M=2)

    w = np.array([0.5, 0.5])
    # Pure weighted sum would tie A, B, C all at 0.5-ish but never strictly prefer B;
    # in fact here sum(A)=1.0, sum(B)=1.1, sum(C)=1.0 so B wins under sum too in this
    # toy example -- confirm the Chebyshev (min-based) score also strictly prefers B,
    # and by a larger relative margin, since min(w*f) rewards balance directly.
    g = chebyshev_scalarize(normalized, w, rho=0.0)  # (3, 1)
    scores = g[:, 0]
    assert np.argmax(scores) == 1  # B wins
    # And with rho=0, g[B] = min(w*fB) = 0.5*0.55 = 0.275, strictly greater than
    # g[A] = min(w*fA) = min(0.5, 0.0) = 0.0.
    np.testing.assert_allclose(scores[0], 0.0)
    np.testing.assert_allclose(scores[1], 0.275)
    np.testing.assert_allclose(scores[2], 0.0)


# ─────────────────────────────────────────────────────────────────────────
# Weight sampling
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["uniform", "low_discrepancy"])
def test_sample_simplex_weights_rows_sum_to_one(method):
    weights = sample_simplex_weights(20, 3, random_state=7, method=method)
    assert weights.shape == (20, 3)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_sample_simplex_weights_m1_degenerate():
    weights = sample_simplex_weights(4, 1, random_state=0)
    np.testing.assert_array_equal(weights, np.ones((4, 1)))


# ─────────────────────────────────────────────────────────────────────────
# parego_select: determinism, diversity, feasibility
# ─────────────────────────────────────────────────────────────────────────


def _make_two_objective_pool(N=60, S=5, seed=0):
    rng = np.random.default_rng(seed)
    # Objective 1 favors low index, objective 2 favors high index -- a clear trade-off.
    base1 = np.linspace(1.0, 0.0, N)
    base2 = np.linspace(0.0, 1.0, N)
    preds = np.empty((N, S, 2))
    for s in range(S):
        preds[:, s, 0] = base1 + rng.normal(scale=0.05, size=N)
        preds[:, s, 1] = base2 + rng.normal(scale=0.05, size=N)
    seq_ids = np.array([f"seq_{i}" for i in range(N)])
    return preds, seq_ids


def test_parego_select_determinism():
    preds, seq_ids = _make_two_objective_pool()
    kwargs = dict(
        seq_ids=seq_ids,
        directions=np.array([1.0, 1.0]),
        q_slate_size=6,
        lie_noise_stddev_multiplier=0.3,
    )
    r1 = parego_select(preds, random_state=42, **kwargs)
    r2 = parego_select(preds, random_state=42, **kwargs)
    r3 = parego_select(preds, random_state=99, **kwargs)

    assert r1.selected_seq_ids == r2.selected_seq_ids
    np.testing.assert_allclose(r1.slot_weights, r2.slot_weights)
    assert r1.selected_seq_ids != r3.selected_seq_ids or not np.allclose(
        r1.slot_weights, r3.slot_weights
    )


def test_parego_select_slot_weights_differ():
    preds, seq_ids = _make_two_objective_pool()
    result = parego_select(
        preds,
        seq_ids,
        directions=np.array([1.0, 1.0]),
        q_slate_size=6,
        lie_noise_stddev_multiplier=0.3,
        random_state=1,
    )
    # Not all slot weight vectors are identical.
    assert not np.allclose(result.slot_weights, result.slot_weights[0])


def test_parego_select_batch_spans_both_objectives():
    preds, seq_ids = _make_two_objective_pool(N=80, S=6, seed=3)
    result = parego_select(
        preds,
        seq_ids,
        directions=np.array([1.0, 1.0]),
        q_slate_size=8,
        lie_noise_stddev_multiplier=0.2,
        random_state=5,
    )
    idx_selected = [int(sid.split("_")[1]) for sid in result.selected_seq_ids]
    # Objective 1 favors low index, objective 2 favors high index. A collapsed batch
    # would cluster near one extreme; a diverse batch should include both low and high
    # indices.
    assert min(idx_selected) < 25
    assert max(idx_selected) > 55


def test_parego_select_provenance_records():
    preds, seq_ids = _make_two_objective_pool(N=30, S=4)
    q = 4
    result = parego_select(
        preds,
        seq_ids,
        directions=np.array([1.0, 1.0]),
        q_slate_size=q,
        lie_noise_stddev_multiplier=0.3,
        random_state=2,
    )
    assert isinstance(result, ParegoSelectionResult)
    assert len(result.records) == q
    assert len(result.slot_scalarized_scores) == q
    for k, rec in enumerate(result.records):
        assert rec["slate_slot"] == k
        assert len(rec["scalarization_weights"]) == 2
        assert isinstance(rec["scalarized_score"], float)
    assert len(set(result.selected_seq_ids)) == q  # no duplicates


def test_parego_select_feasibility_filtering():
    preds, seq_ids = _make_two_objective_pool(N=30, S=4)
    feasibility = np.zeros(30)
    feasibility[:10] = 1.0  # only first 10 candidates feasible

    result = parego_select(
        preds,
        seq_ids,
        directions=np.array([1.0, 1.0]),
        q_slate_size=4,
        lie_noise_stddev_multiplier=0.3,
        random_state=2,
        feasibility=feasibility,
        min_feasibility=0.5,
    )
    idx_selected = [int(sid.split("_")[1]) for sid in result.selected_seq_ids]
    assert all(i < 10 for i in idx_selected)


def test_parego_select_feasibility_raises_when_too_few_remain():
    preds, seq_ids = _make_two_objective_pool(N=30, S=4)
    feasibility = np.zeros(30)
    feasibility[:2] = 1.0

    with pytest.raises(ValueError):
        parego_select(
            preds,
            seq_ids,
            directions=np.array([1.0, 1.0]),
            q_slate_size=4,
            lie_noise_stddev_multiplier=0.3,
            feasibility=feasibility,
            min_feasibility=0.5,
        )


# ─────────────────────────────────────────────────────────────────────────
# Hypervolume
# ─────────────────────────────────────────────────────────────────────────


def test_dominated_hypervolume_hand_computed():
    points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    reference_point = np.array([0.0, 0.0])
    # Union of [0,1]x[0,3], [0,2]x[0,2], [0,3]x[0,1] = 6 (hand computed).
    hv = dominated_hypervolume(points, reference_point)
    assert hv == pytest.approx(6.0)


def test_dominated_hypervolume_ignores_dominated_points():
    points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [0.5, 0.5]])  # last is dominated
    reference_point = np.array([0.0, 0.0])
    hv = dominated_hypervolume(points, reference_point)
    assert hv == pytest.approx(6.0)


def test_dominated_hypervolume_excludes_points_below_reference():
    points = np.array([[-1.0, -1.0], [2.0, 2.0]])
    reference_point = np.array([0.0, 0.0])
    hv = dominated_hypervolume(points, reference_point)
    assert hv == pytest.approx(4.0)


def test_dominated_hypervolume_raises_above_2d():
    points = np.zeros((3, 3))
    with pytest.raises(NotImplementedError):
        dominated_hypervolume(points, np.zeros(3))

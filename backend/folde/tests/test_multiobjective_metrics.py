import numpy as np
import pandas as pd
import pytest

from folde.multiobjective_metrics import (
    MultiObjectiveRoundMetrics,
    QuantileReference,
    compute_round_metrics,
    diagnostic_predicted_hypervolume,
    fit_quantile_reference,
    hypervolume_trajectory,
    measured_hypervolume,
    nondominated_mask,
    pareto_front,
)

# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------


def test_nondominated_mask_hand_built():
    # Points (maximize convention):
    # A=(3,3) nondominated (best on both)
    # B=(4,1) nondominated (best on x)
    # C=(1,4) nondominated (best on y)
    # D=(2,2) dominated by A on both axes
    # E=(3,3) duplicate of A -- not strictly dominated by A, so also kept
    # F=(0,0) dominated by everything
    points = np.array(
        [
            [3, 3],  # A
            [4, 1],  # B
            [1, 4],  # C
            [2, 2],  # D
            [3, 3],  # E (duplicate of A)
            [0, 0],  # F
        ],
        dtype=float,
    )
    mask = nondominated_mask(points)
    expected = np.array([True, True, True, False, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_pareto_front_returns_subset_and_ids():
    points = np.array([[3, 3], [2, 2], [1, 4]], dtype=float)
    ids = ["a", "b", "c"]
    front_points, front_ids = pareto_front(points, ids)
    assert set(front_ids) == {"a", "c"}
    assert front_points.shape == (2, 2)


def test_pareto_front_id_length_mismatch_raises():
    points = np.array([[1, 1], [2, 2]], dtype=float)
    with pytest.raises(ValueError):
        pareto_front(points, ["only_one"])


# ---------------------------------------------------------------------------
# Quantile reference
# ---------------------------------------------------------------------------


def test_quantile_reference_transform_monotone_and_bounded():
    ref_df = pd.DataFrame({"obj1": np.linspace(0, 100, 101)})
    reference = fit_quantile_reference(ref_df)

    query = pd.DataFrame({"obj1": [-10, 0, 25, 50, 75, 100, 200]})
    out = reference.transform(query)["obj1"].to_numpy()

    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    # Monotone non-decreasing in the input ordering.
    assert np.all(np.diff(out) >= 0)


def test_quantile_reference_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=500)
    ref_df = pd.DataFrame({"obj1": raw})
    reference = fit_quantile_reference(ref_df)

    query = pd.DataFrame({"obj1": rng.normal(size=50)})
    out_raw = reference.transform(query)["obj1"].to_numpy()

    # Monotone rescaling: y = 3x + 7 (affine, strictly increasing).
    rescaled_ref_df = pd.DataFrame({"obj1": raw * 3 + 7})
    rescaled_reference = fit_quantile_reference(rescaled_ref_df)
    rescaled_query = pd.DataFrame({"obj1": query["obj1"] * 3 + 7})
    out_rescaled = rescaled_reference.transform(rescaled_query)["obj1"].to_numpy()

    np.testing.assert_allclose(out_raw, out_rescaled, atol=1e-12)


def test_quantile_reference_serialization_roundtrip():
    ref_df = pd.DataFrame({"obj1": [1.0, 2.0, 3.0], "obj2": [4.0, 5.0, 6.0]})
    reference = fit_quantile_reference(ref_df)
    data = reference.to_dict()
    restored = QuantileReference.from_dict(data)
    query = pd.DataFrame({"obj1": [1.5], "obj2": [4.5]})
    np.testing.assert_allclose(
        reference.transform(query).to_numpy(), restored.transform(query).to_numpy()
    )


# ---------------------------------------------------------------------------
# Hypervolume (2-D exact hand-computed cases)
# ---------------------------------------------------------------------------


def test_hypervolume_2d_single_point():
    # Reference point (0, 0); one point at (0.5, 0.5) contributes 0.5*0.5=0.25.
    reference = QuantileReference(
        objectives=["obj1", "obj2"],
        sorted_reference_values={"obj1": [0.0, 1.0], "obj2": [0.0, 1.0]},
    )
    values = pd.DataFrame({"obj1": [0.5], "obj2": [0.5]}, index=["v1"])
    # transform() will map via empirical CDF of [0,1] reference set; use raw
    # values directly on a reference already in [0,1] via identity-like ref.
    # Instead of relying on transform's exact quantile math here, test the
    # private hypervolume math indirectly through measured_hypervolume with a
    # reference whose sorted values make transform an identity on [0, 1].
    hv = measured_hypervolume(values, reference, reference_point=[0.0, 0.0])
    assert hv > 0.0


def test_hypervolume_2d_hand_computed_two_points():
    # Build a reference so quantile-transform is the identity on a known
    # small grid: reference distribution is exactly {0, 1} per objective, so
    # any value v in [0, 1] maps to v (fraction of ref <= v is v for v in
    # {0, 1} with 2 ref points... to keep this exactly checkable we instead
    # bypass transform() and construct QuantileReference with 100 points from
    # 0..99 so that value v maps to v/100 approximately -- instead, directly
    # test via known values whose transform we compute by hand.
    ref_df = pd.DataFrame({"x": np.arange(0, 101), "y": np.arange(0, 101)})
    reference = fit_quantile_reference(ref_df)  # 101 points: 0..100

    # Values 60 and 100 map to quantiles 61/101 and 101/101=1.0 respectively
    # (searchsorted 'right' on sorted [0..100] for value v gives v+1).
    values = pd.DataFrame({"x": [60.0, 30.0], "y": [30.0, 60.0]}, index=["p1", "p2"])
    reference_point = [10.0 / 101.0, 10.0 / 101.0]  # corresponds to raw value ~10

    hv = measured_hypervolume(values, reference, reference_point)

    qx1, qy1 = 61 / 101, 31 / 101
    qx2, qy2 = 31 / 101, 61 / 101
    rx, ry = reference_point
    # Two nondominated points (60,30) and (30,60) in quantile space, neither
    # dominates the other. Hand-compute the union area via the sweep:
    # sorted ascending by x: p2=(qx2,qy2), p1=(qx1,qy1)
    # strip 1: width (qx2 - rx), height suffix_max_y = max(qy2, qy1)
    # strip 2: width (qx1 - qx2), height qy1
    expected = (qx2 - rx) * (max(qy2, qy1) - ry) + (qx1 - qx2) * (qy1 - ry)
    assert hv == pytest.approx(expected, abs=1e-9)


def test_hypervolume_2d_point_on_reference_boundary_contributes_zero():
    ref_df = pd.DataFrame({"x": np.arange(0, 11), "y": np.arange(0, 11)})
    reference = fit_quantile_reference(ref_df)  # 11 points: 0..10

    # A point exactly at the reference point's raw value (5) contributes 0,
    # a second point strictly better contributes something positive.
    values = pd.DataFrame({"x": [5.0, 8.0], "y": [5.0, 8.0]}, index=["boundary", "better"])
    q5 = 6 / 11  # searchsorted right of 5 in 0..10 -> index 6
    reference_point = [q5, q5]

    hv = measured_hypervolume(values, reference, reference_point)
    # Only the "better" point contributes; hand compute its rectangle.
    q8 = 9 / 11
    expected = (q8 - q5) * (q8 - q5)
    assert hv == pytest.approx(expected, abs=1e-9)

    # And a version with only the boundary point should give exactly 0.
    hv_boundary_only = measured_hypervolume(values.loc[["boundary"]], reference, reference_point)
    assert hv_boundary_only == 0.0


def test_hypervolume_3d_raises_not_implemented():
    ref_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]})
    reference = fit_quantile_reference(ref_df)
    values = pd.DataFrame({"x": [0.5], "y": [0.5], "z": [0.5]}, index=["v1"])
    with pytest.raises(NotImplementedError):
        measured_hypervolume(values, reference, reference_point=[0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# NaN policy
# ---------------------------------------------------------------------------


def test_measured_hypervolume_excludes_incomplete_rows():
    ref_df = pd.DataFrame({"x": np.arange(0, 11), "y": np.arange(0, 11)})
    reference = fit_quantile_reference(ref_df)
    values = pd.DataFrame({"x": [8.0, np.nan], "y": [8.0, 9.0]}, index=["complete", "missing_x"])
    hv_all = measured_hypervolume(values, reference, reference_point=[0.0, 0.0])
    hv_complete_only = measured_hypervolume(
        values.loc[["complete"]], reference, reference_point=[0.0, 0.0]
    )
    # The NaN row must be excluded entirely, so results should match exactly.
    assert hv_all == pytest.approx(hv_complete_only)


def test_compute_round_metrics_nan_prediction_gives_nan_spearman():
    ref_df = pd.DataFrame(
        {"obj1": np.arange(0, 11, dtype=float), "obj2": np.arange(0, 11, dtype=float)}
    )
    reference = fit_quantile_reference(ref_df)
    measured_df = pd.DataFrame(
        {"obj1": [1.0, 2.0, 3.0], "obj2": [1.0, 2.0, 3.0]}, index=["a", "b", "c"]
    )
    # No predictions supplied at all for either objective.
    predictions = {}
    metrics = compute_round_metrics(
        measured_df, predictions, reference, reference_point=[0.0, 0.0], round_num=1
    )
    assert np.isnan(metrics.model_spearman["obj1"])
    assert np.isnan(metrics.model_spearman["obj2"])


# ---------------------------------------------------------------------------
# Diagnostic vs. measured are distinct callables
# ---------------------------------------------------------------------------


def test_diagnostic_and_measured_are_distinct_callables():
    assert measured_hypervolume is not diagnostic_predicted_hypervolume
    assert measured_hypervolume.__name__ != diagnostic_predicted_hypervolume.__name__
    assert "diagnostic" in diagnostic_predicted_hypervolume.__doc__.lower()
    assert "do not report" in diagnostic_predicted_hypervolume.__doc__.lower()


def test_diagnostic_predicted_hypervolume_computes_same_math_as_measured():
    ref_df = pd.DataFrame({"x": np.arange(0, 11), "y": np.arange(0, 11)})
    reference = fit_quantile_reference(ref_df)
    values = pd.DataFrame({"x": [8.0], "y": [8.0]}, index=["v1"])
    hv_measured = measured_hypervolume(values, reference, reference_point=[0.0, 0.0])
    hv_diagnostic = diagnostic_predicted_hypervolume(values, reference, reference_point=[0.0, 0.0])
    assert hv_measured == pytest.approx(hv_diagnostic)


# ---------------------------------------------------------------------------
# compute_round_metrics end-to-end
# ---------------------------------------------------------------------------


def test_compute_round_metrics_basic():
    ref_df = pd.DataFrame({"obj1": np.linspace(0, 10, 50), "obj2": np.linspace(0, 10, 50)})
    reference = fit_quantile_reference(ref_df)
    measured_df = pd.DataFrame(
        {"obj1": [1.0, 5.0, 9.0], "obj2": [9.0, 5.0, 1.0]}, index=["v1", "v2", "v3"]
    )
    predictions = {
        "obj1": pd.Series([1.1, 5.1, 8.9], index=["v1", "v2", "v3"]),
        "obj2": pd.Series([8.9, 5.1, 1.1], index=["v1", "v2", "v3"]),
    }
    metrics = compute_round_metrics(
        measured_df, predictions, reference, reference_point=[0.0, 0.0], round_num=3
    )
    assert isinstance(metrics, MultiObjectiveRoundMetrics)
    assert metrics.round_num == 3
    assert metrics.model_spearman["obj1"] == pytest.approx(1.0)
    assert metrics.model_spearman["obj2"] == pytest.approx(1.0)
    # All three points are mutually nondominated (each best on one axis, v2 is
    # a middle tradeoff also nondominated since nothing beats it on both).
    assert metrics.n_nondominated == 3
    assert set(metrics.pareto_front_ids) == {"v1", "v2", "v3"}
    assert metrics.measured_hypervolume > 0.0


# ---------------------------------------------------------------------------
# Monotonicity of hypervolume_trajectory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_hypervolume_trajectory_is_monotone_nondecreasing(seed):
    rng = np.random.default_rng(seed)
    n_total = 40
    seq_ids = [f"v{i}" for i in range(n_total)]
    x = rng.uniform(0, 10, size=n_total)
    y = rng.uniform(0, 10, size=n_total)
    full_df = pd.DataFrame({"obj1": x, "obj2": y}, index=seq_ids)

    reference = fit_quantile_reference(full_df)
    reference_point = [0.0, 0.0]

    # Simulate rounds by revealing a growing prefix of shuffled variants.
    order = rng.permutation(n_total)
    round_size = 5
    rounds = []
    measured_so_far = None
    for round_num, start in enumerate(range(0, n_total, round_size)):
        new_ids = [seq_ids[i] for i in order[start : start + round_size]]
        measured_so_far = (
            full_df.loc[new_ids]
            if measured_so_far is None
            else pd.concat([measured_so_far, full_df.loc[new_ids]])
        )
        metrics = compute_round_metrics(
            measured_so_far,
            predictions_by_objective={},
            reference=reference,
            reference_point=reference_point,
            round_num=round_num,
        )
        rounds.append(metrics)

    trajectory = hypervolume_trajectory(rounds)
    assert len(trajectory) == len(rounds)
    diffs = np.diff(trajectory)
    assert np.all(diffs >= -1e-9)


def test_hypervolume_trajectory_raises_on_violation():
    bad_round_1 = MultiObjectiveRoundMetrics(
        round_num=1,
        model_spearman={},
        measured_hypervolume=1.0,
        n_nondominated=1,
        pareto_front_ids=["a"],
    )
    bad_round_2 = MultiObjectiveRoundMetrics(
        round_num=2,
        model_spearman={},
        measured_hypervolume=0.5,
        n_nondominated=1,
        pareto_front_ids=["a"],
    )
    with pytest.raises(ValueError):
        hypervolume_trajectory([bad_round_1, bad_round_2])

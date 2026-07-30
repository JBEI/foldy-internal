"""Analyze the KCNJ2 multi-objective campaign output.

Raw final hypervolume saturates near 1.0 (see below), so the headline numbers
compress every arm difference into the third decimal. This script reports
hypervolume *regret* against the achievable front instead, which is the standard
MOBO presentation and puts the resolution where the arms actually differ.

Run:
  cd backend && ../.venv/bin/python -u folde/scripts/260730_analyze_multiobj_kcnj2.py
"""

import json
import os

import numpy as np

os.environ.setdefault("FOLDE_CONSTANT_LIAR_DEVICE", "cpu")

from folde.multiobjective_data import (  # noqa: E402
    MULTIOBJECTIVE_DATASETS,
    load_multiobjective_dataset,
)
from folde.multiobjective_metrics import (  # noqa: E402
    fit_quantile_reference,
    measured_hypervolume,
    nondominated_mask,
)

RESULTS = "folde/model_evals/260730-multiobj-kcnj2.json"


def main():
    with open(RESULTS) as fh:
        blob = json.load(fh)

    ds = load_multiobjective_dataset(MULTIOBJECTIVE_DATASETS[blob["dataset"]])
    reference = fit_quantile_reference(ds.activity_df)
    ref_point = tuple(0.0 for _ in blob["objectives"])

    # Achievable ceiling: hypervolume if every variant in the pool were measured.
    hv_star = measured_hypervolume(ds.activity_df, reference, ref_point)
    n_front = int(nondominated_mask(ds.activity_df.to_numpy()).sum())
    print(f"dataset={blob['dataset']}  N={len(ds.activity_df)}  objectives={blob['objectives']}")
    print(f"achievable HV* = {hv_star:.6f}   true Pareto front size = {n_front}\n")

    arms = list(blob["results"])
    print(f"{'arm':14s} {'final HV':>18s} {'regret (HV*-HV)':>22s} {'pareto':>12s}")
    print("-" * 70)
    summary = {}
    for arm in arms:
        finals = np.array([t[-1]["hypervolume"] for t in blob["results"][arm]])
        pareto = np.array([t[-1]["n_pareto"] for t in blob["results"][arm]])
        regret = hv_star - finals
        summary[arm] = {"finals": finals, "regret": regret, "pareto": pareto}
        print(
            f"{arm:14s} {finals.mean():.4f} +/- {finals.std():.4f} "
            f"{regret.mean():>13.5f} +/- {regret.std():.5f} "
            f"{pareto.mean():>6.1f} +/- {pareto.std():.1f}"
        )

    # Per-sim detail: with n=3 the spread matters more than the mean.
    print("\nper-simulation regret (lower is better):")
    for arm in arms:
        vals = "  ".join(f"{r:.5f}" for r in summary[arm]["regret"])
        print(f"  {arm:14s} {vals}")

    print("\nper-round mean regret trajectory:")
    n_rounds = len(blob["results"][arms[0]][0])
    header = "  ".join(f"r{i+1:<6d}" for i in range(n_rounds))
    print(f"  {'arm':14s} {header}")
    for arm in arms:
        traj = np.array([[r["hypervolume"] for r in sim] for sim in blob["results"][arm]])
        mean_regret = hv_star - traj.mean(axis=0)
        row = "  ".join(f"{v:.5f}" for v in mean_regret)
        print(f"  {arm:14s} {row}")

    # Overlap check: does any baseline sim beat the best parego sim?
    print("\noverlap against parego (a single baseline sim beating parego's worst")
    print("means n=3 cannot separate the arms, regardless of the means):")
    p_best, p_worst = summary["parego"]["regret"].min(), summary["parego"]["regret"].max()
    print(f"  parego regret range: [{p_best:.5f}, {p_worst:.5f}]")
    for arm in arms:
        if arm == "parego":
            continue
        beats = int((summary[arm]["regret"] < p_worst).sum())
        print(f"  {arm:14s} sims beating parego's worst: {beats}/{len(summary[arm]['regret'])}")


if __name__ == "__main__":
    main()

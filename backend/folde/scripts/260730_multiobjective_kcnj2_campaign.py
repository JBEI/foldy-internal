"""Simulated multi-objective FolDE campaign on KCNJ2 (function x surface).

Compares ParEGO acquisition against baselines on the spec's Tier-1 primary metric:
cumulative measured hypervolume, quantile-normalized against a shared reference
distribution (the full measured pool) with a pinned reference point.

Arms:
  parego        - randomized augmented-Chebyshev scalarization, weights vary per slot
  fixed_weight  - Chebyshev with a single fixed 50/50 weight for every slot
  single_obj    - greedy constant-liar on objective 0 only (ignores objective 1)
  random        - uniform random selection from the unmeasured pool

All arms select from the identical candidate pool each round: the baseline arms go
through `parego_select` at M=1 rather than `folde.util.constant_liar_sample`, whose
5000-candidate cap would otherwise hand the parego arm a larger search space. See
`_select_scalar`.

Run (note `-u`: stdout is block-buffered when piped, so without it a `tee`d log
stays empty for the whole run):
  cd backend && FOLDE_CONSTANT_LIAR_DEVICE=cpu \\
    ../.venv/bin/python -u folde/scripts/260730_multiobjective_kcnj2_campaign.py
"""

import argparse
import json
import logging
import os
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("FOLDE_CONSTANT_LIAR_DEVICE", "cpu")

from folde.few_shot_models import TorchMLPFewShotModel  # noqa: E402
from folde.multiobjective_data import (  # noqa: E402
    MULTIOBJECTIVE_DATASETS,
    load_multiobjective_dataset,
)
from folde.multiobjective_metrics import (  # noqa: E402
    fit_quantile_reference,
    measured_hypervolume,
    nondominated_mask,
)
from folde.parego import chebyshev_scalarize, parego_select, quantile_normalize  # noqa: E402

# Matches the tuned single-objective config in folde/model_evals/, except device=cpu
# (this host's RTX 5080 is sm_120 and the installed torch has no kernels for it).
RANKER_PARAMS: dict[str, Any] = {
    "hidden_dims": [100, 50],
    "dropout": 0.2,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "ensemble_size": 8,
    "pretrain": True,
    "pretrain_epochs": 10,
    "train_epochs": 200,
    "train_patience": 40,
    "val_frequency": 10,
    "do_validation_with_pair_fraction": 0.2,
    "device": "cpu",
}


def train_objective_rankers(ds, measured_ids, objectives, seed):
    """Fit one independent ensemble per objective (spec 5.3)."""
    rankers = []
    for oi, obj in enumerate(objectives):
        model = TorchMLPFewShotModel(
            wt_aa_seq=ds.wt_sequence, random_state=seed * 100 + oi, **RANKER_PARAMS
        )
        model.pretrain(ds.naturalness_df, ds.embedding_series)
        activity = ds.activity_df.loc[measured_ids, obj].dropna()
        model.fit(ds.naturalness_df, ds.embedding_series, activity)
        rankers.append(model)
    return rankers


def predict_ensemble(rankers, ds, candidate_ids):
    """Return (N, S, M) ensemble predictions over candidate_ids."""
    nat = ds.naturalness_df.loc[candidate_ids]
    emb = ds.embedding_series.loc[candidate_ids]
    per_obj = []
    for model in rankers:
        series_list = model.predict(nat, emb)
        per_obj.append(np.stack([s.to_numpy() for s in series_list], axis=1))  # (N, S)
    return np.stack(per_obj, axis=2)  # (N, S, M)


def select_batch(arm, preds, candidate_ids, q, rng, round_num, lie_schedule):
    """Dispatch to the arm's acquisition rule. Returns list of seq_ids."""
    lie = lie_schedule[min(round_num, len(lie_schedule) - 1)]
    directions = np.ones(preds.shape[2])

    if arm == "random":
        return list(rng.choice(candidate_ids, size=q, replace=False))

    if arm == "parego":
        res = parego_select(
            ensemble_predictions=preds,
            seq_ids=np.asarray(candidate_ids),
            directions=directions,
            q_slate_size=q,
            lie_noise_stddev_multiplier=lie,
            ucb_beta=2.0,
            random_state=int(rng.integers(1 << 30)),
        )
        return list(res.selected_seq_ids)

    if arm == "fixed_weight":
        # Same machinery, but one fixed weight vector for every slot: the
        # "mandatory fixed weighted sum" that spec criterion 7 rejects.
        norm = quantile_normalize(preds, directions)
        w = np.full(preds.shape[2], 1.0 / preds.shape[2])
        scal = chebyshev_scalarize(norm, w, rho=0.05)  # (N, S)
        return _select_scalar(scal, candidate_ids, q, lie, rng)

    if arm == "single_obj":
        # Greedy on objective 0 only; objective 1 is never consulted.
        return _select_scalar(preds[:, :, 0], candidate_ids, q, lie, rng)

    raise ValueError(f"unknown arm {arm}")


def _select_scalar(scalar_preds, candidate_ids, q, lie, rng):
    """Run constant-liar selection on an (N, S) scalar matrix, via parego_select at M=1.

    Deliberately NOT `folde.util.constant_liar_sample`: that builds a dense (N, N)
    covariance and so hard-errors above MAX_POINTS_TO_CONSIDER=5000 (util.py:171),
    which would force these arms to prefilter to 5000 candidates while the parego
    arm searched the full ~6.7k pool. Unequal pools make the cross-arm comparison
    invalid, and prefiltering by mean score preferentially drops exactly the
    low-mean/high-variance candidates a UCB rule might pick.

    `parego_select` with M=1 is provably identical to `constant_liar_sample`
    (test_parego.py::test_m1_equivalence_with_constant_liar_sample) but runs on the
    low-rank covariance, so it has no such cap. Every arm now sees the same pool.
    """
    return list(
        parego_select(
            ensemble_predictions=scalar_preds[:, :, None],  # (N, S, 1)
            seq_ids=np.asarray(candidate_ids),
            directions=np.ones(1),
            q_slate_size=q,
            lie_noise_stddev_multiplier=lie,
            ucb_beta=2.0,
            random_state=int(rng.integers(1 << 30)),
        ).selected_seq_ids
    )


def run_campaign(ds, arm, seed, n_rounds, batch_size, init_size, reference, ref_point, sim=0):
    objectives = list(ds.activity_df.columns)
    rng = np.random.default_rng(seed)
    all_ids = ds.activity_df.index.to_numpy()
    measured = list(rng.choice(all_ids, size=init_size, replace=False))
    lie_schedule = [3.0, 3.0, 30.0, 30.0, 100.0]

    traj = []
    for rnd in range(n_rounds):
        candidates = np.array([i for i in all_ids if i not in set(measured)])
        if arm == "random":
            picked = list(rng.choice(candidates, size=batch_size, replace=False))
        else:
            rankers = train_objective_rankers(ds, measured, objectives, seed + rnd)
            preds = predict_ensemble(rankers, ds, candidates)
            picked = select_batch(arm, preds, candidates, batch_size, rng, rnd, lie_schedule)
        measured.extend(picked)
        hv = measured_hypervolume(ds.activity_df.loc[measured], reference, ref_point)
        nd = int(nondominated_mask(ds.activity_df.loc[measured].to_numpy()).sum())
        traj.append({"round": rnd, "n_measured": len(measured), "hypervolume": hv, "n_pareto": nd})
        # Per-round heartbeat: without this the only output is one line per finished
        # simulation, which makes a healthy multi-hour run indistinguishable from a hang.
        print(
            f"    [{arm} sim{sim}] round {rnd+1}/{n_rounds} pool={len(candidates)} "
            f"n={len(measured)} hv={hv:.4f} pareto={nd}",
            flush=True,
        )
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="KCNJ2")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--init-size", type=int, default=48)
    ap.add_argument("--sims", type=int, default=3)
    ap.add_argument("--arms", default="parego,fixed_weight,single_obj,random")
    ap.add_argument("--out", default="folde/model_evals/260730-multiobj-kcnj2.json")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    ds = load_multiobjective_dataset(MULTIOBJECTIVE_DATASETS[args.dataset])
    objectives = list(ds.activity_df.columns)
    # Shared reference statistics so every arm's hypervolume is on one scale (spec 10.3 item 3).
    reference = fit_quantile_reference(ds.activity_df)
    ref_point = tuple(0.0 for _ in objectives)
    print(f"{args.dataset}: N={len(ds.activity_df)} objectives={objectives}", flush=True)

    results = {}
    for arm in args.arms.split(","):
        arm_traj = []
        for sim in range(args.sims):
            t0 = time.time()
            traj = run_campaign(
                ds,
                arm,
                1000 + sim,
                args.rounds,
                args.batch_size,
                args.init_size,
                reference,
                ref_point,
                sim=sim,
            )
            arm_traj.append(traj)
            print(
                f"  {arm:13s} sim{sim} final_hv={traj[-1]['hypervolume']:.4f} "
                f"pareto={traj[-1]['n_pareto']:3d} ({time.time()-t0:.0f}s)",
                flush=True,
            )
        results[arm] = arm_traj
        finals = [t[-1]["hypervolume"] for t in arm_traj]
        print(
            f"  {arm:13s} MEAN final_hv={np.mean(finals):.4f} +/- {np.std(finals):.4f}\n",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {
                "dataset": args.dataset,
                "objectives": objectives,
                "config": {k: v for k, v in vars(args).items()},
                "ranker_params": RANKER_PARAMS,
                "results": results,
            },
            fh,
            indent=1,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

# Pre-registration: low-N multi-objective acquisition study (Tier 1)

Spec: `2026-07-21-moce-folde-hybrid-spec.md`, Sections 10.3 and 11.10.

## Provenance of this document

The analysis plan below was written into the launch script
`sweep_lowN2.sh` **before** the KCNE1 and OXDA runs produced any output, and
transcribed here unchanged. It was **not** written before KCNJ2, PTEN, or S22A1,
which were run first as a pilot. This is stated plainly because Section 11.10
requires the baseline be declared in advance, and a pre-registration that
misrepresents when it was written is worse than none.

Concretely:

| dataset | run at | covered by this pre-registration? |
|---|---|---|
| KCNJ2, PTEN, S22A1 | 2026-07-30 16:36-18:19 | No -- pilot, results seen first |
| KCNE1, OXDA | 2026-07-30 18:53-19:53 | Yes -- declared before launch |
| RASK | 2026-07-30 19:54- | Yes |

The pooled n=5 and n=6 tests therefore mix pre-registered and post-hoc
observations. Treat the pooled result as **exploratory**, not confirmatory. A
clean confirmatory run would re-run all datasets under this plan.

## Analysis plan

- **Primary metric.** Measured-value hypervolume regret `HV* - HV` after round 5,
  where each objective is quantile-normalized against the full DMS activity
  distribution for that dataset and the reference point is pinned at `(0, 0)`.
  This is Section 10.3(2) (measured values only, never predicted scores) and
  10.3(3) (shared normalization statistics across arms).
- **Comparison.** `parego` vs `fixed_weight`.
- **Pre-declared strongest baseline: `fixed_weight`.** Declared because it is the
  acceptance-criterion-7 foil -- identical machinery, single `[0.5, 0.5]` weight
  vector for every slate slot -- and because it had already *beaten* `parego` on
  PTEN in the pilot. Declaring the arm that has outperformed you is the only
  version of this that carries information.
- **Test.** Two-sided Wilcoxon signed-rank on per-dataset paired differences, one
  observation per dataset (Section 11.10), alpha = 0.05. Report exact p-value,
  median paired difference, and 95% bootstrap CI on that difference.
- **Secondary arms.** `single_obj`, `random`. Exploratory; Benjamini-Hochberg at
  FDR = 0.10 if p-values are computed at all.

## Power caveat, recorded in advance

A two-sided signed-rank test on `n` pairs cannot produce `p < 2/2^n`.

| n datasets | floor | alpha=0.05 reachable? |
|---|---|---|
| 3 | 0.250 | no |
| 5 | 0.0625 | no |
| 6 | 0.03125 | yes |

At n=5 the study **cannot** satisfy acceptance criterion 10 regardless of effect
size. This is why Section 10.10's dataset floor of 5 is wrong and must rise to
>= 6. The five-dataset run was executed to complete the paired set and test the
hypothesis below, not to claim significance.

## Hypothesis under test

`parego`'s advantage over `fixed_weight` scales with the size of the true Pareto
front. Pilot data: `parego` ahead on KCNJ2 (front 11) and S22A1 (front 7), behind
on PTEN (front 6).

Predictions, stated before the runs:

| dataset | front | rho | prediction |
|---|---|---|---|
| OXDA | 5 | +0.232 | `fixed_weight` >= `parego` (replicates PTEN) |
| KCNE1 | 10 | +0.060 | `parego` > `fixed_weight` (replicates KCNJ2) |

OXDA is the falsifier: `parego` winning there kills the hypothesis.

## Campaign configuration (identical across all datasets)

    --rounds 5  --batch-size 16  --init-size 16  --sims 10

`init-size` is one round's worth rather than the script default of 48: a
48-variant free random seed would be 3x a round and would dominate the budget it
exists to bootstrap. Total measured per simulation: 16 + 5x16 = 96.

Ranker: `TorchMLPFewShotModel`, ensemble of 8 per objective, Bradley-Terry loss,
pretrained on E1-600m naturalness and fine-tuned on measured activity, over
ESMC-300M (960-d) embeddings. Device forced to CPU (this host's RTX 5080 is
sm_120 and the installed torch ships no kernels for it).

Arms:

- `parego` -- randomized augmented-Chebyshev, `rho=0.05`, one weight vector per slot
- `fixed_weight` -- same machinery, `w = [0.5, 0.5]` for every slot
- `single_obj` -- greedy constant-liar on objective 0 only
- `random` -- uniform from the unmeasured pool

All arms select from the identical candidate pool via `parego_select` at M=1
rather than `folde.util.constant_liar_sample`, whose 5000-candidate cap would
otherwise hand `parego` a larger search space. Prefiltering by predicted mean was
rejected: it is an exploitation filter upstream of the acquisition function and
would preferentially delete the low-mean/high-variance candidates a UCB rule
exists to find (Section 6.6.6 forbids top-K reduction for this reason).

## Known deviations from Section 11.10

1. Arm 2 of Section 10.1 (naturalness-only selection) is not implemented.
2. The runs use a standalone script rather than `folde/campaign.py`'s
   `simulate_campaign`, which Section 10.1 Tier 1 names and requires unmodified.
3. Sections 6.6.5 (isotonic calibration for interval/constraint objectives) and
   6.6.6 (redundancy penalty) are not exercised; all objectives are directional
   maximize and `redundancy_penalty_fn` is passed as `None`.
4. Section 6.6.1's `direction = -1` branch is never taken.
5. Acceptance criteria 6, 8, and 9 (per-candidate uncertainty and
   distance-from-measured records, full traceability, Section 7.6 manifest) are
   out of scope; they are Section 15.5 milestones 1 and 7.
6. `parego.py:447` clamps variance at zero rather than at the nugget floor, which
   Section 6.6.4 requires. Unresolved at the time of these runs.

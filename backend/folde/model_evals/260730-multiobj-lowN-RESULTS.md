# Results: low-N multi-objective acquisition study (Tier 1, n=6 datasets)

Analysis plan: `260730-multiobj-lowN-PREREGISTRATION.md`. That document is
deliberately **not** amended with these results -- a pre-registration edited after
the outcome is known no longer does the job it exists to do.

## Per-dataset mean hypervolume regret (10 sims each, lower is better)

| dataset | N | front | rho | parego | fixed_weight | single_obj | random | winner |
|---|---|---|---|---|---|---|---|---|
| KCNJ2 | 6789 | 11 | +0.307 | **0.01869** | 0.02097 | 0.02728 | 0.04656 | parego |
| PTEN | 4839 | 6 | +0.373 | 0.03136 | **0.02555** | 0.03421 | 0.04929 | fixed_weight |
| S22A1 | 9715 | 7 | +0.776 | **0.00612** | 0.00957 | 0.00963 | 0.02572 | parego |
| KCNE1 | 2312 | 10 | +0.060 | **0.02452** | 0.04000 | 0.03355 | 0.04908 | parego |
| OXDA | 6387 | 5 | +0.232 | **0.01982** | 0.02257 | 0.02034 | 0.03698 | parego |
| RASK | 23072 | 8 | +0.351 | **0.00152** | 0.00233 | 0.00622 | 0.01825 | parego |

## Primary test (Section 11.10)

Two-sided Wilcoxon signed-rank on per-dataset paired differences, n=6.
Floor is `2/2^6 = 0.03125`, so alpha=0.05 is reachable.

| arm | median diff | wins | p | 95% bootstrap CI | verdict |
|---|---|---|---|---|---|
| **fixed_weight** (pre-declared primary) | +0.00252 | 5/6 | **0.3125** | [-0.00250, +0.00947] | **not significant** |
| single_obj | +0.00410 | 6/6 | 0.0312 | [+0.00168, +0.00881] | significant |
| random | +0.01877 | 6/6 | 0.0312 | [+0.01695, +0.02621] | significant |

Benjamini-Hochberg at FDR=0.10 over the two exploratory arms: both pass.

## Conclusions

**Acceptance criterion 10 is NOT met on the pre-registered primary metric.**
ParEGO vs fixed-weight Chebyshev is p=0.3125 with a confidence interval that
spans zero.

Established:

- ParEGO beats random selection. 6/6 datasets, CI well clear of zero.
- ParEGO beats single-objective greedy. 6/6, effect ~4.5x smaller than vs random.
- The Section 6.6.4 factored low-rank covariance scales. RASK (N=23,072, 4.6x the
  `MAX_POINTS_TO_CONSIDER=5000` cap) completed in 45 minutes. The dense path would
  have required a 23,072^2 covariance rebuilt 16 times per round.

Not established:

- **Acceptance criterion 7.** The criterion asserts a fixed weight vector cannot
  span the nondominated set and that randomized Chebyshev can. After six datasets
  and 240 simulations, measured hypervolume cannot distinguish them. Note KCNE1,
  where `fixed_weight` recovered MORE Pareto points (6.5 vs 5.6) while losing
  badly on hypervolume: if criterion 7 is a claim about front coverage, then
  hypervolume is the wrong instrument and a coverage or spacing metric is needed
  to test it at all. Either the criterion needs a different metric, or it
  overstates the case.

Refuted:

- **The front-size hypothesis** (that ParEGO's advantage scales with true Pareto
  front size), by its own pre-registered falsification test. It predicted
  `fixed_weight >= parego` on OXDA (front 5, the smallest); parego won. Front
  sizes order 11, 6, 7, 10, 5, 8 against winners parego, fixed_weight, parego,
  parego, parego, parego. PTEN remains a single unexplained reversal.

## Why the primary test fails, and what would change it

The failure is not simple underpowering. `single_obj` has a median difference of
+0.0041, only 1.6x larger than `fixed_weight`'s +0.0025, and it reaches
significance. The design resolves effects of that magnitude.

The obstacle is **consistency, not effect size**. `single_obj` and `random` are
swept 6/6; `fixed_weight` is 5/6 because PTEN reverses. A single reversal caps the
signed-rank statistic at p=0.3125 at n=6 regardless of the other five margins.

So more datasets tighten the CI but do not obviously fix this: if the true effect
is "usually slightly better, occasionally worse," a sign-based test will keep
penalizing the reversals. Two things would be more informative than more data:

1. A primary metric that actually measures what criterion 7 claims -- front
   coverage or spacing, not hypervolume.
2. An explanation for PTEN. The one hypothesis tested (front size) is refuted.

## Caveats

- The pooled test mixes pre-registered (KCNE1, OXDA, RASK) and post-hoc (KCNJ2,
  PTEN, S22A1) observations, so it is exploratory rather than confirmatory. See
  the pre-registration's provenance section.
- Six known deviations from Section 11.10 are listed in the pre-registration,
  including the missing naturalness-only arm (Section 10.1 arm 2), use of a
  standalone script rather than `folde/campaign.py::simulate_campaign`, and the
  unresolved variance clamp at `parego.py:447`.
- RASK's regret is ~10x tighter than the other datasets (0.0015 vs 0.02-0.03)
  because 96 measurements against 23,072 candidates leaves every arm near
  HV*=0.999983. Its paired observation carries little dynamic range.

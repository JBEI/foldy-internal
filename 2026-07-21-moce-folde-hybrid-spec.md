# MOCE-FolDE Hybrid Protein Engineering Specification

## 1. Purpose

This document specifies a reusable active-learning system for protein engineering campaigns. The system combines:

- a MOCE-style autoregressive sequence generator for controlled proposal of novel protein variants; and
- a FolDE-style few-shot ranking system using protein language model embeddings, pairwise preference learning, ensemble uncertainty, and diversity-aware batch selection.

The hybrid is intended for new protein engineering campaigns in which experimental capacity is limited, one or more properties must be optimized, and useful variants may require substitutions or higher-order mutation combinations that were not explicitly enumerated in advance.

## 2. Goals

The system shall:

1. Generate novel protein sequences from experimentally characterized seed sequences.
2. Support an arbitrary number of campaign objectives and constraints.
3. Express desired changes using ordered directional control tokens such as `<inc>`, `<dec>`, and optionally `<maintain>`.
4. Rank generated candidates using protein language model embeddings and FolDE-style MLP ensembles.
5. Estimate predictive uncertainty from ensemble disagreement and distance from measured data.
6. Select diverse experimental batches that balance exploitation and exploration.
7. Learn from all accepted measurements after every design-build-test-learn round.
8. Track novelty at the sequence, mutation-combination, substitution, and residue-position levels.
9. Prevent invalid, duplicate, prohibited, or excessively distant variants from entering the experimental queue.
10. Preserve complete provenance so every proposal and model decision can be reproduced.

## 3. Non-goals for Version 1

Version 1 does not attempt to:

- predict protein structure as part of the core ranking loop;
- generate insertions, deletions, domain rearrangements, or variable-length proteins;
- return calibrated physical measurements from the ranking model for objectives that are purely directional. A restricted monotone calibration layer is provided in Version 1, but only for objectives used as interval targets or numeric constraints, and only within the measured range it was fitted on (Section 6.6.5);
- replace assay-specific quality control;
- automatically infer whether objectives should be maximized, minimized, maintained, or constrained; or
- guarantee that natural or high-likelihood sequences are functional for the campaign assay.

Structure, docking, stability, manufacturability, and safety models may be added later as filters or additional objectives through the same scoring interface.

## 4. System Architecture

```text
Accepted measurements
        |
        +---- Pair construction ----> MOCE generator training
        |                                   |
        |                                   v
        |                          Generated sequences
        |                                   |
        |                          Validation and filters
        |                                   |
        |                                   v
        +---- Objective labels ----> PLM embeddings and naturalness
                                            |
                                            v
                                  FolDE ranking ensembles
                                            |
                                            v
                                Multi-objective acquisition
                                            |
                                            v
                                 Experimental batch manifest
                                            |
                                            +----> New measurements
```

The generator, embedding service, objective rankers, acquisition layer, and campaign data store shall be separate modules connected through versioned interfaces. Model implementations must be replaceable without changing the campaign data model.

## 5. Core Components

### 5.1 Controlled sequence generator

The generator accepts a seed sequence and an ordered control vector and autoregressively returns complete candidate sequences.

Reference implementation:

| Property | Reference default |
|---|---|
| Architecture | Encoder-decoder protein transformer |
| Foundation checkpoint | Configurable; the original MOCE implementation uses `Rostlab/prot_t5_xl_uniref50` |
| Adaptation method | LoRA |
| LoRA rank | 16 |
| LoRA scaling | 32 |
| LoRA dropout | 0.05 |
| Generation | Top-k sampling |
| Top-k | 10 |
| Temperature | 1.0, configurable |
| Initial samples per seed | 20 |
| Output | Full amino-acid sequence |

An example prompt for three ordered objectives is:

```text
<inc> <dec> <maintain> M K T ...
```

The meaning and order of tokens must be declared in campaign configuration and stored with every generated sequence.

The `<maintain>` token shall not be used unless the generator's training data contains equivalence pairs supervising it. Section 5.4 excludes sub-threshold pairs from training by default, which means a campaign that declares `goal: maintain` without explicitly enabling equivalence pairs would prompt the generator with a token it has never been trained on, producing undefined behavior. Configuration validation shall reject that combination.

```yaml
objectives:
  - name: objective_1
    token_position: 0
    goal: maximize
  - name: objective_2
    token_position: 1
    goal: minimize
  - name: objective_3
    token_position: 2
    goal: maintain
    target_interval: [0.8, 1.2]
```

The generator shall not be restricted to substitutions already observed experimentally. Restrictions on positions, residues, mutation count, or sequence distance belong in the validation layer rather than in the decoder unless a future constrained-decoding implementation is explicitly enabled.

### 5.2 Embedding and naturalness service

The embedding service converts every measured and generated sequence into a fixed-dimensional representation used by the rankers.

Reference defaults:

| Property | Reference default |
|---|---|
| Embedding model | ESMC-300M |
| Embedding dimension | 960 |
| Naturalness model | Configurable ESMC or ESM-2 checkpoint |
| Inference precision | FP16 or BF16 where supported |
| Stored embedding precision | FP32 unless storage constraints require otherwise |
| Cache key | Model name, exact revision, pooling method, and sequence hash |

Each embedding record shall contain:

- the exact model and revision;
- pooling method;
- sequence hash;
- embedding vector;
- naturalness or sequence log-likelihood score;
- computation timestamp; and
- preprocessing version.

Embedding and naturalness checkpoints must be pinned for the duration of a campaign. Changing either checkpoint creates a new feature space and requires regeneration of all campaign features and retraining of all rankers.

### 5.3 Few-shot objective rankers

Version 1 shall train an independent FolDE-style ensemble for each objective. Independent ensembles simplify missing-label handling and allow objectives to use different assay thresholds or training schedules.

Each ensemble member uses the following reference architecture:

```text
PLM embedding
    |
Linear embedding_dim -> 100
BatchNorm + ReLU + Dropout(0.2)
    |
Linear 100 -> 50
BatchNorm + ReLU + Dropout(0.2)
    |
Linear 50 -> 1
```

Reference training configuration:

| Property | Default |
|---|---|
| Ensemble size | 8 (minimum 3; the constant-liar selector requires at least 3 members to estimate variance) |
| Objective | Relative activity or property ranking |
| Loss | Bradley-Terry pairwise ranking loss |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Weight decay | `1e-5` |
| Maximum epochs | 200 |
| Early-stopping patience | 40 |
| Validation frequency | 10 epochs |
| Validation pairs | 20% nontrivial pair holdout |
| Warm start | Optional naturalness-ranking pretraining |

For objective `j`, the ensemble returns:

```text
mean_score[j] = mean of ensemble predictions
model_uncertainty[j] = standard deviation of ensemble predictions
```

Scores are relative ranking values on an arbitrary latent scale. The scale is not identified: the reference implementation uses bias-free linear layers, so a constant shift cancels in every pairwise difference and the zero point carries no meaning. The scale also changes on every retraining.

Two consequences are binding on the rest of this specification:

1. Raw scores shall not be compared across objectives, and shall not be used in any scale-sensitive computation such as hypervolume. Cross-objective comparison requires the quantile normalization of Section 6.6.2.
2. Raw scores shall not be presented as assay measurements. Objectives that require a value in assay units — interval targets and numeric constraints — shall obtain it through the restricted isotonic calibration layer of Section 6.6.5, whose output shall always be labeled as calibrated and flagged when extrapolating beyond the fitted range.

The reference implementation assigns ensemble member `i` the naturalness column at index `i mod n_columns` during pretraining. Where the naturalness source supplies several model columns, this is a substantial source of ensemble diversity and `ensemble_size` interacts with the column count. Where it supplies only one column — the common case for a single-checkpoint naturalness file — every member pretrains against an identical target and diversity reduces to random initialization and training stochasticity alone.

This matters because the acquisition function derives its entire uncertainty signal from ensemble spread. Campaigns shall record the naturalness column count alongside `ensemble_size`, and shall treat a single-column configuration as a known reduction in the informativeness of `model_uncertainty` rather than assuming the ensemble is diverse by construction.

The pretraining naturalness checkpoint may differ from the checkpoint used as an acquisition feature, and both shall be pinned independently.

### 5.4 Pair construction

Both the generator and rankers learn from relationships between experimentally measured variants. Pair construction must account for experimental noise.

For two measured variants `A` and `B`, objective `j` contributes a directional label only when:

```text
abs(value[j, A] - value[j, B]) >= pair_threshold[j]
```

The label is `<inc>` when `B` is significantly greater than `A` and `<dec>` when it is significantly lower. A `<maintain>` label may be used only when the assay and generator training procedure explicitly support equivalence pairs.

Thresholds may be fixed from assay validation or estimated from replicate variance. The threshold value and method must be versioned with the round.

A single fixed `pair_threshold[j]` applied to aggregated values is only correct when every variant carries a comparable number of replicates. Where replicate counts vary, a variant measured twice and a variant measured eight times have materially different standard errors, and a fixed absolute cutoff will admit unreliable orderings from the sparsely measured variant while discarding reliable ones from the densely measured variant. Implementations shall therefore support a per-pair significance criterion — a Welch-style comparison using each variant's own replicate variance — with the fixed absolute threshold retained as a fallback for assays lacking replicate structure. The criterion in use shall be recorded per round.

```yaml
objectives:
  objective_1:
    goal: maximize
    pair_threshold: 0.25
  objective_2:
    goal: minimize
    pair_threshold: 0.10
```

Pairs below the threshold shall be excluded from Bradley-Terry loss rather than assigned an arbitrary order.

Two consequences shall be tracked rather than ignored. First, excluding sub-threshold pairs biases training toward large-effect comparisons, and the model never receives evidence that two variants are equivalent; this is the same gap that constrains the `<maintain>` token in Section 5.1. Second, the number of admissible pairs grows quadratically in the number of measured variants, so the effective training set size changes from round to round as a joint function of data volume and threshold choice. The pair count, the threshold, and the pair-construction criterion shall each be recorded as round metrics so that changes in model behavior can be attributed to one rather than confounded across all three. Implementations shall support pair subsampling with importance reweighting when the admissible pair count exceeds a configured bound.

## 6. Campaign Workflow

### 6.1 Campaign initialization

Before round 1, the operator provides:

- a wild-type or reference sequence;
- an initial measured variant set, if available;
- objective definitions and directions;
- assay units and quality-control rules;
- experimental-noise thresholds;
- allowed and prohibited mutation rules;
- the maximum experimental batch size; and
- pinned model checkpoints and software versions.

If no activity measurements exist, the first batch may be chosen from a systematic single-mutant library using naturalness-based zero-shot ranking and diversity selection. MOCE fine-tuning begins after sufficient directional pairs are available.

### 6.2 Seed selection

Seeds shall be selected from accepted measured variants using a configurable mixture of:

1. current Pareto-optimal variants;
2. diverse near-Pareto variants;
3. variants with high model uncertainty;
4. historically informative anchors; and
5. the reference sequence.

Suggested allocation after sufficient data exist:

| Seed category | Fraction |
|---|---:|
| Pareto-optimal | 60% |
| Diverse near-Pareto | 20% |
| High uncertainty | 10% |
| Reference or historical anchors | 10% |

Fractions are starting values, not universal requirements. Every selected seed must have accepted assay data for the objectives used in its prompt.

### 6.3 Candidate generation

For each seed and control vector, the generator shall:

1. create the ordered prompt;
2. sample the configured number of complete sequences;
3. retain the sampling seed, temperature, top-k value, checkpoint, and parent sequence;
4. convert valid outputs into mutation notation relative to the campaign reference; and
5. send outputs to the validation layer before embedding or ranking.

Campaigns may allocate most generation to the desired optimization direction while reserving a smaller fraction for alternative directions to maintain model coverage.

### 6.4 Candidate validation and filtering

Version 1 shall reject sequences that:

- contain symbols outside the configured amino-acid alphabet;
- differ in length from the reference sequence;
- contain insertions or deletions;
- duplicate a measured, generated, queued, or selected sequence;
- violate allowed-position or prohibited-position rules;
- violate residue-level constraints;
- fall outside configured mutation-count or edit-distance limits;
- fail required motif constraints; or
- fall below a configured naturalness floor.

Example configuration:

```yaml
candidate_filters:
  alphabet: ACDEFGHIKLMNPQRSTVWY
  substitutions_only: true
  min_mutations: 1
  max_mutations: 6
  allow_unseen_positions: true
  allow_unseen_substitutions: true
  prohibited_positions: []
  minimum_naturalness_percentile: 1
```

Naturalness should normally be a permissive validity filter or acquisition feature, not a replacement for experimental fitness.

### 6.5 Candidate embedding and scoring

Validated candidates shall be deduplicated by sequence hash, embedded in batches, and scored by every applicable objective ensemble.

Each scored candidate shall include:

- mean ranking score per objective;
- ensemble standard deviation per objective;
- naturalness;
- distance to the nearest measured embedding;
- edit distance to the reference and parent;
- generator probability when available; and
- novelty classifications.

### 6.6 Multi-objective acquisition

Version 1 uses ParEGO: randomized augmented-Chebyshev scalarization of quantile-normalized ensemble predictions, selected in batch by the existing constant-liar acquisition. This choice is deliberate. Nondominated sorting with hypervolume contribution requires objective values on a common, scale-meaningful axis with a pinned reference point; Bradley-Terry scores satisfy neither requirement (Section 5.3). ParEGO requires only that objectives be made ordinally comparable, which quantile normalization provides, and it reduces the batch selection problem to a sequence of scalar problems that the existing selector already solves.

#### 6.6.1 Direction convention

All objectives shall first be transformed into a common maximize convention:

```text
directed_score[s, i, j] = direction[j] * mean_score[s, i, j]
```

for ensemble member `s`, candidate `i`, and objective `j`, where `direction[j]` is `+1` for maximization and `-1` for minimization. Objectives declared `maintain` or used as constraints are not scalarized directly; they are handled under Section 6.6.5.

#### 6.6.2 Quantile normalization

Bradley-Terry scores are ordinal within a single trained ensemble and carry an arbitrary latent scale that changes on every retraining. They shall not be compared across objectives in their raw form.

For each objective `j`, a monotone map `Q[j]` shall transform directed scores onto `[0, 1]` using the empirical distribution of predicted scores. Two reference distributions are defined, and they are not interchangeable:

| Map | Reference distribution | Used for |
|---|---|---|
| `Q_pool[j]` | Predicted scores over the current round's validated candidate pool | Selection within a round |
| `Q_measured[j]` | Predicted scores over all accepted measured variants | Reporting, evaluation, cross-round comparison |

Selection requires only within-round comparability, so `Q_pool[j]` shall be used for scalarization. It preserves resolution at the top of the pool, where candidates that exceed every measured variant would otherwise all saturate at `1.0` under `Q_measured[j]`.

Cross-round quantities — including any hypervolume reported under Section 10 — shall use `Q_measured[j]` with a reference point pinned in campaign configuration, because a pool-relative map is not comparable between rounds.

The map shall be applied per ensemble member, using a reference distribution computed from that member's own predictions. Applying a single pooled map across members would compress ensemble disagreement and destroy the uncertainty signal that Section 6.6.4 depends on.

```text
f[s, i, j] = Q_pool[s, j]( directed_score[s, i, j] )      f in [0, 1]
```

#### 6.6.3 Randomized Chebyshev scalarization

For a batch of size `q`, draw `q` weight vectors `w^(1) ... w^(q)` from the `(M-1)`-simplex over the `M` scalarized objectives. Weights shall be drawn from a seeded low-discrepancy or uniform-simplex sequence recorded in the round manifest, so that a round is reproducible.

Each slate slot `k` uses the augmented Chebyshev achievement function:

```text
g[s, i, k] = min_j ( w[k, j] * f[s, i, j] )
             + rho * sum_j ( w[k, j] * f[s, i, j] )
```

with `rho = 0.05` by default. The leading `min` term is what allows ParEGO to recover candidates in nonconvex regions of the Pareto front, which a weighted sum cannot reach. The augmentation term breaks ties among candidates with equal minimum-weighted achievement and prevents selection of weakly dominated points.

Scalarization shall be applied independently to every ensemble member. This yields, for each slot `k`, an `(N, S)` matrix `g[:, :, k]` with the same shape and interpretation as the single-objective ensemble prediction matrix, which is what permits reuse of the existing selector without modification to its scalar core.

#### 6.6.4 Batch selection

Selection shall proceed sequentially over slate slots. Uncertainty shall enter the acquisition exactly once, through the constant-liar upper-confidence bound. Version 1 deliberately provides no per-objective `beta` bonus and no fixed exploitation/uncertainty/random batch quotas: stacking three independent exploration mechanisms makes each of them uninterpretable and makes tuning any one of them unattributable.

For slot `k`:

1. Recompute the ensemble covariance from the scalarized matrix `g[:, :, k]`, including the noise nugget scaled by `lie_noise_stddev_multiplier`.
2. Replay the liar downdate for every candidate already selected in slots `1 ... k-1` as a rank-1 covariance downdate at that candidate's index, with the fake observation set at the configured baseline. The covariance must be rebuilt per slot because the scalarization changed, so the downdate history cannot simply be carried forward.

The covariance shall not be materialized densely. A dense `(N, N)` representation makes step 2 cost `O(q^2 N^2)`, which is prohibitive at realistic pool sizes: measured selection time for `q = 24` grows from 0.3 s at `N = 500` to 137 s at `N = 4000`, extrapolating to several minutes at `N = 6789`.

The covariance is structurally low-rank plus a scaled identity:

```text
Cov = D^T D / S + sigma^2 * I
```

where `D` is the `(S, N)` matrix of ensemble deviations for the current slot's scalarized predictions and `sigma^2` is the lie-noise nugget. Each liar downdate adds exactly one rank-1 term, so after `q` slots the correction has rank at most `S + q`. Every quantity the selection loop requires — the diagonal, a single column, and the rank-1 downdate — is available in this factored form at `O((S + q) N)` rather than `O(N^2)`. Implementations shall maintain the factored representation. Re-symmetrization is then unnecessary because symmetry is structural, but implementations shall still clamp variances at the nugget floor to prevent cancellation from producing negative values under the square root.
3. Mask all previously selected candidates.
4. Apply the redundancy penalty of Section 6.6.6.
5. Select the candidate maximizing `mean + ucb_beta * sigma`.

Exploration pressure over a campaign shall be controlled by `lie_noise_stddev_multiplier`, either fixed or as an explicit per-round schedule. This is the mechanism already validated in the single-objective implementation and it shall remain the single exploration knob.

An optional `random_fraction` of the batch may be reserved for uniformly sampled valid candidates. It defaults to `0`, since the liar noise schedule already supplies exploration; campaigns enabling it shall record the reason.

#### 6.6.5 Constraints and interval objectives

An objective declared `maintain` with a target interval, or used as a hard constraint with a numeric threshold, is specified in assay units. A ranking score cannot be compared to a value in assay units. Such objectives therefore require a calibration layer, and Version 1 shall provide one rather than deferring it.

For each objective used as an interval or threshold constraint, an isotonic regression `C[j]` shall be fitted from ensemble mean ranking score to measured assay value, using accepted measurements only, with cross-validated fitting and out-of-fold residuals retained for uncertainty estimation. Isotonic regression is monotone by construction, so it preserves the ranker's ordering exactly while supplying the missing scale.

Feasibility shall be evaluated probabilistically rather than as a hard point estimate:

```text
feasibility[i, j] = P( C[j](score[i, j]) satisfies the constraint )
```

estimated from the ensemble spread and the calibration residual distribution. Candidates below a configured `min_feasibility` are removed before scalarization. Objectives that are purely directional require no calibration and shall not incur this cost.

Calibrated values shall be labeled as calibrated wherever they are surfaced, and shall not be reported as assay measurements. The calibration is only as valid as the measured range it was fitted on; extrapolation beyond that range shall be flagged in the ranked candidate record.

```yaml
optimization:
  primary_objective: objective_1
  constraints:
    - objective: objective_2
      operator: ">="
      threshold: 0.75
      calibration: isotonic
      min_feasibility: 0.5
    - objective: objective_3
      operator: between
      lower: 0.8
      upper: 1.2
      calibration: isotonic
      min_feasibility: 0.5
```

#### 6.6.6 Candidate pool bounds and redundancy

The scalar selector operates on at most `max_candidates_considered` candidates, default `5000`. When the validated pool exceeds this bound, the pool shall be reduced before selection, and the reduction rule shall be recorded.

Reduction shall not be a simple top-`K` by predicted mean. Under multiple objectives that rule is not well defined, and it is an exploitation bias applied upstream of the acquisition function, which silently discards the uncertain candidates the acquisition exists to find. The pool shall instead be reduced by taking the union, over the drawn weight vectors, of the top candidates under each scalarization, plus a random sample of the remainder. Quotas shall be recorded in the manifest.

A redundancy penalty shall be applied during selection against candidates already chosen in the current batch, computed in both mutation space and embedding space, with configurable per-seed and per-cluster caps. This addresses candidate collapse (Section 12) and is distinct from the liar downdate, which models predictive correlation rather than batch composition.

### 6.7 Experimental batch handoff

Selected candidates shall be frozen into an immutable round manifest containing:

- sequences and mutation notation;
- selection class and reason;
- predicted objective scores and uncertainties;
- parent seed and generator prompt;
- all model and configuration versions;
- plate or batch assignment when applicable; and
- positive, negative, reference, and process controls.

No model retraining may alter a frozen manifest.

### 6.8 Measurement ingestion and retraining

After experimental testing:

1. ingest replicate-level results;
2. apply assay quality control without overwriting raw measurements;
3. aggregate accepted replicates using the configured method;
4. estimate or update experimental-noise thresholds;
5. rebuild cumulative directional pairs;
6. retrain the MOCE adapter on accepted cumulative pairs;
7. restore objective rankers from their pinned naturalness-pretrained initialization;
8. fine-tune rankers on all accepted cumulative measurements; and
9. begin the next campaign round.

For reproducibility, the reference implementation should retrain from pinned base checkpoints using cumulative data. Incremental warm-starting from the previous campaign round may be offered as an explicitly versioned optimization after equivalence testing.

## 7. Data Contracts

### 7.1 Variant record

The variant record is content-addressed and carries only properties intrinsic to the sequence. It shall never carry information about how, when, under what sampling parameters, or how many times the sequence was produced. This record is stable forever: it does not change if the same sequence is generated again by a different seed, control vector, or random seed.

```json
{
  "variant_id": "sha256:...",
  "sequence": "MKT...",
  "mutations": ["A12G", "L48V"],
  "edit_distance_reference": 2
}
```

`variant_id` shall be the sha256 hash of `sequence` and is the sole identity key for deduplication, novelty tracking, embedding, ranking, and measurement joins.

### 7.2 Generation event record

A generation event record describes one sampling event that produced a variant. Because autoregressive sampling repeatedly rediscovers closely related or identical sequences (Section 12, Candidate collapse), the same `variant_id` may be referenced by many generation event records across seeds, rounds, and control vectors. This is expected and is itself a useful redundancy signal; it is the reason generation properties were removed from the variant record.

```json
{
  "generation_event_id": "uuid:...",
  "variant_id": "sha256:...",
  "parent_variant_id": "sha256:...",
  "generation_round": 3,
  "control_tokens": ["inc", "dec", "maintain"],
  "generator_model": "model@revision",
  "generator_adapter": "adapter@revision",
  "temperature": 1.0,
  "top_k": 10,
  "random_seed": 42,
  "generator_log_probability": -12.4
}
```

`parent_variant_id` refers to the seed variant for this event, not to the generated variant's own ancestry outside of generation. A variant reached by a non-generative route (the reference sequence, a systematic single-mutant library entry, or an operator-submitted sequence) has no generation event record.

### 7.3 Embedding record

An embedding record is keyed on the triple of variant, embedding checkpoint, and pooling method, matching the cache key and required fields already specified in Section 5.2. Naturalness may use a checkpoint independent of the embedding checkpoint (Section 5.2), so its model and revision are recorded separately from the embedding's.

```json
{
  "variant_id": "sha256:...",
  "embedding_model": "model@revision",
  "pooling": "campaign_validated_pooling",
  "embedding_vector": [0.0132, -0.884, "..."],
  "naturalness_model": "model@revision",
  "naturalness_score": -0.83,
  "computation_timestamp": "2026-07-30T00:00:00Z",
  "preprocessing_version": "v3"
}
```

`variant_id` already is the sequence hash required by Section 5.2's cache key, so no separate `sequence_hash` field is stored. Changing `embedding_model` or `naturalness_model` (including revision) requires new embedding records rather than mutation of existing ones, consistent with Section 5.2's prohibition on reusing a feature space across checkpoint changes.

### 7.4 Measurement record

Measurements shall be stored in tidy form so objectives can be added without schema migrations. A single `round` field is ambiguous between the round in which the variant's batch was frozen and the round in which this particular measurement was taken; these diverge for retested reference and leading variants (Section 12) and for controls repeated across batches (Section 6.7). The two are therefore stored separately.

```json
{
  "variant_id": "sha256:...",
  "batch_round": 3,
  "measured_round": 5,
  "objective": "objective_1",
  "replicate": 1,
  "value": 2.73,
  "unit": "campaign_defined_unit",
  "assay_batch": "batch_identifier",
  "instrument_id": null,
  "quality_status": "accepted",
  "quality_notes": null
}
```

`batch_round` is the round manifest (Section 7.6) that selected this variant for testing. `measured_round` is the round during which this measurement was ingested (Section 6.8, step 1); for a retest, `batch_round` remains the variant's original selecting round while `measured_round` advances. `instrument_id` is an optional, nullable provenance field to support the assay-batch covariates called for in Section 12; it is not required and shall be left `null` when not tracked.

### 7.5 Ranked candidate record

```json
{
  "variant_id": "sha256:...",
  "predictions": {
    "objective_1": {"mean": 1.42, "stddev": 0.21},
    "objective_2": {"mean": -0.68, "stddev": 0.14}
  },
  "normalized_predictions": {
    "objective_1": {"mean": 0.94, "stddev": 0.06},
    "objective_2": {"mean": 0.71, "stddev": 0.09}
  },
  "normalization_reference": "pool",
  "feasibility": {
    "objective_3": 0.82
  },
  "calibration_extrapolating": {
    "objective_3": false
  },
  "nearest_measured_embedding_distance": 0.19,
  "slate_slot": 4,
  "scalarization_weights": [0.62, 0.38],
  "scalarized_score": 1.07,
  "pareto_front": 0,
  "selection_reason": "argmax_ucb_under_slot_weights",
  "selected": true
}
```

`variant_id` references the record in Section 7.1, unambiguously, because generation metadata no longer lives there.

The record stores both raw and normalized predictions. Raw `predictions` are retained for auditing and for retraining diagnostics; `normalized_predictions` are the quantile-transformed values that selection actually consumed (Section 6.6.2), and `normalization_reference` records which reference distribution was used, since pool-anchored and measured-anchored values are not interchangeable.

`scalarization_weights` records the weight vector drawn for this slate slot under Section 6.6.3, `slate_slot` the slot index, and `scalarized_score` the resulting augmented-Chebyshev value. `pareto_front` is retained only as an optional diagnostic index; it is not a selection criterion in Version 1.

`feasibility` and `calibration_extrapolating` are populated only for objectives carrying an interval target or numeric constraint (Section 6.6.5), and are absent for purely directional objectives. A candidate whose calibration is extrapolating beyond the fitted measured range shall carry the flag even when it is judged feasible.

### 7.6 Round manifest

Every round manifest must contain hashes of:

- input measurement snapshot;
- paired training dataset;
- generator checkpoint and adapter;
- embedding and naturalness checkpoints;
- ranker ensemble checkpoints;
- campaign configuration;
- generated candidate pool, comprising both variant records and their associated generation event records;
- score normalization statistics and, where applicable, fitted calibration maps;
- the drawn scalarization weight vectors and the seed that produced them; and
- final selected batch.

Because a candidate is the join of a variant record and a generation event record rather than a single fused record, the manifest shall hash the round's generation event log alongside the variant set, so that the exact (variant, generation event) pairing considered for selection remains reproducible.

Normalization statistics and calibration maps are hashed because both are fitted from data and both sit directly in the selection path: an identical pool, identical checkpoints, and identical weights can still yield a different batch if either map is refitted. A manifest that omits them does not satisfy Section 11.

## 8. Software Interfaces

```python
class SequenceGenerator:
    def fit(self, paired_dataset, config): ...

    def generate(
        self,
        seeds,
        control_vectors,
        config,
    ) -> list[Candidate]: ...


class SequenceValidator:
    def validate(self, candidates, reference, rules) -> ValidationResult: ...


class SequenceEmbedder:
    def embed(self, sequences) -> dict[str, EmbeddingRecord]: ...

    def naturalness(self, sequences) -> dict[str, float]: ...


class ObjectiveRanker:
    def pretrain(self, embeddings, naturalness, config): ...

    def fit(
        self,
        embeddings,
        measurements,
        pair_criterion,
        config,
    ): ...

    def predict(self, embeddings) -> list[EnsemblePrediction]: ...


class ScoreNormalizer:
    """Monotone map from raw ranking scores onto [0, 1] (Section 6.6.2).

    Fitted per objective and per ensemble member. `reference` selects the
    distribution the map is anchored to: "pool" for within-round selection,
    "measured" for cross-round reporting. The two are not interchangeable.
    """

    def fit(self, ensemble_predictions, reference): ...

    def transform(self, ensemble_predictions) -> NormalizedPredictions: ...


class ObjectiveCalibrator:
    """Monotone score-to-assay-unit map for interval and constraint
    objectives only (Section 6.6.5). Directional objectives skip this.
    """

    def fit(self, ensemble_predictions, measurements, config): ...

    def feasibility(
        self,
        ensemble_predictions,
        constraint,
    ) -> FeasibilityResult: ...

    def is_extrapolating(self, ensemble_predictions) -> list[bool]: ...


class BatchSelector:
    def select(
        self,
        candidates,
        normalized_predictions,
        objective_definitions,
        feasibility,
        batch_size,
        config,
    ) -> list[SelectedCandidate]: ...
```

All interfaces shall accept explicit configuration objects and return serializable provenance. Hidden reliance on mutable global campaign state is prohibited.

`BatchSelector` receives normalized predictions and precomputed feasibility rather than raw scores and raw constraints. Normalization and calibration are separate, independently testable stages precisely because they are the two places where a scale error would silently corrupt selection without producing an error.

The scalar core of `BatchSelector` — the constant-liar upper-confidence-bound loop — shall remain unaware of the number of objectives. Scalarization (Section 6.6.3) reduces every slate slot to an `(N, S)` matrix of the same shape the single-objective implementation already consumes, so multi-objective support shall not require a second selector implementation.

## 9. Configuration Requirements

A complete campaign configuration shall specify:

```yaml
campaign:
  name: example_campaign
  reference_sequence_file: data/reference.fasta
  random_seed: 42

generator:
  model_id: Rostlab/prot_t5_xl_uniref50
  model_revision: pinned_revision
  adapter_method: lora
  top_k: 10
  temperature: 1.0
  candidates_per_seed: 20

embeddings:
  model_id: esmc_300m
  model_revision: pinned_revision
  pooling: campaign_validated_pooling

naturalness:
  model_id: campaign_selected_model
  model_revision: pinned_revision

ranker:
  ensemble_size: 5
  hidden_dims: [100, 50]
  dropout: 0.2
  learning_rate: 0.0003
  weight_decay: 0.00001
  max_epochs: 200
  early_stopping_patience: 40
  loss: bradley_terry

acquisition:
  method: parego_constant_liar
  normalization: quantile_pool          # quantile_pool for selection; quantile_measured for reporting
  chebyshev_rho: 0.05
  weight_sampling: low_discrepancy_simplex
  weight_random_seed: 42
  ucb_beta: 2.0
  lie_noise_stddev_multiplier_schedule: [3.0, 3.0, 30.0, 30.0, 100.0]
  max_candidates_considered: 5000
  random_fraction: 0.0
  redundancy_penalty:
    mutation_space: true
    embedding_space: true
    max_per_seed: 8
    max_per_cluster: 8

calibration:
  method: isotonic                      # required only for interval or constraint objectives
  cross_validation_folds: 5
  min_feasibility: 0.5
  flag_extrapolation: true
```

Objective definitions and candidate filters are campaign-specific sections of the same configuration.

## 10. Evaluation Plan

### 10.1 What historical replay can and cannot measure

The five comparison arms are not uniformly evaluable by deep-mutational-scanning (DMS) replay:

1. random selection;
2. naturalness-only selection;
3. MOCE generation without FolDE ranking;
4. FolDE ranking of a systematic candidate library; and
5. MOCE generation followed by FolDE ranking and hybrid acquisition.

Arms 1, 2, and 4 select from a pre-enumerated candidate library (the full DMS variant set or a systematic single/multi-mutant library). Every candidate they can select has a ground-truth measurement, so historical replay against a DMS dataset is a valid, leakage-controlled evaluation for these arms.

Arms 3 and 5 include a generator that is explicitly designed to emit sequences outside any enumerated set, including combinations and positions never assayed. A DMS dataset has no ground-truth value for such a sequence. There are only two honest ways to close that gap, and neither is offline DMS replay:

- restrict the generator's output to the DMS-covered subset, which measures rejection-sampling behavior on a truncated space, not the generator's actual proposal distribution, and biases every downstream metric (validity rate, diversity, enrichment) in an uncontrolled way; or
- score off-library candidates with a surrogate oracle or wet-lab assay, in which case the arm is no longer "offline DMS replay" but a different evaluation tier with its own error sources.

**Offline DMS replay cannot establish the generator's contribution to campaign outcomes.** Any report that compares arm 5 against arm 4 using only DMS-covered candidates shall be labeled as a ranking/acquisition comparison restricted to enumerable space, not as evidence about generation. This restriction, and the direction of its bias (surrogate-limited candidate diversity, likely undercounting the generator's value on truly novel combinations), must be stated adjacent to any such result.

Evaluation is therefore structured into three tiers:

**Tier 1 — Offline DMS replay (arms 1, 2, 4; ranking and acquisition only).**
Uses the existing simulation harness (`folde/campaign.py`, `simulate_campaign` / `simulate_campaigns` / `simulate_campaigns_with_config_checkpoints`) unmodified: multiple DMS datasets, multiple independent simulations per configuration with fixed per-simulation random seeds, per-round metrics (including model Spearman correlation), and per-mutant records (`round_found`, `activity`, `predicted_activity`, `percentile`). This tier is fully leakage-controlled and repeatable and is the primary evidence source for ranker and acquisition design decisions.

**Tier 2 — Surrogate-oracle study (arms 3 and 5; generation contribution, explicitly labeled as a surrogate).**
A held-out oracle model, trained on a data split disjoint from any data used to fit the campaign's own rankers (different DMS positions/combinations held out, or a separate DMS dataset for the same or a homologous protein), scores off-library generated candidates. The report accompanying any Tier 2 result must state: the oracle's own training data and held-out performance; that oracle agreement is not ground truth and can share the same blind spots as the ranker it is meant to check (both are typically trained on similar low-order mutational data and may extrapolate similarly, or dissimilarly for uninteresting reasons); and that Tier 2 results are hypothesis-generating for Tier 3, not confirmatory on their own.

**Tier 3 — Prospective pilot (arms 3 and 5; the only tier with true ground truth on generated sequences).**
A small wet-lab pilot campaign, run end-to-end through the manifest and measurement-ingestion pipeline described in Sections 6.7-6.8, is required before any claim of generator benefit is treated as established rather than provisional.

### 10.2 Primary campaign metrics

- best measured value per objective, and probability of discovering a top-percentile variant, both computed only against actual assay measurements;
- number and fraction of candidates satisfying all constraints;
- enrichment of successful variants in selected batches relative to random selection;
- experimental measurements required per successful variant;
- generated-candidate validity rate (raw generator output surviving validation; see Section 11.1 for the accompanying acceptance threshold);
- unique substitutions and residue positions explored;
- mutation-space and embedding-space diversity of the selected batch;
- uncertainty calibration and error-versus-distance behavior of the rankers; and
- Pareto hypervolume, subject to the constraints in 10.3.

### 10.3 Hypervolume is a diagnostic on predicted scores and a metric only on measured values

Bradley-Terry ensemble scores are ordinal on a per-training-run latent scale: the reference architecture uses bias-free linear layers throughout (including the final scalar output layer), so neither the zero point nor the scale of `mean_score[j]` is identified, and both can shift after every retrain. Hypervolume is not scale-invariant, so hypervolume computed directly on raw `mean_score[j]` values is an arbitrary, retrain-dependent reweighting of the objectives and is not comparable across rounds, configurations, or arms. This is consistent with Section 6.6, which selects via ParEGO (randomized Chebyshev scalarization over per-objective quantile-normalized scores) rather than nondominated-sorting-plus-hypervolume; Section 10 does not reintroduce hypervolume as a selection rule.

Requirements for any reported hypervolume number:

1. Hypervolume over *predicted* scores, if reported at all, shall be computed only after quantile-normalizing each objective within the round's candidate pool, against a reference point pinned once in campaign configuration for the life of the campaign. It shall be labeled a diagnostic, not a campaign-success metric, since it still describes model belief rather than outcome.
2. Hypervolume over *measured* assay values, computed the same way (quantile-normalized per objective, fixed reference point), is the metric that may be used for cross-round and cross-arm comparison, because it is anchored to ground truth rather than to a training-run-specific score scale.
3. Any comparison of hypervolume across configurations or arms must use the same normalization statistics (fit on a shared reference set, e.g. the full DMS activity distribution for that dataset) so that "one unit of hypervolume" means the same thing in both arms being compared.

### 10.4 Novelty reporting

Novelty shall be reported separately as:

- unseen full sequence;
- unseen combination of previously observed substitutions;
- unseen individual substitution at an observed position; and
- substitution at a previously unobserved position.

### 10.5 Leakage rule

No future-round measurements may be used to generate, train, rank, normalize, or select candidates in an earlier replay round.

## 11. Acceptance Criteria

Version 1 is complete when:

1. **Raw generation validity rate.** At least 15% of raw generator output (before any validation filtering) survives validation and enters the candidate pool, measured per campaign round and averaged over at least 3 rounds. This threshold is set low relative to typical classifier-style validity rates because the generator is a free-form ~3B-parameter encoder-decoder constrained post hoc to emit an exact-length, substitution-only sequence with no constrained decoding (Section 5.1); a large fraction of raw samples are expected to fail the length or edit-type check alone, and setting the bar near 100% would be implausible for this architecture. The 15% threshold is a floor for flagging a broken adapter or a misconfigured control-token vocabulary, not a target for optimization; campaigns are free to report and improve on it. Validator self-consistency (candidates that pass validation continuing to satisfy the same rules on re-check) is retained as a lower-priority, non-blocking regression check rather than an acceptance criterion, since it primarily tests the validator's own determinism.
2. No previously measured or already queued sequence occupies a **discovery** slot in a batch. This deduplication rule applies to the discovery portion of the batch only. Positive, negative, reference, and process controls (Section 6.7), and periodic retests of reference or leading variants (Section 12), are drawn from a separately budgeted control/retest allocation and are exempt from this rule; a manifest's discovery-slot count and control/retest-slot count shall each be recorded and shall sum to the total batch size.
3. Previously unseen substitutions and mutation combinations can pass through generation, validation, embedding, ranking, and selection.
4. Each objective can independently be maximized, minimized, maintained within a range, or used as a constraint. Interval and constraint objectives shall be served by the isotonic calibration layer of Section 6.6.5, with feasibility reported probabilistically and extrapolation beyond the fitted measured range flagged on every affected candidate.
5. Ranker training uses only measurements available before the relevant campaign round.
6. Ensemble uncertainty and distance from measured data are recorded for every ranked candidate.
7. Multi-objective selection can return candidates spanning the nondominated set without a fixed weighted sum. Randomized augmented-Chebyshev scalarization (Section 6.6.3) satisfies this: weights vary per slate slot and the Chebyshev term reaches nonconvex regions of the front that a linear scalarization cannot. A single fixed weight vector applied to every slot does not satisfy this criterion.
8. Every selected candidate is traceable to its seed, prompt, random seed, configuration, data snapshot, and model revisions.
9. A complete campaign round can be reproduced from its manifest.
10. **Statistically supported improvement over baseline, by tier.** For each tier defined in Section 10.1, one primary metric is pre-registered before results are examined; all other metrics listed in Section 10.2 are exploratory and reported without a pass/fail claim.
    - **Tier 1 (arms 1, 2, 4).** Primary metric: measured-value hypervolume (Section 10.3) after the final round, or another single pre-declared metric if the campaign states one in advance. For each of at least 5 DMS datasets, run at least 10 independent simulations per arm per dataset (matching the harness's `number_of_simulations` parameter) under fixed, recorded random seeds, and take the per-dataset mean as one paired observation. Compare the hybrid arm against a **pre-declared** strongest baseline — the baseline arm must be named in the evaluation config before simulations run, not selected post hoc from whichever arm scored lowest. Use a two-sided Wilcoxon signed-rank test on the per-dataset paired differences (appropriate for the expected small number of paired dataset-level observations without assuming normality); report the exact p-value, the median paired difference, and a 95% bootstrap confidence interval on that difference. Significance threshold: alpha = 0.05 on the single pre-registered primary metric.
    - **Tier 2 (arms 3, 5, surrogate oracle).** Primary metric: oracle-scored best value among generated candidates versus the pre-declared strongest Tier-1-evaluable baseline (arm 4) on the same oracle. Same paired-per-dataset design and test as Tier 1. Because the oracle is not ground truth (Section 10.1), a positive Tier 2 result is necessary but not sufficient for the acceptance claim and shall be reported as such.
    - **Tier 3 (prospective pilot).** No inferential test is required for version-1 acceptance; a pilot campaign that completes end-to-end and produces manifests conforming to Section 7.6 satisfies this criterion. A statistical claim of prospective improvement is out of scope for version 1 and deferred to a follow-up campaign with a pre-registered analysis plan.
    - **Multiplicity.** All non-primary metrics from Section 10.2, across all arms and tiers, are exploratory. When they are reported alongside a primary-metric result, p-values (if computed at all for exploratory metrics) shall be adjusted for multiple comparisons using Benjamini-Hochberg at FDR = 0.10, and the report shall state how many exploratory tests were run. No exploratory metric result may be characterized as "significant" or used to claim acceptance on its own.

## 12. Risks and Mitigations

### Generator distribution shift

The generator may produce sequences far outside the ranker's training distribution.

Mitigations:

- edit-distance bounds;
- naturalness floors;
- nearest-measured embedding distance;
- ensemble uncertainty;
- explicit out-of-distribution penalties; and
- gradual increases in allowed mutation order.

### Ranker exploitation

Optimization may find sequences that receive extreme model scores for nonbiological reasons.

Mitigations:

- ensemble models;
- hard biological constraints;
- random exploration;
- acquisition penalties for extreme embedding distance; and
- experimental controls and replicate validation.

### Candidate collapse

Autoregressive sampling may repeatedly generate closely related variants.

Mitigations:

- sequence deduplication;
- mutation-space clustering;
- embedding-space diversification;
- multiple seeds and temperatures; and
- per-seed or per-cluster selection caps.

### Noisy or inconsistent assays

Pairwise methods can amplify incorrect orderings.

Mitigations:

- replicate-level storage;
- assay-batch covariates where appropriate;
- threshold-aware pair construction;
- quality-control exclusion without raw-data deletion; and
- periodic retesting of reference and leading variants.

### Conflicting objectives

Independent objective rankers may favor incompatible regions of sequence space.

Mitigations:

- Pareto acquisition;
- explicit constraints;
- diverse batch selection across Pareto regions; and
- operator-visible trade-off reports before batch freezing.

### Epistatic extrapolation

Rankers trained mostly on single substitutions may be unreliable on higher-order variants.

Mitigations:

- mutation-order curriculum;
- inclusion of measured higher-order variants in every later training round;
- maximum-distance rules; and
- dedicated exploratory allocation for uncertain combinations.

## 13. Extension Points

Future versions may add:

- newer autoregressive or masked protein generators;
- shared-trunk, multi-head objective rankers;
- full calibrated regression heads for all objectives, extending the restricted isotonic calibration that Version 1 applies only to interval and constraint objectives (Section 6.6.5);
- hypervolume-based batch acquisition such as qNEHVI, replacing ParEGO scalarization once objectives are calibrated onto a common scale with a defensible reference point;
- structure confidence, stability, solubility, expression, aggregation, or binding predictors;
- constrained decoding at immutable residues or motifs;
- insertion and deletion support;
- cost-aware and synthesis-aware acquisition;
- assay-batch correction and hierarchical replicate models;
- active selection of control-token directions;
- human-in-the-loop candidate approval; and
- federated campaign learning across related proteins.

All extensions should implement the existing candidate scoring or validation interfaces rather than bypass campaign provenance and selection controls.

## 14. Security, Reproducibility, and Governance

- Raw measurements shall be immutable and access controlled.
- Credentials and remote model tokens shall be supplied through environment variables or an approved secret manager.
- Model revisions, dependency locks, random seeds, and hardware-relevant determinism settings shall be recorded.
- Generated sequences and campaign objectives shall be reviewed under the organization's applicable biosafety and responsible-design process before synthesis.
- Candidate rejection and manual override decisions shall be logged with a reason and operator identity.
- Automatic external synthesis submission is outside the scope of version 1.

## 15. Foldy Integration

This section maps the design in Sections 1-14 onto the running Foldy application (`backend/app`, `backend/folde`, `worker/`) as of this writing. It is written from the code, not from intent; where the code does not do what the rest of this spec assumes, that is called out explicitly as a gap.

### 15.1 Component-to-runtime mapping

Foldy's architecture is React SPA -> Flask-RESTX API -> Postgres, with RQ queues `cpu`, `gpu`, `biggpu`, and `emailparrot` backed by Redis (`backend/app/helpers/rq_helpers.py:23`, queue names confirmed in `backend/app/metrics.py:30-43` and enqueue call sites such as `backend/app/views/few_shot_views.py:210` (`cpu`) and `backend/app/util.py:104-105` (`cpu`/`biggpu`)). There is no queue named `gpu` in use today for heavy PLM work — `few_shot_views.py` enqueues few-shot fitting on `cpu`, and `app/util.py` uses `biggpu` for large-model paths. Worker images are built per-tool (`worker/Dockerfile.esm`, `worker/Dockerfile.boltz`, `worker/Dockerfile.prosst`, `worker/Dockerfile`), each a separate conda env baked into a distinct Docker image, not a shared "GPU worker."

| Spec component | Section | Proposed runtime | Rationale |
|---|---|---|---|
| Controlled sequence generator (5.1) | 5.1, 6.3 | New offline/batch job on `biggpu`, **not** request path | `Rostlab/prot_t5_xl_uniref50` is a ~3B-parameter encoder-decoder. No existing worker Dockerfile loads a model this large; `worker/Dockerfile.esm` installs ESM/E1 (largest current model is ESM3/E1, run via `app/helpers/esm_client.py`) but has no T5/LoRA/PEFT generation stack wired to a job. Generation for a round (many seeds x many samples) is latency-tolerant and should not block an API request; it belongs on the same class of queue as existing large-model inference (`biggpu`), in its own worker image (`worker/Dockerfile.moce` does not exist and must be created) rather than shoehorned into `Dockerfile.esm`, since PEFT/LoRA fine-tuning per round is a distinct, heavier dependency set (see 15.1.1 below). |
| MOCE adapter fine-tuning (6.8 step 6) | 6.8 | `biggpu`, offline/batch, per-round | Same checkpoint, LoRA fine-tune is training, not inference; must run after `evolve_jobs.py`-style few-shot retraining and before the next generation pass. No analogous "train a generator" job exists in `app/jobs/` today — closest precedent is `app/jobs/evolve_jobs.py`, which trains `TorchMLPFewShotModel` few-shot rankers, not a language model. |
| Embedding / naturalness service (5.2) | 5.2, 6.5 | Existing `cpu`/`biggpu` split, reused as-is | Foldy already computes embeddings and naturalness via `Embedding`/`Naturalness` models and ESM clients in `app/helpers/esm_client.py`, enqueued through `app/jobs/esm_jobs.py`. This machinery is the correct home for spec Section 5.2 and needs schema extension (15.2), not a new runtime path. |
| Objective rankers (5.3) | 5.3, 6.8 | `cpu`, matches existing few-shot path | `TorchMLPFewShotModel` (`backend/folde/few_shot_models.py:608`) already implements this architecture: `hidden_dims`, `dropout`, `ensemble_size`, `pretrain`/`train_epochs`, Adam-style optimizer. `app/jobs/evolve_jobs.py` drives it and is enqueued on `cpu` (`few_shot_views.py:210`). The spec's independent-ensemble-per-objective requirement (5.3) is not yet expressed — see 15.2. |
| Validation/filters (6.4) | 6.4 | In-process, wherever candidates are produced (API request for small batches, worker job for generator output) | Cheap, synchronous, deterministic; no queue needed beyond whatever produced the candidates. |
| Multi-objective acquisition / batch selection (6.6) | 6.6 | `cpu` | `constant_liar_sample` in `backend/folde/util.py:145` already runs on CPU/CUDA-optional tensors and is invoked from campaign simulation code, not from a queued job today — it currently runs inside `folde/campaign.py` simulation loops (offline evaluation), not the live campaign flow exposed via `app/views/campaign_views.py`. Wiring it into a live round requires a new job, most naturally on `cpu` given its cost profile. |
| Email/round-complete notifications | 6.7, 6.8 | `emailparrot` | Existing pattern, e.g. `app/views/admin_views.py:232`; reuse directly for round-manifest-ready and retrain-complete notifications. |

**15.1.1 The prot_t5_xl_uniref50 sizing problem.** None of the four worker Dockerfiles (`worker/Dockerfile`, `Dockerfile.boltz`, `Dockerfile.esm`, `Dockerfile.prosst`) install `transformers` T5 classes with a PEFT/LoRA stack pointed at `Rostlab/prot_t5_xl_uniref50`; `Dockerfile.esm` installs `peft==0.9.0` for the E1/ESM path but that is wired to `app/helpers/esm_client.py`, not a generator. A ~3B-parameter encoder-decoder model plus per-round LoRA fine-tuning needs its own worker image and its own queue slot sized for that memory footprint (existing `biggpu` machine classes, per `app/metrics.py`'s `size_biggpu_g`/`normsize_biggpu_g` gauges, are the closest fit, but capacity has not been validated against a 3B model plus PEFT training in the same job). This is unresolved: worker GPU memory budgets for `biggpu` are not documented in the reviewed files.

### 15.2 Schema gaps

Section 5.2 of this spec requires per-embedding provenance (model, exact revision, pooling, sequence hash, preprocessing version). Section 7 requires content-addressed variant IDs and structured per-generation-event and per-round provenance. The current schema (`backend/app/models.py`) does not provide these.

| Area | Spec requirement | Current state | Gap |
|---|---|---|---|
| Embedding model identity | Model + exact revision (5.2) | `Embedding.embedding_model` is a bare `db.String` (`models.py:256`); `Campaign.embedding_model` likewise a bare string defaulting to `"esm2_t33_650M_UR50D"` (`models.py:344`, `369`) | **Must add**: a revision/version column on `Embedding` and `Campaign` (or fold revision into the model-id string with an enforced format, which the spec explicitly says is insufficient — 5.2 wants revision as a first-class field). |
| Embedding pooling method | Required per-record (5.2) | Not present on `Embedding` at all | **Must add**: `pooling_method` column on `Embedding`. |
| Sequence hash / content-addressing | `variant_id: sha256:...` for every variant (7.1) | No hash column anywhere in `models.py`; folds/sequences are identified by `Fold.id`/`Fold.name`, and campaign slates by ad hoc string IDs (`CampaignRound.slate_seq_ids`) | **Must add**: a `sequence_hash` (or full content-addressed `variant_id`) column, likely on a new variant/candidate table (see below), computed and stored at generation/measurement time. |
| Preprocessing version | Required per embedding record (5.2) | Not present | **Must add**: `preprocessing_version` column on `Embedding`. |
| Default model mismatch | Spec reference default is ESMC-300M / 960-dim (5.2, Section 9 YAML `embeddings.model_id: esmc_300m`) | `Campaign` defaults both `naturalness_model` and `embedding_model` to `"esm2_t33_650M_UR50D"` (`models.py:343-344`, constructor `368-369`) | Not a schema gap per se, but a live default mismatch: campaigns created today do not default to the spec's reference embedding model. Config reconciliation (15.4) must decide whether to change the default or treat it as campaign-configurable, which it already nominally is (it's a free string). |
| Round manifest structure | Structured round manifest with hashes of input snapshot, paired dataset, all checkpoints, config, candidate pool, and selected batch (7.6) | `CampaignRound.slate_seq_ids` is a single comma-separated `db.Text` column (`models.py:407`); `input_templates` is likewise a comma-separated text blob (`models.py:409-411`); there is no hash/checksum column for any of these, no per-candidate structured record, and no stored config/checkpoint version snapshot beyond `Campaign.naturalness_model`/`embedding_model` strings | **Must add**: a new `CampaignRoundCandidate` (or similarly named) table with one row per scored/selected candidate — variant id, sequence, mutation notation, per-objective mean/stddev predictions, selection class/reason, parent seed, generator prompt/config — replacing the flat `slate_seq_ids` string. `CampaignRound` needs additional columns or a linked JSON/manifest blob for: input measurement snapshot hash, paired-dataset hash, generator checkpoint+adapter identifiers, ranker ensemble checkpoint identifiers, and campaign config hash. |
| Per-objective independent rankers | One ensemble per objective (5.3) | `FewShot` (`models.py:281`, table `fold_evolution`) stores one `few_shot_params` blob and one `output_fpath` per run, with no explicit per-objective ensemble linkage; `CampaignRound.few_shot_run_id` is a single FK to one `FewShot` row (`models.py:401-406`) | **Must add**: either a `CampaignRoundObjectiveModel` join table (round x objective x few-shot-run) or an objective-keyed structure inside `FewShot`/`Campaign`. Current schema assumes one ranking run per round, not N. |
| Objective/constraint definitions | Structured, campaign-scoped, with direction/target-interval/pair-threshold (5.1, 6.4, 6.6) | No table stores objective definitions at all; `Campaign` has no objectives relation | **Must add**: an `Objective` (or `CampaignObjective`) table: name, direction (maximize/minimize/maintain), target interval, pair threshold, token position — referenced by generator prompts, pair construction, and acquisition. |
| Generator provenance | Checkpoint, adapter, temperature, top-k, random seed, log-probability per variant (7.1) | No columns exist for any of this on any model | **Must add**: a generation event table per Section 7.2, keyed to the candidate/variant table. These are properties of a sampling event, not of a sequence, so they must not be columns on the variant table — the same sequence can be produced by many events. |
| Variant/candidate validation state | Validation pass/fail plus reason (6.4, 14) | No table tracks rejected candidates or rejection reasons | **Must add**: either a status/reason column on the new candidate table, or a separate `CampaignRoundRejectedCandidate` table if volume makes storing all rejects in the main table impractical. |

None of the above should be written as migration files per these instructions; they are listed as required schema work for a follow-on migration (in the style of the existing `backend/migrations/versions/3f8a9c1d2e4b_add_lookup_indexes_and_uniqueness.py`, which only added indexes/uniqueness and did not touch campaign tables).

### 15.3 Where the generator does not fit today

The current campaign flow (`backend/folde/campaign.py`) scores an **enumerated** candidate pool. `CampaignSimulationHelper.get_mutant_pool()` (`folde/campaign.py:71`) calls `get_mutant_pool` from `folde/rust_mutant_pool.py`, which filters a *known, finite* list of `seq_ids` down to those within one substitution of a measured variant (`get_mutant_pool_python`, `folde/rust_mutant_pool.py:39-68`, and its Rust-accelerated counterpart). Everything downstream — activity/naturalness/embedding lookups (`campaign.py:85-91`), pool-relative score normalization, and scalarized UCB selection — assumes this pool is materialized in advance as a pandas-indexable set of `seq_id`s.

A MOCE-style generator breaks this in several concrete ways:

1. **Pool enumeration assumption.** `get_mutant_pool` takes `seq_ids: Sequence[str]` as an already-known candidate universe (typically all single/double mutants of the wild type) and filters it. A generator instead produces sequences on demand; there is no pre-existing `seq_ids` list to filter, so `get_mutant_pool`/`get_accelerated_mutant_pool` cannot be reused unmodified for generator output. The candidate table proposed in 15.2 must become the new source of "the pool for this round," replacing the enumerate-then-filter pattern for generator-fed rounds.

2. **Pool-relative score normalization.** Anywhere scores or percentiles are computed relative to "all mutants in the pool" (e.g., percentile ranks referenced in `folde/types.py`'s `MutantMetrics.percentile`), the pool size and composition must now be defined operationally as "everything generated and validated this round," not "every single/double mutant of the reference." This changes the meaning of percentile and any pool-size-dependent statistic and must be redefined per-round rather than assumed campaign-global.

3. **`MAX_POINTS_TO_CONSIDER = 5000` cap.** `constant_liar_sample` in `folde/util.py:164` hard-caps the ensemble-prediction matrix at 5000 rows and raises `ValueError` above that (`folde/util.py:171-174`). An enumerated single/double-mutant pool for a moderate-length protein can already approach or exceed this in the existing system; a generator makes it worse because there is no natural upper bound on how many candidates can be produced (temperature/top-k sampling can be run indefinitely). Either the generation step must be capped to at most 5000 (or however many the round's compute budget allows) validated, deduplicated candidates before selection, or `constant_liar_sample` must be changed to pre-filter/pre-cluster before scoring. The spec does not currently state a cap on generated-candidate-pool size per round; this must be added as an explicit generation-time bound (e.g., in `candidate_filters` or a new `max_candidates_per_round` config key), and the existing 5000-row ceiling in `util.py` should be treated as an unmodified downstream constraint unless deliberately changed with its own equivalence testing.

4. **Fixed candidate set for reproducibility (Sections 7.6, 11).** The evaluation plan (Section 10) and acceptance criteria (Section 11) assume a round can be reproduced from its manifest. For an enumerated pool this is trivial (the pool is derived deterministically from the reference sequence and measured set). For a generator, reproducibility additionally depends on sampling seed, temperature, top-k, and the exact adapter checkpoint (captured in the generation event record, Section 7.2) — this is achievable but requires that the candidate table in 15.2 actually be populated for every generation event, not just the final selected batch, since the round manifest must hash "generated candidate pool" (7.6) as a distinct artifact from "final selected batch."

### 15.4 Config reconciliation

The system already has a Pydantic config class, `FolDEModelConfig` (`backend/folde/types.py:6-21`), which is what the running few-shot/naturalness pipeline consumes (`few_shot_model_name`, `few_shot_model_params`, `naturalness_model_id`, `embedding_model_id`, etc.). Section 9 of this spec proposes a standalone YAML campaign config covering `campaign`, `generator`, `embeddings`, `naturalness`, `ranker`, and `acquisition` sections.

These are not the same shape, and `FolDEModelConfig` today has no fields for: generator model/adapter/LoRA settings, top-k/temperature/candidates-per-seed, acquisition method/beta/fraction splits, or objective/constraint definitions. It also has no per-objective structure — it is a single-model config, matching the current one-ranker-per-round reality noted in 15.2.

**Recommendation: treat the Section 9 YAML as a superset and extend `FolDEModelConfig` rather than introducing a second, parallel config object.** Concretely:

- Add a `GeneratorConfig` sub-model (model_id, model_revision, adapter_method, lora_rank/alpha/dropout, top_k, temperature, candidates_per_seed) and an optional `generator: Optional[GeneratorConfig] = None` field on `FolDEModelConfig`, so existing few-shot-only campaigns are unaffected.
- Add an `AcquisitionConfig` sub-model matching Section 9 (method, normalization, chebyshev_rho, weight_sampling, weight_random_seed, ucb_beta, lie_noise_stddev_multiplier_schedule, max_candidates_considered, random_fraction, redundancy_penalty) and an `acquisition: Optional[AcquisitionConfig] = None` field. Note that `beta` and the exploitation/uncertainty fraction splits from earlier drafts no longer exist; exploration is controlled solely by `ucb_beta` and the liar-noise schedule (Section 6.6.4).
- Add an `objectives: List[ObjectiveConfig]` field (name, token_position, goal, target_interval, pair_threshold), replacing the implicit single-objective assumption baked into today's one `FewShot` run per round.
- Keep `embedding_model_id`/`naturalness_model_id` as the existing scalar fields, but add companion `embedding_model_revision`/`naturalness_model_revision` fields (str, required once provenance work in 15.2 lands) rather than overloading the id string.
- The Section 9 YAML's `campaign:` block (name, reference_sequence_file, random_seed) maps to fields that already partially exist as `Campaign` ORM columns (`name`, implicit reference via `Fold`) — this block should be treated as request/API-level input that constructs a `Campaign` row plus a `FolDEModelConfig`, not as fields inside `FolDEModelConfig` itself.

The alternative — keeping the spec's YAML as a fully separate config and writing a translation layer into `FolDEModelConfig` at job-submission time — was considered and rejected: it creates two sources of truth for the same run parameters and doubles the validation surface. Extending the existing Pydantic model keeps one schema that both the live system and the hybrid system share, at the cost of `FolDEModelConfig` growing several optional sub-models.

### 15.5 Phasing

Suggested independently shippable milestones, roughly in dependency order:

1. **Provenance schema.** Add the embedding revision/pooling/preprocessing columns and the content-addressed candidate table (15.2), with no behavior change — pure additive migration, backfill-compatible.
2. **Objective-model schema.** Add the `Objective`/`CampaignObjective` table and per-objective ranker linkage, still operating on the existing enumerated-pool flow. This alone lets Section 5.3/5.4/6.6's multi-objective machinery run against today's mutant-pool campaigns before the generator exists.
3. **`FolDEModelConfig` extension.** Land the `GeneratorConfig`/`AcquisitionConfig`/`objectives` fields as optional, unused additions, validated against Section 9 examples, with no runtime consumer yet.
4. **Generator worker (offline/batch only).** Stand up the new worker image and `biggpu` job for MOCE generation and LoRA fine-tuning, producing candidate rows in the new table, run manually / out-of-band from live campaigns for validation against Section 10's evaluation plan (historical replay first).
5. **Pool-cap and normalization fixes.** Address the `MAX_POINTS_TO_CONSIDER` ceiling and pool-relative percentile semantics (15.3) before generator output is allowed to feed live acquisition.
6. **Live round integration.** Wire generator output into `constant_liar_sample`/acquisition as an alternative to `get_mutant_pool`, gated by a campaign-level flag so enumerated-pool campaigns are unaffected; expose through `app/views/campaign_views.py` and `app/jobs/evolve_jobs.py`-equivalent job for generation+retraining.
7. **Round manifest and reproducibility.** Populate the full Section 7.6 manifest (hashes of snapshot, paired dataset, checkpoints, config, pool, selected batch) once the schema and worker paths from steps 1-6 are stable enough that all inputs are actually available to hash.

Each milestone above is additive to the existing schema and job system and can ship without the later ones; milestone 6 is the first that changes live campaign behavior.

## 16. Licensing

Foldy is distributed under a modified BSD license. The referenced MOCE repository does not visibly provide a license file. Before copying or redistributing MOCE source code, obtain licensing clarification from its authors. A clean implementation of the published method behind the versioned interfaces in this specification may be preferable to directly combining repository source files.

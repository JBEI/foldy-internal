# Generative FolDE Multimutant Benchmark Implementation Specification

## 1. Document status

- **Status:** Phases 0-2 executed; Phase 3 pending
- **Date:** 2026-07-30
- **Primary implementation area:** `backend/folde/`
- **Related system specification:** `2026-07-21-moce-folde-hybrid-spec.md`
- **Initial benchmark datasets:** ProteinGym substitution assays already present under
  `backend/folde/data/DMS_ProteinGym_substitutions/`

This document specifies an offline benchmark for determining whether a protein language
model (PLM) used as a candidate generator improves FolDE campaigns beyond strictly
one-mutation-at-a-time exploration.

The benchmark is a prerequisite for production integration of an autoregressive or masked
protein generator. It is intentionally narrower than the MOCE-FolDE hybrid system
specification: this document establishes the interfaces, controls, datasets, and evidence
needed to decide whether generative proposal is useful.

### 1.1 Execution update (2026-08-11)

- Phase 0 dataset manifests, hashes, schemas, and Tier 1 contracts are complete.
- Phase 1 candidate-pool abstractions, oracle boundary, provenance, and synthetic campaign
  contracts are complete.
- Phase 2 Olson Protocol A completed 20 common seeds and 120 campaigns. PLM-plus-FolDE had
  a positive median advantage over adjacent FolDE, but its paired 95% bootstrap interval
  crossed zero, so the preregistered production-integration gate did not pass.
- Phase 3 free ESMC generation has not started.
- A separate multi-objective ALDE screening branch completed 420 campaigns across six
  single-mutant datasets. Heterogeneous proposals passed their proposal-quality gate, but
  the selector and end-to-end gates failed; production Torch-MLP replication is therefore
  not authorized without revising and preregistering the selection hypothesis.
- A preregistered naturalness-aware selector follow-up then completed 540 fresh campaigns.
  A 25% soft naturalness prior was neutral overall and failed its selector and end-to-end
  gates. A diagnostic bottom-quartile naturalness veto beat plain ParEGO on five of six
  datasets, but its paired interval crossed zero. Neither result authorizes production
  replication.

The machine-readable artifacts live under `backend/folde/model_evals/generative/` and
`backend/folde/model_evals/260811-multiobjective-alde-gates/`.

## 2. Decision to be supported

The benchmark must support a defensible decision on the following question:

> Under an equal experimental budget and equal proposal-compute budget, does a
> PLM-generated candidate pool followed by FolDE acquisition discover higher-fitness
> multimutants more efficiently than adjacent mutation, random multimutant generation,
> or combinations of strong single mutants?

The benchmark must separately answer two subordinate questions:

1. **Proposal quality:** Does the generator place useful measured variants into the
   candidate pool?
2. **Selection quality:** Given the same candidate pool, does FolDE select the useful
   variants?

A method must not be credited as a successful generator merely because FolDE rescues a
poor proposal pool, nor credited as a successful selector because it was given a privileged
pool.

## 3. Goals

Version 1 shall:

1. Simulate sequential design-build-test-learn campaigns over measured multimutant
   landscapes.
2. Support candidate generators that propose variants more than one mutation from the
   measured set.
3. Compare all methods under identical round size, number of rounds, proposal count, and
   random seeds.
4. Record candidate-pool provenance before FolDE selection.
5. Evaluate both generator recall and final campaign outcomes.
6. Support mutational-shell protocols such as single-to-double and single-to-triple or
   quadruple exploration.
7. Use measured DMS scores only after a candidate is selected for simulated measurement.
8. Reuse the existing FolDE zero-shot, few-shot, ensemble, and acquisition implementations
   where their semantics are appropriate.
9. Remain generator-agnostic so an autoregressive PLM, ESMC iterative-mask sampler, genetic
   generator, or deterministic baseline can use the same interface.
10. Produce resumable, reproducible, machine-readable result artifacts.

## 4. Non-goals for Version 1

Version 1 does not:

- benchmark insertions, deletions, or variable-length proteins;
- train a Raygun-style ESMC latent decoder;
- claim that ProteinGym results transfer directly to a new wet-lab campaign;
- assign ground-truth fitness to sequences absent from the measured benchmark landscape;
- compare raw DMS scores across different assays;
- fine-tune a large generator online after every simulated round;
- use structure prediction in the core loop;
- replace the existing single-mutant FolDE benchmark; or
- treat PLM likelihood as experimental activity.

All Version 1 variants are fixed-length substitutions relative to the dataset reference
sequence.

## 5. Current repository state

### 5.1 Existing campaign behavior

`CampaignWorldState` currently supports two candidate-universe modes:

- `one_mutation_at_a_time=True`: include single mutants and variants whose allele set is
  one edit from at least one measured variant;
- `one_mutation_at_a_time=False`: expose every unmeasured variant in the finite dataset.

Neither mode represents a dynamic candidate generator. The first mode hard-codes an
adjacency rule. The second gives the selector immediate access to the entire measured
landscape.

Version 1 shall add a candidate-pool strategy interface rather than overload the boolean
with additional meanings.

### 5.2 Existing multimutant benchmark

`backend/folde/scripts/251003_multimutant_benchmark.py` already evaluates:

- `SPG1_STRSG_Olson_2014`;
- `GRB2_HUMAN_Faure_2021`; and
- `PABP_YEAST_Melamed_2013`.

It has locally available ESMC-300M embeddings and ESMC-600M naturalness data for these
assays. It is a useful ranking baseline, but it does not isolate generator quality because
the default configuration exposes the full unmeasured universe.

### 5.3 Existing split metadata

`folde.data.get_proteingym_dataset()` creates:

- `one_vs_many_split` for `SPG1_STRSG_Olson_2014`; and
- `one_vs_many_split`, `two_vs_many_split`, and `three_vs_many_split` for
  `SPG1_STRSG_Wu_2016`.

The current campaign runner interprets `data_split_mode` as the set eligible for the
simulated world, not as an explicit observed-shell versus evaluation-shell split. Version 1
must therefore introduce explicit benchmark split semantics rather than silently reuse that
behavior.

## 6. Dataset inventory and tiers

The local directory contains 217 ProteinGym substitution assays. Sixty-nine contain at
least one multimutant.

### 6.1 Tier 1: primary benchmarks

| Dataset | Singles | Multimutants | Maximum depth | Local PLM features | Role |
|---|---:|---:|---:|---|---|
| `SPG1_STRSG_Olson_2014` | 1,045 | 535,917 | 2 | Yes | Primary single-to-double benchmark |
| `SPG1_STRSG_Wu_2016` | 76 | 149,284 | 4 | No | Primary deeper-shell benchmark |

#### SPG1 Olson properties

- 55 mutated positions.
- All 19 substitutions are represented at every position.
- Every double-mutant component is represented as a measured single.
- The double-mutant library is effectively exhaustive.
- An arbitrary valid double over the assayed region is expected to have measured fitness.
- The assay measures IgG binding.

This is the strongest local dataset for testing candidate generation rather than merely
ranking a sparse fixed library.

#### SPG1 Wu properties

| Mutation depth | Measured variants |
|---:|---:|
| 1 | 76 |
| 2 | 2,091 |
| 3 | 26,019 |
| 4 | 121,174 |

The assay spans four positions and is close to the complete combinatorial landscape. It is
the primary test of direct jumps from single mutants into triple- and quadruple-mutant
shells.

Before full execution, ESMC embeddings and naturalness values must be generated for this
dataset using pinned model revisions and the same pooling convention as the existing FolDE
features.

### 6.2 Tier 2: immediate replication datasets

| Dataset | Singles | Doubles | Notes |
|---|---:|---:|---|
| `GRB2_HUMAN_Faure_2021` | 1,034 | 62,332 | Broad 56-position yeast-growth landscape |
| `PABP_YEAST_Melamed_2013` | 1,187 | 36,521 | Broad 75-position essential-function landscape |

Both datasets already have local ESMC-300M embeddings and ESMC-600M naturalness values.
They are suitable for replication of single-to-double conclusions.

They are not exhaustive double-mutant landscapes. A generator may propose a valid sequence
whose activity was not measured. Generator evaluation on these assays must therefore use
one of the closed-world policies in Section 10.4 and report proposal coverage.

### 6.3 Tier 3: secondary stress tests

| Dataset | Singles | Multimutants | Maximum depth | Intended use |
|---|---:|---:|---:|---|
| `F7YBW8_MESOW_Aakre_2015` | 37 | 9,155 | 4 | Small four-site combinatorial smoke test |
| `GFP_AEQVI_Sarkisyan_2016` | 1,084 | 50,630 | 15 | Broad fluorescence landscape |
| `PHOT_CHLRE_Chen_2023` | 2,122 | 165,407 | 15 | Large fluorescence landscape |
| `CAPSD_AAV2S_Sinai_2021` | 532 | 41,796 | 28 | Deep mutational-distance stress test |
| `HIS7_YEAST_Pokusaeva_2019` | 168 | 495,969 | 28 | Large growth landscape |
| `Q8WTC7_9CNID_Somermeyer_2022` | 1,201 | 32,309 | 43 | Fluorescent-protein design distribution |
| `Q6WV12_9MAXI_Somermeyer_2022` | 1,141 | 30,260 | 13 | Fluorescent-protein design distribution |
| `D7PM05_CLYGR_Somermeyer_2022` | 1,169 | 23,346 | 23 | Fluorescent-protein design distribution |

Tier 3 assays require feature generation and additional analysis of library-construction
bias. They are not release blockers for Version 1.

## 7. Benchmark terminology

- **Reference sequence:** Dataset wild-type sequence from ProteinGym metadata.
- **Mutation depth:** Number of substituted loci relative to the reference sequence.
- **Measured set:** Variants whose DMS score has been revealed to the simulated campaign.
- **Ground-truth universe:** All variants with finite DMS scores in the selected benchmark
  split.
- **Eligible universe:** Ground-truth variants permitted by the protocol's position,
  alphabet, and mutation-depth constraints.
- **Proposal pool:** Ordered set of candidates returned by a generator before activity
  prediction or acquisition.
- **Scored pool:** Proposal pool after validity checks and PLM feature attachment.
- **Selected slate:** Candidates selected for simulated measurement.
- **Proposal budget:** Maximum number of unique valid candidates a generator may place in
  the proposal pool per round.
- **Experimental budget:** Number of variants in the selected slate per round.
- **Coverage:** Fraction of generated valid sequences present in the ground-truth universe.
- **Oracle access:** Reading a candidate's DMS score. Oracle access is forbidden until
  simulated measurement.

## 8. Architecture

```text
Reference sequence + measured variants + round configuration
                           |
                           v
                  CandidatePoolStrategy
                           |
                    proposal records
                           |
                           v
             validation / deduplication / lookup
                           |
                    valid covered pool
                           |
                           v
            embedding and naturalness attachment
                           |
                           v
                FolDE prediction ensemble
                           |
                           v
           acquisition + batch diversity selection
                           |
                    selected slate
                           |
                           v
             DMS oracle reveals selected scores
                           |
                           v
                 next campaign round
```

The generator must not receive DMS scores for unmeasured variants. The benchmark harness
shall pass only the reference sequence, measured observations, constraints, and approved
PLM artifacts into the generator.

## 9. Core interfaces

The exact class names may change during implementation, but the responsibilities and
serialized fields are required.

### 9.1 Mutation representation

```python
class Substitution(BaseModel):
    reference_aa: str
    position: int
    alternate_aa: str


class VariantIdentity(BaseModel):
    sequence: str
    sequence_hash: str
    seq_id: str
    substitutions: tuple[Substitution, ...]
    mutation_depth: int
```

Positions shall use the ProteinGym mutation numbering in persisted provenance. Internal
zero-based sequence indices may be stored separately, but conversion must be validated
against the reference amino acid.

`seq_id` shall use the existing canonical FolDE ordering so the same substitution set maps
to one identifier regardless of generator order.

### 9.2 Proposal record

```python
class CandidateProposal(BaseModel):
    identity: VariantIdentity
    parent_seq_ids: tuple[str, ...]
    generator_name: str
    generator_version: str
    generator_checkpoint: str | None
    generation_seed: int
    proposal_rank: int
    proposal_score: float | None
    proposal_log_probability: float | None
    metadata: dict[str, JsonValue]
```

`proposal_score` may be method-specific. It must never be labeled as activity unless it is
an output from the campaign activity model.

### 9.3 Generator context

```python
class GeneratorContext(BaseModel):
    reference_sequence: str
    measured_variants: tuple[MeasuredVariant, ...]
    allowed_positions: frozenset[int]
    allowed_alphabet: frozenset[str]
    min_mutation_depth: int
    max_mutation_depth: int
    proposal_budget: int
    round_number: int
    random_seed: int
```

### 9.4 Candidate generator protocol

```python
class CandidateGenerator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, context: GeneratorContext) -> Sequence[CandidateProposal]:
        """Return at most context.proposal_budget unique proposals."""
```

The protocol shall have no activity-oracle argument.

### 9.5 Candidate-pool strategy

`CandidatePoolStrategy` may wrap one or more generators and is responsible for:

- allocating proposal budget among parents;
- enforcing global uniqueness;
- mixing local and long-jump proposal channels;
- filling a shortfall with a declared fallback strategy; and
- retaining source-channel provenance.

The first implementation shall support:

```yaml
proposal_mix:
  plm: 0.70
  adjacent: 0.20
  random: 0.10
```

Fractions shall be converted to deterministic integer counts for a fixed budget and seed.

### 9.6 Oracle

```python
class FitnessOracle(Protocol):
    def measure(self, seq_ids: Sequence[str]) -> pd.Series:
        """Reveal measured activity for the selected slate only."""
```

The ProteinGym oracle shall verify that every requested sequence:

- is in the eligible universe;
- has not been measured previously; and
- has a finite activity value.

Oracle lookup calls and returned identifiers shall be recorded.

## 10. Candidate coverage policies

ProteinGym is a finite measured library. A free generator may produce sequences with no
ground truth. The benchmark must declare one of the following policies per protocol.

### 10.1 Exhaustive-landscape policy

Use when all or nearly all valid sequences under the generation constraints are measured,
as with SPG1 Olson doubles or SPG1 Wu's four positions.

- Missing proposals count against coverage.
- Missing proposals are rejected before FolDE scoring.
- The generator may continue sampling until it reaches either the unique-valid proposal
  budget or the configured raw-sampling cap.
- Both raw sample count and accepted proposal count are reported.

### 10.2 Library-constrained scoring policy

The generator assigns scores or probabilities to the eligible measured library, and the
top proposal-budget variants become the proposal pool.

This tests whether the PLM defines a useful proposal distribution but does not test
free-running sequence generation. Results must be labeled **closed-world proposal
ranking**.

### 10.3 Rejection-sampling policy

The generator samples freely; only generated sequences present in the measured library are
retained.

This tests generation and coverage jointly. Low coverage is a real failure mode, not a
reason to grant the generator additional unlimited compute. A raw-sampling cap is required.

### 10.4 Sparse-library recommendation

GRB2, PABP, GFP, PHOT, AAV, and similar sparse libraries shall initially use
library-constrained scoring. Rejection-sampling results may be reported as secondary
coverage experiments, but campaign comparisons must not silently discard methods that
produce unmeasured candidates at different rates.

## 11. Required generator baselines

### 11.1 Adjacent generator

Returns variants one substitution away from at least one measured variant, matching the
semantics of the current accelerated mutant pool.

This is the principal FolDE baseline.

### 11.2 Uniform shell generator

Samples uniformly from eligible unmeasured variants at the configured mutation depth.

This controls for the benefit of merely allowing long jumps.

### 11.3 Top-single combination generator

Constructs multimutants from substitutions observed in high-performing measured singles.

Required variants:

- additive ranking by sum of measured single-mutant effects;
- PLM-naturalness tie-breaking; and
- optional exclusion of pairs at the same locus.

This is a strong, inexpensive baseline and must not be replaced by random generation.

### 11.4 Genetic generator

Uses measured high-fitness parents, mutation, and optional recombination without PLM
guidance.

The mutation-depth and proposal budgets must match the PLM arm.

### 11.5 PLM-only generator

Uses PLM proposal probability or naturalness without the FolDE activity model.

This isolates the value of FolDE selection.

### 11.6 PLM plus FolDE

Uses the same PLM proposal pool as PLM-only, then applies FolDE prediction and acquisition.

The proposal pool must be byte-for-byte identical between the two arms for a fixed
simulation seed and round state.

### 11.7 Full-universe FolDE upper bound

Exposes all eligible measured variants to FolDE.

This is not a deployable generator and shall be labeled an upper-bound selector experiment.
It estimates how much performance is lost through proposal-pool truncation.

## 12. PLM generator implementations

### 12.1 ESMC iterative-mask generator

This is the recommended first learned generator because ESMC conditions each substituted
position on both left and right context.

Reference algorithm:

1. Select one or more measured parents.
2. Draw a target mutation depth from the configured distribution.
3. Choose mutable positions using a configured position policy.
4. Mask selected positions.
5. Obtain conditional residue logits from ESMC.
6. Sample or beam-search alternate residues while forbidding the parent residue when a
   mutation is required.
7. Optionally repeat masked refinement sweeps.
8. Canonicalize, deduplicate, and return proposals.

Required generation parameters:

- model name and exact revision;
- selected parent policy;
- position policy;
- temperature;
- top-k or top-p;
- number of refinement sweeps;
- target-depth distribution;
- raw-sampling cap; and
- random seed.

### 12.2 Autoregressive PLM generator

An autoregressive implementation shall use the same interface. It must document how it
anchors generation to a full parent sequence. Prefix-only generation without a
template-preservation mechanism is not considered equivalent to local multimutant
generation.

Acceptable anchoring mechanisms include:

- an encoder-decoder model conditioned on the complete parent;
- mutation tokens or control prompts learned from variant pairs;
- constrained decoding that preserves nonselected positions; or
- library-constrained scoring of complete candidates.

### 12.3 No learned ESMC decoder requirement

The benchmark shall not depend on a Raygun-style ESMC decoder. If iterative masking or an
existing autoregressive checkpoint fails to improve campaigns, decoder development may be
reconsidered with evidence about the missing capability.

## 13. Parent and position policies

Generators shall declare how parents and mutable positions are selected.

### 13.1 Required parent policies

- reference sequence only;
- best measured activity;
- top-k measured activity, uniformly allocated;
- diversity-aware elite set; and
- mixed reference/elite policy.

The default learned-generator policy after round 1 shall allocate:

| Parent source | Proposal fraction |
|---|---:|
| Top measured elites | 60% |
| Diverse measured variants | 20% |
| Reference sequence | 20% |

### 13.2 Required position policies

- uniform allowed positions;
- PLM entropy;
- measured single-mutant effect;
- FolDE sensitivity, if available; and
- mixed policy.

Position selection must not inspect unmeasured DMS scores.

## 14. Benchmark protocols

### 14.1 Protocol A: exhaustive single-to-double

**Dataset:** `SPG1_STRSG_Olson_2014`

**Purpose:** Primary proof that a learned generator plus FolDE improves two-mutation
exploration.

**Eligible universe:** All measured singles and doubles over positions 228–282.

**Initialization options:**

1. WT plus a random measured single-mutant batch.
2. WT plus a naturalness-selected single-mutant batch.
3. A fixed common set of singles shared across all methods.

Option 3 is the required main comparison because it eliminates first-round variance.

**Campaign defaults:**

| Parameter | Default |
|---|---:|
| Initial measured singles | 32 |
| Round size | 16 |
| Additional rounds | 5 |
| Proposal budget per round | 10,000 |
| Raw PLM sampling cap | 100,000 |
| Target depth after initialization | 2 |
| Simulation seeds | 20 |

All arms receive the same initial measurements for a simulation seed.

### 14.2 Protocol B: shell-jump benchmark

**Dataset:** `SPG1_STRSG_Wu_2016`

**Purpose:** Determine whether direct jumps into triple and quadruple mutants beat
sequential shell traversal.

Required subprotocols:

1. **1 to 2:** initialize with singles; target doubles.
2. **1 to 3:** initialize with singles; target triples.
3. **1 to 4:** initialize with singles; target quadruples.
4. **Curriculum:** permit depth at most `round + 1`.
5. **Mixed jump:** allocate proposals across depths 2, 3, and 4 each round.

Suggested mixed distribution:

```yaml
target_depth_weights:
  2: 0.40
  3: 0.35
  4: 0.25
```

The position set is restricted to 265, 266, 267, and 280 using ProteinGym numbering.

### 14.3 Protocol C: double-mutant replication

**Datasets:**

- `GRB2_HUMAN_Faure_2021`;
- `PABP_YEAST_Melamed_2013`.

Use library-constrained proposal scoring. Keep the same initial-observation, proposal, and
experimental budgets as Protocol A where dataset size permits.

These results establish whether conclusions from the exhaustive Protein G landscape
transfer to broader and sparser mutation regions.

### 14.4 Protocol D: broad sparse-landscape stress test

Run only after Tier 1 and Tier 2 acceptance criteria pass.

For GFP, PHOT, AAV, HIS7, and Somermeyer datasets:

- define explicit mutation-depth bands;
- require minimum sample counts per band;
- use library-constrained scoring for the primary comparison;
- report library-construction and activity-floor effects;
- avoid pooling raw fitness across assays; and
- report each assay independently before any aggregate rank statistic.

## 15. Campaign lifecycle

For every simulation:

1. Load and validate the dataset and reference sequence.
2. Build the eligible universe without exposing its activity values to models.
3. Select the fixed initial measured set using the simulation seed.
4. Reveal initial activity values through the oracle.
5. Pretrain and fit FolDE using only permitted features and revealed activities.
6. For each round:
   1. build `GeneratorContext`;
   2. generate proposals;
   3. canonicalize and deduplicate;
   4. validate sequence, mutation depth, positions, and alphabet;
   5. apply the declared coverage policy;
   6. attach cached PLM features;
   7. fit or update FolDE on measured variants;
   8. predict the proposal pool;
   9. select a slate using the configured acquisition method;
   10. reveal slate activity through the oracle;
   11. save a round checkpoint.
7. Calculate terminal campaign and generator metrics.
8. Write a complete result artifact.

No random operation may use the module-global random generator. Seeds must be derived from:

```text
benchmark seed
  + dataset
  + simulation index
  + method
  + round
  + component
```

The derivation scheme must be stable and recorded.

## 16. FolDE scoring and acquisition

### 16.1 Ranker inputs

The initial implementation shall use:

- ESMC-300M pooled embedding;
- ESMC-600M log naturalness; and
- measured activity labels.

Exact model revisions and pooling method must be pinned in configuration.

### 16.2 Ranker fitting

FolDE shall fit only on measured activity values. Unmeasured proposal embeddings and
naturalness may be used as inference inputs but not as activity labels.

Naturalness pretraining behavior must match the production FolDE configuration under test.

### 16.3 Acquisition

Required selectors:

- predicted mean;
- UCB;
- constant liar; and
- random-within-proposal-pool.

At least one diversity-aware selector is required. It may use:

- embedding distance;
- mutation-set Jaccard distance;
- Hamming distance; or
- an explicit redundancy penalty.

The default selected slate must prevent duplicate sequences and may enforce a maximum
number of variants from one parent.

## 17. Metrics

### 17.1 Primary campaign metrics

Reported after every round and at campaign completion:

- best DMS score found;
- normalized best-found score within the assay;
- percentile of best variant found;
- cumulative top-1% hits;
- cumulative top-10% hits;
- fraction of selected slate in the assay's top 1%;
- simple regret relative to the best eligible variant; and
- area under the best-found-versus-measurements curve.

The primary Version 1 endpoint is:

> Paired difference in area under the best-found curve between PLM-plus-FolDE and adjacent
> FolDE over common datasets and simulation seeds.

### 17.2 Proposal metrics

For every method and round:

- raw sequences sampled;
- syntactically valid sequences;
- unique valid sequences;
- proposal-pool size;
- duplicate rate;
- measured-library coverage;
- mutation-depth distribution;
- parent distribution;
- mean and quantiles of PLM proposal score;
- fraction of eligible top-1% variants included in the proposal pool;
- fraction of eligible top-10% variants included;
- best oracle fitness in the pool, calculated only after the round result is frozen for
  analysis; and
- proposal-pool diversity in sequence and embedding space.

The post-hoc best oracle fitness in the pool must never be fed back into the simulation.

### 17.3 Selection metrics

- best fitness selected divided by best fitness available in the proposal pool;
- top-1% precision in the selected slate;
- proposal-pool top-1% recall versus selected-slate top-1% recall;
- rank of the best selected candidate under FolDE;
- prediction calibration by mutation depth;
- ensemble spread versus absolute prediction error; and
- diversity retained from proposal pool to selected slate.

### 17.4 Epistasis metrics

For datasets with complete component singles, compute for double mutant `AB`:

```text
epistasis(AB) = score(AB) - score(A) - score(B) + score(WT)
```

If the assay does not include an explicit WT score, use only a dataset-validated normalized
WT reference. Otherwise, report a WT-free interaction residual from a model fitted to
single effects and label it accordingly.

Required outputs:

- beneficial-epistasis variants found;
- best beneficial epistasis found;
- fraction of selected doubles whose activity exceeds both component singles; and
- performance on variants whose additive single-mutant baseline is misleading.

### 17.5 Compute metrics

- generator wall time;
- PLM forward passes or tokens evaluated;
- embedding/naturalness wall time;
- FolDE fit and acquisition wall time;
- peak host memory;
- peak GPU memory; and
- serialized artifact sizes.

Methods exceeding the proposal-compute budget must be identified rather than compared as
equal-cost methods.

## 18. Statistical analysis

### 18.1 Paired simulations

All methods shall use:

- identical datasets;
- identical initial measured sets;
- identical experimental budgets; and
- common simulation seeds.

Method comparisons shall therefore use paired differences.

### 18.2 Required summaries

- median paired difference;
- mean paired difference;
- 95% paired bootstrap confidence interval;
- per-dataset result;
- aggregate normalized rank across datasets; and
- win/tie/loss counts across dataset-seed pairs.

Do not pool raw DMS scores across assays.

### 18.3 Multiple comparisons

The report shall distinguish:

- one preregistered primary comparison;
- required baselines; and
- exploratory ablations.

Exploratory p-values must not be presented as confirmatory evidence.

## 19. Leakage prevention

### 19.1 Forbidden inputs

Before simulated measurement, a generator, ranker, or acquisition function must not access:

- DMS scores for unmeasured variants;
- DMS score bins for unmeasured variants;
- rankings derived from the full measured library;
- dataset-specific top variants selected using ground truth; or
- a split constructed using activity values.

### 19.2 PLM training overlap

Pretrained PLMs may have encountered related natural sequences. This is acceptable for the
main benchmark but must be disclosed. No generator may be fine-tuned on the ProteinGym DMS
scores used as the oracle unless the protocol explicitly allocates those measurements to
the simulated measured set.

### 19.3 Feature computation

Embeddings and naturalness for all candidates may be precomputed because they do not use
assay activity. Feature files must not contain DMS activity columns.

### 19.4 Sparse-library selection bias

For sparse datasets, library membership may correlate with the original experiment's
selection process. Results shall be labeled closed-world and must not be interpreted as
unbiased free-generation performance.

### 19.5 Hyperparameter tuning

Generator and FolDE hyperparameters shall be selected using designated development
datasets or inner validation simulations. The Tier 1 final seeds must not be repeatedly
inspected during tuning.

## 20. Configuration schema

Illustrative benchmark configuration:

```yaml
name: generative-folde-v1
seed: 42

datasets:
  - dms_id: SPG1_STRSG_Olson_2014
    protocol: single_to_double
    coverage_policy: exhaustive
    allowed_positions: [228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
                        238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
                        248, 249, 250, 251, 252, 253, 254, 255, 256, 257,
                        258, 259, 260, 261, 262, 263, 264, 265, 266, 267,
                        268, 269, 270, 271, 272, 273, 274, 275, 276, 277,
                        278, 279, 280, 281, 282]

campaign:
  initial_measurements: 32
  round_size: 16
  rounds: 5
  simulations: 20
  proposal_budget: 10000

features:
  embedding_model: esmc_300m
  embedding_revision: PIN_REQUIRED
  embedding_column: embedding
  naturalness_model: esmc_600m
  naturalness_revision: PIN_REQUIRED
  naturalness_columns: [log_wt_marginal]

generator:
  name: esmc_iterative_mask
  model: esmc_300m
  revision: PIN_REQUIRED
  temperature: 1.0
  top_k: 20
  refinement_sweeps: 1
  raw_sampling_cap: 100000
  parent_policy: mixed_elite
  position_policy: uniform
  min_mutation_depth: 2
  max_mutation_depth: 2

folde:
  zero_shot_model_name: NaturalnessZeroShotModel
  few_shot_model_name: TorchMLPFewShotModel
  acquisition: constantliar
  ensemble_size: 5

selection:
  diversity_metric: mutation_jaccard
  minimum_pairwise_distance: 1
  max_per_parent: 4

output:
  checkpoint_dir: folde/model_evals/generative
  save_proposal_pools: true
  compression: zstd
```

Configuration validation shall fail when:

- a requested shell is absent from the dataset;
- the proposal budget is smaller than the round size;
- allowed positions conflict with dataset mutations;
- feature revisions are unpinned in a release run;
- the selected coverage policy is incompatible with the dataset;
- the mutation alphabet contains nonstandard or reference-inconsistent residues; or
- a method cannot meet the declared proposal-compute contract.

## 21. Result artifact

Each run shall save:

```text
benchmark manifest
  dataset manifest
    simulation manifest
      method manifest
        round records
          raw generation summary
          accepted proposal pool
          FolDE predictions
          selected slate
          revealed measurements
          timing and memory
        terminal metrics
```

Required top-level provenance:

- Git commit;
- dirty-worktree flag;
- benchmark configuration hash;
- environment/package lock hash;
- CUDA and device information;
- model names and exact revisions;
- dataset file hashes;
- feature file hashes;
- timestamp; and
- schema version.

Large proposal pools shall be stored separately from the JSON summary in Parquet or another
typed columnar format. Summary artifacts shall reference them by relative path and hash.

## 22. Proposed code organization

```text
backend/folde/
├── candidate_generation/
│   ├── __init__.py
│   ├── base.py                  # Protocols and shared types
│   ├── adjacent.py              # Current adjacency behavior as a strategy
│   ├── random_shell.py
│   ├── top_single.py
│   ├── genetic.py
│   ├── esmc_masked.py
│   ├── autoregressive.py
│   ├── validation.py
│   └── provenance.py
├── benchmarks/
│   ├── multimutant_data.py
│   ├── multimutant_oracle.py
│   ├── multimutant_runner.py
│   ├── multimutant_metrics.py
│   └── schemas.py
├── scripts/
│   ├── run_generative_multimutant_benchmark.py
│   └── prepare_multimutant_features.py
└── tests/
    ├── test_candidate_generation.py
    ├── test_multimutant_oracle.py
    ├── test_multimutant_runner.py
    ├── test_multimutant_metrics.py
    └── test_multimutant_reproducibility.py
```

The implementation may use fewer modules initially, but generator code must not be placed
inside `campaign.py`.

## 23. Required changes to existing modules

### 23.1 `folde/types.py`

Add versioned configuration and result models for:

- proposal budgets;
- generator configuration;
- candidate-pool records;
- shell protocols;
- coverage policies; and
- generator metrics.

Do not add more meanings to `one_mutation_at_a_time`.

### 23.2 `folde/campaign.py`

Refactor the simulation loop so candidate-pool construction is injected.

Required compatibility:

- existing callers without a strategy retain current behavior;
- current single-mutant benchmark results remain reproducible;
- a strategy receives no ground-truth activity series for unmeasured variants; and
- round checkpoints include proposal provenance.

### 23.3 `folde/data.py`

Add utilities to:

- report mutation-depth distributions;
- validate mutation strings against the reference;
- construct explicit observed and target shells;
- expose dataset completeness statistics; and
- load large feature files without materializing unnecessary rows.

The existing Olson and Wu category columns may remain for compatibility but shall not be
the sole representation of shell protocols.

### 23.4 `folde/rust_mutant_pool.py`

Wrap the current behavior as the `AdjacentGenerator` or an adapter used by it. Preserve the
native acceleration and Python equivalence tests.

### 23.5 Feature storage

The current 12 GB Olson embedding CSV is expensive to parse repeatedly. Before large
benchmark sweeps, add one of:

- sharded Parquet;
- Arrow IPC;
- memory-mapped NumPy arrays plus an indexed identifier table; or
- a repository-standard equivalent.

Migration must verify exact identifier and vector equality against a sampled set from the
CSV source.

## 24. Testing requirements

### 24.1 Unit tests

Required:

- mutation parsing and canonicalization;
- reference-residue validation;
- mutation-depth calculation;
- candidate deduplication;
- proposal-budget enforcement;
- coverage-policy behavior;
- generator determinism for fixed seeds;
- seed separation across components;
- no duplicate measurement;
- no oracle access before selection;
- top-single combination correctness;
- adjacent-generator equivalence with current mutant-pool output;
- metric calculations on hand-computed fixtures; and
- configuration validation.

### 24.2 Integration tests

Create a small synthetic landscape with:

- one reference;
- complete singles;
- complete doubles;
- known additive and epistatic optima; and
- deterministic fake embeddings.

The integration suite shall demonstrate:

1. adjacent exploration cannot reach a double before a component single is measured;
2. a long-jump generator can propose the double immediately;
3. FolDE sees only generated candidates;
4. only selected candidates have activity revealed;
5. checkpoint/resume reproduces an uninterrupted run; and
6. proposal and experimental budgets are enforced.

### 24.3 Dataset contract tests

For SPG1 Olson, assert:

- 1,045 singles;
- 535,917 doubles;
- 55 mutated positions;
- maximum depth 2; and
- every double component exists in the single-mutant set.

For SPG1 Wu, assert:

- 76 singles;
- 2,091 doubles;
- 26,019 triples;
- 121,174 quadruples;
- four mutated positions; and
- maximum depth 4.

These tests should inspect only the activity CSVs and should be marked appropriately if
their runtime is unsuitable for the default unit-test suite.

### 24.4 Regression tests

- Existing `test_campaign.py` and `test_campaign_updated.py` must continue to pass.
- Native and Python adjacent-pool implementations must remain equivalent.
- Existing benchmark configuration parsing must remain compatible.

### 24.5 Performance tests

Measure:

- proposal lookup over 500,000 candidates;
- feature-row retrieval for a 10,000-candidate pool;
- peak memory for Olson;
- round checkpoint write time; and
- resume time.

The runner must not reparse the 12 GB Olson embedding CSV for every method or simulation.

## 25. Acceptance criteria

### 25.1 Engineering acceptance

Version 1 is implementation-complete when:

1. At least five required generator baselines use one common interface.
2. ESMC iterative-mask generation is implemented or a stubbed PLM adapter is replaced by a
   declared library-constrained PLM scorer.
3. Protocol A runs end-to-end with checkpoint/resume.
4. Protocol B runs end-to-end after Wu features are prepared.
5. Existing FolDE benchmarks remain functional.
6. No test detects unmeasured oracle access.
7. Result artifacts contain full proposal and selection provenance.
8. A fixed configuration is reproducible across two clean executions.

### 25.2 Scientific go/no-go criteria

The preregistered primary comparison is PLM-plus-FolDE versus adjacent FolDE on Protocol A.

A recommendation to proceed toward production integration requires:

1. Positive median paired improvement in area under the best-found curve.
2. A 95% paired bootstrap confidence interval whose lower bound is above zero on the
   primary Olson comparison, or consistent positive effects across Olson, GRB2, and PABP
   if Olson alone is underpowered for the selected budget.
3. PLM-plus-FolDE outperforming both PLM-only and uniform-shell-plus-FolDE, demonstrating
   contributions from both proposal and selection.
4. No unacceptable collapse in proposal diversity or measured-library coverage.
5. A practically meaningful improvement declared before final execution, such as:
   - at least 20% more top-1% hits under the same experimental budget; or
   - reaching a fixed best-fitness percentile with at least one fewer 16-variant round.

Protocol B is required before claiming benefit for jumps beyond double mutants.

### 25.3 No-go interpretation

Failure must be localized:

- Low top-variant recall in the proposal pool indicates a generator problem.
- Strong proposal pool but weak selected slate indicates a FolDE/acquisition problem.
- Strong closed-world ranking but low free-generation coverage indicates an anchoring or
  constrained-decoding problem.
- Good doubles but poor triples/quadruples indicates distance-dependent surrogate
  extrapolation or generator degradation.

No-go on one component does not imply that all generative campaign approaches are invalid.

## 26. Implementation phases

### Phase 0: dataset audit and manifests

Deliver:

- machine-readable depth/completeness report for all 217 local assays;
- Tier 1 dataset contract tests;
- dataset and feature hashes; and
- benchmark configuration schema.

### Phase 1: candidate-pool abstraction

Deliver:

- `CandidateGenerator` and `CandidatePoolStrategy`;
- adjacent, uniform-shell, and top-single generators;
- oracle boundary;
- proposal provenance; and
- synthetic integration benchmark.

### Phase 2: closed-world Olson benchmark

Deliver:

- library-constrained PLM scorer;
- Protocol A runner;
- random, adjacent, top-single, PLM-only, PLM-plus-FolDE, and full-universe arms;
- paired statistical report; and
- performance-safe Olson feature loading.

This phase can establish whether PLM proposal scores are useful before implementing
free-running generation.

### Phase 3: free ESMC generation

Deliver:

- ESMC iterative-mask generator;
- exhaustive-landscape coverage evaluation;
- matched proposal-compute comparisons; and
- generation ablations for temperature, parent policy, and position policy.

### Phase 4: Wu shell jumps

Deliver:

- ESMC feature generation for all 149,360 Wu variants;
- Protocol B;
- shell-specific FolDE calibration diagnostics; and
- direct-jump versus curriculum comparison.

### Phase 5: replication and stress tests

Deliver:

- GRB2 and PABP results;
- at least one Tier 3 assay;
- cross-dataset normalized summary; and
- recommendation on production integration.

## 27. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Generator proposes variants absent from sparse DMS data | Missing oracle outcomes | Use exhaustive landscapes first; report coverage; use closed-world policy for sparse assays |
| FolDE is overconfident far from measured variants | Poor long-jump selection | Report error by depth; add distance penalty; retain bridge variants |
| PLM likelihood dominates activity prediction | Natural but inactive selections | Separate PLM-only and PLM-plus-FolDE arms |
| Top-single combinations are already sufficient | Learned generator adds complexity without value | Treat top-single combination as a required strong baseline |
| Full-universe access inflates performance | Unrealistic selector estimate | Label as upper bound and enforce proposal budgets elsewhere |
| DMS library-construction bias | Misleading generalization | Use several assay types; distinguish exhaustive from sparse protocols |
| Large CSV feature files exhaust memory | Benchmark instability | Convert to indexed columnar or memory-mapped storage |
| Generator and naturalness model are the same PLM | Correlated scores and circular filtering | Report shared-model configuration; include independent or no-naturalness ablations |
| Model revision drift | Irreproducible features | Pin revisions and hash feature manifests |
| Online generator fine-tuning leaks oracle data | Invalid comparison | Fine-tune only on revealed measurements; record every training example |

## 28. Open decisions

The following must be resolved before the final Protocol A run:

1. Which exact PLM checkpoint and revision will serve as the first generator?
2. Will the primary learned arm use ESMC iterative masking or an existing autoregressive
   model?
3. What is the preregistered practical effect threshold?
4. Which acquisition method is primary: constant liar or UCB?
5. What proposal-compute unit will be enforced across generator families?
6. Should WT activity be represented explicitly or only through ProteinGym normalization?
7. Which feature storage format will replace repeated loading of the Olson CSV?
8. Will model tuning use Aakre, a held-out subset of Olson seeds, or a separate development
   assay?

## 29. Initial recommended execution

The shortest defensible path is:

1. Implement the candidate-pool and oracle boundaries.
2. Run SPG1 Olson in closed-world mode using existing PLM features.
3. Compare adjacent, random shell, top-single combinations, naturalness-only proposals,
   naturalness-plus-FolDE, and full-universe FolDE.
4. Add ESMC iterative-mask free generation and measure coverage on the same exhaustive
   double landscape.
5. Generate Wu features and test direct triple/quadruple jumps.

This sequence answers the highest-value scientific questions before committing to a new
autoregressive model or ESMC latent decoder.

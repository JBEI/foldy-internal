"""Multi-objective dataset loader for FolDE protein-engineering campaigns.

A "multi-objective" dataset is two or more DMS assays measured on the SAME protein
(same wild-type sequence, same mutational coordinate system), sharing a single
embedding feature space so that one ranker per objective can be trained over
that shared space (e.g. KCNJ2 function + surface expression).

This module is intentionally separate from `folde/data.py`: it reuses that
module's low-level helpers (seq_id normalization, validation, sorting) but does
NOT use `_find_foldydata_embedding_file`'s fuzzy substring matching to resolve
embeddings, because that matching is unsafe for multi-objective use (see the
NOTE on `_load_embedding_df` below). All embeddings for a campaign are resolved
from a single, explicitly-named local file.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from app.helpers.sequence_util import (
    allele_set_to_seq_id,
    maybe_get_seq_id_error_message,
    sort_seq_id_list,
)
from folde.data import (
    DATA_DIR,
    DMS_DIR,
    NATURALNESS_DIR,
    _find_foldydata_naturalness_file,
    _parse_embedding_columns_inplace,
    get_dms_metadata,
    maybe_modify_seq_id,
)

logger = logging.getLogger(__name__)

EMBEDDINGS_DIR = DATA_DIR / "embeddings"


# ############################################################################
# SPECS
# ############################################################################


class ObjectiveSpec(BaseModel):
    """One assay/objective within a multi-objective campaign."""

    name: str
    dms_id: str
    goal: Literal["maximize", "minimize", "maintain"] = "maximize"
    # Only meaningful when goal == "maintain": the acceptable range for the objective.
    target_interval: Optional[Tuple[float, float]] = None

    @model_validator(mode="after")
    def _reject_unimplemented_goals(self):
        """Fail loudly on goals that nothing downstream honors yet.

        `minimize` and `maintain` appear in spec 6.6.5 (isotonic calibration,
        probabilistic feasibility), but no calibrator exists and every caller
        currently builds an all-maximize direction vector. Silently accepting
        `minimize` would optimize that objective in the wrong direction. Widen
        this validator only once the calibration layer actually lands.

        A `pair_threshold` field previously lived here and was never read by any
        caller; it was removed rather than left advertising spec 5.4's
        noise-aware pair construction, which is also unimplemented.
        """
        if self.goal != "maximize":
            raise NotImplementedError(
                f"ObjectiveSpec {self.name!r}: goal={self.goal!r} is specified but not "
                "implemented (no isotonic calibrator, no feasibility model). Only "
                "'maximize' is honored today."
            )
        if self.target_interval is not None:
            raise NotImplementedError(
                f"ObjectiveSpec {self.name!r}: target_interval requires the interval/"
                "constraint scoring path of spec 6.6.5, which is not implemented."
            )
        return self


class MultiObjectiveDatasetSpec(BaseModel):
    """A campaign spanning 2+ objectives measured on the same protein."""

    protein: str
    objectives: List[ObjectiveSpec]
    # Which dms_id's embedding file to use for ALL objectives. Must be one of the
    # dms_ids in `objectives` (the one that has a local embedding file on disk).
    embedding_dms_id: str
    embedding_model_id: str = "300m"
    naturalness_model_id: str = "E1-600m"
    expected_embedding_dim: int = 960


# NOTE: PTEN_HUMAN_Matreyek_2021 measures protein stability/abundance and is a
# good future candidate for a "maintain" (constraint-style) objective rather
# than a plain maximize/minimize target, once that scoring mode is wired up
# downstream. For now it is registered as "maximize" like everything else.
MULTIOBJECTIVE_DATASETS: Dict[str, MultiObjectiveDatasetSpec] = {
    "KCNJ2": MultiObjectiveDatasetSpec(
        protein="KCNJ2",
        objectives=[
            ObjectiveSpec(
                name="function",
                dms_id="KCNJ2_MOUSE_Coyote-Maestas_2022_function",
                goal="maximize",
            ),
            ObjectiveSpec(
                name="surface",
                dms_id="KCNJ2_MOUSE_Coyote-Maestas_2022_surface",
                goal="maximize",
            ),
        ],
        embedding_dms_id="KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    ),
    "PTEN": MultiObjectiveDatasetSpec(
        protein="PTEN",
        objectives=[
            ObjectiveSpec(
                name="phosphatase_activity",
                dms_id="PTEN_HUMAN_Mighell_2018",
                goal="maximize",
            ),
            ObjectiveSpec(
                name="stability",
                dms_id="PTEN_HUMAN_Matreyek_2021",
                goal="maximize",
            ),
        ],
        embedding_dms_id="PTEN_HUMAN_Mighell_2018",
    ),
    "S22A1": MultiObjectiveDatasetSpec(
        protein="S22A1",
        objectives=[
            ObjectiveSpec(
                name="activity",
                dms_id="S22A1_HUMAN_Yee_2023_activity",
                goal="maximize",
            ),
            ObjectiveSpec(
                name="abundance",
                dms_id="S22A1_HUMAN_Yee_2023_abundance",
                goal="maximize",
            ),
        ],
        embedding_dms_id="S22A1_HUMAN_Yee_2023_activity",
    ),
    # The three below were added to lift the acceptance suite above the paired
    # Wilcoxon signed-rank floor: at n=5 datasets the minimum achievable
    # two-sided p is 0.0625, so a 5-dataset design cannot clear alpha=0.05 at
    # ANY effect size. Six datasets reach 0.0312.
    "KCNE1": MultiObjectiveDatasetSpec(
        protein="KCNE1",
        objectives=[
            ObjectiveSpec(
                name="function",
                dms_id="KCNE1_HUMAN_Muhammad_2023_function",
                goal="maximize",
            ),
            ObjectiveSpec(
                name="expression",
                dms_id="KCNE1_HUMAN_Muhammad_2023_expression",
                goal="maximize",
            ),
        ],
        embedding_dms_id="KCNE1_HUMAN_Muhammad_2023_function",
    ),
    # NOTE: RASK is the only registered dataset containing multi-mutants (up to
    # doubles), so its seq_ids must be canonicalized through
    # `allele_set_to_seq_id` on BOTH the activity and embedding sides or the
    # join silently drops rows. See 260730_submit_new_dataset_jobs.py.
    "RASK": MultiObjectiveDatasetSpec(
        protein="RASK",
        objectives=[
            ObjectiveSpec(
                name="abundance",
                dms_id="RASK_HUMAN_Weng_2022_abundance",
                goal="maximize",
            ),
            ObjectiveSpec(
                name="binding",
                dms_id="RASK_HUMAN_Weng_2022_binding-DARPin_K55",
                goal="maximize",
            ),
        ],
        embedding_dms_id="RASK_HUMAN_Weng_2022_abundance",
    ),
    "OXDA": MultiObjectiveDatasetSpec(
        protein="OXDA",
        objectives=[
            ObjectiveSpec(
                name="activity",
                dms_id="OXDA_RHOTO_Vanella_2023_activity",
                goal="maximize",
            ),
            ObjectiveSpec(
                name="expression",
                dms_id="OXDA_RHOTO_Vanella_2023_expression",
                goal="maximize",
            ),
        ],
        embedding_dms_id="OXDA_RHOTO_Vanella_2023_activity",
    ),
}


# ############################################################################
# RESULT CONTAINER
# ############################################################################


@dataclass
class MultiObjectiveDataset:
    """Result of loading a MultiObjectiveDatasetSpec.

    All three of activity_df / embedding_series / naturalness_df share an
    identical, aligned index of seq_ids. This is a hard invariant, asserted in
    `load_multiobjective_dataset`.
    """

    activity_df: pd.DataFrame  # index=seq_id, one column per objective name
    embedding_series: pd.Series  # index=seq_id, values are 1-D np.ndarray
    naturalness_df: pd.DataFrame  # index=seq_id, one or more naturalness columns
    objectives: List[ObjectiveSpec]
    wt_sequence: str
    protein: str


# ############################################################################
# LOADING HELPERS
# ############################################################################


def _load_wt_sequence(dms_id: str) -> str:
    dms_metadata = get_dms_metadata()
    row = dms_metadata[dms_metadata["DMS_id"] == dms_id]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one metadata row for {dms_id}, found {len(row)}")
    return row["target_seq"].iloc[0]


def _load_embedding_df(
    embedding_dms_id: str, embedding_model_id: str, expected_embedding_dim: int
) -> pd.DataFrame:
    """Load the single embedding file that will serve ALL objectives of a campaign.

    NOTE: We deliberately do NOT call `folde.data._find_foldydata_embedding_file`
    or its foldydata fallback here. That helper matches embedding files by loose
    substring (`model_token in path.name.lower()`), and for a token like "300m"
    that matches both ESMC-300M (960-dim) and Profluent E1-300m (1024-dim) files
    in a fold's embed/ dir, with `_choose_preferred_file` (alphabetical sort)
    silently picking the wrong one. Since a multi-objective campaign requires a
    single shared feature space for ALL objectives, resolving the embedding
    source ambiguously would be a silent correctness bug. Instead we require an
    explicit local file at the expected ProteinGym-style path and hard-fail if
    it's missing or has the wrong dimensionality.
    """
    embedding_file_path = EMBEDDINGS_DIR / f"{embedding_dms_id}_embedding_{embedding_model_id}.csv"
    if not embedding_file_path.exists():
        raise FileNotFoundError(
            f"Expected explicit local embedding file for multi-objective loading, "
            f"but it does not exist: {embedding_file_path}"
        )
    embedding_df = pd.read_csv(embedding_file_path)
    _parse_embedding_columns_inplace(embedding_df)
    if "seq_id" not in embedding_df.columns or "embedding" not in embedding_df.columns:
        raise ValueError(
            f"Embedding file {embedding_file_path} missing required columns "
            f"'seq_id'/'embedding'; got {embedding_df.columns.tolist()}"
        )
    if len(embedding_df) == 0:
        raise ValueError(f"Embedding file {embedding_file_path} is empty")

    actual_dim = int(np.asarray(embedding_df["embedding"].iloc[0]).shape[0])
    if actual_dim != expected_embedding_dim:
        raise ValueError(
            f"Embedding file {embedding_file_path} has dimension {actual_dim}, expected "
            f"{expected_embedding_dim}. This likely means the wrong embedding model file "
            f"was resolved (e.g. a Profluent E1 file instead of ESMC) — refusing to load, "
            f"since mixing feature spaces across objectives must be impossible."
        )

    embedding_df["seq_id"] = embedding_df["seq_id"].apply(
        lambda x: maybe_modify_seq_id(embedding_dms_id, x)
    )
    embedding_df = embedding_df.set_index("seq_id", drop=False)
    return embedding_df


def _load_naturalness_df(embedding_dms_id: str, naturalness_model_id: str) -> pd.DataFrame:
    """Load the naturalness file shared by all objectives of a campaign.

    Mirrors the naturalness-loading logic in `folde.data.get_proteingym_dataset`
    (local file, else foldydata fallback via `_find_foldydata_naturalness_file`;
    ensemble pivot if a 'model' column with >1 unique value is present). Kept as
    a local copy rather than imported, since that logic lives inline inside
    `get_proteingym_dataset` and is not exposed as a standalone function.
    """
    naturalness_file_path = (
        NATURALNESS_DIR / f"{embedding_dms_id}_naturalness_{naturalness_model_id}.csv"
    )
    if not naturalness_file_path.exists():
        foldy_naturalness = _find_foldydata_naturalness_file(embedding_dms_id, naturalness_model_id)
        if not foldy_naturalness:
            raise FileNotFoundError(
                f"Naturalness file not found for {embedding_dms_id} / {naturalness_model_id}: "
                f"tried {naturalness_file_path} and foldydata fallback."
            )
        naturalness_file_path = foldy_naturalness

    naturalness_df = pd.read_csv(naturalness_file_path)
    if "seq_id" not in naturalness_df.columns:
        raise ValueError(
            f"Naturalness file {naturalness_file_path} missing 'seq_id' column; "
            f"got {naturalness_df.columns.tolist()}"
        )
    naturalness_df["seq_id"] = naturalness_df["seq_id"].apply(
        lambda x: maybe_modify_seq_id(embedding_dms_id, x)
    )
    if "wt_marginal" not in naturalness_df.columns:
        raise ValueError(
            f"Naturalness file {naturalness_file_path} missing 'wt_marginal' column; "
            f"got {naturalness_df.columns.tolist()}"
        )
    if any(naturalness_df["wt_marginal"] < 0):
        raise ValueError(f"wt_marginal for {embedding_dms_id} contains negative values")

    def safe_log(x: float) -> float:
        return np.log(max(x, 1e-20))

    if "model" in naturalness_df.columns and naturalness_df["model"].unique().size > 1:
        naturalness_df["log_wt_marginal"] = naturalness_df["wt_marginal"].apply(safe_log)
        naturalness_df["model"] = naturalness_df["model"].apply(lambda x: f"log_wt_marginal_{x}")
        naturalness_df = naturalness_df.pivot(
            index="seq_id", columns="model", values="log_wt_marginal"
        )
    else:
        naturalness_df = naturalness_df.set_index("seq_id", drop=False)
        naturalness_df["log_wt_marginal"] = naturalness_df["wt_marginal"].apply(safe_log)
        naturalness_df = naturalness_df.drop(columns=["wt_marginal"])
        # Keep only numeric naturalness columns (drop the passthrough 'seq_id' etc.),
        # matching the shape TorchMLPFewShotModel.pretrain expects (it cycles ensemble
        # members across naturalness DataFrame columns).
        naturalness_df = naturalness_df[["log_wt_marginal"]]

    return naturalness_df


def _load_objective_activity_series(objective: ObjectiveSpec) -> pd.Series:
    """Load one objective's DMS_score series, indexed by (normalized) seq_id."""
    dms_file_path = DMS_DIR / f"{objective.dms_id}.csv"
    if not dms_file_path.exists():
        raise FileNotFoundError(f"DMS activity file not found: {dms_file_path}")
    activity_df = pd.read_csv(dms_file_path)
    activity_df["seq_id"] = activity_df["mutant"].apply(
        lambda x: allele_set_to_seq_id(set(x.split(":")))
    )
    activity_df["seq_id"] = activity_df["seq_id"].apply(
        lambda x: maybe_modify_seq_id(objective.dms_id, x)
    )
    dupes = activity_df["seq_id"].duplicated()
    if dupes.any():
        logger.warning(
            f"Dropping {dupes.sum()} duplicate seq_ids from {objective.dms_id} activity data"
        )
        activity_df = activity_df[~dupes]
    activity_df = activity_df.set_index("seq_id", drop=False)
    series = activity_df["DMS_score"].rename(objective.name)
    return series


# ############################################################################
# PUBLIC API
# ############################################################################


def load_multiobjective_dataset(
    spec: MultiObjectiveDatasetSpec, restrict_to_shared: bool = True
) -> MultiObjectiveDataset:
    """Load and align a multi-objective dataset for a single protein.

    Args:
        spec: which protein/objectives/models to load.
        restrict_to_shared: if True, keep only seq_ids measured for EVERY
            objective (intersection). If False, keep the union of seq_ids
            measured for ANY objective (still intersected with embedding and
            naturalness coverage), with NaN in `activity_df` wherever a given
            objective lacks a measurement for that variant.

    Returns:
        A MultiObjectiveDataset with activity_df / embedding_series /
        naturalness_df all sharing an identical, aligned seq_id index.
    """
    if len(spec.objectives) < 2:
        raise ValueError(
            f"MultiObjectiveDatasetSpec for {spec.protein} must have >= 2 objectives, "
            f"got {len(spec.objectives)}"
        )
    if spec.embedding_dms_id not in {o.dms_id for o in spec.objectives}:
        raise ValueError(
            f"embedding_dms_id {spec.embedding_dms_id} is not among the objectives' dms_ids "
            f"for {spec.protein}"
        )

    wt_sequences = {o.dms_id: _load_wt_sequence(o.dms_id) for o in spec.objectives}
    distinct_wt_seqs = set(wt_sequences.values())
    if len(distinct_wt_seqs) != 1:
        raise ValueError(
            f"Objectives for {spec.protein} do not share a wild-type sequence: "
            f"{ {k: len(v) for k, v in wt_sequences.items()} }"
        )
    wt_sequence = distinct_wt_seqs.pop()

    embedding_df = _load_embedding_df(
        spec.embedding_dms_id, spec.embedding_model_id, spec.expected_embedding_dim
    )
    naturalness_df = _load_naturalness_df(spec.embedding_dms_id, spec.naturalness_model_id)

    objective_series = {o.name: _load_objective_activity_series(o) for o in spec.objectives}

    seq_ids_with_embedding = set(embedding_df.index)
    seq_ids_with_naturalness = set(naturalness_df.index)
    # Naturalness is a per-single-substitution table (L x 20), so intersecting
    # against it would drop every multi-mutant. That is not a hypothetical: it
    # silently discarded 22,946 of RASK's 27,814 variants (89%), leaving only
    # singles. Nothing is scored additively for higher-order variants -- there
    # is no such expansion anywhere in the codebase -- so the correct behavior
    # is the one `folde.data.get_proteingym_dataset` already uses: keep the
    # variant and leave naturalness NaN.
    #
    # This is safe because naturalness is consumed in exactly one place.
    # `TorchMLPFewShotModel.pretrain` masks NaN rows out explicitly
    # (`has_naturalness_data = ~naturalness_series.isna()`), and `.predict`
    # accepts a naturalness_df but never reads it -- predictions come from
    # embeddings alone.
    base_index = seq_ids_with_embedding

    if restrict_to_shared:
        final_ids = set(base_index)
        for series in objective_series.values():
            final_ids &= set(series.index)
    else:
        union_activity_ids: set = set()
        for series in objective_series.values():
            union_activity_ids |= set(series.index)
        final_ids = base_index & union_activity_ids

    invalid_seq_ids: List[str] = []
    valid_seq_ids: List[str] = []
    for seq_id in final_ids:
        error_msg = maybe_get_seq_id_error_message(wt_sequence, seq_id)
        if error_msg:
            invalid_seq_ids.append(seq_id)
        else:
            valid_seq_ids.append(seq_id)
    if invalid_seq_ids:
        example_invalid = ", ".join(sorted(invalid_seq_ids)[:5])
        logger.warning(
            f"Dropping {len(invalid_seq_ids)} invalid seq_ids for {spec.protein} "
            f"(e.g., {example_invalid})."
        )

    sorted_ids = sort_seq_id_list(wt_sequence, valid_seq_ids)
    final_index = pd.Index(sorted_ids, name="seq_id")

    if len(final_index) == 0:
        raise ValueError(
            f"No shared/aligned seq_ids found for multi-objective dataset {spec.protein}"
        )

    activity_df = pd.DataFrame(index=final_index)
    for name, series in objective_series.items():
        activity_df[name] = series.reindex(final_index)

    embedding_series = embedding_df.loc[final_index, "embedding"]
    # reindex, not .loc: multi-mutants legitimately have no naturalness row and
    # must come through as NaN rather than raising a KeyError.
    naturalness_df_final = naturalness_df.reindex(final_index)

    # Pretraining needs *some* naturalness signal per ensemble column. Total
    # absence means the naturalness file didn't resolve to this protein at all,
    # which is a silent-wrong-data bug rather than a legitimately sparse table.
    naturalness_coverage = float(naturalness_df_final.notna().any(axis=1).mean())
    if naturalness_coverage == 0.0:
        raise ValueError(
            f"No variant in multi-objective dataset {spec.protein} has naturalness data "
            f"({spec.naturalness_model_id}). The naturalness file likely resolved to the "
            f"wrong protein."
        )
    logger.info(
        f"{spec.protein}: {len(final_index)} variants, naturalness coverage "
        f"{naturalness_coverage:.1%} (multi-mutants carry NaN by design)"
    )

    # Hard invariant: all three must share an identical, aligned index.
    assert activity_df.index.equals(embedding_series.index)
    assert activity_df.index.equals(naturalness_df_final.index)

    return MultiObjectiveDataset(
        activity_df=activity_df,
        embedding_series=embedding_series,
        naturalness_df=naturalness_df_final,
        objectives=list(spec.objectives),
        wt_sequence=wt_sequence,
        protein=spec.protein,
    )


def get_available_multiobjective_datasets() -> List[str]:
    """Return registry keys whose underlying files actually exist on disk."""
    available = []
    for key, spec in MULTIOBJECTIVE_DATASETS.items():
        try:
            embedding_file_path = (
                EMBEDDINGS_DIR / f"{spec.embedding_dms_id}_embedding_{spec.embedding_model_id}.csv"
            )
            if not embedding_file_path.exists():
                continue
            if not all((DMS_DIR / f"{o.dms_id}.csv").exists() for o in spec.objectives):
                continue
            _find_foldydata_naturalness_file(spec.embedding_dms_id, spec.naturalness_model_id)
            naturalness_local = (
                NATURALNESS_DIR
                / f"{spec.embedding_dms_id}_naturalness_{spec.naturalness_model_id}.csv"
            )
            has_naturalness = naturalness_local.exists() or (
                _find_foldydata_naturalness_file(spec.embedding_dms_id, spec.naturalness_model_id)
                is not None
            )
            if not has_naturalness:
                continue
            available.append(key)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Error checking availability of multi-objective dataset {key}: {exc}")
    return available

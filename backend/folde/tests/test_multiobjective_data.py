"""Tests for folde.multiobjective_data."""

import numpy as np
import pytest

from folde.multiobjective_data import (
    MULTIOBJECTIVE_DATASETS,
    MultiObjectiveDatasetSpec,
    ObjectiveSpec,
    get_available_multiobjective_datasets,
    load_multiobjective_dataset,
)

# Datasets whose embedding/naturalness files are expected to be present, with
# their known aligned sizes. This doubles as the strictness gate: a key listed
# here MUST load, so a broken or missing file is a hard failure. Registry
# entries absent from this map are ones whose ESM jobs are still in flight --
# they skip rather than fail, and gain a count here once their embeddings land.
EXPECTED_SHARED_VARIANT_COUNTS = {
    "KCNJ2": 6789,
    "PTEN": 4839,
    "S22A1": 9715,
    "KCNE1": 2312,
    "RASK": 23072,
    "OXDA": 6387,
}

PENDING = sorted(set(MULTIOBJECTIVE_DATASETS) - set(EXPECTED_SHARED_VARIANT_COUNTS))


def test_registry_datasets_are_available_on_disk():
    available = get_available_multiobjective_datasets()
    for key in EXPECTED_SHARED_VARIANT_COUNTS:
        assert key in available, f"{key} was not detected as available"
    # Availability must never claim a dataset that isn't registered.
    assert set(available) <= set(MULTIOBJECTIVE_DATASETS)
    if PENDING:
        print(f"pending (embeddings not yet on disk): {PENDING}")


def _skip_if_pending(key):
    if key in PENDING:
        pytest.skip(f"{key}: embeddings not yet generated (ESM job pending)")


@pytest.mark.parametrize("key", list(MULTIOBJECTIVE_DATASETS.keys()))
def test_registry_entries_resolve_and_load(key):
    _skip_if_pending(key)
    spec = MULTIOBJECTIVE_DATASETS[key]
    dataset = load_multiobjective_dataset(spec, restrict_to_shared=True)

    assert dataset.protein == key
    assert len(dataset.objectives) == len(spec.objectives)
    assert isinstance(dataset.wt_sequence, str) and len(dataset.wt_sequence) > 0

    # Index alignment invariant.
    assert dataset.activity_df.index.equals(dataset.embedding_series.index)
    assert dataset.activity_df.index.equals(dataset.naturalness_df.index)
    assert dataset.activity_df.index.is_unique

    # Activity df has one column per objective.
    expected_columns = {o.name for o in spec.objectives}
    assert set(dataset.activity_df.columns) == expected_columns

    # restrict_to_shared=True means no NaNs remain in any objective column.
    assert not dataset.activity_df.isna().any().any()

    # Embedding dimension check.
    sample_embedding = dataset.embedding_series.iloc[0]
    assert isinstance(sample_embedding, np.ndarray)
    assert sample_embedding.shape[0] == 960

    # Shared-variant count sanity check (within ~1% of the documented value).
    n_variants = len(dataset.activity_df)
    expected = EXPECTED_SHARED_VARIANT_COUNTS[key]
    tolerance = max(1, int(expected * 0.01))
    print(f"{key}: loaded {n_variants} shared variants (expected ~{expected})")
    assert abs(n_variants - expected) <= tolerance, (
        f"{key}: got {n_variants} shared variants, expected ~{expected} " f"(tolerance {tolerance})"
    )


@pytest.mark.parametrize("key", list(MULTIOBJECTIVE_DATASETS.keys()))
def test_restrict_to_shared_false_gives_union_with_nans(key):
    _skip_if_pending(key)
    spec = MULTIOBJECTIVE_DATASETS[key]
    shared = load_multiobjective_dataset(spec, restrict_to_shared=True)
    union = load_multiobjective_dataset(spec, restrict_to_shared=False)

    assert len(union.activity_df) >= len(shared.activity_df)
    assert union.activity_df.index.equals(union.embedding_series.index)
    assert union.activity_df.index.equals(union.naturalness_df.index)
    # The union set should have at least some missing values, since not every
    # embedding-covered variant is measured for every objective.
    assert union.activity_df.isna().any().any()


def test_multi_mutants_are_retained_with_nan_naturalness():
    """Multi-mutants must survive loading even though naturalness has no row for them.

    Naturalness is a per-single-substitution table (L x 20). Intersecting against
    it silently dropped 22,946 of RASK's 27,814 variants (89%), leaving a
    singles-only dataset. Nothing scores higher-order variants additively, so the
    correct behavior -- matching folde.data.get_proteingym_dataset -- is to keep
    the variant with NaN naturalness, which pretrain masks out and predict
    ignores entirely.
    """
    if "RASK" in PENDING:
        pytest.skip("RASK: embeddings not yet generated (ESM job pending)")

    dataset = load_multiobjective_dataset(MULTIOBJECTIVE_DATASETS["RASK"], restrict_to_shared=True)

    multi_mutant_ids = [s for s in dataset.activity_df.index if "_" in s]
    assert len(multi_mutant_ids) > 10000, (
        f"expected RASK's double mutants to survive loading, got only "
        f"{len(multi_mutant_ids)} multi-mutants out of {len(dataset.activity_df)}"
    )

    # Those multi-mutants are exactly the rows with no naturalness.
    has_naturalness = dataset.naturalness_df.notna().any(axis=1)
    assert not has_naturalness.loc[multi_mutant_ids].any()
    # ...and singles still do have it, so the table resolved to the right protein.
    single_ids = [s for s in dataset.activity_df.index if "_" not in s]
    assert has_naturalness.loc[single_ids].all()

    # Embeddings are per-sequence and must be present for every retained variant.
    assert dataset.embedding_series.notna().all()


def test_wrong_dimension_embedding_source_raises():
    """A spec whose embedding_dms_id resolves to a non-960-dim file must raise,
    rather than silently loading (guards against the known _find_foldydata_embedding_file
    substring-matching bug described in the module docstring)."""
    base_spec = MULTIOBJECTIVE_DATASETS["KCNJ2"]
    bad_spec = MultiObjectiveDatasetSpec(
        protein="KCNJ2",
        objectives=list(base_spec.objectives),
        embedding_dms_id=base_spec.embedding_dms_id,
        embedding_model_id=base_spec.embedding_model_id,
        naturalness_model_id=base_spec.naturalness_model_id,
        expected_embedding_dim=1024,  # Wrong on purpose: real file is 960-dim.
    )
    with pytest.raises(ValueError, match="dimension"):
        load_multiobjective_dataset(bad_spec, restrict_to_shared=True)


def test_missing_local_embedding_file_raises_not_silently_falls_back():
    """A dms_id with no local <dms_id>_embedding_<model>.csv file must raise
    FileNotFoundError rather than silently falling back to the fuzzy foldydata
    matcher (which could return the wrong feature space)."""
    bad_spec = MultiObjectiveDatasetSpec(
        protein="KCNJ2",
        objectives=list(MULTIOBJECTIVE_DATASETS["KCNJ2"].objectives),
        embedding_dms_id="KCNJ2_MOUSE_Coyote-Maestas_2022_surface",  # no local embedding file
        embedding_model_id="300m",
        naturalness_model_id="E1-600m",
        expected_embedding_dim=960,
    )
    with pytest.raises(FileNotFoundError):
        load_multiobjective_dataset(bad_spec, restrict_to_shared=True)

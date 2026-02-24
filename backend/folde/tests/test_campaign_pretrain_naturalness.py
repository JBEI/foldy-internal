import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

# `folde.campaign` imports `app.helpers.sequence_util`, which imports dnachisel.
# This test does not exercise dnachisel paths, so a tiny stub keeps test setup lightweight.
if "dnachisel" not in sys.modules:
    dnachisel_stub = types.ModuleType("dnachisel")
    dnachisel_stub.biotools = types.SimpleNamespace()
    sys.modules["dnachisel"] = dnachisel_stub

from folde.campaign import simulate_campaign
from folde.types import FolDEModelConfig, SimulationResult


def _make_mock_dataset(naturalness_offset: float):
    seq_ids = [f"A{i}G" for i in range(1, 13)]
    wt_aa_seq = "M" * 32

    naturalness_df = pd.DataFrame(
        {
            "seq_id": seq_ids,
            "log_wt_marginal": np.arange(len(seq_ids), dtype=float) + naturalness_offset,
        }
    ).set_index("seq_id", drop=False)

    embedding_df = pd.DataFrame(
        {
            "seq_id": seq_ids,
            "embedding": [
                np.array([float(i), float(i + 1), float(i + 2)]) for i in range(len(seq_ids))
            ],
        }
    ).set_index("seq_id", drop=False)

    activity_df = pd.DataFrame(
        {
            "seq_id": seq_ids,
            "mutant": seq_ids,
            "DMS_score": np.linspace(0.0, 1.0, num=len(seq_ids)),
        }
    ).set_index("seq_id", drop=False)

    category_df = pd.DataFrame(index=activity_df.index)
    return wt_aa_seq, naturalness_df, embedding_df, activity_df, category_df


@patch("folde.campaign.run_single_sim_parallel")
@patch("folde.campaign.get_proteingym_dataset")
def test_simulate_campaign_supports_separate_pretrain_naturalness_model(
    mock_get_proteingym_dataset, mock_run_single_sim_parallel
):
    model_config = FolDEModelConfig(
        name="split-naturalness-config",
        naturalness_model_id="zero-shot-nat",
        few_shot_pretrain_naturalness_model_id="pretrain-nat",
        embedding_model_id="300m",
        zero_shot_model_name="NaturalnessZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="TorchMLPFewShotModel",
        few_shot_model_params={},
    )

    zero_shot_dataset = _make_mock_dataset(naturalness_offset=0.0)
    pretrain_dataset = _make_mock_dataset(naturalness_offset=1000.0)
    mock_get_proteingym_dataset.side_effect = [zero_shot_dataset, pretrain_dataset]
    mock_run_single_sim_parallel.return_value = SimulationResult(
        rounds=0,
        variant_pool_size=0,
        round_metrics=[],
        mutant_metrics=[],
    )

    simulate_campaign(
        dms_id="TEST_DMS",
        round_size=1,
        number_of_simulations=1,
        config_list=[model_config],
        max_rounds=1,
        num_workers=1,
    )

    assert mock_get_proteingym_dataset.call_count == 2
    assert mock_get_proteingym_dataset.call_args_list[0].args[:3] == (
        "TEST_DMS",
        "300m",
        "zero-shot-nat",
    )
    assert mock_get_proteingym_dataset.call_args_list[1].args[:3] == (
        "TEST_DMS",
        "300m",
        "pretrain-nat",
    )

    run_args, run_kwargs = mock_run_single_sim_parallel.call_args
    zero_shot_naturalness_df = run_args[3]
    pretrain_naturalness_df = run_kwargs["pretrain_naturalness_df"]

    assert float(zero_shot_naturalness_df["log_wt_marginal"].iloc[0]) == 0.0
    assert float(pretrain_naturalness_df["log_wt_marginal"].iloc[0]) == 1000.0

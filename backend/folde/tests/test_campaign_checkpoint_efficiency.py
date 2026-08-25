from pathlib import Path
from unittest.mock import patch

from folde.campaign import simulate_campaigns_with_config_checkpoints
from folde.types import CampaignResult, FolDEModelConfig, SingleConfigCampaignResult


def _config(name: str) -> FolDEModelConfig:
    return FolDEModelConfig(
        name=name,
        naturalness_model_id="naturalness",
        embedding_model_id="embedding",
        zero_shot_model_name="RandomZeroShotModel",
        zero_shot_model_params={},
        few_shot_model_name="RandomFewShotModel",
        few_shot_model_params={},
    )


def test_checkpoint_runner_is_dms_first_and_shares_dataset_cache(tmp_path: Path) -> None:
    configs = [_config("config-a"), _config("config-b")]
    call_order: list[tuple[str, str]] = []
    cache_ids_by_dms: dict[str, list[int]] = {}

    def fake_simulate_campaign(
        dms_id: str,
        *,
        config_list: list[FolDEModelConfig],
        dataset_cache: dict[object, object],
        **kwargs: object,
    ) -> CampaignResult:
        config = config_list[0]
        call_order.append((dms_id, config.name))
        cache_ids_by_dms.setdefault(dms_id, []).append(id(dataset_cache))
        return CampaignResult(
            dms_id=dms_id,
            round_size=int(kwargs["round_size"]),
            number_of_simulations=int(kwargs["number_of_simulations"]),
            activity_column="DMS_score",
            min_activity=0.0,
            median_activity=0.0,
            max_activity=0.0,
            max_rounds=1,
            random_seed=42,
            config_results=[SingleConfigCampaignResult(config=config, simulation_results=[])],
        )

    with patch("folde.campaign.simulate_campaign", side_effect=fake_simulate_campaign):
        simulate_campaigns_with_config_checkpoints(
            eval_prefix="test",
            dms_ids=["dms-1", "dms-2"],
            config_list=configs,
            checkpoint_dir=str(tmp_path),
            round_size=1,
            number_of_simulations=1,
        )

    assert call_order == [
        ("dms-1", "config-a"),
        ("dms-1", "config-b"),
        ("dms-2", "config-a"),
        ("dms-2", "config-b"),
    ]
    assert len(set(cache_ids_by_dms["dms-1"])) == 1
    assert len(set(cache_ids_by_dms["dms-2"])) == 1
    assert cache_ids_by_dms["dms-1"][0] != cache_ids_by_dms["dms-2"][0]

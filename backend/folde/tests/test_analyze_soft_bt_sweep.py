import json
from pathlib import Path

import pytest

from folde.scripts.analyze_soft_bt_sweep import _arm_metadata, load_round_records


def _config(name: str, *, temperature: float | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "standardized_mse_weight": 0.0,
        "bt_activity_difference_weighting": False,
    }
    if temperature is not None:
        params["bt_soft_target_temperature"] = temperature
    return {
        "name": name,
        "one_mutation_at_a_time": True,
        "few_shot_model_params": params,
    }


def _checkpoint(config: dict[str, object], first_round: list[str]) -> dict[str, object]:
    mutants = [
        {
            "seq_id": seq_id,
            "round_found": 1,
            "percentile": 0.95 - index * 0.01,
        }
        for index, seq_id in enumerate(first_round)
    ]
    mutants.extend(
        [
            {"seq_id": "A1V_C3D", "round_found": 2, "percentile": 0.99},
            {"seq_id": "B2C_C3D", "round_found": 2, "percentile": 0.80},
            {"seq_id": "A1V_D4E", "round_found": 3, "percentile": 0.995},
            {"seq_id": "B2C_D4E", "round_found": 3, "percentile": 0.70},
        ]
    )
    return {
        "name": "test",
        "campaign_results": [
            {
                "dms_id": "SYNTHETIC_DMS",
                "round_size": 2,
                "number_of_simulations": 1,
                "max_rounds": 3,
                "config_results": [
                    {
                        "config": config,
                        "simulation_results": [
                            {
                                "variant_pool_size": 100,
                                "mutant_metrics": mutants,
                                "round_metrics": [
                                    {
                                        "round_num": round_num,
                                        "model_spearman": 0.1 * round_num,
                                        "misc": {"held_out_activity_spearman": 0.2 * round_num},
                                    }
                                    for round_num in range(1, 4)
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _write_checkpoint(
    directory: Path,
    prefix: str,
    config: dict[str, object],
    first_round: list[str],
) -> None:
    path = directory / f"{prefix}_{config['name']}.json"
    path.write_text(json.dumps(_checkpoint(config, first_round)))


def test_arm_metadata_distinguishes_hard_raw_gap_and_soft_targets() -> None:
    hard = _arm_metadata(_config("hard"))
    raw_config = _config("raw")
    raw_config["few_shot_model_params"]["bt_activity_difference_weighting"] = True
    raw = _arm_metadata(raw_config)
    soft = _arm_metadata(_config("soft", temperature=0.25))

    assert hard["family"] == "hard"
    assert raw["family"] == "raw_gap"
    assert soft["family"] == "soft"
    assert soft["arm_short_label"] == "T=0.25"


def test_load_round_records_validates_pairing_and_single_round_one(tmp_path: Path) -> None:
    prefix = "paired"
    first_round = ["A1V", "B2C"]
    _write_checkpoint(tmp_path, prefix, _config("hard"), first_round)
    _write_checkpoint(tmp_path, prefix, _config("soft", temperature=0.25), first_round)

    records, arms = load_round_records(tmp_path, prefix, control_arm="hard")

    assert len(records) == 6
    assert arms["arm"].tolist() == ["hard", "soft"]
    assert records.groupby("round_num").size().to_dict() == {1: 2, 2: 2, 3: 2}


def test_load_round_records_rejects_round_one_multi_mutants(tmp_path: Path) -> None:
    prefix = "invalid"
    _write_checkpoint(tmp_path, prefix, _config("hard"), ["A1V", "B2C"])
    _write_checkpoint(
        tmp_path,
        prefix,
        _config("soft", temperature=0.25),
        ["A1V_C3D", "B2C"],
    )

    with pytest.raises(ValueError, match="non-single mutant"):
        load_round_records(tmp_path, prefix, control_arm="hard")

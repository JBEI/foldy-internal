import pytest

from folde.scripts.run_soft_bt_sweep import (
    LOCAL_MULTI_MUTANT_DMS_IDS,
    build_soft_bt_configs,
    configs_for_benchmark,
    validate_confidence_floors,
    validate_temperatures,
)


def test_local_multi_mutant_suite_covers_all_ready_datasets() -> None:
    assert LOCAL_MULTI_MUTANT_DMS_IDS == [
        "SPG1_STRSG_Olson_2014",
        "GRB2_HUMAN_Faure_2021",
        "PABP_YEAST_Melamed_2013",
        "PHOT_CHLRE_Chen_2023",
        "SPG1_STRSG_Wu_2016",
        "GFP_AEQVI_Sarkisyan_2016",
        "Q8WTC7_9CNID_Somermeyer_2022",
        "Q6WV12_9MAXI_Somermeyer_2022",
        "D7PM05_CLYGR_Somermeyer_2022",
        "RASK_HUMAN_Weng_2022_abundance",
        "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
    ]


def test_build_soft_bt_configs_constructs_paired_ablation_grid() -> None:
    configs = build_soft_bt_configs([0.25, 0.5, 1.0], [0.2], device="cpu")

    assert [config.name for config in configs] == [
        "E1E1-300m-BT-hard",
        "E1E1-300m-BT-rawgap",
        "E1E1-300m-SoftBT-t0p25",
        "E1E1-300m-SoftBT-t0p25-f0p2",
        "E1E1-300m-SoftBT-t0p5",
        "E1E1-300m-SoftBT-t0p5-f0p2",
        "E1E1-300m-SoftBT-t1",
        "E1E1-300m-SoftBT-t1-f0p2",
    ]

    hard_params = configs[0].few_shot_model_params
    assert hard_params["standardized_mse_weight"] == pytest.approx(0.0)
    assert hard_params["bt_activity_difference_weighting"] is False
    assert "bt_soft_target_temperature" not in hard_params

    raw_gap_params = configs[1].few_shot_model_params
    assert raw_gap_params["bt_activity_difference_weighting"] is True
    assert "bt_soft_target_temperature" not in raw_gap_params

    for config in configs[2:]:
        params = config.few_shot_model_params
        assert params["device"] == "cpu"
        assert params["standardized_mse_weight"] == pytest.approx(0.0)
        assert params["bt_activity_difference_weighting"] is False
        assert params["bt_soft_target_temperature"] in (0.25, 0.5, 1.0)
        if "-f" in config.name:
            assert params["bt_soft_target_confidence_floor"] == pytest.approx(0.2)
        else:
            assert "bt_soft_target_confidence_floor" not in params


def test_build_soft_bt_configs_can_omit_controls_and_confidence_arms() -> None:
    configs = build_soft_bt_configs(
        [0.5],
        [],
        device="auto",
        include_hard_control=False,
        include_raw_gap_control=False,
    )

    assert len(configs) == 1
    assert configs[0].name == "E1E1-300m-SoftBT-t0p5"
    assert "device" not in configs[0].few_shot_model_params


def test_configs_for_benchmark_constrains_only_multi_mutant_campaigns() -> None:
    configs = build_soft_bt_configs([0.5], [], device="cpu")

    single_configs = configs_for_benchmark(configs, "single")
    multi_configs = configs_for_benchmark(configs, "multi")

    assert all(not config.one_mutation_at_a_time for config in configs)
    assert all(not config.one_mutation_at_a_time for config in single_configs)
    assert all(config.one_mutation_at_a_time for config in multi_configs)
    assert multi_configs[0] is not configs[0]


def test_configs_for_benchmark_rejects_unknown_benchmark() -> None:
    configs = build_soft_bt_configs([0.5], [], device="cpu")

    with pytest.raises(ValueError, match="Unknown benchmark"):
        configs_for_benchmark(configs, "invalid")


@pytest.mark.parametrize("temperatures", [[], [0.0], [-0.5], [float("nan")], [0.5, 0.5]])
def test_validate_temperatures_rejects_invalid_grids(temperatures: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_temperatures(temperatures)


@pytest.mark.parametrize("floors", [[-0.1], [1.1], [float("inf")], [0.2, 0.2]])
def test_validate_confidence_floors_rejects_invalid_grids(floors: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_confidence_floors(floors)

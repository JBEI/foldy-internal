from folde.scripts.analyze_bt_mse_weight_sweep import (
    CampaignOutcome,
    outcomes_from_evaluation,
    paired_deltas_from_control,
    render_report,
    summarize_outcomes,
)
from folde.scripts.run_bt_mse_weight_sweep import (
    DEFAULT_MSE_WEIGHTS,
    MULTI_MUTANT_DMS_IDS,
    SINGLE_MUTANT_DMS_IDS,
    build_sweep_config,
    build_sweep_configs,
    validate_weights,
)
from folde.types import (
    CampaignResult,
    ModelEvaluation,
    MutantMetrics,
    SimulationResult,
    SingleConfigCampaignResult,
)


def test_sweep_configs_hold_representation_and_validation_protocol_fixed() -> None:
    configs = build_sweep_configs(DEFAULT_MSE_WEIGHTS, device="cuda")

    assert len(configs) == len(DEFAULT_MSE_WEIGHTS)
    assert len({config.name for config in configs}) == len(configs)
    for config, weight in zip(configs, DEFAULT_MSE_WEIGHTS, strict=True):
        params = config.few_shot_model_params
        assert config.naturalness_model_id == "E1-600m_melted"
        assert config.few_shot_pretrain_naturalness_model_id == "E1-600m_melted"
        assert config.embedding_model_id == "300m"
        assert params["standardized_mse_weight"] == weight
        assert params["use_mse_loss"] is False
        assert params["do_holdout_validation"] is True
        assert params["do_validation_with_pair_fraction"] is None
        assert params["bt_activity_difference_weighting"] is False


def test_sweep_uses_the_complete_paper_dataset_lists() -> None:
    assert len(SINGLE_MUTANT_DMS_IDS) == 17
    assert len(MULTI_MUTANT_DMS_IDS) == 3
    assert MULTI_MUTANT_DMS_IDS == [
        "SPG1_STRSG_Olson_2014",
        "GRB2_HUMAN_Faure_2021",
        "PABP_YEAST_Melamed_2013",
    ]


def test_weight_validation_rejects_duplicates_and_out_of_range() -> None:
    try:
        validate_weights([0.1, 0.1])
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("Duplicate weights should fail")

    try:
        build_sweep_config(1.01)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("Out-of-range weights should fail")


def test_outcomes_filter_to_requested_round() -> None:
    config = build_sweep_config(0.2, device="cpu")
    simulation = SimulationResult(
        rounds=3,
        variant_pool_size=100,
        round_metrics=[],
        mutant_metrics=[
            MutantMetrics(
                seq_id="A1G",
                round_found=1,
                activity=1.0,
                predicted_activity=0.5,
                percentile=0.995,
                relevant_mutants=[],
            ),
            MutantMetrics(
                seq_id="A2G",
                round_found=3,
                activity=0.5,
                predicted_activity=0.4,
                percentile=0.95,
                relevant_mutants=[],
            ),
        ],
    )
    evaluation = ModelEvaluation(
        name="test",
        campaign_results=[
            CampaignResult(
                dms_id="example",
                round_size=1,
                number_of_simulations=1,
                activity_column="DMS_score",
                min_activity=0.0,
                median_activity=0.5,
                max_activity=1.0,
                max_rounds=3,
                random_seed=42,
                config_results=[
                    SingleConfigCampaignResult(config=config, simulation_results=[simulation])
                ],
            )
        ],
    )

    round_two = outcomes_from_evaluation(evaluation, benchmark="single", through_round=2)
    round_three = outcomes_from_evaluation(evaluation, benchmark="single", through_round=3)

    assert round_two[0].top_10pct_hits == 1
    assert round_two[0].found_top_1pct == 1
    assert round_three[0].top_10pct_hits == 2


def test_report_selects_macro_top1_winner_and_uses_paired_control() -> None:
    outcomes = [
        CampaignOutcome("single", 0.0, "single-a", 0, 1, 0),
        CampaignOutcome("multi", 0.0, "multi-a", 0, 2, 0),
        CampaignOutcome("single", 0.2, "single-a", 0, 3, 1),
        CampaignOutcome("multi", 0.2, "multi-a", 0, 4, 1),
    ]

    deltas = paired_deltas_from_control(outcomes)
    summaries = summarize_outcomes(outcomes)
    report = render_report(outcomes, summaries, through_round=3)

    assert deltas[("single", 0.2)] == (2.0, 1.0)
    assert deltas[("multi", 0.2)] == (2.0, 1.0)
    assert "Development winner: w=0.2" in report

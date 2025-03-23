from pydantic import BaseModel
from typing import Dict, Any, List


class FolDEModelConfig(BaseModel):
    name: str
    naturalness_model_id: str
    embedding_model_id: str
    zero_shot_model_name: str
    zero_shot_model_params: Dict[str, Any]
    few_shot_model_name: str
    few_shot_model_params: Dict[str, Any]


class MutantMetrics(BaseModel):
    """Stores dense information about each mutants tested in the simulation."""

    seq_id: str
    round_found: int
    activity: float
    predicted_activity: float
    percentile: float
    relevant_mutants: List[str]


class RoundMetrics(BaseModel):
    """Stores expensive-to-compute per-round metrics, such as model characterization."""

    round_num: int
    model_spearman: float
    misc: Dict[str, Any]


class SimulationResult(BaseModel):
    rounds: int
    variant_pool_size: int
    round_metrics: List[RoundMetrics]
    mutant_metrics: List[MutantMetrics]


class SingleConfigCampaignResult(BaseModel):
    config: FolDEModelConfig
    simulation_results: List[SimulationResult]


class CampaignResult(BaseModel):
    dms_id: str
    round_size: int
    number_of_simulations: int
    activity_column: str
    max_rounds: int
    random_seed: int
    config_results: List[SingleConfigCampaignResult]


class ModelEvaluation(BaseModel):
    name: str
    campaign_results: List[CampaignResult]

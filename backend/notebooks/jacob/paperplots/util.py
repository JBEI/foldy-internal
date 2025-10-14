"""Utility functions for paper plots."""


def get_random_baseline_number_of_top_performers(
    n_mutants_per_round: int, n_rounds: int, percentile: float
) -> float:
    """
    Calculate the expected number of top performers under random selection.

    Args:
        n_mutants_per_round: Number of mutants selected per round
        n_rounds: Number of rounds
        percentile: Percentile threshold (e.g., 0.10 for top 10%)

    Returns:
        Expected number of mutants in the top percentile
    """
    total_mutants = n_mutants_per_round * n_rounds
    return total_mutants * percentile


def get_random_baseline_probability_of_top_performer(
    n_mutants_per_round: int, n_rounds: int, percentile: float
) -> float:
    """
    Calculate the probability of finding at least one top performer under random selection.

    Args:
        n_mutants_per_round: Number of mutants selected per round
        n_rounds: Number of rounds
        percentile: Percentile threshold (e.g., 0.01 for top 1%)

    Returns:
        Probability of finding at least one mutant in the top percentile
    """
    total_mutants = n_mutants_per_round * n_rounds
    # P(finding at least one) = 1 - P(missing all)
    # P(missing all) = (1 - percentile)^total_mutants
    prob_missing_all = (1 - percentile) ** total_mutants
    return 1 - prob_missing_all

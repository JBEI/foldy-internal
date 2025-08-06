import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


def parse_oligo_name(sequence_name: str) -> Dict[str, Union[str, bool]]:
    """
    Parse an oligo sequence name to extract components.

    Args:
        sequence_name: The sequence name (e.g., "AP_OplR.W10L.F1" or "AP_OplR.W10.R")

    Returns:
        Dictionary with parsed components:
        - prefix: The protein/construct prefix (e.g., "AP_OplR")
        - allele_id: The mutation identifier (e.g., "W10L" or "W10")
        - direction: "forward" or "reverse"
        - primer_type: "F1" for forward, "R" for reverse
        - is_forward: Boolean indicating if it's a forward primer
        - is_reverse: Boolean indicating if it's a reverse primer
        - original_name: The original sequence name
    """

    # Pattern for forward primers: <prefix>.<allele_id>.F1
    forward_pattern = r"^([^.]+)\.([^.]+)\.F1$"

    # Pattern for reverse primers: <prefix>.<allele_id>.R
    reverse_pattern = r"^([^.]+)\.([^.]+)\.R$"

    result = {
        "original_name": sequence_name,
        "prefix": None,
        "allele_id": None,
        "direction": None,
        "primer_type": None,
        "is_forward": False,
        "is_reverse": False,
    }

    # Try forward pattern first
    forward_match = re.match(forward_pattern, sequence_name)
    if forward_match:
        result["prefix"] = forward_match.group(1)
        result["allele_id"] = forward_match.group(2)
        result["direction"] = "forward"
        result["primer_type"] = "F1"
        result["is_forward"] = True
        return result

    # Try reverse pattern
    reverse_match = re.match(reverse_pattern, sequence_name)
    if reverse_match:
        result["prefix"] = reverse_match.group(1)
        result["allele_id"] = reverse_match.group(2)
        result["direction"] = "reverse"
        result["primer_type"] = "R"
        result["is_reverse"] = True
        return result

    # If no pattern matches, return the result with None values
    return result


def classify_oligos(df: pd.DataFrame, sequence_name_col: str = "Sequence Name") -> pd.DataFrame:
    """
    Classify oligos in a dataframe and add parsed information as new columns.

    Args:
        df: DataFrame containing oligo information
        sequence_name_col: Name of the column containing sequence names

    Returns:
        DataFrame with additional columns for parsed oligo information
    """

    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    # Parse each sequence name
    parsed_data = []
    for seq_name in df[sequence_name_col]:
        parsed_data.append(parse_oligo_name(seq_name))

    # Convert to DataFrame and merge with original
    parsed_df = pd.DataFrame(parsed_data)

    # Add the parsed columns to the result dataframe
    for col in parsed_df.columns:
        if col != "original_name":  # Don't duplicate the original name
            result_df[col] = parsed_df[col]

    return result_df


def get_forward_primers(df: pd.DataFrame, sequence_name_col: str = "Sequence Name") -> pd.DataFrame:
    """
    Filter dataframe to return only forward primers.

    Args:
        df: DataFrame containing oligo information
        sequence_name_col: Name of the column containing sequence names

    Returns:
        DataFrame containing only forward primers
    """
    classified_df = classify_oligos(df, sequence_name_col)
    return classified_df[classified_df["is_forward"] == True]


def get_reverse_primers(df: pd.DataFrame, sequence_name_col: str = "Sequence Name") -> pd.DataFrame:
    """
    Filter dataframe to return only reverse primers.

    Args:
        df: DataFrame containing oligo information
        sequence_name_col: Name of the column containing sequence names

    Returns:
        DataFrame containing only reverse primers
    """
    classified_df = classify_oligos(df, sequence_name_col)
    return classified_df[classified_df["is_reverse"] == True]


def get_primers_by_prefix(
    df: pd.DataFrame, prefix: str, sequence_name_col: str = "Sequence Name"
) -> pd.DataFrame:
    """
    Filter dataframe to return primers for a specific protein/construct prefix.

    Args:
        df: DataFrame containing oligo information
        prefix: The protein/construct prefix to filter by (e.g., "AP_OplR")
        sequence_name_col: Name of the column containing sequence names

    Returns:
        DataFrame containing only primers for the specified prefix
    """
    classified_df = classify_oligos(df, sequence_name_col)
    return classified_df[classified_df["prefix"] == prefix]


def get_primer_pairs(
    df: pd.DataFrame, sequence_name_col: str = "Sequence Name"
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Group primers into forward/reverse pairs by prefix and allele_id.

    Args:
        df: DataFrame containing oligo information
        sequence_name_col: Name of the column containing sequence names

    Returns:
        Nested dictionary: {prefix: {allele_id: {'forward': Series, 'reverse': Series}}}
    """
    classified_df = classify_oligos(df, sequence_name_col)

    pairs = {}

    # Group by prefix and allele_id
    for prefix in classified_df["prefix"].dropna().unique():
        pairs[prefix] = {}
        prefix_df = classified_df[classified_df["prefix"] == prefix]

        for allele_id in prefix_df["allele_id"].dropna().unique():
            allele_df = prefix_df[prefix_df["allele_id"] == allele_id]

            forward_primers = allele_df[allele_df["is_forward"] == True]
            reverse_primers = allele_df[allele_df["is_reverse"] == True]

            pairs[prefix][allele_id] = {
                "forward": forward_primers.iloc[0] if len(forward_primers) > 0 else None,
                "reverse": reverse_primers.iloc[0] if len(reverse_primers) > 0 else None,
            }

    return pairs


def summarize_oligos(df: pd.DataFrame, sequence_name_col: str = "Sequence Name") -> pd.DataFrame:
    """
    Create a summary of oligos by prefix and direction.

    Args:
        df: DataFrame containing oligo information
        sequence_name_col: Name of the column containing sequence names

    Returns:
        DataFrame with summary statistics
    """
    classified_df = classify_oligos(df, sequence_name_col)

    summary = (
        classified_df.groupby(["prefix", "direction"])
        .agg({"allele_id": "count", "Sequence": "count"})
        .rename(columns={"allele_id": "count", "Sequence": "sequence_count"})
    )

    return summary.reset_index()


# Example usage and testing
if __name__ == "__main__":
    # Test the parsing function
    test_names = [
        "AP_OplR.W10L.F1",
        "AP_OplR.W10.R",
        "GAH_DBAT.S23T.F1",
        "ID_CUS.I46F.F1",
        "LK_BorAT.K360A.F1",
        "PW_CYP90.G56F.F1",
    ]

    print("Testing oligo name parsing:")
    for name in test_names:
        parsed = parse_oligo_name(name)
        print(f"{name} -> {parsed}")

    print("\n" + "=" * 50 + "\n")

    # Example of how to use with a dataframe
    sample_data = {
        "Sequence Name": [
            "AP_OplR.W10L.F1",
            "AP_OplR.W10.R",
            "AP_OplR.S51D.F1",
            "AP_OplR.S51.R",
            "GAH_DBAT.S23T.F1",
            "GAH_DBAT.S23.R",
        ],
        "Sequence": [
            "GAACTCTCGACCCACGGCCTGCCACAGCCAGAGCGCCAGGTAC",
            "GCCGTGGGTCGAGAGTTCTGTGGG",
            "GCGCTGGAGTACGCCGGACACGCCGCTCAGCCTGTCGC",
            "CGCGTACTCCAGCGCTGCAG",
            "GGTTGCTCCTAGCCAGCCAACACCTAAAGCCTTTTTTGCAGTTATCAACCCTAGACAACTTACCAG",
            "TTTTAGGTGATGGCTGGCTAGGAGCAACC",
        ],
    }

    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    print("\nClassified DataFrame:")
    classified = classify_oligos(df)
    print(
        classified[
            ["Sequence Name", "prefix", "allele_id", "direction", "is_forward", "is_reverse"]
        ]
    )

    print("\nForward primers only:")
    forward_only = get_forward_primers(df)
    print(forward_only[["Sequence Name", "direction"]])

    print("\nReverse primers only:")
    reverse_only = get_reverse_primers(df)
    print(reverse_only[["Sequence Name", "direction"]])

    print("\nAP_OplR primers only:")
    ap_oplr_only = get_primers_by_prefix(df, "AP_OplR")
    print(ap_oplr_only[["Sequence Name", "prefix", "allele_id"]])

    print("\nSummary:")
    summary = summarize_oligos(df)
    print(summary)

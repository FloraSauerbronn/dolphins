import random
from typing import Dict, List

import pandas as pd


def get_df_with_split_by_audio_chunks_count(
    df: pd.DataFrame,
    audio_name_column: str,
    split_name_to_fraction: Dict[str, int],
    random_seed: int,
) -> pd.DataFrame:
    """
    Adds a split column to a DataFrame based on the audio chunks count.
    The split is done such that the sum of the audio chunks count in each split
    is proportional to the fractions specified in `split_name_to_fraction`.
    """
    assert abs(sum(split_name_to_fraction.values()) - 1) < 0.1e-6

    audio_to_weight = (df.groupby(audio_name_column).size() / len(df)).to_dict()

    split_to_audios = random_split_with_expected_group_sum_fractions(
        audio_to_weight,
        split_name_to_fraction,
        seed=random_seed,
    )
    audio_to_split = {
        audio: split_name
        for split_name, audios in split_to_audios.items()
        for audio in audios
    }
    df_with_splits = df.assign(split_name=df[audio_name_column].map(audio_to_split))
    split_index = (
        df_with_splits.groupby("split_name")
        .rank(method="first", ascending=True)
        .iloc[:, 0]
        .astype(int)
    )
    return (
        df_with_splits.assign(split_index=split_index)
        .sort_values(["split_name", "split_index"])
        .reset_index(drop=True)
    )


def random_split_with_expected_group_sum_fractions(
    split_target_to_weight: Dict[str, float],
    group_name_to_expected_fraction: Dict[str, float],
    seed: int,
) -> Dict[str, List[str]]:
    random.seed(seed)
    weights_sum = sum(split_target_to_weight.values())
    # Compute target sums for each group based on fractions
    group_name_to_target_sum = {
        group_name: expected_fraction * weights_sum
        for group_name, expected_fraction in group_name_to_expected_fraction.items()
    }
    group_name_to_current_sum = {
        group_name: 0.0 for group_name in group_name_to_expected_fraction
    }
    groups = {group_name: [] for group_name in group_name_to_expected_fraction}
    group_names = list(group_name_to_expected_fraction.keys())
    for split_target, to_split_weight in split_target_to_weight.items():
        group_name_to_weight = {}
        total_weight = 0.0
        for group_name in group_names:
            remaining = (
                group_name_to_target_sum[group_name]
                - group_name_to_current_sum[group_name]
            )
            group_weight = max(0.0, remaining)
            group_name_to_weight[group_name] = group_weight
            total_weight += group_weight
        if total_weight > 0.0:
            # Normalize weights to probabilities
            group_probabilities = [
                group_name_to_weight[group_name] / total_weight
                for group_name in group_names
            ]
            # Choose a group based on probabilities
            random_num = random.random()
            cumulative_probability = 0.0
            for group_name, group_probability in zip(group_names, group_probabilities):
                cumulative_probability += group_probability
                if random_num <= cumulative_probability:
                    selected_group = group_name
                    break
        else:
            # All groups have met or exceeded their target sums
            # Assign x to the group with the minimum current sum
            selected_group = min(
                group_name_to_current_sum, key=group_name_to_current_sum.get
            )
        # Assign x to the selected group
        groups[selected_group].append(split_target)
        group_name_to_current_sum[selected_group] += to_split_weight
    return groups

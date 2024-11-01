import pytest

from dolphins.data_split import random_split_with_expected_group_sum_fractions


def test_random_split_with_expected_group_sum_fractions():
    n = 10_000
    split_target_to_weight = {
        str(split_target): split_target for split_target in range(n)
    }
    group_name_to_expected_fraction = {"x": 0.4, "y": 0.6}
    seed = 42

    groups = random_split_with_expected_group_sum_fractions(
        split_target_to_weight, group_name_to_expected_fraction, seed
    )
    group_name_to_actual_fraction = {
        group_name: sum(split_target_to_weight[split_target] for split_target in splits)
        / sum(range(n))
        for group_name, splits in groups.items()
    }

    assert (
        pytest.approx(group_name_to_expected_fraction, abs=1e-3)
        == group_name_to_actual_fraction
    )

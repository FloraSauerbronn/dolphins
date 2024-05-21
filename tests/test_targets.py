import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pathlib import Path

from dolphins.targets import join_target


@pytest.mark.parametrize(
    "join_stategy_name",
    [
        "chunk_contains_percentage_call",
    ],
)
def test_join_target(join_stategy_name: str):
    fixtures_path: Path = Path("tests") / "fixtures" / "join_target" / join_stategy_name
    audio_metadata_df: pd.DataFrame = pd.read_csv(fixtures_path / "audio_metadata.csv")
    labels_df: pd.DataFrame = pd.read_csv(fixtures_path / "labels.csv")
    actual: pd.DataFrame = join_target(
        audio_metadata_df=audio_metadata_df,
        labels_df=labels_df,
        join_stategy_name="chunk_contains_percentage_call",
    )
    expected: pd.DataFrame = pd.read_csv(fixtures_path / "expected.csv")
    assert_frame_equal(actual, expected, check_like=True)

import numpy as np
import pandas as pd
from dolphins.audio_processing import chunk_matrix_on_axis_0, generate_metadata_per_channel


def test_chunk_matrix_on_axis_0():
    data_sample = np.arange(20).reshape(10, 2)
    data_res = chunk_matrix_on_axis_0(data=data_sample, window_size=4, step_size=3)
    expected = np.array(
        [
            [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ],
            [
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
            ],
            [
                [12, 13],
                [14, 15],
                [16, 17],
                [18, 19],
            ],
        ]
    )
    assert np.array_equal(data_res, expected)


def test_generate_metadata_per_channel():
    df = pd.DataFrame({"audio": ["a", "b"]})
    actual = generate_metadata_per_channel(df, 4)
    df_expected = pd.DataFrame({
        "audio": ["a", "b", "a", "b", "a", "b", "a", "b"],
        "channel": [1, 1, 2, 2, 3, 3, 4, 4],
    })
    pd.testing.assert_frame_equal(actual, df_expected)

import numpy as np
from dolphins.audio_processing import chunk_matrix_on_axis_0


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

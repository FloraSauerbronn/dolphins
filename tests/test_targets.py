import pandas as pd
from pandas.testing import assert_frame_equal

from dolphins.targets import join_target


def test_join_target_chunk_contains_percentage_call():
    audio_metadata_df = pd.DataFrame(
        {
            "audio_filename": [
                "audio1.wav",
            ],
            "original_file_length_seconds": [
                365.4,
            ],
            "original_file_sample_rate": [
                96_000,
            ],
            "chunk_index": [
                0,
            ],
            "chunk_start_seconds": [
                0.0,
            ],
            "chunk_end_seconds": [
                2.0,
            ],
            "chunk_file_name": [
                "dolphins/audios/audio1/chunk_0.wav",
            ],
        }
    )
    labels_df = pd.DataFrame(
        {
            "audio_filename": [
                "audio1.wav",
            ],
            "call_channel": [
                1,
            ],
            "label": [
                "click",
            ],
            "call_begin_time": [
                0.0,
            ],
            "call_end_time": [
                0.5,
            ],
            "call_length_seconds": [
                0.5,
            ],
        }
    )
    actual = join_target(audio_metadata_df, labels_df, "chunk_contains_percentage_call")
    expected = pd.DataFrame(
        {
            "audio_filename": [
                "audio1.wav",
            ],
            "original_file_length_seconds": [
                365.4,
            ],
            "original_file_sample_rate": [
                96_000,
            ],
            "chunk_index": [
                0,
            ],
            "chunk_start_seconds": [
                0.0,
            ],
            "chunk_end_seconds": [
                2.0,
            ],
            "chunk_file_name": [
                "dolphins/audios/audio1/chunk_0.wav",
            ],
            "call_channel": [
                1,
            ],
            "label": [
                "click",
            ],
            "call_begin_time": [
                0.0,
            ],
            "call_end_time": [
                0.5,
            ],
            "call_length_seconds": [
                0.5,
            ],
            "call_lenght_within_chunk": [
                0.5,
            ],
        }
    )
    assert_frame_equal(actual, expected, check_like=True)

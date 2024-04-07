from typing import Optional

import pandas as pd
from datasets import Audio, Dataset

from .audio_processing import generate_chunks_for_audios_folder
from .targets import build_labels_df, join_target


def create_dataset(
    audios_folder_name: str,
    chunks_folder_name: str,
    window_seconds: float,
    step_seconds: float,
    sampling_rate: Optional[int],
    mono_channel: bool,
    labels_folder_name: str,
    join_stategy_name: str,
) -> Dataset:
    labels_df: pd.DataFrame = build_labels_df(labels_folder_name)
    audio_metadata_df: pd.DataFrame = generate_chunks_for_audios_folder(
        audios_folder_name, chunks_folder_name, window_seconds, step_seconds
    )
    df: pd.DataFrame = join_target(audio_metadata_df, labels_df, join_stategy_name)
    audio_dataset: Dataset = (
        Dataset.from_pandas(df, preserve_index=False)
        .rename_column("chunk_file_name", "audio")
        .cast_column("audio", Audio(sampling_rate=sampling_rate, mono=mono_channel))
    )
    return audio_dataset


def main():
    create_dataset(
        audios_folder_name="audios",
        chunks_folder_name="chunks",
        window_seconds=8,
        step_seconds=0.25,
        sampling_rate=None,
        mono_channel=False,
        labels_folder_name="labels",
        join_stategy_name="chunk_contains_entire_call"
    )


if __name__ == "__main__":
    main()

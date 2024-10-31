from typing import Optional

import pandas as pd
from datasets import Audio, Dataset

from .audio_processing import generate_chunks_for_audios_folder
from .image_generation import generate_and_save_images_npy
from .targets import build_labels_df, join_target


def create_df(
    audios_folder_name: str,
    chunks_folder_name: str,
    window_seconds: float,
    step_seconds: float,
    labels_folder_name: str,
    join_stategy_name: str,
) -> Dataset:
    labels_df: pd.DataFrame = build_labels_df(labels_folder_name)
    audio_metadata_df: pd.DataFrame = generate_chunks_for_audios_folder(
        audios_folder_name, chunks_folder_name, window_seconds, step_seconds
    )
    df: pd.DataFrame = join_target(audio_metadata_df, labels_df, join_stategy_name)
    return df


def create_dataset(
    df: pd.DataFrame,
    sampling_rate: Optional[int],
    mono_channel: bool,
) -> Dataset:
    audio_dataset: Dataset = (
        Dataset.from_pandas(df, preserve_index=False)
        .rename_column("chunk_file_name", "audio")
        .cast_column("audio", Audio(sampling_rate=sampling_rate, mono=mono_channel))
    )
    return audio_dataset


def main():
    df = create_df(
        audios_folder_name="audios",
        chunks_folder_name="chunks",
        window_seconds=2,
        step_seconds=0.25,
        labels_folder_name="labels",
        join_stategy_name="chunk_contains_percentage_call",
    )
    generate_and_save_images_npy(
        df,
        audio_path_column="chunk_file_name",
        channel_index_column="call_channel",
        output_filename="audio_imgs.npy",
    )


if __name__ == "__main__":
    main()

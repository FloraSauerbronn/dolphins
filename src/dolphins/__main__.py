from typing import Any, Dict

import pandas as pd

from .audio_processing import generate_chunks_for_audios_folder
from .data_split import get_df_with_split_by_audio_chunks_count
from .image_generation import generate_and_save_images_npy
from .targets import build_labels_df, join_target


def create_df(
    audios_folder_name: str,
    chunks_folder_name: str,
    window_seconds: float,
    step_seconds: float,
    labels_folder_name: str,
    join_stategy_name: str,
    sql_query_params: Dict[str, Any],
    num_channels: int,
) -> pd.DataFrame:
    labels_df: pd.DataFrame = build_labels_df(labels_folder_name)
    audio_metadata_df: pd.DataFrame = generate_chunks_for_audios_folder(
        audios_folder_name,
        chunks_folder_name,
        window_seconds,
        step_seconds,
        num_channels,
    )
    df: pd.DataFrame = join_target(
        audio_metadata_df, labels_df, join_stategy_name, sql_query_params
    )
    return df


def main():
    df: pd.DataFrame = create_df(
        audios_folder_name="data/audios",
        chunks_folder_name="chunks",
        window_seconds=2,
        step_seconds=0.25,
        labels_folder_name="data/labels",
        join_stategy_name="chunk_contains_percentage_call",
        sql_query_params={
            "minimum_percentage_of_call_in_chunk": 0.6,
        },
        num_channels=4,
    ).query("label != 'whistle'")

    split_proportions = {
        "train": 0.7,
        "val": 0.2,
        "test": 0.1,
    }
    df_with_splits = get_df_with_split_by_audio_chunks_count(
        df,
        audio_name_column="audio_filename",
        split_name_to_fraction=split_proportions,
        random_seed=42,
    )
    df_with_splits.to_parquet("data/tables/df_with_splits.parquet")

    for split_name in split_proportions:
        generate_and_save_images_npy(
            df_with_splits.query(f"split_name == '{split_name}'"),
            audio_path_column="chunk_file_name",
            channel_column="channel",
            output_filename=f"data/npys/audio_imgs_{split_name}.npy",
            image_generation_params={
                "output_image_dimension_dots": 224,
                "frame_size": 2048,
                "hop_size": 512,
                "min_frequency": 15_000,
                "max_frequency": 48_000,
            },
        )


if __name__ == "__main__":
    main()

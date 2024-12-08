from typing import Any, Dict, List

import pandas as pd

from .audio_processing import generate_chunks_for_audios_folder
from .data_split import get_df_with_split_by_audio_chunks_count
from .image_generation import generate_and_save_images_npy
from .targets import build_labels_df, join_target
from .utils import save_table


def create_df(
    audios_folder_name: str,
    chunks_folder_name: str,
    window_seconds: float,
    step_seconds: float,
    labels_folder_name: str,
    join_stategy_name: str,
    sql_query_params: Dict[str, Any],
    num_channels: int,
    labels_to_remove: List[str],
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
    return df[~df["label"].isin(labels_to_remove)]


def run(config: Dict[str, Any]):
    folders: Dict[str, str] = config["folders"]

    base_metadata_df: pd.DataFrame = create_df(
        audios_folder_name=folders.get("audios_folder_name"),
        chunks_folder_name=folders.get("chunks_folder_name"),
        labels_folder_name=folders.get("labels_folder_name"),
        join_stategy_name="chunk_contains_percentage_call",
        **config.get("chunk_params"),
        num_channels=config.get("num_channels"),
        labels_to_remove=config.get("labels_to_remove"),
    )
    save_table(
        base_metadata_df,
        folder_path=folders.get("tables_folder_name"),
        file_name="base_metadata",
    )

    split_params: Dict[str, Any] = config.get("split")
    df_with_splits = get_df_with_split_by_audio_chunks_count(
        base_metadata_df,
        audio_name_column="audio_filename",
        **split_params,
    )
    save_table(
        df_with_splits,
        folder_path=folders.get("tables_folder_name"),
        file_name="metadata_with_splits",
    )

    for split_name in split_params.get("split_name_to_fraction"):
        generate_and_save_images_npy(
            df_with_splits,
            split_name=split_name,
            audio_path_column="chunk_file_name",
            channel_column="channel",
            output_filename=f"{folders.get('npys_folder_name')}/audio_imgs_{split_name}.npy",
            image_generation_params=config.get("image_generation_params"),
        )

from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd


def build_labels_df(labels_folder_name: str) -> pd.DataFrame:
    labels_folder = Path(labels_folder_name)

    dfs_for_concatenation: List[pd.DataFrame] = []
    for labels_file in labels_folder.glob("*.txt"):
        df: pd.DataFrame = pd.read_csv(labels_file, sep="\t")
        df.columns = df.columns.str.capitalize()
        df_to_append = (
            df.rename(
                columns={
                    "Channel": "channel",
                    "Type": "label",
                    "Begin time (s)": "call_begin_time",
                    "End time (s)": "call_end_time",
                }
            )
            .assign(
                audio_filename=labels_file.name.split(".")[0] + ".wav",
                call_length_seconds=lambda x: x.call_end_time - x.call_begin_time,
                label=lambda x: x.label.str.capitalize().replace(
                    {"^W.*": "whistle", "^C.*": "click", "^L.*":"click"}, regex=True
                ),
            )
            .dropna(subset=["label"])[
                [
                    "audio_filename",
                    "channel",
                    "label",
                    "call_begin_time",
                    "call_end_time",
                    "call_length_seconds",
                ]
            ]
        )
        dfs_for_concatenation.append(df_to_append)
    labels_df: pd.DataFrame = pd.concat(dfs_for_concatenation, ignore_index=True)
    return labels_df


def join_target(
    audio_metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    join_stategy_name: str,
    sql_query_params: Dict[str, Any],
) -> pd.DataFrame:
    sql_path: Path = Path(".") / "src" / "dolphins" / "sql" / f"{join_stategy_name}.sql"
    with open(sql_path, "r") as file:
        sql_query: str = file.read().format(**sql_query_params)
        df: pd.DataFrame = duckdb.sql(sql_query).to_df()
    return df

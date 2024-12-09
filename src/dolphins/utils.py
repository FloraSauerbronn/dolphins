from functools import wraps
from pathlib import Path
from time import time
from typing import Any, Dict

import pandas as pd


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time()
        result = f(*args, **kw)
        end_time = time()
        print(
            f"function:{f.__name__} args:[{args}, {kw}] took: {(end_time-start_time):.4f} sec"
        )
        return result

    return wrap


def save_table(
    df: pd.DataFrame,
    config: Dict[str, Any],
    table_name_key: str,
):
    folder = Path(config["folders"]["tables_folder_name"])
    file_name = config["table_names"][table_name_key]
    df.to_parquet(folder / f"{file_name}.parquet", index=False)
    df.to_csv(folder / f"{file_name}.csv", index=False)


def read_table(
    config: Dict[str, Any],
    table_name_key: str,
) -> pd.DataFrame:
    folder = Path(config["folders"]["tables_folder_name"])
    file_name = config["table_names"][table_name_key]
    return pd.read_parquet(folder / f"{file_name}.parquet")

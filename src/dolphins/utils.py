from functools import wraps
from pathlib import Path
from time import time

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
    folder_path: str,
    file_name: str,
):
    folder = Path(folder_path)
    df.to_parquet(folder / f"{file_name}.parquet", index=False)
    df.to_csv(folder / f"{file_name}.csv", index=False)

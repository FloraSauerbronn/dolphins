import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def undersample(
    df: pd.DataFrame,
    split_to_undersample: str,
) -> pd.DataFrame:
    undersampler = RandomUnderSampler(
        sampling_strategy="majority",
        random_state=42,
    )
    df_to_undersample = df.query(f"split_name == '{split_to_undersample}'")
    labels = df_to_undersample["label"]
    df_undersampled, _ = undersampler.fit_resample(df_to_undersample, labels)

    return pd.concat(
        [df.query(f"split_name != '{split_to_undersample}'"), df_undersampled]
    )

import io
from typing import Dict

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from npy_append_array import NpyAppendArray
from tqdm import tqdm


def generate_image_array_from_audio(
    audio_path: str,
    channel_index: int,
    frame_size: int,
    hop_size: int,
    output_image_dimension_dots: int,
    min_frequency: int,
    max_frequency: int,
) -> np.ndarray:
    audio_time_series, sampling_rate = librosa.load(
        audio_path,
        sr=None,
        mono=False,
    )
    stft = librosa.stft(
        audio_time_series[channel_index, :],
        n_fft=frame_size,
        hop_length=hop_size,
    )
    y_log_scale = librosa.amplitude_to_db(np.abs(stft))

    dimension_inches = 1
    fig, ax = plt.subplots(
        figsize=(dimension_inches, dimension_inches),
        dpi=output_image_dimension_dots,
    )
    ax.axis("off")
    ax.margins(0, 0)
    ax.set_ylim(ymin=min_frequency, ymax=max_frequency)

    img = librosa.display.specshow(
        y_log_scale,
        sr=sampling_rate,
        hop_length=hop_size,
        x_axis="time",
        y_axis="linear",
        cmap="gray",
        fmin=min_frequency,
        fmax=max_frequency,
        ax=ax,
    )
    img.set_clim(-60, 10)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout(pad=0)

    with io.BytesIO() as buf:
        fig.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=output_image_dimension_dots,
        )
        buf.seek(0)
        # Read image and covert from RGBA to RGB
        image = plt.imread(buf)[..., :3].T
    plt.close()

    return image


def generate_and_save_images_npy(
    df: pd.DataFrame,
    split_name: str,
    audio_path_column: str,
    channel_column: str,
    output_filename: str,
    image_generation_params: Dict[str, int],
):
    df_filtered = df.query(f"split_name == '{split_name}'").sort_values(
        ["split_name", "split_index"]
    )
    print(f"\nGenerating images and saving to {output_filename}")
    with NpyAppendArray(output_filename, delete_if_exists=True) as npaa:
        for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
            image = generate_image_array_from_audio(
                **image_generation_params,
                audio_path=row[audio_path_column],
                channel_index=row[channel_column] - 1,
            )
            npaa.append(np.expand_dims(image, axis=0))

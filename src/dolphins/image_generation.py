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
    channel_index: int = 0,
    frame_size: int = 2048,
    hop_size: int = 512,
    output_image_dimension_dots: int = 224,
    min_frequency: int = 15_000,
    max_frequency: int = 48_000,
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
    audio_path_column: str,
    channel_index_column: str,
    output_filename: str,
    image_generation_params: Dict[str, int] = {},
):
    print(f"\nGenerating images and saving to {output_filename}")
    with NpyAppendArray(output_filename, delete_if_exists=True) as npaa:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            channel_index = np.nan_to_num(row[channel_index_column], nan=1).astype(int) - 1
            image = generate_image_array_from_audio(
                **image_generation_params,
                audio_path=row[audio_path_column],
                channel_index=channel_index,
            )
            npaa.append(np.expand_dims(image, axis=0))

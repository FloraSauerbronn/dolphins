import io

import librosa
import matplotlib.pyplot as plt
import numpy as np


def generate_image_array_from_audio(
    audio_path: str,
    call_channel: int = 0,
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
        audio_time_series[call_channel,:],
        n_fft=frame_size,
        hop_length=hop_size,
    )
    y_log_scale = librosa.amplitude_to_db(np.abs(stft))

    dimension_inches = 1
    fig, ax = plt.subplots(
        figsize=(dimension_inches, dimension_inches),
        dpi=output_image_dimension_dots,
    )
    ax.axis('off')
    ax.margins(0, 0)
    ax.set_ylim(ymin=min_frequency, ymax=max_frequency)

    img = librosa.display.specshow(
        y_log_scale,
        sr=sampling_rate,
        hop_length=hop_size,
        x_axis='time',
        y_axis='linear',
        cmap='gray',
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
            format='png',
            bbox_inches='tight',
            pad_inches=0,
            dpi=output_image_dimension_dots,
        )
        buf.seek(0)
        # Read image and covert from RGBA to RGB
        image = plt.imread(buf)[..., :3]
    plt.close()

    return image



audio_path = "/Users/pedro.igor/dev/personal/dolphins/audios/chunks/LPS1142017_MF_20170804_084350_893/chunk_1112.wav"

arr = generate_image_array_from_audio(audio_path)
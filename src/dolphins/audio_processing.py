from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm


def read_wav_file(audio_path: str) -> Dict[str, Any]:
    sample_rate, data = wavfile.read(audio_path)
    num_samples, num_channels = data.shape
    length_seconds = num_samples / sample_rate
    return {
        "sample_rate": sample_rate,
        "length_seconds": length_seconds,
        "data": data,
        "num_samples": num_samples,
        "num_channels": num_channels,
    }


def chunk_matrix_on_axis_0(data, window_size: int, step_size: int) -> np.ndarray:
    num_elements_outside_first_chunk = data.shape[0] - window_size
    if num_elements_outside_first_chunk % step_size != 0:
        raise ValueError(
            f"Data size minus window size must ({num_elements_outside_first_chunk}) be a multiple of window size ({step_size})"
        )

    chunked_shape = (
        num_elements_outside_first_chunk // step_size + 1,
        window_size,
        data.shape[1],
    )
    chunked_strides = (
        data.strides[0] * step_size,  # number of bytes to skip to get to the next chunk
        data.strides[0],  # number of bytes of each row
        data.strides[1],  # number of bytes of each single element
    )
    return np.lib.stride_tricks.as_strided(
        data, shape=chunked_shape, strides=chunked_strides, writeable=False
    )


def chunk_audio_data(
    audio: Dict[str, Any], window_seconds: float, step_seconds: float
) -> np.ndarray:
    window_size = int(window_seconds * audio["sample_rate"])
    step_size = int(step_seconds * audio["sample_rate"])
    return chunk_matrix_on_axis_0(audio["data"], window_size, step_size)


def round_audio(
    audio: Dict[str, Any], window_seconds: float, step_seconds: float
) -> Dict[str, Any]:
    window_size = int(window_seconds * audio["sample_rate"])
    step_size = int(step_seconds * audio["sample_rate"])
    num_elements_outside_first_chunk = audio["num_samples"] - window_size
    num_elems_to_keep = audio["num_samples"] - (
        num_elements_outside_first_chunk % step_size
    )
    rounded_audio_data = audio["data"][:num_elems_to_keep]
    return {
        "sample_rate": audio["sample_rate"],
        "length_seconds": num_elems_to_keep / audio["sample_rate"],
        "data": rounded_audio_data,
        "num_samples": num_elems_to_keep,
        "num_channels": audio["num_channels"],
    }


def generate_chunks_for_audios_folder(
    audios_folder_name: str,
    chunks_folder_name: str,
    window_seconds: float,
    step_seconds: float,
) -> pd.DataFrame:
    audios_folder = Path(audios_folder_name)
    chunks_folder = audios_folder / chunks_folder_name
    chunks_folder.mkdir(parents=True, exist_ok=True)

    metadata: List[Dict[str, Any]] = []
    for audio_file in audios_folder.glob("*.wav"):
        print(f"Generating chunks for {audio_file.name}")
        audio = read_wav_file(audio_file)
        rounded_audio = round_audio(audio, window_seconds, step_seconds)
        chunked_audio_data = chunk_audio_data(
            audio=rounded_audio,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
        audio_chunk_folder = chunks_folder / audio_file.stem
        audio_chunk_folder.mkdir(parents=True, exist_ok=True)
        for index, audio_chunk in tqdm(
            enumerate(chunked_audio_data), total=len(chunked_audio_data)
        ):
            chunk_file = audio_chunk_folder / f"chunk_{index}.wav"
            chunk_metadata = {
                "audio_filename": audio_file.name,
                "original_file_length_seconds": audio["length_seconds"],
                "original_file_sample_rate": audio["sample_rate"],
                "chunk_index": index,
                "chunk_start_seconds": index * step_seconds,
                "chunk_end_seconds": (index * step_seconds) + window_seconds,
                "chunk_file_name": chunk_file.resolve().__str__(),
            }
            wavfile.write(
                filename=chunk_file,
                rate=audio["sample_rate"],
                data=audio_chunk,
            )
            metadata.append(chunk_metadata)

    return pd.DataFrame(metadata)

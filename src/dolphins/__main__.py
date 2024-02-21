from .audio_processing import create_audio_dataset


def main():
    create_audio_dataset(
        audios_folder_name="audios",
        chunks_folder_name="chunks",
        window_seconds=1,
        step_seconds=0.25,
        sampling_rate=None,
        mono_channel=False,
    )


if __name__ == "__main__":
    main()

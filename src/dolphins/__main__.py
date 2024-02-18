from .audio_processing import generate_chunks_for_audios_folder


def main():
    generate_chunks_for_audios_folder(
        audios_folder_name="audios",
        chunks_folder_name="chunks",
        window_seconds=1,
        step_seconds=0.25,
    )


if __name__ == "__main__":
    main()

CONFIG = {
    "folders": {
        "audios_folder_name": "E:/all_audios",
        "chunks_folder_name": "E:/datasets/2sec-60perc-gray/chunks",
        "labels_folder_name": "data/labels",
        "npys_folder_name": "E:/datasets/2sec-60perc-gray",
        "tables_folder_name": "data/tables",
    },
    "table_names": {
        "base_metadata": "base_metadata",
        "metadata_with_splits": "metadata_with_splits",
    },
    "chunk_params": {
        "window_seconds": 2,
        "step_seconds": 0.25,
        "sql_query_params": {
            "minimum_percentage_of_call_in_chunk": 0.6,
        },
    },
    "num_channels": 4,
    "labels_to_remove": ["whistle"],
    "split": {
        "split_name_to_fraction": {
            "train": 0.7,
            "val": 0.2,
            "test": 0.1,
        },
        "random_seed": 42,
    },
    "image_generation_params": {
        "output_image_dimension_dots": 224,
        "frame_size": 2048,
        "hop_size": 512,
        "min_frequency": 15_000,
        "max_frequency": 48_000,
    },
}

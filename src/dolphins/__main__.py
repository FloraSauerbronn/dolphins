import click

from .config import CONFIG
from .controllers import (
    run_all,
    run_chunks_and_base_metadata,
    run_images,
    run_splits,
)


@click.group()
def cli():
    pass


@cli.command("run-chunks-and-base-metadata")
def chunks_and_base_metadata():
    """
    Generates the audio chunks and base metadata table.
    """
    run_chunks_and_base_metadata(CONFIG)


@cli.command("run-splits")
def splits():
    """
    Generates the metadata table with splits.
    """
    run_splits(CONFIG)


@cli.command("run-images")
def images():
    """
    Generates the spectrogram images for each audio chunk and channel and saves them into npy files for each data split.
    """
    run_images(CONFIG)


@cli.command("run-all")
def all():
    """
    Runs all the steps in the pipeline.
    """
    run_all(CONFIG)


if __name__ == "__main__":
    cli()

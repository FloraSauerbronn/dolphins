# Automated Delphinid Click Identification in Seismic Acoustic Surveys for Environmental Impact Assessment

This repository contains the code implementation of the methodology proposed in the paper **"Automated Delphinid Click Identification in Seismic Acoustic Surveys for Environmental Impact Assessment"**. The project aims to support environmental monitoring by automatically identifying delphinid clicks in acoustic data collected during seismic surveys using passive acoustic monitoring (PAM).

## Acknowledgements

This research was funded by public resources allocated to the project supporting the **Ocean Dynamics Laboratory** at the **Federal University of Santa Catarina (UFSC)**.  
It was developed as part of the master's research of **Flora Medeiros Sauerbronn**, under the supervision of **Professor Antonio Härter Fetter** (UFSC) and co-supervision of **Professor Andrea Dalbel Soares** from **BIOSONAR**.

We would also like to acknowledge the essential contributions of **Pedro Igor de Araújo Oliveira** and **Ingridy Moara Severino**, who were actively involved in the development and execution of this project.

Audio recordings and manual annotations (labels) used in this research can be accessed at:  
**[INSERT DATA REPOSITORY LINK HERE]**

## Installation

* Install [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#getting-started)
* Create virtual environment:
    ```bash
    uv venv  # Create a virtual environment at .venv.
    ```
* Activate virtual environment:
    ```bash
    source .venv/bin/activate
    ```
* Install the project itself:

    For minimal dependencies:
    ```bash
    uv pip install -e .
    ```
    Or with development dependencies:
    ```bash
    uv pip install -e .[dev]
    ```

## Usage

### Configuration
* The `config.py` sets all the configurations for the project.
* Audio files should be placed in the `audios` folder as specified in the config.
* Annotation CSVs should be placed in the `labels` directory as specified in the config. Obs.: each CSV should have a prefix equal to the name of the corresponding audio file plus a suffix separated by a dot (`.`).

### Check the commands available in the CLI with
```bash
python -m dolphins
```

## Compile dependencies

### Main dependencies only
```bash
uv pip compile pyproject.toml -o requirements.txt
```

### With development dependencies
```bash
uv pip compile pyproject.toml -o requirements_dev.txt --extra dev
```

## Run tests

* Make sure to have the development dependencies installed.
* Run tests with:
```bash
pytest
```

## Linter & formatter

* Make sure to have the development dependencies installed.
* Run linter with:
```bash
ruff check src/ tests/
```
* Run formatter with:
```bash
ruff format src/ tests/
```

## Repository Structure

Below is an overview of the main folders included in this repository:

### `data/`
Main directory for input and output data used throughout the project.

- **`audios/`**: Contains raw audio files, as collected during the pre-watch phase of seismic surveys by the onboard environmental monitoring team. No preprocessing is applied at this stage.
- **`chunks/`**: Output directory containing audio segments ("chunks") extracted from the raw recordings. These are the samples used to train the machine learning model.
- **`labels/`**: Contains `.txt` files with manual annotations and cataloging of the raw audio data.
- **`npys/`**: Stores `.npy` files that compile training, validation, and testing datasets. These are created from the chunked samples and used directly by the model.
- **`tables/`**: Tabular metadata files that describe the content of each chunk, indicating whether it contains clicks, whistles, or background noise.

### `src/dolphins/`
This folder contains all source code necessary to reproduce the pipeline proposed in the paper, from audio preprocessing to dataset preparation and model training.

#### Subfolders:
- **`sql/`**  
  Contains a DuckDB query that applies the thresholding rule described in the paper. It defines how many delphinid calls (clicks) a chunk must contain to be labeled as a positive sample.

#### Python Scripts:
- **`__init__.py`** and **`__main__.py`**  
  Initialization and entry point for the package.

- **`audio_processing.py`**  
  Includes all functions related to raw audio preprocessing, such as filtering, chunking, and signal transformation.

- **`config.py`**  
  The only file intended to be modified by the user. It stores the configuration parameters such as chunk size, spectrogram characteristics, and file paths.

- **`controllers.py`**  
  Manages and orchestrates the execution of different modules in the pipeline.

- **`data_Split.py`**  
  Responsible for generating the training, validation, and test datasets from the preprocessed data.

- **`image_generation.py`**  
  Handles the creation and formatting of spectrogram images used as model input.

- **`model_data_loader.py`**  
  Implements the `DataLoader` used during model training and evaluation.

- **`targets.py`**  
  Parses and processes label files, mapping delphinid events to corresponding audio chunks.

- **`undersampling.py`**  
  Applies undersampling rules to balance the dataset by reducing overrepresented classes.

- **`utils.py`**  
  Contains auxiliary functions used across different modules.

### Contact information
* Flora Medeiros Sauerbronn: [flora.ufsc24@gmail.com]
* Prof. Antonio Fetter: [antoniofetter@gmail.com]
* Andrea Dalben: [biosonardalben@gmail.com]


<h3>This project was a partnership between:</h3>

<p align="center">
  <img src="logos/ufsc.png" alt="UFSC" width="150" style="margin-right: 40px;">
  <img src="logos/logo_Biosonar.png" alt="BIOSONAR" width="200">
</p>

<p align="center">
  <strong>Federal University of Santa Catarina</strong> &nbsp;&nbsp;&nbsp;&nbsp; <strong>BIOSONAR</strong>
</p>

# dolphins
Consultancy for Flora's 🌺 master thesis.

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

### Placing data files
* Audio files should be placed in the `audios` directory.
* Annotation CSVs should be placed in the `labels` directory. Obs.: each CSV should have a prefix equal to the name of the corresponding audio file plus a suffix separated by a dot (`.`).

### Run the main script with
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

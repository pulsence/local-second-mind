# Development Setup

This guide explains how to set up a local development environment for LSM.

## Requirements

- Python 3.10 or newer
- Git
- Optional: CUDA-capable GPU for faster embedding

## Install the Package

From the repository root:

```bash
pip install -e .
```

For dev tooling (tests, linting):

```bash
pip install -e ".[dev]"
```

## Configure Environment

1. Copy the example env file:

```bash
cp .env.example .env
```

2. Add API keys (OpenAI, Brave) to `.env`.

```bash
OPENAI_API_KEY=...
BRAVE_API_KEY=...
```

## Create a Config File

Start from `example_config.json` and edit `roots` and other settings.

```bash
cp example_config.json config.json
```

## Run Ingest

```bash
python -m lsm ingest --config config.json
```

## Run Query

```bash
python -m lsm query --config config.json
```

## Use the Unified Shell

```bash
python -m lsm shell --config config.json
```

## Optional OCR Setup

OCR requires the Tesseract binary in PATH.

- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
- macOS: `brew install tesseract`
- Linux: `apt install tesseract-ocr`

## IDE Tips

- Use a virtualenv (`python -m venv .venv`).
- Configure your IDE to use the venv interpreter.
- Enable linting for `ruff` if available.

## Common Dev Tasks

- Run tests: `pytest`
- Lint: `ruff check .`
- Format: `ruff format .`

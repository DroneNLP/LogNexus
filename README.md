# LogNexs (re: Log Nexus)

LogNexs (re: Log Nexus) is a command-line sentence segmentation tool for decrypted drone
flight log messages. It uses a domain-tuned DistilBERT NER model to split noisy,
multi-sentence flight log messages into semantically complete sentence records
for downstream forensic review and analysis.

## Features

- Batch-processes decrypted DJI CSV flight logs from an input directory.
- Extracts `APP.tip` and `APP.warning` messages into a message-level timeline.
- Downloads the required Hugging Face model with `lognexus-download`.
- Exports results as nested JSON or exploded XLSX rows.
- Provides a SoPID-style forensic inference pipeline through `lognexus-pipeline`.
- Supports optional CUDA inference when PyTorch detects an available GPU.

## Installation

Create and activate a Python environment, then install from the repository:

```bash
git clone https://github.com/DroneNLP/LogNexus.git
cd LogNexus
python -m pip install .
```

Or install the PyPI distribution after release:

```bash
python -m pip install LogNexs
```

LogNexs depends on PyTorch and `simpletransformers`. For GPU usage, install the
PyTorch build that matches your CUDA environment before running LogNexs.

## Model Setup

The NER model is not bundled into the Python package. Download it separately:

```bash
lognexus-download
```

By default, this downloads:

```text
swardiantara/LogNexus-distilbert-base-uncased
```

into:

```text
./model
```

Custom download location:

```bash
lognexus-download --model_dir /path/to/model
```

## Input Data

Place decrypted `.csv` flight logs in the input directory. The CSV files must
contain these columns:

```text
CUSTOM.date [local]
CUSTOM.updateTime [local]
APP.tip
APP.warning
```

LogNexs reads each non-empty `APP.tip` and `APP.warning` cell as a separate log
message while preserving the original date and time values.

## Usage

### Sentence Extraction

Basic run:

```bash
lognexus
```

This uses:

```text
input:  ./evidence
output: ./output
model:  ./model
format: json
```

Custom paths:

```bash
lognexus --input_dir /path/to/logs --output_dir /path/to/results --model_dir /path/to/model --format json
```

XLSX output:

```bash
lognexus --format xlsx
```

GPU inference:

```bash
lognexus --cuda
```

If CUDA is requested but unavailable, LogNexs falls back to CPU.

### SoPID-Style Inference Pipeline

The `lognexus-pipeline` command ports the working inference pipeline from
SoPID into the LogNexus package structure. It supports two paradigms:

- `message`: classifies whole log messages using the Hugging Face sentiment
  model `swardiantara/drone-sentiment`.
- `segment`: segments messages with the SoPID NER model and classifies each
  unique segment with a local DroPTC classifier.

Recommended message-level run:

```bash
lognexus-pipeline --paradigm message --evidence-dir ./evidence --output-dir ./pipeline-output
```

Segment-level run:

```bash
lognexus-pipeline \
  --paradigm segment \
  --model-name swardiantara/SoPID-bert-base-cased \
  --model-type bert \
  --pretokenizer spacy \
  --tag-scheme bioes \
  --droptc-model-dir ./best-model/droptc \
  --evidence-dir ./evidence \
  --output-dir ./pipeline-output
```

Pipeline evidence can be flat (`evidence/*.csv`) or grouped by drone model
(`evidence/{drone-model}/*.csv`). Outputs are written under:

```text
pipeline-output/{message|segment}-{before|after}/run-{n}/{drone-model}/{flight-log}/
```

Each processed log gets:

- `unique_events.xlsx`: deduplicated messages or segments for manual review.
- `timeline.json`: full forensic timeline with propagated labels.
- `timing.json`: per-log timing.
- `prediction.json`: segment CoNLL-style predictions, for segment runs only.

The run folder also gets `timing_summary.json`.

## Output Formats

JSON output keeps one record per original message and stores extracted sentences
as a list:

```json
[
  {
    "date": "5/12/2025",
    "time": "8:27:36.34 AM",
    "message": "Failsafe RTH.; RC signal lost. Returning to home.",
    "sentence": [
      "Failsafe RTH",
      "RC signal lost",
      "Returning to home"
    ]
  }
]
```

XLSX output explodes the `sentence` list so each extracted sentence gets its own
spreadsheet row.

## Development

Install the lightweight test dependencies without downloading the ML stack:

```bash
python -m pip install pytest pandas openpyxl
python -m pip install -e . --no-deps
```

Run tests:

```bash
pytest
```

Build package artifacts:

```bash
python -m build
twine check dist/*
```

## Publishing Note

The PyPI distribution name is `LogNexs`. The internal Python import package and
the installed console commands remain `lognexus` and `lognexus-download` for
compatibility with the original tool and paper terminology.

## Citation

```bibtex
@misc{Silalahi2025LogNexus,
  title = {LogNexus: A Foundational Segmentation Tool for Drone Flight Log Messages},
  publisher = {Code Ocean},
  year = {2025},
  note = {[Source Code]},
  author = {Swardiantara Silalahi and Tohari Ahmad and Hudan Studiawan}
}
```

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from lognexus.model import LogNexusModel


SOURCE_COLUMNS = [
    "CUSTOM.date [local]",
    "CUSTOM.updateTime [local]",
    "APP.tip",
    "APP.warning",
]
OUTPUT_COLUMNS = ["date", "time", "message"]


@dataclass
class ProcessResult:
    processed_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _detect_skiprows(filepath):
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as file:
        first_line = file.readline()
    normalized_first_line = first_line.lower()
    has_header = all(column.lower() in normalized_first_line for column in SOURCE_COLUMNS[:2])
    return None if has_header else 1


def _resolve_source_columns(columns):
    normalized = {str(column).strip().lower(): column for column in columns}
    resolved = {}
    missing = []
    for expected in SOURCE_COLUMNS:
        column = normalized.get(expected.lower())
        if column is None:
            missing.append(expected)
        else:
            resolved[expected] = column

    if missing:
        expected = ", ".join(SOURCE_COLUMNS)
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing required column(s): {missing_text}. Expected columns: {expected}"
        )

    return resolved


def load_and_extract_log(filepath):
    """
    Read a decrypted DJI CSV flight log and extract message-level timeline rows.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Log file not found: {filepath}")
    if filepath.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported input file type: {filepath.suffix}. Expected .csv")

    file_df = pd.read_csv(filepath, skiprows=_detect_skiprows(filepath))
    columns = _resolve_source_columns(file_df.columns)
    timeline_df = file_df[
        [
            columns["CUSTOM.date [local]"],
            columns["CUSTOM.updateTime [local]"],
            columns["APP.tip"],
            columns["APP.warning"],
        ]
    ]

    rows = []
    for _, row in timeline_df.iterrows():
        date = row[columns["CUSTOM.date [local]"]]
        time = row[columns["CUSTOM.updateTime [local]"]]
        tip = row[columns["APP.tip"]]
        warning = row[columns["APP.warning"]]

        if not pd.isna(tip):
            rows.append({"date": date, "time": time, "message": tip})
        if not pd.isna(warning):
            rows.append({"date": date, "time": time, "message": warning})

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def process_logs(input_dir, output_dir, model_path, output_format="xlsx", use_cuda=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_path = Path(model_path)
    output_format = output_format.lower()

    if output_format not in {"xlsx", "json"}:
        raise ValueError("output_format must be either 'xlsx' or 'json'")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model directory does not exist: {model_path}")

    print(f"[-] Loading LogNexus model from: {model_path}")
    if use_cuda:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required when --cuda is requested.") from exc
        if not torch.cuda.is_available():
            print("[!] CUDA requested but not available. Falling back to CPU.")
            use_cuda = False
    ner_model = LogNexusModel(model_path, use_cuda=use_cuda)
    print("[+] Model loaded successfully.")

    os.makedirs(output_dir, exist_ok=True)
    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".csv")

    if not files:
        print(f"[!] No log files found in {input_dir}")
        return ProcessResult()

    print(f"[-] Found {len(files)} log file(s). Start processing...")
    result = ProcessResult()

    try:
        from tqdm import tqdm

        file_iterator = tqdm(files, desc="Processing")
    except ImportError:
        file_iterator = files

    for input_path in file_iterator:
        try:
            df = load_and_extract_log(input_path)
            messages = df["message"].fillna("").astype(str).tolist()
            df["sentence"] = ner_model.predict_sentences(messages)
            if output_format == "json":
                out_file = input_path.with_suffix(".json").name
                with open(output_dir / out_file, "w", encoding="utf-8") as file:
                    json.dump(df.to_dict(orient="records"), file, indent=2, default=str)
            else:
                out_file = f"{input_path.stem}_processed.xlsx"
                df.explode("sentence").to_excel(output_dir / out_file, index=False)
            result.processed_files.append(out_file)
        except Exception as e:
            error = f"{input_path.name}: {e}"
            result.errors.append(error)
            print(f"[!] Error processing {error}")

    print(f"[+] Completed. Results in: {output_dir}")
    return result

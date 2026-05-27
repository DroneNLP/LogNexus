import json

import pandas as pd
import pytest

from lognexus import core


HEADER = "CUSTOM.date [local],CUSTOM.updateTime [local],APP.tip,APP.warning\n"


class FakeModel:
    def __init__(self, model_path, use_cuda=False):
        self.model_path = model_path
        self.use_cuda = use_cuda

    def predict_sentences(self, messages):
        return [[message] for message in messages]


def write_csv(path, body, header=HEADER):
    path.write_text(header + body, encoding="utf-8")


def test_load_and_extract_log_expands_tip_and_warning_rows(tmp_path):
    log_file = tmp_path / "flight.csv"
    write_csv(
        log_file,
        "2025-05-12,8:27:36 AM,Failsafe RTH,RC signal weak\n"
        "2025-05-12,8:27:37 AM,,Landing\n",
    )

    df = core.load_and_extract_log(log_file)

    assert list(df.columns) == ["date", "time", "message"]
    assert df.to_dict(orient="records") == [
        {"date": "2025-05-12", "time": "8:27:36 AM", "message": "Failsafe RTH"},
        {"date": "2025-05-12", "time": "8:27:36 AM", "message": "RC signal weak"},
        {"date": "2025-05-12", "time": "8:27:37 AM", "message": "Landing"},
    ]


def test_load_and_extract_log_supports_prefixed_csv_export(tmp_path):
    log_file = tmp_path / "flight.csv"
    log_file.write_text(
        "metadata exported by viewer\n"
        + HEADER
        + "2025-05-12,8:27:36 AM,Failsafe RTH,\n",
        encoding="utf-8",
    )

    df = core.load_and_extract_log(log_file)

    assert df["message"].tolist() == ["Failsafe RTH"]


def test_load_and_extract_log_reports_missing_columns(tmp_path):
    log_file = tmp_path / "flight.csv"
    log_file.write_text("date,time,message\n2025-05-12,8:27:36 AM,Failsafe RTH\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required column"):
        core.load_and_extract_log(log_file)


def test_process_logs_writes_json_and_reports_outputs(tmp_path, monkeypatch):
    input_dir = tmp_path / "evidence"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "model"
    input_dir.mkdir()
    model_dir.mkdir()
    write_csv(input_dir / "flight.csv", "2025-05-12,8:27:36 AM,Failsafe RTH,\n")
    monkeypatch.setattr(core, "LogNexusModel", FakeModel)

    result = core.process_logs(input_dir, output_dir, model_dir, output_format="json")

    assert result.errors == []
    assert result.processed_files == ["flight.json"]
    output = json.loads((output_dir / "flight.json").read_text(encoding="utf-8"))
    assert output == [
        {
            "date": "2025-05-12",
            "time": "8:27:36 AM",
            "message": "Failsafe RTH",
            "sentence": ["Failsafe RTH"],
        }
    ]


def test_process_logs_writes_exploded_xlsx(tmp_path, monkeypatch):
    input_dir = tmp_path / "evidence"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "model"
    input_dir.mkdir()
    model_dir.mkdir()
    write_csv(input_dir / "flight.csv", "2025-05-12,8:27:36 AM,Failsafe RTH,\n")
    monkeypatch.setattr(core, "LogNexusModel", FakeModel)

    result = core.process_logs(input_dir, output_dir, model_dir, output_format="xlsx")

    assert result.errors == []
    assert result.processed_files == ["flight_processed.xlsx"]
    written = pd.read_excel(output_dir / "flight_processed.xlsx")
    assert written["sentence"].tolist() == ["Failsafe RTH"]


def test_process_logs_rejects_missing_input_dir(tmp_path):
    with pytest.raises(NotADirectoryError):
        core.process_logs(tmp_path / "missing", tmp_path / "output", tmp_path)

import json

import pandas as pd

from lognexus import inference


HEADER = "CUSTOM.date [local],CUSTOM.updateTime [local],APP.message,APP.tip,APP.warning\n"


class FakeSentimentPipeline:
    def __call__(self, texts, batch_size=32, truncation=True):
        return [
            {"label": "negative" if "warning" in text.lower() else "positive"}
            for text in texts
        ]


def write_csv(path, body, header=HEADER):
    path.write_text(header + body, encoding="utf-8")


def test_extract_messages_reads_message_tip_and_warning(tmp_path):
    csv_path = tmp_path / "flight.csv"
    write_csv(
        csv_path,
        "2025-05-12,8:27:36 AM,SystemOK,Failsafe RTH,RC warning\n",
    )

    messages = inference.extract_messages(csv_path)

    assert [message["message_type"] for message in messages] == [
        "message",
        "tip",
        "warning",
    ]
    assert messages[0]["message"] == "System OK."


def test_message_pipeline_writes_unique_events_timeline_and_timing(tmp_path):
    csv_path = tmp_path / "flight.csv"
    output_dir = tmp_path / "out"
    write_csv(
        csv_path,
        "2025-05-12,8:27:36 AM,,Failsafe RTH,RC warning\n"
        "2025-05-12,8:27:37 AM,,Failsafe RTH,\n",
    )

    timing = inference.process_flight_log(
        csv_path=csv_path,
        paradigm="message",
        output_dir=output_dir,
        sentiment_pipeline=FakeSentimentPipeline(),
    )

    assert timing["n_messages"] == 3
    assert timing["n_unique_events"] == 2
    unique = pd.read_excel(output_dir / "unique_events.xlsx")
    assert unique["predicted_label"].tolist() == ["Normal", "Problem"]
    timeline = json.loads((output_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["total_messages"] == 3
    assert (output_dir / "timing.json").is_file()


def test_process_evidence_supports_grouped_device_directories(tmp_path):
    evidence_dir = tmp_path / "evidence"
    device_dir = evidence_dir / "dji-fpv"
    device_dir.mkdir(parents=True)
    write_csv(device_dir / "flight.csv", "2025-05-12,8:27:36 AM,,Failsafe RTH,\n")

    summary = inference.process_evidence(
        evidence_dir=evidence_dir,
        output_dir=tmp_path / "pipeline-output",
        paradigm="message",
        sentiment_pipeline=FakeSentimentPipeline(),
    )

    assert summary["n_logs"] == 1
    assert summary["n_messages_total"] == 1

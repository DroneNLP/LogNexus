import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from lognexus.decode import flatten_group_segments, reconstruct_segments_group_aware
from lognexus.preprocessing import preprocess_log
from lognexus.problem_model import load_droptc
from lognexus.tokenization import PRETOKENIZERS, get_tokenizer


LABELS = {
    "iob2": ["O", "B-SEG", "I-SEG"],
    "bioes": ["O", "B-SEG", "I-SEG", "E-SEG", "S-SEG"],
}
TAG_SCHEMES = ["iob2", "bioes"]

logger = logging.getLogger(__name__)
_TOKENIZER_CACHE: Dict[str, object] = {}


def _configure_torch_runtime():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for LogNexus pipeline inference.") from exc

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch


def _get_cached_tokenizer(name: str):
    if name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[name] = get_tokenizer(name)
    return _TOKENIZER_CACHE[name]


def _start_step() -> Tuple[datetime, float]:
    return datetime.now(), perf_counter()


def _end_step(wall_start: datetime, perf_start: float) -> Dict:
    return {
        "started_at": wall_start.isoformat(timespec="milliseconds"),
        "duration_s": round(perf_counter() - perf_start, 4),
    }


def load_ner_model(
    model_name: str,
    model_type: str,
    tag_scheme: str,
    use_cuda: bool = False,
):
    try:
        from simpletransformers.ner import NERModel
    except ImportError as exc:
        raise ImportError("simpletransformers is required for segment inference.") from exc

    return NERModel(
        model_type=model_type,
        model_name=model_name,
        labels=LABELS[tag_scheme],
        use_cuda=use_cuda,
    )


def load_sentiment_pipeline(model_name: str, device_index: int):
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ImportError("transformers is required for message inference.") from exc

    return hf_pipeline("text-classification", model=model_name, device=device_index)


def _csv_skiprows(csv_path: Path) -> int:
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as file:
        first_line = file.readline().strip()
    if first_line == "sep=,":
        return 1
    if "CUSTOM.date [local]" not in first_line and "CUSTOM.updateTime [local]" not in first_line:
        return 1
    return 0


def extract_messages(csv_path: Path) -> List[Dict]:
    """
    Extract APP.message, APP.tip, and APP.warning rows from a decrypted DJI CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evidence file not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        skiprows=_csv_skiprows(csv_path),
        encoding="utf-8-sig",
        low_memory=False,
    )
    df.columns = df.columns.str.strip()

    messages: List[Dict] = []
    for _, row in df.iterrows():
        date = str(row.get("CUSTOM.date [local]", ""))
        time_value = str(row.get("CUSTOM.updateTime [local]", ""))
        for column, message_type in [
            ("APP.message", "message"),
            ("APP.tip", "tip"),
            ("APP.warning", "warning"),
        ]:
            if column in df.columns and pd.notna(row.get(column)) and str(row[column]).strip():
                raw = str(row[column]).strip()
                messages.append(
                    {
                        "date": date,
                        "time": time_value,
                        "message_type": message_type,
                        "raw_message": raw,
                        "message": preprocess_log(raw),
                    }
                )
    return messages


def predict_segments(model, messages: List[Dict], pretokenizer: str, tag_scheme: str) -> List[Dict]:
    tokenizer = _get_cached_tokenizer(pretokenizer)
    all_token_texts: List[List[str]] = []
    empty_indices: set[int] = set()

    for index, message in enumerate(messages):
        token_texts = [token.text for token in tokenizer.tokenize(message["raw_message"])]
        all_token_texts.append(token_texts)
        if not token_texts:
            empty_indices.add(index)

    sentences = [
        " ".join(tokens)
        for index, tokens in enumerate(all_token_texts)
        if index not in empty_indices
    ]

    batch_predictions: List[List[str]] = []
    if sentences:
        predictions, _raw_outputs = model.predict(sentences)
        for sentence_predictions in predictions:
            batch_predictions.append(
                [
                    label
                    for token_dict in sentence_predictions
                    for _token, label in token_dict.items()
                ]
            )

    results: List[Dict] = []
    prediction_index = 0
    for index, message in enumerate(messages):
        token_texts = all_token_texts[index]
        if index in empty_indices:
            result = dict(message)
            result["tokens"] = []
            result["predicted_labels"] = []
            result["predicted_groups"] = []
            results.append(result)
            continue

        predicted_tags = batch_predictions[prediction_index]
        prediction_index += 1
        if len(predicted_tags) > len(token_texts):
            predicted_tags = predicted_tags[:len(token_texts)]
        elif len(predicted_tags) < len(token_texts):
            predicted_tags.extend(["O"] * (len(token_texts) - len(predicted_tags)))

        result = dict(message)
        result["tokens"] = token_texts
        result["predicted_labels"] = predicted_tags
        result["predicted_groups"] = reconstruct_segments_group_aware(
            message["raw_message"],
            token_texts,
            predicted_tags,
            pretokenizer,
            tag_scheme,
        )
        results.append(result)

    return results


def build_unique_events(timeline: List[Dict], paradigm: str) -> Tuple[Dict[str, int], List[Dict]]:
    message2id: Dict[str, int] = {}
    unique_events: List[Dict] = []

    if paradigm == "message":
        for entry in timeline:
            raw_text = entry["raw_message"]
            key = raw_text.lower().strip()
            if key not in message2id:
                event_id = len(message2id)
                message2id[key] = event_id
                unique_events.append(
                    {
                        "event_id": event_id,
                        "date": entry["date"],
                        "time": entry["time"],
                        "message": raw_text,
                        "predicted_label": None,
                    }
                )
        return message2id, unique_events

    for entry in timeline:
        for segment in flatten_group_segments(entry.get("predicted_groups", [])):
            key = segment.lower().strip()
            if not key:
                continue
            if key not in message2id:
                event_id = len(message2id)
                message2id[key] = event_id
                unique_events.append(
                    {
                        "event_id": event_id,
                        "date": entry["date"],
                        "time": entry["time"],
                        "message": entry["raw_message"],
                        "segment": segment,
                        "predicted_label": None,
                    }
                )
    return message2id, unique_events


def build_unique_events_from_labeled(timeline: List[Dict], paradigm: str) -> List[Dict]:
    seen: Dict[str, int] = {}
    unique_events: List[Dict] = []

    if paradigm == "message":
        for entry in timeline:
            raw_text = entry["raw_message"]
            key = raw_text.lower().strip()
            if key not in seen:
                event_id = len(seen)
                seen[key] = event_id
                unique_events.append(
                    {
                        "event_id": event_id,
                        "date": entry["date"],
                        "time": entry["time"],
                        "message": raw_text,
                        "predicted_label": entry.get("predicted_label", "Unknown"),
                    }
                )
        return unique_events

    for entry in timeline:
        for group in entry.get("predicted_groups", []):
            for segment_info in group.get("segments", []):
                segment = segment_info["text"]
                key = segment.lower().strip()
                if not key:
                    continue
                if key not in seen:
                    event_id = len(seen)
                    seen[key] = event_id
                    unique_events.append(
                        {
                            "event_id": event_id,
                            "date": entry["date"],
                            "time": entry["time"],
                            "message": entry["raw_message"],
                            "segment": segment,
                            "raw_label": segment_info.get("raw_label", "Unknown"),
                            "predicted_label": segment_info.get("predicted_label", "Unknown"),
                        }
                    )
    return unique_events


def predict_labels_message(unique_events: List[Dict], sentiment_pipeline, batch_size: int = 32) -> List[Dict]:
    if not unique_events:
        return unique_events

    texts = [event["message"] for event in unique_events]
    results = sentiment_pipeline(texts, batch_size=batch_size, truncation=True)
    for event, result in zip(unique_events, results):
        raw_label = result["label"].lower()
        event["predicted_label"] = "Normal" if raw_label == "positive" else "Problem"
    return unique_events


def _predict_segment_texts(
    texts: List[str],
    droptc_model,
    droptc_tokenizer,
    idx2label: Dict[str, str],
    device,
    max_seq_length: int,
    batch_size: int,
) -> List[str]:
    if not texts:
        return []

    import torch

    all_labels: List[str] = []
    droptc_model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = droptc_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            logits = droptc_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = logits.argmax(dim=-1).cpu().tolist()
        all_labels.extend(idx2label[str(prediction)] for prediction in predictions)
    return all_labels


def predict_labels_segment(
    unique_events: List[Dict],
    droptc_model,
    droptc_tokenizer,
    idx2label: Dict[str, str],
    device,
    max_seq_length: int = 128,
    batch_size: int = 32,
) -> List[Dict]:
    labels = _predict_segment_texts(
        [event["segment"] for event in unique_events],
        droptc_model,
        droptc_tokenizer,
        idx2label,
        device,
        max_seq_length,
        batch_size,
    )
    for event, label in zip(unique_events, labels):
        event["raw_label"] = label
        event["predicted_label"] = "Normal" if label == "Normal" else "Problem"
    return unique_events


def predict_labels_all_message(timeline: List[Dict], sentiment_pipeline, batch_size: int = 32) -> List[Dict]:
    if not timeline:
        return timeline

    results = sentiment_pipeline(
        [entry["raw_message"] for entry in timeline],
        batch_size=batch_size,
        truncation=True,
    )
    for entry, result in zip(timeline, results):
        raw_label = result["label"].lower()
        entry["predicted_label"] = "Normal" if raw_label == "positive" else "Problem"
    return timeline


def predict_labels_all_segment(
    timeline: List[Dict],
    droptc_model,
    droptc_tokenizer,
    idx2label: Dict[str, str],
    device,
    max_seq_length: int = 128,
    batch_size: int = 32,
) -> List[Dict]:
    all_pairs: List[Tuple[int, str]] = []
    for index, entry in enumerate(timeline):
        for segment in flatten_group_segments(entry.get("predicted_groups", [])):
            all_pairs.append((index, segment))

    labels = _predict_segment_texts(
        [segment for _index, segment in all_pairs],
        droptc_model,
        droptc_tokenizer,
        idx2label,
        device,
        max_seq_length,
        batch_size,
    )

    entry_segments: Dict[int, List[Dict]] = {index: [] for index in range(len(timeline))}
    for (entry_index, segment), label in zip(all_pairs, labels):
        entry_segments[entry_index].append(
            {
                "text": segment,
                "raw_label": label,
                "predicted_label": "Normal" if label == "Normal" else "Problem",
            }
        )

    for index, entry in enumerate(timeline):
        labeled = entry_segments[index]
        label_index = 0
        for group in entry.get("predicted_groups", []):
            count = len(group.get("segments", []))
            group["segments"] = labeled[label_index:label_index + count]
            label_index += count
    return timeline


def export_unique_events(unique_events: List[Dict], output_dir: Path, paradigm: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if paradigm == "message":
        columns = ["event_id", "date", "time", "message", "predicted_label"]
    else:
        columns = ["event_id", "date", "time", "message", "segment", "raw_label", "predicted_label"]

    df = pd.DataFrame(unique_events)
    df = df[[column for column in columns if column in df.columns]]
    output_path = output_dir / "unique_events.xlsx"
    df.to_excel(output_path, index=False)
    return output_path


def propagate_labels(
    timeline: List[Dict],
    message2id: Dict[str, int],
    unique_events: List[Dict],
    paradigm: str,
) -> List[Dict]:
    id2label = {event["event_id"]: event["predicted_label"] for event in unique_events}

    for entry in timeline:
        if paradigm == "message":
            event_id = message2id.get(entry["raw_message"].lower().strip())
            entry["event_id"] = event_id
            entry["predicted_label"] = id2label.get(event_id, "Unknown")
            continue

        for group in entry.get("predicted_groups", []):
            labeled_segments = []
            for segment in group.get("segments", []):
                event_id = message2id.get(segment.lower().strip())
                labeled_segments.append(
                    {
                        "text": segment,
                        "event_id": event_id,
                        "predicted_label": id2label.get(event_id, "Unknown"),
                    }
                )
            group["segments"] = labeled_segments
    return timeline


def process_flight_log(
    csv_path: Path,
    paradigm: str,
    output_dir: Path,
    dedup_order: str = "before",
    run_idx: int = 1,
    ner_model=None,
    pretokenizer: Optional[str] = None,
    tag_scheme: Optional[str] = None,
    droptc_model=None,
    droptc_tokenizer=None,
    droptc_config: Optional[Dict] = None,
    sentiment_pipeline=None,
    device=None,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: Dict[str, Dict] = {}

    wall, perf = _start_step()
    messages = extract_messages(csv_path)
    steps["extract"] = _end_step(wall, perf)
    if not messages:
        return {"evidence_file": csv_path.name, "skipped": True, "steps": steps}

    if paradigm == "segment":
        wall, perf = _start_step()
        timeline = predict_segments(ner_model, messages, pretokenizer, tag_scheme)
        steps["segment"] = _end_step(wall, perf)
        segment_count = sum(
            len(flatten_group_segments(entry.get("predicted_groups", [])))
            for entry in timeline
        )
    else:
        timeline = messages
        segment_count = 0

    if dedup_order == "before":
        wall, perf = _start_step()
        message2id, unique_events = build_unique_events(timeline, paradigm)
        steps["dedup"] = _end_step(wall, perf)

        wall, perf = _start_step()
        if paradigm == "message":
            unique_events = predict_labels_message(unique_events, sentiment_pipeline)
        else:
            unique_events = predict_labels_segment(
                unique_events,
                droptc_model,
                droptc_tokenizer,
                droptc_config["idx2label"],
                device,
                max_seq_length=droptc_config.get("max_seq_length", 128),
            )
        steps["predict"] = _end_step(wall, perf)

        wall, perf = _start_step()
        export_unique_events(unique_events, output_dir, paradigm)
        steps["export"] = _end_step(wall, perf)

        wall, perf = _start_step()
        timeline = propagate_labels(timeline, message2id, unique_events, paradigm)
        steps["propagate"] = _end_step(wall, perf)

    else:
        wall, perf = _start_step()
        if paradigm == "message":
            timeline = predict_labels_all_message(timeline, sentiment_pipeline)
        else:
            timeline = predict_labels_all_segment(
                timeline,
                droptc_model,
                droptc_tokenizer,
                droptc_config["idx2label"],
                device,
                max_seq_length=droptc_config.get("max_seq_length", 128),
            )
        steps["predict"] = _end_step(wall, perf)

        wall, perf = _start_step()
        unique_events = build_unique_events_from_labeled(timeline, paradigm)
        steps["dedup"] = _end_step(wall, perf)

        wall, perf = _start_step()
        export_unique_events(unique_events, output_dir, paradigm)
        steps["export"] = _end_step(wall, perf)

    wall, perf = _start_step()
    if paradigm == "segment":
        with open(output_dir / "prediction.json", "w", encoding="utf-8") as file:
            json.dump(
                [
                    {
                        "raw_message": entry["raw_message"],
                        "tokens": entry.get("tokens", []),
                        "true_labels": None,
                        "predicted_labels": entry.get("predicted_labels", []),
                    }
                    for entry in timeline
                ],
                file,
                indent=2,
                ensure_ascii=False,
            )
        for entry in timeline:
            entry.pop("tokens", None)
            entry.pop("predicted_labels", None)

    with open(output_dir / "timeline.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "evidence_file": csv_path.name,
                "paradigm": paradigm,
                "dedup_order": dedup_order,
                "total_messages": len(timeline),
                "timeline": timeline,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    steps["report"] = _end_step(wall, perf)

    timing = {
        "evidence_file": csv_path.name,
        "paradigm": paradigm,
        "dedup_order": dedup_order,
        "run_idx": run_idx,
        "n_messages": len(messages),
        "n_unique_events": len(unique_events),
        "steps": steps,
        "total_duration_s": round(sum(step["duration_s"] for step in steps.values()), 4),
    }
    if paradigm == "segment":
        timing["n_segments"] = segment_count

    with open(output_dir / "timing.json", "w", encoding="utf-8") as file:
        json.dump(timing, file, indent=2, ensure_ascii=False)
    return timing


def process_evidence(
    evidence_dir: Path,
    output_dir: Path,
    paradigm: str,
    dedup_order: str = "before",
    run_idx: int = 1,
    model_load_timing: Optional[Dict] = None,
    ner_model=None,
    pretokenizer: Optional[str] = None,
    tag_scheme: Optional[str] = None,
    droptc_model=None,
    droptc_tokenizer=None,
    droptc_config: Optional[Dict] = None,
    sentiment_pipeline=None,
    device=None,
) -> Dict:
    run_dir = Path(output_dir) / f"{paradigm}-{dedup_order}" / f"run-{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    evidence_dir = Path(evidence_dir)
    device_dirs = sorted(path for path in evidence_dir.iterdir() if path.is_dir())
    if not device_dirs:
        device_dirs = [evidence_dir]

    overall_wall_start = datetime.now()
    overall_perf_start = perf_counter()
    all_timings: List[Dict] = []

    for device_dir in device_dirs:
        for csv_path in sorted(device_dir.glob("*.csv")):
            log_output = run_dir / device_dir.name / csv_path.stem
            all_timings.append(
                process_flight_log(
                    csv_path=csv_path,
                    paradigm=paradigm,
                    output_dir=log_output,
                    dedup_order=dedup_order,
                    run_idx=run_idx,
                    ner_model=ner_model,
                    pretokenizer=pretokenizer,
                    tag_scheme=tag_scheme,
                    droptc_model=droptc_model,
                    droptc_tokenizer=droptc_tokenizer,
                    droptc_config=droptc_config,
                    sentiment_pipeline=sentiment_pipeline,
                    device=device,
                )
            )

    step_totals: Dict[str, float] = {}
    n_messages_total = 0
    n_unique_total = 0
    n_segments_total = 0
    for timing in all_timings:
        if timing.get("skipped"):
            continue
        n_messages_total += timing.get("n_messages", 0)
        n_unique_total += timing.get("n_unique_events", 0)
        n_segments_total += timing.get("n_segments", 0)
        for step_name, step_data in timing.get("steps", {}).items():
            step_totals[step_name] = round(
                step_totals.get(step_name, 0.0) + step_data["duration_s"],
                4,
            )

    summary = {
        "paradigm": paradigm,
        "dedup_order": dedup_order,
        "run_idx": run_idx,
        "started_at": overall_wall_start.isoformat(timespec="milliseconds"),
        "n_logs": len([timing for timing in all_timings if not timing.get("skipped")]),
        "n_messages_total": n_messages_total,
        "n_unique_events_total": n_unique_total,
        "model_load": model_load_timing or {},
        "step_totals_s": {
            step_name: {"total_duration_s": duration}
            for step_name, duration in step_totals.items()
        },
        "grand_total_duration_s": round(perf_counter() - overall_perf_start, 4),
        "per_log": all_timings,
    }
    if paradigm == "segment":
        summary["n_segments_total"] = n_segments_total

    with open(run_dir / "timing_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SoPID-style problem identification pipeline for drone flight logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--paradigm", choices=["message", "segment"], required=True)
    parser.add_argument("--evidence-dir", type=Path, default=Path("./evidence"))
    parser.add_argument("--output-dir", type=Path, default=Path("./pipeline-output"))
    parser.add_argument("--dedup-order", choices=["before", "after"], default="before")
    parser.add_argument("--run-idx", type=int, default=1)
    parser.add_argument("--model-name", default="swardiantara/SoPID-bert-base-cased")
    parser.add_argument(
        "--model-type",
        choices=["bert", "mobilebert", "roberta", "albert", "distilbert"],
        default="bert",
    )
    parser.add_argument("--pretokenizer", choices=PRETOKENIZERS, default="spacy")
    parser.add_argument("--tag-scheme", choices=TAG_SCHEMES, default="bioes")
    parser.add_argument("--droptc-model-dir", type=Path, default=Path("best-model/droptc"))
    parser.add_argument("--sentiment-model", default="swardiantara/drone-sentiment")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)

    if not args.evidence_dir.is_dir():
        logger.error("Evidence directory does not exist: %s", args.evidence_dir)
        return 1

    try:
        torch = _configure_torch_runtime()
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        use_cuda = device.type == "cuda"

        model_load_timing: Dict[str, Dict] = {}
        ner_model = None
        droptc_model = None
        droptc_tokenizer = None
        droptc_config = None
        sentiment_pipeline = None

        if args.paradigm == "segment":
            wall, perf = _start_step()
            ner_model = load_ner_model(
                args.model_name,
                args.model_type,
                args.tag_scheme,
                use_cuda=use_cuda,
            )
            model_load_timing["ner_segmenter"] = _end_step(wall, perf)

            wall, perf = _start_step()
            droptc_model, droptc_tokenizer, droptc_config = load_droptc(
                device,
                model_dir=args.droptc_model_dir,
            )
            model_load_timing["droptc_classifier"] = _end_step(wall, perf)
        else:
            wall, perf = _start_step()
            sentiment_pipeline = load_sentiment_pipeline(
                args.sentiment_model,
                device_index=0 if use_cuda else -1,
            )
            model_load_timing["sentiment_classifier"] = _end_step(wall, perf)

        process_evidence(
            evidence_dir=args.evidence_dir,
            output_dir=args.output_dir,
            paradigm=args.paradigm,
            dedup_order=args.dedup_order,
            run_idx=args.run_idx,
            model_load_timing=model_load_timing,
            ner_model=ner_model,
            pretokenizer=args.pretokenizer,
            tag_scheme=args.tag_scheme,
            droptc_model=droptc_model,
            droptc_tokenizer=droptc_tokenizer,
            droptc_config=droptc_config,
            sentiment_pipeline=sentiment_pipeline,
            device=device,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

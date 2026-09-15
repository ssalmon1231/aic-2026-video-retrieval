"""Strict JSON loaders and CSV/ZIP exporters for AIC submissions."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .contracts import Task
from .evaluation import (
    EvaluationError,
    FrameInterval,
    KisGroundTruth,
    KisResponse,
    MAX_RESPONSES,
    QaGroundTruth,
    QaResponse,
    TrakeGroundTruth,
    TrakeResponse,
    validate_ranked_responses,
)

GroundTruth = KisGroundTruth | QaGroundTruth | TrakeGroundTruth
Response = KisResponse | QaResponse | TrakeResponse


def load_query(path: str | Path) -> tuple[Task, GroundTruth, tuple[Response, ...]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read evaluation input {source}: {error}") from error
    if not isinstance(payload, dict):
        raise EvaluationError("evaluation input root must be an object")
    return parse_query(payload)


def parse_query(payload: dict[str, Any]) -> tuple[Task, GroundTruth, tuple[Response, ...]]:
    task = _parse_task(payload.get("task"))
    ground_truth_payload = _require_object(payload, "ground_truth")
    responses_payload = payload.get("responses")
    if not isinstance(responses_payload, list):
        raise EvaluationError("responses must be an array")

    if task is Task.TEXTUAL_KIS:
        ground_truth: GroundTruth = KisGroundTruth(
            _string(ground_truth_payload, "video_id"),
            _interval(ground_truth_payload),
        )
        responses: tuple[Response, ...] = tuple(
            KisResponse(_string(_object(value, rank), "video_id"), _integer(_object(value, rank), "frame_id"))
            for rank, value in enumerate(responses_payload, start=1)
        )
    elif task is Task.QA:
        answers = ground_truth_payload.get("answers")
        if not isinstance(answers, list) or any(not isinstance(answer, str) for answer in answers):
            raise EvaluationError("ground_truth.answers must be an array of strings")
        ground_truth = QaGroundTruth(
            _string(ground_truth_payload, "video_id"),
            _interval(ground_truth_payload),
            tuple(answers),
        )
        responses = tuple(
            QaResponse(
                _string(_object(value, rank), "video_id"),
                _integer(_object(value, rank), "frame_id"),
                _string(_object(value, rank), "answer"),
            )
            for rank, value in enumerate(responses_payload, start=1)
        )
    else:
        interval_payloads = ground_truth_payload.get("intervals")
        if not isinstance(interval_payloads, list):
            raise EvaluationError("ground_truth.intervals must be an array")
        ground_truth = TrakeGroundTruth(
            _string(ground_truth_payload, "video_id"),
            tuple(_interval(_object(value, index)) for index, value in enumerate(interval_payloads, start=1)),
        )
        parsed: list[Response] = []
        for rank, value in enumerate(responses_payload, start=1):
            response = _object(value, rank)
            frame_ids = response.get("frame_ids")
            if not isinstance(frame_ids, list) or any(
                isinstance(frame_id, bool) or not isinstance(frame_id, int)
                for frame_id in frame_ids
            ):
                raise EvaluationError(f"response {rank}.frame_ids must be an array of integers")
            parsed.append(TrakeResponse(_string(response, "video_id"), tuple(frame_ids)))
        responses = tuple(parsed)

    validate_ranked_responses(responses)
    return task, ground_truth, responses


def write_submission_csv(path: str | Path, responses: Sequence[Response]) -> Path:
    """Write one headerless UTF-8 CSV required for one official query."""
    destination = Path(path)
    if destination.suffix.lower() != ".csv":
        raise EvaluationError("submission output must use the .csv extension")
    if not destination.name or destination.name != Path(destination.name).name:
        raise EvaluationError("submission CSV must use a plain filename")
    _validate_submission_responses(responses)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for response in responses:
            writer.writerow(_csv_row(response))
    return destination


def build_submission_zip(
    destination: str | Path,
    query_responses: Mapping[str, Sequence[Response]],
) -> Path:
    """Build official ZIP with headerless per-query CSVs inside `submission/`."""
    target = Path(destination)
    if target.suffix.lower() != ".zip":
        raise EvaluationError("submission package must use the .zip extension")
    if not query_responses:
        raise EvaluationError("submission package requires at least one query CSV")
    names = tuple(query_responses)
    if len(set(names)) != len(names):
        raise EvaluationError("submission CSV filenames must be unique")
    for name in names:
        _validate_submission_filename(name)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "submission"
            for name, responses in query_responses.items():
                write_submission_csv(root / name, responses)
            with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=9) as archive:
                for csv_path in sorted(root.glob("*.csv")):
                    archive.write(csv_path, csv_path.relative_to(root.parent).as_posix())
        temporary.replace(target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def _validate_submission_responses(responses: Sequence[Response]) -> None:
    validate_ranked_responses(responses)
    if any(not isinstance(response, (KisResponse, QaResponse, TrakeResponse)) for response in responses):
        raise EvaluationError("submission responses have an unsupported type")
    for response in responses:
        _validate_submission_video_id(response.video_id)
        if isinstance(response, QaResponse) and len(response.answer) > 100:
            raise EvaluationError("Q&A response answer exceeds 100 characters")


def _csv_row(response: Response) -> tuple[str | int, ...]:
    if isinstance(response, KisResponse):
        return response.video_id, response.frame_id
    if isinstance(response, QaResponse):
        return response.video_id, response.frame_id, response.answer
    if isinstance(response, TrakeResponse):
        return (response.video_id, *response.frame_ids)
    raise EvaluationError("submission response has an unsupported type")


def _validate_submission_video_id(video_id: str) -> None:
    if video_id.endswith(".mp4"):
        raise EvaluationError("submission video_id must not include .mp4")
    if any(character in video_id for character in ",\r\n"):
        raise EvaluationError("submission video_id contains CSV-special characters")


def _validate_submission_filename(name: str) -> None:
    path = Path(name)
    if not isinstance(name, str) or path.name != name or path.suffix.lower() != ".csv":
        raise EvaluationError("submission CSV filename must be a plain .csv filename")


def _parse_task(value: Any) -> Task:
    try:
        return Task(value)
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"unsupported task: {value!r}") from error


def _interval(payload: dict[str, Any]) -> FrameInterval:
    return FrameInterval(_integer(payload, "start"), _integer(payload, "end"))


def _require_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise EvaluationError(f"{key} must be an object")
    return value


def _object(value: Any, position: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"item {position} must be an object")
    return value


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise EvaluationError(f"{key} must be a string")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{key} must be an integer")
    return value

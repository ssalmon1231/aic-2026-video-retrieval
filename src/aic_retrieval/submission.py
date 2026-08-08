"""Strict JSON loaders for AIC ground truth and ranked responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import Task
from .evaluation import (
    EvaluationError,
    FrameInterval,
    KisGroundTruth,
    KisResponse,
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

"""Deterministic AIC task scoring and top-k metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from .contracts import Task

CUTOFFS = (1, 5, 20, 50, 100)
MAX_RESPONSES = 100


class EvaluationError(ValueError):
    """Raised when ground truth or ranked responses violate the contract."""


@dataclass(frozen=True, slots=True)
class FrameInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise EvaluationError(
                f"invalid frame interval [{self.start}, {self.end}]"
            )

    def contains(self, frame_id: int) -> bool:
        return self.start <= frame_id <= self.end


@dataclass(frozen=True, slots=True)
class KisGroundTruth:
    video_id: str
    interval: FrameInterval

    def __post_init__(self) -> None:
        _require_video_id(self.video_id)


@dataclass(frozen=True, slots=True)
class QaGroundTruth:
    video_id: str
    interval: FrameInterval
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_video_id(self.video_id)
        if not self.answers or any(not answer.strip() for answer in self.answers):
            raise EvaluationError("Q&A ground truth requires non-empty answers")


@dataclass(frozen=True, slots=True)
class TrakeGroundTruth:
    video_id: str
    intervals: tuple[FrameInterval, ...]

    def __post_init__(self) -> None:
        _require_video_id(self.video_id)
        if not self.intervals:
            raise EvaluationError("TRAKE ground truth requires at least one event")


@dataclass(frozen=True, slots=True)
class KisResponse:
    video_id: str
    frame_id: int

    def __post_init__(self) -> None:
        _require_response_identity(self.video_id, (self.frame_id,))


@dataclass(frozen=True, slots=True)
class QaResponse:
    video_id: str
    frame_id: int
    answer: str

    def __post_init__(self) -> None:
        _require_response_identity(self.video_id, (self.frame_id,))
        if not self.answer.strip():
            raise EvaluationError("Q&A response answer must not be empty")


@dataclass(frozen=True, slots=True)
class TrakeResponse:
    video_id: str
    frame_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_response_identity(self.video_id, self.frame_ids)
        if not self.frame_ids:
            raise EvaluationError("TRAKE response requires at least one frame")
        if any(current <= previous for previous, current in zip(self.frame_ids, self.frame_ids[1:])):
            raise EvaluationError("TRAKE response frames must be strictly increasing")


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    task: Task
    matcher: str
    response_scores: tuple[float, ...]
    recall_at: tuple[tuple[int, float], ...]
    final_score: float
    query_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "task": self.task.value,
            "matcher": self.matcher,
            "response_scores": list(self.response_scores),
            "recall_at": {str(cutoff): score for cutoff, score in self.recall_at},
            "final_score": self.final_score,
        }


AnswerMatcher = Callable[[str, tuple[str, ...]], bool]
GroundTruthT = TypeVar("GroundTruthT")
ResponseT = TypeVar("ResponseT")


def exact_answer_matcher(answer: str, expected: tuple[str, ...]) -> bool:
    normalized = _normalize_answer(answer)
    return any(normalized == _normalize_answer(candidate) for candidate in expected)


def score_kis(ground_truth: KisGroundTruth, response: KisResponse) -> float:
    return float(
        response.video_id == ground_truth.video_id
        and ground_truth.interval.contains(response.frame_id)
    )


def score_qa(
    ground_truth: QaGroundTruth,
    response: QaResponse,
    *,
    answer_matcher: AnswerMatcher = exact_answer_matcher,
) -> float:
    localized = (
        response.video_id == ground_truth.video_id
        and ground_truth.interval.contains(response.frame_id)
    )
    return float(localized and answer_matcher(response.answer, ground_truth.answers))


def score_trake(ground_truth: TrakeGroundTruth, response: TrakeResponse) -> float:
    if len(response.frame_ids) != len(ground_truth.intervals):
        raise EvaluationError(
            f"TRAKE event count {len(response.frame_ids)} != expected "
            f"{len(ground_truth.intervals)}"
        )
    if response.video_id != ground_truth.video_id:
        return 0.0
    matched = sum(
        interval.contains(frame_id)
        for interval, frame_id in zip(ground_truth.intervals, response.frame_ids)
    )
    return matched / len(ground_truth.intervals)


def evaluate_ranked(
    task: Task,
    ground_truth: GroundTruthT,
    responses: Sequence[ResponseT],
    *,
    answer_matcher: AnswerMatcher = exact_answer_matcher,
    matcher_name: str = "exact",
    query_id: str = "",
) -> QueryMetrics:
    validate_ranked_responses(responses)
    if task is Task.TEXTUAL_KIS:
        if not isinstance(ground_truth, KisGroundTruth) or any(
            not isinstance(response, KisResponse) for response in responses
        ):
            raise EvaluationError("Textual KIS records have the wrong type")
        scores = tuple(score_kis(ground_truth, response) for response in responses)
        matcher_name = "exact"
    elif task is Task.QA:
        if not isinstance(ground_truth, QaGroundTruth) or any(
            not isinstance(response, QaResponse) for response in responses
        ):
            raise EvaluationError("Q&A records have the wrong type")
        scores = tuple(
            score_qa(ground_truth, response, answer_matcher=answer_matcher)
            for response in responses
        )
    elif task is Task.TRAKE:
        if not isinstance(ground_truth, TrakeGroundTruth) or any(
            not isinstance(response, TrakeResponse) for response in responses
        ):
            raise EvaluationError("TRAKE records have the wrong type")
        scores = tuple(score_trake(ground_truth, response) for response in responses)
        matcher_name = "exact"
    else:
        raise EvaluationError(f"unsupported task: {task}")

    recall_at = tuple((cutoff, _best_at(scores, cutoff)) for cutoff in CUTOFFS)
    final_score = sum(score for _, score in recall_at) / len(recall_at)
    return QueryMetrics(task, matcher_name, scores, recall_at, final_score, query_id)


def validate_ranked_responses(responses: Sequence[object]) -> None:
    if len(responses) > MAX_RESPONSES:
        raise EvaluationError(
            f"response count {len(responses)} exceeds maximum {MAX_RESPONSES}"
        )
    seen: set[object] = set()
    for rank, response in enumerate(responses, start=1):
        if response in seen:
            raise EvaluationError(f"duplicate response at rank {rank}")
        seen.add(response)


def aggregate_queries(metrics: Sequence[QueryMetrics]) -> float:
    if not metrics:
        return 0.0
    return sum(metric.final_score for metric in metrics) / len(metrics)


def _best_at(scores: Sequence[float], cutoff: int) -> float:
    return max(scores[:cutoff], default=0.0)


def _normalize_answer(value: str) -> str:
    return " ".join(value.casefold().split())


def _require_video_id(video_id: str) -> None:
    if not video_id.strip():
        raise EvaluationError("video_id must not be empty")


def _require_response_identity(video_id: str, frame_ids: Sequence[int]) -> None:
    _require_video_id(video_id)
    if any(
        isinstance(frame_id, bool)
        or not isinstance(frame_id, int)
        or frame_id < 0
        for frame_id in frame_ids
    ):
        raise EvaluationError("frame IDs must be non-negative integers")

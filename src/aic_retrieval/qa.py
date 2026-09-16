"""Automatic Q&A contracts and deterministic bilingual query routing."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, Sequence

from .evaluation import MAX_RESPONSES, QaResponse

ANSWER_TYPES = frozenset(
    {
        "count",
        "visible_text",
        "name",
        "number",
        "speech",
        "color",
        "person",
        "location",
        "object",
        "action",
        "unknown",
    }
)

_QUESTION_MARKER = re.compile(
    r"(?<!\w)(?:câu\s+hỏi|cau\s+hoi|question)\s*[:：]",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?:\s+)")
_ALLOWED_CONTROLS = frozenset("\t\n\r")

_VISIBLE_TEXT_PATTERNS = (
    r"\b(?:biển|bảng|màn hình|nhãn|dòng chữ|chữ)\b.*\b(?:ghi|viết|hiển thị|đọc)\b",
    r"\b(?:ghi|viết|hiển thị)\s+(?:gì|chữ gì|nội dung gì)\b",
    r"\bwhat\s+(?:is|was)\s+(?:written|displayed)\b",
    r"\bwhat\s+does\s+(?:the\s+)?(?:sign|screen|label|text)\s+(?:say|read)\b",
    r"\b(?:read|written|displayed)\s+(?:on|in)\b",
)
_NUMBER_PATTERNS = (
    r"\b(?:biển số|số áo|số hiệu|tỷ số|điểm số)\b",
    r"\b(?:tỉ|tỷ)\s+số\s+(?:là\s+)?(?:bao nhiêu|mấy|gì)\b",
    r"\b(?:số|number)\s+(?:mấy|nào|gì|bao nhiêu)\b",
    r"\bwhat\s+(?:is|was)\s+(?:the\s+)?score\b",
    r"\bwhat\s+number\b",
    r"\b(?:jersey|license|plate|score)\s+number\b",
)
_COUNT_PATTERNS = (
    r"\bbao\s+nhiêu\b",
    r"\bmấy\s+(?:người|con|chiếc|cái|vật|xe|lần)\b",
    r"\bsố\s+lượng\b",
    r"\bhow\s+many\b",
    r"\bnumber\s+of\b",
)
_SPEECH_PATTERNS = (
    r"\b(?:nói|phát biểu|hỏi|trả lời)\s+(?:gì|điều gì|câu gì)\b",
    r"\bnghe\s+(?:thấy\s+)?(?:gì|điều gì)\b",
    r"\bwhat\s+(?:did|does|do|is|was|were)\b.*\b(?:say|saying|ask|answer)\b",
    r"\bwhat\s+can\s+be\s+heard\b",
)
_COLOR_PATTERNS = (
    r"\bmàu\s+(?:gì|nào)\b",
    r"\bwhat\s+colou?r\b",
    r"\bwhich\s+colou?r\b",
)
_NAME_PATTERNS = (
    r"\btên\b.*\b(?:gì|là gì|nào)\b",
    r"\b(?:tên|name)\s+(?:của|of)\b",
    r"\bwhat(?:'s|\s+is|\s+was)\b.*\bname\b",
)
_PERSON_PATTERNS = (
    r"\bai\b",
    r"\bngười\s+(?:nào|đó)\s+là\b",
    r"\bwho\b",
)
_LOCATION_PATTERNS = (
    r"\bở\s+đâu\b",
    r"\bđịa\s+điểm\s+(?:nào|gì)\b",
    r"\bnơi\s+(?:nào|đâu)\b",
    r"\bwhere\b",
    r"\bwhat\s+(?:place|location)\b",
)
_ACTION_PATTERNS = (
    r"\b(?:đang|đã|sẽ)\s+làm\s+gì\b",
    r"\b(?:đã|sẽ)\s+(?:xảy ra|diễn ra)\s+(?:gì|thế nào)\b",
    r"\b(?:chuyện|điều)\s+gì\s+(?:đang\s+)?xảy ra\b",
    r"\bhành\s+động\s+(?:gì|nào)\b",
    r"\bwhat\s+(?:is|are|was|were)\b.*\bdoing\b",
    r"\bwhat\s+happens?\b",
    r"\bwhat\s+action\b",
)
_OBJECT_PATTERNS = (
    r"\b(?:vật|đồ vật|đối tượng|loại xe|phương tiện)\s+(?:gì|nào)\b",
    r"\b(?:cầm|giữ|mang|lấy|đặt|ăn|uống|mua)\s+(?:cái|vật|thứ|đồ)?\s*gì\b",
    r"\b(?:cái|chiếc|con)\s+gì\b",
    r"\bwhat\s+(?:object|item|vehicle|animal)\b",
    r"\bwhich\s+(?:object|item|vehicle|animal)\b",
    r"\bwhat\b.*\b(?:holding|carrying|wearing|buying|eating|drinking)\b",
)


class QaError(ValueError):
    """Raised when an automatic Q&A contract or raw prompt is invalid."""


class AutomaticQaPipeline(Protocol):
    """Prompt-only public boundary implemented by later automatic Q&A phases."""

    def answer_query(
        self,
        raw_text: str,
    ) -> tuple[tuple[QaResponse, ...], QaPipelineStatus]: ...


@dataclass(frozen=True, slots=True)
class QaQuery:
    raw_text: str
    event_description: str
    question: str
    answer_type: str
    use_ocr: bool
    use_asr: bool

    def __post_init__(self) -> None:
        _require_text("raw_text", self.raw_text)
        _require_text("event_description", self.event_description)
        _require_text("question", self.question)
        if not isinstance(self.answer_type, str) or self.answer_type not in ANSWER_TYPES:
            raise QaError(
                "answer_type must be one of: " + ", ".join(sorted(ANSWER_TYPES))
            )
        if not isinstance(self.use_ocr, bool) or not isinstance(self.use_asr, bool):
            raise QaError("use_ocr and use_asr must be booleans")
        if self.use_ocr != (self.answer_type in {"visible_text", "name", "number"}):
            raise QaError("use_ocr disagrees with answer_type routing")
        if self.use_asr != (self.answer_type == "speech"):
            raise QaError("use_asr disagrees with answer_type routing")


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    video_id: str
    start_frame_id: int
    end_frame_id: int
    seed_frame_ids: tuple[int, ...]
    sampled_frame_ids: tuple[int, ...]
    retrieval_score: float
    raw_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_text("video_id", self.video_id)
        _require_frame_id("start_frame_id", self.start_frame_id)
        _require_frame_id("end_frame_id", self.end_frame_id)
        if self.end_frame_id < self.start_frame_id:
            raise QaError("end_frame_id must be at least start_frame_id")
        _require_frame_tuple("seed_frame_ids", self.seed_frame_ids)
        _require_frame_tuple("sampled_frame_ids", self.sampled_frame_ids)
        if tuple(sorted(self.seed_frame_ids)) != self.seed_frame_ids:
            raise QaError("seed_frame_ids must be sorted")
        if tuple(sorted(self.sampled_frame_ids)) != self.sampled_frame_ids:
            raise QaError("sampled_frame_ids must be sorted")
        if len(set(self.seed_frame_ids)) != len(self.seed_frame_ids):
            raise QaError("seed_frame_ids must not contain duplicates")
        if len(set(self.sampled_frame_ids)) != len(self.sampled_frame_ids):
            raise QaError("sampled_frame_ids must not contain duplicates")
        for name, values in (
            ("seed_frame_ids", self.seed_frame_ids),
            ("sampled_frame_ids", self.sampled_frame_ids),
        ):
            if any(
                frame_id < self.start_frame_id or frame_id > self.end_frame_id
                for frame_id in values
            ):
                raise QaError(f"{name} must stay inside the evidence window")
        _require_score("retrieval_score", self.retrieval_score, -1.0, 1.0)
        if not isinstance(self.raw_ranks, tuple) or not self.raw_ranks:
            raise QaError("raw_ranks must be a non-empty tuple")
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
            for rank in self.raw_ranks
        ):
            raise QaError("raw_ranks must contain positive integers")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    slot: int
    frame_id: int
    image: object

    def __post_init__(self) -> None:
        _require_frame_id("slot", self.slot)
        _require_frame_id("frame_id", self.frame_id)
        if self.image is None:
            raise QaError("image must not be None")

    def to_dict(self) -> dict[str, int]:
        return {"slot": self.slot, "frame_id": self.frame_id}


@dataclass(frozen=True, slots=True)
class AnswerHypothesis:
    video_id: str
    frame_id: int
    raw_answer: str
    normalized_answer: str
    retrieval_score: float
    support_score: float
    agreement_score: float
    joint_score: float

    def __post_init__(self) -> None:
        _require_text("video_id", self.video_id)
        _require_frame_id("frame_id", self.frame_id)
        _require_text("raw_answer", self.raw_answer)
        _require_text("normalized_answer", self.normalized_answer)
        if self.normalized_answer != normalize_answer(self.raw_answer):
            raise QaError("normalized_answer disagrees with raw_answer")
        _require_score("retrieval_score", self.retrieval_score, -1.0, 1.0)
        _require_score("support_score", self.support_score, 0.0, 1.0)
        _require_score("agreement_score", self.agreement_score, 0.0, 1.0)
        _require_finite("joint_score", self.joint_score)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QaPipelineStatus:
    query_parsed: bool
    retrieved_windows: int
    answered_windows: int
    valid_hypotheses: int
    response_count: int
    vlm_circuit_open: bool
    elapsed_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.query_parsed, bool):
            raise QaError("query_parsed must be boolean")
        if not isinstance(self.vlm_circuit_open, bool):
            raise QaError("vlm_circuit_open must be boolean")
        for name in (
            "retrieved_windows",
            "answered_windows",
            "valid_hypotheses",
            "response_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise QaError(f"{name} must be a non-negative integer")
        if self.answered_windows > self.retrieved_windows:
            raise QaError("answered_windows cannot exceed retrieved_windows")
        if self.response_count > MAX_RESPONSES:
            raise QaError(f"response_count cannot exceed {MAX_RESPONSES}")
        _require_finite("elapsed_ms", self.elapsed_ms)
        if self.elapsed_ms < 0:
            raise QaError("elapsed_ms must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_qa_query(raw_text: str) -> QaQuery:
    """Parse one raw bilingual prompt without model calls or content rewriting."""

    _require_text("raw_text", raw_text)
    stripped = raw_text.strip()
    split = _split_explicit_marker(stripped) or _split_final_question(stripped)
    if split is None:
        event_description = stripped
        question = stripped
    else:
        event_description, question = split
    answer_type = route_answer_type(question)
    return QaQuery(
        raw_text=raw_text,
        event_description=event_description,
        question=question,
        answer_type=answer_type,
        use_ocr=answer_type in {"visible_text", "name", "number"},
        use_asr=answer_type == "speech",
    )


def normalize_answer(answer: str) -> str:
    """Normalize an answer only for deduplication and matching."""

    _require_text("answer", answer)
    return " ".join(unicodedata.normalize("NFC", answer).casefold().split())


def route_answer_type(question: str) -> str:
    """Route a question to one bounded answer category using fixed rules."""

    _require_text("question", question)
    normalized = " ".join(unicodedata.normalize("NFC", question).casefold().split())
    routes = (
        ("number", _NUMBER_PATTERNS),
        ("count", _COUNT_PATTERNS),
        ("name", _NAME_PATTERNS),
        ("visible_text", _VISIBLE_TEXT_PATTERNS),
        ("speech", _SPEECH_PATTERNS),
        ("color", _COLOR_PATTERNS),
        ("person", _PERSON_PATTERNS),
        ("location", _LOCATION_PATTERNS),
        ("action", _ACTION_PATTERNS),
        ("object", _OBJECT_PATTERNS),
    )
    for answer_type, patterns in routes:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return answer_type
    return "unknown"


def _split_explicit_marker(text: str) -> tuple[str, str] | None:
    matches = tuple(_QUESTION_MARKER.finditer(text))
    if not matches:
        return None
    marker = matches[-1]
    question = text[marker.end() :].strip()
    if not question:
        return None
    event_description = text[: marker.start()].strip() or text
    return event_description, question


def _split_final_question(text: str) -> tuple[str, str] | None:
    if not text.endswith("?"):
        return None
    boundaries = tuple(_SENTENCE_BOUNDARY.finditer(text))
    if boundaries:
        boundary = boundaries[-1]
        event_description = text[: boundary.start() + 1].strip()
        question = text[boundary.end() :].strip()
        if event_description and _is_single_question(question):
            return event_description, question

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and _is_single_question(lines[-1]):
        return "\n".join(lines[:-1]), lines[-1]
    return None


def _is_single_question(value: str) -> bool:
    return bool(value) and value.endswith("?") and not any(
        punctuation in value[:-1] for punctuation in ".!?"
    )




@dataclass(frozen=True, slots=True)
class QaPipeline:
    """Deterministic orchestration over retrieval, evidence, and answer engine."""

    retrieve: Callable[[QaQuery], Sequence[EvidenceWindow]]
    answer_engine: Any
    load_evidence: Callable[[EvidenceWindow], Sequence[FrameEvidence]] | None = None
    max_responses: int = MAX_RESPONSES

    def __post_init__(self) -> None:
        if not callable(self.retrieve):
            raise QaError("retrieve must be callable")
        if self.answer_engine is None:
            raise QaError("answer_engine must not be None")
        if isinstance(self.max_responses, bool) or not isinstance(self.max_responses, int) or not 1 <= self.max_responses <= MAX_RESPONSES:
            raise QaError(f"max_responses must be in [1, {MAX_RESPONSES}]")

    def answer_query(self, raw_text: str) -> tuple[tuple[QaResponse, ...], QaPipelineStatus]:
        import time

        started = time.perf_counter()
        query_parsed = False
        windows: Sequence[EvidenceWindow] = ()
        hypotheses: list[AnswerHypothesis] = []
        circuit_open = False
        try:
            query = parse_qa_query(raw_text)
        except (QaError, TypeError, ValueError):
            elapsed = (time.perf_counter() - started) * 1000.0
            return (), QaPipelineStatus(False, 0, 0, 0, 0, False, elapsed)
        query_parsed = True
        try:
            windows = tuple(self.retrieve(query))
        except (QaError, RuntimeError, TypeError, ValueError):
            elapsed = (time.perf_counter() - started) * 1000.0
            return (), QaPipelineStatus(True, 0, 0, 0, 0, False, elapsed)

        answered = 0
        for window in windows:
            try:
                evidence = _validate_evidence(
                    window,
                    (
                        tuple(self.load_evidence(window))
                        if self.load_evidence is not None
                        else tuple(
                            FrameEvidence(slot, frame_id, frame_id)
                            for slot, frame_id in enumerate(window.sampled_frame_ids)
                        )
                    ),
                )
                output = self.answer_engine.answer(query.question, query.event_description, evidence)
                from .qa_models import parse_answer_protocol
                parsed = parse_answer_protocol(output, len(evidence))
                frame_id = evidence[parsed.slot].frame_id
                hypotheses.append(AnswerHypothesis(
                    window.video_id,
                    frame_id,
                    parsed.answer,
                    normalize_answer(parsed.answer),
                    window.retrieval_score,
                    1.0,
                    0.0,
                    window.retrieval_score,
                ))
                answered += 1
            except Exception:
                if getattr(self.answer_engine, "circuit_open", False):
                    circuit_open = True
                continue

        hypotheses.sort(key=lambda item: (-item.joint_score, item.video_id, item.frame_id, item.normalized_answer))
        responses: list[QaResponse] = []
        seen: set[tuple[str, int, str]] = set()
        for hypothesis in hypotheses:
            identity = (hypothesis.video_id, hypothesis.frame_id, hypothesis.normalized_answer)
            if identity in seen:
                continue
            seen.add(identity)
            responses.append(QaResponse(hypothesis.video_id, hypothesis.frame_id, hypothesis.raw_answer))
            if len(responses) >= self.max_responses:
                break
        elapsed = (time.perf_counter() - started) * 1000.0
        return tuple(responses), QaPipelineStatus(
            query_parsed, len(windows), answered, len(hypotheses), len(responses), circuit_open, elapsed
        )


def _validate_evidence(
    window: EvidenceWindow,
    evidence: Sequence[FrameEvidence],
) -> tuple[FrameEvidence, ...]:
    if not evidence:
        raise QaError("evidence must not be empty")
    validated = tuple(evidence)
    if any(not isinstance(frame, FrameEvidence) for frame in validated):
        raise QaError("evidence must contain FrameEvidence records")
    if tuple(frame.slot for frame in validated) != tuple(range(len(validated))):
        raise QaError("evidence slots must be contiguous from zero")
    frame_ids = tuple(frame.frame_id for frame in validated)
    if frame_ids != window.sampled_frame_ids:
        raise QaError("evidence frame IDs must match sampled_frame_ids")
    if any(
        frame_id < window.start_frame_id or frame_id > window.end_frame_id
        for frame_id in frame_ids
    ):
        raise QaError("evidence frame IDs must stay inside the evidence window")
    return validated


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise QaError(f"{name} must be a non-empty string")
    if any(
        unicodedata.category(character) == "Cc" and character not in _ALLOWED_CONTROLS
        for character in value
    ):
        raise QaError(f"{name} contains a forbidden control character")


def _require_frame_id(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QaError(f"{name} must be a non-negative integer")


def _require_frame_tuple(name: str, value: Any) -> None:
    if not isinstance(value, tuple) or not value:
        raise QaError(f"{name} must be a non-empty tuple")
    for frame_id in value:
        _require_frame_id(name, frame_id)


def _require_finite(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise QaError(f"{name} must be a finite number")


def _require_score(name: str, value: Any, minimum: float, maximum: float) -> None:
    _require_finite(name, value)
    if not minimum <= value <= maximum:
        raise QaError(f"{name} must be in [{minimum}, {maximum}]")

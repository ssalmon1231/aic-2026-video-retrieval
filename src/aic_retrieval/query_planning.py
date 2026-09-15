"""Bounded private query plans for compositional Textual KIS retrieval."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from .query import normalize_query
from .reranking import PLANNER_MODEL_ID, PLANNER_REVISION, QwenPlanner, RerankingError

MAX_VISUAL_CLAUSES = 4
MAX_ORDERED_EVENTS = 3
MAX_EXACT_TEXTS = 4
MAX_GENERATED_TEXT_LENGTH = 320
MAX_EXACT_TEXT_LENGTH = 160
MAX_NEW_TOKENS = 256
PLAN_KEYS = {
    "holistic_visual",
    "visual_clauses",
    "ordered_events",
    "exact_texts",
}
EVENT_KEYS = {"visual", "exact_text"}
VISUAL_QUERY_KINDS = {"raw", "holistic", "clause"}


class QueryPlanningError(ValueError):
    """Raised when query-plan inputs or generated output violate bounds."""


@dataclass(frozen=True, slots=True)
class VisualQuery:
    text: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in VISUAL_QUERY_KINDS:
            raise QueryPlanningError("visual query kind is invalid")
        maximum = 2000 if self.kind == "raw" else MAX_GENERATED_TEXT_LENGTH
        normalized = _bounded_text(self.text, maximum)
        object.__setattr__(self, "text", normalized)


@dataclass(frozen=True, slots=True)
class OrderedEvent:
    visual: str
    exact_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "visual",
            _bounded_text(self.visual, MAX_GENERATED_TEXT_LENGTH),
        )
        if self.exact_text is not None:
            object.__setattr__(
                self,
                "exact_text",
                _bounded_text(self.exact_text, MAX_EXACT_TEXT_LENGTH),
            )


@dataclass(frozen=True, slots=True)
class QueryPlan:
    visual_queries: tuple[VisualQuery, ...]
    ordered_events: tuple[OrderedEvent, ...]
    exact_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.visual_queries, tuple)
            or not all(isinstance(item, VisualQuery) for item in self.visual_queries)
        ):
            raise QueryPlanningError("visual queries must be a VisualQuery tuple")
        if (
            not isinstance(self.ordered_events, tuple)
            or not all(isinstance(item, OrderedEvent) for item in self.ordered_events)
        ):
            raise QueryPlanningError("ordered events must be an OrderedEvent tuple")
        if not self.visual_queries or self.visual_queries[0].kind != "raw":
            raise QueryPlanningError("query plan must begin with the raw visual query")
        if sum(item.kind == "raw" for item in self.visual_queries) != 1:
            raise QueryPlanningError("query plan must contain exactly one raw visual query")
        if sum(item.kind == "holistic" for item in self.visual_queries) > 1:
            raise QueryPlanningError("query plan permits at most one holistic query")
        if sum(item.kind == "clause" for item in self.visual_queries) > MAX_VISUAL_CLAUSES:
            raise QueryPlanningError("query plan contains too many visual clauses")
        if len(self.ordered_events) > MAX_ORDERED_EVENTS:
            raise QueryPlanningError("query plan contains too many ordered events")
        if not isinstance(self.exact_texts, tuple):
            raise QueryPlanningError("query plan exact texts must be a tuple")
        if len(self.exact_texts) > MAX_EXACT_TEXTS:
            raise QueryPlanningError("query plan contains too many exact texts")

        exact_texts = tuple(
            _bounded_text(value, MAX_EXACT_TEXT_LENGTH) for value in self.exact_texts
        )
        if len(set(exact_texts)) != len(exact_texts):
            raise QueryPlanningError("query plan exact texts must be unique")
        event_texts = tuple(event.exact_text for event in self.ordered_events)
        if any(value is not None and value not in exact_texts for value in event_texts):
            raise QueryPlanningError("ordered event exact text must appear in exact_texts")
        object.__setattr__(self, "exact_texts", exact_texts)


class QwenQueryPlanner:
    """Pinned warm planner whose generated plan remains process-private."""

    def __init__(
        self,
        model_id: str = PLANNER_MODEL_ID,
        revision: str = PLANNER_REVISION,
        *,
        device: str = "cuda",
    ) -> None:
        self._runtime = QwenPlanner(model_id, revision, device=device)

    @property
    def last_generated_tokens(self) -> int:
        return self._runtime.last_generated_tokens

    def plan(self, raw_query: str) -> QueryPlan:
        query = normalize_query(raw_query)
        content = self._runtime.generate_json(
            _planner_prompt(query),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        return parse_query_plan(content, query)


def baseline_query_plan(raw_query: str) -> QueryPlan:
    query = normalize_query(raw_query)
    return QueryPlan((VisualQuery(query, "raw"),), (), ())


def parse_query_plan(content: str, raw_query: str) -> QueryPlan:
    query = normalize_query(raw_query)
    normalized = _normalize_plan_content(content)
    try:
        payload: Any = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise QueryPlanningError("query planner output must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != PLAN_KEYS:
        raise QueryPlanningError("query planner output has invalid fields")

    holistic = _optional_text(payload["holistic_visual"], MAX_GENERATED_TEXT_LENGTH)
    clauses = _unique_texts(
        payload["visual_clauses"],
        MAX_VISUAL_CLAUSES,
        MAX_GENERATED_TEXT_LENGTH,
    )
    exact_texts = _unique_texts(
        payload["exact_texts"],
        MAX_EXACT_TEXTS,
        MAX_EXACT_TEXT_LENGTH,
    )
    events = _ordered_events(payload["ordered_events"], exact_texts)
    if any(value not in query for value in exact_texts):
        raise QueryPlanningError("exact text must be a verbatim raw-query substring")

    visuals = [VisualQuery(query, "raw")]
    if holistic is not None:
        visuals.append(VisualQuery(holistic, "holistic"))
    visuals.extend(VisualQuery(clause, "clause") for clause in clauses)
    return QueryPlan(tuple(visuals), events, exact_texts)


def _ordered_events(value: Any, exact_texts: tuple[str, ...]) -> tuple[OrderedEvent, ...]:
    if not isinstance(value, list) or len(value) > MAX_ORDERED_EVENTS:
        raise QueryPlanningError("ordered_events must be a bounded JSON array")
    events: list[OrderedEvent] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != EVENT_KEYS:
            raise QueryPlanningError("ordered event has invalid fields")
        exact_text = item["exact_text"]
        if exact_text is not None:
            exact_text = _bounded_text(exact_text, MAX_EXACT_TEXT_LENGTH)
            if exact_text not in exact_texts:
                raise QueryPlanningError(
                    "ordered event exact text must appear in exact_texts"
                )
        events.append(
            OrderedEvent(
                _bounded_text(item["visual"], MAX_GENERATED_TEXT_LENGTH),
                exact_text,
            )
        )
    if len(set(events)) != len(events):
        raise QueryPlanningError("ordered events must be unique")
    return tuple(events)


def _optional_text(value: Any, maximum: int) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, maximum)


def _unique_texts(value: Any, maximum_count: int, maximum_length: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_count:
        raise QueryPlanningError("query planner text list exceeds its bound")
    texts = tuple(_bounded_text(item, maximum_length) for item in value)
    if len(set(texts)) != len(texts):
        raise QueryPlanningError("query planner text list contains duplicates")
    return texts


def _bounded_text(value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        raise QueryPlanningError("query planner text must be a string")
    if any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in value
    ):
        raise QueryPlanningError("query planner text contains forbidden controls")
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized:
        raise QueryPlanningError("query planner text must not be empty")
    if len(normalized) > maximum:
        raise QueryPlanningError("query planner text exceeds its length bound")
    return normalized


def _normalize_plan_content(content: Any) -> str:
    if not isinstance(content, str):
        raise QueryPlanningError("query planner output must be JSON text")
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[0].strip().casefold() not in {"```", "```json"}:
        raise QueryPlanningError("query planner output must be plain JSON or one JSON fence")
    if lines[-1].strip() != "```" or any("```" in line for line in lines[1:-1]):
        raise QueryPlanningError("query planner output must be plain JSON or one JSON fence")
    body = "\n".join(lines[1:-1]).strip()
    if not body:
        raise QueryPlanningError("query planner output must contain JSON")
    return body


def _planner_prompt(query: str) -> str:
    example = {
        "holistic_visual": "two men carrying an object on a street",
        "visual_clauses": ["one man in a blue shirt", "one man in a white shirt"],
        "ordered_events": [
            {"visual": "two men carrying an object", "exact_text": None},
            {"visual": "close view of a street sign", "exact_text": "Đường Tiền Lân 11"},
        ],
        "exact_texts": ["Đường Tiền Lân 11"],
    }
    return (
        "Plan a Vietnamese video-search query for CLIP and OCR. Return one minified "
        "JSON object only with exactly these keys in this order: holistic_visual, "
        "visual_clauses, ordered_events, exact_texts. holistic_visual is one short "
        "English visible-scene string or null. visual_clauses contains 0 to 4 unique "
        "short English strings. Preserve visible colors, counts, objects, posture, "
        "setting, and relations; omit intent or facts that cannot be seen. "
        "ordered_events contains 0 to 3 objects with exactly visual and exact_text; "
        "use query order, English visual text, and null unless that event visibly "
        "contains an exact string. exact_texts contains 0 to 4 unique visible strings "
        "copied verbatim as contiguous substrings of the query, preserving spelling, "
        "accents, letters, digits, whitespace, and punctuation. "
        "Every non-null event exact_text must also occur in exact_texts. Do not "
        "translate exact text. No explanation or markdown. Valid shape: "
        f"{json.dumps(example, ensure_ascii=False, separators=(',', ':'))}. Query: {query}"
    )


__all__ = [
    "MAX_EXACT_TEXTS",
    "MAX_ORDERED_EVENTS",
    "MAX_VISUAL_CLAUSES",
    "OrderedEvent",
    "QueryPlan",
    "QueryPlanningError",
    "QwenQueryPlanner",
    "VisualQuery",
    "baseline_query_plan",
    "parse_query_plan",
]

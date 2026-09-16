"""Deterministic TRAKE event parsing and monotonic frame alignment."""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from typing import Sequence

from .evaluation import TrakeResponse
from .index import SearchHit
from .qa import QaError


@dataclass(frozen=True, slots=True)
class TrakeEvent:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise QaError("TRAKE event text must be non-empty")
        if any(ord(char) < 32 and char not in "\t\n\r" for char in self.text):
            raise QaError("TRAKE event text contains forbidden control character")


@dataclass(frozen=True, slots=True)
class TrakeQuery:
    raw_text: str
    events: tuple[TrakeEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str) or not self.raw_text.strip():
            raise QaError("TRAKE raw_text must be non-empty")
        if not isinstance(self.events, tuple) or not self.events:
            raise QaError("TRAKE requires at least one event")


def parse_trake_query(raw_text: str) -> TrakeQuery:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise QaError("TRAKE raw_text must be non-empty")
    text = " ".join(raw_text.split())
    numbered_events = tuple(
        match.group("text").strip(" ;,")
        for match in re.finditer(
            r"(?:^|\s)E\d+\s*:?[ \t]*(?P<text>.*?)(?=\s+E\d+\s*:?[ \t]*|$)",
            text,
            re.IGNORECASE,
        )
        if match.group("text").strip(" ;,")
    )
    if numbered_events:
        return TrakeQuery(raw_text, tuple(TrakeEvent(part) for part in numbered_events))
    parts = [part.strip(" ;,\t") for part in text.replace("→", "|").split("|")]
    if len(parts) == 1:
        parts = [part.strip() for part in text.split(" sau đó ") if part.strip()]
    return TrakeQuery(raw_text, tuple(TrakeEvent(part) for part in parts if part))


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    gap_penalty: float = 0.01
    max_candidates_per_event: int = 50
    max_videos: int = 100
    max_paths: int = 100

    def __post_init__(self) -> None:
        if not math.isfinite(self.gap_penalty) or self.gap_penalty < 0:
            raise QaError("gap_penalty must be finite and non-negative")
        for name in ("max_candidates_per_event", "max_videos", "max_paths"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise QaError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class AlignmentPath:
    score: float
    response: TrakeResponse


def align_event_hits_ranked(
    event_hits: Sequence[Sequence[SearchHit]],
    *,
    config: AlignmentConfig = AlignmentConfig(),
) -> tuple[AlignmentPath, ...]:
    if not isinstance(event_hits, Sequence) or not event_hits or any(not hits for hits in event_hits):
        return ()
    candidates: list[list[SearchHit]] = []
    for hits in event_hits:
        if not isinstance(hits, Sequence):
            raise QaError("event hits must be sequences")
        if any(not isinstance(hit, SearchHit) for hit in hits):
            raise QaError("event hits must contain SearchHit records")
        ordered = sorted(hits, key=lambda hit: (-hit.score, hit.row))[: config.max_candidates_per_event]
        if not ordered:
            return ()
        candidates.append(ordered)

    videos = sorted({hit.video_id for hits in candidates for hit in hits})[: config.max_videos]
    paths: list[AlignmentPath] = []
    for video in videos:
        per_event = [
            sorted((hit for hit in hits if hit.video_id == video), key=lambda hit: hit.original_frame_id)
            for hits in candidates
        ]
        if any(not hits for hits in per_event):
            continue
        states = {
            hit.row: (float(hit.score), (hit.original_frame_id,))
            for hit in per_event[0]
        }
        for hits in per_event[1:]:
            previous = sorted(
                ((value[1][-1], value[0], value[1]) for value in states.values()),
                key=lambda item: item[0],
            )
            next_states = {}
            best_prev = None
            previous_index = 0
            for hit in hits:
                while previous_index < len(previous) and previous[previous_index][0] < hit.original_frame_id:
                    frame, score, path = previous[previous_index]
                    candidate = (score + config.gap_penalty * frame, tuple(-value for value in path), path)
                    if best_prev is None or candidate[:2] > best_prev[:2]:
                        best_prev = candidate
                    previous_index += 1
                if best_prev is not None:
                    score = best_prev[0] + hit.score - config.gap_penalty * hit.original_frame_id
                    next_states[hit.row] = (score, best_prev[2] + (hit.original_frame_id,))
            states = next_states
            if not states:
                break
        paths.extend(
            AlignmentPath(score=value[0], response=TrakeResponse(video, value[1]))
            for value in states.values()
        )
    paths.sort(key=functools.cmp_to_key(_compare_alignment_paths))
    unique: list[AlignmentPath] = []
    seen: set[TrakeResponse] = set()
    for path in paths:
        if path.response in seen:
            continue
        seen.add(path.response)
        unique.append(path)
        if len(unique) >= config.max_paths:
            break
    return tuple(unique)


def _compare_alignment_paths(left: AlignmentPath, right: AlignmentPath) -> int:
    left_key = (
        left.score,
        tuple(-frame for frame in left.response.frame_ids),
        left.response.video_id,
    )
    right_key = (
        right.score,
        tuple(-frame for frame in right.response.frame_ids),
        right.response.video_id,
    )
    return (right_key > left_key) - (right_key < left_key)


def align_event_hits(
    event_hits: Sequence[Sequence[SearchHit]],
    *,
    config: AlignmentConfig = AlignmentConfig(),
) -> TrakeResponse | None:
    ranked = align_event_hits_ranked(event_hits, config=config)
    return None if not ranked else ranked[0].response


__all__ = [
    "AlignmentConfig",
    "AlignmentPath",
    "TrakeEvent",
    "TrakeQuery",
    "align_event_hits",
    "align_event_hits_ranked",
    "parse_trake_query",
]

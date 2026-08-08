"""Deterministic ranking transforms for CLIP retrieval candidates."""

from __future__ import annotations

from bisect import bisect_left, insort
from collections.abc import Sequence
from typing import Protocol, TypeVar


class RankingError(ValueError):
    """Raised when a ranking transform cannot preserve its contract."""


class TemporalCandidate(Protocol):
    video_id: str
    original_frame_id: int


CandidateT = TypeVar("CandidateT", bound=TemporalCandidate)


def temporal_deduplicate(
    candidates: Sequence[CandidateT], frame_window: int
) -> tuple[CandidateT, ...]:
    if isinstance(frame_window, bool) or not isinstance(frame_window, int) or frame_window < 0:
        raise RankingError("frame_window must be a non-negative integer")
    if frame_window == 0:
        return tuple(candidates)

    kept: list[CandidateT] = []
    frames_by_video: dict[str, list[int]] = {}
    for candidate in candidates:
        frames = frames_by_video.setdefault(candidate.video_id, [])
        position = bisect_left(frames, candidate.original_frame_id)
        neighbors = frames[max(0, position - 1) : position + 1]
        if any(
            abs(candidate.original_frame_id - frame_id) <= frame_window
            for frame_id in neighbors
        ):
            continue
        kept.append(candidate)
        insort(frames, candidate.original_frame_id)
    return tuple(kept)


def limit_per_video(
    candidates: Sequence[CandidateT], maximum: int | None
) -> tuple[CandidateT, ...]:
    if maximum is None:
        return tuple(candidates)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise RankingError("maximum per video must be a positive integer or null")

    counts: dict[str, int] = {}
    kept: list[CandidateT] = []
    for candidate in candidates:
        count = counts.get(candidate.video_id, 0)
        if count >= maximum:
            continue
        kept.append(candidate)
        counts[candidate.video_id] = count + 1
    return tuple(kept)

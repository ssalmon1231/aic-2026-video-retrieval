"""Bounded monotonic alignment for ordered video-search events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .index import SearchHit


class TemporalAlignmentError(ValueError):
    """Raised when ordered-event evidence violates temporal alignment bounds."""


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    max_frame_gap: int = 9000
    rank_constant: int = 60
    gap_penalty: float = 0.25
    full_coverage_bonus: float = 1.0
    video_boost_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_frame_gap", "rank_constant"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TemporalAlignmentError(f"{name} must be a positive integer")
        for name in ("gap_penalty", "full_coverage_bonus", "video_boost_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise TemporalAlignmentError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EventEvidence:
    hit: SearchHit
    rank: int
    score: float
    event_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.hit, SearchHit):
            raise TemporalAlignmentError("event evidence hit is invalid")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise TemporalAlignmentError("event evidence rank must be positive")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise TemporalAlignmentError("event evidence score must be finite")
        if (
            isinstance(self.event_index, bool)
            or not isinstance(self.event_index, int)
            or self.event_index < 0
        ):
            raise TemporalAlignmentError("event evidence index must be non-negative")


@dataclass(frozen=True, slots=True)
class TemporalChain:
    video_id: str
    events: tuple[EventEvidence, ...]
    coverage: int
    full_coverage: bool
    score: float
    representative: SearchHit

    def __post_init__(self) -> None:
        if not self.video_id or not self.events:
            raise TemporalAlignmentError("temporal chain must contain events")
        if self.coverage != len(self.events) or self.coverage <= 0:
            raise TemporalAlignmentError("temporal chain coverage is invalid")
        if not math.isfinite(self.score):
            raise TemporalAlignmentError("temporal chain score must be finite")
        if self.representative not in tuple(item.hit for item in self.events):
            raise TemporalAlignmentError("representative must belong to temporal chain")


def align_ordered_events(
    event_lists: Sequence[tuple[SearchHit, ...]],
    config: TemporalConfig = TemporalConfig(),
    *,
    prefer_final_event: bool = False,
) -> tuple[TemporalChain, ...]:
    if (
        isinstance(event_lists, (str, bytes))
        or not isinstance(event_lists, Sequence)
        or not event_lists
    ):
        raise TemporalAlignmentError("ordered alignment requires event hit lists")
    if len(event_lists) > 4:
        raise TemporalAlignmentError("ordered alignment supports at most four events")

    per_event: list[dict[str, tuple[EventEvidence, ...]]] = []
    videos: set[str] = set()
    for hits in event_lists:
        if not isinstance(hits, tuple):
            raise TemporalAlignmentError("event hit list must be a tuple")
        rows: set[int] = set()
        grouped: dict[str, list[EventEvidence]] = {}
        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, SearchHit) or hit.row in rows or not math.isfinite(hit.score):
                raise TemporalAlignmentError("event hit list contains invalid rows")
            rows.add(hit.row)
            evidence = EventEvidence(
                hit,
                rank,
                1.0 / (config.rank_constant + rank),
                len(per_event),
            )
            grouped.setdefault(hit.video_id, []).append(evidence)
            videos.add(hit.video_id)
        per_event.append(
            {
                video_id: tuple(
                    sorted(
                        values,
                        key=lambda item: (
                            item.hit.original_frame_id,
                            item.rank,
                            item.hit.row,
                        ),
                    )
                )
                for video_id, values in grouped.items()
            }
        )

    chains: list[TemporalChain] = []
    if not isinstance(prefer_final_event, bool):
        raise TemporalAlignmentError("prefer_final_event must be boolean")
    for video_id in videos:
        chain = _best_video_chain(
            video_id,
            per_event,
            config,
            prefer_final_event=prefer_final_event,
        )
        if chain is not None:
            chains.append(chain)
    return tuple(
        sorted(
            chains,
            key=lambda chain: (
                not chain.full_coverage,
                -chain.coverage,
                -chain.score,
                min(item.rank for item in chain.events),
                chain.representative.row,
                chain.video_id,
            ),
        )
    )


def apply_temporal_boost(
    fused_hits: tuple[SearchHit, ...],
    chains: Sequence[TemporalChain],
    config: TemporalConfig = TemporalConfig(),
) -> tuple[SearchHit, ...]:
    if not isinstance(fused_hits, tuple):
        raise TemporalAlignmentError("fused hits must be a tuple")
    best_by_video: dict[str, TemporalChain] = {}
    for chain in chains:
        if not isinstance(chain, TemporalChain):
            raise TemporalAlignmentError("temporal chain input is invalid")
        current = best_by_video.get(chain.video_id)
        if current is None or chain.score > current.score:
            best_by_video[chain.video_id] = chain

    boosted: list[SearchHit] = []
    for hit in fused_hits:
        chain = best_by_video.get(hit.video_id)
        boost = (
            config.video_boost_weight * chain.score
            if chain is not None and hit.row == chain.representative.row
            else 0.0
        )
        score = hit.score + boost
        if not math.isfinite(score):
            raise TemporalAlignmentError("temporal boost produced invalid score")
        boosted.append(
            SearchHit(
                score=score,
                row=hit.row,
                video_id=hit.video_id,
                keyframe_id=hit.keyframe_id,
                original_frame_id=hit.original_frame_id,
                keyframe_path=hit.keyframe_path,
            )
        )
    return tuple(sorted(boosted, key=lambda hit: (-hit.score, hit.row)))


def _best_video_chain(
    video_id: str,
    per_event: Sequence[dict[str, tuple[EventEvidence, ...]]],
    config: TemporalConfig,
    *,
    prefer_final_event: bool,
) -> TemporalChain | None:
    previous_states: list[tuple[tuple[EventEvidence, ...], float]] = []
    best: tuple[tuple[EventEvidence, ...], float] | None = None
    total_events = len(per_event)
    for grouped in per_event:
        current_states: list[tuple[tuple[EventEvidence, ...], float]] = []
        for evidence in grouped.get(video_id, ()):
            candidates = [((evidence,), evidence.score)]
            for chain, score in previous_states:
                gap = (
                    evidence.hit.original_frame_id
                    - chain[-1].hit.original_frame_id
                )
                if gap <= 0 or gap > config.max_frame_gap:
                    continue
                candidates.append(
                    (
                        (*chain, evidence),
                        score
                        + evidence.score
                        - config.gap_penalty * gap / config.max_frame_gap,
                    )
                )
            state = min(
                candidates,
                key=lambda candidate: _state_key(candidate, total_events, config),
            )
            current_states.append(state)
            if best is None or _state_key(
                state, total_events, config
            ) < _state_key(best, total_events, config):
                best = state
        previous_states = current_states
    if best is None:
        return None
    events, base_score = best
    coverage = len(events)
    full = coverage == len(per_event)
    score = base_score + (config.full_coverage_bonus if full else 0.0)
    representative = min(
        events,
        key=lambda item: (
            (
                0
                if prefer_final_event
                and full
                and item.event_index == len(per_event) - 1
                else 1
            ),
            -item.score,
            item.rank,
            item.hit.row,
        ),
    ).hit
    return TemporalChain(video_id, events, coverage, full, score, representative)


def _state_key(
    state: tuple[tuple[EventEvidence, ...], float],
    total_events: int,
    config: TemporalConfig,
) -> tuple[object, ...]:
    chain, score = state
    full = len(chain) == total_events
    adjusted = score + (config.full_coverage_bonus if full else 0.0)
    return (
        not full,
        -len(chain),
        -adjusted,
        min(item.rank for item in chain),
        tuple(item.hit.row for item in chain),
    )


__all__ = [
    "EventEvidence",
    "TemporalAlignmentError",
    "TemporalChain",
    "TemporalConfig",
    "align_ordered_events",
    "apply_temporal_boost",
]

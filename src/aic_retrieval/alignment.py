"""Retrieval adapter for TRAKE event alignment."""

from __future__ import annotations

from .evaluation import TrakeResponse
from .index import ExactIndex
from .qa import QaError
from .query import QueryEncoderRuntime
from .trake import AlignmentConfig, TrakeQuery, align_event_hits, align_event_hits_ranked


def retrieve_and_align_ranked(
    index: ExactIndex,
    encoder: QueryEncoderRuntime,
    query: TrakeQuery,
    *,
    config: AlignmentConfig = AlignmentConfig(),
    candidate_depth: int = 100,
) -> tuple[TrakeResponse, ...]:
    """Retrieve bounded per-event candidates then return ranked same-video paths."""
    if isinstance(candidate_depth, bool) or not isinstance(candidate_depth, int) or candidate_depth <= 0:
        raise QaError("candidate_depth must be a positive integer")
    if not isinstance(query, TrakeQuery):
        raise QaError("query must be a TrakeQuery")
    event_hits = []
    for event in query.events:
        encoded = encoder.encode([event.text])
        event_hits.append(index.search(encoded.vectors[0], limit=candidate_depth))
    return tuple(path.response for path in align_event_hits_ranked(event_hits, config=config))


def retrieve_and_align(
    index: ExactIndex,
    encoder: QueryEncoderRuntime,
    query: TrakeQuery,
    *,
    config: AlignmentConfig = AlignmentConfig(),
    candidate_depth: int = 100,
) -> TrakeResponse | None:
    """Compatibility boundary returning best ranked TRAKE path."""
    responses = retrieve_and_align_ranked(
        index, encoder, query, config=config, candidate_depth=candidate_depth
    )
    return None if not responses else responses[0]


__all__ = ["retrieve_and_align", "retrieve_and_align_ranked"]

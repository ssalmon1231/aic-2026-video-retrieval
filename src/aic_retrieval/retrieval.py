"""Vector-input Textual KIS retrieval over the canonical exact index."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .evaluation import MAX_RESPONSES, KisResponse, validate_ranked_responses
from .index import ExactIndex, SearchHit
from .ranking import RankingError, limit_per_video, temporal_deduplicate


class RetrievalError(ValueError):
    """Raised when retrieval inputs or output limits violate the contract."""


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    candidate_depth: int = 500
    result_limit: int = MAX_RESPONSES
    temporal_window: int = 0
    max_results_per_video: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_depth, bool)
            or not isinstance(self.candidate_depth, int)
            or self.candidate_depth <= 0
        ):
            raise RetrievalError("candidate_depth must be a positive integer")
        if (
            isinstance(self.result_limit, bool)
            or not isinstance(self.result_limit, int)
            or not 1 <= self.result_limit <= MAX_RESPONSES
        ):
            raise RetrievalError(
                f"result_limit must be an integer in [1, {MAX_RESPONSES}]"
            )
        if self.candidate_depth < self.result_limit:
            raise RetrievalError("candidate_depth must be at least result_limit")
        try:
            temporal_deduplicate((), self.temporal_window)
            limit_per_video((), self.max_results_per_video)
        except RankingError as error:
            raise RetrievalError(str(error)) from error


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    score: float
    raw_rank: int
    video_id: str
    keyframe_id: str
    original_frame_id: int
    keyframe_path: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    raw_candidates: tuple[RetrievalCandidate, ...]
    final_candidates: tuple[RetrievalCandidate, ...]
    responses: tuple[KisResponse, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_candidates": [asdict(candidate) for candidate in self.raw_candidates],
            "final_candidates": [
                asdict(candidate) for candidate in self.final_candidates
            ],
            "responses": [asdict(response) for response in self.responses],
        }


def retrieve_kis(
    index: ExactIndex,
    query_vector: np.ndarray,
    config: RetrievalConfig = RetrievalConfig(),
    *,
    rerank: Callable[[ExactIndex, tuple[SearchHit, ...]], tuple[SearchHit, ...]] | None = None,
) -> RetrievalResult:
    vector = validate_query_vector(query_vector, index.metadata.dimension)
    baseline_hits = index.search(vector, limit=config.candidate_depth)
    hits = baseline_hits
    if rerank is not None:
        try:
            candidate_hits = rerank(index, baseline_hits)
            hits = _validate_reranked_hits(candidate_hits, baseline_hits)
        except (OSError, RuntimeError, TypeError, ValueError):
            hits = baseline_hits
    exact_ranks = {hit.row: rank for rank, hit in enumerate(baseline_hits, start=1)}
    return retrieval_result_from_hits(
        index,
        hits,
        config,
        raw_ranks=exact_ranks,
    )


def retrieval_result_from_hits(
    index: ExactIndex,
    hits: tuple[SearchHit, ...],
    config: RetrievalConfig,
    *,
    raw_ranks: dict[int, int] | None = None,
) -> RetrievalResult:
    """Build canonical KIS responses from a validated ranked index shortlist."""

    if not isinstance(hits, tuple):
        raise RetrievalError("ranked hits must be a tuple")
    ranks = (
        raw_ranks
        if raw_ranks is not None
        else {hit.row: rank for rank, hit in enumerate(hits, start=1)}
    )
    if set(ranks) != {hit.row for hit in hits}:
        raise RetrievalError("raw ranks must cover every unique ranked hit")
    if (
        any(isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0 for rank in ranks.values())
        or len(set(ranks.values())) != len(ranks)
    ):
        raise RetrievalError("raw ranks must be unique positive integers")

    seen_rows: set[int] = set()
    raw: list[RetrievalCandidate] = []
    for hit in hits:
        if not isinstance(hit, SearchHit) or hit.row in seen_rows:
            raise RetrievalError("ranked hits must contain unique SearchHit rows")
        if not math.isfinite(hit.score) or not 0 <= hit.row < len(index.keyframes):
            raise RetrievalError("ranked hit score or row is invalid")
        record = index.keyframes[hit.row]
        if (
            hit.video_id != record.video_id
            or hit.keyframe_id != record.keyframe_id
            or hit.original_frame_id != record.original_frame_id
            or hit.keyframe_path != record.keyframe_path
        ):
            raise RetrievalError("ranked hit provenance disagrees with index lookup")
        seen_rows.add(hit.row)
        raw.append(
            RetrievalCandidate(
                score=hit.score,
                raw_rank=ranks[hit.row],
                video_id=hit.video_id,
                keyframe_id=hit.keyframe_id,
                original_frame_id=hit.original_frame_id,
                keyframe_path=hit.keyframe_path,
            )
        )

    raw_candidates = tuple(raw)
    ranked = temporal_deduplicate(raw_candidates, config.temporal_window)
    ranked = _deduplicate_response_identities(ranked)
    ranked = limit_per_video(ranked, config.max_results_per_video)
    final = ranked[: config.result_limit]
    responses = tuple(
        KisResponse(candidate.video_id, candidate.original_frame_id)
        for candidate in final
    )
    validate_ranked_responses(responses)
    return RetrievalResult(raw_candidates, final, responses)


def _validate_reranked_hits(
    reranked: tuple[SearchHit, ...],
    baseline: tuple[SearchHit, ...],
) -> tuple[SearchHit, ...]:
    if not isinstance(reranked, tuple) or len(reranked) != len(baseline):
        raise RetrievalError("reranker must preserve shortlist length")
    baseline_rows = {hit.row for hit in baseline}
    reranked_rows = {hit.row for hit in reranked}
    if reranked_rows != baseline_rows or len(reranked_rows) != len(reranked):
        raise RetrievalError("reranker must preserve unique shortlist rows")
    if any(not math.isfinite(hit.score) for hit in reranked):
        raise RetrievalError("reranker scores must be finite")
    return reranked


def _deduplicate_response_identities(
    candidates: tuple[RetrievalCandidate, ...],
) -> tuple[RetrievalCandidate, ...]:
    seen: set[tuple[str, int]] = set()
    kept: list[RetrievalCandidate] = []
    for candidate in candidates:
        identity = (candidate.video_id, candidate.original_frame_id)
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(candidate)
    return tuple(kept)


def validate_query_vector(query_vector: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(query_vector, dtype=np.float32)
    if vector.shape != (dimension,):
        raise RetrievalError(f"query vector shape {vector.shape} != ({dimension},)")
    if not np.isfinite(vector).all():
        raise RetrievalError("query vector contains NaN or infinity")
    if float(np.linalg.norm(vector)) == 0:
        raise RetrievalError("query vector norm must be positive")
    return vector


def load_config(path: str | Path) -> RetrievalConfig:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalError(f"cannot read retrieval config {source}: {error}") from error
    if not isinstance(payload, dict):
        raise RetrievalError("retrieval config root must be an object")
    allowed = {
        "candidate_depth",
        "result_limit",
        "temporal_window",
        "max_results_per_video",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RetrievalError(f"unknown retrieval config fields: {', '.join(unknown)}")
    try:
        return RetrievalConfig(**payload)
    except TypeError as error:
        raise RetrievalError(f"invalid retrieval config: {error}") from error

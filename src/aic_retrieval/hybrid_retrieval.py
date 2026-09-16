"""Bounded multi-query rank fusion for compositional Textual KIS."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .index import ExactIndex, SearchHit
from .ocr import OcrIndex, OcrMatch
from .query import QueryEncoderRuntime, QueryEncodingError
from .query_planning import QueryPlan, baseline_query_plan
from .retrieval import (
    RetrievalConfig,
    RetrievalError,
    RetrievalResult,
    retrieval_result_from_hits,
    retrieve_kis,
)
from .temporal import TemporalConfig, align_ordered_events, apply_temporal_boost


class HybridRetrievalError(ValueError):
    """Raised when hybrid retrieval configuration or evidence is invalid."""


class QueryPlanner(Protocol):
    def plan(self, raw_query: str) -> QueryPlan: ...


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    enabled: bool = False
    raw_candidate_depth: int = 500
    auxiliary_candidate_depth: int = 300
    raw_weight: float = 2.0
    holistic_weight: float = 1.5
    clause_weight: float = 1.0
    ordered_event_weight: float = 1.0
    rrf_k: int = 60
    planner_budget_seconds: float = 3.0
    temporal_enabled: bool = True
    temporal_max_frame_gap: int = 9000
    temporal_gap_penalty: float = 0.25
    temporal_full_coverage_bonus: float = 1.0
    temporal_video_boost_weight: float = 1.0
    ocr_enabled: bool = True
    ocr_candidate_depth: int = 100
    ocr_weight: float = 0.75
    ocr_fuzzy_threshold: float = 0.72

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (self.enabled, self.temporal_enabled, self.ocr_enabled)
        ):
            raise HybridRetrievalError("enabled flags must be boolean")
        for name in (
            "raw_candidate_depth",
            "auxiliary_candidate_depth",
            "rrf_k",
            "temporal_max_frame_gap",
            "ocr_candidate_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise HybridRetrievalError(f"{name} must be a positive integer")
        weights = (
            self.raw_weight,
            self.holistic_weight,
            self.clause_weight,
            self.ordered_event_weight,
            self.ocr_weight,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in weights
        ):
            raise HybridRetrievalError("fusion weights must be finite and positive")
        if (
            isinstance(self.planner_budget_seconds, bool)
            or not isinstance(self.planner_budget_seconds, (int, float))
            or not math.isfinite(self.planner_budget_seconds)
            or self.planner_budget_seconds <= 0
        ):
            raise HybridRetrievalError(
                "planner_budget_seconds must be finite and positive"
            )
        if (
            isinstance(self.ocr_fuzzy_threshold, bool)
            or not isinstance(self.ocr_fuzzy_threshold, (int, float))
            or not math.isfinite(self.ocr_fuzzy_threshold)
            or not 0 <= self.ocr_fuzzy_threshold <= 1
        ):
            raise HybridRetrievalError("ocr_fuzzy_threshold must be in [0, 1]")
        try:
            TemporalConfig(
                max_frame_gap=self.temporal_max_frame_gap,
                rank_constant=self.rrf_k,
                gap_penalty=self.temporal_gap_penalty,
                full_coverage_bonus=self.temporal_full_coverage_bonus,
                video_boost_weight=self.temporal_video_boost_weight,
            )
        except ValueError as error:
            raise HybridRetrievalError(str(error)) from error


@dataclass(frozen=True, slots=True)
class HybridRetrievalStatus:
    applied: bool
    fallback: bool
    planner_fallback: bool
    semantic_lists: int
    fused_candidates: int
    ocr_available: bool = False
    ocr_applied: bool = False
    ocr_matches: int = 0
    planner_elapsed_ms: float = 0.0
    encoding_elapsed_ms: float = 0.0
    search_elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.applied,
                self.fallback,
                self.planner_fallback,
                self.ocr_available,
                self.ocr_applied,
            )
        ):
            raise HybridRetrievalError("hybrid status flags must be boolean")
        if self.ocr_applied and not self.ocr_available:
            raise HybridRetrievalError("OCR cannot be applied when unavailable")
        for value in (self.semantic_lists, self.fused_candidates, self.ocr_matches):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HybridRetrievalError("hybrid status counts must be non-negative")
        for value in (
            self.planner_elapsed_ms,
            self.encoding_elapsed_ms,
            self.search_elapsed_ms,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise HybridRetrievalError(
                    "hybrid status timings must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    retrieval: RetrievalResult
    status: HybridRetrievalStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.retrieval.to_dict(),
            "hybrid": {
                "applied": self.status.applied,
                "fallback": self.status.fallback,
                "planner_fallback": self.status.planner_fallback,
                "semantic_lists": self.status.semantic_lists,
                "fused_candidates": self.status.fused_candidates,
                "ocr_available": self.status.ocr_available,
                "ocr_applied": self.status.ocr_applied,
                "ocr_matches": self.status.ocr_matches,
                "planner_elapsed_ms": self.status.planner_elapsed_ms,
                "encoding_elapsed_ms": self.status.encoding_elapsed_ms,
                "search_elapsed_ms": self.status.search_elapsed_ms,
            },
        }


@dataclass(frozen=True, slots=True)
class RankedList:
    hits: tuple[SearchHit, ...]
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.hits, tuple) or not self.hits:
            raise HybridRetrievalError("ranked list must contain hits")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise HybridRetrievalError("ranked-list weight must be finite and positive")
        rows = tuple(hit.row for hit in self.hits)
        if len(set(rows)) != len(rows):
            raise HybridRetrievalError("ranked list rows must be unique")


class HybridTextRetriever:
    """Warm text pipeline with exact-baseline fallback on auxiliary failure."""

    def __init__(
        self,
        index: ExactIndex,
        retrieval_config: RetrievalConfig,
        hybrid_config: HybridRetrievalConfig,
        multilingual_encoder: QueryEncoderRuntime,
        english_encoder: QueryEncoderRuntime,
        planner: QueryPlanner,
        *,
        ocr_index: OcrIndex | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if (
            hybrid_config.enabled
            and retrieval_config.candidate_depth < hybrid_config.raw_candidate_depth
        ):
            raise HybridRetrievalError(
                "retrieval candidate_depth must cover raw_candidate_depth"
            )
        for encoder in (multilingual_encoder, english_encoder):
            if encoder.expected_dimension != index.metadata.dimension:
                raise HybridRetrievalError("encoder dimension disagrees with index")
        self.index = index
        self.retrieval_config = retrieval_config
        self.hybrid_config = hybrid_config
        self.multilingual_encoder = multilingual_encoder
        self.english_encoder = english_encoder
        if ocr_index is not None and (
            ocr_index.metadata.manifest_sha256 != index.metadata.manifest_sha256
            or ocr_index.metadata.index_rows != len(index.keyframes)
        ):
            raise HybridRetrievalError("OCR index provenance disagrees with vector index")
        self.planner = planner
        self.ocr_index = ocr_index
        self.clock = clock
        self.last_raw_provenance: Any | None = None

    def retrieve(self, raw_query: str) -> HybridRetrievalResult:
        encoded_started = self.clock()
        raw_encoding = self.multilingual_encoder.encode([raw_query])
        self.last_raw_provenance = getattr(raw_encoding, "provenance", None)
        encoding_elapsed = self.clock() - encoded_started
        raw_vector = raw_encoding.vectors[0]
        baseline = retrieve_kis(
            self.index,
            raw_vector,
            self.retrieval_config,
        )
        disabled_status = HybridRetrievalStatus(
            applied=False,
            fallback=False,
            planner_fallback=False,
            semantic_lists=1,
            fused_candidates=len(baseline.raw_candidates),
            ocr_available=self.ocr_index is not None,
            planner_elapsed_ms=0.0,
            encoding_elapsed_ms=encoding_elapsed * 1000.0,
            search_elapsed_ms=0.0,
        )
        if not self.hybrid_config.enabled:
            return HybridRetrievalResult(baseline, disabled_status)

        planner_started = self.clock()
        planner_fallback = False
        try:
            plan = self.planner.plan(raw_query)
        except (OSError, RuntimeError, TypeError, ValueError):
            plan = baseline_query_plan(raw_query)
            planner_fallback = True
        planner_elapsed = self.clock() - planner_started
        if planner_elapsed > self.hybrid_config.planner_budget_seconds:
            plan = baseline_query_plan(raw_query)
            planner_fallback = True
        if planner_fallback:
            status = HybridRetrievalStatus(
                applied=False,
                fallback=False,
                planner_fallback=True,
                semantic_lists=1,
                fused_candidates=len(baseline.raw_candidates),
                ocr_available=self.ocr_index is not None,
                planner_elapsed_ms=planner_elapsed * 1000.0,
                encoding_elapsed_ms=encoding_elapsed * 1000.0,
                search_elapsed_ms=0.0,
            )
            return HybridRetrievalResult(baseline, status)

        try:
            search_started = self.clock()
            raw_hits = self.index.search(
                raw_vector,
                limit=self.hybrid_config.raw_candidate_depth,
            )
            lists = [RankedList(raw_hits, self.hybrid_config.raw_weight)]
            event_hit_lists: list[tuple[SearchHit, ...]] = []
            auxiliary = tuple(item for item in plan.visual_queries if item.kind != "raw")
            for item in auxiliary:
                try:
                    encode_started = self.clock()
                    vector = self.english_encoder.encode([item.text]).vectors[0]
                    encoding_elapsed += self.clock() - encode_started
                    weight = (
                        self.hybrid_config.holistic_weight
                        if item.kind == "holistic"
                        else self.hybrid_config.clause_weight
                    )
                    lists.append(
                        RankedList(
                            self.index.search(
                                vector,
                                limit=self.hybrid_config.auxiliary_candidate_depth,
                            ),
                            weight,
                        )
                    )
                except (OSError, RuntimeError, TypeError, ValueError, QueryEncodingError):
                    continue
            for event in plan.ordered_events:
                try:
                    encode_started = self.clock()
                    vector = self.english_encoder.encode([event.visual]).vectors[0]
                    encoding_elapsed += self.clock() - encode_started
                    event_hits = self.index.search(
                        vector,
                        limit=self.hybrid_config.auxiliary_candidate_depth,
                    )
                except (OSError, RuntimeError, TypeError, ValueError, QueryEncodingError):
                    event_hit_lists.append(())
                    continue
                event_hit_lists.append(event_hits)
                lists.append(
                    RankedList(
                        event_hits,
                        self.hybrid_config.ordered_event_weight,
                    )
                )
            fused = reciprocal_rank_fusion(
                tuple(lists),
                rrf_k=self.hybrid_config.rrf_k,
            )
            ocr_matches: tuple[OcrMatch, ...] = ()
            ocr_event_hits: dict[str, tuple[SearchHit, ...]] = {}
            if (
                self.hybrid_config.ocr_enabled
                and self.ocr_index is not None
                and plan.exact_texts
            ):
                allowed_rows = {hit.row for hit in fused}
                per_text: list[tuple[OcrMatch, ...]] = []
                for exact_text in plan.exact_texts:
                    semantic_matches = self.ocr_index.search(
                        exact_text,
                        allowed_rows=allowed_rows,
                        limit=self.hybrid_config.ocr_candidate_depth,
                        fuzzy_threshold=self.hybrid_config.ocr_fuzzy_threshold,
                    )
                    exact_matches = self.ocr_index.exact_candidates(
                        exact_text,
                        limit=self.hybrid_config.ocr_candidate_depth,
                    )
                    matches = _merge_ocr_matches(
                        semantic_matches,
                        exact_matches,
                        limit=self.hybrid_config.ocr_candidate_depth,
                    )
                    per_text.append(matches)
                    if matches:
                        ocr_event_hits[exact_text] = _ocr_hits(self.index, matches)
                ocr_matches = _best_ocr_matches(per_text)
                if ocr_matches:
                    lists.append(
                        RankedList(
                            _ocr_hits(self.index, ocr_matches),
                            self.hybrid_config.ocr_weight,
                        )
                    )
                    fused = reciprocal_rank_fusion(
                        tuple(lists),
                        rrf_k=self.hybrid_config.rrf_k,
                    )
            if self.hybrid_config.temporal_enabled:
                temporal_lists: list[tuple[SearchHit, ...]] = []
                for event, visual_hits in zip(
                    plan.ordered_events,
                    event_hit_lists,
                    strict=True,
                ):
                    if event.exact_text is not None:
                        exact_hits = ocr_event_hits.get(event.exact_text)
                        if exact_hits is None:
                            temporal_lists = []
                            break
                        temporal_lists.append(exact_hits)
                    else:
                        temporal_lists.append(visual_hits)
                if len(temporal_lists) >= 2:
                    temporal_config = TemporalConfig(
                        max_frame_gap=self.hybrid_config.temporal_max_frame_gap,
                        rank_constant=self.hybrid_config.rrf_k,
                        gap_penalty=self.hybrid_config.temporal_gap_penalty,
                        full_coverage_bonus=self.hybrid_config.temporal_full_coverage_bonus,
                        video_boost_weight=self.hybrid_config.temporal_video_boost_weight,
                    )
                    chains = align_ordered_events(
                        tuple(temporal_lists),
                        temporal_config,
                        prefer_final_event=plan.ordered_events[-1].exact_text is not None,
                    )
                    fused = apply_temporal_boost(fused, chains, temporal_config)
            search_elapsed = self.clock() - search_started
            result = retrieval_result_from_hits(
                self.index,
                fused,
                self.retrieval_config,
            )
            if len(lists) == 1 and not ocr_matches:
                status = HybridRetrievalStatus(
                    applied=False,
                    fallback=False,
                    planner_fallback=False,
                    semantic_lists=1,
                    fused_candidates=len(baseline.raw_candidates),
                    ocr_available=self.ocr_index is not None,
                    planner_elapsed_ms=planner_elapsed * 1000.0,
                    encoding_elapsed_ms=encoding_elapsed * 1000.0,
                    search_elapsed_ms=search_elapsed * 1000.0,
                )
                return HybridRetrievalResult(baseline, status)
            status = HybridRetrievalStatus(
                applied=True,
                fallback=False,
                planner_fallback=planner_fallback,
                semantic_lists=len(lists) - bool(ocr_matches),
                fused_candidates=len(fused),
                ocr_available=self.ocr_index is not None,
                ocr_applied=bool(ocr_matches),
                ocr_matches=len(ocr_matches),
                planner_elapsed_ms=planner_elapsed * 1000.0,
                encoding_elapsed_ms=encoding_elapsed * 1000.0,
                search_elapsed_ms=search_elapsed * 1000.0,
            )
            return HybridRetrievalResult(result, status)
        except (OSError, RuntimeError, TypeError, ValueError, QueryEncodingError):
            status = HybridRetrievalStatus(
                applied=False,
                fallback=True,
                planner_fallback=planner_fallback,
                semantic_lists=1,
                fused_candidates=len(baseline.raw_candidates),
                ocr_available=self.ocr_index is not None,
                planner_elapsed_ms=planner_elapsed * 1000.0,
                encoding_elapsed_ms=encoding_elapsed * 1000.0,
                search_elapsed_ms=0.0,
            )
            return HybridRetrievalResult(baseline, status)


def _ocr_hits(index: ExactIndex, matches: Sequence[OcrMatch]) -> tuple[SearchHit, ...]:
    hits: list[SearchHit] = []
    for match in matches:
        if match.row >= len(index.keyframes):
            raise HybridRetrievalError("OCR match row lies outside vector index")
        keyframe = index.keyframes[match.row]
        hits.append(
            SearchHit(
                score=match.score,
                row=match.row,
                video_id=keyframe.video_id,
                keyframe_id=keyframe.keyframe_id,
                original_frame_id=keyframe.original_frame_id,
                keyframe_path=keyframe.keyframe_path,
            )
        )
    return tuple(hits)


def _merge_ocr_matches(
    *groups: Sequence[OcrMatch],
    limit: int,
) -> tuple[OcrMatch, ...]:
    by_row: dict[int, OcrMatch] = {}
    for matches in groups:
        for match in matches:
            previous = by_row.get(match.row)
            if previous is None or (match.score, match.exact, match.confidence) > (
                previous.score,
                previous.exact,
                previous.confidence,
            ):
                by_row[match.row] = match
    return tuple(
        sorted(
            by_row.values(),
            key=lambda match: (
                -match.score,
                not match.exact,
                -match.confidence,
                match.row,
            ),
        )[:limit]
    )


def _best_ocr_matches(
    per_text: Sequence[tuple[OcrMatch, ...]],
) -> tuple[OcrMatch, ...]:
    by_row: dict[int, OcrMatch] = {}
    for matches in per_text:
        for match in matches:
            previous = by_row.get(match.row)
            if previous is None or (match.score, match.exact, match.confidence) > (
                previous.score,
                previous.exact,
                previous.confidence,
            ):
                by_row[match.row] = match
    return tuple(
        sorted(
            by_row.values(),
            key=lambda match: (
                -match.score,
                not match.exact,
                -match.confidence,
                match.row,
            ),
        )
    )


def reciprocal_rank_fusion(
    ranked_lists: Sequence[RankedList],
    *,
    rrf_k: int = 60,
) -> tuple[SearchHit, ...]:
    if (
        isinstance(ranked_lists, (str, bytes))
        or not isinstance(ranked_lists, Sequence)
        or not ranked_lists
    ):
        raise HybridRetrievalError("rank fusion requires ranked lists")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise HybridRetrievalError("rrf_k must be a positive integer")

    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    source_hits: dict[int, SearchHit] = {}
    for ranked in ranked_lists:
        if not isinstance(ranked, RankedList):
            raise HybridRetrievalError("rank fusion input is invalid")
        for rank, hit in enumerate(ranked.hits, start=1):
            if not isinstance(hit, SearchHit) or not math.isfinite(hit.score):
                raise HybridRetrievalError("rank fusion hit is invalid")
            existing = source_hits.get(hit.row)
            if existing is not None and (
                existing.video_id,
                existing.keyframe_id,
                existing.original_frame_id,
                existing.keyframe_path,
            ) != (
                hit.video_id,
                hit.keyframe_id,
                hit.original_frame_id,
                hit.keyframe_path,
            ):
                raise HybridRetrievalError("rank fusion row provenance conflicts")
            source_hits.setdefault(hit.row, hit)
            scores[hit.row] = scores.get(hit.row, 0.0) + float(ranked.weight) / (
                rrf_k + rank
            )
            best_rank[hit.row] = min(best_rank.get(hit.row, rank), rank)

    if not scores or not all(math.isfinite(value) for value in scores.values()):
        raise HybridRetrievalError("rank fusion produced invalid scores")
    rows = sorted(scores, key=lambda row: (-scores[row], best_rank[row], row))
    return tuple(
        SearchHit(
            score=scores[row],
            row=row,
            video_id=source_hits[row].video_id,
            keyframe_id=source_hits[row].keyframe_id,
            original_frame_id=source_hits[row].original_frame_id,
            keyframe_path=source_hits[row].keyframe_path,
        )
        for row in rows
    )


def load_config(path: str | Path) -> HybridRetrievalConfig:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HybridRetrievalError("cannot read hybrid retrieval config") from error
    if not isinstance(payload, dict):
        raise HybridRetrievalError("hybrid retrieval config root must be an object")
    allowed = {
        "enabled",
        "raw_candidate_depth",
        "auxiliary_candidate_depth",
        "raw_weight",
        "holistic_weight",
        "clause_weight",
        "ordered_event_weight",
        "rrf_k",
        "planner_budget_seconds",
        "temporal_enabled",
        "temporal_max_frame_gap",
        "temporal_gap_penalty",
        "temporal_full_coverage_bonus",
        "temporal_video_boost_weight",
        "ocr_enabled",
        "ocr_candidate_depth",
        "ocr_weight",
        "ocr_fuzzy_threshold",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise HybridRetrievalError("unknown hybrid retrieval config fields")
    try:
        return HybridRetrievalConfig(**payload)
    except TypeError as error:
        raise HybridRetrievalError("invalid hybrid retrieval config") from error


__all__ = [
    "HybridRetrievalConfig",
    "HybridRetrievalError",
    "HybridRetrievalResult",
    "HybridRetrievalStatus",
    "HybridTextRetriever",
    "RankedList",
    "load_config",
    "reciprocal_rank_fusion",
]

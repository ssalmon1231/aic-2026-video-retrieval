"""Bounded temporal evidence windows for automatic Q&A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .contracts import VideoRecord
from .index import SearchHit
from .qa import EvidenceWindow, FrameEvidence, QaError


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    window_seconds: float = 4.0
    max_windows: int = 20
    max_seeds_per_video: int = 3
    samples_per_window: int = 5
    near_seed_radius: int = 0
    refine_window_seconds: float = 1.0
    max_refined_windows: int = 20

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise QaError("window_seconds must be finite and positive")
        if (
            not math.isfinite(self.refine_window_seconds)
            or self.refine_window_seconds <= 0
        ):
            raise QaError("refine_window_seconds must be finite and positive")
        for name in (
            "max_windows",
            "max_seeds_per_video",
            "samples_per_window",
            "max_refined_windows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise QaError(f"{name} must be a positive integer")
        if isinstance(self.near_seed_radius, bool) or not isinstance(self.near_seed_radius, int) or self.near_seed_radius < 0:
            raise QaError("near_seed_radius must be a non-negative integer")


def build_evidence_windows(
    hits: Iterable[SearchHit],
    videos: Iterable[VideoRecord],
    config: EvidenceConfig = EvidenceConfig(),
) -> tuple[EvidenceWindow, ...]:
    """Group ranked keyframe hits into deterministic, bounded video windows."""
    ranked_hits = tuple(hits)
    video_map = {video.video_id: video for video in videos}
    grouped: dict[str, list[SearchHit]] = {}
    for hit in ranked_hits:
        if not isinstance(hit, SearchHit):
            raise QaError("evidence hits must contain SearchHit records")
        video = video_map.get(hit.video_id)
        if video is None:
            continue
        if not 0 <= hit.original_frame_id < video.frame_count:
            continue
        grouped.setdefault(hit.video_id, []).append(hit)

    windows: list[EvidenceWindow] = []
    global_ranks = {hit.row: rank for rank, hit in enumerate(ranked_hits, start=1)}
    for video_id, video_hits in grouped.items():
        video = video_map[video_id]
        ordered = sorted(video_hits, key=lambda item: (item.original_frame_id, item.row))
        seeds: list[SearchHit] = []
        seen: set[int] = set()
        for hit in sorted(video_hits, key=lambda item: (-item.score, item.row)):
            if hit.original_frame_id not in seen:
                seeds.append(hit)
                seen.add(hit.original_frame_id)
            if len(seeds) >= config.max_seeds_per_video:
                break
        for seed in seeds:
            center = seed.original_frame_id
            radius = max(1, int(math.ceil(config.window_seconds * video.fps / 2)))
            start = max(0, center - radius)
            end = min(video.frame_count - 1, center + radius)
            window_hits = [hit for hit in ordered if start <= hit.original_frame_id <= end]
            raw_ranks = tuple(sorted({global_ranks[hit.row] for hit in window_hits}))
            if not raw_ranks:
                raw_ranks = (global_ranks[seed.row],)
            sampled = sample_frame_ids(
                start,
                end,
                tuple(sorted({hit.original_frame_id for hit in window_hits})),
                config.samples_per_window,
                near_seed_radius=config.near_seed_radius,
            )
            windows.append(EvidenceWindow(
                video_id=video_id,
                start_frame_id=start,
                end_frame_id=end,
                seed_frame_ids=tuple(sorted({hit.original_frame_id for hit in seeds if start <= hit.original_frame_id <= end})) or (center,),
                sampled_frame_ids=sampled,
                retrieval_score=float(seed.score),
                raw_ranks=raw_ranks,
            ))
    windows.sort(key=lambda item: (-item.retrieval_score, item.raw_ranks, item.video_id, item.start_frame_id))
    return windows[: config.max_windows]


def refine_evidence_windows(
    windows: Sequence[EvidenceWindow],
    videos: Iterable[VideoRecord],
    config: EvidenceConfig = EvidenceConfig(),
) -> tuple[EvidenceWindow, ...]:
    """Center bounded fine windows on coarse evidence without changing provenance."""
    video_map = {video.video_id: video for video in videos}
    refined: list[EvidenceWindow] = []
    for window in windows:
        video = video_map.get(window.video_id)
        if video is None:
            continue
        center = window.seed_frame_ids[0]
        radius = max(1, int(math.ceil(config.refine_window_seconds * video.fps / 2)))
        start = max(0, center - radius)
        end = min(video.frame_count - 1, center + radius)
        seeds = tuple(frame for frame in window.seed_frame_ids if start <= frame <= end)
        if not seeds:
            seeds = (center,)
        sampled = sample_frame_ids(
            start,
            end,
            seeds,
            config.samples_per_window,
            near_seed_radius=config.near_seed_radius,
        )
        refined.append(EvidenceWindow(
            video_id=window.video_id,
            start_frame_id=start,
            end_frame_id=end,
            seed_frame_ids=seeds,
            sampled_frame_ids=sampled,
            retrieval_score=window.retrieval_score,
            raw_ranks=window.raw_ranks,
        ))
    refined.sort(
        key=lambda item: (
            -item.retrieval_score,
            item.raw_ranks,
            item.video_id,
            item.start_frame_id,
        )
    )
    return tuple(refined[: config.max_refined_windows])


def merge_ranked_hits(
    ranked_hit_lists: Sequence[Sequence[SearchHit]],
) -> tuple[SearchHit, ...]:
    """Fuse retrieval variants with reciprocal ranks and stable keyframe provenance."""
    merged: dict[int, tuple[float, SearchHit]] = {}
    for hits in ranked_hit_lists:
        if not isinstance(hits, Sequence):
            raise QaError("ranked hit lists must be sequences")
        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, SearchHit):
                raise QaError("ranked hit lists must contain SearchHit records")
            score, canonical = merged.get(hit.row, (0.0, hit))
            merged[hit.row] = (score + 1.0 / (60 + rank), canonical)
    return tuple(
        item[1]
        for item in sorted(
            merged.values(),
            key=lambda item: (
                -item[0],
                item[1].row,
                item[1].video_id,
                item[1].original_frame_id,
            ),
        )
    )


def sample_frame_ids(
    start_frame_id: int,
    end_frame_id: int,
    seed_frame_ids: tuple[int, ...],
    sample_count: int,
    *,
    near_seed_radius: int = 0,
) -> tuple[int, ...]:
    if start_frame_id < 0 or end_frame_id < start_frame_id:
        raise QaError("invalid sampling bounds")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise QaError("sample_count must be a positive integer")
    seeds = set(seed_frame_ids)
    if any(frame < start_frame_id or frame > end_frame_id for frame in seeds):
        raise QaError("seed frame lies outside sampling bounds")
    selected = set(seeds)
    if near_seed_radius:
        for seed in seeds:
            selected.add(max(start_frame_id, seed - near_seed_radius))
            selected.add(min(end_frame_id, seed + near_seed_radius))
    span = end_frame_id - start_frame_id
    if sample_count == 1:
        selected.add(start_frame_id)
    else:
        for index in range(sample_count):
            selected.add(start_frame_id + (span * index) // (sample_count - 1))
    return tuple(sorted(selected))


def decode_window(
    video_path: str,
    window: EvidenceWindow,
    decoder: Callable[[str, int], Any],
) -> tuple[FrameEvidence, ...]:
    """Decode sampled frames in memory; decoder owns BGR/RGB conversion."""
    evidence: list[FrameEvidence] = []
    for slot, frame_id in enumerate(window.sampled_frame_ids):
        image = decoder(video_path, frame_id)
        if image is None:
            raise QaError(f"decoder returned no frame for {frame_id}")
        evidence.append(FrameEvidence(slot, frame_id, image))
    return tuple(evidence)


__all__ = ["EvidenceConfig", "build_evidence_windows", "decode_window", "sample_frame_ids"]

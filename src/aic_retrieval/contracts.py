"""Canonical IDs and manifest records shared by every AIC task."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


class ManifestError(ValueError):
    """Raised when dataset identity or frame mapping cannot be trusted."""


class Task(str, Enum):
    TEXTUAL_KIS = "textual-kis"
    QA = "qa"
    TRAKE = "trake"


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    video_path: str
    fps: float
    frame_count: int
    duration: float
    batch_id: str
    metadata_path: str | None = None

    def validate(self) -> None:
        if not self.video_id.strip():
            raise ManifestError("video_id must not be empty")
        if not self.video_path.strip():
            raise ManifestError(f"{self.video_id}: video_path must not be empty")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ManifestError(f"{self.video_id}: fps must be finite and positive")
        if self.frame_count <= 0:
            raise ManifestError(f"{self.video_id}: frame_count must be positive")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ManifestError(f"{self.video_id}: duration must be finite and positive")
        expected = self.frame_count / self.fps
        tolerance = max(1.0, 2.0 / self.fps)
        if abs(self.duration - expected) > tolerance:
            raise ManifestError(
                f"{self.video_id}: duration {self.duration} disagrees with "
                f"frame_count/fps {expected:.6f}"
            )
        if not self.batch_id.strip():
            raise ManifestError(f"{self.video_id}: batch_id must not be empty")


@dataclass(frozen=True, slots=True)
class KeyframeRecord:
    video_id: str
    keyframe_id: str
    keyframe_path: str
    keyframe_ordinal: int
    original_frame_id: int
    clip_row: int | None = None
    object_path: str | None = None

    def validate(self, video: VideoRecord) -> None:
        if self.video_id != video.video_id:
            raise ManifestError(
                f"{self.keyframe_id}: video_id {self.video_id!r} does not match "
                f"{video.video_id!r}"
            )
        if not self.keyframe_id.strip():
            raise ManifestError("keyframe_id must not be empty")
        if not self.keyframe_path.strip():
            raise ManifestError(f"{self.keyframe_id}: keyframe_path must not be empty")
        if self.keyframe_ordinal < 0:
            raise ManifestError(f"{self.keyframe_id}: keyframe_ordinal must be non-negative")
        if not 0 <= self.original_frame_id < video.frame_count:
            raise ManifestError(
                f"{self.keyframe_id}: original_frame_id {self.original_frame_id} "
                f"outside [0, {video.frame_count - 1}]"
            )
        if self.clip_row is not None and self.clip_row < 0:
            raise ManifestError(f"{self.keyframe_id}: clip_row must be non-negative")


@dataclass(frozen=True, slots=True)
class Manifest:
    videos: tuple[VideoRecord, ...]
    keyframes: tuple[KeyframeRecord, ...]
    version: int = MANIFEST_VERSION

    def validate(self) -> None:
        if self.version != MANIFEST_VERSION:
            raise ManifestError(
                f"unsupported manifest version {self.version}; expected {MANIFEST_VERSION}"
            )

        videos: dict[str, VideoRecord] = {}
        for video in self.videos:
            video.validate()
            if video.video_id in videos:
                raise ManifestError(f"duplicate video_id: {video.video_id}")
            videos[video.video_id] = video

        keyframe_ids: set[tuple[str, str]] = set()
        ordinals: set[tuple[str, int]] = set()
        clip_rows: set[tuple[str, int]] = set()
        previous_frame: dict[str, int] = {}
        for keyframe in self.keyframes:
            video = videos.get(keyframe.video_id)
            if video is None:
                raise ManifestError(
                    f"{keyframe.keyframe_id}: unknown video_id {keyframe.video_id!r}"
                )
            keyframe.validate(video)
            identity = (keyframe.video_id, keyframe.keyframe_id)
            ordinal = (keyframe.video_id, keyframe.keyframe_ordinal)
            if identity in keyframe_ids:
                raise ManifestError(f"duplicate keyframe identity: {identity}")
            if ordinal in ordinals:
                raise ManifestError(f"duplicate keyframe ordinal: {ordinal}")
            keyframe_ids.add(identity)
            ordinals.add(ordinal)

            if keyframe.clip_row is not None:
                clip_row = (keyframe.video_id, keyframe.clip_row)
                if clip_row in clip_rows:
                    raise ManifestError(f"duplicate CLIP row: {clip_row}")
                clip_rows.add(clip_row)

            prior = previous_frame.get(keyframe.video_id)
            if prior is not None and keyframe.original_frame_id <= prior:
                raise ManifestError(
                    f"{keyframe.video_id}: keyframes must be ordered by strictly "
                    "increasing original_frame_id"
                )
            previous_frame[keyframe.video_id] = keyframe.original_frame_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "videos": [asdict(video) for video in self.videos],
            "keyframes": [asdict(keyframe) for keyframe in self.keyframes],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manifest:
        try:
            manifest = cls(
                version=int(payload["version"]),
                videos=tuple(VideoRecord(**record) for record in payload["videos"]),
                keyframes=tuple(
                    KeyframeRecord(**record) for record in payload["keyframes"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestError(f"invalid manifest schema: {error}") from error
        manifest.validate()
        return manifest


def load_manifest(path: str | Path) -> Manifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be an object")
    return Manifest.from_dict(payload)


def save_manifest(manifest: Manifest, path: str | Path) -> None:
    manifest.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)

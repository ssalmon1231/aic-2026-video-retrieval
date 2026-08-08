"""Lazy exact-frame decoding by canonical original frame ID."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .contracts import VideoRecord


class VideoDecodeError(ValueError):
    """Raised when a requested source-video frame cannot be decoded exactly."""


def decode_frame(video_path: str | Path, frame_id: int) -> Any:
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise VideoDecodeError("frame_id must be a non-negative integer")
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoDecodeError(f"decoder could not open video {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0 and frame_id >= frame_count:
            raise VideoDecodeError(
                f"frame_id {frame_id} outside [0, {frame_count - 1}]"
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = capture.read()
        if not success or frame is None:
            raise VideoDecodeError(f"could not decode frame {frame_id}")
        decoded_id = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if decoded_id != frame_id:
            raise VideoDecodeError(
                f"decoder returned frame {decoded_id}, expected {frame_id}"
            )
        return frame
    finally:
        capture.release()


def iter_frame_window(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
) -> Iterator[tuple[int, Any]]:
    if (
        isinstance(start_frame, bool)
        or isinstance(end_frame, bool)
        or not isinstance(start_frame, int)
        or not isinstance(end_frame, int)
        or start_frame < 0
        or end_frame < start_frame
    ):
        raise VideoDecodeError(
            f"invalid frame window [{start_frame}, {end_frame}]"
        )
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoDecodeError(f"decoder could not open video {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0 and end_frame >= frame_count:
            raise VideoDecodeError(
                f"end_frame {end_frame} outside [0, {frame_count - 1}]"
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_id in range(start_frame, end_frame + 1):
            success, frame = capture.read()
            if not success or frame is None:
                raise VideoDecodeError(f"could not decode frame {frame_id}")
            decoded_id = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if decoded_id != frame_id:
                raise VideoDecodeError(
                    f"decoder returned frame {decoded_id}, expected {frame_id}"
                )
            yield frame_id, frame
    finally:
        capture.release()


def decode_manifest_frame(
    dataset_root: str | Path,
    video: VideoRecord,
    frame_id: int,
) -> Any:
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or not 0 <= frame_id < video.frame_count:
        raise VideoDecodeError(
            f"frame_id {frame_id} outside [0, {video.frame_count - 1}]"
        )
    return decode_frame(Path(dataset_root) / video.video_path, frame_id)


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "video decoding requires opencv-python-headless; install the 'video' extra"
        ) from error
    return cv2

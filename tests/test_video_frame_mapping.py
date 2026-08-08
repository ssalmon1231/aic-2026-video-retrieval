from __future__ import annotations

import unittest
from unittest import mock

from aic_retrieval.contracts import VideoRecord
from aic_retrieval.video import (
    VideoDecodeError,
    decode_frame,
    decode_manifest_frame,
    iter_frame_window,
)


class FakeCapture:
    def __init__(
        self,
        frames: list[object],
        *,
        opened: bool = True,
        position_offset: int = 0,
    ) -> None:
        self.frames = frames
        self.opened = opened
        self.position = 0
        self.position_offset = position_offset
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        if property_id == FakeCv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if property_id == FakeCv2.CAP_PROP_POS_FRAMES:
            return float(self.position + self.position_offset)
        raise AssertionError(property_id)

    def set(self, property_id: int, value: int) -> bool:
        if property_id != FakeCv2.CAP_PROP_POS_FRAMES:
            raise AssertionError(property_id)
        self.position = int(value)
        return True

    def read(self) -> tuple[bool, object | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_POS_FRAMES = 2

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.paths: list[str] = []

    def VideoCapture(self, path: str) -> FakeCapture:
        self.paths.append(path)
        return self.capture


class VideoFrameMappingTests(unittest.TestCase):
    def test_decode_exact_frame_and_release(self) -> None:
        capture = FakeCapture(["zero", "one", "two"])
        fake_cv2 = FakeCv2(capture)
        with mock.patch("aic_retrieval.video._cv2", return_value=fake_cv2):
            self.assertEqual(decode_frame("video.mp4", 1), "one")
        self.assertTrue(capture.released)

    def test_iter_window_is_inclusive_and_releases(self) -> None:
        capture = FakeCapture(["zero", "one", "two", "three"])
        with mock.patch(
            "aic_retrieval.video._cv2", return_value=FakeCv2(capture)
        ):
            self.assertEqual(
                list(iter_frame_window("video.mp4", 1, 3)),
                [(1, "one"), (2, "two"), (3, "three")],
            )
        self.assertTrue(capture.released)

    def test_rejects_out_of_bounds_and_decoder_mismatch(self) -> None:
        with mock.patch(
            "aic_retrieval.video._cv2",
            return_value=FakeCv2(FakeCapture(["zero"])),
        ):
            with self.assertRaisesRegex(VideoDecodeError, "outside"):
                decode_frame("video.mp4", 1)

        capture = FakeCapture(["zero"], position_offset=1)
        with mock.patch(
            "aic_retrieval.video._cv2", return_value=FakeCv2(capture)
        ):
            with self.assertRaisesRegex(VideoDecodeError, "expected 0"):
                decode_frame("video.mp4", 0)
        self.assertTrue(capture.released)

    def test_manifest_decoder_validates_original_frame_id(self) -> None:
        video = VideoRecord(
            video_id="L01_V001",
            video_path="Videos/L01_V001.mp4",
            fps=25.0,
            frame_count=3,
            duration=0.12,
            batch_id="batch-1",
        )
        with self.assertRaisesRegex(VideoDecodeError, "outside"):
            decode_manifest_frame("dataset", video, -1)
        with mock.patch(
            "aic_retrieval.video.decode_frame", return_value="frame"
        ) as decoder:
            self.assertEqual(decode_manifest_frame("dataset", video, 2), "frame")
        self.assertTrue(str(decoder.call_args.args[0]).endswith("Videos\\L01_V001.mp4"))
        self.assertEqual(decoder.call_args.args[1], 2)


if __name__ == "__main__":
    unittest.main()

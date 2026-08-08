from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aic_retrieval.contracts import (
    KeyframeRecord,
    Manifest,
    ManifestError,
    VideoRecord,
    load_manifest,
    save_manifest,
)


def valid_manifest() -> Manifest:
    return Manifest(
        videos=(
            VideoRecord(
                video_id="L01_V001",
                video_path="Videos/L01_V001.mp4",
                fps=25.0,
                frame_count=250,
                duration=10.0,
                batch_id="batch-1",
            ),
        ),
        keyframes=(
            KeyframeRecord(
                video_id="L01_V001",
                keyframe_id="001",
                keyframe_path="Keyframes/L01_V001/001.jpg",
                keyframe_ordinal=0,
                original_frame_id=25,
                clip_row=0,
            ),
            KeyframeRecord(
                video_id="L01_V001",
                keyframe_id="002",
                keyframe_path="Keyframes/L01_V001/002.jpg",
                keyframe_ordinal=1,
                original_frame_id=100,
                clip_row=1,
            ),
        ),
    )


class ManifestContractTests(unittest.TestCase):
    def test_manifest_round_trip(self) -> None:
        manifest = valid_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            save_manifest(manifest, path)
            self.assertEqual(load_manifest(path), manifest)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 1)

    def test_rejects_out_of_bounds_original_frame(self) -> None:
        manifest = valid_manifest()
        invalid = Manifest(
            videos=manifest.videos,
            keyframes=(
                KeyframeRecord(
                    video_id="L01_V001",
                    keyframe_id="bad",
                    keyframe_path="Keyframes/L01_V001/bad.jpg",
                    keyframe_ordinal=0,
                    original_frame_id=250,
                    clip_row=0,
                ),
            ),
        )
        with self.assertRaisesRegex(ManifestError, "outside"):
            invalid.validate()

    def test_rejects_non_monotonic_mapping(self) -> None:
        manifest = valid_manifest()
        invalid = Manifest(
            videos=manifest.videos,
            keyframes=(manifest.keyframes[1], manifest.keyframes[0]),
        )
        with self.assertRaisesRegex(ManifestError, "strictly increasing"):
            invalid.validate()

    def test_rejects_duplicate_clip_row(self) -> None:
        manifest = valid_manifest()
        second = manifest.keyframes[1]
        invalid = Manifest(
            videos=manifest.videos,
            keyframes=(
                manifest.keyframes[0],
                KeyframeRecord(
                    video_id=second.video_id,
                    keyframe_id=second.keyframe_id,
                    keyframe_path=second.keyframe_path,
                    keyframe_ordinal=second.keyframe_ordinal,
                    original_frame_id=second.original_frame_id,
                    clip_row=0,
                ),
            ),
        )
        with self.assertRaisesRegex(ManifestError, "duplicate CLIP row"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()

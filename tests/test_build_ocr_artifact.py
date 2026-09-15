from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aic_retrieval.contracts import KeyframeRecord
from aic_retrieval.index import ExactIndex, IndexMetadata
from aic_retrieval.ocr import OcrError
from scripts.build_ocr_artifact import _extract_records, _parse_output


def test_index() -> ExactIndex:
    records = tuple(
        KeyframeRecord(
            video_id="video",
            keyframe_id=f"{row:03d}",
            keyframe_path=f"frame-{row}.jpg",
            keyframe_ordinal=row,
            original_frame_id=10 + row,
            clip_row=row,
        )
        for row in range(2)
    )
    return ExactIndex(
        np.eye(2, dtype=np.float32),
        records,
        IndexMetadata(
            1,
            "numpy-flat-ip",
            "model",
            "preprocessing",
            2,
            2,
            "float32",
            True,
            "a" * 64,
        ),
    )


class FakeRuntime:
    def predict(self, paths):
        return (
            SimpleNamespace(
                json={
                    "res": {
                        "rec_texts": ["79H-6072", "low"],
                        "rec_scores": [0.95, 0.1],
                        "rec_boxes": [[0, 0, 10, 10], [1, 1, 2, 2]],
                    }
                }
            ),
            SimpleNamespace(
                json={
                    "res": {
                        "rec_texts": ["Đường Tiền Lân 11"],
                        "rec_scores": [0.85],
                        "rec_boxes": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                    }
                }
            ),
        )


class BuildOcrArtifactTests(unittest.TestCase):
    def test_extracts_canonical_records_and_filters_confidence(self) -> None:
        index = test_index()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for item in index.keyframes:
                (root / item.keyframe_path).write_bytes(b"private-frame")
            records = tuple(_extract_records(index, root, FakeRuntime(), 2, 0.35))
        self.assertEqual(tuple(item.row for item in records), (0, 1))
        self.assertEqual(records[0].raw_text, "79H-6072")
        self.assertEqual(records[1].box, (0.0, 0.0, 10.0, 0.0, 10.0, 5.0, 0.0, 5.0))

    def test_missing_keyframe_and_invalid_payload_fail_closed(self) -> None:
        index = test_index()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(OcrError, "missing canonical"):
                tuple(_extract_records(index, Path(temporary), FakeRuntime(), 2, 0.35))
        with self.assertRaisesRegex(OcrError, "misses recognition"):
            _parse_output({"res": {}})
        with self.assertRaisesRegex(OcrError, "inconsistent"):
            _parse_output(
                {
                    "rec_texts": ["one"],
                    "rec_scores": [],
                    "rec_boxes": [],
                }
            )
        for score in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(score=score):
                with self.assertRaisesRegex(OcrError, r"finite and in \[0, 1\]"):
                    _parse_output(
                        {
                            "rec_texts": ["one"],
                            "rec_scores": [score],
                            "rec_boxes": [[0, 0, 1, 1]],
                        }
                    )


if __name__ == "__main__":
    unittest.main()

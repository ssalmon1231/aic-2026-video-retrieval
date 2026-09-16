from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aic_retrieval.contracts import KeyframeRecord
from aic_retrieval.index import ExactIndex, IndexMetadata
from aic_retrieval.ocr import (
    DESCRIPTOR_NAME,
    RECORDS_NAME,
    OcrError,
    OcrIndex,
    OcrProvenance,
    OcrRecord,
    alphanumeric_fold,
    create_ocr_artifact,
    load_ocr_artifact,
    normalize_visible_text,
)


def test_index(manifest_sha256: str = "a" * 64) -> ExactIndex:
    records = tuple(
        KeyframeRecord(
            video_id=f"video-{row // 2}",
            keyframe_id=f"{row:03d}",
            keyframe_path=f"{row}.jpg",
            keyframe_ordinal=row % 2,
            original_frame_id=100 + row * 10,
            clip_row=0,
        )
        for row in range(6)
    )
    vectors = np.eye(6, dtype=np.float32)
    return ExactIndex(
        vectors,
        records,
        IndexMetadata(
            1,
            "numpy-flat-ip",
            "model",
            "preprocessing",
            6,
            6,
            "float32",
            True,
            manifest_sha256,
        ),
    )


def provenance() -> OcrProvenance:
    return OcrProvenance(
        "paddleocr",
        "3.2.0",
        "detector",
        "recognizer",
        "model-revision",
    )


def record(index: ExactIndex, row: int, text: str, confidence: float = 0.9) -> OcrRecord:
    keyframe = index.keyframes[row]
    return OcrRecord.create(
        row,
        keyframe.video_id,
        keyframe.original_frame_id,
        text,
        confidence,
        (0.0, 0.0, 10.0, 10.0),
    )


def sample_records(index: ExactIndex) -> tuple[OcrRecord, ...]:
    return (
        record(index, 0, "79H-6072"),
        record(index, 1, "Đường Tiền Lân 11"),
        record(index, 2, "Bánh bèo cô Anh"),
        record(index, 3, "Sạp 96"),
        record(index, 4, "Đường Tiền Lân l1", 0.8),
        record(index, 5, "AB", 0.99),
    )


class OcrSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = test_index()
        self.ocr = OcrIndex(
            metadata=create_metadata(self.index, sample_records(self.index)),
            records=sample_records(self.index),
        )

    def test_normalizes_nfc_whitespace_and_alphanumeric_text(self) -> None:
        self.assertEqual(
            normalize_visible_text("  Đường\n  Tiền   Lân 11 "),
            "Đường Tiền Lân 11",
        )
        self.assertEqual(alphanumeric_fold("79H-6072"), "79h6072")
        self.assertEqual(alphanumeric_fold("Sạp 96"), "sap96")
        self.assertEqual(alphanumeric_fold("Đường Tiền Lân 11"), "duongtienlan11")

    def test_plate_punctuation_and_case_fold_match(self) -> None:
        matches = self.ocr.search("79h6072")
        self.assertEqual(matches[0].row, 0)
        self.assertTrue(matches[0].exact)

    def test_diacritic_phrase_shop_and_stall_match(self) -> None:
        for query, expected_row in (
            ("ĐƯỜNG TIỀN LÂN 11", 1),
            ("bánh bèo cô Anh", 2),
            ("sạp 96", 3),
        ):
            with self.subTest(query=query):
                match = self.ocr.search(query)[0]
                self.assertEqual(match.row, expected_row)
                self.assertTrue(match.exact)

    def test_bounded_noise_matches_but_short_strings_do_not_fuzzy_match(self) -> None:
        noisy = self.ocr.search("Đường Tiền Lân 11")
        self.assertIn(4, tuple(match.row for match in noisy))
        self.assertFalse(next(match for match in noisy if match.row == 4).exact)
        self.assertEqual(self.ocr.search("AC"), ())
        self.assertEqual(self.ocr.exact_candidates("tiền"), ())

    def test_allowed_rows_bound_search(self) -> None:
        matches = self.ocr.search("Đường Tiền Lân 11", allowed_rows={4})
        self.assertEqual(tuple(match.row for match in matches), (4,))
        self.assertEqual(self.ocr.search("Đường Tiền Lân 11", allowed_rows=set()), ())

    def test_exact_candidates_use_only_full_inverted_keys(self) -> None:
        self.assertEqual(
            tuple(match.row for match in self.ocr.exact_candidates("79H6072")),
            (0,),
        )
        self.assertEqual(
            tuple(
                match.row
                for match in self.ocr.exact_candidates("duong tien lan 11")
            ),
            (1,),
        )
        phrase_matches = self.ocr.exact_candidates("Tiền Lân")
        self.assertEqual(tuple(match.row for match in phrase_matches), (1, 4))
        self.assertTrue(all(match.exact for match in phrase_matches))


class OcrArtifactTests(unittest.TestCase):
    def test_round_trip_is_deterministic(self) -> None:
        index = test_index()
        records = sample_records(index)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ocr"
            metadata = create_ocr_artifact(root, index, tuple(reversed(records)), provenance())
            loaded = load_ocr_artifact(root, index)
            self.assertEqual(loaded.metadata, metadata)
            self.assertEqual(
                tuple(item.row for item in loaded.records),
                tuple(range(len(records))),
            )
            self.assertEqual(
                json.loads((root / DESCRIPTOR_NAME).read_text(encoding="utf-8"))[
                    "record_count"
                ],
                len(records),
            )

    def test_duplicate_records_and_wrong_row_identity_fail_closed(self) -> None:
        index = test_index()
        duplicate = record(index, 0, "same")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(OcrError, "duplicate"):
                create_ocr_artifact(
                    Path(temporary) / "duplicate",
                    index,
                    (duplicate, duplicate),
                    provenance(),
                )
            wrong = OcrRecord.create(
                0,
                "wrong-video",
                index.keyframes[0].original_frame_id,
                "text",
                0.9,
                (0.0, 0.0, 1.0, 1.0),
            )
            with self.assertRaisesRegex(OcrError, "identity"):
                create_ocr_artifact(
                    Path(temporary) / "wrong",
                    index,
                    (wrong,),
                    provenance(),
                )

    def test_manifest_mismatch_changed_bytes_and_extra_member_fail_closed(self) -> None:
        index = test_index()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ocr"
            create_ocr_artifact(root, index, sample_records(index), provenance())
            with self.assertRaisesRegex(OcrError, "manifest"):
                load_ocr_artifact(root, test_index("b" * 64))

            records_path = root / RECORDS_NAME
            records_path.write_bytes(records_path.read_bytes() + b" ")
            with self.assertRaisesRegex(OcrError, "checksum"):
                load_ocr_artifact(root, index)

            create_ocr_artifact(root, index, sample_records(index), provenance())
            (root / "private-labels.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(OcrError, "unexpected"):
                load_ocr_artifact(root, index)

    def test_missing_artifact_has_neutral_optional_loader(self) -> None:
        from aic_retrieval.ocr import load_optional_ocr_artifact

        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            self.assertIsNone(load_optional_ocr_artifact(missing, test_index()))


def create_metadata(
    index: ExactIndex,
    records: tuple[OcrRecord, ...],
):
    from aic_retrieval.ocr import OcrArtifactMetadata

    return OcrArtifactMetadata(
        1,
        index.metadata.manifest_sha256,
        len(index.keyframes),
        len(records),
        "b" * 64,
        provenance(),
    )


if __name__ == "__main__":
    unittest.main()

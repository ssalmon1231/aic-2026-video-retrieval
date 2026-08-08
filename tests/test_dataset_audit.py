from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from aic_retrieval.data import (
    AuditConfig,
    MappingRecord,
    VideoProbe,
    _candidate_frame_ids,
    _read_mapping,
    _sample_evenly,
    audit_dataset,
    diagnose_dataset,
    load_config,
    object_schema_warnings,
    publish_audit,
    summarize_object_schemas,
)


class DatasetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for directory in (
            "Videos",
            "Keyframes/L01_V001",
            "Map-keyframes",
            "CLIP-features",
            "Objects/L01_V001",
            "Metadata",
            "output",
        ):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / "Videos/L01_V001.mp4").write_bytes(b"fixture")
        (self.root / "Keyframes/L01_V001/001.jpg").write_bytes(b"frame-1")
        (self.root / "Keyframes/L01_V001/002.jpg").write_bytes(b"frame-2")
        (self.root / "Map-keyframes/L01_V001.csv").write_text(
            "n,frame_idx\n001,10\n002,50\n",
            encoding="utf-8",
        )
        (self.root / "Objects/L01_V001/001.json").write_text("[]\n", encoding="utf-8")
        (self.root / "Metadata/L01_V001.json").write_text("{}\n", encoding="utf-8")
        np.save(
            self.root / "CLIP-features/L01_V001.npy",
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        self.config = AuditConfig(
            version=1,
            batch_id="batch-1",
            dataset_root=self.root,
            video_glob="Videos/**/*.*",
            keyframe_glob="Keyframes/**/*.jpg",
            mapping_glob="Map-keyframes/**/*.csv",
            clip_glob="CLIP-features/**/*.npy",
            object_glob="Objects/**/*.json",
            metadata_glob="Metadata/**/*.json",
            output_manifest=self.root / "output/manifest.json",
            output_report=self.root / "output/report.json",
            expected_clip_dimension=2,
            verify_frame_mapping=False,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def video_probe(_: Path) -> VideoProbe:
        return VideoProbe(fps=25.0, frame_count=250, duration=10.0)

    def test_builds_manifest_without_mutating_source(self) -> None:
        original_video = (self.root / "Videos/L01_V001.mp4").read_bytes()
        result = audit_dataset(self.config, video_prober=self.video_probe)
        self.assertTrue(result.valid, result.issues)
        self.assertEqual(len(result.manifest.videos), 1)  # type: ignore[union-attr]
        self.assertEqual(
            [record.original_frame_id for record in result.manifest.keyframes],  # type: ignore[union-attr]
            [10, 50],
        )
        self.assertEqual(
            result.manifest.keyframes[1].clip_row,  # type: ignore[union-attr]
            1,
        )
        publish_audit(self.config, result)
        self.assertTrue(self.config.output_manifest.exists())
        self.assertTrue(json.loads(self.config.output_report.read_text())["valid"])
        self.assertEqual((self.root / "Videos/L01_V001.mp4").read_bytes(), original_video)

    def test_clip_row_mismatch_blocks_manifest(self) -> None:
        np.save(
            self.root / "CLIP-features/L01_V001.npy",
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        result = audit_dataset(self.config, video_prober=self.video_probe)
        self.assertFalse(result.valid)
        self.assertIsNone(result.manifest)
        self.assertIn("clip-row-mismatch", {issue.code for issue in result.issues})

    def test_clip_dimension_mismatch_blocks_manifest(self) -> None:
        np.save(
            self.root / "CLIP-features/L01_V001.npy",
            np.ones((2, 3), dtype=np.float32),
        )
        result = audit_dataset(self.config, video_prober=self.video_probe)
        self.assertFalse(result.valid)
        self.assertIn("clip-dimension-mismatch", {issue.code for issue in result.issues})

    def test_non_normalized_clip_is_warning_only(self) -> None:
        np.save(
            self.root / "CLIP-features/L01_V001.npy",
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        )
        result = audit_dataset(self.config, video_prober=self.video_probe)
        self.assertTrue(result.valid, result.issues)
        self.assertIn("clip-not-normalized", {issue.code for issue in result.issues})

    def test_invalid_support_json_is_warning_only(self) -> None:
        (self.root / "Metadata/L01_V001.json").write_text("{", encoding="utf-8")
        (self.root / "Objects/L01_V001/001.json").write_text("{", encoding="utf-8")
        result = audit_dataset(self.config, video_prober=self.video_probe)
        self.assertTrue(result.valid, result.issues)
        self.assertIn("metadata-invalid", {issue.code for issue in result.issues})
        self.assertIn("object-invalid", {issue.code for issue in result.issues})

    def test_invalid_audit_removes_stale_manifest(self) -> None:
        valid = audit_dataset(self.config, video_prober=self.video_probe)
        publish_audit(self.config, valid)
        self.assertTrue(self.config.output_manifest.exists())
        (self.root / "Map-keyframes/L01_V001.csv").unlink()
        invalid = audit_dataset(self.config, video_prober=self.video_probe)
        publish_audit(self.config, invalid)
        self.assertFalse(self.config.output_manifest.exists())

    def test_missing_metadata_is_warning_only(self) -> None:
        (self.root / "Metadata/L01_V001.json").unlink()
        result = audit_dataset(self.config, video_prober=self.video_probe)
        self.assertTrue(result.valid, result.issues)
        self.assertIn("metadata-missing", {issue.code for issue in result.issues})

    def test_missing_mapping_is_error(self) -> None:
        (self.root / "Map-keyframes/L01_V001.csv").unlink()
        result = audit_dataset(self.config, video_prober=self.video_probe)
        self.assertFalse(result.valid)
        self.assertIn("mapping-missing", {issue.code for issue in result.issues})

    def test_frame_mapping_failure_blocks_manifest(self) -> None:
        config = replace(self.config, verify_frame_mapping=True)
        with mock.patch(
            "aic_retrieval.data.verify_frame_mapping",
            side_effect=ValueError("decoded frame differs"),
        ):
            result = audit_dataset(config, video_prober=self.video_probe)
        self.assertFalse(result.valid)
        self.assertIn("frame-mapping-mismatch", {issue.code for issue in result.issues})

    def test_single_frame_mapping_sample_uses_middle_value(self) -> None:
        self.assertEqual(_sample_evenly([10, 20, 30], 1), [20])

    def test_mapping_loader_preserves_optional_timestamp_fields(self) -> None:
        path = self.root / "Map-keyframes/L01_V001.csv"
        path.write_text(
            "n,pts_time,fps,frame_idx\n001,0.0,25,0\n002,0.04,25,1\n",
            encoding="utf-8",
        )
        records = _read_mapping(path, "n", "frame_idx")
        self.assertEqual(
            records,
            [
                MappingRecord("001", 0, 0, 0.0, 25.0),
                MappingRecord("002", 1, 1, 0.04, 25.0),
            ],
        )

    def test_candidate_frames_include_csv_timestamp_and_neighbors(self) -> None:
        self.assertEqual(
            _candidate_frame_ids(MappingRecord("002", 1, 0, 0.04, 25.0)),
            {
                "csv": 0,
                "csv_plus_1": 1,
                "timestamp": 1,
                "timestamp_minus_1": 0,
                "timestamp_plus_1": 2,
            },
        )

    def test_timestamp_mapping_builds_manifest_and_preserves_rows(self) -> None:
        (self.root / "Map-keyframes/L01_V001.csv").write_text(
            "n,pts_time,fps,frame_idx\n001,0.4,25,10\n002,2.0,25,10\n",
            encoding="utf-8",
        )
        result = audit_dataset(
            replace(self.config, frame_id_source="timestamp"),
            video_prober=self.video_probe,
        )
        self.assertTrue(result.valid, result.issues)
        self.assertEqual(
            [record.original_frame_id for record in result.manifest.keyframes],  # type: ignore[union-attr]
            [10, 50],
        )
        self.assertEqual(
            [record.clip_row for record in result.manifest.keyframes],  # type: ignore[union-attr]
            [0, 1],
        )

    def test_timestamp_mapping_requires_complete_fields(self) -> None:
        result = audit_dataset(
            replace(self.config, frame_id_source="timestamp"),
            video_prober=self.video_probe,
        )
        self.assertFalse(result.valid)
        self.assertTrue(
            any("timestamp frame ID requires" in issue.message for issue in result.issues)
        )

    def test_config_loads_explicit_timestamp_source(self) -> None:
        path = self.root / "dataset.json"
        payload = {
            **{field: str(getattr(self.config, field)) for field in (
                "batch_id",
                "dataset_root",
                "video_glob",
                "keyframe_glob",
                "mapping_glob",
                "clip_glob",
                "object_glob",
                "metadata_glob",
                "output_manifest",
                "output_report",
            )},
            "version": 1,
            "frame_id_source": "timestamp",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(load_config(path).frame_id_source, "timestamp")

    def test_parallel_object_arrays_accept_numeric_score_strings(self) -> None:
        warnings = object_schema_warnings(
            {
                "detection_scores": ["0.9", "0"],
                "detection_class_names": ["person", "car"],
                "detection_class_entities": ["person", "vehicle"],
                "detection_boxes": [
                    ["0.1", "0.2", "0.3", "0.4"],
                    [1, 2, 3, 4],
                ],
                "detection_class_labels": [1, "car"],
            }
        )
        self.assertEqual(warnings, [])

    def test_parallel_object_arrays_reject_bad_values_without_leaking_them(self) -> None:
        secret = "private-score"
        warnings = object_schema_warnings(
            {
                "detection_scores": [secret],
                "detection_class_names": [3],
                "detection_class_entities": ["person"],
                "detection_boxes": [[0.1, 0.2, 0.3]],
                "detection_class_labels": [True],
            }
        )
        self.assertEqual(len(warnings), 4)
        self.assertNotIn(secret, " ".join(warnings))

    def test_parallel_object_arrays_reject_invalid_string_coordinates(self) -> None:
        for coordinate in ("private-coordinate", "nan", "inf", "-0.1"):
            with self.subTest(coordinate=coordinate):
                warnings = object_schema_warnings(
                    {
                        "detection_scores": ["0.9"],
                        "detection_class_names": ["person"],
                        "detection_class_entities": ["person"],
                        "detection_boxes": [[coordinate, "0.2", "0.3", "0.4"]],
                        "detection_class_labels": ["1"],
                    }
                )
                self.assertEqual(warnings, ["detection 0 has invalid bbox"])
                self.assertNotIn(coordinate, warnings[0])

    def test_parallel_object_schema_summary_is_aggregate_only(self) -> None:
        valid = self.root / "Objects/L01_V001/001.json"
        invalid_lengths = self.root / "Objects/L01_V001/002.json"
        valid.write_text(
            json.dumps(
                {
                    "detection_scores": [0.9],
                    "detection_class_names": ["person"],
                    "detection_class_entities": ["person"],
                    "detection_boxes": [[0.1, 0.2, 0.3, 0.4]],
                    "detection_class_labels": [1],
                }
            ),
            encoding="utf-8",
        )
        invalid_lengths.write_text(
            json.dumps(
                {
                    "detection_scores": [1.2, 0.5],
                    "detection_class_names": ["person"],
                    "detection_class_entities": ["person"],
                    "detection_boxes": [[0.1, 0.2, 0.3]],
                    "detection_class_labels": [1],
                }
            ),
            encoding="utf-8",
        )
        summary = summarize_object_schemas([valid, invalid_lengths])
        self.assertEqual(summary["equal_length_files"], 1)
        self.assertEqual(summary["unequal_or_missing_length_files"], 1)
        self.assertEqual(summary["box_shape_counts"], {"list[3]": 1, "list[4]": 1})
        self.assertEqual(summary["scores_outside_unit_interval"], 1)
        self.assertNotIn("person", json.dumps(summary))

    def test_object_schema_summary_counts_invalid_roots(self) -> None:
        invalid_json = self.root / "Objects/L01_V001/001.json"
        non_object = self.root / "Objects/L01_V001/002.json"
        invalid_json.write_text("{", encoding="utf-8")
        non_object.write_text("[]", encoding="utf-8")

        summary = summarize_object_schemas([invalid_json, non_object])
        self.assertEqual(summary["sampled_files"], 2)
        self.assertEqual(summary["invalid_json_files"], 1)
        self.assertEqual(summary["non_object_root_files"], 1)
        self.assertEqual(summary["unequal_or_missing_length_files"], 1)

    def test_diagnostic_detects_duplicates_without_relaxing_manifest(self) -> None:
        (self.root / "Map-keyframes/L01_V001.csv").write_text(
            "n,pts_time,fps,frame_idx\n001,0.0,25,0\n002,0.04,25,0\n",
            encoding="utf-8",
        )
        (self.root / "Objects/L01_V001/001.json").write_text(
            json.dumps(
                {
                    "detection_scores": [0.9],
                    "detection_class_names": ["person"],
                    "detection_class_entities": ["person"],
                    "detection_boxes": [[0.1, 0.2, 0.3, 0.4]],
                    "detection_class_labels": [1],
                }
            ),
            encoding="utf-8",
        )
        (self.root / "Objects/L01_V001/002.json").write_text(
            (self.root / "Objects/L01_V001/001.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        def compare(
            _: Path, samples: list[tuple[Path, dict[str, int]]]
        ) -> list[dict[str, float]]:
            return [
                {label: float(abs(frame_id - 1)) for label, frame_id in candidates.items()}
                for _, candidates in samples
            ]

        report = diagnose_dataset(
            self.config,
            frame_comparer=compare,
            frame_sample_limit=10,
            object_sample_limit=10,
        )
        self.assertTrue(report["evidence_ready"])
        self.assertEqual(report["frame_mapping"]["duplicate_video_count"], 1)
        self.assertEqual(report["frame_mapping"]["compared_rows"], 2)
        self.assertEqual(report["frame_mapping"]["winner_counts"]["timestamp"], 1)
        self.assertFalse(audit_dataset(self.config, video_prober=self.video_probe).valid)

    def test_incomplete_diagnostic_is_not_ready(self) -> None:
        report = diagnose_dataset(
            self.config,
            frame_comparer=lambda _video, _samples: [],
            object_sample_limit=1,
        )
        self.assertFalse(report["evidence_ready"])
        self.assertEqual(report["frame_mapping"]["duplicate_video_count"], 0)


if __name__ == "__main__":
    unittest.main()

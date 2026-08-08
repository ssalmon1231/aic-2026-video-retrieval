from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from aic_retrieval.contracts import KeyframeRecord, Manifest, VideoRecord, save_manifest
from aic_retrieval.index import (
    IndexError,
    build_index,
    load_config,
    load_index,
    save_index,
)


VIDEO_ID = "L01_V001"


def manifest_for(rows: int = 3) -> Manifest:
    return Manifest(
        videos=(
            VideoRecord(
                video_id=VIDEO_ID,
                video_path=f"Videos/{VIDEO_ID}.mp4",
                fps=25.0,
                frame_count=250,
                duration=10.0,
                batch_id="batch-1",
            ),
        ),
        keyframes=tuple(
            KeyframeRecord(
                video_id=VIDEO_ID,
                keyframe_id=f"{row + 1:03d}",
                keyframe_path=f"Keyframes/{VIDEO_ID}/{row + 1:03d}.jpg",
                keyframe_ordinal=row,
                original_frame_id=(row + 1) * 25,
                clip_row=row,
            )
            for row in range(rows)
        ),
    )


def write_features(root: Path, vectors: np.ndarray) -> None:
    destination = root / "CLIP-features" / "batch-1"
    destination.mkdir(parents=True)
    np.save(destination / f"{VIDEO_ID}.npy", vectors)


def write_real_layout_features(root: Path, vectors: np.ndarray) -> None:
    destination = root / "clip-features-32-aic25-b1" / "clip-features-32"
    destination.mkdir(parents=True)
    np.save(destination / f"{VIDEO_ID}.npy", vectors)


class IndexRoundTripTests(unittest.TestCase):
    def test_build_search_save_load_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_features(
                root,
                np.array(
                    [
                        [3.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [1.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            index = build_index(
                manifest_for(),
                root,
                model_id="test-model",
                preprocessing="test-preprocessing",
                manifest_sha256="abc",
            )
            self.assertTrue(
                np.allclose(np.linalg.norm(index.vectors, axis=1), np.ones(3))
            )
            hit = index.search(np.array([0.0, 9.0, 0.0]), limit=1)[0]
            self.assertEqual(
                (hit.video_id, hit.keyframe_id, hit.original_frame_id),
                (VIDEO_ID, "002", 50),
            )

            destination = root / "active.npz"
            save_index(index, destination)
            loaded = load_index(destination)
            self.assertEqual(loaded.metadata, index.metadata)
            self.assertEqual(loaded.keyframes, index.keyframes)
            self.assertTrue(np.array_equal(loaded.vectors, index.vectors))
            self.assertEqual(loaded.search(index.vectors[0]), index.search(index.vectors[0]))

    def test_build_accepts_real_feature_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_real_layout_features(root, np.eye(2, dtype=np.float32))
            index = build_index(
                manifest_for(2),
                root,
                model_id="test-model",
                preprocessing="test-preprocessing",
                manifest_sha256="abc",
                feature_glob=(
                    "clip-features-32-aic25-b1/clip-features-32/{video_id}.npy"
                ),
            )
            self.assertEqual(index.metadata.rows, 2)

    def test_duplicate_feature_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_features(root, np.eye(2, dtype=np.float32))
            write_real_layout_features(root, np.eye(2, dtype=np.float32))
            with self.assertRaisesRegex(IndexError, "found 2"):
                build_index(
                    manifest_for(2),
                    root,
                    model_id="test-model",
                    preprocessing="test-preprocessing",
                    manifest_sha256="abc",
                    feature_glob="**/{video_id}.npy",
                )

    def test_duplicate_vectors_are_valid_ties(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_features(root, np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
            index = build_index(
                manifest_for(2),
                root,
                model_id="test-model",
                preprocessing="test-preprocessing",
                manifest_sha256="abc",
            )
            save_index(index, root / "active.npz")
            self.assertEqual(
                tuple(hit.keyframe_id for hit in index.search(index.vectors[1], limit=2)),
                ("001", "002"),
            )

    def test_failed_staging_verification_preserves_active_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_features(root, np.eye(2, dtype=np.float32))
            index = build_index(
                manifest_for(2),
                root,
                model_id="test-model",
                preprocessing="test-preprocessing",
                manifest_sha256="abc",
            )
            destination = root / "active.npz"
            save_index(index, destination)
            active = destination.read_bytes()

            with mock.patch(
                "aic_retrieval.index.verify_index",
                side_effect=IndexError("forced verification failure"),
            ):
                with self.assertRaisesRegex(IndexError, "forced verification failure"):
                    save_index(index, destination)
            self.assertEqual(destination.read_bytes(), active)
            self.assertFalse((root / ".active.npz.staging.npz").exists())

    def test_rejects_bad_features_and_query(self) -> None:
        cases = (
            (np.array([[0.0, 0.0]], dtype=np.float32), "zero-norm"),
            (np.array([[np.nan, 1.0]], dtype=np.float32), "NaN or infinity"),
        )
        for vectors, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_features(root, vectors)
                with self.assertRaisesRegex(IndexError, message):
                    build_index(
                        manifest_for(1),
                        root,
                        model_id="test-model",
                        preprocessing="test-preprocessing",
                        manifest_sha256="abc",
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_features(root, np.eye(2, dtype=np.float32))
            index = build_index(
                manifest_for(2),
                root,
                model_id="test-model",
                preprocessing="test-preprocessing",
                manifest_sha256="abc",
            )
            with self.assertRaisesRegex(IndexError, "query shape"):
                index.search(np.ones(3, dtype=np.float32))

    def test_config_loader_and_manifest_hash_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            save_manifest(manifest_for(1), manifest_path)
            payload = {
                "manifest_path": str(manifest_path),
                "dataset_root": str(root),
                "output_path": str(root / "index.npz"),
                "model_id": "test-model",
                "preprocessing": "test-preprocessing",
            }
            config_path = root / "index.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(config.manifest_path, manifest_path)
            self.assertEqual(
                config.feature_glob,
                "CLIP-features/**/{video_id}.npy",
            )
            payload["feature_glob"] = "features/{video_id}.npy"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_config(config_path).feature_glob,
                "features/{video_id}.npy",
            )
            payload["model_id"] = ""
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(IndexError, "non-empty strings"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()

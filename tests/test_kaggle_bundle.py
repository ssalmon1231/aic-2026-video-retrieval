from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from scripts.prepare_kaggle_bundle import build_bundle

ROOT = Path(__file__).parents[1]
REQUIRED = {
    "pyproject.toml",
    "src/aic_retrieval/__init__.py",
    "src/aic_retrieval/hybrid_retrieval.py",
    "src/aic_retrieval/ocr.py",
    "src/aic_retrieval/qa.py",
    "src/aic_retrieval/qa_evidence.py",
    "src/aic_retrieval/qa_models.py",
    "src/aic_retrieval/query_pack.py",
    "src/aic_retrieval/alignment.py",
    "src/aic_retrieval/trake.py",
    "src/aic_retrieval/query_planning.py",
    "src/aic_retrieval/temporal.py",
    "scripts/audit_dataset.py",
    "scripts/build_index.py",
    "scripts/verify_index.py",
    "scripts/search.py",
    "scripts/benchmark_kis.py",
    "scripts/build_ocr_artifact.py",
    "scripts/prepare_kaggle_bundle.py",
    "scripts/run_query_pack.py",
    "scripts/verify_python310_ast.py",
    "config/dataset.example.yaml",
    "config/index.example.yaml",
    "config/query-encoder.example.yaml",
    "config/query-encoder-multilingual.example.yaml",
    "config/reranker-disabled.example.yaml",
    "config/hybrid-retrieval.example.yaml",
    "config/qa-baseline.yaml",
    "config/trake-baseline.yaml",
    "config/retrieval-baseline.yaml",
    "config/kis-benchmark.example.yaml",
    "docs/kaggle-runbook.md",
    "tests/test_build_ocr_artifact.py",
    "tests/test_hybrid_retrieval.py",
    "tests/test_ocr.py",
    "tests/test_qa_pipeline.py",
    "tests/test_query_planning.py",
    "tests/test_temporal.py",
}
FORBIDDEN_SUFFIXES = {".avi", ".faiss", ".mkv", ".mp4", ".npy", ".npz", ".pdf", ".webm"}


class KaggleBundleTests(unittest.TestCase):
    def test_bundle_is_deterministic_and_source_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.zip"
            second = Path(temporary) / "second.zip"
            first_count, first_checksum = build_bundle(ROOT, first)
            second_count, second_checksum = build_bundle(ROOT, second)
            self.assertEqual(first_count, second_count)
            self.assertEqual(first_checksum, second_checksum)

            with ZipFile(first) as archive:
                names = archive.namelist()
                self.assertEqual(names, sorted(names))
                self.assertTrue(REQUIRED <= set(names))
                for name in names:
                    path = PurePosixPath(name)
                    self.assertFalse(path.is_absolute())
                    self.assertNotIn("..", path.parts)
                    self.assertFalse({".git", "plans", "data", "dataset"} & set(path.parts))
                    self.assertNotIn(path.suffix.lower(), FORBIDDEN_SUFFIXES)
                    self.assertFalse(name.endswith(".zip"))

                extraction = Path(temporary) / "extracted"
                archive.extractall(extraction)
                self.assertTrue((extraction / "pyproject.toml").is_file())
                self.assertTrue((extraction / "src/aic_retrieval/__init__.py").is_file())

    def test_rejects_non_zip_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, r"\.zip"):
                build_bundle(ROOT, Path(temporary) / "bundle.bin")

    def test_cli_builds_parseable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle.zip"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_kaggle_bundle.py"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(output.is_file())
            self.assertIn("bundle ready:", completed.stdout)
            self.assertIn("sha256:", completed.stdout)
            with ZipFile(output) as archive:
                self.assertIn("pyproject.toml", archive.namelist())


if __name__ == "__main__":
    unittest.main()

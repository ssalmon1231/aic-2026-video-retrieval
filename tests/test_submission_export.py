from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from aic_retrieval.evaluation import KisResponse, QaResponse, TrakeResponse
from aic_retrieval.submission import build_submission_zip, write_submission_csv


class SubmissionExportTests(unittest.TestCase):
    def test_writes_headerless_utf8_csv_with_quoted_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "query-3-qa.csv"
            write_submission_csv(path, (QaResponse("L01_V028", 3450, "Có 3 người, gồm nam và nữ"),))
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                'L01_V028,3450,"Có 3 người, gồm nam và nữ"\n',
            )
            with path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(next(csv.reader(stream)), ["L01_V028", "3450", "Có 3 người, gồm nam và nữ"])

    def test_rejects_oversized_answer_and_video_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "query-3-qa.csv"
            with self.assertRaisesRegex(ValueError, "100"):
                write_submission_csv(path, (QaResponse("L01_V028", 1, "x" * 101),))
            with self.assertRaisesRegex(ValueError, r"\.mp4"):
                write_submission_csv(path, (KisResponse("L01_V028.mp4", 1),))

    def test_builds_zip_with_submission_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "team.zip"
            build_submission_zip(
                output,
                {
                    "query-1-kis.csv": (KisResponse("L00_V000", 1234),),
                    "query-3-qa.csv": (QaResponse("L01_V028", 3450, "5"),),
                    "query-4-trake.csv": (TrakeResponse("L10_V001", (1200, 1850, 2100, 2450)),),
                },
            )
            with ZipFile(output) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "submission/query-1-kis.csv",
                        "submission/query-3-qa.csv",
                        "submission/query-4-trake.csv",
                    ],
                )
                self.assertEqual(archive.read("submission/query-4-trake.csv").decode("utf-8"), "L10_V001,1200,1850,2100,2450\n")


if __name__ == "__main__":
    unittest.main()

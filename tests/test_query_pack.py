from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aic_retrieval.contracts import Task
from aic_retrieval.query_pack import QueryPackDiagnostic, load_query_pack, summarize_diagnostics


class QueryPackTests(unittest.TestCase):
    def test_loads_recognized_suffixes_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "query-2-qa.txt").write_text("question", encoding="utf-8")
            (root / "query-1-kis.txt").write_text("query", encoding="utf-8")
            (root / "query-3-trake.txt").write_text("event", encoding="utf-8")
            (root / "ignored.md").write_text("ignored", encoding="utf-8")
            entries = load_query_pack(root)
        self.assertEqual(
            [(entry.query_id, entry.task) for entry in entries],
            [
                ("query-1-kis", Task.TEXTUAL_KIS),
                ("query-2-qa", Task.QA),
                ("query-3-trake", Task.TRAKE),
            ],
        )

    def test_rejects_empty_recognized_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "query-1-kis.txt").write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                load_query_pack(root)

    def test_summary_never_contains_predictions_or_query_text(self) -> None:
        report = summarize_diagnostics((
            QueryPackDiagnostic("query-1-kis", Task.TEXTUAL_KIS, 100, 1.0, "completed"),
            QueryPackDiagnostic("query-2-qa", Task.QA, 0, 2.0, "fail-closed"),
        ))
        self.assertEqual(report["query_count"], 2)
        self.assertNotIn("queries", report)
        self.assertEqual(report["by_status"], {"completed": 1, "fail-closed": 1})


if __name__ == "__main__":
    unittest.main()

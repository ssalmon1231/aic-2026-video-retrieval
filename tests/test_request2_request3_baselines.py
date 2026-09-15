from __future__ import annotations

import unittest

from aic_retrieval.qa_evidence import sample_frame_ids
from aic_retrieval.qa_models import parse_answer_protocol
from aic_retrieval.trake import parse_trake_query


class QaEvidenceTests(unittest.TestCase):
    def test_sampling_is_sorted_bounded_and_deterministic(self) -> None:
        expected = (0, 5, 10)
        self.assertEqual(sample_frame_ids(0, 10, (5,), 3), expected)
        self.assertEqual(sample_frame_ids(0, 10, (5,), 3), expected)

    def test_sampling_rejects_outside_seed(self) -> None:
        with self.assertRaises(ValueError):
            sample_frame_ids(0, 10, (11,), 3)


class QaModelProtocolTests(unittest.TestCase):
    def test_protocol_parses_slot_and_preserves_answer(self) -> None:
        parsed = parse_answer_protocol("1\t Màu xanh ", 2)
        self.assertEqual((parsed.slot, parsed.answer), (1, "Màu xanh"))

    def test_protocol_rejects_extra_text(self) -> None:
        with self.assertRaises(ValueError):
            parse_answer_protocol("0: blue", 1)

    def test_protocol_enforces_competition_length_limit(self) -> None:
        with self.assertRaises(ValueError):
            parse_answer_protocol("0\t" + "x" * 101, 1)


class TrakeTests(unittest.TestCase):
    def test_parser_preserves_event_order(self) -> None:
        query = parse_trake_query("người vào cửa hàng sau đó người rời đi")
        self.assertEqual(tuple(event.text for event in query.events), ("người vào cửa hàng", "người rời đi"))

    def test_parser_reads_numbered_event_lines(self) -> None:
        query = parse_trake_query("E1: chuẩn bị nguyên liệu. E2: nấu thức ăn. E3: bày món.")
        self.assertEqual(
            tuple(event.text for event in query.events),
            ("chuẩn bị nguyên liệu.", "nấu thức ăn.", "bày món."),
        )


if __name__ == "__main__":
    unittest.main()

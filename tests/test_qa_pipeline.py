from __future__ import annotations

import inspect
import math
import unittest

from aic_retrieval.qa import (
    ANSWER_TYPES,
    AnswerHypothesis,
    AutomaticQaPipeline,
    EvidenceWindow,
    FrameEvidence,
    QaError,
    QaPipelineStatus,
    QaQuery,
    normalize_answer,
    parse_qa_query,
    route_answer_type,
)


class QaQueryRoutingTests(unittest.TestCase):
    def test_explicit_vietnamese_marker_preserves_raw_prompt(self) -> None:
        raw = "  Tìm cảnh người đàn ông không đội mũ. Câu hỏi: Người đó cầm gì?  "
        query = parse_qa_query(raw)
        self.assertEqual(query.raw_text, raw)
        self.assertEqual(
            query.event_description,
            "Tìm cảnh người đàn ông không đội mũ.",
        )
        self.assertEqual(query.question, "Người đó cầm gì?")
        self.assertEqual(query.answer_type, "object")
        self.assertFalse(query.use_ocr)
        self.assertFalse(query.use_asr)
        self.assertIn("không", query.event_description)

    def test_explicit_english_marker_is_case_insensitive(self) -> None:
        query = parse_qa_query(
            "Find the woman near the car. QUESTION: What color is the car?"
        )
        self.assertEqual(query.event_description, "Find the woman near the car.")
        self.assertEqual(query.question, "What color is the car?")
        self.assertEqual(query.answer_type, "color")

    def test_last_explicit_marker_wins_deterministically(self) -> None:
        raw = "Question: appears on a sign. Event continues. Question: What is written?"
        query = parse_qa_query(raw)
        self.assertEqual(
            query.event_description,
            "Question: appears on a sign. Event continues.",
        )
        self.assertEqual(query.question, "What is written?")
        self.assertEqual(query.answer_type, "visible_text")
        self.assertTrue(query.use_ocr)

    def test_final_interrogative_clause_splits_without_marker(self) -> None:
        query = parse_qa_query(
            "Một người bước vào cửa hàng. Người đó đang làm gì?"
        )
        self.assertEqual(query.event_description, "Một người bước vào cửa hàng.")
        self.assertEqual(query.question, "Người đó đang làm gì?")
        self.assertEqual(query.answer_type, "action")

    def test_final_line_question_splits_without_marker(self) -> None:
        query = parse_qa_query("A person enters a room\nWhere are they?")
        self.assertEqual(query.event_description, "A person enters a room")
        self.assertEqual(query.question, "Where are they?")
        self.assertEqual(query.answer_type, "location")

    def test_ambiguous_prompt_uses_full_query_fallback(self) -> None:
        for raw in (
            "Tìm cảnh có người đứng cạnh xe",
            "Ai đang đứng cạnh xe?",
            "Find a person, perhaps outside; who is near the vehicle?",
        ):
            with self.subTest(raw=raw):
                query = parse_qa_query(raw)
                self.assertEqual(query.raw_text, raw)
                self.assertEqual(query.event_description, raw)
                self.assertEqual(query.question, raw)

    def test_marker_without_question_falls_back_to_full_prompt(self) -> None:
        raw = "Tìm một người đang chạy. Câu hỏi:"
        query = parse_qa_query(raw)
        self.assertEqual(query.event_description, raw)
        self.assertEqual(query.question, raw)

    def test_marker_word_inside_another_word_does_not_split(self) -> None:
        raw = "A questionnaire appears on screen. What color is it?"
        query = parse_qa_query(raw)
        self.assertEqual(
            query.event_description,
            "A questionnaire appears on screen.",
        )
        self.assertEqual(query.question, "What color is it?")

    def test_unicode_and_unaccented_explicit_markers_split(self) -> None:
        cases = (
            (
                "Tìm cảnh có biển. Câu hỏi：Biển ghi gì?",
                "Tìm cảnh có biển.",
                "Biển ghi gì?",
            ),
            (
                "Find the sign. Question：What does the sign say?",
                "Find the sign.",
                "What does the sign say?",
            ),
            (
                "Tim canh co xe. Cau hoi: Xe mau gi?",
                "Tim canh co xe.",
                "Xe mau gi?",
            ),
        )
        for raw, event, question in cases:
            with self.subTest(raw=raw):
                parsed = parse_qa_query(raw)
                self.assertEqual(parsed.event_description, event)
                self.assertEqual(parsed.question, question)

    def test_rejects_empty_non_string_and_forbidden_controls(self) -> None:
        for value in ("", " \n\t ", None, 42, "valid\x00invalid", "bad\x1btext"):
            with self.subTest(value=value):
                with self.assertRaises(QaError):
                    parse_qa_query(value)  # type: ignore[arg-type]

    def test_multiline_and_standard_whitespace_controls_are_allowed(self) -> None:
        raw = "Tìm cảnh người đi bộ.\nCâu hỏi:\tCó bao nhiêu người?"
        query = parse_qa_query(raw)
        self.assertEqual(query.raw_text, raw)
        self.assertEqual(query.answer_type, "count")

    def test_bounded_bilingual_answer_type_router(self) -> None:
        cases = {
            "Có bao nhiêu người trong phòng?": ("count", False, False),
            "How many cars are visible?": ("count", False, False),
            "Biển hiệu ghi gì?": ("visible_text", True, False),
            "What does the sign say?": ("visible_text", True, False),
            "Biển số xe là số nào?": ("number", True, False),
            "Tỷ số là bao nhiêu?": ("number", True, False),
            "What is the jersey number?": ("number", True, False),
            "What is the score?": ("number", True, False),
            "Người đàn ông nói gì?": ("speech", False, True),
            "What did the woman say?": ("speech", False, True),
            "Chiếc ô màu gì?": ("color", False, False),
            "What color is the umbrella?": ("color", False, False),
            "Tên của cửa hàng là gì?": ("name", True, False),
            "Tên người đó là gì?": ("name", True, False),
            "What is the store name?": ("name", True, False),
            "Ai đang cầm chiếc ô?": ("person", False, False),
            "Who is holding the umbrella?": ("person", False, False),
            "Sự kiện diễn ra ở đâu?": ("location", False, False),
            "Where does this happen?": ("location", False, False),
            "Người đó đang làm gì?": ("action", False, False),
            "Chuyện gì đang xảy ra?": ("action", False, False),
            "What is the person doing?": ("action", False, False),
            "Đó là loại xe gì?": ("object", False, False),
            "Người đó đang cầm gì?": ("object", False, False),
            "What is the person holding?": ("object", False, False),
            "What object is on the table?": ("object", False, False),
            "Điều gì đáng chú ý?": ("unknown", False, False),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                query = parse_qa_query(question)
                self.assertEqual(
                    (query.answer_type, query.use_ocr, query.use_asr),
                    expected,
                )
                self.assertIn(query.answer_type, ANSWER_TYPES)

    def test_public_routing_entry_points_have_no_human_controls(self) -> None:
        self.assertEqual(tuple(inspect.signature(parse_qa_query).parameters), ("raw_text",))
        self.assertEqual(tuple(inspect.signature(route_answer_type).parameters), ("question",))
        self.assertEqual(tuple(inspect.signature(normalize_answer).parameters), ("answer",))
        self.assertEqual(
            tuple(inspect.signature(AutomaticQaPipeline.answer_query).parameters),
            ("self", "raw_text"),
        )
        forbidden = {
            "answer",
            "frame",
            "frame_id",
            "selected_frame",
            "rank",
            "rank_override",
            "correction_callback",
        }
        for entry_point in (
            parse_qa_query,
            route_answer_type,
            AutomaticQaPipeline.answer_query,
        ):
            self.assertTrue(
                forbidden.isdisjoint(inspect.signature(entry_point).parameters)
            )

    def test_same_input_produces_equal_contract(self) -> None:
        raw = "Tìm cảnh một chiếc xe dừng lại. Câu hỏi: Xe màu gì?"
        self.assertEqual(parse_qa_query(raw), parse_qa_query(raw))


class QaContractTests(unittest.TestCase):
    def test_qa_query_rejects_invalid_router_contract(self) -> None:
        with self.assertRaisesRegex(QaError, "answer_type"):
            QaQuery("raw", "event", "question", "free_form", False, False)
        with self.assertRaisesRegex(QaError, "use_ocr"):
            QaQuery("raw", "event", "question", "visible_text", False, False)
        with self.assertRaisesRegex(QaError, "use_asr"):
            QaQuery("raw", "event", "question", "speech", False, False)

    def test_evidence_window_validates_bounds_order_scores_and_types(self) -> None:
        window = EvidenceWindow(
            video_id="video-1",
            start_frame_id=10,
            end_frame_id=30,
            seed_frame_ids=(15, 25),
            sampled_frame_ids=(10, 15, 20, 25, 30),
            retrieval_score=0.8,
            raw_ranks=(1, 3),
        )
        self.assertEqual(window.to_dict()["sampled_frame_ids"], (10, 15, 20, 25, 30))
        invalid = (
            ({"end_frame_id": 9}, "end_frame_id"),
            ({"seed_frame_ids": ()}, "non-empty"),
            ({"seed_frame_ids": (25, 15)}, "sorted"),
            ({"sampled_frame_ids": (10, 10)}, "duplicates"),
            ({"sampled_frame_ids": (5, 10)}, "inside"),
            ({"retrieval_score": math.nan}, "finite"),
            ({"retrieval_score": 1.1}, r"\[-1.0, 1.0\]"),
            ({"raw_ranks": (0,)}, "positive"),
            ({"raw_ranks": [1]}, "tuple"),
        )
        values = {
            "video_id": "video-1",
            "start_frame_id": 10,
            "end_frame_id": 30,
            "seed_frame_ids": (15, 25),
            "sampled_frame_ids": (10, 15, 20, 25, 30),
            "retrieval_score": 0.8,
            "raw_ranks": (1, 3),
        }
        for updates, message in invalid:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(QaError, message):
                    EvidenceWindow(**{**values, **updates})  # type: ignore[arg-type]

    def test_frame_evidence_never_serializes_image(self) -> None:
        image = object()
        evidence = FrameEvidence(slot=2, frame_id=42, image=image)
        self.assertIs(evidence.image, image)
        self.assertEqual(evidence.to_dict(), {"slot": 2, "frame_id": 42})
        self.assertNotIn("image", evidence.to_dict())
        for kwargs in (
            {"slot": True, "frame_id": 1, "image": image},
            {"slot": 0, "frame_id": -1, "image": image},
            {"slot": 0, "frame_id": 1, "image": None},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(QaError):
                    FrameEvidence(**kwargs)

    def test_answer_hypothesis_preserves_raw_answer_and_rejects_non_finite(self) -> None:
        hypothesis = AnswerHypothesis(
            video_id="video-1",
            frame_id=42,
            raw_answer="  Màu Xanh  ",
            normalized_answer="màu xanh",
            retrieval_score=0.8,
            support_score=1.0,
            agreement_score=0.5,
            joint_score=0.75,
        )
        self.assertEqual(hypothesis.raw_answer, "  Màu Xanh  ")
        self.assertEqual(hypothesis.to_dict()["raw_answer"], "  Màu Xanh  ")
        self.assertEqual(normalize_answer("  MÀU   XANH  "), "màu xanh")
        with self.assertRaisesRegex(QaError, "disagrees"):
            AnswerHypothesis(
                video_id="video-1",
                frame_id=42,
                raw_answer="Màu Xanh",
                normalized_answer="red",
                retrieval_score=0.8,
                support_score=1.0,
                agreement_score=0.5,
                joint_score=0.75,
            )
        values = hypothesis.to_dict()
        for field in (
            "retrieval_score",
            "support_score",
            "agreement_score",
            "joint_score",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(QaError, "finite"):
                    AnswerHypothesis(**{**values, field: math.inf})
        for field, value in (
            ("retrieval_score", -1.1),
            ("support_score", -0.1),
            ("support_score", 1.1),
            ("agreement_score", 1.1),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(QaError, "must be in"):
                    AnswerHypothesis(**{**values, field: value})

    def test_pipeline_status_validates_counts_and_elapsed_time(self) -> None:
        status = QaPipelineStatus(
            query_parsed=True,
            retrieved_windows=5,
            answered_windows=3,
            valid_hypotheses=2,
            response_count=2,
            vlm_circuit_open=False,
            elapsed_ms=12.5,
        )
        self.assertEqual(status.to_dict()["response_count"], 2)
        invalid = (
            ({"answered_windows": 6}, "cannot exceed"),
            ({"response_count": 101}, "cannot exceed"),
            ({"retrieved_windows": -1}, "non-negative"),
            ({"elapsed_ms": -0.1}, "non-negative"),
            ({"elapsed_ms": math.nan}, "finite"),
            ({"query_parsed": 1}, "boolean"),
        )
        values = status.to_dict()
        for updates, message in invalid:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(QaError, message):
                    QaPipelineStatus(**{**values, **updates})


if __name__ == "__main__":
    unittest.main()

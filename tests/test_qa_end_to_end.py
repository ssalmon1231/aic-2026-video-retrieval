from __future__ import annotations

import unittest

from aic_retrieval.qa import EvidenceWindow, FrameEvidence, QaPipeline
from aic_retrieval.qa_evidence import build_evidence_windows
from aic_retrieval.qa_models import FailClosedAnswerEngine
from aic_retrieval.index import SearchHit
from aic_retrieval.contracts import VideoRecord


class _Engine:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = []

    def answer(self, question, event_description, frames):
        self.calls.append((question, event_description, tuple(frames)))
        return self.output


class QaPipelineEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = EvidenceWindow(
            video_id="v1",
            start_frame_id=10,
            end_frame_id=30,
            seed_frame_ids=(20,),
            sampled_frame_ids=(10, 20, 30),
            retrieval_score=0.8,
            raw_ranks=(1,),
        )

    def test_valid_engine_maps_slot_to_canonical_frame(self) -> None:
        engine = _Engine("1\tblue")
        pipeline = QaPipeline(lambda query: (self.window,), engine)
        responses, status = pipeline.answer_query("A car is visible. Question: What color is it?")
        self.assertEqual(responses[0].video_id, "v1")
        self.assertEqual(responses[0].frame_id, 20)
        self.assertEqual(responses[0].answer, "blue")
        self.assertEqual(status.response_count, 1)
        self.assertEqual(status.answered_windows, 1)
        self.assertEqual(engine.calls[0][2][1].frame_id, 20)

    def test_decoder_evidence_controls_frame_mapping(self) -> None:
        engine = _Engine("0\ttext")
        evidence = lambda window: tuple(
            FrameEvidence(slot, frame_id, object())
            for slot, frame_id in enumerate(window.sampled_frame_ids)
        )
        pipeline = QaPipeline(lambda query: (self.window,), engine, evidence)
        responses, _ = pipeline.answer_query("Find sign. Question: What is written?")
        self.assertEqual((responses[0].frame_id, responses[0].answer), (10, "text"))

    def test_decoder_evidence_must_preserve_sampled_frame_mapping(self) -> None:
        evidence = lambda window: (FrameEvidence(0, 12, object()),)
        pipeline = QaPipeline(lambda query: (self.window,), _Engine("0\ttext"), evidence)
        responses, status = pipeline.answer_query("Find sign. Question: What is written?")
        self.assertEqual(responses, ())
        self.assertEqual(status.answered_windows, 0)

    def test_malformed_engine_output_fails_closed(self) -> None:
        pipeline = QaPipeline(lambda query: (self.window,), _Engine("answer only"))
        responses, status = pipeline.answer_query("Find sign. Question: What is written?")
        self.assertEqual(responses, ())
        self.assertEqual(status.answered_windows, 0)
        self.assertEqual(status.valid_hypotheses, 0)

    def test_unavailable_engine_fails_closed(self) -> None:
        pipeline = QaPipeline(lambda query: (self.window,), FailClosedAnswerEngine())
        responses, status = pipeline.answer_query("Find sign. Question: What is written?")
        self.assertEqual(responses, ())
        self.assertEqual(status.retrieved_windows, 1)

    def test_retrieval_failure_reports_parse_success(self) -> None:
        def fail(query):
            raise RuntimeError("retrieval unavailable")

        pipeline = QaPipeline(fail, _Engine("0\tanswer"))
        _, status = pipeline.answer_query("Find sign. Question: What is written?")
        self.assertTrue(status.query_parsed)


class EvidenceWindowBuilderTests(unittest.TestCase):
    def test_builder_clamps_and_keeps_exact_provenance(self) -> None:
        video = VideoRecord("v1", "v1.mp4", 2.5, 11, 4.4, "b1")
        hits = (
            SearchHit(0.9, 4, "v1", "k4", 10, "k4.jpg"),
            SearchHit(0.8, 2, "v1", "k2", 2, "k2.jpg"),
        )
        windows = build_evidence_windows(hits, (video,))
        self.assertTrue(windows)
        self.assertLessEqual(windows[0].end_frame_id, 10)
        self.assertEqual(windows[0].video_id, "v1")


if __name__ == "__main__":
    unittest.main()

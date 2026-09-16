from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from aic_retrieval.contracts import VideoRecord
from aic_retrieval.qa import EvidenceWindow, FrameEvidence, QaPipeline
from aic_retrieval.qa_evidence import decode_window
from aic_retrieval.qa_models import FailClosedAnswerEngine


class _Engine:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = []

    def answer(self, question, event_description, frames):
        self.calls.append((question, event_description, tuple(frames)))
        return self.output


class QaDecoderIntegrationTests(unittest.TestCase):
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

    def test_decode_window_preserves_sampled_frame_ids(self) -> None:
        calls: list[tuple[str, int]] = []

        def decoder(path: str, frame_id: int) -> object:
            calls.append((path, frame_id))
            return {"frame_id": frame_id}

        evidence = decode_window("Videos/v1.mp4", self.window, decoder)
        self.assertEqual(tuple(item.frame_id for item in evidence), (10, 20, 30))
        self.assertEqual(calls, [("Videos/v1.mp4", 10), ("Videos/v1.mp4", 20), ("Videos/v1.mp4", 30)])

    def test_pipeline_passes_decoded_pixels_and_maps_slot(self) -> None:
        engine = _Engine("2\tblue")
        evidence = lambda window: decode_window(
            "Videos/v1.mp4", window, lambda path, frame_id: (path, frame_id)
        )
        pipeline = QaPipeline(lambda query: (self.window,), engine, evidence)
        responses, status = pipeline.answer_query(
            "A car is visible. Question: What color is it?"
        )
        self.assertEqual((responses[0].frame_id, responses[0].answer), (30, "blue"))
        self.assertEqual(engine.calls[0][2][2].image, ("Videos/v1.mp4", 30))
        self.assertEqual(status.answered_windows, 1)

    def test_decoder_failure_fails_closed(self) -> None:
        def fail(window):
            raise ValueError("decoder unavailable")

        pipeline = QaPipeline(lambda query: (self.window,), _Engine("0\tanswer"), fail)
        responses, status = pipeline.answer_query(
            "Find sign. Question: What is written?"
        )
        self.assertEqual(responses, ())
        self.assertEqual(status.retrieved_windows, 1)
        self.assertEqual(status.answered_windows, 0)

    def test_runner_decoder_uses_dataset_root_and_manifest_video(self) -> None:
        video = VideoRecord("v1", "Videos/v1.mp4", 25.0, 100, 4.0, "batch")
        with mock.patch(
            "aic_retrieval.video.decode_frame", return_value="pixel"
        ) as decoder:
            from aic_retrieval.video import decode_manifest_frame

            self.assertEqual(decode_manifest_frame(Path("/private/dataset"), video, 30), "pixel")
        self.assertEqual(decoder.call_args.args[0], Path("/private/dataset") / "Videos/v1.mp4")


if __name__ == "__main__":
    unittest.main()

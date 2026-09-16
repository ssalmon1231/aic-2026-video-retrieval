from __future__ import annotations

import unittest

from aic_retrieval.evaluation import score_trake, TrakeGroundTruth, FrameInterval
from aic_retrieval.index import SearchHit
from aic_retrieval.trake import AlignmentConfig, align_event_hits


def hit(score: float, row: int, video: str, frame: int) -> SearchHit:
    return SearchHit(score, row, video, f"k{row}", frame, f"k{row}.jpg")


class TrakeAlignmentTests(unittest.TestCase):
    def test_finds_ordered_same_video_path(self) -> None:
        result = align_event_hits((
            (hit(.9, 1, "v1", 10), hit(.8, 2, "v2", 2)),
            (hit(.9, 3, "v1", 20), hit(.95, 4, "v2", 1)),
            (hit(.9, 5, "v1", 30), hit(.8, 6, "v2", 3)),
        ))
        self.assertEqual((result.video_id, result.frame_ids), ("v1", (10, 20, 30)))

    def test_rejects_reverse_order(self) -> None:
        result = align_event_hits((
            (hit(.9, 1, "v1", 20),),
            (hit(.9, 2, "v1", 10),),
        ))
        self.assertIsNone(result)

    def test_rejects_missing_video_coverage(self) -> None:
        result = align_event_hits((
            (hit(.9, 1, "v1", 10),),
            (hit(.9, 2, "v2", 20),),
        ))
        self.assertIsNone(result)

    def test_rejects_non_search_hit_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "SearchHit"):
            align_event_hits(((hit(.9, 1, "v1", 10), "bad"),))

    def test_tie_break_is_deterministic(self) -> None:
        result = align_event_hits((
            (hit(.5, 2, "v2", 10), hit(.5, 1, "v1", 10)),
            (hit(.5, 4, "v2", 20), hit(.5, 3, "v1", 20)),
        ), config=AlignmentConfig(gap_penalty=0))
        self.assertEqual(result.video_id, "v2")

    def test_evaluator_gives_partial_event_credit(self) -> None:
        result = align_event_hits((
            (hit(.9, 1, "v1", 10),),
            (hit(.9, 2, "v1", 20),),
            (hit(.9, 3, "v1", 30),),
            (hit(.9, 4, "v1", 40),),
        ))
        ground_truth = TrakeGroundTruth("v1", (
            FrameInterval(10, 10), FrameInterval(20, 20),
            FrameInterval(31, 31), FrameInterval(40, 40),
        ))
        self.assertEqual(score_trake(ground_truth, result), .75)


if __name__ == "__main__":
    unittest.main()

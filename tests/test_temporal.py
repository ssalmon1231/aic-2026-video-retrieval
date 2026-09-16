from __future__ import annotations

import unittest

from aic_retrieval.index import SearchHit
from aic_retrieval.temporal import (
    TemporalAlignmentError,
    TemporalConfig,
    align_ordered_events,
    apply_temporal_boost,
)


def hit(row: int, video: str, frame: int, score: float = 1.0) -> SearchHit:
    return SearchHit(score, row, video, f"{row:03d}", frame, f"{row}.jpg")


class TemporalAlignmentTests(unittest.TestCase):
    def test_full_ordered_chain_beats_partial_and_selects_strongest_event(self) -> None:
        first = (
            hit(0, "full", 10, 0.9),
            hit(1, "partial", 20, 0.8),
        )
        second = (
            hit(2, "full", 40, 0.95),
            hit(3, "reversed", 80, 0.9),
        )
        chains = align_ordered_events(
            (first, second),
            TemporalConfig(max_frame_gap=100, gap_penalty=0.0),
        )
        self.assertEqual(chains[0].video_id, "full")
        self.assertTrue(chains[0].full_coverage)
        self.assertEqual(chains[0].coverage, 2)
        self.assertEqual(chains[0].representative.row, 0)
        partial = next(chain for chain in chains if chain.video_id == "partial")
        self.assertFalse(partial.full_coverage)

    def test_reversed_or_excessive_gap_never_gets_full_coverage(self) -> None:
        first = (
            hit(0, "reversed", 100),
            hit(1, "far", 10),
        )
        second = (
            hit(2, "reversed", 50),
            hit(3, "far", 1000),
        )
        chains = align_ordered_events(
            (first, second),
            TemporalConfig(max_frame_gap=100),
        )
        self.assertFalse(any(chain.full_coverage for chain in chains))
        self.assertTrue(all(chain.coverage == 1 for chain in chains))

    def test_chain_cannot_skip_middle_event_and_still_count_full(self) -> None:
        chains = align_ordered_events(
            (
                (hit(0, "video-a", 10),),
                (),
                (hit(1, "video-a", 20),),
            ),
            TemporalConfig(max_frame_gap=100, gap_penalty=0.0),
        )
        chain = next(value for value in chains if value.video_id == "video-a")
        self.assertFalse(chain.full_coverage)
        self.assertEqual(chain.coverage, 1)

    def test_exact_text_mode_selects_final_event(self) -> None:
        chains = align_ordered_events(
            (
                (hit(0, "video-a", 10),),
                (hit(1, "video-a", 20),),
            ),
            TemporalConfig(max_frame_gap=100, gap_penalty=0.0),
            prefer_final_event=True,
        )
        self.assertEqual(chains[0].representative.row, 1)

    def test_equal_chain_ties_are_deterministic(self) -> None:
        chains = align_ordered_events(
            (
                (hit(0, "video-a", 10), hit(1, "video-b", 10)),
                (hit(2, "video-a", 20), hit(3, "video-b", 20)),
            ),
            TemporalConfig(max_frame_gap=100, gap_penalty=0.0),
        )
        self.assertEqual(tuple(chain.video_id for chain in chains), ("video-a", "video-b"))

    def test_boost_only_changes_representative_row(self) -> None:
        event_lists = (
            (hit(0, "video-a", 10),),
            (hit(1, "video-a", 20),),
        )
        config = TemporalConfig(
            max_frame_gap=100,
            gap_penalty=0.0,
            full_coverage_bonus=1.0,
            video_boost_weight=2.0,
        )
        chain = align_ordered_events(event_lists, config)[0]
        fused = (
            hit(2, "video-b", 5, 0.8),
            hit(0, "video-a", 10, 0.1),
            hit(1, "video-a", 20, 0.09),
        )
        boosted = apply_temporal_boost(fused, (chain,), config)
        self.assertEqual(boosted[0].row, chain.representative.row)
        unchanged = next(value for value in boosted if value.row == 1)
        self.assertEqual(unchanged.score, 0.09)

    def test_rejects_invalid_config_and_duplicate_rows(self) -> None:
        with self.assertRaisesRegex(TemporalAlignmentError, "max_frame_gap"):
            TemporalConfig(max_frame_gap=0)
        with self.assertRaisesRegex(TemporalAlignmentError, "invalid rows"):
            align_ordered_events(((hit(0, "v", 1), hit(0, "v", 2)),))
        with self.assertRaisesRegex(TemporalAlignmentError, "at most four"):
            align_ordered_events(((hit(0, "v", 1),),) * 5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aic_retrieval.contracts import KeyframeRecord
from aic_retrieval.hybrid_retrieval import (
    HybridRetrievalConfig,
    HybridRetrievalError,
    HybridTextRetriever,
    RankedList,
    load_config,
    reciprocal_rank_fusion,
)
from aic_retrieval.index import ExactIndex, IndexMetadata
from aic_retrieval.ocr import OcrArtifactMetadata, OcrIndex, OcrProvenance, OcrRecord
from aic_retrieval.query_planning import OrderedEvent, QueryPlan, VisualQuery
from aic_retrieval.retrieval import RetrievalConfig, retrieve_kis


class FakeEncoder:
    def __init__(self, vectors: dict[str, np.ndarray], dimension: int = 4) -> None:
        self.vectors = vectors
        self.expected_dimension = dimension
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts):
        values = tuple(texts)
        self.calls.append(values)
        return SimpleNamespace(
            vectors=np.stack([self.vectors[text] for text in values]).astype(np.float32),
            provenance=None,
        )


class Planner:
    def __init__(self, plan: QueryPlan | Exception) -> None:
        self.value = plan
        self.calls: list[str] = []

    def plan(self, query: str) -> QueryPlan:
        self.calls.append(query)
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_index() -> ExactIndex:
    vectors = np.eye(4, dtype=np.float32)
    records = tuple(
        KeyframeRecord(
            video_id=f"video-{row}",
            keyframe_id="001",
            keyframe_path=f"{row}.jpg",
            keyframe_ordinal=0,
            original_frame_id=10 + row,
            clip_row=0,
        )
        for row in range(4)
    )
    return ExactIndex(
        vectors,
        records,
        IndexMetadata(
            1,
            "numpy-flat-ip",
            "model",
            "preprocessing",
            4,
            4,
            "float32",
            True,
            "a" * 64,
        ),
    )


def test_ocr(index: ExactIndex, rows: tuple[tuple[int, str], ...]) -> OcrIndex:
    records = tuple(
        OcrRecord.create(
            row,
            index.keyframes[row].video_id,
            index.keyframes[row].original_frame_id,
            text,
            0.9,
            (0.0, 0.0, 1.0, 1.0),
        )
        for row, text in rows
    )
    return OcrIndex(
        OcrArtifactMetadata(
            1,
            index.metadata.manifest_sha256,
            len(index.keyframes),
            len(records),
            "a" * 64,
            OcrProvenance("engine", "version", "det", "rec", "revision"),
        ),
        records,
    )


def enabled_config(**changes: object) -> HybridRetrievalConfig:
    values: dict[str, object] = {
        "enabled": True,
        "raw_candidate_depth": 2,
        "auxiliary_candidate_depth": 2,
    }
    values.update(changes)
    return HybridRetrievalConfig(**values)  # type: ignore[arg-type]


class HybridRetrievalTests(unittest.TestCase):
    def test_rrf_unions_lists_boosts_overlap_and_breaks_ties_by_row(self) -> None:
        index = test_index()
        first = index.search(np.array([1.0, 0.9, 0.0, 0.0]), limit=2)
        second = index.search(np.array([0.0, 1.0, 0.9, 0.0]), limit=2)
        fused = reciprocal_rank_fusion(
            (RankedList(first, 1.0), RankedList(second, 1.0)),
            rrf_k=60,
        )
        self.assertEqual(tuple(hit.row for hit in fused), (1, 0, 2))
        self.assertGreater(fused[0].score, fused[1].score)

        tie = reciprocal_rank_fusion(
            (
                RankedList((first[0],), 1.0),
                RankedList((second[1],), 1.0),
            ),
            rrf_k=60,
        )
        self.assertEqual(tuple(hit.row for hit in tie), (0, 2))

    def test_auxiliary_list_recovers_target_missing_from_raw_shortlist(self) -> None:
        index = test_index()
        raw_query = "query Việt dài"
        plan = QueryPlan(
            (
                VisualQuery(raw_query, "raw"),
                VisualQuery("white truck driving through water at night", "holistic"),
            ),
            (),
            (),
        )
        multilingual = FakeEncoder(
            {raw_query: np.array([1.0, 0.8, 0.0, 0.0], dtype=np.float32)}
        )
        english = FakeEncoder(
            {
                "white truck driving through water at night": np.array(
                    [0.0, 0.0, 1.0, 0.9], dtype=np.float32
                )
            }
        )
        retriever = HybridTextRetriever(
            index,
            RetrievalConfig(candidate_depth=2, result_limit=2),
            enabled_config(raw_weight=1.0, holistic_weight=2.0),
            multilingual,  # type: ignore[arg-type]
            english,  # type: ignore[arg-type]
            Planner(plan),
        )
        result = retriever.retrieve(raw_query)
        rows = tuple(
            next(
                row
                for row, record in enumerate(index.keyframes)
                if record.video_id == candidate.video_id
            )
            for candidate in result.retrieval.raw_candidates
        )
        self.assertIn(2, rows)
        self.assertEqual(rows[0], 2)
        self.assertTrue(result.status.applied)
        self.assertFalse(result.status.fallback)
        self.assertEqual(result.status.semantic_lists, 2)

    def test_ocr_exact_text_boosts_bounded_semantic_candidate(self) -> None:
        index = test_index()
        query = "xe cạnh biển số 79H-6072"
        plan = QueryPlan(
            (VisualQuery(query, "raw"), VisualQuery("white vehicle", "holistic")),
            (),
            ("79H-6072",),
        )
        retriever = HybridTextRetriever(
            index,
            RetrievalConfig(candidate_depth=4, result_limit=4),
            enabled_config(
                raw_candidate_depth=4,
                auxiliary_candidate_depth=4,
                raw_weight=1.0,
                holistic_weight=1.0,
                ocr_weight=4.0,
            ),
            FakeEncoder(
                {query: np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float32)}
            ),  # type: ignore[arg-type]
            FakeEncoder(
                {"white vehicle": np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float32)}
            ),  # type: ignore[arg-type]
            Planner(plan),
            ocr_index=test_ocr(index, ((2, "79H6072"),)),
        )
        result = retriever.retrieve(query)
        self.assertEqual(result.retrieval.raw_candidates[0].video_id, "video-2")
        self.assertTrue(result.status.ocr_available)
        self.assertTrue(result.status.ocr_applied)
        self.assertEqual(result.status.ocr_matches, 1)
        content = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertNotIn("79H-6072", content)
        self.assertNotIn("79H6072", content)

    def test_exact_ocr_recovers_row_missing_from_semantic_union(self) -> None:
        index = test_index()
        query = "xe cạnh biển số 79H-6072"
        plan = QueryPlan((VisualQuery(query, "raw"),), (), ("79H-6072",))
        retriever = HybridTextRetriever(
            index,
            RetrievalConfig(candidate_depth=2, result_limit=2),
            enabled_config(
                raw_candidate_depth=2,
                auxiliary_candidate_depth=2,
                raw_weight=1.0,
                ocr_weight=4.0,
            ),
            FakeEncoder(
                {query: np.array([1.0, 0.9, 0.0, 0.0], dtype=np.float32)}
            ),  # type: ignore[arg-type]
            FakeEncoder({}),  # type: ignore[arg-type]
            Planner(plan),
            ocr_index=test_ocr(index, ((2, "79H6072"),)),
        )
        result = retriever.retrieve(query)
        rows = tuple(
            next(
                row
                for row, record in enumerate(index.keyframes)
                if record.video_id == candidate.video_id
            )
            for candidate in result.retrieval.raw_candidates
        )
        self.assertNotIn(2, tuple(hit.row for hit in index.search(
            np.array([1.0, 0.9, 0.0, 0.0], dtype=np.float32), limit=2
        )))
        self.assertEqual(rows[0], 2)
        self.assertTrue(result.status.ocr_applied)

    def test_ordered_exact_text_uses_ocr_as_temporal_event(self) -> None:
        records = (
            KeyframeRecord("video-target", "001", "0.jpg", 0, 10, 0),
            KeyframeRecord("video-target", "002", "1.jpg", 1, 20, 1),
            KeyframeRecord("video-other", "001", "2.jpg", 0, 30, 0),
            KeyframeRecord("video-other", "002", "3.jpg", 1, 40, 1),
        )
        index = ExactIndex(
            np.eye(4, dtype=np.float32),
            records,
            IndexMetadata(
                1,
                "numpy-flat-ip",
                "model",
                "preprocessing",
                4,
                4,
                "float32",
                True,
                "a" * 64,
            ),
        )
        query = "hai người mang đồ sau đó Đường Tiền Lân 11"
        plan = QueryPlan(
            (VisualQuery(query, "raw"),),
            (
                OrderedEvent("two people carrying an object"),
                OrderedEvent("street sign", "Đường Tiền Lân 11"),
            ),
            ("Đường Tiền Lân 11",),
        )
        retriever = HybridTextRetriever(
            index,
            RetrievalConfig(candidate_depth=4, result_limit=4, temporal_window=0),
            enabled_config(
                raw_candidate_depth=4,
                auxiliary_candidate_depth=4,
                raw_weight=1.0,
                ordered_event_weight=1.0,
                ocr_weight=1.0,
                temporal_gap_penalty=0.0,
                temporal_full_coverage_bonus=2.0,
                temporal_video_boost_weight=2.0,
            ),
            FakeEncoder(
                {query: np.array([0.1, 0.1, 1.0, 0.9], dtype=np.float32)}
            ),  # type: ignore[arg-type]
            FakeEncoder(
                {
                    "two people carrying an object": np.array(
                        [1.0, 0.0, 0.9, 0.0], dtype=np.float32
                    ),
                    "street sign": np.array(
                        [0.0, 0.9, 0.0, 1.0], dtype=np.float32
                    ),
                }
            ),  # type: ignore[arg-type]
            Planner(plan),
            ocr_index=test_ocr(index, ((1, "Đường Tiền Lân 11"),)),
        )
        result = retriever.retrieve(query)
        self.assertEqual(result.retrieval.raw_candidates[0].video_id, "video-target")
        self.assertTrue(result.status.ocr_applied)

    def test_missing_ocr_artifact_is_neutral(self) -> None:
        index = test_index()
        query = "private visible text"
        plan = QueryPlan((VisualQuery(query, "raw"),), (), ("private sign",))
        retriever = HybridTextRetriever(
            index,
            RetrievalConfig(candidate_depth=2, result_limit=2),
            enabled_config(),
            FakeEncoder(
                {query: np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32)}
            ),  # type: ignore[arg-type]
            FakeEncoder({}),  # type: ignore[arg-type]
            Planner(plan),
        )
        result = retriever.retrieve(query)
        self.assertFalse(result.status.ocr_available)
        self.assertFalse(result.status.ocr_applied)
        self.assertFalse(result.status.fallback)

    def test_planner_failure_keeps_raw_semantic_path(self) -> None:
        index = test_index()
        query = "private query"
        vector = np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32)
        config = RetrievalConfig(candidate_depth=2, result_limit=2)
        retriever = HybridTextRetriever(
            index,
            config,
            enabled_config(),
            FakeEncoder({query: vector}),  # type: ignore[arg-type]
            FakeEncoder({}),  # type: ignore[arg-type]
            Planner(RuntimeError("private generated output")),
        )
        result = retriever.retrieve(query)
        self.assertEqual(result.retrieval, retrieve_kis(index, vector, config))
        self.assertTrue(result.status.planner_fallback)
        self.assertFalse(result.status.fallback)
        self.assertEqual(result.status.semantic_lists, 1)

    def test_auxiliary_encoder_failure_returns_exact_baseline(self) -> None:
        index = test_index()
        query = "private query"
        vector = np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32)
        plan = QueryPlan(
            (VisualQuery(query, "raw"), VisualQuery("english", "holistic")),
            (),
            (),
        )

        class BrokenEncoder(FakeEncoder):
            def encode(self, texts):
                raise RuntimeError("private encoder failure")

        config = RetrievalConfig(candidate_depth=2, result_limit=2)
        retriever = HybridTextRetriever(
            index,
            config,
            enabled_config(),
            FakeEncoder({query: vector}),  # type: ignore[arg-type]
            BrokenEncoder({}),  # type: ignore[arg-type]
            Planner(plan),
        )
        result = retriever.retrieve(query)
        self.assertEqual(result.retrieval, retrieve_kis(index, vector, config))
        self.assertFalse(result.status.applied)
        self.assertFalse(result.status.fallback)
        content = json.dumps(result.to_dict())
        self.assertNotIn(query, content)
        self.assertNotIn("english", content)
        self.assertNotIn("private encoder failure", content)

    def test_failed_auxiliary_list_does_not_discard_successful_list(self) -> None:
        index = test_index()
        query = "private query"
        vector = np.array([1.0, 0.9, 0.0, 0.0], dtype=np.float32)
        plan = QueryPlan(
            (
                VisualQuery(query, "raw"),
                VisualQuery("working english", "holistic"),
                VisualQuery("broken english", "clause"),
            ),
            (),
            (),
        )

        class PartialEncoder(FakeEncoder):
            def encode(self, texts):
                values = tuple(texts)
                if values == ("broken english",):
                    raise RuntimeError("private encoder failure")
                return super().encode(values)

        retriever = HybridTextRetriever(
            index,
            RetrievalConfig(candidate_depth=2, result_limit=2),
            enabled_config(raw_weight=1.0, holistic_weight=3.0),
            FakeEncoder({query: vector}),  # type: ignore[arg-type]
            PartialEncoder(
                {
                    "working english": np.array(
                        [0.0, 0.0, 1.0, 0.9], dtype=np.float32
                    )
                }
            ),  # type: ignore[arg-type]
            Planner(plan),
        )
        result = retriever.retrieve(query)
        self.assertFalse(hasattr(retriever, "last_plan"))
        self.assertEqual(result.retrieval.raw_candidates[0].video_id, "video-2")
        self.assertTrue(result.status.applied)
        self.assertFalse(result.status.fallback)
        self.assertEqual(result.status.semantic_lists, 2)

    def test_disabled_mode_matches_exact_baseline_without_planner(self) -> None:
        index = test_index()
        query = "raw"
        vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        planner = Planner(AssertionError("planner must not run"))
        config = RetrievalConfig(candidate_depth=2, result_limit=2)
        retriever = HybridTextRetriever(
            index,
            config,
            HybridRetrievalConfig(
                enabled=False,
                raw_candidate_depth=500,
                auxiliary_candidate_depth=300,
            ),
            FakeEncoder({query: vector}),  # type: ignore[arg-type]
            FakeEncoder({}),  # type: ignore[arg-type]
            planner,
        )
        result = retriever.retrieve(query)
        self.assertEqual(result.retrieval, retrieve_kis(index, vector, config))
        self.assertFalse(result.status.applied)
        self.assertEqual(planner.calls, [])

    def test_config_and_dimensions_are_strict(self) -> None:
        for changes, message in (
            ({"rrf_k": 0}, "rrf_k"),
            ({"raw_weight": 0.0}, "weights"),
            ({"planner_budget_seconds": float("nan")}, "planner_budget"),
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(HybridRetrievalError, message):
                    enabled_config(**changes)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hybrid.json"
            path.write_text('{"enabled": true, "unknown": 1}', encoding="utf-8")
            with self.assertRaisesRegex(HybridRetrievalError, "unknown"):
                load_config(path)

        with self.assertRaisesRegex(HybridRetrievalError, "dimension"):
            HybridTextRetriever(
                test_index(),
                RetrievalConfig(candidate_depth=2, result_limit=2),
                enabled_config(),
                FakeEncoder({}, dimension=3),  # type: ignore[arg-type]
                FakeEncoder({}),  # type: ignore[arg-type]
                Planner(QueryPlan((VisualQuery("raw", "raw"),), (), ())),
            )


if __name__ == "__main__":
    unittest.main()

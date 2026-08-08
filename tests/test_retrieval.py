from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aic_retrieval.contracts import KeyframeRecord
from aic_retrieval.index import ExactIndex, IndexMetadata, save_index
from aic_retrieval.query import (
    QueryBatchEncoding,
    QueryEncoderProvenance,
    QueryEncoding,
)
from aic_retrieval.reranking import RerankStatus
from aic_retrieval.retrieval import (
    RetrievalConfig,
    RetrievalError,
    load_config,
    retrieve_kis,
)


ROOT = Path(__file__).parents[1]


def test_index(rows: int = 4) -> ExactIndex:
    vectors = np.zeros((rows, rows), dtype=np.float32)
    np.fill_diagonal(vectors, 1.0)
    keyframes = tuple(
        KeyframeRecord(
            video_id="video-a" if row < 3 else "video-b",
            keyframe_id=f"{row + 1:03d}",
            keyframe_path=f"Keyframes/{row + 1:03d}.jpg",
            keyframe_ordinal=row if row < 3 else 0,
            original_frame_id=(10, 15, 100, 12)[row],
            clip_row=row if row < 3 else 0,
        )
        for row in range(rows)
    )
    metadata = IndexMetadata(
        version=1,
        backend="numpy-flat-ip",
        model_id="test-model",
        preprocessing="test-preprocessing",
        dimension=rows,
        rows=rows,
        dtype="float32",
        normalized=True,
        manifest_sha256="abc",
    )
    return ExactIndex(vectors, keyframes, metadata)


class RetrievalTests(unittest.TestCase):
    def test_raw_ranking_preserves_provenance_and_original_frame(self) -> None:
        index = test_index()
        result = retrieve_kis(
            index,
            np.array([0.9, 0.8, 0.1, 0.7], dtype=np.float32),
            RetrievalConfig(candidate_depth=4, result_limit=3),
        )
        self.assertEqual(
            tuple(candidate.raw_rank for candidate in result.raw_candidates),
            (1, 2, 3, 4),
        )
        self.assertEqual(
            tuple((response.video_id, response.frame_id) for response in result.responses),
            (("video-a", 10), ("video-a", 15), ("video-b", 12)),
        )
        self.assertEqual(result.final_candidates[0].keyframe_id, "001")

    def test_temporal_dedup_and_video_cap_preserve_raw_pool(self) -> None:
        index = test_index()
        result = retrieve_kis(
            index,
            np.array([0.9, 0.8, 0.7, 0.75], dtype=np.float32),
            RetrievalConfig(
                candidate_depth=4,
                result_limit=2,
                temporal_window=10,
                max_results_per_video=1,
            ),
        )
        self.assertEqual(len(result.raw_candidates), 4)
        self.assertEqual(
            tuple(candidate.keyframe_id for candidate in result.final_candidates),
            ("001", "004"),
        )

    def test_disabled_ranking_preserves_raw_order(self) -> None:
        index = test_index()
        result = retrieve_kis(
            index,
            np.ones(4, dtype=np.float32),
            RetrievalConfig(candidate_depth=4, result_limit=4),
        )
        self.assertEqual(result.final_candidates, result.raw_candidates)

    def test_reranker_preserves_original_exact_raw_ranks(self) -> None:
        index = test_index()
        query = np.array([0.9, 0.8, 0.1, 0.7], dtype=np.float32)

        def reverse(
            index: ExactIndex,
            hits: tuple[object, ...],
        ) -> tuple[object, ...]:
            return tuple(reversed(hits))

        result = retrieve_kis(
            index,
            query,
            RetrievalConfig(candidate_depth=4, result_limit=4),
            rerank=reverse,  # type: ignore[arg-type]
        )
        self.assertEqual(
            tuple(candidate.raw_rank for candidate in result.raw_candidates),
            (4, 3, 2, 1),
        )

    def test_invalid_or_failed_reranker_preserves_exact_baseline(self) -> None:
        index = test_index()
        query = np.array([0.9, 0.8, 0.1, 0.7], dtype=np.float32)
        config = RetrievalConfig(candidate_depth=4, result_limit=3)
        baseline = retrieve_kis(index, query, config)

        def failed(index: ExactIndex, hits: tuple[object, ...]) -> tuple[object, ...]:
            raise RuntimeError("reranker failed")

        def invalid(index: ExactIndex, hits: tuple[object, ...]) -> tuple[object, ...]:
            return hits[:-1]

        self.assertEqual(
            retrieve_kis(index, query, config, rerank=failed),  # type: ignore[arg-type]
            baseline,
        )
        self.assertEqual(
            retrieve_kis(index, query, config, rerank=invalid),  # type: ignore[arg-type]
            baseline,
        )

    def test_exact_response_duplicates_are_removed_after_raw_ranking(self) -> None:
        index = test_index()
        duplicate = KeyframeRecord(
            video_id="video-a",
            keyframe_id="duplicate",
            keyframe_path="Keyframes/duplicate.jpg",
            keyframe_ordinal=3,
            original_frame_id=10,
            clip_row=3,
        )
        index = ExactIndex(
            index.vectors,
            index.keyframes[:3] + (duplicate,),
            index.metadata,
        )
        result = retrieve_kis(
            index,
            np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            RetrievalConfig(candidate_depth=4, result_limit=3),
        )
        self.assertEqual(len(result.raw_candidates), 4)
        self.assertEqual(
            tuple((response.video_id, response.frame_id) for response in result.responses),
            (("video-a", 10), ("video-a", 15), ("video-a", 100)),
        )

    def test_rejects_invalid_query_and_config(self) -> None:
        index = test_index()
        queries = (
            (np.ones(3, dtype=np.float32), "shape"),
            (np.array([np.nan, 0.0, 0.0, 0.0]), "NaN or infinity"),
            (np.zeros(4, dtype=np.float32), "norm"),
        )
        for query, message in queries:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RetrievalError, message):
                    retrieve_kis(index, query)
        with self.assertRaisesRegex(RetrievalError, "at least result_limit"):
            RetrievalConfig(candidate_depth=1, result_limit=2)
        with self.assertRaisesRegex(RetrievalError, r"\[1, 100\]"):
            RetrievalConfig(result_limit=101)
        with self.assertRaisesRegex(RetrievalError, "frame_window"):
            RetrievalConfig(temporal_window=-1)
        with self.assertRaisesRegex(RetrievalError, "maximum per video"):
            RetrievalConfig(max_results_per_video=0)

    def test_result_limit_enforces_top_100_contract(self) -> None:
        rows = 101
        vectors = np.eye(rows, dtype=np.float32)
        keyframes = tuple(
            KeyframeRecord(
                video_id=f"video-{row}",
                keyframe_id="001",
                keyframe_path=f"Keyframes/{row}.jpg",
                keyframe_ordinal=0,
                original_frame_id=row,
                clip_row=0,
            )
            for row in range(rows)
        )
        index = ExactIndex(
            vectors,
            keyframes,
            IndexMetadata(
                1,
                "numpy-flat-ip",
                "test-model",
                "test-preprocessing",
                rows,
                rows,
                "float32",
                True,
                "abc",
            ),
        )
        result = retrieve_kis(
            index,
            np.ones(rows, dtype=np.float32),
            RetrievalConfig(candidate_depth=101, result_limit=100),
        )
        self.assertEqual(len(result.responses), 100)
        self.assertEqual(len(set(result.responses)), 100)

    def test_config_loader_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text('{"unexpected": true}', encoding="utf-8")
            with self.assertRaisesRegex(RetrievalError, "unknown"):
                load_config(path)

    def test_cli_writes_parseable_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.npz"
            vector_path = root / "query.npy"
            config_path = root / "config.json"
            output_path = root / "result.json"
            save_index(test_index(), index_path)
            np.save(vector_path, np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32))
            config_path.write_text(
                json.dumps(
                    {
                        "candidate_depth": 4,
                        "result_limit": 2,
                        "temporal_window": 0,
                        "max_results_per_video": None,
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/search.py"),
                    "--index",
                    str(index_path),
                    "--query-vector",
                    str(vector_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["responses"]), 2)
            self.assertEqual(payload["responses"][0], {"video_id": "video-a", "frame_id": 10})
            self.assertEqual(payload["index"]["manifest_sha256"], "abc")
            self.assertIsNone(payload["encoder"])
            self.assertEqual(payload["encoding_elapsed_ms"], 0.0)
            self.assertGreaterEqual(payload["retrieval_elapsed_ms"], 0.0)
            self.assertEqual(
                payload["elapsed_ms"],
                payload["retrieval_elapsed_ms"],
            )

    def test_text_cli_uses_encoder_without_leaking_query(self) -> None:
        from scripts import search

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.npz"
            encoder_path = root / "encoder.json"
            config_path = root / "retrieval.json"
            output_path = root / "result.json"
            save_index(test_index(), index_path)
            encoder_path.write_text("{}", encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "candidate_depth": 4,
                        "result_limit": 2,
                        "temporal_window": 0,
                        "max_results_per_video": None,
                    }
                ),
                encoding="utf-8",
            )
            provenance = QueryEncoderProvenance(
                model_id="openai/clip-vit-base-patch32",
                revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                tokenizer_class="CLIPTokenizer",
                tokenizer_use_fast=False,
                normalization="Unicode NFC",
                output_dtype="float32",
                normalized=True,
                image_features="empirically matched",
            )
            encoded = QueryEncoding(
                np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32),
                provenance,
            )
            private_query = "private query must not be serialized"
            with mock.patch.object(
                search,
                "load_encoder_config",
                return_value=object(),
            ), mock.patch.object(
                search,
                "encode_text",
                return_value=encoded,
            ) as encoder:
                self.assertEqual(
                    search.main(
                        [
                            "--index",
                            str(index_path),
                            "--query-text",
                            private_query,
                            "--encoder-config",
                            str(encoder_path),
                            "--config",
                            str(config_path),
                            "--output",
                            str(output_path),
                        ]
                    ),
                    0,
                )

            content = output_path.read_text(encoding="utf-8")
            payload = json.loads(content)
            encoder.assert_called_once()
            self.assertEqual(encoder.call_args.args[0], private_query)
            self.assertNotIn(private_query, content)
            self.assertEqual(payload["encoder"]["revision"], provenance.revision)
            self.assertGreaterEqual(payload["encoding_elapsed_ms"], 0.0)
            self.assertGreaterEqual(payload["retrieval_elapsed_ms"], 0.0)

    def test_text_cli_serializes_multilingual_provenance_without_query(self) -> None:
        from scripts import search

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.npz"
            encoder_path = root / "encoder.json"
            config_path = root / "retrieval.json"
            output_path = root / "result.json"
            save_index(test_index(), index_path)
            encoder_path.write_text("{}", encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "candidate_depth": 4,
                        "result_limit": 2,
                        "temporal_window": 0,
                        "max_results_per_video": None,
                    }
                ),
                encoding="utf-8",
            )
            provenance = QueryEncoderProvenance(
                model_id="sentence-transformers/clip-ViT-B-32-multilingual-v1",
                revision="58edf8cada9e398793dca955574a48cbb7f18be2",
                tokenizer_class=None,
                tokenizer_use_fast=None,
                normalization="Unicode NFC",
                output_dtype="float32",
                normalized=True,
                image_features="model-card aligned; benchmark pending",
                backend="sentence-transformers",
                runtime_class="SentenceTransformer",
            )
            encoded = QueryEncoding(
                np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32),
                provenance,
            )
            private_query = "một người đàn ông đang lái xe trên đường"
            with mock.patch.object(
                search,
                "load_encoder_config",
                return_value=object(),
            ), mock.patch.object(
                search,
                "encode_text",
                return_value=encoded,
            ):
                self.assertEqual(
                    search.main(
                        [
                            "--index",
                            str(index_path),
                            "--query-text",
                            private_query,
                            "--encoder-config",
                            str(encoder_path),
                            "--config",
                            str(config_path),
                            "--output",
                            str(output_path),
                        ]
                    ),
                    0,
                )

            content = output_path.read_text(encoding="utf-8")
            payload = json.loads(content)
            self.assertNotIn(private_query, content)
            self.assertEqual(payload["encoder"]["backend"], "sentence-transformers")
            self.assertEqual(payload["encoder"]["runtime_class"], "SentenceTransformer")
            self.assertIsNone(payload["encoder"]["tokenizer_use_fast"])

    def test_text_cli_reranker_status_is_private_safe(self) -> None:
        from scripts import search

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.npz"
            encoder_path = root / "encoder.json"
            reranker_path = root / "reranker.json"
            config_path = root / "retrieval.json"
            output_path = root / "result.json"
            save_index(test_index(), index_path)
            encoder_path.write_text("{}", encoding="utf-8")
            reranker_path.write_text("{}", encoding="utf-8")
            config_path.write_text(
                json.dumps({"candidate_depth": 4, "result_limit": 2}),
                encoding="utf-8",
            )
            provenance = QueryEncoderProvenance(
                model_id="sentence-transformers/clip-ViT-B-32-multilingual-v1",
                revision="58edf8cada9e398793dca955574a48cbb7f18be2",
                tokenizer_class=None,
                tokenizer_use_fast=None,
                normalization="Unicode NFC",
                output_dtype="float32",
                normalized=True,
                image_features="model-card aligned; benchmark pending",
                backend="sentence-transformers",
                runtime_class="SentenceTransformer",
            )
            private_query = "private motorcycle query"
            generated_plan = "private generated bicycle confusion"

            class Runtime:
                def __init__(self, config, *, expected_dimension: int) -> None:
                    self.config = config
                    self.expected_dimension = expected_dimension

                def encode(self, texts):
                    return QueryBatchEncoding(
                        np.array([[1.0, 0.5, 0.0, 0.0]], dtype=np.float32),
                        provenance,
                    )

            class Reranker:
                def __init__(self, *args, **kwargs) -> None:
                    self.last_status = RerankStatus(
                        applied=True,
                        fallback=False,
                        circuit_open=False,
                        planner_elapsed_ms=1.0,
                        scoring_elapsed_ms=2.0,
                    )

                def rerank(self, query, index, hits):
                    self.private_plan = generated_plan
                    return hits

            enabled = SimpleNamespace(
                enabled=True,
                planner_model_id="planner",
                planner_revision="revision",
            )
            with mock.patch.object(
                search,
                "load_encoder_config",
                return_value=SimpleNamespace(device="cuda"),
            ), mock.patch.object(
                search,
                "QueryEncoderRuntime",
                Runtime,
            ), mock.patch.object(
                search,
                "load_reranker_config",
                return_value=enabled,
            ), mock.patch.object(
                search,
                "QwenPlanner",
                return_value=object(),
            ), mock.patch.object(
                search,
                "ContrastiveReranker",
                Reranker,
            ):
                self.assertEqual(
                    search.main(
                        [
                            "--index",
                            str(index_path),
                            "--query-text",
                            private_query,
                            "--encoder-config",
                            str(encoder_path),
                            "--reranker-config",
                            str(reranker_path),
                            "--dataset-root",
                            str(root),
                            "--config",
                            str(config_path),
                            "--output",
                            str(output_path),
                        ]
                    ),
                    0,
                )

            content = output_path.read_text(encoding="utf-8")
            payload = json.loads(content)
            self.assertNotIn(private_query, content)
            self.assertNotIn(generated_plan, content)
            self.assertEqual(
                payload["reranker"],
                {
                    "applied": True,
                    "fallback": False,
                    "circuit_open": False,
                    "planner_elapsed_ms": 1.0,
                    "scoring_elapsed_ms": 2.0,
                },
            )

    def test_text_cli_planner_load_failure_returns_private_safe_baseline(self) -> None:
        from scripts import search

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.npz"
            encoder_path = root / "encoder.json"
            reranker_path = root / "reranker.json"
            config_path = root / "retrieval.json"
            output_path = root / "result.json"
            save_index(test_index(), index_path)
            encoder_path.write_text("{}", encoding="utf-8")
            reranker_path.write_text("{}", encoding="utf-8")
            config_path.write_text(
                json.dumps({"candidate_depth": 4, "result_limit": 2}),
                encoding="utf-8",
            )
            provenance = QueryEncoderProvenance(
                model_id="sentence-transformers/clip-ViT-B-32-multilingual-v1",
                revision="58edf8cada9e398793dca955574a48cbb7f18be2",
                tokenizer_class=None,
                tokenizer_use_fast=None,
                normalization="Unicode NFC",
                output_dtype="float32",
                normalized=True,
                image_features="model-card aligned; benchmark pending",
                backend="sentence-transformers",
                runtime_class="SentenceTransformer",
            )
            private_query = "private planner load failure query"

            class Runtime:
                def __init__(self, config, *, expected_dimension: int) -> None:
                    self.config = config
                    self.expected_dimension = expected_dimension

                def encode(self, texts):
                    return QueryBatchEncoding(
                        np.array([[1.0, 0.5, 0.0, 0.0]], dtype=np.float32),
                        provenance,
                    )

            enabled = SimpleNamespace(
                enabled=True,
                planner_model_id="planner",
                planner_revision="revision",
            )
            with mock.patch.object(
                search,
                "load_encoder_config",
                return_value=SimpleNamespace(device="cuda"),
            ), mock.patch.object(
                search,
                "QueryEncoderRuntime",
                Runtime,
            ), mock.patch.object(
                search,
                "load_reranker_config",
                return_value=enabled,
            ), mock.patch.object(
                search,
                "QwenPlanner",
                side_effect=search.RerankingError("private model loader failure"),
            ):
                self.assertEqual(
                    search.main(
                        [
                            "--index",
                            str(index_path),
                            "--query-text",
                            private_query,
                            "--encoder-config",
                            str(encoder_path),
                            "--reranker-config",
                            str(reranker_path),
                            "--dataset-root",
                            str(root),
                            "--config",
                            str(config_path),
                            "--output",
                            str(output_path),
                        ]
                    ),
                    0,
                )

            content = output_path.read_text(encoding="utf-8")
            payload = json.loads(content)
            baseline = retrieve_kis(
                test_index(),
                np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32),
                RetrievalConfig(candidate_depth=4, result_limit=2),
            )
            self.assertNotIn(private_query, content)
            self.assertNotIn("private model loader failure", content)
            self.assertEqual(payload["responses"], baseline.to_dict()["responses"])
            self.assertEqual(
                payload["reranker"],
                {
                    "applied": False,
                    "fallback": True,
                    "circuit_open": True,
                    "planner_elapsed_ms": 0.0,
                    "scoring_elapsed_ms": 0.0,
                },
            )

    def test_cli_rejects_invalid_query_mode_combinations(self) -> None:
        cases = (
            (["--query-text", "private query"], "--encoder-config is required"),
            (
                [
                    "--query-vector",
                    "query.npy",
                    "--encoder-config",
                    "encoder.json",
                ],
                "--encoder-config is only valid",
            ),
            (
                [
                    "--query-vector",
                    "query.npy",
                    "--query-text",
                    "private query",
                ],
                "not allowed with argument",
            ),
            (
                [
                    "--query-vector",
                    "query.npy",
                    "--reranker-config",
                    "reranker.json",
                    "--dataset-root",
                    "dataset",
                ],
                "reranker options are only valid",
            ),
            (
                [
                    "--query-text",
                    "private query",
                    "--encoder-config",
                    "encoder.json",
                    "--reranker-config",
                    "reranker.json",
                ],
                "must be supplied together",
            ),
        )
        for query_arguments, message in cases:
            with self.subTest(query_arguments=query_arguments):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/search.py"),
                        "--index",
                        "index.npz",
                        *query_arguments,
                        "--config",
                        "retrieval.json",
                    ],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(message, completed.stderr)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aic_retrieval.benchmark import (
    BenchmarkError,
    KisBenchmarkQuery,
    KisBenchmarkSet,
    QueryEncoder,
    load_query_set,
    load_query_vectors,
    run_kis_benchmark,
    validate_query_vectors,
)
from aic_retrieval.contracts import KeyframeRecord
from aic_retrieval.evaluation import FrameInterval, KisGroundTruth
from aic_retrieval.hybrid_retrieval import HybridRetrievalConfig
from aic_retrieval.index import ExactIndex, IndexMetadata, save_index
from aic_retrieval.retrieval import RetrievalConfig, retrieve_kis

ROOT = Path(__file__).parents[1]


def test_index() -> ExactIndex:
    return ExactIndex(
        np.eye(3, dtype=np.float32),
        (
            KeyframeRecord("video-a", "001", "Keyframes/a-001.jpg", 0, 10, 0),
            KeyframeRecord("video-b", "001", "Keyframes/b-001.jpg", 0, 20, 0),
            KeyframeRecord("video-c", "001", "Keyframes/c-001.jpg", 0, 30, 0),
        ),
        IndexMetadata(
            1,
            "numpy-flat-ip",
            "image-model",
            "image-preprocessing",
            3,
            3,
            "float32",
            True,
            "manifest-hash",
        ),
    )


def test_query_set() -> KisBenchmarkSet:
    return KisBenchmarkSet(
        QueryEncoder("text-model", "text-preprocessing"),
        (
            KisBenchmarkQuery(
                "query-1", KisGroundTruth("video-a", FrameInterval(10, 12))
            ),
            KisBenchmarkQuery(
                "query-2", KisGroundTruth("video-c", FrameInterval(30, 32))
            ),
        ),
    )


def write_query_set(path: Path, payload: dict[str, object] | None = None) -> None:
    if payload is None:
        payload = {
            "version": 1,
            "task": "textual-kis",
            "query_encoder": {
                "model_id": "text-model",
                "preprocessing": "text-preprocessing",
            },
            "queries": [
                {
                    "query_id": "query-1",
                    "ground_truth": {
                        "video_id": "video-a",
                        "start": 10,
                        "end": 12,
                    },
                },
                {
                    "query_id": "query-2",
                    "ground_truth": {
                        "video_id": "video-c",
                        "start": 30,
                        "end": 32,
                    },
                },
            ],
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


class KisBenchmarkTests(unittest.TestCase):
    def test_run_preserves_order_metrics_and_provenance(self) -> None:
        ticks = iter((1.0, 1.001, 2.0, 2.003))
        result = run_kis_benchmark(
            test_index(),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            test_query_set(),
            RetrievalConfig(candidate_depth=3, result_limit=3),
            name="baseline",
            code_revision="revision",
            index_size_mb=2.5,
            clock=lambda: next(ticks),
            memory_reader=lambda: 12.5,
        )
        payload = result.to_dict()
        self.assertEqual(
            [latency["query_id"] for latency in payload["query_latencies_ms"]],
            ["query-1", "query-2"],
        )
        self.assertEqual(
            [query["query_id"] for query in payload["experiment"]["queries"]],
            ["query-1", "query-2"],
        )
        self.assertEqual(payload["experiment"]["metrics"]["final_score"], 1.0)
        self.assertEqual(
            payload["experiment"]["config"]["query_encoder"]["model_id"],
            "text-model",
        )
        self.assertEqual(
            payload["experiment"]["config"]["index"]["manifest_sha256"],
            "manifest-hash",
        )
        self.assertEqual(payload["experiment"]["resources"]["peak_memory_mb"], 12.5)

    def test_unavailable_memory_serializes_null_without_ground_truth(self) -> None:
        ticks = iter((1.0, 1.001, 2.0, 2.001))
        result = run_kis_benchmark(
            test_index(),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            test_query_set(),
            RetrievalConfig(candidate_depth=3, result_limit=3),
            name="baseline",
            code_revision="revision",
            index_size_mb=2.5,
            clock=lambda: next(ticks),
            memory_reader=lambda: None,
        )
        content = result.to_json()
        self.assertIsNone(json.loads(content)["experiment"]["resources"]["peak_memory_mb"])
        self.assertNotIn("ground_truth", content)
        self.assertNotIn('"start"', content)
        self.assertNotIn('"end"', content)

    def test_in_memory_runner_receives_query_ids_only(self) -> None:
        index = test_index()
        config = RetrievalConfig(candidate_depth=3, result_limit=3)
        received: list[str] = []
        vectors = {
            "query-1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "query-2": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        private_queries = {
            "query-1": "private raw query one",
            "query-2": "private raw query two",
        }

        def run(query_id: str):
            received.append(query_id)
            self.assertIn(query_id, private_queries)
            return retrieve_kis(index, vectors[query_id], config)

        ticks = iter((1.0, 1.001, 2.0, 2.001))
        result = run_kis_benchmark(
            index,
            None,
            test_query_set(),
            config,
            name="private-runner",
            code_revision="revision",
            index_size_mb=2.5,
            query_runner=run,
            clock=lambda: next(ticks),
            memory_reader=lambda: None,
        )
        content = result.to_json()
        self.assertEqual(received, ["query-1", "query-2"])
        for private_query in private_queries.values():
            self.assertNotIn(private_query, content)

    def test_runner_and_vectors_are_mutually_exclusive(self) -> None:
        arguments = (
            test_index(),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            test_query_set(),
            RetrievalConfig(candidate_depth=3, result_limit=3),
        )
        keywords = {
            "name": "invalid",
            "code_revision": "revision",
            "index_size_mb": 2.5,
        }
        with self.assertRaisesRegex(BenchmarkError, "exactly one"):
            run_kis_benchmark(*arguments, query_runner=lambda query_id: object(), **keywords)
        with self.assertRaisesRegex(BenchmarkError, "exactly one"):
            run_kis_benchmark(arguments[0], None, *arguments[2:], **keywords)
        with self.assertRaisesRegex(BenchmarkError, "RetrievalResult"):
            run_kis_benchmark(
                arguments[0],
                None,
                *arguments[2:],
                query_runner=lambda query_id: object(),
                **keywords,
            )
        with self.assertRaisesRegex(BenchmarkError, "runner_config requires"):
            run_kis_benchmark(
                *arguments,
                runner_config={"mode": "invalid"},
                **keywords,
            )

    def test_rejects_duplicate_empty_ids_and_wrong_task(self) -> None:
        duplicate_queries = (
            test_query_set().queries[0],
            replace(test_query_set().queries[1], query_id="query-1"),
        )
        with self.assertRaisesRegex(BenchmarkError, "unique"):
            KisBenchmarkSet(test_query_set().query_encoder, duplicate_queries)
        with self.assertRaisesRegex(BenchmarkError, "must not be empty"):
            KisBenchmarkQuery(" ", test_query_set().queries[0].ground_truth)
        with self.assertRaisesRegex(BenchmarkError, "textual-kis"):
            KisBenchmarkSet(
                test_query_set().query_encoder,
                test_query_set().queries,
                task="qa",  # type: ignore[arg-type]
            )

    def test_query_set_loader_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.json"
            cases = (
                (
                    {
                        "version": 1,
                        "task": "textual-kis",
                        "query_encoder": {
                            "model_id": "model",
                            "preprocessing": "prep",
                        },
                        "queries": [
                            {
                                "query_id": "q",
                                "ground_truth": {
                                    "video_id": "v",
                                    "start": 3,
                                    "end": 2,
                                },
                            }
                        ],
                    },
                    "invalid frame interval",
                ),
                (
                    {
                        "version": 1,
                        "task": "textual-kis",
                        "query_encoder": {
                            "model_id": "model",
                            "preprocessing": "prep",
                            "unexpected": True,
                        },
                        "queries": [],
                    },
                    "unknown",
                ),
                (
                    {
                        "version": 1,
                        "task": "qa",
                        "query_encoder": {
                            "model_id": "model",
                            "preprocessing": "prep",
                        },
                        "queries": [],
                    },
                    "textual-kis",
                ),
            )
            for payload, message in cases:
                with self.subTest(message=message):
                    write_query_set(path, payload)
                    with self.assertRaisesRegex(BenchmarkError, message):
                        load_query_set(path)

    def test_rejects_invalid_vector_matrix(self) -> None:
        cases = (
            (np.ones((1, 3), dtype=np.float32), "shape"),
            (np.ones((2, 2), dtype=np.float32), "shape"),
            (
                np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                "NaN or infinity",
            ),
            (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), "zero-norm"),
            (
                np.array([[1e300, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
                "NaN or infinity",
            ),
        )
        for matrix, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(BenchmarkError, message):
                    validate_query_vectors(matrix, query_count=2, dimension=3)

    def test_query_vector_loader_rejects_npz_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.npz"
            np.savez(path, vectors=np.eye(2, dtype=np.float32))
            with self.assertRaisesRegex(BenchmarkError, "single .npy array"):
                load_query_vectors(path)

    def test_private_query_text_loader_is_strict_and_process_private(self) -> None:
        from scripts.benchmark_kis import load_private_query_texts

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "private-queries.json"
            private_query = "private exact visible-text query 79H-6072"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "queries": {
                            "query-1": private_query,
                            "query-2": "private second query",
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_private_query_texts(path)
            self.assertEqual(loaded["query-1"], private_query)

            for payload in (
                {"version": 1, "queries": {}},
                {"version": 2, "queries": {"query-1": private_query}},
                {"version": 1, "queries": {"": private_query}},
                {"version": 1, "queries": {"query-1": " "}},
                {"version": 1, "queries": {}, "private_labels": []},
            ):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(BenchmarkError):
                        load_private_query_texts(path)

    def test_private_raw_text_cli_runs_disabled_hybrid_as_b0(self) -> None:
        from scripts import benchmark_kis

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.npz"
            query_set_path = root / "queries.json"
            private_path = root / "private-queries.json"
            retrieval_path = root / "retrieval.json"
            output_path = root / "benchmark.json"
            save_index(test_index(), index_path)
            write_query_set(query_set_path)
            private_queries = {
                "query-1": "private raw query one",
                "query-2": "private raw query two",
            }
            private_path.write_text(
                json.dumps({"version": 1, "queries": private_queries}),
                encoding="utf-8",
            )
            retrieval_path.write_text(
                json.dumps({"candidate_depth": 3, "result_limit": 3}),
                encoding="utf-8",
            )
            vectors = {
                private_queries["query-1"]: np.array([1.0, 0.0, 0.0]),
                private_queries["query-2"]: np.array([0.0, 0.0, 1.0]),
            }

            class Runtime:
                def __init__(self, config, *, expected_dimension: int) -> None:
                    self.expected_dimension = expected_dimension

                def encode(self, texts):
                    text = tuple(texts)[0]
                    return SimpleNamespace(
                        vectors=np.array([vectors[text]], dtype=np.float32),
                        provenance=None,
                    )

            with mock.patch.object(
                benchmark_kis,
                "load_encoder_config",
                return_value=SimpleNamespace(
                    device="cuda",
                    backend="sentence-transformers",
                    model_id="multilingual",
                    revision="5" * 40,
                ),
            ) as load_encoder, mock.patch.object(
                benchmark_kis,
                "load_hybrid_config",
                return_value=HybridRetrievalConfig(
                    enabled=False,
                    raw_candidate_depth=3,
                    auxiliary_candidate_depth=3,
                ),
            ), mock.patch.object(
                benchmark_kis,
                "QueryEncoderRuntime",
                Runtime,
            ), mock.patch.object(
                benchmark_kis,
                "HybridTextRetriever",
                side_effect=AssertionError("hybrid retriever must not load"),
            ), mock.patch.object(
                benchmark_kis,
                "QwenQueryPlanner",
                side_effect=AssertionError("planner must not load"),
            ):
                self.assertEqual(
                    benchmark_kis.main(
                        [
                            "--index",
                            str(index_path),
                            "--private-query-texts",
                            str(private_path),
                            "--encoder-config",
                            str(root / "encoder.json"),
                            "--english-encoder-config",
                            str(root / "english.json"),
                            "--hybrid-config",
                            str(root / "hybrid.json"),
                            "--query-set",
                            str(query_set_path),
                            "--retrieval-config",
                            str(retrieval_path),
                            "--name",
                            "private-b0",
                            "--code-revision",
                            "test-revision",
                            "--output",
                            str(output_path),
                        ]
                    ),
                    0,
                )
            self.assertEqual(load_encoder.call_count, 1)
            content = output_path.read_text(encoding="utf-8")
            for text in private_queries.values():
                self.assertNotIn(text, content)
            payload = json.loads(content)
            self.assertEqual(
                payload["experiment"]["metrics"]["final_score"],
                1.0,
            )
            runner = payload["experiment"]["config"]["runner"]
            self.assertEqual(runner["mode"], "raw-text-baseline")
            self.assertFalse(runner["hybrid"]["enabled"])
            self.assertNotIn("english_encoder", runner)

    def test_private_raw_text_cli_uses_ids_and_serializes_no_text(self) -> None:
        from scripts import benchmark_kis

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.npz"
            query_set_path = root / "queries.json"
            private_path = root / "private-queries.json"
            retrieval_path = root / "retrieval.json"
            output_path = root / "benchmark.json"
            save_index(test_index(), index_path)
            write_query_set(query_set_path)
            private_queries = {
                "query-1": "private raw query one",
                "query-2": "private raw query two",
            }
            private_path.write_text(
                json.dumps({"version": 1, "queries": private_queries}),
                encoding="utf-8",
            )
            retrieval_path.write_text(
                json.dumps({"candidate_depth": 3, "result_limit": 3}),
                encoding="utf-8",
            )
            vectors = {
                private_queries["query-1"]: np.array([1.0, 0.0, 0.0]),
                private_queries["query-2"]: np.array([0.0, 0.0, 1.0]),
            }
            received: list[str] = []

            class Runtime:
                def __init__(self, config, *, expected_dimension: int) -> None:
                    self.expected_dimension = expected_dimension

                def encode(self, texts):
                    text = tuple(texts)[0]
                    received.append(text)
                    return SimpleNamespace(
                        vectors=np.array([vectors[text]], dtype=np.float32),
                        provenance=None,
                    )

            class Retriever:
                def __init__(
                    self,
                    index,
                    retrieval_config,
                    hybrid_config,
                    multilingual,
                    english,
                    planner,
                    *,
                    ocr_index=None,
                ) -> None:
                    self.index = index
                    self.config = retrieval_config
                    self.runtime = multilingual

                def retrieve(self, text):
                    encoded = self.runtime.encode([text])
                    return SimpleNamespace(
                        retrieval=retrieve_kis(
                            self.index,
                            encoded.vectors[0],
                            self.config,
                        ),
                        status=SimpleNamespace(
                            applied=True,
                            fallback=False,
                            planner_fallback=False,
                            ocr_applied=False,
                        ),
                    )

            with mock.patch.object(
                benchmark_kis,
                "load_encoder_config",
                return_value=SimpleNamespace(
                    device="cuda",
                    backend="sentence-transformers",
                    model_id="multilingual",
                    revision="5" * 40,
                ),
            ), mock.patch.object(
                benchmark_kis,
                "load_hybrid_config",
                return_value=HybridRetrievalConfig(
                    enabled=True,
                    raw_candidate_depth=3,
                    auxiliary_candidate_depth=3,
                ),
            ), mock.patch.object(
                benchmark_kis,
                "QueryEncoderRuntime",
                Runtime,
            ), mock.patch.object(
                benchmark_kis,
                "QwenQueryPlanner",
                return_value=object(),
            ), mock.patch.object(
                benchmark_kis,
                "HybridTextRetriever",
                Retriever,
            ):
                self.assertEqual(
                    benchmark_kis.main(
                        [
                            "--index",
                            str(index_path),
                            "--private-query-texts",
                            str(private_path),
                            "--encoder-config",
                            str(root / "encoder.json"),
                            "--english-encoder-config",
                            str(root / "english.json"),
                            "--hybrid-config",
                            str(root / "hybrid.json"),
                            "--query-set",
                            str(query_set_path),
                            "--retrieval-config",
                            str(retrieval_path),
                            "--name",
                            "private-synthetic",
                            "--code-revision",
                            "test-revision",
                            "--output",
                            str(output_path),
                        ]
                    ),
                    0,
                )
            content = output_path.read_text(encoding="utf-8")
            self.assertEqual(received, list(private_queries.values()))
            for text in private_queries.values():
                self.assertNotIn(text, content)
            self.assertNotIn("ground_truth", content)
            payload = json.loads(content)
            self.assertEqual(
                payload["experiment"]["metrics"]["final_score"],
                1.0,
            )
            runner = payload["experiment"]["config"]["runner"]
            self.assertEqual(runner["mode"], "hybrid")
            self.assertTrue(runner["hybrid"]["enabled"])
            self.assertEqual(runner["english_encoder"]["revision"], "5" * 40)
            self.assertEqual(
                runner["outcomes"],
                {
                    "applied": 2,
                    "fallback": 0,
                    "planner_fallback": 0,
                    "raw_baseline": 0,
                    "ocr_applied": 0,
                },
            )

    def test_cli_writes_parseable_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.npz"
            vectors_path = root / "queries.npy"
            query_set_path = root / "queries.json"
            retrieval_path = root / "retrieval.json"
            output_path = root / "benchmark.json"
            save_index(test_index(), index_path)
            np.save(
                vectors_path,
                np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            )
            write_query_set(query_set_path)
            retrieval_path.write_text(
                json.dumps({"candidate_depth": 3, "result_limit": 3}),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/benchmark_kis.py"),
                    "--index",
                    str(index_path),
                    "--query-vectors",
                    str(vectors_path),
                    "--query-set",
                    str(query_set_path),
                    "--retrieval-config",
                    str(retrieval_path),
                    "--name",
                    "synthetic",
                    "--code-revision",
                    "test-revision",
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("Final Score 1.000000", completed.stdout)
            self.assertEqual(payload["experiment"]["metrics"]["query_count"], 2)
            self.assertEqual(payload["experiment"]["metrics"]["final_score"], 1.0)
            self.assertEqual(len(payload["query_latencies_ms"]), 2)
            self.assertNotIn("ground_truth", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

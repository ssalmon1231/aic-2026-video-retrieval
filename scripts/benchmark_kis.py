"""Benchmark vector-input Textual KIS retrieval on labeled queries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from aic_retrieval.benchmark import (
    BenchmarkError,
    load_query_set,
    load_query_vectors,
    run_kis_benchmark,
)
from aic_retrieval.hybrid_retrieval import (
    HybridRetrievalError,
    HybridTextRetriever,
    load_config as load_hybrid_config,
)
from aic_retrieval.index import IndexError, load_index
from aic_retrieval.ocr import OcrError, load_optional_ocr_artifact
from aic_retrieval.query import QueryEncoderRuntime, QueryEncodingError
from aic_retrieval.query import load_config as load_encoder_config
from aic_retrieval.query_planning import QwenQueryPlanner
from aic_retrieval.reranking import RerankingError
from aic_retrieval.retrieval import RetrievalError, load_config, retrieve_kis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark an AIC Textual KIS index using precomputed vectors or a "
            "private raw-text hybrid runner"
        )
    )
    parser.add_argument("--index", required=True, type=Path)
    query_input = parser.add_mutually_exclusive_group(required=True)
    query_input.add_argument("--query-vectors", type=Path)
    query_input.add_argument("--private-query-texts", type=Path)
    parser.add_argument("--encoder-config", type=Path)
    parser.add_argument("--english-encoder-config", type=Path)
    parser.add_argument("--hybrid-config", type=Path)
    parser.add_argument("--ocr-artifact", type=Path)
    parser.add_argument("--query-set", required=True, type=Path)
    parser.add_argument("--retrieval-config", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    private_mode = arguments.private_query_texts is not None
    private_options = (
        arguments.encoder_config,
        arguments.english_encoder_config,
        arguments.hybrid_config,
    )
    if private_mode and any(value is None for value in private_options):
        parser.error(
            "--encoder-config, --english-encoder-config, and --hybrid-config "
            "are required with --private-query-texts"
        )
    if not private_mode and any(value is not None for value in private_options):
        parser.error("text pipeline options require --private-query-texts")
    if arguments.ocr_artifact is not None and not private_mode:
        parser.error("--ocr-artifact requires --private-query-texts")

    try:
        index = load_index(arguments.index)
        query_set = load_query_set(arguments.query_set)
        retrieval_config = load_config(arguments.retrieval_config)
        query_vectors = None
        query_runner = None
        runner_config = None
        retriever = None
        hybrid_outcomes: list[Any] = []
        if private_mode:
            private_queries = load_private_query_texts(arguments.private_query_texts)
            expected_ids = {query.query_id for query in query_set.queries}
            if set(private_queries) != expected_ids:
                raise BenchmarkError(
                    "private query-text IDs must exactly match benchmark query IDs"
                )
            multilingual_config = load_encoder_config(arguments.encoder_config)
            hybrid_config = load_hybrid_config(arguments.hybrid_config)
            if not hybrid_config.enabled and arguments.ocr_artifact is not None:
                raise BenchmarkError(
                    "--ocr-artifact requires an enabled hybrid config"
                )
            multilingual_runtime = QueryEncoderRuntime(
                multilingual_config,
                expected_dimension=index.metadata.dimension,
            )
            if hybrid_config.enabled:
                english_config = load_encoder_config(arguments.english_encoder_config)
                retriever = HybridTextRetriever(
                    index,
                    retrieval_config,
                    hybrid_config,
                    multilingual_runtime,
                    QueryEncoderRuntime(
                        english_config,
                        expected_dimension=index.metadata.dimension,
                    ),
                    QwenQueryPlanner(device=multilingual_config.device),
                    ocr_index=load_optional_ocr_artifact(arguments.ocr_artifact, index),
                )

                def run_private_query(query_id: str):
                    outcome = retriever.retrieve(private_queries[query_id])
                    hybrid_outcomes.append(outcome.status)
                    return outcome.retrieval

            else:

                def run_private_query(query_id: str):
                    encoded = multilingual_runtime.encode([private_queries[query_id]])
                    return retrieve_kis(
                        index,
                        encoded.vectors[0],
                        retrieval_config,
                    )

            query_runner = run_private_query
            runner_config = {
                "mode": "hybrid" if hybrid_config.enabled else "raw-text-baseline",
                "multilingual_encoder": encoder_config_identity(multilingual_config),
                "hybrid": asdict(hybrid_config),
                "ocr_artifact_attached": arguments.ocr_artifact is not None,
            }
            if hybrid_config.enabled:
                runner_config["english_encoder"] = encoder_config_identity(
                    english_config
                )
        else:
            query_vectors = load_query_vectors(arguments.query_vectors)
        result = run_kis_benchmark(
            index,
            query_vectors,
            query_set,
            retrieval_config,
            name=arguments.name,
            code_revision=arguments.code_revision,
            index_size_mb=arguments.index.stat().st_size / (1024.0 * 1024.0),
            query_runner=query_runner,
            runner_config=runner_config,
        )
        if retriever is not None:
            result = with_hybrid_outcomes(result, hybrid_outcomes)
        result.save(arguments.output)
    except BenchmarkError as error:
        parser.exit(2, f"aic-benchmark-kis: {error}\n")
    except (
        HybridRetrievalError,
        IndexError,
        OcrError,
        OSError,
        QueryEncodingError,
        RerankingError,
        RetrievalError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.exit(2, f"aic-benchmark-kis: {type(error).__name__}\n")

    metrics = result.experiment.to_dict()["metrics"]
    print(
        f"benchmark completed: {metrics['query_count']} queries, "
        f"Final Score {metrics['final_score']:.6f}"
    )
    return 0


def with_hybrid_outcomes(result: Any, statuses: list[Any]) -> Any:
    expected = result.experiment.to_dict()["metrics"]["query_count"]
    if len(statuses) != expected:
        raise BenchmarkError("hybrid status count disagrees with benchmark queries")
    counts = Counter(
        "fallback"
        if status.fallback
        else "planner-fallback"
        if status.planner_fallback
        else "applied"
        if status.applied
        else "raw-baseline"
        for status in statuses
    )
    runner = dict(result.experiment.config["runner"])
    runner["outcomes"] = {
        "applied": counts["applied"],
        "fallback": counts["fallback"],
        "planner_fallback": counts["planner-fallback"],
        "raw_baseline": counts["raw-baseline"],
        "ocr_applied": sum(status.ocr_applied for status in statuses),
    }
    experiment = replace(
        result.experiment,
        config={**result.experiment.config, "runner": runner},
    )
    return replace(result, experiment=experiment)


def encoder_config_identity(config: Any) -> dict[str, Any]:
    identity = {
        "backend": getattr(config, "backend", None),
        "model_id": getattr(config, "model_id", None),
        "revision": getattr(config, "revision", None),
    }
    if not all(isinstance(value, str) and value for value in identity.values()):
        raise BenchmarkError("query encoder config identity is invalid")
    return identity


def load_private_query_texts(path: str | Path) -> dict[str, str]:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError("cannot read private query-text mapping") from error
    if not isinstance(payload, dict) or set(payload) != {"version", "queries"}:
        raise BenchmarkError("private query-text mapping has invalid fields")
    if payload["version"] != 1 or not isinstance(payload["queries"], dict):
        raise BenchmarkError("private query-text mapping schema is invalid")
    queries: dict[str, str] = {}
    for query_id, text in payload["queries"].items():
        if (
            not isinstance(query_id, str)
            or not query_id.strip()
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise BenchmarkError("private query-text mapping contains invalid values")
        queries[query_id] = text
    if not queries:
        raise BenchmarkError("private query-text mapping must not be empty")
    return queries


if __name__ == "__main__":
    raise SystemExit(main())

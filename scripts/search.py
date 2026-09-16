"""Run Textual KIS retrieval from a vector or pinned CLIP text encoder."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aic_retrieval.hybrid_retrieval import (
    HybridTextRetriever,
    load_config as load_hybrid_config,
)
from aic_retrieval.index import IndexError, load_index
from aic_retrieval.ocr import OcrError, load_optional_ocr_artifact
from aic_retrieval.query import (
    QueryEncoderRuntime,
    QueryEncodingError,
    encode_text,
    load_config as load_encoder_config,
)
from aic_retrieval.query_planning import QwenQueryPlanner
from aic_retrieval.reranking import (
    ContrastiveReranker,
    QwenPlanner,
    RerankingError,
    RerankStatus,
    load_config as load_reranker_config,
)
from aic_retrieval.retrieval import load_config, retrieve_kis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search an AIC index using a vector or pinned CLIP text encoder"
    )
    parser.add_argument("--index", required=True, type=Path)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--query-vector", type=Path)
    query.add_argument("--query-text")
    parser.add_argument("--encoder-config", type=Path)
    parser.add_argument("--hybrid-config", type=Path)
    parser.add_argument("--english-encoder-config", type=Path)
    parser.add_argument("--ocr-artifact", type=Path)
    parser.add_argument("--reranker-config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.query_text is not None and arguments.encoder_config is None:
        parser.error("--encoder-config is required with --query-text")
    if arguments.query_vector is not None and arguments.encoder_config is not None:
        parser.error("--encoder-config is only valid with --query-text")
    if arguments.query_vector is not None and (
        arguments.reranker_config is not None or arguments.dataset_root is not None
    ):
        parser.error("reranker options are only valid with --query-text")
    if arguments.query_vector is not None and any(
        value is not None
        for value in (
            arguments.hybrid_config,
            arguments.english_encoder_config,
            arguments.ocr_artifact,
        )
    ):
        parser.error("hybrid options are only valid with --query-text")
    hybrid_requested = arguments.hybrid_config is not None
    if hybrid_requested != (arguments.english_encoder_config is not None):
        parser.error(
            "--hybrid-config and --english-encoder-config must be supplied together"
        )
    if arguments.ocr_artifact is not None and not hybrid_requested:
        parser.error("--ocr-artifact requires --hybrid-config")
    if hybrid_requested and (
        arguments.reranker_config is not None or arguments.dataset_root is not None
    ):
        parser.error("hybrid and contrastive reranker modes are mutually exclusive")
    if (arguments.reranker_config is None) != (arguments.dataset_root is None):
        parser.error("--reranker-config and --dataset-root must be supplied together")

    index = load_index(arguments.index)
    startup_ms = 0.0
    encoding_ms = 0.0
    encoder = None
    reranker = None
    hybrid_result = None
    hybrid_config = None
    hybrid_startup_fallback = False
    reranker_status = RerankStatus(False, False, False, 0.0, 0.0)
    config = load_config(arguments.config)
    if arguments.query_vector is not None:
        try:
            query_vector = np.load(arguments.query_vector, allow_pickle=False)
        except (OSError, ValueError) as error:
            parser.error(f"cannot read query vector {arguments.query_vector}: {error}")
    else:
        try:
            encoder_config = load_encoder_config(arguments.encoder_config)
            if hybrid_requested:
                hybrid_config = load_hybrid_config(arguments.hybrid_config)
                if not hybrid_config.enabled:
                    started = time.perf_counter()
                    single = encode_text(
                        arguments.query_text,
                        encoder_config,
                        expected_dimension=index.metadata.dimension,
                    )
                    encoding_ms = (time.perf_counter() - started) * 1000.0
                    query_vector = single.vector
                    encoder = asdict(single.provenance)
                else:
                    started = time.perf_counter()
                    multilingual_runtime = QueryEncoderRuntime(
                        encoder_config,
                        expected_dimension=index.metadata.dimension,
                    )
                    startup_ms = (time.perf_counter() - started) * 1000.0
                    try:
                        started = time.perf_counter()
                        english_runtime = QueryEncoderRuntime(
                            load_encoder_config(arguments.english_encoder_config),
                            expected_dimension=index.metadata.dimension,
                        )
                        planner = QwenQueryPlanner(device=encoder_config.device)
                        ocr_index = load_optional_ocr_artifact(
                            arguments.ocr_artifact,
                            index,
                        )
                        startup_ms += (time.perf_counter() - started) * 1000.0
                    except (
                        IndexError,
                        OcrError,
                        OSError,
                        QueryEncodingError,
                        RerankingError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        hybrid_startup_fallback = True
                        started = time.perf_counter()
                        encoded = multilingual_runtime.encode([arguments.query_text])
                        encoding_ms = (time.perf_counter() - started) * 1000.0
                        query_vector = encoded.vectors[0]
                        encoder = asdict(encoded.provenance)
                    else:
                        hybrid = HybridTextRetriever(
                            index,
                            config,
                            hybrid_config,
                            multilingual_runtime,
                            english_runtime,
                            planner,
                            ocr_index=ocr_index,
                        )
                        hybrid_result = hybrid.retrieve(arguments.query_text)
                        if hybrid.last_raw_provenance is None:
                            raise QueryEncodingError(
                                "hybrid raw encoder did not return provenance"
                            )
                        encoder = asdict(hybrid.last_raw_provenance)
            elif arguments.reranker_config is None:
                started = time.perf_counter()
                single = encode_text(
                    arguments.query_text,
                    encoder_config,
                    expected_dimension=index.metadata.dimension,
                )
                encoding_ms = (time.perf_counter() - started) * 1000.0
                query_vector = single.vector
                encoder = asdict(single.provenance)
            else:
                started = time.perf_counter()
                encoder_runtime = QueryEncoderRuntime(
                    encoder_config,
                    expected_dimension=index.metadata.dimension,
                )
                encoded = encoder_runtime.encode([arguments.query_text])
                encoding_ms = (time.perf_counter() - started) * 1000.0
                query_vector = encoded.vectors[0]
                encoder = asdict(encoded.provenance)
                reranker_config = load_reranker_config(arguments.reranker_config)
                if reranker_config.enabled:
                    try:
                        planner = QwenPlanner(
                            reranker_config.planner_model_id,
                            reranker_config.planner_revision,
                            device=encoder_config.device,
                        )
                    except RerankingError:
                        reranker_status = RerankStatus(
                            False, True, True, 0.0, 0.0
                        )
                    else:
                        reranker = ContrastiveReranker(
                            reranker_config,
                            planner,
                            encoder_runtime,
                            arguments.dataset_root,
                        )
        except (QueryEncodingError, RerankingError, OSError, RuntimeError, TypeError, ValueError) as error:
            parser.error(str(error))

    started = time.perf_counter()
    if hybrid_result is not None:
        result = hybrid_result.retrieval
    else:
        result = retrieve_kis(
            index,
            query_vector,
            config,
            rerank=(
                (lambda idx, hits: reranker.rerank(arguments.query_text, idx, hits))
                if reranker is not None
                else None
            ),
        )
    retrieval_ms = (time.perf_counter() - started) * 1000.0
    if hybrid_result is not None:
        encoding_ms = hybrid_result.status.encoding_elapsed_ms
        retrieval_ms = (
            hybrid_result.status.planner_elapsed_ms
            + hybrid_result.status.search_elapsed_ms
        )
    payload = {
        "index": asdict(index.metadata),
        "config": asdict(config),
        "encoder": encoder,
        "reranker": asdict(
            reranker.last_status if reranker is not None else reranker_status
        ),
        "startup_elapsed_ms": startup_ms,
        "encoding_elapsed_ms": encoding_ms,
        "retrieval_elapsed_ms": retrieval_ms,
        "elapsed_ms": startup_ms + encoding_ms + retrieval_ms,
        **(
            hybrid_result.to_dict()
            if hybrid_result is not None
            else result.to_dict()
        ),
    }
    if hybrid_result is None and hybrid_config is not None:
        payload["hybrid"] = {
            "applied": False,
            "fallback": hybrid_startup_fallback,
            "planner_fallback": False,
            "semantic_lists": 1,
            "fused_candidates": len(result.raw_candidates),
            "ocr_available": False,
            "ocr_applied": False,
            "ocr_matches": 0,
            "planner_elapsed_ms": 0.0,
            "encoding_elapsed_ms": encoding_ms,
            "search_elapsed_ms": retrieval_ms,
        }
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(content, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

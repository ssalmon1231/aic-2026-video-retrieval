"""Run Textual KIS retrieval from a vector or pinned CLIP text encoder."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aic_retrieval.index import load_index
from aic_retrieval.query import (
    QueryEncoderRuntime,
    QueryEncodingError,
    encode_text,
    load_config as load_encoder_config,
)
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
    if (arguments.reranker_config is None) != (arguments.dataset_root is None):
        parser.error("--reranker-config and --dataset-root must be supplied together")

    index = load_index(arguments.index)
    encoding_ms = 0.0
    encoder = None
    reranker = None
    reranker_status = RerankStatus(False, False, False, 0.0, 0.0)
    if arguments.query_vector is not None:
        try:
            query_vector = np.load(arguments.query_vector, allow_pickle=False)
        except (OSError, ValueError) as error:
            parser.error(f"cannot read query vector {arguments.query_vector}: {error}")
    else:
        try:
            encoder_config = load_encoder_config(arguments.encoder_config)
            started = time.perf_counter()
            if arguments.reranker_config is None:
                single = encode_text(
                    arguments.query_text,
                    encoder_config,
                    expected_dimension=index.metadata.dimension,
                )
                query_vector = single.vector
                encoder = asdict(single.provenance)
            else:
                encoder_runtime = QueryEncoderRuntime(
                    encoder_config,
                    expected_dimension=index.metadata.dimension,
                )
                encoded = encoder_runtime.encode([arguments.query_text])
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
            encoding_ms = (time.perf_counter() - started) * 1000.0
        except (QueryEncodingError, RerankingError) as error:
            parser.error(str(error))

    config = load_config(arguments.config)
    started = time.perf_counter()
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
    payload = {
        "index": asdict(index.metadata),
        "config": asdict(config),
        "encoder": encoder,
        "reranker": asdict(
            reranker.last_status if reranker is not None else reranker_status
        ),
        "encoding_elapsed_ms": encoding_ms,
        "retrieval_elapsed_ms": retrieval_ms,
        "elapsed_ms": encoding_ms + retrieval_ms,
        **result.to_dict(),
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

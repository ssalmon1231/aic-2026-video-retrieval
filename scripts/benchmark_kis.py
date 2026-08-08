"""Benchmark vector-input Textual KIS retrieval on labeled queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from aic_retrieval.benchmark import (
    BenchmarkError,
    load_query_set,
    load_query_vectors,
    run_kis_benchmark,
)
from aic_retrieval.index import IndexError, load_index
from aic_retrieval.retrieval import RetrievalError, load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark an AIC Textual KIS index using labeled ground truth and "
            "precomputed embeddings from an explicitly identified compatible encoder"
        )
    )
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--query-vectors", required=True, type=Path)
    parser.add_argument("--query-set", required=True, type=Path)
    parser.add_argument("--retrieval-config", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    try:
        index = load_index(arguments.index)
        query_vectors = load_query_vectors(arguments.query_vectors)
        query_set = load_query_set(arguments.query_set)
        retrieval_config = load_config(arguments.retrieval_config)
        result = run_kis_benchmark(
            index,
            query_vectors,
            query_set,
            retrieval_config,
            name=arguments.name,
            code_revision=arguments.code_revision,
            index_size_mb=arguments.index.stat().st_size / (1024.0 * 1024.0),
        )
        result.save(arguments.output)
    except (BenchmarkError, IndexError, OSError, RetrievalError) as error:
        parser.exit(2, f"aic-benchmark-kis: {error}\n")

    metrics = result.experiment.to_dict()["metrics"]
    print(
        f"benchmark completed: {metrics['query_count']} queries, "
        f"Final Score {metrics['final_score']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

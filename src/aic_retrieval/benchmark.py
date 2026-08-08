"""Batch Textual KIS benchmark over precomputed query embeddings."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import Task
from .evaluation import FrameInterval, KisGroundTruth, QueryMetrics, evaluate_ranked
from .experiment import ExperimentRecord, ResourceMetrics
from .index import ExactIndex
from .retrieval import RetrievalConfig, RetrievalResult, retrieve_kis

BENCHMARK_VERSION = 1


class BenchmarkError(ValueError):
    """Raised when benchmark inputs or provenance cannot be trusted."""


@dataclass(frozen=True, slots=True)
class QueryEncoder:
    model_id: str
    preprocessing: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.preprocessing.strip():
            raise BenchmarkError(
                "query encoder model_id and preprocessing must not be empty"
            )


@dataclass(frozen=True, slots=True)
class KisBenchmarkQuery:
    query_id: str
    ground_truth: KisGroundTruth

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise BenchmarkError("query_id must not be empty")


@dataclass(frozen=True, slots=True)
class KisBenchmarkSet:
    query_encoder: QueryEncoder
    queries: tuple[KisBenchmarkQuery, ...]
    version: int = BENCHMARK_VERSION
    task: Task = Task.TEXTUAL_KIS

    def __post_init__(self) -> None:
        if self.version != BENCHMARK_VERSION:
            raise BenchmarkError(f"unsupported benchmark version {self.version}")
        if self.task is not Task.TEXTUAL_KIS:
            raise BenchmarkError("benchmark task must be textual-kis")
        if not self.queries:
            raise BenchmarkError("benchmark requires at least one query")
        query_ids = tuple(query.query_id for query in self.queries)
        if len(set(query_ids)) != len(query_ids):
            raise BenchmarkError("benchmark query IDs must be unique")


@dataclass(frozen=True, slots=True)
class QueryLatency:
    query_id: str
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class KisBenchmarkResult:
    experiment: ExperimentRecord
    query_latencies_ms: tuple[QueryLatency, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment.to_dict(),
            "query_latencies_ms": [
                asdict(latency) for latency in self.query_latencies_ms
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(self.to_json(), encoding="utf-8")
        temporary.replace(destination)


def load_query_set(path: str | Path) -> KisBenchmarkSet:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot read benchmark query set {source}: {error}") from error
    if not isinstance(payload, dict):
        raise BenchmarkError("benchmark query-set root must be an object")
    _require_fields(
        payload,
        required={"version", "task", "query_encoder", "queries"},
        context="benchmark query set",
    )

    version = _integer(payload, "version", "benchmark query set")
    try:
        task = Task(payload["task"])
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"unsupported benchmark task: {payload['task']!r}") from error

    encoder_payload = _object(payload["query_encoder"], "query_encoder")
    _require_fields(
        encoder_payload,
        required={"model_id", "preprocessing"},
        context="query_encoder",
    )
    encoder = QueryEncoder(
        _string(encoder_payload, "model_id", "query_encoder"),
        _string(encoder_payload, "preprocessing", "query_encoder"),
    )

    query_payloads = payload["queries"]
    if not isinstance(query_payloads, list):
        raise BenchmarkError("queries must be an array")
    queries: list[KisBenchmarkQuery] = []
    for position, value in enumerate(query_payloads, start=1):
        context = f"query {position}"
        query_payload = _object(value, context)
        _require_fields(
            query_payload,
            required={"query_id", "ground_truth"},
            context=context,
        )
        ground_truth_payload = _object(
            query_payload["ground_truth"], f"{context}.ground_truth"
        )
        _require_fields(
            ground_truth_payload,
            required={"video_id", "start", "end"},
            context=f"{context}.ground_truth",
        )
        try:
            ground_truth = KisGroundTruth(
                _string(
                    ground_truth_payload,
                    "video_id",
                    f"{context}.ground_truth",
                ),
                FrameInterval(
                    _integer(
                        ground_truth_payload,
                        "start",
                        f"{context}.ground_truth",
                    ),
                    _integer(
                        ground_truth_payload,
                        "end",
                        f"{context}.ground_truth",
                    ),
                ),
            )
        except ValueError as error:
            raise BenchmarkError(f"{context}: {error}") from error
        queries.append(
            KisBenchmarkQuery(
                _string(query_payload, "query_id", context),
                ground_truth,
            )
        )
    return KisBenchmarkSet(encoder, tuple(queries), version, task)


def load_query_vectors(path: str | Path) -> np.ndarray:
    source = Path(path)
    try:
        vectors = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read query vectors {source}: {error}") from error
    if not isinstance(vectors, np.ndarray):
        vectors.close()
        raise BenchmarkError(f"query vectors {source} must be a single .npy array")
    return vectors


def validate_query_vectors(
    vectors: np.ndarray,
    *,
    query_count: int,
    dimension: int,
) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"query vectors must be numeric: {error}") from error
    expected = (query_count, dimension)
    if matrix.shape != expected:
        raise BenchmarkError(f"query vector shape {matrix.shape} != {expected}")
    if not np.isfinite(matrix).all():
        raise BenchmarkError("query vectors contain NaN or infinity")
    norms = np.linalg.norm(matrix.astype(np.float64, copy=False), axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise BenchmarkError("query vectors contain invalid or zero-norm rows")
    return matrix


def run_kis_benchmark(
    index: ExactIndex,
    query_vectors: np.ndarray | None,
    query_set: KisBenchmarkSet,
    retrieval_config: RetrievalConfig,
    *,
    name: str,
    code_revision: str,
    index_size_mb: float,
    query_runner: Callable[[str], RetrievalResult] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    memory_reader: Callable[[], float | None] | None = None,
) -> KisBenchmarkResult:
    if not name.strip() or not code_revision.strip():
        raise BenchmarkError("benchmark name and code_revision must not be empty")
    if not math.isfinite(index_size_mb) or index_size_mb < 0:
        raise BenchmarkError("index_size_mb must be finite and non-negative")
    if (query_vectors is None) == (query_runner is None):
        raise BenchmarkError("supply exactly one of query_vectors or query_runner")
    matrix = (
        validate_query_vectors(
            query_vectors,
            query_count=len(query_set.queries),
            dimension=index.metadata.dimension,
        )
        if query_vectors is not None
        else None
    )

    metrics: list[QueryMetrics] = []
    latencies: list[QueryLatency] = []
    for position, query in enumerate(query_set.queries):
        started = clock()
        result = (
            query_runner(query.query_id)
            if query_runner is not None
            else retrieve_kis(index, matrix[position], retrieval_config)
        )
        if not isinstance(result, RetrievalResult):
            raise BenchmarkError("query_runner must return RetrievalResult")
        elapsed_ms = (clock() - started) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise BenchmarkError("benchmark clock returned invalid elapsed time")
        latencies.append(QueryLatency(query.query_id, elapsed_ms))
        metrics.append(
            evaluate_ranked(
                Task.TEXTUAL_KIS,
                query.ground_truth,
                result.responses,
                query_id=query.query_id,
            )
        )

    elapsed_values = np.asarray(
        [latency.elapsed_ms for latency in latencies], dtype=np.float64
    )
    read_memory = memory_reader or peak_process_memory_mb
    resources = ResourceMetrics(
        latency_ms_p50=float(np.percentile(elapsed_values, 50)),
        latency_ms_p95=float(np.percentile(elapsed_values, 95)),
        peak_memory_mb=read_memory(),
        index_size_mb=index_size_mb,
    )
    experiment = ExperimentRecord(
        name=name,
        manifest_sha256=index.metadata.manifest_sha256,
        code_revision=code_revision,
        config={
            "index": asdict(index.metadata),
            "query_encoder": asdict(query_set.query_encoder),
            "retrieval": asdict(retrieval_config),
        },
        queries=tuple(metrics),
        resources=resources,
        notes=(
            "Query encoder compatibility requires labeled retrieval evidence; "
            "matching vector dimensions alone is not proof."
        ),
    )
    return KisBenchmarkResult(experiment, tuple(latencies))


def peak_process_memory_mb() -> float | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss) / 1024.0


def _require_fields(
    payload: dict[str, Any], *, required: set[str], context: str
) -> None:
    missing = sorted(required - payload.keys())
    unknown = sorted(payload.keys() - required)
    if missing:
        raise BenchmarkError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise BenchmarkError(f"{context} has unknown fields: {', '.join(unknown)}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context} must be an object")
    return value


def _string(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise BenchmarkError(f"{context}.{key} must be a string")
    return value


def _integer(payload: dict[str, Any], key: str, context: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"{context}.{key} must be an integer")
    return value

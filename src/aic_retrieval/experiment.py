"""Byte-stable experiment records and cutoff comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evaluation import CUTOFFS, QueryMetrics, aggregate_queries


@dataclass(frozen=True, slots=True)
class ResourceMetrics:
    latency_ms_p50: float
    latency_ms_p95: float
    peak_memory_mb: float | None
    index_size_mb: float

    def __post_init__(self) -> None:
        values = (
            self.latency_ms_p50,
            self.latency_ms_p95,
            self.index_size_mb,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("resource metrics must be finite and non-negative")
        if self.peak_memory_mb is not None and (
            not math.isfinite(self.peak_memory_mb) or self.peak_memory_mb < 0
        ):
            raise ValueError("peak memory must be finite, non-negative, or null")


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    name: str
    manifest_sha256: str
    code_revision: str
    config: Mapping[str, Any]
    queries: tuple[QueryMetrics, ...]
    resources: ResourceMetrics
    notes: str = ""

    def __post_init__(self) -> None:
        query_ids = tuple(metric.query_id for metric in self.queries)
        if any(not query_id.strip() for query_id in query_ids):
            raise ValueError("experiment metrics require non-empty query IDs")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("experiment query IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        cutoff_means = {
            str(cutoff): _mean(_recall(metric, cutoff) for metric in self.queries)
            for cutoff in CUTOFFS
        }
        return {
            "name": self.name,
            "manifest_sha256": self.manifest_sha256,
            "code_revision": self.code_revision,
            "config": dict(self.config),
            "metrics": {
                "query_count": len(self.queries),
                "recall_at": cutoff_means,
                "final_score": aggregate_queries(self.queries),
            },
            "queries": [metric.to_dict() for metric in self.queries],
            "resources": asdict(self.resources),
            "notes": self.notes,
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


@dataclass(frozen=True, slots=True)
class PromotionComparison:
    baseline: str
    candidate: str
    recall_at_delta: tuple[tuple[int, float], ...]
    final_score_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "recall_at_delta": {
                str(cutoff): delta for cutoff, delta in self.recall_at_delta
            },
            "final_score_delta": self.final_score_delta,
        }


def compare_experiments(
    baseline: ExperimentRecord, candidate: ExperimentRecord
) -> PromotionComparison:
    if baseline.manifest_sha256 != candidate.manifest_sha256:
        raise ValueError("experiments must use the same data manifest")
    baseline_ids = tuple(metric.query_id for metric in baseline.queries)
    candidate_ids = tuple(metric.query_id for metric in candidate.queries)
    if baseline_ids != candidate_ids:
        raise ValueError("experiments must evaluate the same ordered query IDs")
    deltas = tuple(
        (
            cutoff,
            _mean(_recall(metric, cutoff) for metric in candidate.queries)
            - _mean(_recall(metric, cutoff) for metric in baseline.queries),
        )
        for cutoff in CUTOFFS
    )
    return PromotionComparison(
        baseline=baseline.name,
        candidate=candidate.name,
        recall_at_delta=deltas,
        final_score_delta=(
            aggregate_queries(candidate.queries) - aggregate_queries(baseline.queries)
        ),
    )


def _recall(metric: QueryMetrics, cutoff: int) -> float:
    return dict(metric.recall_at)[cutoff]


def _mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else 0.0

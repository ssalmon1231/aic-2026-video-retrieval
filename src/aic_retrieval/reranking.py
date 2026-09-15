"""Bounded object-aware contrastive reranking for exact KIS shortlists."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .data import ObjectDetection, load_object_detections
from .index import ExactIndex, SearchHit
from .query import QueryEncoderRuntime

PLANNER_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PLANNER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PLAN_KEYS = {"positive", "negatives", "required_objects", "excluded_objects"}
MAX_NEGATIVES = 3
MAX_OBJECT_VOCABULARY = 64
MAX_NEW_TOKENS = 96


class RerankingError(ValueError):
    """Raised when reranking configuration or output is invalid."""


@dataclass(frozen=True, slots=True)
class ContrastivePlan:
    positive: str
    negatives: tuple[str, ...]
    required_objects: tuple[str, ...]
    excluded_objects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    enabled: bool = False
    planner_model_id: str = PLANNER_MODEL_ID
    planner_revision: str = PLANNER_REVISION
    positive_weight: float = 0.0
    negative_weight: float = 0.0
    object_weight: float = 0.0
    planner_budget_seconds: float = 1.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise RerankingError("enabled must be boolean")
        if self.enabled and (
            self.planner_model_id != PLANNER_MODEL_ID
            or self.planner_revision != PLANNER_REVISION
        ):
            raise RerankingError("enabled reranker must use the pinned planner")
        weights = (self.positive_weight, self.negative_weight, self.object_weight)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in weights
        ):
            raise RerankingError("reranking weights must be finite and non-negative")
        if self.enabled and not any(weights):
            raise RerankingError("enabled reranker requires at least one nonzero weight")
        if (
            isinstance(self.planner_budget_seconds, bool)
            or not isinstance(self.planner_budget_seconds, (int, float))
            or not math.isfinite(self.planner_budget_seconds)
            or self.planner_budget_seconds <= 0
        ):
            raise RerankingError("planner_budget_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class RerankStatus:
    applied: bool
    fallback: bool
    circuit_open: bool
    planner_elapsed_ms: float
    scoring_elapsed_ms: float


class Planner(Protocol):
    def plan(self, query: str, vocabulary: Sequence[str]) -> ContrastivePlan: ...


class QwenPlanner:
    """Warm pinned deterministic planner; generated content never leaves this object."""

    def __init__(
        self,
        model_id: str = PLANNER_MODEL_ID,
        revision: str = PLANNER_REVISION,
        *,
        device: str = "cuda",
    ) -> None:
        if model_id != PLANNER_MODEL_ID or revision != PLANNER_REVISION:
            raise RerankingError("planner must use the pinned checkpoint")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RerankingError("planner requires optional clip dependencies") from error
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=False,
            )
            model_options: dict[str, Any] = {
                "revision": revision,
                "trust_remote_code": False,
            }
            if device.startswith("cuda"):
                model_options["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
            loaded_revision = getattr(model.config, "_commit_hash", None)
            if loaded_revision is not None and loaded_revision != revision:
                raise RerankingError("loaded planner revision differs from config")
            model = model.to(device)
            model.eval()
        except RerankingError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RerankingError("cannot load planner") from error
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._last_generated_tokens = 0

    @property
    def last_generated_tokens(self) -> int:
        return self._last_generated_tokens

    def plan(self, query: str, vocabulary: Sequence[str]) -> ContrastivePlan:
        content = self.generate_json(
            _planner_prompt(query, vocabulary),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        return parse_plan(content, vocabulary)

    def generate_json(self, prompt: str, *, max_new_tokens: int) -> str:
        self._last_generated_tokens = 0
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise RerankingError("planner generation inputs are invalid")
        try:
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(rendered, return_tensors="pt").to(self._device)
            input_length = int(inputs["input_ids"].shape[1])
            with self._torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
            generated = output[0][input_length:]
            self._last_generated_tokens = int(generated.shape[0])
            return self._tokenizer.decode(generated, skip_special_tokens=True)
        except RerankingError:
            raise
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            raise RerankingError("planner generation failed") from error


class ContrastiveReranker:
    def __init__(
        self,
        config: RerankerConfig,
        planner: Planner,
        encoder: QueryEncoderRuntime,
        dataset_root: str | Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.planner = planner
        self.encoder = encoder
        self.dataset_root = Path(dataset_root)
        self.clock = clock
        self.circuit_open = False
        self.last_status = RerankStatus(False, False, False, 0.0, 0.0)

    def rerank(
        self,
        query: str,
        index: ExactIndex,
        hits: tuple[SearchHit, ...],
    ) -> tuple[SearchHit, ...]:
        if not self.config.enabled or self.circuit_open:
            self.last_status = RerankStatus(
                False, self.circuit_open, self.circuit_open, 0.0, 0.0
            )
            return hits
        try:
            detections = (
                tuple(
                    load_object_detections(
                        self.dataset_root,
                        index.keyframes[hit.row].object_path,
                    )
                    for hit in hits
                )
                if self.config.object_weight
                else ((),) * len(hits)
            )
            vocabulary = build_object_vocabulary(detections)
            started = self.clock()
            plan = self.planner.plan(query, vocabulary)
            planner_elapsed = self.clock() - started
            if planner_elapsed > self.config.planner_budget_seconds:
                self.circuit_open = True
                self.last_status = RerankStatus(
                    False, True, True, planner_elapsed * 1000.0, 0.0
                )
                return hits
            scoring_started = self.clock()
            ranked = score_shortlist(
                index,
                hits,
                detections,
                plan,
                self.encoder,
                self.config,
            )
            scoring_elapsed = self.clock() - scoring_started
            self.last_status = RerankStatus(
                True,
                False,
                False,
                planner_elapsed * 1000.0,
                scoring_elapsed * 1000.0,
            )
            return ranked
        except (OSError, RuntimeError, TypeError, ValueError):
            self.last_status = RerankStatus(False, True, self.circuit_open, 0.0, 0.0)
            return hits


def score_shortlist(
    index: ExactIndex,
    hits: tuple[SearchHit, ...],
    detections: tuple[tuple[ObjectDetection, ...], ...],
    plan: ContrastivePlan,
    encoder: QueryEncoderRuntime,
    config: RerankerConfig,
) -> tuple[SearchHit, ...]:
    if len(detections) != len(hits):
        raise RerankingError("object detection count must match shortlist")
    need_contrastive = bool(config.positive_weight or config.negative_weight)
    if need_contrastive:
        texts = (plan.positive, *plan.negatives)
        encoded = encoder.encode(texts).vectors
        rows = np.fromiter((hit.row for hit in hits), dtype=np.int64)
        similarities = index.vectors[rows] @ encoded.T
        positive = similarities[:, 0]
        negative = (
            np.max(similarities[:, 1:], axis=1)
            if plan.negatives
            else np.zeros(len(hits), dtype=np.float32)
        )
    else:
        positive = negative = np.zeros(len(hits), dtype=np.float32)
    objects = (
        np.asarray(
            [object_consistency(value, plan) for value in detections],
            dtype=np.float32,
        )
        if config.object_weight
        else np.zeros(len(hits), dtype=np.float32)
    )
    base = np.fromiter((hit.score for hit in hits), dtype=np.float32)
    scores = (
        base
        + config.positive_weight * positive
        - config.negative_weight * negative
        + config.object_weight * objects
    )
    if scores.shape != (len(hits),) or not np.isfinite(scores).all():
        raise RerankingError("reranker produced invalid scores")
    order = sorted(range(len(hits)), key=lambda row: (-float(scores[row]), row))
    return tuple(
        SearchHit(
            score=float(scores[row]),
            row=hits[row].row,
            video_id=hits[row].video_id,
            keyframe_id=hits[row].keyframe_id,
            original_frame_id=hits[row].original_frame_id,
            keyframe_path=hits[row].keyframe_path,
        )
        for row in order
    )


def object_consistency(
    detections: Sequence[ObjectDetection],
    plan: ContrastivePlan,
) -> float:
    confidence: dict[str, float] = defaultdict(float)
    for detection in detections:
        for label in detection.labels:
            confidence[label] = max(confidence[label], detection.score)
    required = (
        sum(confidence[label] for label in plan.required_objects)
        / len(plan.required_objects)
        if plan.required_objects
        else 0.0
    )
    excluded = max(
        (confidence[label] for label in plan.excluded_objects), default=0.0
    )
    return required - excluded


def build_object_vocabulary(
    detections: Sequence[Sequence[ObjectDetection]],
) -> tuple[str, ...]:
    confidence: dict[str, float] = defaultdict(float)
    for candidate in detections:
        for detection in candidate:
            for label in detection.labels:
                confidence[label] += detection.score
    return tuple(
        label
        for label, _ in sorted(confidence.items(), key=lambda item: (-item[1], item[0]))[
            :MAX_OBJECT_VOCABULARY
        ]
    )


def _planner_prompt(query: str, vocabulary: Sequence[str]) -> str:
    vocabulary_json = json.dumps(list(vocabulary), ensure_ascii=False)
    return (
        "Analyze the visual search query. Return one minified JSON object only. "
        'Valid example: {"positive":"person outdoors","negatives":[],'
        '"required_objects":[],"excluded_objects":[]}. Use exactly those four keys '
        "in that order. After each key write a colon and its JSON value. positive "
        "must be one short visual string. negatives must be a JSON array of 0 to 3 "
        "unique strings. Object arrays must contain unique labels only from the "
        "supplied vocabulary. When vocabulary is empty, both object arrays must be "
        "[]. Close every string, array, and the object. No explanation or markdown. "
        f"Vocabulary: {vocabulary_json}. Query: {query}"
    )


def parse_plan(content: str, vocabulary: Sequence[str]) -> ContrastivePlan:
    normalized = _normalize_plan_content(content)
    try:
        payload: Any = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise RerankingError("planner output must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != PLAN_KEYS:
        raise RerankingError("planner output has invalid fields")
    allowed = set(vocabulary)
    positive = _nonempty_text(payload["positive"])
    negatives = _unique_texts(payload["negatives"], MAX_NEGATIVES)
    required = _object_labels(payload["required_objects"], allowed)
    excluded = _object_labels(payload["excluded_objects"], allowed)
    if set(required) & set(excluded):
        raise RerankingError("object labels cannot be both required and excluded")
    return ContrastivePlan(positive, negatives, required, excluded)


def _normalize_plan_content(content: Any) -> str:
    if not isinstance(content, str):
        raise RerankingError("planner output must be JSON text")
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[0].strip().casefold() not in {"```", "```json"}:
        raise RerankingError("planner output must be plain JSON or one JSON fence")
    if lines[-1].strip() != "```" or any("```" in line for line in lines[1:-1]):
        raise RerankingError("planner output must be plain JSON or one JSON fence")
    body = "\n".join(lines[1:-1]).strip()
    if not body:
        raise RerankingError("planner output must contain JSON")
    return body


def _nonempty_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerankingError("planner text must be non-empty")
    return " ".join(value.split())


def _unique_texts(value: Any, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise RerankingError("planner text list exceeds its bound")
    texts = tuple(_nonempty_text(item) for item in value)
    if len(set(texts)) != len(texts):
        raise RerankingError("planner text list contains duplicates")
    return texts


def _object_labels(value: Any, allowed: set[str]) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RerankingError("planner object labels must be a list")
    if not allowed:
        return ()
    labels = tuple(_nonempty_text(item).casefold() for item in value)
    if len(set(labels)) != len(labels) or not set(labels) <= allowed:
        raise RerankingError("planner object labels are invalid")
    return labels


def load_config(path: str | Path) -> RerankerConfig:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RerankingError("cannot read reranker config") from error
    if not isinstance(payload, dict):
        raise RerankingError("reranker config root must be an object")
    allowed = {
        "enabled",
        "planner_model_id",
        "planner_revision",
        "positive_weight",
        "negative_weight",
        "object_weight",
        "planner_budget_seconds",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise RerankingError("unknown reranker config fields")
    try:
        return RerankerConfig(**payload)
    except TypeError as error:
        raise RerankingError("invalid reranker config") from error

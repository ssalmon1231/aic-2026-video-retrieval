"""Pinned text encoders for the empirically matched OpenAI CLIP image index."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OPENAI_CLIP_BACKEND = "openai-clip"
SENTENCE_TRANSFORMERS_BACKEND = "sentence-transformers"
SUPPORTED_BACKENDS = {OPENAI_CLIP_BACKEND, SENTENCE_TRANSFORMERS_BACKEND}
REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
NORMALIZATION = "Unicode NFC; whitespace collapsed; content otherwise unchanged"
IMAGE_PROVENANCE = "empirically matched; source generator/revision unknown"
MULTILINGUAL_IMAGE_PROVENANCE = (
    "model-card aligned with clip-ViT-B-32; exact AIC index compatibility "
    "pending labeled benchmark"
)


class QueryEncodingError(ValueError):
    """Raised when text encoding configuration or output cannot be trusted."""


@dataclass(frozen=True, slots=True)
class QueryEncoderConfig:
    model_id: str
    revision: str
    tokenizer_use_fast: bool | None
    device: str
    backend: str = OPENAI_CLIP_BACKEND

    def __post_init__(self) -> None:
        for name in ("model_id", "revision", "device", "backend"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise QueryEncodingError(f"{name} must be a non-empty string")
            if "unverified" in value.lower():
                raise QueryEncodingError(f"{name} must not contain 'unverified'")
        if self.backend not in SUPPORTED_BACKENDS:
            raise QueryEncodingError(
                f"backend must be one of: {', '.join(sorted(SUPPORTED_BACKENDS))}"
            )
        if REVISION_PATTERN.fullmatch(self.revision) is None:
            raise QueryEncodingError("revision must be an exact 40-character commit hash")
        if self.revision != self.revision.lower():
            raise QueryEncodingError("revision must use lowercase hexadecimal")
        if self.backend == OPENAI_CLIP_BACKEND:
            if self.tokenizer_use_fast is not False:
                raise QueryEncodingError(
                    "tokenizer_use_fast must be false for the verified baseline"
                )
        elif self.tokenizer_use_fast is not None:
            raise QueryEncodingError(
                "tokenizer_use_fast is not configurable for sentence-transformers"
            )


@dataclass(frozen=True, slots=True)
class QueryEncoderProvenance:
    model_id: str
    revision: str
    tokenizer_class: str | None
    tokenizer_use_fast: bool | None
    normalization: str
    output_dtype: str
    normalized: bool
    image_features: str
    backend: str = OPENAI_CLIP_BACKEND
    runtime_class: str = "CLIPModel"


@dataclass(frozen=True, slots=True)
class QueryEncoding:
    vector: np.ndarray
    provenance: QueryEncoderProvenance


@dataclass(frozen=True, slots=True)
class QueryBatchEncoding:
    vectors: np.ndarray
    provenance: QueryEncoderProvenance


class QueryEncoderRuntime:
    """Warm pinned text encoder shared by repeated and batched queries."""

    def __init__(self, config: QueryEncoderConfig, *, expected_dimension: int) -> None:
        _validate_expected_dimension(expected_dimension)
        self.config = config
        self.expected_dimension = expected_dimension
        if config.backend == OPENAI_CLIP_BACKEND:
            self._runtime = _load_runtime(config)
        else:
            self._runtime = _load_multilingual_runtime(config)

    def encode(self, texts: Sequence[str]) -> QueryBatchEncoding:
        normalized = _normalize_queries(texts)
        if self.config.backend == OPENAI_CLIP_BACKEND:
            return _encode_openai_clip_batch(
                normalized,
                self.config,
                self.expected_dimension,
                self._runtime,
            )
        return _encode_sentence_transformers_batch(
            normalized,
            self.config,
            self.expected_dimension,
            self._runtime,
        )


def normalize_query(text: str) -> str:
    if not isinstance(text, str):
        raise QueryEncodingError("query text must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    if not normalized:
        raise QueryEncodingError("query text must not be empty")
    return normalized


def encode_text(
    text: str,
    config: QueryEncoderConfig,
    *,
    expected_dimension: int,
) -> QueryEncoding:
    encoded = encode_texts([text], config, expected_dimension=expected_dimension)
    return QueryEncoding(encoded.vectors[0], encoded.provenance)


def encode_texts(
    texts: Sequence[str],
    config: QueryEncoderConfig,
    *,
    expected_dimension: int,
    runtime: QueryEncoderRuntime | None = None,
) -> QueryBatchEncoding:
    _validate_expected_dimension(expected_dimension)
    encoder = runtime or QueryEncoderRuntime(
        config,
        expected_dimension=expected_dimension,
    )
    if encoder.config != config or encoder.expected_dimension != expected_dimension:
        raise QueryEncodingError("query encoder runtime does not match the requested config")
    return encoder.encode(texts)


def _normalize_queries(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
        raise QueryEncodingError("query batch must be a non-empty sequence")
    return tuple(normalize_query(text) for text in texts)


def _validate_expected_dimension(expected_dimension: int) -> None:
    if (
        isinstance(expected_dimension, bool)
        or not isinstance(expected_dimension, int)
        or expected_dimension <= 0
    ):
        raise QueryEncodingError("expected_dimension must be a positive integer")


def load_config(path: str | Path) -> QueryEncoderConfig:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QueryEncodingError(f"cannot read query encoder config {source}: {error}") from error
    if not isinstance(payload, dict):
        raise QueryEncodingError("query encoder config root must be an object")

    backend = payload.get("backend", OPENAI_CLIP_BACKEND)
    common = {"model_id", "revision", "device"}
    if backend == OPENAI_CLIP_BACKEND:
        required = common | {"tokenizer_use_fast"}
        allowed = required | {"backend"}
    else:
        required = common | {"backend"}
        allowed = required
    missing = sorted(required - set(payload))
    if missing:
        raise QueryEncodingError(
            f"query encoder config missing fields: {', '.join(missing)}"
        )
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise QueryEncodingError(
            f"unknown query encoder config fields: {', '.join(unknown)}"
        )

    values = dict(payload)
    values.setdefault("backend", OPENAI_CLIP_BACKEND)
    values.setdefault("tokenizer_use_fast", None)
    try:
        return QueryEncoderConfig(**values)
    except TypeError as error:
        raise QueryEncodingError(f"invalid query encoder config: {error}") from error


def _encode_openai_clip_batch(
    normalized: tuple[str, ...],
    config: QueryEncoderConfig,
    expected_dimension: int,
    runtime: tuple[Any, Any, Any],
) -> QueryBatchEncoding:
    torch, tokenizer, model = runtime
    try:
        max_length = model.config.text_config.max_position_embeddings
        tokens = tokenizer(
            list(normalized),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise QueryEncodingError(
                "tokenizer output must contain input_ids and attention_mask"
            )
        input_ids = tokens["input_ids"].to(config.device)
        attention_mask = tokens["attention_mask"].to(config.device)
        with torch.inference_mode():
            output = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            projected = model.text_projection(output.pooler_output).float()
        vectors = projected.detach().cpu().numpy()
    except QueryEncodingError:
        raise
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise QueryEncodingError(f"text encoding failed: {error}") from error

    vectors = _normalize_matrix(vectors, len(normalized), expected_dimension)
    return QueryBatchEncoding(
        vectors=vectors,
        provenance=QueryEncoderProvenance(
            model_id=config.model_id,
            revision=config.revision,
            tokenizer_class=type(tokenizer).__name__,
            tokenizer_use_fast=config.tokenizer_use_fast,
            normalization=NORMALIZATION,
            output_dtype=str(vectors.dtype),
            normalized=True,
            image_features=IMAGE_PROVENANCE,
            backend=config.backend,
            runtime_class=type(model).__name__,
        ),
    )


def _encode_sentence_transformers_batch(
    normalized: tuple[str, ...],
    config: QueryEncoderConfig,
    expected_dimension: int,
    model: Any,
) -> QueryBatchEncoding:
    try:
        embeddings = np.asarray(
            model.encode(
                list(normalized),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise QueryEncodingError(f"text encoding failed: {error}") from error
    vectors = _normalize_matrix(embeddings, len(normalized), expected_dimension)
    return QueryBatchEncoding(
        vectors=vectors,
        provenance=QueryEncoderProvenance(
            model_id=config.model_id,
            revision=config.revision,
            tokenizer_class=None,
            tokenizer_use_fast=None,
            normalization=NORMALIZATION,
            output_dtype=str(vectors.dtype),
            normalized=True,
            image_features=MULTILINGUAL_IMAGE_PROVENANCE,
            backend=config.backend,
            runtime_class=type(model).__name__,
        ),
    )


def _load_runtime(config: QueryEncoderConfig) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoTokenizer, CLIPModel
    except ImportError as error:
        raise QueryEncodingError(
            "text encoding requires optional dependencies: pip install -e '.[clip]'"
        ) from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            use_fast=config.tokenizer_use_fast,
        )
        model = CLIPModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            use_safetensors=False,
        )
        _validate_loaded_revision(model.config, config.revision)
        model = model.to(config.device)
        model.eval()
    except QueryEncodingError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise QueryEncodingError(
            f"cannot load {config.model_id}@{config.revision}: {error}"
        ) from error
    return torch, tokenizer, model


def _load_multilingual_runtime(config: QueryEncoderConfig) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise QueryEncodingError(
            "multilingual text encoding requires optional dependencies: "
            "pip install -e '.[multilingual]'"
        ) from error

    try:
        model = SentenceTransformer(
            config.model_id,
            revision=config.revision,
            device=config.device,
            trust_remote_code=False,
        )
        _validate_loaded_revision(_sentence_transformer_config(model), config.revision)
    except QueryEncodingError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise QueryEncodingError(
            f"cannot load {config.model_id}@{config.revision}: {error}"
        ) from error
    return model


def _sentence_transformer_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is not None:
        return config
    first_module = getattr(model, "_first_module", None)
    if not callable(first_module):
        return None
    module = first_module()
    auto_model = getattr(module, "auto_model", None)
    return getattr(auto_model, "config", getattr(module, "config", None))


def _validate_loaded_revision(config: Any, expected_revision: str) -> None:
    loaded_revision = getattr(config, "_commit_hash", None)
    if loaded_revision is not None and loaded_revision != expected_revision:
        raise QueryEncodingError(
            "loaded checkpoint revision "
            f"{loaded_revision!r} != configured {expected_revision}"
        )


def _normalize_matrix(
    vectors: Any,
    expected_rows: int,
    expected_dimension: int,
) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    expected_shape = (expected_rows, expected_dimension)
    if array.shape != expected_shape:
        raise QueryEncodingError(
            f"encoded query batch shape {array.shape} != {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise QueryEncodingError("encoded query batch contains NaN or infinity")
    norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise QueryEncodingError("encoded query norm must be positive")
    normalized = np.asarray(array / norms, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise QueryEncodingError("normalized query contains NaN or infinity")
    return normalized


def _normalize_vector(vector: Any, expected_dimension: int) -> np.ndarray:
    array = np.asarray(vector)
    if array.shape != (expected_dimension,):
        raise QueryEncodingError(
            f"encoded query shape {array.shape} != ({expected_dimension},)"
        )
    return _normalize_matrix(array[None, :], 1, expected_dimension)[0]

"""Exact NumPy vector index with canonical keyframe provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import KeyframeRecord, Manifest, load_manifest

INDEX_VERSION = 1


class IndexError(ValueError):
    """Raised when vectors, lookup rows, or index metadata cannot be trusted."""


@dataclass(frozen=True, slots=True)
class IndexConfig:
    manifest_path: Path
    dataset_root: Path
    output_path: Path
    model_id: str
    preprocessing: str
    feature_glob: str = "CLIP-features/**/{video_id}.npy"


@dataclass(frozen=True, slots=True)
class SearchHit:
    score: float
    row: int
    video_id: str
    keyframe_id: str
    original_frame_id: int
    keyframe_path: str


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    version: int
    backend: str
    model_id: str
    preprocessing: str
    dimension: int
    rows: int
    dtype: str
    normalized: bool
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ExactIndex:
    vectors: np.ndarray
    keyframes: tuple[KeyframeRecord, ...]
    metadata: IndexMetadata

    def __post_init__(self) -> None:
        vectors = np.asarray(self.vectors)
        if vectors.ndim != 2:
            raise IndexError(f"vectors must be 2D, got {vectors.shape}")
        if vectors.shape != (len(self.keyframes), self.metadata.dimension):
            raise IndexError(
                f"vector shape {vectors.shape} disagrees with lookup/metadata"
            )
        if self.metadata.version != INDEX_VERSION:
            raise IndexError(f"unsupported index version {self.metadata.version}")
        if self.metadata.backend != "numpy-flat-ip":
            raise IndexError(f"unsupported index backend {self.metadata.backend}")
        if self.metadata.rows != len(self.keyframes):
            raise IndexError("metadata row count disagrees with lookup")
        if self.metadata.dtype != str(vectors.dtype):
            raise IndexError("metadata dtype disagrees with vectors")
        if not self.metadata.normalized:
            raise IndexError("metadata marks vectors as unnormalized")
        if not np.isfinite(vectors).all():
            raise IndexError("vectors contain NaN or infinity")
        if not np.allclose(
            np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-4, atol=1e-4
        ):
            raise IndexError("index vectors are not unit-normalized")
        vectors.setflags(write=False)

    def search(self, query: np.ndarray, limit: int = 100) -> tuple[SearchHit, ...]:
        if limit <= 0:
            raise IndexError("search limit must be positive")
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (self.metadata.dimension,):
            raise IndexError(
                f"query shape {vector.shape} != ({self.metadata.dimension},)"
            )
        if not np.isfinite(vector).all():
            raise IndexError("query contains NaN or infinity")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise IndexError("query norm must be positive")
        scores = self.vectors @ (vector / norm)
        count = min(limit, len(self.keyframes))
        order = np.argsort(-scores, kind="stable")[:count]
        return tuple(
            SearchHit(
                score=float(scores[row]),
                row=int(row),
                video_id=self.keyframes[row].video_id,
                keyframe_id=self.keyframes[row].keyframe_id,
                original_frame_id=self.keyframes[row].original_frame_id,
                keyframe_path=self.keyframes[row].keyframe_path,
            )
            for row in order
        )


def build_index(
    manifest: Manifest,
    dataset_root: str | Path,
    *,
    model_id: str,
    preprocessing: str,
    manifest_sha256: str,
    feature_glob: str = "CLIP-features/**/{video_id}.npy",
) -> ExactIndex:
    manifest.validate()
    root = Path(dataset_root)
    if not model_id.strip() or not preprocessing.strip() or not feature_glob.strip():
        raise IndexError("model_id, preprocessing, and feature_glob must not be empty")
    if "{video_id}" not in feature_glob:
        raise IndexError("feature_glob must contain {video_id}")
    ordered = tuple(
        sorted(
            manifest.keyframes,
            key=lambda record: (record.video_id, record.keyframe_ordinal),
        )
    )
    arrays: dict[str, np.ndarray] = {}
    vectors: list[np.ndarray] = []
    try:
        for record in ordered:
            if record.clip_row is None:
                raise IndexError(
                    f"{record.video_id}/{record.keyframe_id}: missing CLIP row"
                )
            if record.video_id not in arrays:
                feature_path = _feature_path(root, feature_glob, record.video_id)
                try:
                    array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError) as error:
                    raise IndexError(f"cannot read {feature_path}: {error}") from error
                if array.ndim != 2:
                    raise IndexError(f"{feature_path}: expected 2D feature matrix")
                arrays[record.video_id] = array
            array = arrays[record.video_id]
            if record.clip_row >= array.shape[0]:
                raise IndexError(
                    f"{record.video_id}/{record.keyframe_id}: CLIP row "
                    f"{record.clip_row} outside [0, {array.shape[0] - 1}]"
                )
            vectors.append(np.asarray(array[record.clip_row], dtype=np.float32).copy())
    finally:
        for array in arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    if not vectors:
        raise IndexError("manifest contains no indexable keyframes")
    try:
        matrix = np.stack(vectors)
    except ValueError as error:
        raise IndexError(f"feature vectors have inconsistent dimensions: {error}") from error
    matrix = _normalize_rows(matrix)
    metadata = IndexMetadata(
        version=INDEX_VERSION,
        backend="numpy-flat-ip",
        model_id=model_id,
        preprocessing=preprocessing,
        dimension=int(matrix.shape[1]),
        rows=int(matrix.shape[0]),
        dtype=str(matrix.dtype),
        normalized=True,
        manifest_sha256=manifest_sha256,
    )
    return ExactIndex(matrix, ordered, metadata)


def build_from_config(config: IndexConfig) -> ExactIndex:
    manifest = load_manifest(config.manifest_path)
    return build_index(
        manifest,
        config.dataset_root,
        model_id=config.model_id,
        preprocessing=config.preprocessing,
        manifest_sha256=sha256_file(config.manifest_path),
        feature_glob=config.feature_glob,
    )


def save_index(index: ExactIndex, destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.staging.npz")
    try:
        np.savez(
            staging,
            vectors=index.vectors,
            lookup_json=np.array(
                json.dumps(
                    [asdict(record) for record in index.keyframes],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
            metadata_json=np.array(
                json.dumps(asdict(index.metadata), ensure_ascii=False, sort_keys=True)
            ),
        )
        loaded = load_index(staging)
        verify_index(index, loaded)
        staging.replace(target)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def load_index(source: str | Path) -> ExactIndex:
    path = Path(source)
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata_payload = json.loads(str(archive["metadata_json"]))
            lookup_payload = json.loads(str(archive["lookup_json"]))
            vectors = np.asarray(archive["vectors"], dtype=np.float32).copy()
        metadata = IndexMetadata(**metadata_payload)
        keyframes = tuple(KeyframeRecord(**record) for record in lookup_payload)
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise IndexError(f"cannot load index {path}: {error}") from error
    return ExactIndex(vectors, keyframes, metadata)


def verify_index(expected: ExactIndex, actual: ExactIndex) -> None:
    if expected.metadata != actual.metadata or expected.keyframes != actual.keyframes:
        raise IndexError("saved index metadata or lookup changed after reload")
    probes = _sample_rows(len(expected.keyframes), 3)
    for row in probes:
        expected_hits = expected.search(expected.vectors[row], limit=min(10, len(expected.keyframes)))
        actual_hits = actual.search(expected.vectors[row], limit=min(10, len(actual.keyframes)))
        if expected_hits != actual_hits:
            raise IndexError(f"saved index search changed for probe row {row}")
        expected_score = float(actual.vectors[row] @ actual.vectors[row])
        if not np.isclose(actual_hits[0].score, expected_score, rtol=1e-5, atol=1e-6):
            raise IndexError(f"self-query row {row} did not return a top-scoring match")


def load_config(path: str | Path) -> IndexConfig:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IndexError(f"cannot read index config {source}: {error}") from error
    required = {"manifest_path", "dataset_root", "output_path", "model_id", "preprocessing"}
    allowed = required | {"feature_glob"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        missing = sorted(required - set(payload) if isinstance(payload, dict) else required)
        raise IndexError(f"index config missing fields: {', '.join(missing)}")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise IndexError(f"unknown index config fields: {', '.join(unknown)}")
    values = {field: payload[field] for field in required}
    feature_glob = payload.get("feature_glob", "CLIP-features/**/{video_id}.npy")
    if any(not isinstance(value, str) or not value.strip() for value in (*values.values(), feature_glob)):
        raise IndexError("index config fields must be non-empty strings")
    if "{video_id}" not in feature_glob:
        raise IndexError("feature_glob must contain {video_id}")
    return IndexConfig(
        manifest_path=Path(values["manifest_path"]),
        dataset_root=Path(values["dataset_root"]),
        output_path=Path(values["output_path"]),
        model_id=values["model_id"],
        preprocessing=values["preprocessing"],
        feature_glob=feature_glob,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_path(root: Path, feature_glob: str, video_id: str) -> Path:
    matches = sorted(root.glob(feature_glob.format(video_id=video_id)))
    if len(matches) != 1:
        raise IndexError(
            f"expected one feature array for {video_id}, found {len(matches)}"
        )
    return matches[0]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if not np.isfinite(matrix).all():
        raise IndexError("feature vectors contain NaN or infinity")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise IndexError("feature vectors contain zero-norm rows")
    return np.asarray(matrix / norms, dtype=np.float32)


def _sample_rows(length: int, count: int) -> tuple[int, ...]:
    if length <= count:
        return tuple(range(length))
    return tuple(
        sorted(
            {
                round(position * (length - 1) / (count - 1))
                for position in range(count)
            }
        )
    )

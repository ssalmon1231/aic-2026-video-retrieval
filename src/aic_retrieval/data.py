"""Read-only Kaggle dataset inventory and canonical manifest construction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypeVar

import numpy as np

from .contracts import KeyframeRecord, Manifest, VideoRecord, save_manifest

IssueLevel = Literal["error", "warning"]
FrameIdSource = Literal["csv", "timestamp"]
T = TypeVar("T")
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
OBJECT_FIELDS = (
    "detection_scores",
    "detection_class_names",
    "detection_class_entities",
    "detection_boxes",
    "detection_class_labels",
)
DIAGNOSTIC_FRAME_SAMPLE_LIMIT = 256
DIAGNOSTIC_OBJECT_SAMPLE_LIMIT = 64


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    score: float
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MappingRecord:
    keyframe_id: str
    ordinal: int
    original_frame_id: int
    pts_time: float | None = None
    fps: float | None = None


@dataclass(frozen=True, slots=True)
class AuditConfig:
    version: int
    batch_id: str
    dataset_root: Path
    video_glob: str
    keyframe_glob: str
    mapping_glob: str
    clip_glob: str
    object_glob: str
    metadata_glob: str
    output_manifest: Path
    output_report: Path
    expected_clip_dimension: int | None = 512
    verify_frame_mapping: bool = True
    frame_mapping_samples: int = 3
    frame_mapping_max_mae: float = 20.0
    fail_on_warning: bool = False
    keyframe_id_column: str = "n"
    original_frame_id_column: str = "frame_idx"
    frame_id_source: FrameIdSource = "csv"


@dataclass(frozen=True, slots=True)
class AuditIssue:
    level: IssueLevel
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class VideoProbe:
    fps: float
    frame_count: int
    duration: float


@dataclass(frozen=True, slots=True)
class AuditResult:
    manifest: Manifest | None
    counts: dict[str, int]
    issues: tuple[AuditIssue, ...]

    @property
    def valid(self) -> bool:
        return self.manifest is not None and not any(
            issue.level == "error" for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "counts": self.counts,
            "issues": [asdict(issue) for issue in self.issues],
        }


def load_config(path: str | Path) -> AuditConfig:
    """Load JSON-compatible YAML without adding a YAML dependency."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read audit config {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("audit config root must be an object")

    required = {
        "version",
        "batch_id",
        "dataset_root",
        "video_glob",
        "keyframe_glob",
        "mapping_glob",
        "clip_glob",
        "object_glob",
        "metadata_glob",
        "output_manifest",
        "output_report",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"audit config missing fields: {', '.join(missing)}")
    if int(payload["version"]) != 1:
        raise ValueError(f"unsupported audit config version: {payload['version']}")

    config = AuditConfig(
        version=1,
        batch_id=str(payload["batch_id"]),
        dataset_root=Path(payload["dataset_root"]),
        video_glob=str(payload["video_glob"]),
        keyframe_glob=str(payload["keyframe_glob"]),
        mapping_glob=str(payload["mapping_glob"]),
        clip_glob=str(payload["clip_glob"]),
        object_glob=str(payload["object_glob"]),
        metadata_glob=str(payload["metadata_glob"]),
        output_manifest=Path(payload["output_manifest"]),
        output_report=Path(payload["output_report"]),
        expected_clip_dimension=(
            int(payload["expected_clip_dimension"])
            if payload.get("expected_clip_dimension") is not None
            else None
        ),
        verify_frame_mapping=bool(payload.get("verify_frame_mapping", True)),
        frame_mapping_samples=int(payload.get("frame_mapping_samples", 3)),
        frame_mapping_max_mae=float(payload.get("frame_mapping_max_mae", 20.0)),
        fail_on_warning=bool(payload.get("fail_on_warning", False)),
        keyframe_id_column=str(payload.get("keyframe_id_column", "n")),
        original_frame_id_column=str(
            payload.get("original_frame_id_column", "frame_idx")
        ),
        frame_id_source=str(payload.get("frame_id_source", "csv")),
    )
    if not config.batch_id.strip():
        raise ValueError("batch_id must not be empty")
    if config.expected_clip_dimension is not None and config.expected_clip_dimension <= 0:
        raise ValueError("expected_clip_dimension must be positive or null")
    if config.frame_mapping_samples <= 0:
        raise ValueError("frame_mapping_samples must be positive")
    if config.frame_mapping_max_mae < 0:
        raise ValueError("frame_mapping_max_mae must be non-negative")
    if config.frame_id_source not in ("csv", "timestamp"):
        raise ValueError("frame_id_source must be 'csv' or 'timestamp'")
    return config


def probe_video(path: Path) -> VideoProbe:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "video probing requires opencv-python-headless; install the 'video' extra"
        ) from error

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("decoder could not open video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError(f"invalid probe result: fps={fps}, frame_count={frame_count}")
    return VideoProbe(fps=fps, frame_count=frame_count, duration=frame_count / fps)


def probe_clip(path: Path) -> tuple[int, int, str, bool, bool]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"empty feature shape {array.shape}")
    sample = np.asarray(array[: min(256, array.shape[0])], dtype=np.float32)
    finite = bool(np.isfinite(sample).all())
    normalized = finite and bool(
        np.allclose(np.linalg.norm(sample, axis=1), 1.0, rtol=1e-3, atol=1e-3)
    )
    return (
        int(array.shape[0]),
        int(array.shape[1]),
        str(array.dtype),
        finite,
        normalized,
    )


def probe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(str(error)) from error


def _valid_score(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    try:
        score = float(value)
    except (ValueError, OverflowError):
        return False
    return math.isfinite(score) and 0 <= score <= 1


def _valid_coordinate(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    try:
        coordinate = float(value)
    except (ValueError, OverflowError):
        return False
    return math.isfinite(coordinate) and coordinate >= 0


def _valid_box(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_valid_coordinate(coordinate) for coordinate in value)
    )


def _parallel_object_warnings(payload: dict[str, Any]) -> list[str]:
    missing = [field for field in OBJECT_FIELDS if field not in payload]
    if missing:
        return [f"parallel object schema missing fields: {', '.join(missing)}"]
    if any(not isinstance(payload[field], list) for field in OBJECT_FIELDS):
        return ["parallel object fields must be arrays"]

    lengths = {len(payload[field]) for field in OBJECT_FIELDS}
    if len(lengths) != 1:
        return ["parallel object arrays must have equal lengths"]

    warnings: list[str] = []
    count = lengths.pop()
    if count > 100:
        warnings.append(f"contains {count} detections; expected at most 100")
    for index in range(count):
        if not _valid_score(payload["detection_scores"][index]):
            warnings.append(f"detection {index} has invalid score")
        if not isinstance(payload["detection_class_names"][index], str):
            warnings.append(f"detection {index} has invalid class name")
        if not isinstance(payload["detection_class_entities"][index], str):
            warnings.append(f"detection {index} has invalid class entity")
        if not isinstance(payload["detection_class_labels"][index], (str, int)) or isinstance(
            payload["detection_class_labels"][index], bool
        ):
            warnings.append(f"detection {index} has invalid class label")
        if not _valid_box(payload["detection_boxes"][index]):
            warnings.append(f"detection {index} has invalid bbox")
    return warnings


def object_schema_warnings(payload: Any) -> list[str]:
    if isinstance(payload, list):
        detections = payload
    elif isinstance(payload, dict):
        if any(field in payload for field in OBJECT_FIELDS):
            return _parallel_object_warnings(payload)
        detections = next(
            (
                payload[key]
                for key in ("detections", "objects", "results")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if detections is None:
            return ["unrecognized object JSON schema"]
    else:
        return ["object JSON root must be an array or object"]

    warnings: list[str] = []
    if len(detections) > 100:
        warnings.append(f"contains {len(detections)} detections; expected at most 100")
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            warnings.append(f"detection {index} must be an object")
            continue
        score = detection.get("score", detection.get("confidence"))
        if score is not None and not _valid_score(score):
            warnings.append(f"detection {index} has invalid score")
        category = next(
            (
                detection[key]
                for key in ("category", "label", "class_name", "name", "class_id")
                if key in detection
            ),
            None,
        )
        if category is not None and not isinstance(category, (str, int)):
            warnings.append(f"detection {index} has invalid category")
        bbox = detection.get("bbox", detection.get("box"))
        if bbox is not None and not _valid_box(bbox):
            warnings.append(f"detection {index} has invalid bbox")
    return warnings


def parse_object_detections(payload: Any) -> tuple[ObjectDetection, ...]:
    if object_schema_warnings(payload):
        raise ValueError("invalid object detections")
    if isinstance(payload, dict) and any(field in payload for field in OBJECT_FIELDS):
        return tuple(
            ObjectDetection(
                score=float(payload["detection_scores"][index]),
                labels=_normalized_object_labels(
                    payload["detection_class_names"][index],
                    payload["detection_class_entities"][index],
                    payload["detection_class_labels"][index],
                ),
            )
            for index in range(len(payload["detection_scores"]))
        )

    detections = payload if isinstance(payload, list) else next(
        payload[key] for key in ("detections", "objects", "results") if key in payload
    )
    parsed: list[ObjectDetection] = []
    for detection in detections:
        score = detection.get("score", detection.get("confidence", 1.0))
        category = next(
            (
                detection[key]
                for key in ("category", "label", "class_name", "name", "class_id")
                if key in detection
            ),
            None,
        )
        labels = _normalized_object_labels(category)
        if labels:
            parsed.append(ObjectDetection(float(score), labels))
    return tuple(parsed)


def load_object_detections(
    dataset_root: str | Path,
    object_path: str | None,
) -> tuple[ObjectDetection, ...]:
    if not object_path:
        return ()
    root = Path(dataset_root).resolve()
    source = (root / object_path).resolve()
    try:
        source.relative_to(root)
    except ValueError:
        return ()
    try:
        return parse_object_detections(probe_json(source))
    except (OSError, ValueError):
        return ()


def _normalized_object_labels(*values: Any) -> tuple[str, ...]:
    labels: list[str] = []
    for value in values:
        label = " ".join(str(value).casefold().split()) if value is not None else ""
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


# ponytail: supports the two observed object encodings; add a schema registry only if a third appears.


def compare_frame_candidates(
    video_path: Path,
    samples: list[tuple[Path, dict[str, int]]],
) -> list[dict[str, float]]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "frame comparison requires opencv-python-headless; install the 'video' extra"
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    comparisons: list[dict[str, float]] = []
    try:
        if not capture.isOpened():
            raise ValueError("decoder could not open video")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for keyframe_path, candidates in samples:
            expected = cv2.imread(str(keyframe_path), cv2.IMREAD_COLOR)
            if expected is None:
                raise ValueError(f"could not decode keyframe {keyframe_path}")
            maes_by_frame: dict[int, float] = {}
            for frame_id in sorted(set(candidates.values())):
                if frame_id < 0 or (frame_count > 0 and frame_id >= frame_count):
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                success, actual = capture.read()
                if not success or actual is None:
                    continue
                if actual.shape[:2] != expected.shape[:2]:
                    actual = cv2.resize(
                        actual,
                        (expected.shape[1], expected.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                maes_by_frame[frame_id] = float(
                    np.mean(
                        np.abs(
                            actual.astype(np.float32) - expected.astype(np.float32)
                        )
                    )
                )
            comparison = {
                label: maes_by_frame[frame_id]
                for label, frame_id in candidates.items()
                if frame_id in maes_by_frame
            }
            if not comparison:
                raise ValueError(f"could not decode candidate frames for {keyframe_path}")
            comparisons.append(comparison)
    finally:
        capture.release()
    return comparisons


def verify_frame_mapping(
    video_path: Path,
    samples: list[tuple[Path, int]],
    max_mae: float,
) -> list[tuple[int, float]]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "frame mapping verification requires opencv-python-headless; "
            "install the 'video' extra"
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    results: list[tuple[int, float]] = []
    try:
        if not capture.isOpened():
            raise ValueError("decoder could not open video")
        for keyframe_path, original_frame_id in samples:
            expected = cv2.imread(str(keyframe_path), cv2.IMREAD_COLOR)
            if expected is None:
                raise ValueError(f"could not decode keyframe {keyframe_path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, original_frame_id)
            success, actual = capture.read()
            if not success or actual is None:
                raise ValueError(f"could not decode frame {original_frame_id}")
            if actual.shape[:2] != expected.shape[:2]:
                actual = cv2.resize(
                    actual,
                    (expected.shape[1], expected.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            mae = float(
                np.mean(
                    np.abs(
                        actual.astype(np.float32) - expected.astype(np.float32)
                    )
                )
            )
            results.append((original_frame_id, mae))
            if mae > max_mae:
                raise ValueError(
                    f"frame {original_frame_id} differs from keyframe "
                    f"(MAE {mae:.3f} > {max_mae:.3f})"
                )
    finally:
        capture.release()
    return results


def _sample_evenly(values: list[T], count: int) -> list[T]:
    if not values:
        return []
    if count == 1:
        return [values[len(values) // 2]]
    if len(values) <= count:
        return values.copy()
    indexes = {
        round(position * (len(values) - 1) / (count - 1))
        for position in range(count)
    }
    return [values[index] for index in sorted(indexes)]


def _discover(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _unique_by_stem(paths: Iterable[Path]) -> tuple[dict[str, Path], list[str]]:
    values: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        if path.stem in values:
            duplicates.append(path.stem)
        else:
            values[path.stem] = path
    return values, duplicates


def _keyframes_by_video(paths: Iterable[Path]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for path in paths:
        video_id = path.parent.name
        result.setdefault(video_id, {})[path.stem] = path
    return result


def _matching_keyframe(files: dict[str, Path], value: str) -> Path | None:
    candidates = [value, value.lstrip("0") or "0"]
    try:
        number = int(value)
    except ValueError:
        number = None
    if number is not None:
        candidates.extend((str(number), f"{number:03d}", f"{number:04d}"))
    for candidate in candidates:
        if candidate in files:
            return files[candidate]
    return None


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _optional_mapping_float(row: dict[str, str], field: str, row_number: int) -> float | None:
    raw = (row.get(field) or "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0 or (field == "fps" and value == 0):
        requirement = "finite and positive" if field == "fps" else "finite and non-negative"
        raise ValueError(f"row {row_number}: {field} must be {requirement}")
    return value


def _read_mapping(
    path: Path,
    keyframe_id_column: str,
    original_frame_id_column: str,
) -> list[MappingRecord]:
    records: list[MappingRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {keyframe_id_column, original_frame_id_column}
        if not required <= fields:
            raise ValueError(
                f"missing mapping columns {sorted(required - fields)}; found {sorted(fields)}"
            )
        for ordinal, row in enumerate(reader):
            row_number = ordinal + 2
            raw_id = (row.get(keyframe_id_column) or "").strip()
            raw_frame = (row.get(original_frame_id_column) or "").strip()
            if not raw_id or not raw_frame:
                raise ValueError(f"row {row_number}: empty keyframe/frame ID")
            records.append(
                MappingRecord(
                    keyframe_id=raw_id,
                    ordinal=ordinal,
                    original_frame_id=int(raw_frame),
                    pts_time=_optional_mapping_float(row, "pts_time", row_number),
                    fps=_optional_mapping_float(row, "fps", row_number),
                )
            )
    if not records:
        raise ValueError("mapping contains no rows")
    return records


def _frame_id(record: MappingRecord, source: FrameIdSource) -> int:
    if source == "csv":
        return record.original_frame_id
    if record.pts_time is None or record.fps is None:
        raise ValueError(
            f"row {record.ordinal + 2}: timestamp frame ID requires pts_time and fps"
        )
    return round(record.pts_time * record.fps)


def _candidate_frame_ids(record: MappingRecord) -> dict[str, int]:
    candidates = {
        "csv": record.original_frame_id,
        "csv_minus_1": record.original_frame_id - 1,
        "csv_plus_1": record.original_frame_id + 1,
    }
    if record.pts_time is not None and record.fps is not None:
        timestamp_frame = round(record.pts_time * record.fps)
        candidates.update(
            {
                "timestamp": timestamp_frame,
                "timestamp_minus_1": timestamp_frame - 1,
                "timestamp_plus_1": timestamp_frame + 1,
            }
        )
    return {label: frame_id for label, frame_id in candidates.items() if frame_id >= 0}


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def summarize_object_schemas(paths: list[Path]) -> dict[str, Any]:
    missing_fields: Counter[str] = Counter()
    field_types = {field: Counter() for field in OBJECT_FIELDS}
    item_types = {field: Counter() for field in OBJECT_FIELDS}
    box_shapes: Counter[str] = Counter()
    sampled_files = len(paths)
    invalid_json_files = 0
    non_object_root_files = 0
    equal_length_files = 0
    unequal_length_files = 0
    scores_outside_unit_interval = 0

    for path in paths:
        try:
            payload = probe_json(path)
        except ValueError:
            invalid_json_files += 1
            continue
        if not isinstance(payload, dict):
            non_object_root_files += 1
            for field in OBJECT_FIELDS:
                missing_fields[field] += 1
            unequal_length_files += 1
            continue

        lengths: list[int] = []
        for field in OBJECT_FIELDS:
            if field not in payload:
                missing_fields[field] += 1
                continue
            value = payload[field]
            field_types[field][_value_type(value)] += 1
            if not isinstance(value, list):
                continue
            lengths.append(len(value))
            item_types[field].update(_value_type(item) for item in value)
            if field == "detection_boxes":
                for box in value:
                    if isinstance(box, list):
                        box_shapes[f"list[{len(box)}]"] += 1
                    else:
                        box_shapes[_value_type(box)] += 1
            elif field == "detection_scores":
                for score in value:
                    if not _valid_score(score):
                        scores_outside_unit_interval += 1
        if len(lengths) == len(OBJECT_FIELDS) and len(set(lengths)) == 1:
            equal_length_files += 1
        else:
            unequal_length_files += 1

    return {
        "sampled_files": sampled_files,
        "invalid_json_files": invalid_json_files,
        "non_object_root_files": non_object_root_files,
        "equal_length_files": equal_length_files,
        "unequal_or_missing_length_files": unequal_length_files,
        "missing_field_counts": dict(sorted(missing_fields.items())),
        "field_type_counts": {
            field: dict(sorted(counts.items())) for field, counts in field_types.items()
        },
        "item_type_counts": {
            field: dict(sorted(counts.items())) for field, counts in item_types.items()
        },
        "box_shape_counts": dict(sorted(box_shapes.items())),
        "scores_outside_unit_interval": scores_outside_unit_interval,
    }


def diagnose_dataset(
    config: AuditConfig,
    *,
    frame_comparer: Callable[
        [Path, list[tuple[Path, dict[str, int]]]], list[dict[str, float]]
    ] = compare_frame_candidates,
    frame_sample_limit: int = DIAGNOSTIC_FRAME_SAMPLE_LIMIT,
    object_sample_limit: int = DIAGNOSTIC_OBJECT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    root = config.dataset_root
    if not root.is_dir():
        raise ValueError(f"dataset root not found: {root}")
    if frame_sample_limit <= 0 or object_sample_limit <= 0:
        raise ValueError("diagnostic sample limits must be positive")

    video_paths = [
        path
        for path in _discover(root, config.video_glob)
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    mapping_paths = _discover(root, config.mapping_glob)
    keyframe_paths = _discover(root, config.keyframe_glob)
    object_paths = _discover(root, config.object_glob)
    videos_by_id, duplicate_videos = _unique_by_stem(video_paths)
    mappings_by_id, duplicate_mappings = _unique_by_stem(mapping_paths)
    keyframes_by_video = _keyframes_by_video(keyframe_paths)

    mapping_errors = 0
    duplicate_video_count = 0
    adjacent_duplicate_pairs = 0
    incomplete_timestamp_rows = 0
    diagnostic_rows: list[tuple[Path, Path, MappingRecord]] = []
    for video_id in sorted(set(videos_by_id) & set(mappings_by_id)):
        try:
            mapping = _read_mapping(
                mappings_by_id[video_id],
                config.keyframe_id_column,
                config.original_frame_id_column,
            )
        except (OSError, UnicodeError, ValueError):
            mapping_errors += 1
            continue
        involved: set[int] = set()
        for position in range(1, len(mapping)):
            if mapping[position].original_frame_id == mapping[position - 1].original_frame_id:
                adjacent_duplicate_pairs += 1
                involved.update((position - 1, position))
        if not involved:
            continue
        duplicate_video_count += 1
        for position in sorted(involved):
            record = mapping[position]
            if record.pts_time is None or record.fps is None:
                incomplete_timestamp_rows += 1
                continue
            keyframe_path = _matching_keyframe(
                keyframes_by_video.get(video_id, {}), record.keyframe_id
            )
            if keyframe_path is not None:
                diagnostic_rows.append((videos_by_id[video_id], keyframe_path, record))

    sampled_rows = _sample_evenly(diagnostic_rows, frame_sample_limit)
    rows_by_video: dict[Path, list[tuple[Path, dict[str, int]]]] = {}
    differing_csv_timestamp_candidates = 0
    for video_path, keyframe_path, record in sampled_rows:
        candidates = _candidate_frame_ids(record)
        if candidates.get("csv") != candidates.get("timestamp"):
            differing_csv_timestamp_candidates += 1
        rows_by_video.setdefault(video_path, []).append((keyframe_path, candidates))

    mae_sums: Counter[str] = Counter()
    mae_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    comparison_errors = 0
    compared_rows = 0
    tie_rows = 0
    for video_path, samples in rows_by_video.items():
        try:
            comparisons = frame_comparer(video_path, samples)
        except (OSError, RuntimeError, ValueError):
            comparison_errors += len(samples)
            continue
        if len(comparisons) != len(samples):
            comparison_errors += len(samples)
            continue
        for comparison in comparisons:
            finite = {
                label: float(mae)
                for label, mae in comparison.items()
                if math.isfinite(mae) and mae >= 0
            }
            if not finite:
                comparison_errors += 1
                continue
            compared_rows += 1
            mae_sums.update(finite)
            mae_counts.update(finite.keys())
            best = min(finite.values())
            winners = [
                label
                for label, mae in finite.items()
                if math.isclose(mae, best, rel_tol=1e-9, abs_tol=1e-9)
            ]
            winner_counts.update(winners)
            tie_rows += len(winners) > 1

    mean_mae = {
        label: mae_sums[label] / mae_counts[label]
        for label in sorted(mae_counts)
    }
    sampled_object_paths = _sample_evenly(object_paths, object_sample_limit)
    object_schema = summarize_object_schemas(sampled_object_paths)
    evidence_ready = (
        not duplicate_videos
        and not duplicate_mappings
        and mapping_errors == 0
        and duplicate_video_count > 0
        and incomplete_timestamp_rows == 0
        and compared_rows == len(sampled_rows)
        and compared_rows > 0
        and object_schema["sampled_files"] > 0
        and object_schema["invalid_json_files"] == 0
        and object_schema["non_object_root_files"] == 0
    )
    return {
        "evidence_ready": evidence_ready,
        "counts": {
            "videos": len(video_paths),
            "mappings": len(mapping_paths),
            "keyframes": len(keyframe_paths),
            "objects": len(object_paths),
        },
        "frame_mapping": {
            "duplicate_video_count": duplicate_video_count,
            "adjacent_duplicate_pairs": adjacent_duplicate_pairs,
            "eligible_duplicate_rows": len(diagnostic_rows),
            "sampled_rows": len(sampled_rows),
            "compared_rows": compared_rows,
            "comparison_errors": comparison_errors,
            "incomplete_timestamp_rows": incomplete_timestamp_rows,
            "differing_csv_timestamp_candidates": differing_csv_timestamp_candidates,
            "winner_counts": dict(sorted(winner_counts.items())),
            "tie_rows": tie_rows,
            "mean_mae": mean_mae,
        },
        "object_schema": object_schema,
        "integrity": {
            "duplicate_video_ids": len(duplicate_videos),
            "duplicate_mapping_ids": len(duplicate_mappings),
            "mapping_errors": mapping_errors,
        },
    }


def publish_diagnostic(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def audit_dataset(
    config: AuditConfig,
    *,
    video_prober: Callable[[Path], VideoProbe] = probe_video,
) -> AuditResult:
    root = config.dataset_root
    issues: list[AuditIssue] = []
    if not root.is_dir():
        return AuditResult(
            manifest=None,
            counts={},
            issues=(
                AuditIssue("error", "dataset-root-missing", "dataset root not found", str(root)),
            ),
        )

    video_paths = [
        path
        for path in _discover(root, config.video_glob)
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    keyframe_paths = _discover(root, config.keyframe_glob)
    mapping_paths = _discover(root, config.mapping_glob)
    clip_paths = _discover(root, config.clip_glob)
    object_paths = _discover(root, config.object_glob)
    metadata_paths = _discover(root, config.metadata_glob)
    counts = {
        "videos": len(video_paths),
        "keyframes": len(keyframe_paths),
        "mappings": len(mapping_paths),
        "clip_arrays": len(clip_paths),
        "object_files": len(object_paths),
        "metadata_files": len(metadata_paths),
    }

    videos_by_id, duplicate_videos = _unique_by_stem(video_paths)
    mappings_by_id, duplicate_mappings = _unique_by_stem(mapping_paths)
    clips_by_id, duplicate_clips = _unique_by_stem(clip_paths)
    metadata_by_id, duplicate_metadata = _unique_by_stem(metadata_paths)
    keyframes_by_video = _keyframes_by_video(keyframe_paths)
    objects_by_video = _keyframes_by_video(object_paths)

    for metadata_path in metadata_paths:
        try:
            payload = probe_json(metadata_path)
            if not isinstance(payload, dict):
                issues.append(
                    AuditIssue(
                        "warning",
                        "metadata-schema-unrecognized",
                        "metadata JSON root is not an object",
                        _relative(metadata_path, root),
                    )
                )
        except ValueError as error:
            issues.append(
                AuditIssue(
                    "warning",
                    "metadata-invalid",
                    str(error),
                    _relative(metadata_path, root),
                )
            )

    for object_path in object_paths:
        try:
            warnings = object_schema_warnings(probe_json(object_path))
        except ValueError as error:
            warnings = [str(error)]
        for warning in warnings:
            issues.append(
                AuditIssue(
                    "warning",
                    "object-invalid",
                    warning,
                    _relative(object_path, root),
                )
            )

    for kind, duplicates in (
        ("video", duplicate_videos),
        ("mapping", duplicate_mappings),
        ("clip", duplicate_clips),
        ("metadata", duplicate_metadata),
    ):
        for identity in duplicates:
            issues.append(
                AuditIssue("error", f"duplicate-{kind}", f"duplicate {kind} ID {identity}")
            )

    video_records: list[VideoRecord] = []
    keyframe_records: list[KeyframeRecord] = []
    for video_id, video_path in videos_by_id.items():
        try:
            probe = video_prober(video_path)
        except (OSError, RuntimeError, ValueError) as error:
            issues.append(
                AuditIssue("error", "video-probe-failed", str(error), _relative(video_path, root))
            )
            continue

        video_records.append(
            VideoRecord(
                video_id=video_id,
                video_path=_relative(video_path, root) or "",
                fps=probe.fps,
                frame_count=probe.frame_count,
                duration=probe.duration,
                batch_id=config.batch_id,
                metadata_path=_relative(metadata_by_id.get(video_id), root),
            )
        )
        if video_id not in metadata_by_id:
            issues.append(
                AuditIssue("warning", "metadata-missing", "metadata not found", video_id)
            )

        mapping_path = mappings_by_id.get(video_id)
        if mapping_path is None:
            issues.append(AuditIssue("error", "mapping-missing", "mapping not found", video_id))
            continue
        try:
            raw_mapping = _read_mapping(
                mapping_path,
                config.keyframe_id_column,
                config.original_frame_id_column,
            )
            mapping = [
                MappingRecord(
                    record.keyframe_id,
                    record.ordinal,
                    _frame_id(record, config.frame_id_source),
                    record.pts_time,
                    record.fps,
                )
                for record in raw_mapping
            ]
        except (OSError, UnicodeError, ValueError) as error:
            issues.append(
                AuditIssue("error", "mapping-invalid", str(error), _relative(mapping_path, root))
            )
            continue

        feature_path = clips_by_id.get(video_id)
        clip_rows: int | None = None
        if feature_path is None:
            issues.append(AuditIssue("warning", "clip-missing", "CLIP array not found", video_id))
        else:
            try:
                clip_rows, clip_dimension, _, finite, normalized = probe_clip(
                    feature_path
                )
                if not finite:
                    issues.append(
                        AuditIssue(
                            "error",
                            "clip-non-finite",
                            "sampled CLIP values contain NaN or infinity",
                            _relative(feature_path, root),
                        )
                    )
                elif not normalized:
                    issues.append(
                        AuditIssue(
                            "warning",
                            "clip-not-normalized",
                            "sampled CLIP rows are not unit-normalized",
                            _relative(feature_path, root),
                        )
                    )
                if (
                    config.expected_clip_dimension is not None
                    and clip_dimension != config.expected_clip_dimension
                ):
                    issues.append(
                        AuditIssue(
                            "error",
                            "clip-dimension-mismatch",
                            f"CLIP dimension {clip_dimension} != expected "
                            f"{config.expected_clip_dimension}",
                            _relative(feature_path, root),
                        )
                    )
                if clip_rows != len(mapping):
                    issues.append(
                        AuditIssue(
                            "error",
                            "clip-row-mismatch",
                            f"CLIP rows {clip_rows} != mapping rows {len(mapping)}",
                            _relative(feature_path, root),
                        )
                    )
            except (OSError, ValueError) as error:
                issues.append(
                    AuditIssue("error", "clip-invalid", str(error), _relative(feature_path, root))
                )

        frame_files = keyframes_by_video.get(video_id, {})
        mapped_keyframes: list[tuple[Path, int]] = []
        for mapping_record in mapping:
            keyframe_path = _matching_keyframe(frame_files, mapping_record.keyframe_id)
            if keyframe_path is None:
                issues.append(
                    AuditIssue(
                        "error",
                        "keyframe-missing",
                        f"mapped keyframe {mapping_record.keyframe_id} not found",
                        video_id,
                    )
                )
                continue
            mapped_keyframes.append((keyframe_path, mapping_record.original_frame_id))
            object_path = _matching_keyframe(
                objects_by_video.get(video_id, {}), mapping_record.keyframe_id
            )
            keyframe_records.append(
                KeyframeRecord(
                    video_id=video_id,
                    keyframe_id=mapping_record.keyframe_id,
                    keyframe_path=_relative(keyframe_path, root) or "",
                    keyframe_ordinal=mapping_record.ordinal,
                    original_frame_id=mapping_record.original_frame_id,
                    clip_row=mapping_record.ordinal if clip_rows is not None else None,
                    object_path=_relative(object_path, root),
                )
            )

        if config.verify_frame_mapping and len(mapped_keyframes) == len(mapping):
            samples = _sample_evenly(mapped_keyframes, config.frame_mapping_samples)
            try:
                verify_frame_mapping(video_path, samples, config.frame_mapping_max_mae)
            except (OSError, RuntimeError, ValueError) as error:
                issues.append(
                    AuditIssue(
                        "error",
                        "frame-mapping-mismatch",
                        str(error),
                        video_id,
                    )
                )

    for orphan_video_id in sorted(set(mappings_by_id) - set(videos_by_id)):
        issues.append(
            AuditIssue("warning", "orphan-mapping", "mapping has no video", orphan_video_id)
        )
    for orphan_video_id in sorted(set(keyframes_by_video) - set(videos_by_id)):
        issues.append(
            AuditIssue("warning", "orphan-keyframes", "keyframes have no video", orphan_video_id)
        )

    manifest: Manifest | None = None
    blocking = any(issue.level == "error" for issue in issues)
    if not blocking:
        candidate = Manifest(tuple(video_records), tuple(keyframe_records))
        try:
            candidate.validate()
        except ValueError as error:
            issues.append(AuditIssue("error", "manifest-invalid", str(error)))
        else:
            manifest = candidate
    return AuditResult(manifest=manifest, counts=counts, issues=tuple(issues))


def publish_audit(config: AuditConfig, result: AuditResult) -> None:
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_report.with_name(f".{config.output_report.name}.tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(config.output_report)
    if result.manifest is not None:
        save_manifest(result.manifest, config.output_manifest)
    else:
        try:
            config.output_manifest.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit an AIC dataset without mutating it")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)

    try:
        config = load_config(arguments.config)
        result = audit_dataset(config)
        publish_audit(config, result)
    except (OSError, ValueError) as error:
        parser.exit(2, f"aic-audit-dataset: {error}\n")
    errors = sum(issue.level == "error" for issue in result.issues)
    warnings = sum(issue.level == "warning" for issue in result.issues)
    print(
        f"audit {'passed' if result.valid else 'failed'}: "
        f"{result.counts.get('videos', 0)} videos, {errors} errors, {warnings} warnings"
    )
    return 1 if errors or (config.fail_on_warning and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build private OCR records from canonical AIC keyframes."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aic_retrieval.index import ExactIndex, load_index
from aic_retrieval.ocr import (
    OcrError,
    OcrProvenance,
    OcrRecord,
    create_ocr_artifact,
)

DEFAULT_DETECTION_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_RECOGNITION_MODEL = "latin_PP-OCRv5_mobile_rec"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a private OCR artifact from canonical AIC keyframes"
    )
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--detection-model", default=DEFAULT_DETECTION_MODEL)
    parser.add_argument("--recognition-model", default=DEFAULT_RECOGNITION_MODEL)
    parser.add_argument("--model-revision", required=True)
    arguments = parser.parse_args(argv)
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not math.isfinite(arguments.min_confidence) or not 0 <= arguments.min_confidence <= 1:
        parser.error("--min-confidence must be finite and in [0, 1]")
    for name, value in (
        ("--device", arguments.device),
        ("--detection-model", arguments.detection_model),
        ("--recognition-model", arguments.recognition_model),
        ("--model-revision", arguments.model_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            parser.error(f"{name} must not be empty")

    try:
        index = load_index(arguments.index)
        runtime = _load_runtime(
            arguments.device,
            arguments.detection_model,
            arguments.recognition_model,
        )
        records = tuple(
            _extract_records(
                index,
                arguments.dataset_root,
                runtime,
                arguments.batch_size,
                arguments.min_confidence,
            )
        )
        metadata = create_ocr_artifact(
            arguments.output,
            index,
            records,
            OcrProvenance(
                engine="paddleocr",
                package_version=importlib.metadata.version("paddleocr"),
                detection_model=arguments.detection_model,
                recognition_model=arguments.recognition_model,
                model_revision=arguments.model_revision,
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError, OcrError) as error:
        parser.error(str(error))
    print(
        f"OCR artifact built: {metadata.record_count} records, "
        f"{metadata.index_rows} keyframes"
    )
    return 0


def _load_runtime(device: str, detection_model: str, recognition_model: str) -> Any:
    if not device.strip() or not detection_model.strip() or not recognition_model.strip():
        raise OcrError("OCR runtime options must not be empty")
    try:
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise OcrError(
            "PaddleOCR is unavailable; install T4-validated private runtime packages"
        ) from error
    try:
        return PaddleOCR(
            device=device,
            text_detection_model_name=detection_model,
            text_recognition_model_name=recognition_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as error:
        raise OcrError("cannot initialize pinned PaddleOCR runtime") from error


def _extract_records(
    index: ExactIndex,
    dataset_root: Path,
    runtime: Any,
    batch_size: int,
    min_confidence: float,
) -> Iterator[OcrRecord]:
    for start in range(0, len(index.keyframes), batch_size):
        stop = min(start + batch_size, len(index.keyframes))
        paths = tuple(dataset_root / item.keyframe_path for item in index.keyframes[start:stop])
        missing = next((path for path in paths if not path.is_file()), None)
        if missing is not None:
            raise OcrError(f"missing canonical keyframe {missing}")
        try:
            outputs = tuple(runtime.predict([str(path) for path in paths]))
        except Exception as error:
            raise OcrError(f"OCR inference failed for rows {start}:{stop}") from error
        if len(outputs) != len(paths):
            raise OcrError("OCR runtime returned wrong batch result count")
        for offset, output in enumerate(outputs):
            row = start + offset
            for text, confidence, box in _parse_output(output):
                if confidence < min_confidence:
                    continue
                keyframe = index.keyframes[row]
                yield OcrRecord.create(
                    row,
                    keyframe.video_id,
                    keyframe.original_frame_id,
                    text,
                    confidence,
                    box,
                )


def _parse_output(output: Any) -> tuple[tuple[str, float, tuple[float, ...]], ...]:
    payload = getattr(output, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None:
        payload = output
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]
    if not isinstance(payload, dict):
        raise OcrError("OCR result payload must be an object")
    expected = {"rec_texts", "rec_scores", "rec_boxes"}
    if not expected <= payload.keys():
        raise OcrError("OCR result payload misses recognition fields")
    texts = payload["rec_texts"]
    scores = payload["rec_scores"]
    boxes = payload["rec_boxes"]
    if (
        isinstance(texts, (str, bytes))
        or not isinstance(texts, (list, tuple))
        or not isinstance(scores, (list, tuple))
        or not isinstance(boxes, (list, tuple))
        or not len(texts) == len(scores) == len(boxes)
    ):
        raise OcrError("OCR recognition fields have inconsistent lengths")
    parsed: list[tuple[str, float, tuple[float, ...]]] = []
    for text, score, box in zip(texts, scores, boxes, strict=True):
        if not isinstance(text, str):
            raise OcrError("OCR recognized text must be a string")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= float(score) <= 1.0
        ):
            raise OcrError("OCR recognition confidence must be finite and in [0, 1]")
        if not isinstance(box, (list, tuple)):
            raise OcrError("OCR recognition box must be a sequence")
        flattened = _flatten_box(box)
        parsed.append((text, float(score), flattened))
    return tuple(parsed)


def _flatten_box(box: list[Any] | tuple[Any, ...]) -> tuple[float, ...]:
    if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
        return tuple(float(value) for value in box)
    if len(box) == 4 and all(
        isinstance(point, (list, tuple)) and len(point) == 2 for point in box
    ):
        return tuple(float(value) for point in box for value in point)
    raise OcrError("OCR recognition box must contain four values or four points")


if __name__ == "__main__":
    raise SystemExit(main())

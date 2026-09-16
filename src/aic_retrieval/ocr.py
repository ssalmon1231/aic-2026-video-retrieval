"""Private OCR artifact contracts and bounded visible-text retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from .index import ExactIndex, SearchHit

OCR_ARTIFACT_VERSION = 1
DESCRIPTOR_NAME = "ocr-artifact.json"
RECORDS_NAME = "ocr-records.jsonl"
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
ALPHANUMERIC_PATTERN = re.compile(r"[^0-9a-z]+")
FOLD_TRANSLATION = str.maketrans({"đ": "d", "ł": "l", "ø": "o", "æ": "ae", "œ": "oe"})


class OcrError(ValueError):
    """Raised when OCR artifacts or searches cannot be trusted."""


@dataclass(frozen=True, slots=True)
class OcrProvenance:
    engine: str
    package_version: str
    detection_model: str
    recognition_model: str
    model_revision: str

    def __post_init__(self) -> None:
        for name in (
            "engine",
            "package_version",
            "detection_model",
            "recognition_model",
            "model_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise OcrError(f"OCR provenance {name} must not be empty")


@dataclass(frozen=True, slots=True)
class OcrRecord:
    row: int
    video_id: str
    frame_id: int
    raw_text: str
    normalized_text: str
    alphanumeric_text: str
    confidence: float
    box: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.row, bool) or not isinstance(self.row, int) or self.row < 0:
            raise OcrError("OCR row must be a non-negative integer")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise OcrError("OCR video_id must not be empty")
        if (
            isinstance(self.frame_id, bool)
            or not isinstance(self.frame_id, int)
            or self.frame_id < 0
        ):
            raise OcrError("OCR frame_id must be a non-negative integer")
        raw = normalize_visible_text(self.raw_text)
        if self.normalized_text != raw.casefold():
            raise OcrError("OCR normalized_text disagrees with raw_text")
        if self.alphanumeric_text != alphanumeric_fold(self.raw_text):
            raise OcrError("OCR alphanumeric_text disagrees with raw_text")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise OcrError("OCR confidence must be finite and in [0, 1]")
        if not isinstance(self.box, tuple) or len(self.box) not in {4, 8}:
            raise OcrError("OCR box must contain four or eight coordinates")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in self.box
        ):
            raise OcrError("OCR box coordinates must be finite numbers")

    @classmethod
    def create(
        cls,
        row: int,
        video_id: str,
        frame_id: int,
        raw_text: str,
        confidence: float,
        box: Sequence[float],
    ) -> OcrRecord:
        normalized = normalize_visible_text(raw_text)
        return cls(
            row,
            video_id,
            frame_id,
            normalized,
            normalized.casefold(),
            alphanumeric_fold(normalized),
            confidence,
            tuple(float(value) for value in box),
        )


@dataclass(frozen=True, slots=True)
class OcrArtifactMetadata:
    version: int
    manifest_sha256: str
    index_rows: int
    record_count: int
    records_sha256: str
    provenance: OcrProvenance

    def __post_init__(self) -> None:
        if self.version != OCR_ARTIFACT_VERSION:
            raise OcrError(f"unsupported OCR artifact version {self.version}")
        for name in ("manifest_sha256", "records_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise OcrError(f"{name} must be a lowercase SHA-256")
        for name in ("index_rows", "record_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise OcrError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class OcrMatch:
    row: int
    score: float
    exact: bool
    confidence: float

    def __post_init__(self) -> None:
        if isinstance(self.row, bool) or not isinstance(self.row, int) or self.row < 0:
            raise OcrError("OCR match row must be non-negative")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise OcrError("OCR match score must be finite and in [0, 1]")
        if not isinstance(self.exact, bool):
            raise OcrError("OCR match exact flag must be boolean")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise OcrError("OCR match confidence must be finite and in [0, 1]")


class OcrIndex:
    def __init__(
        self,
        metadata: OcrArtifactMetadata,
        records: tuple[OcrRecord, ...],
    ) -> None:
        if len(records) != metadata.record_count:
            raise OcrError("OCR record count disagrees with descriptor")
        self.metadata = metadata
        self.records = records
        tokens: dict[str, set[int]] = {}
        normalized: dict[str, list[int]] = {}
        alphanumeric: dict[str, list[int]] = {}
        rows: dict[int, list[int]] = {}
        for position, record in enumerate(records):
            rows.setdefault(record.row, []).append(position)
            normalized.setdefault(record.normalized_text, []).append(position)
            for token in _tokens(record.normalized_text):
                tokens.setdefault(token, set()).add(position)
            if record.alphanumeric_text:
                alphanumeric.setdefault(record.alphanumeric_text, []).append(position)
        self._tokens = {key: frozenset(value) for key, value in tokens.items()}
        self._normalized = {
            key: _ordered_positions(value, records) for key, value in normalized.items()
        }
        self._alphanumeric = {
            key: _ordered_positions(value, records) for key, value in alphanumeric.items()
        }
        self._rows = {key: tuple(value) for key, value in rows.items()}

    def exact_candidates(
        self,
        text: str,
        *,
        limit: int = 100,
    ) -> tuple[OcrMatch, ...]:
        """Return bounded exact-text candidates from inverted indexes only."""

        query = normalize_visible_text(text).casefold()
        folded = alphanumeric_fold(text)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise OcrError("OCR result limit must be positive")
        if not _exact_query_allowed(query, folded):
            return ()
        positions: set[int] = set(self._normalized.get(query, ()))
        query_tokens = tuple(token for token in _tokens(query) if len(token) >= 2)
        if query_tokens:
            postings = [self._tokens.get(token, frozenset()) for token in query_tokens]
            if all(postings):
                positions.update(set.intersection(*(set(posting) for posting in postings)))
        if _folded_exact_allowed(folded):
            positions.update(self._alphanumeric.get(folded, ()))
        ordered = _ordered_positions(positions, self.records)[:limit]
        return self._exact_matches(query, folded, ordered, limit)

    def search(
        self,
        text: str,
        *,
        allowed_rows: Iterable[int] | None = None,
        limit: int = 100,
        fuzzy_threshold: float = 0.72,
    ) -> tuple[OcrMatch, ...]:
        query = normalize_visible_text(text).casefold()
        folded = alphanumeric_fold(text)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise OcrError("OCR result limit must be positive")
        if (
            isinstance(fuzzy_threshold, bool)
            or not isinstance(fuzzy_threshold, (int, float))
            or not math.isfinite(fuzzy_threshold)
            or not 0.0 <= fuzzy_threshold <= 1.0
        ):
            raise OcrError("OCR fuzzy threshold must be in [0, 1]")
        allowed = None if allowed_rows is None else _validate_rows(allowed_rows)
        candidates = self._candidate_positions(query, folded, allowed)
        matches: dict[int, OcrMatch] = {}
        for position in candidates:
            record = self.records[position]
            exact_text = query in record.normalized_text
            exact_folded = bool(
                folded
                and _folded_exact_allowed(folded)
                and folded in record.alphanumeric_text
            )
            exact = exact_text or exact_folded
            if exact:
                similarity = 1.0
            else:
                similarity = max(
                    SequenceMatcher(None, query, record.normalized_text).ratio(),
                    (
                        SequenceMatcher(None, folded, record.alphanumeric_text).ratio()
                        if len(folded) >= 6 and len(record.alphanumeric_text) >= 6
                        else 0.0
                    ),
                )
            if not exact and similarity < fuzzy_threshold:
                continue
            score = min(
                1.0,
                (0.85 if exact else 0.65) * similarity + 0.15 * record.confidence,
            )
            match = OcrMatch(record.row, score, exact, record.confidence)
            previous = matches.get(record.row)
            if previous is None or _match_key(match) < _match_key(previous):
                matches[record.row] = match
        return tuple(sorted(matches.values(), key=_match_key)[:limit])

    def _exact_matches(
        self,
        query: str,
        folded: str,
        positions: Iterable[int],
        limit: int,
    ) -> tuple[OcrMatch, ...]:
        matches: dict[int, OcrMatch] = {}
        for position in positions:
            record = self.records[position]
            exact = query in record.normalized_text or bool(
                folded
                and _folded_exact_allowed(folded)
                and folded in record.alphanumeric_text
            )
            if not exact:
                continue
            match = OcrMatch(
                record.row,
                min(1.0, 0.85 + 0.15 * record.confidence),
                True,
                record.confidence,
            )
            previous = matches.get(record.row)
            if previous is None or _match_key(match) < _match_key(previous):
                matches[record.row] = match
        return tuple(sorted(matches.values(), key=_match_key)[:limit])

    def _candidate_positions(
        self,
        query: str,
        folded: str,
        allowed_rows: frozenset[int] | None,
    ) -> tuple[int, ...]:
        positions: set[int] = set()
        for token in _tokens(query):
            if len(token) >= 2:
                positions.update(self._tokens.get(token, ()))
        if _folded_exact_allowed(folded):
            positions.update(self._alphanumeric.get(folded, ()))
        if allowed_rows is not None:
            allowed_positions = {
                position
                for row in allowed_rows
                for position in self._rows.get(row, ())
            }
            if not allowed_positions:
                return ()
            if positions:
                positions &= allowed_positions
            elif _fuzzy_allowed(query, folded):
                positions = allowed_positions
        if not positions and allowed_rows is None and _fuzzy_allowed(query, folded):
            positions = set(range(len(self.records)))
        return tuple(sorted(positions))


def normalize_visible_text(text: str) -> str:
    if not isinstance(text, str):
        raise OcrError("visible text must be a string")
    if any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in text
    ):
        raise OcrError("visible text contains forbidden controls")
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    if not normalized:
        raise OcrError("visible text must not be empty")
    return normalized


def alphanumeric_fold(text: str) -> str:
    normalized = normalize_visible_text(text)
    decomposed = unicodedata.normalize("NFKD", normalized).casefold()
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).translate(FOLD_TRANSLATION)
    return ALPHANUMERIC_PATTERN.sub("", without_marks)


def create_ocr_artifact(
    destination: str | Path,
    index: ExactIndex,
    records: Sequence[OcrRecord],
    provenance: OcrProvenance,
) -> OcrArtifactMetadata:
    target = Path(destination)
    ordered = tuple(
        sorted(records, key=lambda item: (item.row, item.frame_id, item.raw_text, item.box))
    )
    _validate_records(index, ordered)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        records_path = staging / RECORDS_NAME
        with records_path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in ordered:
                stream.write(
                    json.dumps(
                        asdict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        metadata = OcrArtifactMetadata(
            OCR_ARTIFACT_VERSION,
            index.metadata.manifest_sha256,
            len(index.keyframes),
            len(ordered),
            _sha256(records_path),
            provenance,
        )
        descriptor = {
            **asdict(metadata),
            "members": {"records": RECORDS_NAME},
        }
        (staging / DESCRIPTOR_NAME).write_text(
            json.dumps(
                descriptor,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        load_ocr_artifact(staging, index)
        backup = target.with_name(f".{target.name}.backup")
        if backup.exists():
            _remove_path(backup)
        if target.exists():
            if not target.is_dir():
                raise OcrError("OCR artifact destination must be a directory")
            target.replace(backup)
        try:
            staging.replace(target)
        except OSError:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        else:
            if backup.exists():
                _remove_tree(backup)
        return metadata
    except Exception:
        if staging.exists():
            _remove_tree(staging)
        raise


def load_optional_ocr_artifact(
    source: str | Path | None,
    index: ExactIndex,
) -> OcrIndex | None:
    if source is None or not Path(source).exists():
        return None
    return load_ocr_artifact(source, index)


def load_ocr_artifact(source: str | Path, index: ExactIndex) -> OcrIndex:
    root = Path(source)
    descriptor_path = root / DESCRIPTOR_NAME
    records_path = root / RECORDS_NAME
    try:
        descriptor: Any = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OcrError("cannot read OCR artifact descriptor") from error
    expected_keys = {
        "version",
        "manifest_sha256",
        "index_rows",
        "record_count",
        "records_sha256",
        "provenance",
        "members",
    }
    if not isinstance(descriptor, dict) or set(descriptor) != expected_keys:
        raise OcrError("OCR artifact descriptor has invalid fields")
    if descriptor["members"] != {"records": RECORDS_NAME}:
        raise OcrError("OCR artifact member mapping is invalid")
    try:
        members = tuple(root.iterdir())
    except OSError as error:
        raise OcrError("cannot read OCR artifact directory") from error
    actual_files = {path.name for path in members if path.is_file()}
    if (
        actual_files != {DESCRIPTOR_NAME, RECORDS_NAME}
        or len(members) != len(actual_files)
    ):
        raise OcrError("OCR artifact contains missing or unexpected files")
    try:
        metadata = OcrArtifactMetadata(
            version=descriptor["version"],
            manifest_sha256=descriptor["manifest_sha256"],
            index_rows=descriptor["index_rows"],
            record_count=descriptor["record_count"],
            records_sha256=descriptor["records_sha256"],
            provenance=OcrProvenance(**descriptor["provenance"]),
        )
    except (KeyError, TypeError) as error:
        raise OcrError("OCR artifact descriptor schema is invalid") from error
    if metadata.manifest_sha256 != index.metadata.manifest_sha256:
        raise OcrError("OCR artifact manifest hash disagrees with index")
    if metadata.index_rows != len(index.keyframes):
        raise OcrError("OCR artifact index row count disagrees with index")
    if _sha256(records_path) != metadata.records_sha256:
        raise OcrError("OCR records checksum mismatch")

    records: list[OcrRecord] = []
    try:
        with records_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise OcrError("OCR records must not contain blank lines")
                payload: Any = json.loads(line)
                if not isinstance(payload, dict):
                    raise OcrError("OCR record must be an object")
                expected_record_keys = {
                    "row",
                    "video_id",
                    "frame_id",
                    "raw_text",
                    "normalized_text",
                    "alphanumeric_text",
                    "confidence",
                    "box",
                }
                if set(payload) != expected_record_keys:
                    raise OcrError("OCR record has invalid fields")
                payload["box"] = tuple(payload["box"])
                records.append(OcrRecord(**payload))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise OcrError("cannot read OCR records") from error
    values = tuple(records)
    _validate_records(index, values)
    return OcrIndex(metadata, values)


def fuse_ocr_matches(
    hits: tuple[SearchHit, ...],
    matches: Sequence[OcrMatch],
    *,
    weight: float,
) -> tuple[SearchHit, ...]:
    if not isinstance(hits, tuple):
        raise OcrError("OCR fusion hits must be a tuple")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
    ):
        raise OcrError("OCR fusion weight must be finite and non-negative")
    rows: set[int] = set()
    for hit in hits:
        if not isinstance(hit, SearchHit) or hit.row in rows or not math.isfinite(hit.score):
            raise OcrError("OCR fusion hits contain invalid rows")
        rows.add(hit.row)
    by_row: dict[int, OcrMatch] = {}
    for match in matches:
        if not isinstance(match, OcrMatch):
            raise OcrError("OCR fusion match is invalid")
        previous = by_row.get(match.row)
        if previous is None or _match_key(match) < _match_key(previous):
            by_row[match.row] = match
    boosted = tuple(
        SearchHit(
            score=hit.score + weight * (by_row[hit.row].score if hit.row in by_row else 0.0),
            row=hit.row,
            video_id=hit.video_id,
            keyframe_id=hit.keyframe_id,
            original_frame_id=hit.original_frame_id,
            keyframe_path=hit.keyframe_path,
        )
        for hit in hits
    )
    return tuple(sorted(boosted, key=lambda hit: (-hit.score, hit.row)))


def _validate_records(index: ExactIndex, records: tuple[OcrRecord, ...]) -> None:
    identities: set[tuple[int, str, tuple[float, ...]]] = set()
    for record in records:
        if record.row >= len(index.keyframes):
            raise OcrError("OCR row lies outside index")
        keyframe = index.keyframes[record.row]
        if (
            record.video_id != keyframe.video_id
            or record.frame_id != keyframe.original_frame_id
        ):
            raise OcrError("OCR record identity disagrees with index lookup")
        identity = (record.row, record.raw_text, record.box)
        if identity in identities:
            raise OcrError("OCR artifact contains duplicate records")
        identities.add(identity)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.casefold()))


def _exact_query_allowed(query: str, folded: str) -> bool:
    tokens = _tokens(query)
    has_digit = any(character.isdigit() for character in folded)
    return has_digit and len(folded) >= 4 or len(tokens) >= 2 and len(folded) >= 6


def _folded_exact_allowed(folded: str) -> bool:
    has_digit = any(character.isdigit() for character in folded)
    return len(folded) >= (4 if has_digit else 6)


def _fuzzy_allowed(query: str, folded: str) -> bool:
    tokens = _tokens(query)
    has_strong_number = any(
        token.isdigit() and len(token) >= 2
        or any(character.isdigit() for character in token) and len(token) >= 4
        for token in tokens
    )
    return len(folded) >= 6 and (len(tokens) >= 2 or has_strong_number)


def _validate_rows(rows: Iterable[int]) -> frozenset[int]:
    try:
        values = frozenset(rows)
    except TypeError as error:
        raise OcrError("allowed OCR rows must be iterable") from error
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise OcrError("allowed OCR rows must be non-negative integers")
    return values


def _match_key(match: OcrMatch) -> tuple[object, ...]:
    return (-match.score, not match.exact, -match.confidence, match.row)


def _ordered_positions(
    positions: Iterable[int],
    records: tuple[OcrRecord, ...],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            positions,
            key=lambda position: (
                -records[position].confidence,
                records[position].row,
                position,
            ),
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_path(path: Path) -> None:
    if not path.is_dir():
        path.unlink()
        return
    for child in path.iterdir():
        _remove_path(child)
    path.rmdir()


def _remove_tree(path: Path) -> None:
    if not path.is_dir():
        raise OcrError("OCR artifact path must be a directory")
    _remove_path(path)


__all__ = [
    "DESCRIPTOR_NAME",
    "OCR_ARTIFACT_VERSION",
    "OcrArtifactMetadata",
    "OcrError",
    "OcrIndex",
    "OcrMatch",
    "OcrProvenance",
    "OcrRecord",
    "RECORDS_NAME",
    "alphanumeric_fold",
    "create_ocr_artifact",
    "fuse_ocr_matches",
    "load_ocr_artifact",
    "load_optional_ocr_artifact",
    "normalize_visible_text",
]

"""Private-safe query-pack discovery and aggregate diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .contracts import Task
from .qa import QaError

_SUFFIX_TO_TASK = {
    "-kis.txt": Task.TEXTUAL_KIS,
    "-qa.txt": Task.QA,
    "-trake.txt": Task.TRAKE,
}


@dataclass(frozen=True, slots=True)
class QueryPackEntry:
    query_id: str
    task: Task
    text: str

    def __post_init__(self) -> None:
        if not self.query_id or Path(self.query_id).name != self.query_id:
            raise QaError("query_id must be a plain filename")
        if not self.text.strip():
            raise QaError("query text must be non-empty")


@dataclass(frozen=True, slots=True)
class QueryPackDiagnostic:
    query_id: str
    task: Task
    response_count: int
    elapsed_ms: float
    status: str

    def __post_init__(self) -> None:
        if self.response_count < 0:
            raise QaError("response_count must be non-negative")
        if self.elapsed_ms < 0:
            raise QaError("elapsed_ms must be non-negative")
        if not self.status:
            raise QaError("status must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_query_pack(directory: str | Path) -> tuple[QueryPackEntry, ...]:
    """Read recognized query files without exposing their text in diagnostics."""
    root = Path(directory)
    if not root.is_dir():
        raise QaError("query pack directory does not exist")
    entries: list[QueryPackEntry] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        task = _task_for_name(path.name)
        if task is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise QaError(f"cannot read query file {path.name}") from error
        entries.append(QueryPackEntry(path.stem, task, text))
    if not entries:
        raise QaError("query pack has no recognized query files")
    query_ids = tuple(entry.query_id for entry in entries)
    if len(query_ids) != len(set(query_ids)):
        raise QaError("query pack has duplicate query IDs")
    return tuple(entries)


def summarize_diagnostics(
    diagnostics: Iterable[QueryPackDiagnostic],
) -> dict[str, object]:
    """Produce aggregate-only diagnostics with no query text or predictions."""
    values = tuple(diagnostics)
    by_task: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for value in values:
        by_task[value.task.value] = by_task.get(value.task.value, 0) + 1
        by_status[value.status] = by_status.get(value.status, 0) + 1
    return {
        "query_count": len(values),
        "by_task": dict(sorted(by_task.items())),
        "by_status": dict(sorted(by_status.items())),
        "response_count": sum(value.response_count for value in values),
    }


def _task_for_name(name: str) -> Task | None:
    if Path(name).name != name:
        return None
    for suffix, task in _SUFFIX_TO_TASK.items():
        if name.endswith(suffix):
            return task
    return None


__all__ = [
    "QueryPackDiagnostic",
    "QueryPackEntry",
    "load_query_pack",
    "summarize_diagnostics",
]

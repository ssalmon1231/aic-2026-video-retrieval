"""Strict, lazy answer-engine boundaries for automatic Q&A."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .qa import FrameEvidence, QaError

MAX_ANSWER_LENGTH = 100


class AnswerEngine(Protocol):
    def answer(self, question: str, event_description: str, frames: Sequence[FrameEvidence]) -> str:
        ...


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    slot: int
    answer: str


def parse_answer_protocol(output: str, frame_count: int) -> ParsedAnswer:
    if not isinstance(output, str) or not output.strip():
        raise QaError("answer engine output must be non-empty text")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise QaError("frame_count must be a positive integer")
    if "\n" in output or "\r" in output:
        raise QaError("answer engine output must contain exactly one line")
    parts = output.split("\t")
    if len(parts) != 2:
        raise QaError("answer engine output must be <frame_slot>\\t<short_answer>")
    slot_text, answer = parts
    if not re.fullmatch(r"[0-9]+", slot_text):
        raise QaError("answer frame slot must be a non-negative integer")
    slot = int(slot_text)
    if slot >= frame_count:
        raise QaError("answer frame slot is outside sampled evidence")
    if not answer.strip() or len(answer.strip()) > MAX_ANSWER_LENGTH:
        raise QaError("answer must be non-empty and bounded")
    if any(ord(char) < 32 and char not in "\t" for char in answer):
        raise QaError("answer contains forbidden control characters")
    return ParsedAnswer(slot, answer.strip())


class FailClosedAnswerEngine:
    """Engine boundary used when no promoted VLM is available."""

    def answer(self, question: str, event_description: str, frames: Sequence[FrameEvidence]) -> str:
        raise QaError("no promoted answer engine is available")


__all__ = ["AnswerEngine", "FailClosedAnswerEngine", "MAX_ANSWER_LENGTH", "ParsedAnswer", "parse_answer_protocol"]

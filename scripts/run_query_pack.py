#!/usr/bin/env python3
"""Run a private AIC query pack and create official CSV/ZIP outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aic_retrieval.alignment import retrieve_and_align_ranked
from aic_retrieval.contracts import Task, load_manifest
from aic_retrieval.index import load_index
from aic_retrieval.qa import QaPipeline
from aic_retrieval.qa_evidence import EvidenceConfig, build_evidence_windows, decode_window
from aic_retrieval.qa_models import FailClosedAnswerEngine
from aic_retrieval.query import QueryEncoderRuntime, load_config as load_encoder_config
from aic_retrieval.query_pack import QueryPackDiagnostic, load_query_pack, summarize_diagnostics
from aic_retrieval.retrieval import load_config as load_retrieval_config, retrieve_kis
from aic_retrieval.submission import build_submission_zip
from aic_retrieval.trake import AlignmentConfig, parse_trake_query
from aic_retrieval.video import decode_manifest_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a private AIC query pack and create official submission ZIP"
    )
    parser.add_argument("--query-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--encoder-config", required=True, type=Path)
    parser.add_argument("--retrieval-config", required=True, type=Path)
    parser.add_argument("--qa-config", required=True, type=Path)
    parser.add_argument("--trake-config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    entries = load_query_pack(args.query_dir)
    if any(entry.task is Task.QA for entry in entries) and args.manifest is None:
        parser.error("--manifest is required when the query pack contains Q&A")

    index = load_index(args.index)
    encoder = QueryEncoderRuntime(
        load_encoder_config(args.encoder_config),
        expected_dimension=index.metadata.dimension,
    )
    retrieval_config = load_retrieval_config(args.retrieval_config)
    evidence_config = EvidenceConfig(**{
        key: value for key, value in _load_object(args.qa_config).items()
        if key in {
            "window_seconds",
            "max_windows",
            "max_seeds_per_video",
            "samples_per_window",
            "near_seed_radius",
            "refine_window_seconds",
            "max_refined_windows",
        }
    })
    alignment_config = AlignmentConfig(**_load_object(args.trake_config))
    videos = load_manifest(args.manifest).videos if args.manifest is not None else ()

    video_map = {video.video_id: video for video in videos}

    def retrieve_evidence(query):
        encoded = encoder.encode([query.event_description])
        hits = index.search(encoded.vectors[0], limit=min(500, len(index.keyframes)))
        return build_evidence_windows(hits, videos, evidence_config)

    def load_qa_evidence(window):
        video = video_map.get(window.video_id)
        if video is None:
            raise ValueError(f"unknown evidence video: {window.video_id}")
        return decode_window(
            video.video_path,
            window,
            lambda _video_path, frame_id: decode_manifest_frame(
                args.dataset_root, video, frame_id
            ),
        )

    qa_pipeline = QaPipeline(
        retrieve_evidence,
        FailClosedAnswerEngine(),
        load_evidence=load_qa_evidence,
    )
    query_responses = {}
    diagnostics = []
    for entry in entries:
        started = time.perf_counter()
        try:
            if entry.task is Task.TEXTUAL_KIS:
                encoded = encoder.encode([entry.text])
                responses = retrieve_kis(index, encoded.vectors[0], retrieval_config).responses
                status = "completed"
            elif entry.task is Task.QA:
                responses, qa_status = qa_pipeline.answer_query(entry.text)
                status = "completed" if responses else (
                    "fail-closed" if qa_status.retrieved_windows else "no-evidence"
                )
            else:
                query = parse_trake_query(entry.text)
                responses = retrieve_and_align_ranked(
                    index,
                    encoder,
                    query,
                    config=alignment_config,
                )
                status = "completed" if responses else "no-monotonic-path"
        except (RuntimeError, TypeError, ValueError):
            responses = ()
            status = "failed"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        query_responses[f"{entry.query_id}.csv"] = responses
        diagnostics.append(
            QueryPackDiagnostic(
                entry.query_id,
                entry.task,
                len(responses),
                elapsed_ms,
                status,
            )
        )

    build_submission_zip(args.output, query_responses)
    report = {
        "summary": summarize_diagnostics(diagnostics),
        "queries": [diagnostic.to_dict() for diagnostic in diagnostics],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"query pack completed: {report['summary']['query_count']} queries")
    print(f"submission ZIP: {args.output.resolve()}")
    print(f"safe report: {args.report.resolve()}")
    return 0


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read config: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"config must be an object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

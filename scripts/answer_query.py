#!/usr/bin/env python3
"""Prompt-only automatic Q&A baseline CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aic_retrieval.contracts import load_manifest
from aic_retrieval.index import load_index
from aic_retrieval.qa import QaPipeline
from aic_retrieval.qa_evidence import (
    EvidenceConfig,
    build_evidence_windows,
    decode_window,
    merge_ranked_hits,
    refine_evidence_windows,
)
from aic_retrieval.qa_models import FailClosedAnswerEngine
from aic_retrieval.query import QueryEncoderRuntime, load_config as load_encoder_config
from aic_retrieval.video import decode_manifest_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run prompt-only automatic Q&A baseline")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--encoder-config", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    index = load_index(args.index)
    manifest = load_manifest(args.manifest)
    encoder = QueryEncoderRuntime(
        load_encoder_config(args.encoder_config),
        expected_dimension=index.metadata.dimension,
    )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evidence_config = EvidenceConfig(**{
        key: value for key, value in config.items()
        if key in {"window_seconds", "max_windows", "max_seeds_per_video", "samples_per_window", "near_seed_radius", "refine_window_seconds", "max_refined_windows"}
    })
    retrieval_depth = int(config.get("retrieval_depth", min(500, len(index.keyframes))))
    use_event_variant = bool(config.get("use_event_variant", True))
    video_map = {video.video_id: video for video in manifest.videos}

    def retrieve(query):
        variants = [query.raw_text]
        if use_event_variant and query.event_description != query.raw_text:
            variants.append(query.event_description)
        encoded = encoder.encode(variants)
        hit_lists = [
            index.search(vector, limit=min(retrieval_depth, len(index.keyframes)))
            for vector in encoded.vectors
        ]
        hits = merge_ranked_hits(hit_lists)
        coarse = build_evidence_windows(hits, manifest.videos, evidence_config)
        return refine_evidence_windows(coarse, manifest.videos, evidence_config)

    def load_evidence(window):
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

    responses, status = QaPipeline(
        retrieve,
        FailClosedAnswerEngine(),
        load_evidence=load_evidence,
    ).answer_query(args.prompt)
    payload = {
        "responses": [asdict(response) for response in responses],
        "status": status.to_dict(),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prompt-only TRAKE baseline CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aic_retrieval.alignment import retrieve_and_align_ranked
from aic_retrieval.index import load_index
from aic_retrieval.query import QueryEncoderRuntime, load_config as load_encoder_config
from aic_retrieval.trake import AlignmentConfig, parse_trake_query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run prompt-only TRAKE baseline")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--encoder-config", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    index = load_index(args.index)
    encoder = QueryEncoderRuntime(
        load_encoder_config(args.encoder_config),
        expected_dimension=index.metadata.dimension,
    )
    config = AlignmentConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    query = parse_trake_query(args.prompt)
    results = retrieve_and_align_ranked(index, encoder, query, config=config)
    payload = {"responses": [asdict(result) for result in results]}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

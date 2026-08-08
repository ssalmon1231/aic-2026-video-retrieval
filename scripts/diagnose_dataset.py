"""Collect aggregate evidence for unresolved Kaggle dataset contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from aic_retrieval.data import (
    diagnose_dataset,
    load_config,
    publish_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose AIC frame mapping and object schemas without mutating input"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    try:
        config = load_config(arguments.config)
        report = diagnose_dataset(config)
        publish_diagnostic(report, arguments.output)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"aic-diagnose-dataset: {error}\n")

    frame_mapping = report["frame_mapping"]
    print(
        f"diagnostic {'ready' if report['evidence_ready'] else 'incomplete'}: "
        f"{frame_mapping['duplicate_video_count']} duplicate-frame videos, "
        f"{frame_mapping['compared_rows']} rows compared"
    )
    print(f"report: {arguments.output.resolve()}")
    return 0 if report["evidence_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

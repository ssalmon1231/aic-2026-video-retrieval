"""Build and atomically publish the exact CLIP index."""

from __future__ import annotations

import argparse
from pathlib import Path

from aic_retrieval.index import build_from_config, load_config, save_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the exact AIC vector index")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    index = build_from_config(config)
    save_index(index, config.output_path)
    print(
        f"index built: {index.metadata.rows} rows, "
        f"{index.metadata.dimension} dimensions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

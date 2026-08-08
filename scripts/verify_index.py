"""Reload and verify a published exact CLIP index."""

from __future__ import annotations

import argparse
from pathlib import Path

from aic_retrieval.index import load_index, verify_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an exact AIC vector index")
    parser.add_argument("index", type=Path)
    arguments = parser.parse_args()
    index = load_index(arguments.index)
    verify_index(index, index)
    print(
        f"index verified: {index.metadata.rows} rows, "
        f"{index.metadata.dimension} dimensions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

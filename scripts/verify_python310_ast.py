"""Verify project Python and notebook code with Python 3.10 grammar."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def main() -> int:
    python_files = tuple(
        path
        for root in ("src", "scripts", "tests")
        for path in Path(root).rglob("*.py")
        if path.name != Path(__file__).name
    )
    for path in python_files:
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 10),
        )

    notebooks = tuple(Path("notebooks").glob("*.ipynb"))
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for position, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            ast.parse(
                "".join(cell.get("source", [])),
                filename=f"{path}:cell-{position}",
                feature_version=(3, 10),
            )
    print(
        f"Python 3.10 AST: {len(python_files)} files, "
        f"{len(notebooks)} notebooks OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

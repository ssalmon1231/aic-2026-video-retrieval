"""Build a deterministic, source-only Kaggle upload bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
REQUIRED_FILES = (
    Path("pyproject.toml"),
    Path("config/dataset.example.yaml"),
    Path("config/index.example.yaml"),
    Path("config/query-encoder.example.yaml"),
    Path("config/query-encoder-multilingual.example.yaml"),
    Path("config/reranker-disabled.example.yaml"),
    Path("config/hybrid-retrieval.example.yaml"),
    Path("config/qa-baseline.yaml"),
    Path("config/trake-baseline.yaml"),
    Path("config/retrieval-baseline.yaml"),
    Path("config/kis-benchmark.example.yaml"),
    Path("docs/competition-requirements.md"),
    Path("docs/evaluation-protocol.md"),
    Path("docs/kaggle-runbook.md"),
)


class BundleError(ValueError):
    """Raised when the upload bundle cannot be built safely."""


def bundle_files(root: Path) -> tuple[Path, ...]:
    project_root = root.resolve()
    relative_paths = set(REQUIRED_FILES)
    relative_paths.update(
        path.relative_to(project_root)
        for directory in ("src", "scripts")
        for path in (project_root / directory).rglob("*.py")
        if path.is_file()
    )
    relative_paths.update(
        path.relative_to(project_root)
        for path in (project_root / "tests").glob("test_*.py")
        if path.is_file()
    )

    for relative_path in relative_paths:
        source = project_root / relative_path
        if not source.is_file():
            raise BundleError(f"required bundle file not found: {relative_path.as_posix()}")
        if _contains_symlink(project_root, source):
            raise BundleError(f"bundle file must not be a symlink: {relative_path.as_posix()}")
    return tuple(sorted(relative_paths, key=lambda path: path.as_posix()))


def build_bundle(root: Path, destination: Path) -> tuple[int, str]:
    project_root = root.resolve()
    target = destination.resolve()
    if target.suffix.lower() != ".zip":
        raise BundleError("bundle output must use the .zip extension")
    files = bundle_files(project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for relative_path in files:
                info = ZipInfo(relative_path.as_posix(), FIXED_TIMESTAMP)
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (project_root / relative_path).read_bytes())
        temporary.replace(target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return len(files), sha256_file(target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_symlink(root: Path, path: Path) -> bool:
    current = path
    while current != root:
        if current.is_symlink():
            return True
        current = current.parent
    return root.is_symlink()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a deterministic source-only Kaggle upload bundle"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dist/aic-retrieval-kaggle.zip",
    )
    arguments = parser.parse_args()
    try:
        file_count, checksum = build_bundle(project_root, arguments.output)
    except (BundleError, OSError) as error:
        parser.exit(2, f"aic-prepare-kaggle-bundle: {error}\n")
    print(f"bundle ready: {arguments.output.resolve()}")
    print(f"files: {file_count}")
    print(f"sha256: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

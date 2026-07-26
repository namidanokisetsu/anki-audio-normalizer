#!/usr/bin/env python3
"""Build a minimal, cross-platform .ankiaddon archive."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "audio-normalizer.ankiaddon"
# A fixed fallback keeps local builds byte-for-byte reproducible. Release jobs
# can provide SOURCE_DATE_EPOCH when a different canonical timestamp is wanted.
DEFAULT_SOURCE_DATE_EPOCH = 1785024000  # 2026-07-26 00:00:00 UTC
ROOT_FILES = (
    "__init__.py",
    "ffmpeg.py",
    "config.json",
    "config.md",
    "manifest.json",
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
)


def source_files():
    for relative in ROOT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required add-on file is missing: {relative}")
        yield path
    yield from sorted((ROOT / "normalization").glob("*.py"))
    user_files_marker = ROOT / "user_files" / "README.txt"
    if user_files_marker.is_file():
        yield user_files_marker


def build(output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".audio_normalizer_addon_", suffix=".zip", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in source_files():
                relative = path.relative_to(ROOT).as_posix()
                epoch = int(
                    os.environ.get(
                        "SOURCE_DATE_EPOCH", str(DEFAULT_SOURCE_DATE_EPOCH)
                    )
                )
                timestamp = time.gmtime(max(epoch, 315532800))[:6]
                info = zipfile.ZipInfo(relative, date_time=timestamp)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        validate(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate(archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if "__init__.py" not in names or len(names) != len(set(names)):
            raise ValueError("Archive is missing its entry point or has duplicates.")
        forbidden = ("__pycache__", "meta.json", ".git/", "tests/")
        if any(any(part in name for part in forbidden) for name in names):
            raise ValueError("Archive contains development or private files.")
        manifest = json.loads(archive.read("manifest.json"))
        if not manifest.get("package") or not manifest.get("name"):
            raise ValueError("Archive manifest is incomplete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"archive path (default: {DEFAULT_OUTPUT.name})",
    )
    arguments = parser.parse_args()
    build(arguments.output)
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()

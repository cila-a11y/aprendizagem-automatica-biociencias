#!/usr/bin/env python3
"""Rebuild the verified support ZIP from its 252 explicit repository files.

Uses only the Python standard library and no network. Compression takes place
in memory. No output is written unless both complete-archive hashes match.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import zipfile
import zlib


EXPECTED_SHA256 = "aebaa44152f485c158db057a6c5c506a8a0eb139498514126c311a0e33bddb01"
EXPECTED_GIT_SHA1 = "c0d52a02d2eced5a97d11c3656011895c3e48a59"
PREFIX = "material_apoio_biociencias/"


def rebuild(root, metadata):
    if (metadata["format"] != 1 or metadata["archive_prefix"] != PREFIX
            or metadata["expected_sha256"] != EXPECTED_SHA256
            or metadata["expected_git_blob_sha1"] != EXPECTED_GIT_SHA1
            or metadata["expected_entries"] != 252
            or len(metadata["entries"]) != 252
            or metadata["compression_level"] != 6):
        raise ValueError("Unexpected archive metadata")
    buffer = io.BytesIO()
    seen = set()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=False) as archive:
        archive.comment = bytes.fromhex(metadata["archive_comment_hex"])
        for name, time_index, attr_index, size, expected_hash in metadata["entries"]:
            relative = PurePosixPath(name)
            if (not name or relative.is_absolute() or ".." in relative.parts
                    or "\\" in name or relative.as_posix() != name
                    or name in seen
                    or relative.suffix.lower() in {".tex", ".bib", ".docx"}):
                raise ValueError("Unexpected input path: " + name)
            seen.add(name)
            source = (root / name).resolve(strict=True)
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("Input is outside the source root: " + name)
            data = source.read_bytes()
            if len(data) != size or hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError("Input differs from the verified package: " + name)
            info = zipfile.ZipInfo(PREFIX + name,
                                   tuple(metadata["date_times"][time_index]))
            for attribute, value in metadata["zipinfo"].items():
                setattr(info, attribute, value)
            info.external_attr = metadata["external_attrs"][attr_index]
            info.extra = b""
            info.comment = b""
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=6)
    result = buffer.getvalue()
    sha256 = hashlib.sha256(result).hexdigest()
    git_sha = hashlib.sha1(b"blob " + str(len(result)).encode("ascii")
                          + b"\0" + result).hexdigest()
    if (len(result) != metadata["expected_size"] or sha256 != EXPECTED_SHA256
            or git_sha != EXPECTED_GIT_SHA1):
        raise ValueError("Archive bytes differ; output was not written. "
                         "sha256=" + sha256 + "; zlib=" + zlib.ZLIB_RUNTIME_VERSION)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Repository root containing the 252 package files")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="Explicit archive_metadata.json file")
    parser.add_argument("--output", required=True, type=Path,
                        help="Destination ZIP; an existing different file is never overwritten")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    result = rebuild(root, metadata)
    output = args.output.resolve()
    written = False
    if output.exists():
        if output.read_bytes() != result:
            raise FileExistsError("Refusing to replace a different existing output: " + str(output))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(result)
        written = True
    print(json.dumps({"output": str(output), "written": written,
                      "bytes": len(result), "sha256": EXPECTED_SHA256,
                      "git_blob_sha1": EXPECTED_GIT_SHA1,
                      "python": platform.python_version(),
                      "zlib_build": zlib.ZLIB_VERSION,
                      "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError, TypeError, zipfile.BadZipFile) as error:
        print("Archive rebuild failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)

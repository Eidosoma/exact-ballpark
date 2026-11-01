# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Where the CUDA kernels are built, which nvcc builds them, and their sources as the compiler sees them.

- `cache_root()`: the folder the kernels are built into (each kernel in its own subfolder, unless its own variable
  names another): `$EXACT_BALLPARK_CACHE`, else `$XDG_CACHE_HOME/exact-ballpark`, else `~/.cache/exact-ballpark`.
- `default_nvcc()`: `$EXACT_BALLPARK_NVCC`, else `$CUDA_HOME/bin/nvcc`, else the `nvcc` on `PATH`. Where there is none
  it returns a path that does not exist, and a build says so.
- `read_source(path)`: a kernel's source as the compiler sees it. The C library's transcribed functions
  (LGPL-2.1-or-later) live in their own headers, `glibc_*.cuh`, which the MIT kernels include where the functions
  belong; each such `#include` line is replaced by its header's code (the header's licence comment dropped). A build
  hashes, scans and compiles that text, one translation unit, so every check reads the code that is compiled.
- `SourcePath`: a path whose `read_text()` is `read_source`'s (and whose `read_bytes()` is that text's UTF-8 bytes),
  so that a kernel module's `SOURCE` reads as the compiler sees it.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
_INCLUDE = re.compile(r'^#include "(glibc_[a-z0-9_]+\.cuh)"[^\n]*\n', re.M)


def cache_root() -> Path:
    """The kernels' build folder."""
    d = os.environ.get("EXACT_BALLPARK_CACHE")
    if d:
        return Path(d)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "exact-ballpark"


def default_nvcc() -> Path:
    """The nvcc the kernels are built with."""
    d = os.environ.get("EXACT_BALLPARK_NVCC")
    if d:
        return Path(d)
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / "nvcc").exists():
        return Path(home) / "bin" / "nvcc"
    found = shutil.which("nvcc")
    return Path(found) if found else Path("/nonexistent/nvcc")


def header_code(name: str, folder: Path | None = None) -> str:
    """A glibc_*.cuh header's code: the text after its licence comment (the comment lines up to the first empty
    line). The header is looked up beside the including file, then in the package."""
    for d in ([Path(folder)] if folder is not None else []) + [PACKAGE]:
        p = d / name
        if p.is_file():
            text = Path.read_text(p)
            head, sep, code = text.partition("\n\n")
            if not sep or not all(line.startswith("//") for line in head.splitlines()):
                raise ValueError(f"{p}: a glibc header starts with its licence comment and an empty line")
            return code
    raise FileNotFoundError(f"{name}: not beside the source nor in the package")


def expand(text: str, folder: Path | None = None) -> str:
    """`text` with every `#include "glibc_*.cuh"` line replaced by that header's code."""
    return _INCLUDE.sub(lambda m: header_code(m.group(1), folder), text)


def read_source(path) -> str:
    """A kernel source file's text as the compiler sees it (its glibc headers inlined)."""
    p = Path(path)
    return expand(Path.read_text(p), p.parent)


class SourcePath(Path):
    """A kernel source's path: read_text() returns the inlined text (read_source), and read_bytes() that text's UTF-8
    bytes, which a build's manifest hashes."""

    def read_text(self, encoding=None, errors=None, newline=None):
        return expand(super().read_text(encoding, errors, newline), self.parent)

    def read_bytes(self):
        return self.read_text().encode()

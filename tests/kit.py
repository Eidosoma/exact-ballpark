# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The conformance kit's configuration, as the tests read it: `conformance/kit.toml`, or the file that
`$EXACT_BALLPARK_KIT` names. It says where the pinned upstream clones and the compiled harnesses are; a relative path
is relative to the configuration file's folder. A harness that is not built is a path that does not exist, and the
tests that need it are skipped."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config_file() -> Path:
    return Path(os.environ.get("EXACT_BALLPARK_KIT") or ROOT / "conformance" / "kit.toml")


def path(name: str) -> Path:
    """The configured path called `name` (a key of the [paths] table)."""
    f = config_file()
    with open(f, "rb") as fh:
        value = tomllib.load(fh)["paths"][name]
    p = Path(value).expanduser()
    return p if p.is_absolute() else f.parent / p


def harness(name: str) -> Path:
    """A compiled harness: core, modes, space or full."""
    return path(f"harness_{name}")

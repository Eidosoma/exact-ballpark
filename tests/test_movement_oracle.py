# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance row 2: the scalar movement reference against Destiny's own
functions compiled verbatim (conformance/build.py), on generated cases.

Both integration branches (the Taylor branch with a small friction), several steps, step lengths of 0.25 to 2 s,
goto points inside and outside the homing distance, zero direction vectors, STOP with vertical damping; positions and
velocities compared for bit equality. Development cases (seed 1) were used while writing the reference; held-out
cases (seed 2) are a separate test. Skipped when the harness has not been built on this machine.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from movement_cases import cases, reference

from kit import harness as kit_harness  # noqa: E402  (the conformance kit's paths: conformance/kit.toml)
HARNESS = kit_harness("core")
pytestmark = pytest.mark.skipif(not HARNESS.is_file(), reason="Destiny harness not built (conformance/build.py)")


def oracle(cs: list[tuple]) -> list[tuple]:
    text = "".join(" ".join(repr(x) for x in c) + "\n" for c in cs)
    res = subprocess.run([str(HARNESS)], input=text, capture_output=True, text=True, check=True, timeout=120)
    return [(int(a), int(b), *map(float, rest)) for a, b, *rest in (line.split() for line in res.stdout.splitlines())]


def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def compare(seed: int) -> None:
    cs = cases(seed)
    want = oracle(cs)
    got = [row for c in cs for row in reference(c)]
    assert len(got) == len(want)
    diffs = [(g[0], g[1], i) for g, w in zip(got, want) for i in range(2, 8) if bits(g[i]) != bits(w[i])]
    assert not diffs, f"{len(diffs)} values differ from the compiled oracle, first {diffs[:5]}"


def test_reference_matches_compiled_destiny_development_cases():
    compare(1)


def test_reference_matches_compiled_destiny_held_out_cases():
    compare(2)


def test_the_taylor_branch_is_exercised():
    cs = [c for c in cases(1) if c[1] < 1e-10 * c[5] * c[6] * c[2] * 0.001]
    assert len(cs) >= 30

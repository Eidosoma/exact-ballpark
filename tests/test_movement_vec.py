# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance row 3: the batched movement implementation against the
scalar reference, bit for bit, every step, on the CPU and (when one is visible) the GPU; the same cases as row 2."""

from __future__ import annotations

import struct
from collections import defaultdict

import pytest
import torch

from movement_cases import cases, reference
from exact_ballpark.vec import VecBallpark


def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def batched(cs: list[tuple], device: str) -> dict:
    """Run the cases batched (one VecBallpark per friction and step length); rows (id, step) -> (p, v)."""
    groups = defaultdict(list)
    for c in cs:
        groups[(c[1], c[2])].append(c)
    out = {}
    for (friction, tick), g in groups.items():
        park = VecBallpark(len(g), friction=friction, tick_interval_ms=tick, device=device)
        park.load([{"mass": c[5], "agility": c[6], "max_vel": c[7], "speed_fraction": c[8], "pos": c[9:12],
                    "vel": c[12:15], "goto": c[15:18], "mode": c[4]} for c in g])
        for s in range(1, max(c[3] for c in g) + 1):
            park.evolve()
            p, v = park.pos.cpu().tolist(), park.vel.cpu().tolist()
            for i, c in enumerate(g):
                if s <= c[3]:
                    out[(c[0], s)] = (*p[i], *v[i])
    return out


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible GPU"))])
@pytest.mark.parametrize("seed", [1, 2])
def test_batched_matches_reference_bit_for_bit(device, seed):
    cs = cases(seed)
    got = batched(cs, device)
    want = {(r[0], r[1]): r[2:] for c in cs for r in reference(c)}
    assert set(got) == set(want)
    diffs = [(k, i) for k, w in want.items() for i in range(6) if bits(got[k][i]) != bits(w[i])]
    assert not diffs, f"{len(diffs)} values differ, first {diffs[:5]}"


def test_batch_order_does_not_change_any_ball():
    cs = cases(1)[:120]
    a = batched(cs, "cpu")
    b = batched(list(reversed(cs)), "cpu")
    assert all(bits(a[k][i]) == bits(b[k][i]) for k in a for i in range(6))

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The scalar reference (exact_ballpark.full, FullBallpark) against Destiny's own code compiled verbatim with DirectXMath
(conformance/build_full.py, full_cases.HARNESS) on the flight scenarios of tests/flight_cases.py
(the flight modes with dynamical orientation on and off and every setter in flight) and tests/mode_cases.py's scenarios
converted to full_cases' records with dynamical orientation on. Every field of every ball-state row (S rows: the whole
ball, its orientation, partition filing and sensor fields included), every step, bit for bit. These rows are the ones
tests/test_movement_vec_modes.py holds the GPU kernel to, so the chain kernel = reference = compiled Destiny closes.

The first run of this comparison (23 October 2026) found that EntityWarpIn with dynamical orientation snaps the rotation
to the new velocity (Ballpark.cpp:4392-4395), which the reference did not: fixed in FullBallpark.entity_warp_in; the
omission, planted again, must be caught. Development cases are seed 1, held-out seed 2.
"""

from __future__ import annotations

import pytest

import flight_cases as flc
import full_cases as fc
import mode_cases as mc
from exact_ballpark.full import FullBallpark
from exact_ballpark.modes import ModeBallpark

needs_harness = pytest.mark.skipif(not fc.HARNESS.exists(), reason="the compiled full harness is not built")


def scenarios(seed: int, which: str) -> list:
    if which == "flight":
        return flc.cases(seed)
    return [c for c in (flc.from_mode_case(c, True) for c in mc.cases(seed)) if c]


def differences(cs: list) -> list:
    """(case, step, ball, field) where the reference and the compiled code differ, 100 scenarios an oracle call."""
    bad = []
    for k in range(0, len(cs), 100):
        chunk = cs[k:k + 100]
        want = {(r[0], r[1], r[2]): r for r in fc.oracle(chunk)["S"]}
        got = {}
        for c in chunk:
            for r in fc.reference(c)["S"]:
                got[(r[0], r[1], r[2])] = r
        if set(want) != set(got):
            return [("rows", len(want), len(got), sorted(set(want) ^ set(got))[:5])]
        for key, w in want.items():
            bad += [(*key, f) for f, a, b in zip(fc.S_FIELDS, w, got[key]) if not fc.same(a, b)]
    return bad


@needs_harness
@pytest.mark.parametrize("which", ["flight", "mc_do"])
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_reference_is_destinys_compiled_code(seed, which):
    bad = differences(scenarios(seed, which))
    assert not bad, (len(bad), bad[:8])


@needs_harness
def test_the_warp_ins_snap_is_caught_when_left_out(monkeypatch):
    """EntityWarpIn without its SnapOrientation (modes.py's alone), planted again: the compiled code differs."""
    monkeypatch.setattr(FullBallpark, "entity_warp_in", ModeBallpark.entity_warp_in)
    cs = [c for c in scenarios(1, "flight") if c["kind"] == "warpin" and c["do"]][:20]
    assert differences(cs), "the planted omission went unnoticed"


def test_the_scenarios_warp_in_with_dynamical_orientation():
    cs = scenarios(1, "flight")
    assert sum(c["do"] and any(r[0] == "K" and r[2] == fc.WARPIN_OP for r in c["records"]) for c in cs) >= 20

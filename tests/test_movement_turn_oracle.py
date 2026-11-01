# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against compiled code for turning (dynamical orientation, MOV-728, and
the orientation without it, MOV-729), turned balls through every flight mode (MOV-725's turned launcher and MOV-726's
turned formation leader through DirectXMath's rotation), and the setters in flight with every side effect (MOV-704,
MOV-711, MOV-713; SetMaxSpeed's and SetBallRadius's re-filing, the decay factor's refresh) of the reference
(exact_ballpark.full) against Destiny's own code compiled verbatim with DirectXMath's scalar path
(conformance/build_full.py, full_cases.HARNESS).

Generated scenarios (tests/turn_cases.py, four families: "vec" and "vec_do", ships turning hard under every setter;
"turned_modes", follow, orbit (both laws), warp, missile launch and formations around turning ships, with
GetRotatedVector queries; "setters", massive ships meeting while their values change, both collision methods and both
views). After every step everything full_cases records is compared for equality (doubles by their bits). Development
cases are seed 1; held-out cases (seed 2) are a separate test, first run after the generator was frozen. The
generator's first run found a reference bug, fixed in ref.py: a NaN mass or agility (SetBallMass and SetBallAgility
store it) gave the decay factor 0 where Destiny's SetBallTimeFactor gives NaN. Skipped when the harness has not been
built here.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import pytest

import full_cases as fc
import exact_ballpark.full as full
import exact_ballpark.ref as ref
import exact_ballpark.space as space
import turn_cases as tc
from exact_ballpark.full import FullBallpark
from exact_ballpark.space import DestinyUndefined

pytestmark = pytest.mark.skipif(not fc.HARNESS.is_file(),
                                reason="Destiny full harness not built (conformance/build_full.py)")

_RUNS: dict = {}


def run(seed: int) -> tuple:
    """(scenarios, Destiny's rows, the reference's rows, the undefined scenarios, branch hits), cached per seed."""
    if seed not in _RUNS:
        cs = tc.cases(seed)
        rows, undefined, coverage = defaultdict(list), {}, Counter()
        for c in cs:
            t = Counter()
            try:
                out = fc.reference(c, t)
            except DestinyUndefined as exc:
                undefined[c["id"]] = str(exc)
            else:
                for k, v in out.items():
                    rows[k].extend(v)
            coverage.update(t.keys())
        want = fc.oracle([c for c in cs if c["id"] not in undefined])
        _RUNS[seed] = (cs, want, rows, undefined, coverage)
    return _RUNS[seed]


def check(seed: int) -> None:
    cs, want, got, undefined, _ = run(seed)
    diffs = fc.differences(want, got, undefined)
    assert not diffs, {k: (len(v), v[:5]) for k, v in diffs.items()}
    assert not undefined, undefined
    assert not got["X"] and not want["X"]
    assert len(got["S"]) > 100000 and len(got["Q"]) > 2000, {k: len(v) for k, v in got.items()}


def test_reference_matches_compiled_destiny_development_cases():
    check(1)


def test_reference_matches_compiled_destiny_held_out_cases():
    check(2)


BRANCHES = [
    "orient.thrust", "orient.behind", "orient.coast", "torque.full", "torque.brake_only", "torque.none",     # MOV-728
    "torque.flip", "roll.turn", "roll.brake_only", "roll.none", "snap.at_rest", "orbit_new.heading_normal",
    "ypr.evolve", "ypr.snap", "cap_acceleration",                                                           # MOV-729
    "launch.stationary", "launch.aimed", "launch.straight", "formation.slot",                               # 725, 726
    "warp.align_wait", "warp.align_ok", "warp.drop_out", "orbit_new.default_normal", "orbit_old.tangent",   # modes
    "gtf.capped", "potential.free", "iterative.found",                                                      # setters
]


@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_every_branch_of_turning_is_reached(seed):
    cs, want, _, _, coverage = run(seed)
    thin = {b: coverage[b] for b in BRANCHES if coverage[b] < 5}
    assert not thin, f"branches reached by fewer than 5 scenarios: {thin}"
    kinds = Counter(c["kind"] for c in cs)
    assert kinds == Counter(tc.FAMILY_SIZES), kinds
    turned = sum(1 for q in want["Q"] if q[2] == fc.ROTATED_Q and q[5] == 1)       # GetRotatedVector answered
    assert turned > 1000, turned
    ops = Counter(rec[2] for c in cs for rec in c["records"] if rec[0] == "K")
    for op in tc.VEC_OPS:
        assert ops[op] >= 100, (op, ops[op])
    nan_setters = sum(1 for c in cs for rec in c["records"] if rec[0] == "K" and rec[2] in (fc.MASS_OP, fc.AGILITY_OP)
                      and math.isnan(rec[7][3]))
    assert nan_setters >= 10, nan_setters


# ---- planted faults: each must be caught by the development comparison, on the scenarios it concerns ----

_BY_CASE: dict = {}


def _want_by_case() -> dict:
    if not _BY_CASE:
        cs, want, _, undefined, _ = run(1)
        for kind, rows in want.items():
            for row in rows:
                _BY_CASE.setdefault(row[0], defaultdict(list))[kind].append(row)
    return _BY_CASE


def _caught(kinds: tuple) -> bool:
    cs, _, _, undefined, _ = run(1)
    by_case = _want_by_case()
    for c in cs:
        if c["kind"] not in kinds or c["id"] in undefined:
            continue
        try:
            got = fc.reference(c)
        except DestinyUndefined:
            return True
        if fc.differences(by_case.get(c["id"], {}), got, {}):
            return True
    return False


def _old_decay(self, b):
    m = b.mass * b.agility
    b.time_factor = math.exp(-self.friction * self.dt / m) if m > 0 else 0.0


def _max_speed_not_refiled(self, bid, v):
    v = space._float32(v)
    b = self.balls.get(bid)
    if v < 0.0 or b is None:
        return
    b.max_vel = v


def _radius_not_refiled(self, b, r):
    if r < 0.0:
        return
    b.radius = r


def _unclamped(self, bid, s):
    s = space._float32(s)
    b = self.balls.get(bid)
    if math.isfinite(s) and b is not None:
        b.speed_fraction = s


def _velocity_without_snap(self, bid, x, y, z):
    b = self.balls.get(bid)
    if b is not None:
        b.vel = [float(x), float(y), float(z)]
        b.old_vel = list(b.vel)


def _roll_mirrored(orig):
    def f(self, b, heading_torque):
        return -orig(self, b, heading_torque)
    return f


VEC, VEC_DO, MODES, SETTERS = ("vec",), ("vec_do",), ("turned_modes",), ("setters",)
FAULTS = {
    # MOV-704: the decay factor 0 for a NaN mass or agility (the reference's form before 23 October 2026)
    "decay_nan_as_zero": (VEC + VEC_DO, lambda mp: mp.setattr(ref.Ballpark, "_refresh_time_factor", _old_decay)),
    # MOV-711: SetMaxSpeed without re-filing the ball in the partition
    "max_speed_not_refiled": (SETTERS, lambda mp: mp.setattr(space.SpaceBallpark, "set_max_velocity",
                                                             _max_speed_not_refiled)),
    # MOV-711: SetBallRadius without re-filing
    "radius_not_refiled": (SETTERS, lambda mp: mp.setattr(space.SpaceBallpark, "_set_ball_radius",
                                                          _radius_not_refiled)),
    # SetSpeedFraction without its clamp to [0, 1]
    "speed_fraction_unclamped": (VEC + VEC_DO, lambda mp: mp.setattr(space.SpaceBallpark, "set_speed_fraction",
                                                                     _unclamped)),
    # MOV-713: SetBallVelocity without the orientation's snap
    "velocity_without_snap": (VEC + VEC_DO, lambda mp: mp.setattr(FullBallpark, "set_velocity",
                                                                  _velocity_without_snap)),
    # MOV-728: the roll torque mirrored
    "roll_mirrored": (VEC_DO, lambda mp: mp.setattr(FullBallpark, "_goto_roll_breaks",
                                                    _roll_mirrored(FullBallpark._goto_roll_breaks))),
    # MOV-725, MOV-726: turned balls' vectors not turned (DirectXMath's rotation as the identity)
    "directxmath_identity": (MODES, lambda mp: mp.setattr(full, "q_rotation_ypr",
                                                          lambda yaw, pitch, roll: full.QUAT_IDENTITY)),
    # MOV-729: the roll speed's decay -0.8
    "roll_speed_decay": (VEC + MODES, lambda mp: mp.setattr(full, "ROLL_SPEED_DECAY", -0.8)),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    kinds, patch = FAULTS[fault]
    patch(monkeypatch)
    assert _caught(kinds), f"the planted fault {fault} went unnoticed"

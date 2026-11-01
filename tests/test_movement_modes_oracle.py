# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against compiled code for the flight modes: the FOLLOW, ORBIT and WARP reference
(exact_ballpark.modes) against Destiny's own code compiled verbatim (conformance/build_modes.py), whose tick
(`Ballpark::Evolve`, `EvolveBehaviorForBall`), mode functions, commands and warp predicates are extracted unchanged.

Generated scenarios (tests/mode_cases.py): follow at, inside, outside and on its target, of stationary, moving, fixed
and following targets; both orbit laws near, at, far and on centre, negative ranges, zero cruise speed and heavy ships;
warps short (a goto), medium, long and very long at several warp factors and minimum ranges, aligned and unaligned
starts, the 180-tick forced start, drop-out and the overrun; followers released at activation with ids on both sides of
the warper; commands during alignment and warp proper; EntityWarpIn; the commands' guards; and multi-ball mixtures with
inert modes, at ticks of 0.25 to 20 s, three frictions and both the server's and a client's view. After every tick every
ball's state is compared for bit equality: position, velocity, goto point and behaviour acceleration (doubles), mode,
effect stamp, massive and sensor flags, follow id, owner id, followed ball, follow range, warp length and follower set;
and each ball's own sequence of events (DoModeChange, OnExitWarp, OnActivatingWarp, OnDeactivatingWarp and partition
insertions). Development cases (seed 1) were used while writing the reference; held-out cases (seed 2) are a separate
test, first run after the development cases passed. The branch test shows every branch of the slice is reached; the
planted faults show the comparison catches small errors. Skipped when the harness has not been built on this machine.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

import mode_cases as mc
import exact_ballpark.modes as modes

pytestmark = pytest.mark.skipif(not mc.HARNESS.is_file(),
                                reason="Destiny modes harness not built (conformance/build_modes.py)")

_ORACLE: dict = {}


def oracle_run(seed: int) -> tuple:
    if seed not in _ORACLE:
        cs = mc.cases(seed)
        _ORACLE[seed] = (cs, *mc.oracle(cs))
    return _ORACLE[seed]


def differences(seed: int) -> tuple[list, list]:
    """(state differences as (case, step, ball, field), balls whose event sequences differ)."""
    cs, want_rows, want_events = oracle_run(seed)
    got_rows, got_events = [], []
    for c in cs:
        r, e = mc.reference(c)
        got_rows += r
        got_events += e
    assert len(got_rows) == len(want_rows)
    state = [(w[0], w[1], w[2], mc.FIELDS[i]) for w, g in zip(want_rows, got_rows)
             for i in range(len(mc.FIELDS)) if not mc.same(w[i], g[i])]
    we, ge = mc.per_ball(want_events), mc.per_ball(got_events)
    return state, sorted(k for k in set(we) | set(ge) if we.get(k) != ge.get(k))


def test_reference_matches_compiled_destiny_development_cases():
    state, events = differences(1)
    assert not state, f"{len(state)} values differ from the compiled oracle, first {state[:5]}"
    assert not events, f"{len(events)} balls' event sequences differ, first {events[:5]}"


def test_reference_matches_compiled_destiny_held_out_cases():
    state, events = differences(2)
    assert not state, f"{len(state)} values differ from the compiled oracle, first {state[:5]}"
    assert not events, f"{len(events)} balls' event sequences differ, first {events[:5]}"


BRANCHES = [
    "follow.coincident",                                                     # MOV-714
    "orbit_old.tangent", "orbit_old.inside", "orbit_old.real_part",          # MOV-717, MOV-718
    "orbit_new.on_target", "orbit_new.default_normal",                       # MOV-719
    "orbit_new.angle_clamp_low", "orbit_new.angle_clamp_high", "orbit_new.dist_floor", "orbit_new.circle",
    "orbit_new.factor_floor", "orbit_new.factor_log", "orbit_new.factor_linear",           # MOV-720
    "gtf.orbit_thrust", "gtf.no_accel", "gtf.capped", "gtf.free",                           # MOV-721
    "orbit_thrust.capped", "orbit_thrust.off",                                              # MOV-722
    "release.stop", "release.goto_direction",                                               # MOV-730
    "warp.short_goto", "warp.repeat_noop",                                                  # TRV-701, TRV-702
    "warp.align_wait", "warp.align_ok", "warp.align_forced",                                # TRV-705, TRV-706
    "warp.speed_limited", "warp.accel", "warp.speed_kept", "warp.cruise", "warp.decel",      # TRV-708 to TRV-712
    "warp.drop_out", "warp.overrun", "warp.drop_out_insert",                                # TRV-713, TRV-714
    "warpin.warp", "warpin.goto",                                                           # TRV-715
]


@pytest.mark.parametrize("seed", [1, 2])
def test_every_branch_of_the_slice_is_reached(seed):
    cs, _, events = oracle_run(seed)
    per_case = Counter()
    for c in cs:
        t = Counter()
        mc.reference(c, t)
        per_case.update(t.keys())
    thin = {b: per_case[b] for b in BRANCHES if per_case[b] < 5}
    assert not thin, f"branches reached by fewer than 5 cases: {thin}"
    names = Counter(e[2] for e in events)
    assert all(names[n] > 0 for n in ("DoModeChange", "OnExitWarp", "OnActivatingWarp", "OnDeactivatingWarp",
                                      "partition_insert")), names
    assert {e[4] for e in events if e[2] == "OnExitWarp"} == {0, 1}             # SetMode's and WarpTo's
    kinds = Counter(c["kind"] for c in cs)
    assert min(kinds.values()) >= 60 and sum(kinds.values()) >= 1200, kinds


def test_the_harness_runs_destinys_own_code():
    m = json.loads(mc.MANIFEST.read_text())
    assert m["destiny_commit"] == "114e89fa584f0e1a81eb0a679aabf41af8870072"
    assert m["command"][:5] == ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math"]
    for name in ("Ballpark::Evolve", "Ballpark::EvolveBehaviorForBall", "Ballpark::EvolveFollow",
                 "Ballpark::EvolveOldStyleOrbit", "Ballpark::EvolveNewStyleOrbit", "Ballpark::GetOrbitalNormal",
                 "Ballpark::DefaultOrbitalNormal", "Ballpark::GotoThrustFollow", "Ballpark::OrbitThrust",
                 "Ballpark::EvolveWarp", "Ballpark::RealWarp", "Ballpark::SetupWarpConstants",
                 "Ballpark::WarpDistance", "Ballpark::WarpTo", "Ballpark::EntityWarpIn", "Ballpark::FollowBall",
                 "Ballpark::Orbit", "Ballpark::StopAllFollowers", "Ballpark::Stop(Ball*)", "Ball::SetMode",
                 "Ball::IsAlignedForWarp", "Ballpark::Integrate", "Ballpark::GotoThrust"):
        assert len(m["extracted_sha256"][name]) == 64, name
    # the four functions M2's harness also extracts are byte-identical extractions (conformance/build.py)
    assert m["extracted_sha256"]["Ballpark::Integrate"].startswith("b03b4d51cb37")
    assert m["extracted_sha256"]["Ballpark::GotoThrust"].startswith("cdf7cfbb2b28")
    assert all("#pragma once" in w for w in m["compiler_warnings"])


def test_the_warp_model_reads_no_ballpark_warp_speed():
    """TRV-716: the ballpark's mWarpSpeed is declared, initialised and exposed, and nothing else reads it; the warp
    model compiled into the harness has no such member, and the reference has none."""
    m = json.loads(mc.MANIFEST.read_text())
    assert m["mWarpSpeed_references_under_src"] == ["src/Ballpark.cpp:101", "src/Ballpark.h:95",
                                                    "src/Ballpark_Blue.cpp:99"]
    assert m["mWarpSpeed_in_extracted_code"] is False
    assert not any("warp_speed" in name for name in vars(modes.ModeBallpark()))


# ---- planted faults: each must be caught by the seed-1 comparison ----

def _drop_threshold(orig):
    def f(self, b, p, v, t):
        mv = b.max_vel
        b.max_vel = mv * 1.5
        try:
            return orig(self, b, p, v, t)
        finally:
            b.max_vel = mv
    return f


def _stop_everyone(orig):
    def f(self, b):                                  # CONF-706's rejected reading: every follower stops
        for fid in sorted(b.followers):
            self._stop(self.balls[fid])
    return f


def _half_weight(orig):
    def f(self, b, target, tv, weight):
        return orig(self, b, target, tv, 0.5)
    return f


@pytest.mark.parametrize("fault", ["MAX_ALIGN_TICKS", "WARP_FACTOR_TO_DECELERATION", "AU", "ORBITAL_PRECESSION",
                                   "MIN_WARP_DISTANCE_SQ", "drop_out_threshold", "every_follower_stops",
                                   "no_extra_step_at_drop_out", "follow_weight"])
def test_planted_faults_are_caught(fault, monkeypatch):
    constants = {"MAX_ALIGN_TICKS": 179, "WARP_FACTOR_TO_DECELERATION": 1.0 / 3000.0000001,
                 "AU": modes.AU * (1 + 2 ** -52), "ORBITAL_PRECESSION": 0.0011, "MIN_WARP_DISTANCE_SQ": 4.0e10}
    P = modes.ModeBallpark
    if fault in constants:
        monkeypatch.setattr(modes, fault, constants[fault])
    elif fault == "drop_out_threshold":                                         # TRV-713
        monkeypatch.setattr(P, "_warp_distance", _drop_threshold(P._warp_distance))
    elif fault == "every_follower_stops":                                       # MOV-730
        monkeypatch.setattr(P, "_stop_all_followers", _stop_everyone(P._stop_all_followers))
    elif fault == "follow_weight":                                              # MOV-721
        monkeypatch.setattr(P, "_goto_thrust_follow", _half_weight(P._goto_thrust_follow))
    else:                                                                       # TRV-714: commit the drop-out point
        orig_wd, orig_int, armed = P._warp_distance, P.integrate, []

        def wd(self, b, p, v, t):
            out = orig_wd(self, b, p, v, t)
            if b.mode == modes.STOP:
                armed.append(True)
            return out

        def integ(self, p, v, a, m, k, tf, t):
            if armed:
                armed.pop()
                return p, v
            return orig_int(self, p, v, a, m, k, tf, t)
        monkeypatch.setattr(P, "_warp_distance", wd)
        monkeypatch.setattr(P, "integrate", integ)
    state, events = differences(1)
    assert state or events, f"the planted fault {fault} went unnoticed"


def test_the_release_order_of_followers_does_not_matter(monkeypatch):
    """Destiny walks an unordered_set (StopAllFollowers); the reference walks the ids in ascending order. Walking them
    in descending order changes no state and no ball's own event sequence."""
    P = modes.ModeBallpark

    def descending(self, b):
        for fid in sorted(b.followers, reverse=True):
            f = self.balls.get(fid)
            if f is None or f.follow_id != b.id or f.follow_ptr != b.id:
                b.followers.discard(fid)
                continue
            if f.mode == modes.MISSILE or (f.is_interactive and f.mode == modes.ORBIT):
                if f.mode == modes.MISSILE:
                    f.is_massive = False
                self._goto_direction(f, list(f.vel))
            else:
                self._stop(f)
    monkeypatch.setattr(P, "_stop_all_followers", descending)
    state, events = differences(1)
    assert not state and not events

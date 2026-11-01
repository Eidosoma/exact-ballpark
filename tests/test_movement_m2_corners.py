# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The two early goto/stop paths, exact_ballpark.ref.Ballpark and exact_ballpark.vec.VecBallpark (on the CPU and, when one
is visible, the GPU), against exact_ballpark.full.FullBallpark where they used to differ from it. FullBallpark is held bit for bit to Destiny's compiled code
(tests/test_movement_full_oracle.py, tests/test_movement_turn_oracle.py). The cases:

- a NaN mass, at AddBall and through SetBallMass (the decay factor NaN, not 0), and a NaN agility (SetBallAgility too);
- the other values AddBall clamps, as NaN and as -0 (Destiny's `x < 0 ? 0 : x` keeps both; Python's max(0.0, x) made a
  NaN 0 and a -0 +0), and a tiny negative float that rounds to -0; vec.py's missing clamp of a negative mass;
- a -0 thrust component (a goto whose offset underflows) over a -0 velocity: Destiny integrates mLastG + mLastC (zero
  without collisions), so the velocity's zero is +0.

Every field every tick, bit for bit (two NaNs equal). Planted: either correction undone must be caught. That nothing
else changed is the before-and-after runs of the movement tests and conformance rows 1
to 3 (tests/test_movement_ref.py, test_movement_oracle.py, test_movement_vec.py), unchanged.
"""

from __future__ import annotations

import math
import struct

import pytest
import torch

from exact_ballpark import ref as R
from exact_ballpark import vec as VB
from exact_ballpark.full import FullBallpark
from exact_ballpark.ref import GOTO, STOP, Ballpark
from exact_ballpark.vec import VecBallpark

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
NAN = math.nan
TICKS = 6


def same(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b) or (a != a and b != b)


def ball(**kw) -> dict:
    """A free ball: Destiny's test hull (python/destiny/test/helpers.py) unless overridden; `goto` puts it in GOTO."""
    b = {"mass": 13_000_000.0, "radius": 2.0, "max_vel": 10.0, "agility": 0.9, "speed_fraction": 0.95,
         "pos": (1.0, 2.0, 3.0), "vel": (0.5, -0.25, 0.125), "goto": None, "set_mass": None, "set_agility": None}
    b.update(kw)
    return b


CASES = {
    "nan_mass_at_add": ball(mass=NAN, goto=(1e4, 0.0, 0.0)),
    "nan_mass_in_stop": ball(mass=NAN),
    "nan_mass_set": ball(goto=(1e4, 5.0, -3.0), set_mass=(2, NAN)),
    "nan_agility_at_add": ball(agility=NAN, goto=(-2e3, 1.0, 1.0)),
    "nan_agility_set": ball(set_agility=(3, NAN)),
    "nan_max_vel": ball(max_vel=NAN, goto=(5e3, 0.0, 0.0)),
    "nan_speed_fraction": ball(speed_fraction=NAN, goto=(5e3, 0.0, 0.0)),
    "nan_radius": ball(radius=NAN),
    "minus_zero_values": ball(radius=-0.0, max_vel=-0.0, speed_fraction=-0.0),
    "minus_zero_mass_in_stop": ball(mass=-0.0),
    "tiny_negative_floats": ball(radius=-1e-50, max_vel=-1e-50, speed_fraction=-1e-50, agility=-1e-50),
    "negative_values": ball(radius=-3.0, max_vel=-10.0, speed_fraction=-0.5, agility=-2.0),
    "negative_mass_in_stop": ball(mass=-5e6),
    # the thrust's y component underflows to -0 (goto 0 below a position of 5e-324, 1e17 m away) over a -0 velocity
    "minus_zero_thrust": ball(pos=(0.0, 5e-324, 0.0), vel=(0.0, -0.0, 0.0), goto=(1e17, 0.0, 0.0)),
    "minus_zero_thrust_z": ball(pos=(0.0, 0.0, 5e-324), vel=(0.0, 0.0, -0.0), goto=(-1e17, 0.0, 0.0),
                                mass=2e9, agility=3.0),
}
FIELDS = ("pos", "vel", "mass", "agility", "max_vel", "speed_fraction", "time_factor", "radius")
VEC_FIELDS = FIELDS[:-1]                                    # VecBallpark keeps no radius


def run_full(c: dict) -> list:
    p = FullBallpark()
    p.add_ball(1, c["mass"], c["radius"], c["max_vel"], is_free=True, x=c["pos"][0], y=c["pos"][1], z=c["pos"][2],
               vx=c["vel"][0], vy=c["vel"][1], vz=c["vel"][2], agility=c["agility"], speed_fraction=c["speed_fraction"])
    return _run(p, c, 1)


def run_ref(c: dict) -> list:
    p = Ballpark()
    p.add_ball(1, c["mass"], c["radius"], c["max_vel"], c["pos"], c["vel"], c["agility"], c["speed_fraction"])
    return _run(p, c, 1)


def _state(b) -> dict:
    return {"pos": list(b.pos), "vel": list(b.vel), "mass": b.mass, "agility": b.agility, "max_vel": b.max_vel,
            "speed_fraction": b.speed_fraction, "time_factor": b.time_factor, "radius": b.radius}


def _run(p, c: dict, bid: int) -> list:
    if c["goto"] is not None:
        p.goto_point(bid, *c["goto"])
    out = [_state(p.balls[bid])]
    for tick in range(1, TICKS):
        if c["set_mass"] and c["set_mass"][0] == tick:
            p.set_mass(bid, c["set_mass"][1])
        if c["set_agility"] and c["set_agility"][0] == tick:
            p.set_agility(bid, c["set_agility"][1])
        p.evolve()
        out.append(_state(p.balls[bid]))
    return out


def run_vec(cs: list, device: str) -> list:
    """Every case a row of one VecBallpark (no setters there: the cases without them); its rows' states a tick."""
    full = [run_full(c) for c in cs]                        # GotoPoint's own effects, as Destiny applies them
    park = VecBallpark(len(cs), device=device)
    park.load([{"mass": c["mass"], "agility": c["agility"], "max_vel": c["max_vel"],
                "speed_fraction": c["speed_fraction"], "pos": c["pos"], "vel": c["vel"],
                "goto": c["goto"] or (0.0, 0.0, 0.0), "mode": GOTO if c["goto"] is not None else STOP} for c in cs])
    park.speed_fraction = torch.tensor([f[0]["speed_fraction"] for f in full], dtype=torch.float64, device=device)
    out = [[] for _ in cs]
    for tick in range(TICKS):
        if tick:
            park.evolve()
        cols = {f: getattr(park, f).cpu().tolist() for f in VEC_FIELDS}
        for i in range(len(cs)):
            out[i].append({f: cols[f][i] for f in VEC_FIELDS})
    return out


def differences(a: list, b: list) -> list:
    bad = []
    for tick, (x, y) in enumerate(zip(a, b)):
        for f in x:
            u, v = x[f], y[f]
            if isinstance(u, list):
                if not all(same(p, q) for p, q in zip(u, v)):
                    bad.append((tick, f, u, v))
            elif not same(u, v):
                bad.append((tick, f, u, v))
    return bad


@pytest.mark.parametrize("name", sorted(CASES))
def test_ref_ballpark_is_destinys_on_the_corner_cases(name):
    c = CASES[name]
    bad = differences(run_ref(c), run_full(c))
    assert not bad, bad[:4]


@pytest.mark.parametrize("device", DEVICES)
def test_vec_ballpark_is_destinys_on_the_corner_cases(device):
    names = [n for n in sorted(CASES) if not (CASES[n]["set_mass"] or CASES[n]["set_agility"])]
    got = run_vec([CASES[n] for n in names], device)
    for n, g in zip(names, got):
        bad = differences(g, run_full(CASES[n]))
        assert not bad, (n, bad[:4])


def test_the_corner_cases_are_corner_cases():
    """The cases reach what they name: a NaN decay factor, -0 values kept, and a -0 thrust component."""
    full = {n: run_full(c) for n, c in CASES.items()}
    assert math.isnan(full["nan_mass_at_add"][0]["time_factor"]) and math.isnan(full["nan_mass_set"][2]["time_factor"])
    assert struct.pack("<d", full["minus_zero_values"][0]["max_vel"]) == struct.pack("<d", -0.0)
    assert struct.pack("<d", full["tiny_negative_floats"][0]["speed_fraction"]) == struct.pack("<d", -0.0)
    p = Ballpark()
    c = CASES["minus_zero_thrust"]
    p.add_ball(1, c["mass"], c["radius"], c["max_vel"], c["pos"], c["vel"], c["agility"], c["speed_fraction"])
    p.goto_point(1, *c["goto"])
    p.evolve()
    g = p.balls[1].last_g
    assert struct.pack("<d", g[1]) == struct.pack("<d", -0.0)                     # mLastG's -0
    assert struct.pack("<d", p.balls[1].vel[1]) == struct.pack("<d", 0.0)          # the velocity's +0 (Destiny's)


@pytest.mark.parametrize("fault", ["max_clamps", "positive_guards", "decay_rule", "last_g_alone"])
def test_planted_faults_are_caught(fault, monkeypatch):
    """Each correction undone, in the scalar path or the batched one, must change a comparison."""
    if fault == "max_clamps":                       # the old AddBall: max(0.0, x) and `> 0`
        def old_add(self, bid, mass, radius, max_vel, pos, vel, agility, speed_fraction):
            agility = R.f32(agility)
            b = R.Ball(id=bid, mass=max(0.0, float(mass)), radius=R.f32(max(0.0, radius)),
                       max_vel=R.f32(max(0.0, max_vel)), agility=agility if agility > 0 else 1.0,
                       speed_fraction=R.f32(max(0.0, speed_fraction)), pos=[float(c) for c in pos],
                       vel=[float(c) for c in vel])
            self._refresh_time_factor(b)
            self.balls[bid] = b
            return b
        monkeypatch.setattr(Ballpark, "add_ball", old_add)
    elif fault == "positive_guards":                # the old setters: refuse unless positive
        def old_set_mass(self, bid, mass):
            if not mass > 0.0:
                return
            b = self.balls[bid]
            b.mass = float(mass)
            self._refresh_time_factor(b)
        monkeypatch.setattr(Ballpark, "set_mass", old_set_mass)
    elif fault == "decay_rule":                     # the old batched decay: exp(...) only if m > 0
        monkeypatch.setattr(VB, "_c_float", lambda x: R.f32(x))
        old = VecBallpark.load

        def old_load(self, rows):
            old(self, rows)
            m = (self.mass * self.agility).tolist()
            self.time_factor = torch.tensor([math.exp(-self.friction * self.dt / x) if x > 0 else 0.0 for x in m],
                                            dtype=torch.float64, device=self.dev)
        monkeypatch.setattr(VecBallpark, "load", old_load)
    else:                                           # integrating mLastG alone
        def old_integrate_evolve(self):
            order = sorted(self.balls)
            for bid in order:
                b = self.balls[bid]
                b.last_g = [0.0, 0.0, 0.0]
                if b.mode == GOTO:
                    b.last_g = self.goto_thrust(b)
                elif b.mode == STOP:
                    b.vel[1] = (b.vel[1] - 0.07 * b.vel[1]) * R.STOP_DAMP
            new = {bid: self.integrate(self.balls[bid].pos, self.balls[bid].vel, self.balls[bid].last_g,
                                       self.balls[bid].mass * self.balls[bid].agility, self.friction,
                                       self.balls[bid].time_factor, self.dt) for bid in order}
            for bid in order:
                self.balls[bid].pos, self.balls[bid].vel = new[bid]
            self.current_time += 1
        monkeypatch.setattr(Ballpark, "evolve", old_integrate_evolve)
    def differs(c: dict) -> bool:
        try:
            return bool(differences(run_ref(c), run_full(c)))
        except ZeroDivisionError:                   # the old clamp's zero mass: Python's division raises
            return True

    caught = sum(differs(c) for c in CASES.values())
    names = [n for n in sorted(CASES) if not (CASES[n]["set_mass"] or CASES[n]["set_agility"])]
    caught += sum(bool(differences(g, run_full(CASES[n])))
                  for n, g in zip(names, run_vec([CASES[n] for n in names], "cpu")))
    assert caught, f"the planted fault {fault} went unnoticed"

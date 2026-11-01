# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The scalar movement reference: Destiny's ballpark evolution for the GOTO and STOP modes, one ball at a time.

A transcription of Carbon's `destiny` (github.com/carbonengine/destiny, MIT, (c) CCP Games) at revision 114e89fa. Rules implemented, by rule id:

- MOV-701, MOV-702, MOV-703: closed-form integration of m dv/dt = m a - k v over a step, with m = mass x agility,
  k = friction (default 1e6), both branches (`Ballpark::Integrate`, Ballpark.cpp:751).
- MOV-704: the cached one-tick factor exp(-k dt / m), refreshed when mass or agility is set (0 when m <= 0, NaN for a
  NaN m, as Destiny's SetBallTimeFactor; corrected 23 October 2026).
- MOV-705, MOV-706, MOV-707: goto thrust, maxThrust = k x speedFraction x maxVel / m along the unit vector to the
  goto point, scaled by (L^2 / D^2)^2 inside the one-tick distance D (`Ballpark::GotoThrust`, Ballpark.cpp:1397).
- MOV-708, MOV-709: GotoPoint stops the ball, sets the point and a speed fraction of 1 if it was 0; GotoDirection
  aims 1e17 m away; Stop leaves the ball coasting.
- MOV-710: in STOP the vertical velocity is damped before integration (`Ballpark::EvolveStop`, Ballpark.cpp:1339).
- MOV-711, MOV-712: AddBall's clamps; radius, max velocity, agility, speed fraction and follow range held as
  32-bit floats and promoted to double in the formulas. The clamps and the setters' guards are written as Destiny
  writes them (`x < 0 ? 0 : x`, `agility <= 0 ? 1 : agility`, SetBallMass and SetBallAgility refusing `<= 0`), so a
  NaN is kept and a -0 stays -0 (corrected 31 October 2026: Python's max(0.0, x) made a NaN 0 and a
  -0 +0, and the setters refused a NaN that Destiny stores).
- TIME-701, TIME-706, TIME-707: dt = 1 s by default; each tick computes every ball's mode acceleration first, then
  integrates every ball from its current state into temporaries, then commits; balls in ascending id. The integration
  takes mLastG + mLastC (Ballpark.cpp:487; mLastC is zero without collisions), so a -0 thrust component is +0 there
  (corrected 31 October 2026; it took mLastG alone).

Not implemented here: warp, follow, orbit, missiles, formations, collisions, partitions and bubbles (later
milestones). Plain Python floats are IEEE doubles and `math.exp` is the C library's, as Destiny's `exp` on Linux.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

STOP, GOTO = 2, 0          # Destiny's DSTBALL_GOTO is 0 and DSTBALL_STOP 2 (IDstConstants.h)
STOP_DAMP = 0.9345794392523364485981308411215   # EvolveStop's constant (Ballpark.cpp:1343), as a double


def f32(x: float) -> float:
    """Round a double to the nearest 32-bit float and back (a C++ float member read into a double)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _c_float(x: float) -> float:
    """A double passed as a C++ float parameter: rounded to float, an out-of-range value an infinity (the hardware's
    conversion, as exact_ballpark.modes._float32; f32 alone raises there)."""
    try:
        return f32(x)
    except OverflowError:
        return math.copysign(math.inf, x)


def _normalized(v: list) -> list:
    """Destiny's Vector3d::Normalize: multiply by the reciprocal of the length; a zero vector stays zero."""
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm == 0.0:
        return list(v)
    inv = 1.0 / norm
    return [v[0] * inv, v[1] * inv, v[2] * inv]


@dataclass
class Ball:
    id: int
    mass: float                       # double
    radius: float                     # float
    max_vel: float                    # float
    agility: float                    # float
    speed_fraction: float             # float
    pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mode: int = STOP
    goto: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    time_factor: float = 0.0
    last_g: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


class Ballpark:
    def __init__(self, friction: float = 1.0e6, tick_interval_ms: int = 1000):
        self.friction = friction
        self.tick_interval = tick_interval_ms
        self.dt = tick_interval_ms * 0.001
        self.balls: dict[int, Ball] = {}
        self.current_time = 0

    # ---- properties (MOV-711, MOV-712, MOV-704) ----

    def _refresh_time_factor(self, b: Ball) -> None:
        """SetBallTimeFactor (Ballpark.cpp:4968): 0 if mass x agility <= 0, else exp(-k dt / m), so a NaN mass or
        agility (SetBallMass and SetBallAgility store a NaN: their guards are `<= 0`) gives NaN, as Destiny does."""
        m = b.mass * b.agility
        b.time_factor = 0.0 if m <= 0.0 else math.exp(-self.friction * self.dt / m)

    def add_ball(self, bid: int, mass: float, radius: float, max_vel: float, pos, vel, agility: float,
                 speed_fraction: float) -> Ball:
        """AddBall (Ballpark.cpp:3391-3400): the float values rounded to float, then clamped as Destiny clamps them
        (a NaN kept, a -0 kept), the mass a double."""
        mass = float(mass)
        radius, max_vel = _c_float(radius), _c_float(max_vel)
        agility, speed_fraction = _c_float(agility), _c_float(speed_fraction)
        b = Ball(id=bid, mass=0.0 if mass < 0.0 else mass, radius=0.0 if radius < 0.0 else radius,
                 max_vel=0.0 if max_vel < 0.0 else max_vel, agility=1.0 if agility <= 0.0 else agility,
                 speed_fraction=0.0 if speed_fraction < 0.0 else speed_fraction,
                 pos=[float(c) for c in pos], vel=[float(c) for c in vel])
        self._refresh_time_factor(b)
        self.balls[bid] = b
        return b

    def set_mass(self, bid: int, mass: float) -> None:            # SetBallMass: ignored if <= 0 (a NaN is stored)
        mass = float(mass)
        if mass <= 0.0:
            return
        b = self.balls[bid]
        b.mass = mass
        self._refresh_time_factor(b)

    def set_agility(self, bid: int, agility: float) -> None:      # SetBallAgility: ignored if <= 0 (a NaN is stored)
        agility = f32(agility)
        if agility <= 0.0:
            return
        b = self.balls[bid]
        b.agility = agility
        self._refresh_time_factor(b)

    def set_max_velocity(self, bid: int, v: float) -> None:       # SetMaxSpeed: ignored if negative; no factor refresh
        v = f32(v)
        if v < 0.0:
            return
        self.balls[bid].max_vel = v

    def set_speed_fraction(self, bid: int, s: float) -> None:     # SetSpeedFraction: finite only, clamped to [0, 1]
        s = f32(s)
        if not math.isfinite(s):
            return
        self.balls[bid].speed_fraction = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s

    def set_radius(self, bid: int, r: float) -> None:             # SetBallRadius: ignored if negative
        r = f32(r)
        if r < 0.0:
            return
        self.balls[bid].radius = r

    # ---- commands (MOV-708, MOV-709) ----

    def stop(self, bid: int) -> None:
        self.balls[bid].mode = STOP

    def goto_point(self, bid: int, x: float, y: float, z: float) -> None:
        b = self.balls[bid]
        if not all(math.isfinite(c) for c in (x, y, z)):
            return
        self.stop(bid)
        b.goto = [float(x), float(y), float(z)]
        if b.speed_fraction == 0.0:
            b.speed_fraction = 1.0
        b.mode = GOTO

    def goto_direction(self, bid: int, x: float, y: float, z: float) -> None:
        b = self.balls[bid]
        d = _normalized([float(x), float(y), float(z)])
        self.goto_point(bid, *(b.pos[i] + d[i] * 1.0e17 for i in range(3)))

    # ---- one tick (TIME-706, TIME-707) ----

    def goto_thrust(self, b: Ball) -> list:
        a = [b.goto[i] - b.pos[i] for i in range(3)]
        length2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        dist = float(b.speed_fraction) * b.max_vel * self.dt
        max_thrust = self.friction * b.speed_fraction * float(b.max_vel) / (b.mass * b.agility)
        unit = _normalized(a)
        if length2 < dist * dist:
            coff = length2 / (dist * dist)
            return [c * (max_thrust * coff * coff) for c in unit]
        return [c * max_thrust for c in unit]

    def integrate(self, p: list, v: list, a: list, m: float, k: float, time_factor: float, t: float) -> tuple:
        if t == 0.0:
            return p, v
        if t != self.dt:
            time_factor = math.exp(-k / m * t)
        k2 = k * k
        ook2 = 1.0 / k2
        ook = 1.0 / k
        if k < 1e-10 * m * t:
            # `v * k / m * t` on a Vector3d: its operator / multiplies by the reciprocal (Vector3d.h), so (v*k)*(1/m)*t
            inv_m = 1.0 / m
            pn = [(m * (a[i] * (-k * t) + k * (a[i] * t + v[i] * k * inv_m * t)) + p[i] * k2) * ook2 for i in range(3)]
        else:
            pn = [(m * (m * a[i] * (time_factor - 1.0) + k * (a[i] * t + v[i] - time_factor * v[i])) + p[i] * k2) * ook2
                  for i in range(3)]
        vn = [(m * a[i] - (m * a[i] - v[i] * k) * time_factor) * ook for i in range(3)]
        return pn, vn

    def evolve(self) -> None:
        order = sorted(self.balls)
        for bid in order:                                   # (1) mode accelerations
            b = self.balls[bid]
            b.last_g = [0.0, 0.0, 0.0]
            if b.mode == GOTO:
                b.last_g = self.goto_thrust(b)
            elif b.mode == STOP:
                b.vel[1] = (b.vel[1] - 0.07 * b.vel[1]) * STOP_DAMP
        new = {}
        for bid in order:                                   # (3) integrate into temporaries: mLastG + mLastC (zero)
            b = self.balls[bid]
            a = [c + 0.0 for c in b.last_g]
            new[bid] = self.integrate(b.pos, b.vel, a, b.mass * b.agility, self.friction, b.time_factor, self.dt)
        for bid in order:                                   # (4) commit
            self.balls[bid].pos, self.balls[bid].vel = new[bid]
        self.current_time += 1                              # (7)

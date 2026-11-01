# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The batched movement implementation: Destiny's GOTO and STOP evolution for B balls at once, as float64 tensors.

It must reproduce `ref.py` bit for bit (conformance row 3, tests/test_movement_vec.py), on the CPU and the GPU. That
is possible because a tick needs only correctly rounded IEEE operations (+, -, *, /, sqrt): the one-tick decay factor
exp(-k dt / m) is computed once on the host with the C library's `exp` when mass or agility is set (MOV-704) and
stored per ball. To keep the bits, every expression keeps Destiny's operation order, sums of squares are written
left to right instead of reduced, vector normalisation and vector division multiply by the reciprocal (Vector3d.h),
and nothing is fused or compiled (no torch.compile: a fused multiply-add would change the last bits).

Two early corner cases corrected (31 October 2026; they changed nothing for finite, non-negative
values): `load` clamps as Destiny's AddBall does (a negative mass 0, a NaN kept, a -0 kept; the decay factor 0 only
for mass x agility <= 0, so NaN for a NaN), and `evolve` integrates mLastG + mLastC (zero without collisions), so a -0
thrust component is +0 there, as in Destiny, exact_ballpark.ref and exact_ballpark.vec_orient.
"""

from __future__ import annotations

import math

import torch

from .ref import GOTO, STOP, STOP_DAMP, _c_float

F64 = torch.float64


class VecBallpark:
    """B independent balls, one per row (one ball per world in M2's movement tests)."""

    def __init__(self, n: int, friction: float = 1.0e6, tick_interval_ms: int = 1000, device="cpu"):
        self.n, self.dev = n, torch.device(device)
        self.friction, self.dt = float(friction), tick_interval_ms * 0.001
        z = lambda *s: torch.zeros(*s, dtype=F64, device=self.dev)
        self.pos, self.vel, self.goto = z(n, 3), z(n, 3), z(n, 3)
        self.mass, self.agility, self.max_vel, self.speed_fraction, self.time_factor = z(n), z(n), z(n), z(n), z(n)
        self.mode = torch.full((n,), STOP, dtype=torch.int64, device=self.dev)

    def load(self, rows: list[dict]) -> None:
        """Set every ball from host values as AddBall does (Ballpark.cpp:3391-3400): the float32 properties rounded,
        then each clamped as Destiny clamps it (a NaN kept, a -0 kept), and the decay factor computed here
        (SetBallTimeFactor: 0 if mass x agility <= 0, else exp(-k dt / m))."""
        t = lambda xs: torch.tensor(xs, dtype=F64, device=self.dev)
        mass = [float(r["mass"]) for r in rows]
        mass = [0.0 if m < 0.0 else m for m in mass]
        agility = [_c_float(r["agility"]) for r in rows]
        agility = [1.0 if a <= 0.0 else a for a in agility]
        max_vel = [_c_float(r["max_vel"]) for r in rows]
        speed_fraction = [_c_float(r["speed_fraction"]) for r in rows]
        self.mass = t(mass)
        self.agility = t(agility)
        self.max_vel = t([0.0 if v < 0.0 else v for v in max_vel])
        self.speed_fraction = t([0.0 if f < 0.0 else f for f in speed_fraction])
        tf = []
        for ms, ag in zip(mass, agility):
            m = ms * ag
            tf.append(0.0 if m <= 0.0 else math.exp(-self.friction * self.dt / m))
        self.time_factor = t(tf)
        self.pos = t([list(map(float, r["pos"])) for r in rows])
        self.vel = t([list(map(float, r["vel"])) for r in rows])
        self.goto = t([list(map(float, r.get("goto", (0.0, 0.0, 0.0)))) for r in rows])
        self.mode = torch.tensor([r["mode"] for r in rows], dtype=torch.int64, device=self.dev)

    def _goto_thrust(self) -> torch.Tensor:
        a = self.goto - self.pos
        length2 = a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1] + a[:, 2] * a[:, 2]
        dist = self.speed_fraction * self.max_vel * self.dt
        max_thrust = self.friction * self.speed_fraction * self.max_vel / (self.mass * self.agility)
        norm = torch.sqrt(length2)
        inv = torch.where(norm == 0.0, torch.ones_like(norm), 1.0 / torch.where(norm == 0.0, torch.ones_like(norm), norm))
        unit = torch.where((norm == 0.0)[:, None], a, a * inv[:, None])
        dd = dist * dist
        coff = length2 / torch.where(dd == 0.0, torch.ones_like(dd), dd)
        scale = torch.where(length2 < dd, max_thrust * coff * coff, max_thrust)
        return unit * scale[:, None]

    def evolve(self) -> None:
        goto, stop = self.mode == GOTO, self.mode == STOP
        acc = torch.where(goto[:, None], self._goto_thrust(), torch.zeros_like(self.pos))
        vy = self.vel[:, 1]
        self.vel[:, 1] = torch.where(stop, (vy - 0.07 * vy) * STOP_DAMP, vy)
        m = (self.mass * self.agility)[:, None]
        k, t, tf = self.friction, self.dt, self.time_factor[:, None]
        k2 = k * k
        ook2, ook = 1.0 / k2, 1.0 / k
        p, v, a = self.pos, self.vel, acc + 0.0         # mLastG + mLastC (zero without collisions): -0 becomes +0
        exact = (m * (m * a * (tf - 1.0) + k * (a * t + v - tf * v)) + p * k2) * ook2
        taylor = (m * (a * (-k * t) + k * (a * t + v * k * (1.0 / m) * t)) + p * k2) * ook2
        small = (k < 1e-10 * m * t)
        self.pos = torch.where(small, taylor, exact)
        self.vel = (m * a - (m * a - v * k) * tf) * ook

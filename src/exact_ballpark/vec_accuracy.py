# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's accuracy helpers batched: `GetAccuracy` (Ballpark.cpp:5156; CMB-702) and
`Ball::CalculateRelativeVelocities` (Ball.cpp:161; CMB-701) for B (shooter, target) pairs at once, as float64 tensors
on the CPU or a GPU.

It must reproduce `exact_ballpark.full.FullBallpark.get_accuracy` and `relative_velocities` bit for bit, which are held
bit for bit to Destiny's own code compiled verbatim (tests/test_movement_full_oracle.py), so it is exact against Destiny
(Carbon's `destiny` at 114e89fa, MIT, (c) CCP Games). Checked by
tests/test_movement_vec_accuracy.py on generated pairs, on the CPU and the GPU, with planted faults.

**Exactness.** Every operation is a correctly rounded IEEE operation (+, -, *, /, sqrt) in the reference's order, with
tensor divisors (CUDA divides by a host scalar through its reciprocal), sums of squares written left to right, nothing
fused or compiled. The power 2^E is the C library's pow (glibc 2.34 aarch64, as Destiny's compiled code and the
reference's `math.pow` call it), transcribed operation for operation in `exact_ballpark.vec_libm.pow`; the same function
runs on the device in `exact_ballpark.vec_combat_cuda` (vec_orient_cuda.cu's g_pow). The reference took 2^E with
Python's `math.pow`, which raises OverflowError for E of 1024 or more (a speed2 made negative by rounding times a large
k^2) where Destiny's pow returns +inf; this port changed it to the C library's pow (equal wherever math.pow
returns), a finding in the reference, not in Destiny (tests/test_movement_vec_accuracy.py pins it). The division by
dist * dist cannot meet a zero divisor: dist is the square root of a sum that is at least the least subnormal.

**The pairs.** Row i is one call: the shooter's and the target's position, velocity and radius (the radius a float
value held in float64, as the reference holds it), the call's arguments, and two masks the caller derives from its
park: `same` (the shooter is the target: src == dst) and `present` (both balls exist). With dynamical orientation
CalculateRelativeVelocities also takes the shooter's angular velocity (floats held in float64).

This module is Destiny's helper as the engine computes it (pow, not a portable exp2).
"""

from __future__ import annotations

import torch

from . import vec_libm as L

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
UNSET = -7777.0                    # the harness driver's mark for a value CalculateRelativeVelocities left unset
REL, RADIAL, TRANSVERSAL, ANGULAR = 1, 2, 4, 8      # CalculateRelativeVelocities' flags (Ball.cpp:161)


def _dot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vector3d's dot product, x first (Vector3d.h:49): u0 v0 + u1 v1 + u2 v2, left to right."""
    return u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1] + u[..., 2] * v[..., 2]


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vector3d::Cross (Vector3d.h:74), components in Destiny's order."""
    return torch.stack((u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1], u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                        u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]), dim=-1)


def _safe(x: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    """x with 1.0 where `bad` (a divisor for rows whose result is not used)."""
    return torch.where(bad, torch.ones_like(x), x)


def get_accuracy(pos_s: torch.Tensor, vel_s: torch.Tensor, rad_s: torch.Tensor, pos_d: torch.Tensor,
                 vel_d: torch.Tensor, rad_d: torch.Tensor, optimal: torch.Tensor, falloff: torch.Tensor,
                 tracking: torch.Tensor, signature: torch.Tensor, optimal_signature: torch.Tensor,
                 same: torch.Tensor | None = None, present: torch.Tensor | None = None) -> tuple:
    """GetAccuracy (Ballpark.cpp:5156; CMB-702) for each row: (2^E, the surface distance, the angular speed), as
    FullBallpark.get_accuracy returns them. The early returns give (0, optimal, tracking): the same ball, a missing
    ball, a negative optimal range, a zero tracking speed, signature radius or falloff, coincident centres."""
    f = lambda x: torch.as_tensor(x, dtype=F64, device=pos_s.device)
    optimal, falloff, tracking, signature, optimal_signature = (f(x) for x in (optimal, falloff, tracking, signature,
                                                                               optimal_signature))
    B = pos_s.shape[0]
    same = torch.zeros(B, dtype=BOOL, device=pos_s.device) if same is None else same.to(BOOL)
    present = torch.ones(B, dtype=BOOL, device=pos_s.device) if present is None else present.to(BOOL)
    early = same | ~present | (optimal < 0.0) | (tracking == 0.0) | (signature == 0.0) | (falloff == 0.0)
    radial = pos_d - pos_s
    dist = torch.sqrt(_dot(radial, radial))
    short = (dist - rad_d) - rad_s
    short = torch.where(short < 0.0, torch.zeros_like(short), short)
    early = early | (dist == 0.0)
    combined = vel_d - vel_s
    radial_speed = _dot(combined, radial) / _safe(dist, dist == 0.0)
    speed2 = (_dot(combined, combined) - radial_speed * radial_speed) / (dist * dist)
    exponent = optimal_signature / (tracking * signature)
    exponent = exponent * (-exponent * speed2)
    gap = short - optimal
    exponent = torch.where(short > optimal, exponent - (gap * gap) / (falloff * falloff), exponent)
    acc = L.pow(torch.full_like(exponent, 2.0), exponent)
    zero = torch.zeros_like(acc)
    return (torch.where(early, zero, acc), torch.where(early, optimal, short),
            torch.where(early, tracking, torch.sqrt(speed2)))


def relative_velocities(pos_s: torch.Tensor, vel_s: torch.Tensor, rad_s: torch.Tensor, pos_t: torch.Tensor,
                        vel_t: torch.Tensor, rad_t: torch.Tensor, flags: torch.Tensor,
                        ang_vel_s: torch.Tensor | None = None, dynamical: bool = False,
                        present: torch.Tensor | None = None) -> tuple:
    """Ball::CalculateRelativeVelocities (Ball.cpp:161; CMB-701) for each row: (relative velocity [B, 3], radial
    speed, surface distance, transversal speed, angular speed), each -7777 where it is not computed, as
    FullBallpark.relative_velocities returns them. `flags` per row (REL 1, RADIAL 2, TRANSVERSAL 4, ANGULAR 8; each
    implies the ones before it); with dynamical orientation the shooter's own rotation (cross(angVel, radial) x dist)
    is taken off the transversal velocity before the angular speed."""
    dev = pos_s.device
    B = pos_s.shape[0]
    flags = torch.as_tensor(flags, dtype=I64, device=dev).expand(B)
    present = torch.ones(B, dtype=BOOL, device=dev) if present is None else present.to(BOOL)
    c_ang = present & ((flags & ANGULAR) != 0)
    c_trans = present & (((flags & TRANSVERSAL) != 0) | c_ang)
    c_rad = present & (((flags & RADIAL) != 0) | c_trans)
    c_rel = present & (((flags & REL) != 0) | c_rad)
    unset = torch.full((B,), UNSET, dtype=F64, device=dev)
    rel = vel_t - vel_s
    radial = pos_t - pos_s
    dist = torch.sqrt(_dot(radial, radial))
    far = dist != 0.0                                   # coincident centres: only the relative velocity
    short = (dist - rad_t) - rad_s
    short = torch.where(short < 0.0, torch.zeros_like(short), short)
    inv = torch.ones_like(dist) / _safe(dist, ~far)     # Vector3d / double: times the reciprocal
    unit = radial * inv[:, None]
    rad = _dot(rel, unit)
    tv = rel - unit * rad[:, None]
    trans = torch.sqrt(_dot(tv, tv))
    if dynamical:
        if ang_vel_s is None:
            raise ValueError("dynamical orientation needs the shooter's angular velocity")
        omega = _cross(ang_vel_s.to(F64), unit)
        tv2 = tv - omega * dist[:, None]
        ang = torch.sqrt(_dot(tv2, tv2)) / _safe(dist, ~far)
    else:
        ang = trans / _safe(dist, ~far)
    rel_out = torch.where(c_rel[:, None], rel, torch.full_like(rel, UNSET))
    c_rad, c_trans, c_ang = c_rad & far, c_trans & far, c_ang & far
    return (rel_out, torch.where(c_rad, rad, unset), torch.where(c_rad, short, unset),
            torch.where(c_trans, trans, unset), torch.where(c_ang, ang, unset))


def from_park(park, src: torch.Tensor, dst: torch.Tensor) -> tuple:
    """The pairs' tensors from a batched park with `pos`, `vel`, `radius` and `ids` rows (VecModesPark, VecOrientPark,
    exact_ballpark.vec_combat.VecCombatPark): src and dst are rows (-1: missing). Returns (pos_s, vel_s, rad_s, pos_d,
    vel_d, rad_d, same, present). The park's `exists` mask, where it has one, marks a removed row as missing; the
    same ball is the same row (index work only)."""
    src, dst = src.to(I64), dst.to(I64)
    n = park.pos.shape[0]
    ok = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    s, d = torch.where(ok, src, 0), torch.where(ok, dst, 0)
    exists = getattr(park, "exists", None)
    if exists is not None:
        ok = ok & exists[s] & exists[d]
    return (park.pos[s], park.vel[s], park.radius[s], park.pos[d], park.vel[d], park.radius[d], ok & (s == d), ok)

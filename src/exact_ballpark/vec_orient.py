# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's turning batched: the orientation of B balls (with dynamical orientation, or the yaw, pitch and roll
without it) and the setters that change a ship's values in flight, around GOTO and STOP, as float64 tensors on the CPU
or the GPU.

It must reproduce `exact_ballpark.full.FullBallpark` bit for bit (balls added with the Python AddBall, free, not massive,
so they never meet; a client's view), which is itself held bit for bit to Destiny's own code compiled verbatim
(tests/test_movement_full_oracle.py, tests/test_movement_turn_oracle.py), so it is exact against Destiny (Carbon's
`destiny` at 114e89fa, MIT, (c) CCP Games). Checked by
tests/test_movement_vec_orient.py on generated balls, on the CPU and the GPU. `VecOrientPark` extends vec.py's
`VecBallpark` (B independent balls, one per row) without changing it. Rules, by rule id:

- MOV-728, dynamical orientation (`dynamical_orientation=True`; Destiny's g_useDynamicalOrientation, off by default):
  above 1 % of the maximum acceleration the heading torque (`GotoTorqueBreaks`, Ballpark.cpp:1616) and the roll torque
  (`GotoRollBreaks`, 1751) are searched, the rotation a tick on is predicted (`ApplyTorque`, 1936), and the thrust is
  laid along that heading, scaled by cos^2 of its angle to the wanted thrust (zero at 90 degrees or more); below it the
  thrust is kept and only the roll is steered; STOP's vertical damping is skipped; each tick ends with ApplyTorque's
  rotation and angular velocity (EvolveBehaviorForBall and Evolve, Ballpark.cpp:840-890, 421). All of it in C++ float
  arithmetic, each float operation rounded to binary32, with the C library's float functions from
  `exact_ballpark.vec_libm` (expf, logf, acosf, atan2f, sinf, cosf; pow for kappa = tf^n), exact on every device.
  AddBall and SetBallVelocity snap the rotation to the velocity (`SnapOrientation`, Ball.cpp:290).
- MOV-729, the orientation without it (`CalculateYawPitchRoll`, Ball.cpp:335; `Modulo2pi`, 777): yaw and pitch of the
  velocity mixed with the goto direction (of the goto direction at rest), smoothed by the mass, the roll driven by the
  yaw's change; AddBall and SetBallVelocity snap it. In double arithmetic, with the C library's atan2, asin and sin:
  with `ypr="host"` (the default) evaluated on the host (Python's math, the C library's own functions: exact by
  construction, at the cost of two round trips a tick), with `ypr="cuda"` on the device by glibc's own algorithms
  transcribed bit for bit (`exact_ballpark.vec_ypr_cuda`, held to the C library and to the host path by
  tests/test_movement_vec_ypr_cuda.py), so a tick has no host round trip and can be captured as a CUDA graph
  (`vec_ypr_cuda.YprTickGraph`); everything else on the device either way.
- MOV-711, MOV-713, the setters with every side effect Destiny applies (Ballpark.cpp:4652-5025): SetBallMass and
  SetBallAgility refresh the one-tick decay exp(-k dt / (mass agility)) (MOV-704; glibc's exp on the device through
  `exact_ballpark.vec_exp`), SetMaxSpeed and SetBallRadius store the value (Destiny also re-files the ball in its
  partition, which this module, like vec.py, does not have; `refiled` counts the calls), SetSpeedFraction clamps to
  [0, 1] and refuses a non-finite value, SetMaxAngularSpeed and SetBallAngularAgility (the turning values),
  SetBallAngularVelocity and SetBallRotation (normalised above 1e-20, the rotation stopped), SetBallVelocity (and the
  snap); each refuses the values Destiny refuses (negative, zero or NaN, as coded). The commands are GotoPoint,
  GotoDirection and Stop (MOV-708, MOV-709).

**Exactness.** Only correctly rounded IEEE operations, in the reference's order: +, -, *, sqrt, division with a tensor
divisor, comparisons and integer steps; float operations as a double rounded to float (exact for +, -, *, / and sqrt
of floats, 53 >= 2 x 24 + 2); no fused multiply-add outside vec_exp's and vec_libm's emulations, no compilation, no
float reductions (the torque search's first passing step is found by an integer count). The torque searches run every
candidate step at once: GotoTorqueBreaks and GotoRollBreaks try n1 = 1, 2, ... 99 accelerating ticks and keep the
first whose torque is below the maximum, each with kappa1 = tf^n1 formed by repeated float products, as Destiny's loop
forms it; here the 99 products are a table made once a tick and every candidate is computed, then the first that
passes is taken. The outer correction (at most 5 rounds) runs as masked rounds.

**Two implementations of dynamical orientation** (`kernel=`): "torch" (the default, above: the exactness reference)
and "cuda", one CUDA kernel a tick running the scalar algorithm a thread a ship, each ship leaving its torque searches
when its own rounds end (`exact_ballpark.vec_orient_cuda`; every rounding pinned, held to this module and to the
reference bit for bit by tests/test_movement_vec_orient_cuda.py). The kernel also commits ApplyTorque's rotation and
angular velocity, which the torch path does at the end of `evolve`; everything else in the tick is the same code.

(1 November 2026): OnTick's schedule for every world at its own clock (`VecOrientPark.on_tick`,
FullBallpark.on_tick's) and its per-world mask in every park's tick and CUDA graph (`masked_tick`, `GraphMask`).

Not here: modes other than GOTO and STOP (follow, orbit, warp), missiles, formations, collisions and
the partition, OnTick's time stamps, the server's view.
"""

from __future__ import annotations

import math

import torch

from . import vec_libm as L
from .ref import GOTO, STOP, STOP_DAMP
from .vec import VecBallpark
from .vec_exp import exp as vexp

F64, I64 = torch.float64, torch.int64
HALF_PI, TWO_PI = 1.57079632679489655, 6.2831853071795862   # Ball.cpp:13-15
ROLL_SPEED_ACCELERATION, ROLL_SPEED_DECAY = -3.0, -0.9      # Ballpark.cpp:115-118 (MOV-729)
ROLL_ACCELERATION, ROLL_DECAY = 1.0, -0.5
TORQUE_STEPS, TORQUE_ROUNDS = 100, 5                        # GotoTorqueBreaks' and GotoRollBreaks' loops
KERNELS = ("torch", "cuda")                                 # dynamical orientation's two implementations
YPR_PATHS = ("host", "cuda")                                # where the yaw, pitch and roll call atan2, asin, sin
f32 = L.f32


# ======================================================================== C++ float vectors, quaternions, matrices

def _t(v, like: torch.Tensor) -> torch.Tensor:
    """A Python float as a tensor shaped like `like`; a tensor unchanged."""
    return v if torch.is_tensor(v) else torch.full_like(like, v)


def _fdiv(a, b) -> torch.Tensor:
    """float / float, IEEE (an infinity or NaN for a zero divisor); the divisor a tensor."""
    return f32(a / b) if torch.is_tensor(b) else f32(a / _t(b, a))


def _fmax(a, b):
    """std::max: (a < b) ? b : a."""
    return torch.where(a < b, b, a)


def _fmin(a, b):
    """std::min: (b < a) ? b : a."""
    return torch.where(b < a, b, a)


def v3_mul(a: tuple, f) -> tuple:
    return (f32(a[0] * f), f32(a[1] * f), f32(a[2] * f))


def v3_add(a: tuple, b: tuple) -> tuple:
    return (f32(a[0] + b[0]), f32(a[1] + b[1]), f32(a[2] + b[2]))


def v3_sub(a: tuple, b: tuple) -> tuple:
    return (f32(a[0] - b[0]), f32(a[1] - b[1]), f32(a[2] - b[2]))


def v3_div(a: tuple, f) -> tuple:
    """Vector3 / float: multiplied by 1.0f / f."""
    return v3_mul(a, _fdiv(_t(1.0, a[0]), f))


def v3_dot(a: tuple, b: tuple) -> torch.Tensor:
    return f32(f32(f32(a[0] * b[0]) + f32(a[1] * b[1])) + f32(a[2] * b[2]))


def v3_len(a: tuple) -> torch.Tensor:
    return f32(torch.sqrt(v3_dot(a, a)))


def v3_cross(a: tuple, b: tuple) -> tuple:
    return (f32(f32(a[1] * b[2]) - f32(a[2] * b[1])), f32(f32(a[2] * b[0]) - f32(a[0] * b[2])),
            f32(f32(a[0] * b[1]) - f32(a[1] * b[0])))


def v3_normalize(v: tuple) -> tuple:
    """Normalize(Vector3): by the largest magnitude first, then by the length (a zero length left as it is)."""
    mx = _fmax(_fmax(v[0].abs(), v[1].abs()), v[2].abs())
    minified = v3_div(v, torch.where(mx != 0.0, mx, 1.0))
    length = v3_len(minified)
    return v3_mul(minified, torch.where(length != 0.0, _fdiv(_t(1.0, length), length), length))


def q_mul(q: tuple, o: tuple) -> tuple:
    """Quaternion::operator*(other)."""
    x, y, z, w = q
    ox, oy, oz, ow = o
    return (f32(f32(f32(f32(ow * x) + f32(ox * w)) + f32(oy * z)) - f32(oz * y)),
            f32(f32(f32(f32(ow * y) - f32(ox * z)) + f32(oy * w)) + f32(oz * x)),
            f32(f32(f32(f32(ow * z) + f32(ox * y)) - f32(oy * x)) + f32(oz * w)),
            f32(f32(f32(f32(ow * w) - f32(ox * x)) - f32(oy * y)) - f32(oz * z)))


def q_len(q: tuple) -> torch.Tensor:
    return f32(torch.sqrt(f32(f32(f32(f32(q[0] * q[0]) + f32(q[1] * q[1])) + f32(q[2] * q[2])) + f32(q[3] * q[3]))))


def q_div(q: tuple, f) -> tuple:
    d = _fdiv(_t(1.0, f), f)
    return (f32(q[0] * d), f32(q[1] * d), f32(q[2] * d), f32(q[3] * d))


def q_conj(q: tuple) -> tuple:
    return (-q[0], -q[1], -q[2], q[3])


def q_rotation(axis: tuple, angle: torch.Tensor) -> tuple:
    """RotationQuaternion(axis, angle): sinf and cosf of the half angle."""
    t = v3_normalize(axis)
    h = _fdiv(angle, _t(2.0, angle))
    s = L.sinf(h)
    return (f32(s * t[0]), f32(s * t[1]), f32(s * t[2]), L.cosf(h))


def m_rotation_q(q: tuple) -> tuple:
    """RotationMatrix(Quaternion), rows m[0] to m[3] (the last column and row constant)."""
    x, y, z, w = q
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    return ((f32(1.0 - f32(2.0 * f32(f32(y * y) + f32(z * z)))), f32(2.0 * f32(f32(x * y) + f32(z * w))),
             f32(2.0 * f32(f32(x * z) - f32(y * w))), zero),
            (f32(2.0 * f32(f32(x * y) - f32(z * w))), f32(1.0 - f32(2.0 * f32(f32(x * x) + f32(z * z)))),
             f32(2.0 * f32(f32(y * z) + f32(x * w))), zero),
            (f32(2.0 * f32(f32(x * z) + f32(y * w))), f32(2.0 * f32(f32(y * z) - f32(x * w))),
             f32(1.0 - f32(2.0 * f32(f32(x * x) + f32(y * y)))), zero),
            (zero, zero, zero, one))


def transform_coord(v: tuple, m: tuple) -> tuple:
    """TransformCoord: the row vector times the matrix, divided by the homogeneous norm (every sum kept, so the signs
    of zero are Destiny's)."""
    x, y, z = (_t(c, m[0][0]) for c in v)
    norm = f32(f32(f32(f32(m[0][3] * x) + f32(m[1][3] * y)) + f32(m[2][3] * z)) + m[3][3])
    out = tuple(_fdiv(f32(f32(f32(f32(x * m[0][j]) + f32(y * m[1][j])) + f32(z * m[2][j])) + m[3][j]), norm)
                for j in range(3))
    return tuple(torch.where(norm != 0.0, c, 0.0) for c in out)


def _cols(t: torch.Tensor) -> tuple:
    return tuple(t.unbind(1))


def _stack(v: tuple) -> torch.Tensor:
    return torch.stack(v, dim=1)


def _where(mask: torch.Tensor, a: tuple, b: tuple) -> tuple:
    return tuple(torch.where(mask, x, y) for x, y in zip(a, b))


def _to_int32(x: torch.Tensor) -> torch.Tensor:
    """int(x) of a float on aarch64 (fcvtzs): truncation, saturating, NaN to 0 (as int64)."""
    x = torch.where(x != x, 0.0, x)
    return torch.trunc(torch.clamp(x, -2147483648.0, 2147483647.0)).to(I64)


# ============================================================================= OnTick's per-world mask

def masked_tick(park, active: torch.Tensor, tick, in_place: bool) -> None:
    """One tick of the rows in `active` [n] (OnTick's worlds with a tick, each world's rows together), every other row
    left as it was, every per-row tensor included: `tick()`, the park's own unmasked
    tick (unchanged), runs for every row, then each per-row tensor (`park.row_tensors()`) of the rows outside `active`
    is given back its value from before the tick. Exact by construction: a park's tick reads and writes only rows of
    each row's own world (a follower's, missile's or formation member's target, a mushroom's members and a collision's
    candidates are rows of its world; VecOrientPark's rows are independent), so the rows in `active` hold the unmasked
    tick's values and the others their own, each a selection (torch.where) of the one or the other, bit for bit; no
    host synchronisation. `in_place` (the kernel parks, whose ticks write their tensors in place, so that CUDA graphs
    and the layer's views keep their buffers; and their ticks on tensors): the values before are copies and the
    put-back writes into the park's tensors as the tick left them (refused if two of them share storage); otherwise
    (VecOrientPark's tick, which rebinds every tensor it changes and writes none in place) the values before are the
    tensors themselves, and each tensor the tick rebound is rebound to the selection. The park's `_ticked` holds (its
    tick count, the mask) for `events()`."""
    act = active.to(device=park.dev, dtype=torch.bool).reshape(park.n)
    names = park.row_tensors()
    if in_place:
        before = {k: getattr(park, k).clone() for k in names}
    else:
        before = {k: getattr(park, k) for k in names}
    tick()
    if in_place:
        ptrs = [getattr(park, k).untyped_storage().data_ptr() for k in names]
        if len(set(ptrs)) != len(ptrs):
            raise RuntimeError("a masked tick in place needs every per-row tensor in its own storage")
    for k, old in before.items():
        cur = getattr(park, k)
        if not in_place and cur is old:
            continue
        if cur.shape != old.shape or cur.dtype != old.dtype:
            raise RuntimeError(f"the tick changed {k}'s shape or dtype ({tuple(old.shape)} {old.dtype} to "
                               f"{tuple(cur.shape)} {cur.dtype})")
        sel = act.reshape((park.n,) + (1,) * (cur.dim() - 1))
        if in_place:
            cur.copy_(torch.where(sel, cur, old))
        else:
            setattr(park, k, torch.where(sel, cur, old))
    park._ticked = (park.current_time, act)


class GraphMask:
    """OnTick's per-world mask inside a CUDA graph: `mask` [n], a device buffer written before each
    replay (`set`), and a copy of each of the graph's state buffers (`buf`, the park's per-row tensors the captured
    kernels update in place); the capture saves the buffers before the tick (`save`) and gives the rows outside the
    mask their saved values after it (`put_back`), a selection as `masked_tick` makes in place; `mark` records the
    mask for `events()`. No allocation at a replay, no host synchronisation."""

    def __init__(self, buf: dict, n: int, dev):
        bad = [k for k, b in buf.items() if b.dim() < 1 or b.shape[0] != n]
        if bad:
            raise ValueError(f"OnTick's mask in a graph needs per-row buffers, not {bad}")
        self.buf, self.n = buf, n
        self.mask = torch.ones(n, dtype=torch.bool, device=dev)
        self.saved = {k: torch.empty_like(b) for k, b in buf.items()}

    def save(self) -> None:
        for k, b in self.buf.items():
            self.saved[k].copy_(b)

    def put_back(self) -> None:
        for k, b in self.buf.items():
            sel = self.mask.reshape((self.n,) + (1,) * (b.dim() - 1))
            b.copy_(torch.where(sel, b, self.saved[k]))

    def set(self, active: torch.Tensor | None) -> None:
        if active is None:
            self.mask.fill_(True)
        else:
            self.mask.copy_(active.to(device=self.mask.device, dtype=torch.bool).reshape(self.n))

    def mark(self, park) -> None:
        park._ticked = (park.current_time, self.mask.clone())


# ================================================================================================ the batched park

class VecOrientPark(VecBallpark):
    """B independent balls, one per row, with Destiny's orientation (dynamical or not) and setters."""

    def __init__(self, n: int, friction: float = 1.0e6, tick_interval_ms: int = 1000, device="cpu",
                 dynamical_orientation: bool = False, kernel: str = "torch", ypr: str = "host"):
        super().__init__(n, friction, tick_interval_ms, device)
        self.dynamical_orientation = bool(dynamical_orientation)
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, not {kernel!r}")
        if kernel == "cuda" and not (self.dynamical_orientation and self.dev.type == "cuda"):
            raise ValueError('kernel="cuda" is dynamical orientation on a CUDA device')
        if ypr not in YPR_PATHS:
            raise ValueError(f"ypr must be one of {YPR_PATHS}, not {ypr!r}")
        if ypr == "cuda" and (self.dynamical_orientation or self.dev.type != "cuda"):
            raise ValueError('ypr="cuda" is the yaw, pitch and roll without dynamical orientation on a CUDA device')
        self.kernel = kernel
        self.ypr = ypr
        self.fdt = float(L.f32(torch.tensor([self.dt], dtype=F64)).item())   # float(dt)
        self.ffriction = float(L.f32(torch.tensor([self.friction], dtype=F64)).item())
        z = lambda *s: torch.zeros(*s, dtype=F64, device=self.dev)
        self.radius, self.max_ang_vel, self.rot_agility, self.rot_roll = z(n), z(n), z(n), z(n)
        self.old_pos, self.old_vel, self.last_g, self.ang_vel, self.old_ang_vel, self.torque = (
            z(n, 3), z(n, 3), z(n, 3), z(n, 3), z(n, 3), z(n, 3))
        self.rot, self.old_rot = z(n, 4), z(n, 4)
        self.rot[:, 3] = 1.0
        self.old_rot[:, 3] = 1.0
        self.yaw, self.old_yaw, self.pitch, self.old_pitch = z(n), z(n), z(n), z(n)
        self.roll, self.old_roll, self.roll_speed, self.old_roll_speed, self.yaw_delta = z(n), z(n), z(n), z(n), z(n)
        self.refiled = torch.zeros(n, dtype=I64, device=self.dev)            # SetMaxSpeed / SetBallRadius calls
        ndt = torch.arange(1, TORQUE_STEPS, dtype=F64, device=self.dev) * self.dt
        self._ndt = f32(ndt)                                                 # float(n1 dt), n1 = 1 .. 99
        self.current_time = 0

    # ---- loading: the Python AddBall (PyAddBall, Thunkers.cpp:150) of free, non-massive balls ----

    def load(self, rows: list[dict]) -> None:
        """Every ball as the Python AddBall adds a new free ball: AddBall's clamps (MOV-711), the decay factor, mode
        STOP; then, with dynamical orientation, the rotation at rest snapped to the velocity (AddDynamicallyOriented-
        Ball, SnapOrientation), without it the yaw, pitch and roll snapped (AddOldStyleOrientedBall). Keys: mass,
        radius, max_vel, pos, vel, agility, speed_fraction, max_ang_vel (default 1), rot_agility (default 0.9)."""
        t = lambda xs: torch.tensor(xs, dtype=F64, device=self.dev)
        ff = lambda xs: f32(t([float(x) for x in xs]))
        self.mass = t([float(r["mass"]) for r in rows])
        self.mass = torch.where(self.mass < 0.0, 0.0, self.mass)
        self.radius = ff(r["radius"] for r in rows)
        self.radius = torch.where(self.radius < 0.0, 0.0, self.radius)
        self.max_vel = ff(r["max_vel"] for r in rows)
        self.max_vel = torch.where(self.max_vel < 0.0, 0.0, self.max_vel)
        self.agility = ff(r["agility"] for r in rows)
        self.agility = torch.where(self.agility <= 0.0, 1.0, self.agility)
        self.speed_fraction = ff(r["speed_fraction"] for r in rows)
        self.speed_fraction = torch.where(self.speed_fraction < 0.0, 0.0, self.speed_fraction)
        self.pos = t([[float(c) for c in r["pos"]] for r in rows])
        self.vel = t([[float(c) for c in r["vel"]] for r in rows])
        self.old_pos = self.pos - self.vel * self.dt
        self.old_vel = self.vel.clone()
        self.goto = torch.zeros_like(self.pos)
        self.mode = torch.full((len(rows),), STOP, dtype=I64, device=self.dev)
        self.last_g = torch.zeros_like(self.pos)
        self.time_factor = self._decay(self.mass * self.agility)
        every = torch.ones(len(rows), dtype=torch.bool, device=self.dev)
        z = torch.zeros(len(rows), dtype=F64, device=self.dev)
        if self.dynamical_orientation:                  # only AddDynamicallyOrientedBall takes the turning values
            self.max_ang_vel = ff(r.get("max_ang_vel", 1.0) for r in rows)
            self.rot_agility = ff(r.get("rot_agility", 0.9) for r in rows)
            self.max_ang_vel = torch.where(self.max_ang_vel < 0.0, 0.0, self.max_ang_vel)
            self.rot_agility = torch.where(self.rot_agility <= 0.0, 1.0, self.rot_agility)
            rot = (z, z, z, z + 1.0)
            w = (z, z, z)
            back = q_rotation(w, f32(-self.fdt * v3_len(w)))              # mOldRot of a new ball: rot x RotationQ
            self.rot, self.old_rot = _stack(rot), _stack(q_mul(rot, back))  # (angVel, -dt |angVel|)
            self.ang_vel, self.old_ang_vel = _stack(w), _stack(w)
            self._snap_orientation(every)
        else:                                           # Ball::Ball's defaults: 1 rad/s, 0.9 (as floats)
            self.max_ang_vel, self.rot_agility = z + 1.0, f32(z + 0.9)
            self.rot, self.old_rot = _stack((z, z, z, z + 1.0)), _stack((z, z, z, z + 1.0))
            self.ang_vel, self.old_ang_vel = torch.zeros_like(self.pos), torch.zeros_like(self.pos)
            self.yaw, self.old_yaw, self.pitch, self.old_pitch = z.clone(), z.clone(), z.clone(), z.clone()
            self.roll, self.old_roll, self.roll_speed, self.old_roll_speed = z.clone(), z.clone(), z.clone(), z.clone()
            self.yaw_delta = z.clone()
            self._yaw_pitch_roll(every, snap=True)
        self.torque, self.rot_roll = torch.zeros_like(self.pos), z.clone()

    def _decay(self, m: torch.Tensor) -> torch.Tensor:
        """SetBallTimeFactor (MOV-704): 0 if mass x agility <= 0, else exp(-k dt / m) (NaN for a NaN m)."""
        return torch.where(m <= 0.0, 0.0, vexp(torch.full_like(m, -self.friction * self.dt) / m))

    # ---- commands (MOV-708, MOV-709) ----

    def goto_point(self, mask: torch.Tensor, point: torch.Tensor) -> None:
        """GotoPoint: refused for a non-finite point; Stop, the point, a zero speed fraction made 1, then GOTO."""
        ok = mask & torch.isfinite(point).all(dim=1)
        self.goto = torch.where(ok[:, None], point.to(F64), self.goto)
        self.speed_fraction = torch.where(ok & (self.speed_fraction == 0.0), 1.0, self.speed_fraction)
        self.mode = torch.where(ok, GOTO, self.mode)

    def goto_direction(self, mask: torch.Tensor, direction: torch.Tensor) -> None:
        """GotoDirection: a point 1e17 m along the unit direction (refused when not finite)."""
        d = direction.to(F64)
        ok = mask & torch.isfinite(d).all(dim=1)
        norm = torch.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        inv = torch.ones_like(norm) / torch.where(norm == 0.0, 1.0, norm)
        unit = torch.where((norm == 0.0)[:, None], d, d * inv[:, None])
        self.goto_point(ok, self.pos + unit * 1.0e17)

    def stop(self, mask: torch.Tensor) -> None:
        self.mode = torch.where(mask, STOP, self.mode)

    # ---- setters (MOV-711, MOV-713; Ballpark.cpp:4652-5025) ----

    def set_mass(self, mask: torch.Tensor, mass: torch.Tensor) -> None:
        """SetBallMass: ignored unless positive (a NaN is stored, as coded); the decay factor refreshed."""
        m = mass.to(F64)
        ok = mask & ~(m <= 0.0)
        self.mass = torch.where(ok, m, self.mass)
        self.time_factor = torch.where(ok, self._decay(self.mass * self.agility), self.time_factor)

    def set_agility(self, mask: torch.Tensor, agility: torch.Tensor) -> None:
        """SetBallAgility: the float ignored unless positive; the decay factor refreshed."""
        a = f32(agility.to(F64))
        ok = mask & ~(a <= 0.0)
        self.agility = torch.where(ok, a, self.agility)
        self.time_factor = torch.where(ok, self._decay(self.mass * self.agility), self.time_factor)

    def set_max_velocity(self, mask: torch.Tensor, v: torch.Tensor) -> None:
        """SetMaxSpeed: the float ignored if negative; Destiny then re-files the ball in its partition (counted)."""
        v = f32(v.to(F64))
        ok = mask & ~(v < 0.0)
        self.max_vel = torch.where(ok, v, self.max_vel)
        self.refiled += ok.to(I64)

    def set_speed_fraction(self, mask: torch.Tensor, s: torch.Tensor) -> None:
        """SetSpeedFraction: the float refused unless finite, clamped to [0, 1]."""
        s = f32(s.to(F64))
        ok = mask & torch.isfinite(s)
        s = torch.where(s < 0.0, 0.0, torch.where(s > 1.0, 1.0, s))
        self.speed_fraction = torch.where(ok, s, self.speed_fraction)

    def set_radius(self, mask: torch.Tensor, r: torch.Tensor) -> None:
        """SetBallRadius: the float ignored if negative; Destiny then re-files the ball (counted)."""
        r = f32(r.to(F64))
        ok = mask & ~(r < 0.0)
        self.radius = torch.where(ok, r, self.radius)
        self.refiled += ok.to(I64)

    def set_max_angular_speed(self, mask: torch.Tensor, w: torch.Tensor) -> None:
        """SetMaxAngularSpeed: the float ignored if negative."""
        w = f32(w.to(F64))
        ok = mask & ~(w < 0.0)
        self.max_ang_vel = torch.where(ok, w, self.max_ang_vel)

    def set_rot_agility(self, mask: torch.Tensor, a: torch.Tensor) -> None:
        """SetBallAngularAgility: the float ignored unless positive."""
        a = f32(a.to(F64))
        ok = mask & ~(a <= 0.0)
        self.rot_agility = torch.where(ok, a, self.rot_agility)

    def set_angular_velocity(self, mask: torch.Tensor, w: torch.Tensor) -> None:
        """SetBallAngularVelocity: the new and old angular velocity, as floats."""
        w = f32(w.to(F64))
        self.ang_vel = torch.where(mask[:, None], w, self.ang_vel)
        self.old_ang_vel = torch.where(mask[:, None], w, self.old_ang_vel)

    def set_rotation(self, mask: torch.Tensor, q: torch.Tensor) -> None:
        """SetBallRotation: normalised when longer than 1e-20 (otherwise ignored), the rotation stopped."""
        rot = _cols(f32(q.to(F64)))
        length = q_len(rot)
        ok = mask & (length > 1e-20)
        rot = _stack(q_div(rot, torch.where(ok, length, 1.0)))
        self.rot = torch.where(ok[:, None], rot, self.rot)
        self.old_rot = torch.where(ok[:, None], rot, self.old_rot)
        self.ang_vel = torch.where(ok[:, None], 0.0, self.ang_vel)
        self.old_ang_vel = torch.where(ok[:, None], 0.0, self.old_ang_vel)

    def set_velocity(self, mask: torch.Tensor, v: torch.Tensor) -> None:
        """SetBallVelocity (MOV-713): the new and old velocity, then the orientation snapped."""
        v = v.to(F64)
        self.vel = torch.where(mask[:, None], v, self.vel)
        self.old_vel = torch.where(mask[:, None], v, self.old_vel)
        if self.dynamical_orientation:
            self._snap_orientation(mask)
        else:
            self._yaw_pitch_roll(mask, snap=True)

    # ---- the orientation without dynamical orientation (MOV-729) ----

    def _yaw_pitch_roll(self, mask: torch.Tensor, snap: bool) -> None:
        """Ball::CalculateYawPitchRoll for GOTO and STOP: atan2, asin and sin on the host (Python's math is the C
        library's) or, with ypr="cuda", on the device (exact_ballpark.vec_ypr_cuda: glibc's own algorithms, the same
        bits); the rest on the device in the reference's order."""
        old_pitch, old_yaw, old_roll, old_roll_speed = self.pitch, self.yaw, self.roll, self.roll_speed
        mm = self.mass / torch.full_like(self.mass, self.friction)    # a tensor divisor (CUDA: no reciprocal)
        mm = torch.where(mm != 0.0, torch.full_like(mm, 100.0) / torch.where(mm != 0.0, mm, 1.0), 100.0)
        v = self.vel
        vel2 = v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2]
        a = self.goto - self.pos
        if snap:
            direction = torch.where((vel2 == 0.0)[:, None], a, v)
        else:
            norm = torch.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1] + a[:, 2] * a[:, 2])
            inv = torch.ones_like(norm) / torch.where(norm == 0.0, 1.0, norm)
            offset = torch.where((norm == 0.0)[:, None], a, a * inv[:, None])
            mixed = v + offset * (torch.sqrt(vel2) * 0.3)[:, None]
            direction = torch.where((vel2 == 0.0)[:, None], a, mixed)
        r = torch.sqrt(direction[:, 0] * direction[:, 0] + direction[:, 1] * direction[:, 1]
                       + direction[:, 2] * direction[:, 2])
        sine = direction[:, 1] / torch.where(r > 0.0, r, 1.0)
        sine = _fmin(torch.ones_like(sine), _fmax(-torch.ones_like(sine), sine))
        if self.ypr == "cuda":
            from . import vec_ypr_cuda as Y
            yaw = Y.atan2(direction[:, 0].contiguous(), direction[:, 2].contiguous())
            pitch = -Y.asin(sine)
        else:
            host = torch.stack((direction[:, 0], direction[:, 2], sine), dim=1).cpu().tolist()
            yaw = torch.tensor([math.atan2(d0, d2) for d0, d2, _ in host], dtype=F64, device=self.dev)
            pitch = torch.tensor([-math.asin(s) for _, _, s in host], dtype=F64, device=self.dev)
        pitch = torch.where(r > 0.0, pitch, 0.0)
        old_yaw_m, yaw = self._modulo2pi(old_yaw, yaw)
        old_pitch_m, pitch = self._modulo2pi(old_pitch, pitch)
        if snap:
            z = torch.zeros_like(yaw)
            new = dict(yaw=yaw, pitch=pitch, old_yaw=old_yaw_m, old_pitch=old_pitch_m, roll=z, old_roll=z,
                       roll_speed=z, old_roll_speed=z, yaw_delta=z)
        else:
            tmp = torch.ones_like(mm) / (1.0 + mm * 0.5)
            yaw = (old_yaw_m + mm * 0.5 * yaw) * tmp
            pitch = (old_pitch_m + mm * 0.5 * pitch) * tmp
            if self.ypr == "cuda":
                yaw_delta = Y.asin_sin(yaw - old_yaw_m)
            else:
                change = (yaw - old_yaw_m).cpu().tolist()
                yaw_delta = torch.tensor([math.asin(math.sin(x)) for x in change], dtype=F64, device=self.dev)
            roll_speed = self.roll_speed + yaw_delta * ROLL_SPEED_ACCELERATION
            roll_speed = roll_speed + old_roll_speed * ROLL_SPEED_DECAY * self.dt
            roll_speed = _fmax(torch.full_like(mm, -HALF_PI), _fmin(torch.full_like(mm, HALF_PI), roll_speed))
            roll = self.roll + roll_speed * ROLL_ACCELERATION * self.dt
            roll = roll + old_roll * ROLL_DECAY * self.dt
            roll = _fmax(torch.full_like(mm, -HALF_PI), _fmin(torch.full_like(mm, HALF_PI), roll))
            new = dict(yaw=yaw, pitch=pitch, old_yaw=old_yaw_m, old_pitch=old_pitch_m, roll=roll, old_roll=old_roll,
                       roll_speed=roll_speed, old_roll_speed=old_roll_speed, yaw_delta=yaw_delta)
        for k, val in new.items():
            setattr(self, k, torch.where(mask, val, getattr(self, k)))

    @staticmethod
    def _modulo2pi(a: torch.Tensor, b: torch.Tensor) -> tuple:
        """Modulo2pi: both reduced by fmod (exact), then b moved by 2 pi toward a."""
        b, a = torch.fmod(b, TWO_PI), torch.fmod(a, TWO_PI)
        d0, dp, dm = (b - a).abs(), (b + TWO_PI - a).abs(), (b - TWO_PI - a).abs()
        out = torch.where(d0 < dp, torch.where(d0 < dm, b, b - TWO_PI), torch.where(dp < dm, b + TWO_PI, b - TWO_PI))
        return a, out

    # ---- dynamical orientation (MOV-728), in float ----

    def _t_scale(self) -> torch.Tensor:
        """float(friction) / (float(mass) agility rotAgility)."""
        return _fdiv(torch.full_like(self.mass, self.ffriction), f32(f32(f32(self.mass) * self.agility)
                                                                     * self.rot_agility))

    def _tau(self) -> torch.Tensor:
        """(float(mass) agility rotAgility) / float(friction)."""
        return _fdiv(f32(f32(f32(self.mass) * self.agility) * self.rot_agility), torch.full_like(self.mass,
                                                                                                self.ffriction))

    def _snap_orientation(self, mask: torch.Tensor) -> None:
        """Ball::SnapOrientation (Ball.cpp:290): turned to the velocity and rolled upright, the rotation stopped; at a
        speed of 1e-6 m/s or less nothing changes."""
        v = self.vel
        moving = mask & ((v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2]) > 1e-12)
        d = _cols(f32(v))
        len_dir = v3_len(d)
        zero = torch.zeros_like(len_dir)
        vec = v3_cross((zero, zero, zero + 1.0), d)
        angle = L.acosf(_fmax(zero - 1.0, _fmin(zero + 1.0, _fdiv(d[2], len_dir))))
        rot = q_rotation(vec, angle)
        up = transform_coord((0.0, 1.0, 0.0), m_rotation_q(q_conj(rot)))
        up_roll = torch.where(up[2] < L.f32(zero + 0.99), L.atan2f(-up[0], up[1]), 0.0)
        heading = transform_coord((0.0, 0.0, 1.0), m_rotation_q(rot))
        rot = q_mul(rot, q_rotation(heading, up_roll))
        rot = _where(len_dir < 1e-12, (zero, zero, zero, zero + 1.0), rot)
        rot = _stack(rot)
        self.rot = torch.where(moving[:, None], rot, self.rot)
        self.old_rot = torch.where(moving[:, None], rot, self.old_rot)
        self.ang_vel = torch.where(moving[:, None], 0.0, self.ang_vel)
        self.old_ang_vel = torch.where(moving[:, None], 0.0, self.old_ang_vel)

    def _kappas(self, tf: torch.Tensor) -> torch.Tensor:
        """kappa1 for n1 = 1 .. 99: kappa1 = float(kappa1 tf), from 1, as the torque loops form it; shape (B, 99)."""
        out, k = [], torch.ones_like(tf)
        for _ in range(1, TORQUE_STEPS):
            k = f32(k * tf)
            out.append(k)
        return torch.stack(out, dim=1)

    def _n2_alpha2(self, omega0, len_omega0, tf, t_scale, vector: bool) -> tuple:
        """CalculateN2Alpha2 (Ballpark.cpp:1580, 1598): the braking ticks n2, kappa2 = tf^n2 and the braking torque,
        when the turning rate is above 1e-10."""
        live = len_omega0 > 1e-10
        lo = torch.where(live, len_omega0, 1.0)
        factor = _fdiv(self.max_ang_vel, lo)
        n2 = _to_int32(torch.ceil(_fdiv(L.logf(_fdiv(factor, f32(1.0 + factor))), f32(-self.fdt * t_scale))))
        n2 = torch.where(live, n2, 0)
        kappa2 = torch.where(live, f32(L.pow(tf, n2.to(F64))), 1.0)
        g = _fdiv(f32(t_scale * kappa2), f32(1.0 - kappa2))
        if vector:
            alpha2 = v3_mul((-omega0[0], -omega0[1], -omega0[2]), g)
            z = torch.zeros_like(lo)
            alpha2 = _where(live, alpha2, (z, z, z))
        else:
            alpha2 = torch.where(live, f32(-omega0 * g), 0.0)
        return n2, kappa2, alpha2

    def _search(self, target, omega0, tau, kap, n2, kappa2, max_alpha, vector: bool) -> tuple:
        """The inner loop of GotoTorqueBreaks or GotoRollBreaks: the first n1 in 1 .. 99 whose torque is below the
        maximum (every candidate computed, the first found by an integer count), or n1 = 100 with the 99th; returns
        (n1, alpha1, kappa1)."""
        tau2, n2dt = tau[:, None], f32(n2.to(F64) * self.dt)[:, None]
        kappa2b = kappa2[:, None]
        k1 = kap                                                      # (B, 99)
        free = (n2 == 0)[:, None]
        num0 = f32(tau2 * f32(1.0 - k1))
        den0 = f32(tau2 * f32(self._ndt[None, :] - num0))
        num1 = f32(tau2 - _fdiv(f32(f32(k1 * kappa2b) * n2dt), f32(1.0 - kappa2b)))
        den1 = f32(tau2 * f32(self._ndt[None, :] - _fdiv(f32(f32(n2dt * kappa2b) * f32(1.0 - k1)), f32(1.0 - kappa2b))))
        num, den = torch.where(free, num0, num1), torch.where(free, den0, den1)
        if vector:
            om = tuple(c[:, None] for c in omega0)
            tg = tuple(c[:, None] for c in target)
            a1 = v3_div(v3_sub(tg, v3_mul(om, num)), den)
            ok = v3_len(a1) < max_alpha[:, None]
        else:
            a1 = _fdiv(f32(target[:, None] - f32(omega0[:, None] * num)), den)
            ok = a1.abs() < max_alpha[:, None]
        lead = (ok.to(I64).cumsum(dim=1) == 0).to(I64).sum(dim=1)     # failing steps before the first that passes
        n1 = torch.where(lead < TORQUE_STEPS - 1, lead + 1, TORQUE_STEPS)
        pick = torch.clamp(lead, max=TORQUE_STEPS - 2)[:, None]
        if vector:
            alpha1 = tuple(c.gather(1, pick)[:, 0] for c in a1)
        else:
            alpha1 = a1.gather(1, pick)[:, 0]
        return n1, alpha1, k1.gather(1, pick)[:, 0]

    def _torque_rounds(self, target, omega0, len_omega0, tau, tf, kap, t_scale, max_alpha, live, vector: bool):
        """The outer loop (at most 5 rounds): the braking check, the n1 search, n2 again from the angular velocity
        reached; a ball leaves the loop when braking alone suffices or n2 repeats."""
        n2, kappa2, alpha2 = self._n2_alpha2(omega0, len_omega0, tf, t_scale, vector)
        n1 = torch.zeros_like(n2)
        if vector:
            alpha1 = (torch.zeros_like(tau),) * 3
        else:
            alpha1 = torch.zeros_like(tau)
        kappa1 = torch.ones_like(tau)
        done = ~live
        for _ in range(TORQUE_ROUNDS):
            n2dt = f32(n2.to(F64) * self.dt)
            if vector:
                theta2 = v3_add(v3_mul(v3_mul(alpha2, tau), f32(n2dt - f32(tau * f32(1.0 - kappa2)))),
                                v3_mul(v3_mul(omega0, tau), f32(1.0 - kappa2)))
                close = v3_len(v3_sub(target, theta2)) < 1e-3
            else:
                theta2 = f32(f32(f32(tau * alpha2) * f32(n2dt - f32(tau * f32(1.0 - kappa2))))
                             + f32(f32(tau * omega0) * f32(1.0 - kappa2)))
                close = f32(target - theta2).abs() < 1e-3
            brake = ~done & (n2 > 0) & close
            n1 = torch.where(brake, 0, n1)
            done = done | brake
            go = ~done
            if not bool(go.any()):
                break
            m1, a1, k1 = self._search(target, omega0, tau, kap, n2, kappa2, max_alpha, vector)
            n1 = torch.where(go, m1, n1)
            alpha1 = _where(go, a1, alpha1) if vector else torch.where(go, a1, alpha1)
            kappa1 = torch.where(go, k1, kappa1)
            if vector:
                omega1 = v3_add(v3_mul(alpha1, f32(tau * f32(1.0 - kappa1))), v3_mul(omega0, kappa1))
                n2n, kappa2n, alpha2n = self._n2_alpha2(omega1, v3_len(omega1), tf, t_scale, True)
            else:
                omega1 = f32(f32(f32(alpha1 * tau) * f32(1.0 - kappa1)) + f32(omega0 * kappa1))
                n2n, kappa2n, alpha2n = self._n2_alpha2(omega1, omega1.abs(), tf, t_scale, False)
            same = n2n == n2
            n2 = torch.where(go, n2n, n2)
            kappa2 = torch.where(go, kappa2n, kappa2)
            alpha2 = _where(go, alpha2n, alpha2) if vector else torch.where(go, alpha2n, alpha2)
            done = done | (go & same)
        return n1, alpha1, n2, alpha2

    def _goto_torque_breaks(self, direction: tuple, tf, kap, live: torch.Tensor) -> tuple:
        """GotoTorqueBreaks (Ballpark.cpp:1616): the heading torque toward `direction` (a float vector)."""
        t_scale = self._t_scale()
        max_alpha = f32(self.max_ang_vel * t_scale)
        len_dir = v3_len(direction)
        heading = v3_normalize(transform_coord((0.0, 0.0, 1.0), m_rotation_q(_cols(self.rot))))
        rot = v3_div(v3_cross(heading, direction), len_dir)
        rot_len = v3_len(rot)
        one = torch.ones_like(len_dir)
        rot_angle = L.acosf(_fmax(-one, _fmin(one, _fdiv(v3_dot(direction, heading), len_dir))))
        turned = v3_mul(rot, _fdiv(rot_angle, rot_len))
        zero = torch.zeros_like(one)
        flip = (-rot_angle, zero, zero)
        rot = _where(rot_len > 1e-12, turned, _where(rot_angle > 1.0, flip, rot))
        w = _cols(self.ang_vel)
        omega0 = v3_sub(w, v3_mul(heading, v3_dot(heading, w)))
        len_omega0 = v3_len(omega0)
        quiet = (len_omega0 * self.dt < 1e-4) & (rot_angle < 1e-4)
        live = live & ~(max_alpha <= 0.0) & ~(len_dir == 0.0) & ~quiet
        tau = _fdiv(one, t_scale)
        n1, alpha1, n2, alpha2 = self._torque_rounds(rot, omega0, len_omega0, tau, tf, kap, t_scale, max_alpha, live,
                                                     True)
        full = v3_mul(v3_normalize(alpha1), max_alpha)
        out = _where(n1 > 0, _where(n1 > 1, full, alpha1), _where(n2 > 0, alpha2, (zero,) * 3))
        return _where(live, out, (zero,) * 3)

    def _goto_roll_breaks(self, heading_torque: tuple, tf, kap) -> torch.Tensor:
        """GotoRollBreaks (Ballpark.cpp:1751): the roll torque, rolling into strong turns, otherwise upright."""
        t_scale = self._t_scale()
        max_alpha = f32(t_scale * self.max_ang_vel)
        mat = m_rotation_q(q_conj(_cols(self.rot)))
        ship_omega = transform_coord(_cols(self.ang_vel), mat)
        omega_roll = ship_omega[2]
        len_torque = v3_len(heading_torque)
        zero = torch.zeros_like(omega_roll)
        ship_torque = transform_coord(heading_torque, mat)
        flat = (ship_omega[0], ship_omega[1], zero)
        strong = len_torque > 0.05 * max_alpha
        into = strong & ((v3_dot(flat, ship_torque) >= 0.0)
                         | (v3_len(flat) < f32(L.f32(zero + 0.01) * self.max_ang_vel)))
        turn_roll = torch.where(into, L.atan2f(-ship_torque[1], -ship_torque[0]), 0.0)
        weight = torch.where(into, _fdiv(len_torque, t_scale), 0.0)
        weight = torch.where(weight > 1.0, 1.0, weight)
        up = transform_coord((0.0, 1.0, 0.0), mat)
        up_roll = torch.where(up[2] < L.f32(zero + 0.95), L.atan2f(-up[0], up[1]), 0.0)
        desired = torch.where(weight < L.f32(zero + 0.05), up_roll, f32(weight * turn_roll))
        live = ~(max_alpha <= 0.0) & ~((omega_roll.abs() < 1e-4) & (desired.abs() < 1e-4))
        tau = _fdiv(torch.ones_like(t_scale), t_scale)
        n1, alpha1, n2, alpha2 = self._torque_rounds(desired, omega_roll, omega_roll.abs(), tau, tf, kap, t_scale,
                                                     max_alpha, live, False)
        full = f32(torch.where(alpha1 >= 0.0, 1.0, -1.0) * max_alpha)
        out = torch.where(n1 > 0, torch.where(n1 > 1, full, alpha1), torch.where(n2 > 0, alpha2, 0.0))
        return torch.where(live, out, 0.0)

    def _apply_torque(self, torque: tuple, roll: torch.Tensor, tau: torch.Tensor) -> tuple:
        """ApplyTorque (Ballpark.cpp:1936) over one tick (fraction 1): heading and roll apart; (rotation, angular
        velocity) as float tuples."""
        rotation, angvel = _cols(self.rot), _cols(self.ang_vel)
        fdt = self.fdt
        one = torch.ones_like(tau)
        tf = L.expf(_fdiv(f32(one * (-fdt * 1.0)), tau))
        heading = transform_coord((0.0, 0.0, 1.0), m_rotation_q(rotation))
        omega_roll = v3_dot(angvel, heading)
        omega_heading = v3_sub(angvel, v3_mul(heading, omega_roll))
        span = f32(f32(one * (1.0 * fdt)) - f32(tau * f32(1.0 - tf)))
        rot_roll = f32(tau * f32(f32(roll * span) + f32(omega_roll * f32(1.0 - tf))))
        rot_heading = v3_mul(v3_add(v3_mul(torque, span), v3_mul(omega_heading, f32(1.0 - tf))), tau)
        angle_heading = v3_len(rot_heading)
        apply = angle_heading > 1e-10
        zero = torch.zeros_like(one)
        ident = (zero, zero, zero, one)
        quat_heading = _where(apply, q_rotation(rot_heading, angle_heading), ident)
        heading_rotated = _where(apply, transform_coord(heading, m_rotation_q(quat_heading)), heading)
        angvel = v3_add(v3_mul(torque, f32(tau * f32(1.0 - tf))), v3_mul(omega_heading, tf))
        angvel = v3_add(angvel, v3_mul(heading_rotated, f32(f32(tau * roll) - f32(f32(f32(tau * roll) - omega_roll)
                                                                                   * tf))))
        rotation = _where(apply, q_mul(rotation, quat_heading), rotation)
        rotation = q_mul(rotation, q_rotation(heading_rotated, rot_roll))
        norm = q_len(rotation)
        rotation = _where(norm > 0.0, q_div(rotation, torch.where(norm > 0.0, norm, 1.0)), rotation)
        return rotation, angvel

    # ---- one tick (Ballpark::Evolve, TIME-706, TIME-707) ----

    def evolve(self, active: torch.Tensor | None = None) -> None:
        """Behaviours (GOTO's thrust, STOP's damping without dynamical orientation, the orientation's torque and the
        thrust along the heading with it), then each ball's end-of-tick state (and rotation), then the commit (and the
        yaw, pitch and roll without dynamical orientation). `active` [n] (OnTick's per-world mask): only
        those rows tick, every other row keeps every per-row tensor bit for bit (`masked_tick`)."""
        if active is not None:
            masked_tick(self, active, lambda: VecOrientPark.evolve(self), in_place=False)
            return
        goto = self.mode == GOTO
        a = torch.where(goto[:, None], self._goto_thrust(), torch.zeros_like(self.pos))
        kernel = self.dynamical_orientation and self.kernel == "cuda"
        if not self.dynamical_orientation:
            vy = self.vel[:, 1]
            self.vel = self.vel.clone()
            self.vel[:, 1] = torch.where(self.mode == STOP, (vy - 0.07 * vy) * STOP_DAMP, vy)
        elif kernel:                                    # one launch: _orient and the rotation's commit below
            from .vec_orient_cuda import orient_park
            a = orient_park(self, a)
        else:
            a = self._orient(a)
        self.last_g = a
        a = a + torch.zeros_like(a)                     # mLastG + mLastC (zero without collisions): -0 becomes +0
        m = (self.mass * self.agility)[:, None]
        k, t, tf = self.friction, self.dt, self.time_factor[:, None]
        k2 = k * k
        ook2, ook = 1.0 / k2, 1.0 / k
        p, v = self.pos, self.vel
        exact = (m * (m * a * (tf - 1.0) + k * (a * t + v - tf * v)) + p * k2) * ook2
        taylor = (m * (a * (-k * t) + k * (a * t + v * k * (1.0 / m) * t)) + p * k2) * ook2
        small = (k < 1e-10 * m * t)
        new_pos = torch.where(small, taylor, exact)
        new_vel = (m * a - (m * a - v * k) * tf) * ook
        if self.dynamical_orientation and not kernel:   # ApplyTorque a tick on: as _orient predicted it
            rot, w = self._next_rotation
            rot, w = _stack(rot), _stack(w)
            rot_ok = torch.isfinite(f32(f32(f32(f32(rot[:, 0] * rot[:, 0]) + f32(rot[:, 1] * rot[:, 1]))
                                            + f32(rot[:, 2] * rot[:, 2])) + f32(rot[:, 3] * rot[:, 3])))
            w_ok = torch.isfinite(v3_dot(_cols(w), _cols(w)))
            self.rot, self.old_rot = (torch.where(rot_ok[:, None], rot, self.rot),
                                      torch.where(rot_ok[:, None], self.rot, rot))
            self.ang_vel, self.old_ang_vel = (torch.where(w_ok[:, None], w, self.ang_vel),
                                              torch.where(w_ok[:, None], self.ang_vel, w))
        self.old_pos, self.pos = self.pos, new_pos
        self.old_vel, self.vel = self.vel, new_vel
        if not self.dynamical_orientation:
            self._yaw_pitch_roll(torch.ones_like(goto), snap=False)
        self.current_time += 1

    # ---- OnTick (Ballpark::OnTick, TIME-702 to TIME-705), every world at its own clock ----

    _ROW_EXCLUDED = ("_ndt", "time", "first_time")      # not ball state: the torque table, OnTick's clocks

    def row_tensors(self) -> list:
        """The names of the park's per-row tensors (first dimension the rows, n), which a tick may change and a masked
        tick puts back (`masked_tick`); `_ROW_EXCLUDED` names the tensors that are not a ball's state."""
        return [k for k, v in vars(self).items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == self.n
                and k not in self._ROW_EXCLUDED]

    def _on_tick_master(self) -> bool:
        """Whether OnTick cuts a backlog of more than five intervals to one (the master's resynchronisation, TIME-703):
        a client's view here (VecCombatPark: its `master`)."""
        return False

    def _bring_out_on_tick(self) -> None:
        """OnTick's BringOutTheDeads (Ballpark.cpp:231; TIME-713) before the interval check: no ball is ever removed
        here (VecCombatPark: its moribund balls due)."""

    def _world_split(self, x: torch.Tensor) -> torch.Tensor:
        """Whether some world's rows hold different values of x [n] (a 0-dim bool tensor, no host synchronisation):
        never here, where every row is its own world (VecModesPark: by its `world`)."""
        return torch.zeros((), dtype=torch.bool, device=self.dev)

    def on_tick(self, sim_time, tick=None, after=None) -> list:
        """Ballpark::OnTick (Ballpark.cpp:219-330; TIME-702 to TIME-705; exact_ballpark.full.FullBallpark.on_tick) for
        every world at once, each at its own simulation time: `sim_time` [n], a row's world's time in
        100 ns units (a world's rows share it, as they share its clock: refused otherwise). BringOutTheDeads first
        (`_bring_out_on_tick`); less than one whole interval (the tick interval x 10,000) since the world's `time`:
        nothing, its time kept; on a master's park (`_on_tick_master`) a backlog of more than five intervals cut to one;
        a world's first call evolves once (Destiny's DoPreTick and DoPostTick are the server's: none here); every other
        world runs its whole intervals, the worlds with an interval left ticking together under OnTick's mask
        (`tick(rows)`, by default `evolve(active=rows)`; a tick of every row is the unmasked tick, `tick(None)`), the
        remainder carried in its time. `time` [n] (Be::Time mTime) and `first_time` [n] (mFirstTime) are made at the
        first call, 0 and true as Ballpark's; a world's tick count, where the park keeps one, is its `clock`
        (VecModesPark). `after(rows)` is called after each tick (the caller's events). Returns the masks of the ticks
        run, in order. Host synchronisations: the round count and whether a world's rows disagree; a round's
        all-rows test."""
        from .full import MAX_CATCH_UP_TICKS
        n, d = self.n, self.dev
        if getattr(self, "time", None) is None:
            self.time = torch.zeros(n, dtype=I64, device=d)
            self.first_time = torch.ones(n, dtype=torch.bool, device=d)
        sim = torch.as_tensor(sim_time, dtype=I64).to(d).reshape(n)
        interval = int(round(self.dt * 1000.0)) * 10000                   # mTickInterval x 10,000 (dt = ms x 0.001)
        self._bring_out_on_tick()
        last = sim - self.time
        go = ~(last < interval)                                         # too early: nothing (Ballpark.cpp:235)
        k = torch.where(go, torch.div(last, interval, rounding_mode="floor"), 0)
        if self._on_tick_master():                                      # the master's resynchronisation (246-257)
            resync = go & (k > MAX_CATCH_UP_TICKS)
            last = torch.where(resync, interval, last)
            k = torch.where(resync, 1, k)
        first = go & self.first_time                                    # the first call: Evolve once (260-273)
        rest = go & ~first
        top = torch.maximum(torch.where(rest, k, 0).max(), first.any().to(I64)) if n else torch.zeros((), dtype=I64)
        top, split = torch.stack((top.to(I64), self._world_split(sim).to(I64))).tolist() if n else (0, 0)
        if split:
            raise ValueError("on_tick: a world's rows hold different simulation times")
        tick = tick or (lambda rows: self.evolve(active=rows))
        masks = []
        for j in range(top):                                            # each whole interval (275-322)
            m = (rest & (k > j)) | (first & (j == 0))
            tick(None if bool(m.all()) else m)
            if after is not None:
                after(m)
            masks.append(m)
        self.first_time = self.first_time & ~first
        self.time = torch.where(go, sim - torch.where(rest, last - k * interval, 0), self.time)
        return masks

    def _orient(self, a: torch.Tensor) -> torch.Tensor:
        """EvolveBehaviorForBall's dynamical-orientation branch (Ballpark.cpp:840-890; MOV-728)."""
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
        accel = f32(torch.sqrt(ax * ax + ay * ay + az * az))
        max_accel = _fdiv(f32(f32(torch.full_like(ax, self.ffriction) * self.speed_fraction) * self.max_vel),
                          f32(f32(self.mass) * self.agility))
        thrust = accel > 1e-2 * max_accel
        t_scale = self._t_scale()
        tf = L.expf(f32(-self.fdt * t_scale))
        kap = self._kappas(tf)
        a32 = _cols(f32(a))
        torque = self._goto_torque_breaks(a32, tf, kap, thrust)
        torque = _where(thrust, torque, (torch.zeros_like(ax),) * 3)
        self.torque = _stack(torque)
        self.rot_roll = self._goto_roll_breaks(torque, tf, kap)
        rotation, angvel = self._apply_torque(torque, self.rot_roll, self._tau())
        self._next_rotation = (rotation, angvel)        # the same call ends the tick (same inputs)
        heading = transform_coord((0.0, 0.0, 1.0), m_rotation_q(rotation))
        cosangle = _fmin(torch.ones_like(accel), _fdiv(v3_dot(heading, a32), accel))
        h = _stack(v3_mul(heading, f32(f32(cosangle * cosangle) * accel)))
        steered = torch.where((cosangle <= 0.0)[:, None], 0.0, h)
        return torch.where(thrust[:, None], steered, a)

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's flight batched on the GPU: GOTO, STOP, FOLLOW, ORBIT (both laws) and WARP for B balls, with the modes'
transitions, the setters and dynamical orientation, one CUDA kernel a tick (exact_ballpark.vec_modes_cuda), and the
same tick on tensors on the CPU (`evolve_torch`; `evolve` takes it on any device but a CUDA one).

It must reproduce the scalar reference bit for bit: `exact_ballpark.modes.ModeBallpark` (FOLLOW, ORBIT and WARP over
ref.py's GOTO and STOP) and, for dynamical orientation and the setters, `exact_ballpark.full.FullBallpark` (balls added
with the Python AddBall, none colliding, every bubble live: a client's view), which are themselves held bit for bit to
Destiny's own code compiled verbatim (tests/test_movement_modes_oracle.py, tests/test_movement_full_oracle.py,
tests/test_movement_turn_oracle.py); so it is exact against Destiny (Carbon's `destiny` at 114e89fa, MIT, (c) CCP
Games). Checked by tests/test_movement_vec_modes.py. `VecModesPark` extends
`exact_ballpark.vec_orient.VecOrientPark` (its setters, its rotation snap) without changing it. Rules, by rule id:

- MOV-701 to MOV-713: GOTO and STOP as vec.py and vec_orient.py (the closed-form integrator with its Taylor branch,
  GotoThrust, STOP's vertical damping without dynamical orientation, the commands, AddBall's clamps, the setters).
- MOV-714, MOV-715: FOLLOW (`EvolveFollow`) and the guards of `FollowBall` and `Orbit` (a non-finite range, the ball
  itself, a missing, moribund, cloaked or other-bubble target).
- MOV-716 to MOV-722: ORBIT under the old law (`EvolveOldStyleOrbit`) and the new (`GetOrbitalNormal`,
  `EvolveNewStyleOrbit`, `GotoThrustFollow`, `OrbitThrust`), chosen by the park's `use_new_orbit` (Destiny's
  process-global useNewOrbit); with dynamical orientation the new orbit's normal falls back on the ship's heading.
- MOV-728: dynamical orientation (`dynamical_orientation=True`) in every mode, a ball in warp proper turning toward
  its exit point; vec_orient_cuda's device code.
- MOV-730 (in part), MOV-735: when a ball starts warp proper its followers are released (an interactive orbiter by
  GotoDirection along its velocity, any other follower stops), at the warper's turn in the tick: a follower with a
  lower id has had its own turn already. Mode changes post no events here (`ModeBallpark.events`).
- TRV-701 to TRV-715: WarpTo (the short goto, the repeated no-op, the minimum range's bits, the warp factor),
  alignment and activation, RealWarp, the warp path, dropping out (TRV-714's extra step from the drop-out point) and
  EntityWarpIn; a Stop during warp ends it where it is, the ball left non-massive and its sensor off, as the public code
  does.
- MOV-732, MOV-734, MOV-736 (the reference is exact_ballpark.full): SetBallFree (a ball
  made free evolves from the next tick; fixing one stops it, zeroes its velocity, snaps the rotation with dynamical
  orientation and clears mLastG; it then never moves), SetBallRigid (Stop, then RIGID), the Python SetBoid (BOID
  without a Stop, or a Stop), SetBallTroll (Stop, free, interactive, due at the clock + the delay, a delay below 1 made
  1, TROLL) and PetrifyTroll at the end of the tick it is due (fixed, at rest, non-interactive, RIGID; in the kernel).

**The worlds.** Each row is a ball of one world: `ids` are Destiny's ball ids (the order of a tick's phase 1 within a
world, and the old orbit's precession), `world` groups the rows, `clock` is each row's world time (Destiny's
mCurrentTime: the old orbit's angles and the warp's times read it); a FOLLOW or ORBIT ball's `follow_ptr` is its
target's row, in the same world. A world's balls must all be in the park, each world's rows sharing one clock. The
follower set of a ball is implicit: the balls whose follow_ptr is its row and whose follow id is its id (`followers`),
the members StopAllFollowers acts on (Ballpark.cpp:5433-5450). These are the balls in FOLLOW or ORBIT, and a ball that
SetBoid took out of FOLLOW or ORBIT without a Stop: Stop(Ball*) clears the follow fields and leaves the set only for a
ball in a follow mode, so such a ball stays a member in BOID and any mode after it (stopped, or a GOTO) until a new
follow command or a WarpTo replaces its follow id. Destiny's set may also hold stale entries whose follow id names
another ball (the same ball after a new follow command); StopAllFollowers erases them without effect, and
`followers` leaves them out (Destiny's GetFollowers would list them).

**Exactness.** The kernel (vec_modes_cuda) runs the reference's own operations in their order with every rounding
pinned and glibc's functions transcribed (sin, cos, sincos and log from exact_ballpark.dlibm, exp from
exact_ballpark.vec_exp, the float functions and pow from vec_libm). The commands and setters here are the reference's on
tensors: only correctly rounded operations in the reference's order, tensor divisors. The tick on tensors
(`evolve_torch`) is VecComposedPark's phase 1, warp path and drop-out, moved here
verbatim (VecCombatPark's and VecComposedPark's own parts are hooks), with the kernel's commit and PetrifyTroll:
the reference's operations in its order on tensors, the double sin, cos, sincos and log through the C library on
the host (exact by construction: the reference calls the same functions), exp glibc's (vec_exp), the float
functions vec_libm's; held to ModeBallpark and FullBallpark on the kernel's own scenarios, with no tolerance, by
tests/test_movement_vec_modes.py (and on the leftovers, under OnTick, and under the server and space layers by
their tests).

Not here: collisions (exact_ballpark.vec_collide), the partition, bubbles and sensors (every bubble live, no events),
missiles, formations, mushrooms, the yaw, pitch and roll without dynamical orientation (MOV-729: display only, never
fed back into these modes' flight; exact_ballpark.vec_ypr_cuda has them), OnTick's time stamps (so SetBallFree's
mNewTime is not kept). VecCombatPark and VecComposedPark batch the
commands for their own ticks too (their kernels and torch ticks hold the troll's petrification and the follower sets);
`_g11` refuses them only for a park whose tick is none of the three.
"""

from __future__ import annotations

import math

import torch

from . import modes as MO
from . import vec_exp
from . import vec_libm as L
from . import vec_modes_cuda as K
from .modes import (AU, BOID, FOLLOW, INERT_MODES, MAX_ALIGN_TICKS, MIN_WARP_DISTANCE_SQ, MISSILE, ORBIT,
                    ORBITAL_PRECESSION, RIGID, TROLL, WARP, WARP_FACTOR_TO_ACCELERATION,
                    WARP_FACTOR_TO_AU_PER_SECOND, WARP_FACTOR_TO_DECELERATION)
from .ref import GOTO, STOP, STOP_DAMP
from .vec_collide import _to_int64
from .vec_orient import (VecOrientPark, _cols, _fdiv, _fmin, _stack, _where, m_rotation_q, masked_tick, q_mul,
                         q_rotation, transform_coord, v3_dot, v3_len, v3_mul)

F64, I64 = torch.float64, torch.int64
f32 = L.f32
LOADABLE = (STOP,) + tuple(INERT_MODES)            # AddBall's STOP, or an inert mode (ModeBallpark.add_ball)
TROLL_MIN_DELAY = 1                                 # SetBallTroll (Ballpark.cpp:6300); a test holds it to full.py's

# ---- the flight tick on tensors: its constants and tensor helpers, moved here
# verbatim from exact_ballpark.vec_composed, and RotationQuaternion(yaw, pitch, roll) from exact_ballpark.vec_combat ----

TRUNC7 = 10000000               # _trunc7's scale (modes.py:91)
RF_SCALE = 40000.0              # EvolveOldStyleOrbit's radial falloff, exp(-(r - dist)^2 / 40000) (modes.py:449)
GOTO_FAR = 10.0 * AU            # the old orbit's goto point, 10 AU along the thrust (modes.py:460)
NORMAL_MIN = 1e-12              # GetOrbitalNormal's parallel test (modes.py:484)
ANGLE_SCALE = 0.1               # the new orbit's angle and distance floor (modes.py:500, 507)
ANGLE_ZERO = 1e-10              # the new orbit's circle and linear tests (modes.py:511, 520)
FACTOR_FLOOR = -0.999           # the new orbit's factor floor (modes.py:517)
LOG_FLOOR = 0.001               # math.log(0.001) below the floor (modes.py:518)
SLOW_TARGET = 5e-2              # GotoThrustFollow's slow target (modes.py:541)
THRUST_OFF = 0.1                # OrbitThrust's cut (modes.py:581)
ALIGN_COS = 0.01                # IsAlignedForWarp (modes.py:331)
ALIGN_SPEED2 = 0.5625           # 75 % of maxVel, squared (modes.py:332)
DROP_SPEED = 100.0              # the drop-out speed's cap (modes.py:632)
REST_SQ = 0.1                   # the drop-out's re-filing at rest, and the commit's (modes.py:711, 723)
DIR_DIST = 1.0e17               # GotoDirection's point (modes.py:241)


# ---- tensor helpers: the reference's vector arithmetic, components left to right ----

def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _len(a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_dot(a, a))


def _unit(v: torch.Tensor) -> torch.Tensor:
    """ref.py's _normalized: times the reciprocal of the length; a zero vector stays as it is."""
    n = _len(v)
    inv = torch.ones_like(n) / torch.where(n == 0.0, 1.0, n)
    return torch.where((n == 0.0)[..., None], v, v * inv[..., None])


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vector3d::Cross (modes.py's _cross): components in Destiny's order."""
    return torch.stack((u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1], u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                        u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]), dim=-1)


def _div(a, b) -> torch.Tensor:
    """IEEE division a / b with tensors on both sides (a Python number becomes a tensor first)."""
    if not isinstance(b, torch.Tensor):
        b = torch.full_like(a, float(b))
    if not isinstance(a, torch.Tensor):
        a = torch.full_like(b, float(a))
    return a / b


def _c_call(fn, x: float) -> float:
    """The C library's double function at x through Python's math (which calls it), with C's results where Python
    raises (outside the reference's domain: the reference itself raises there)."""
    try:
        return fn(x)
    except ValueError:
        if fn is math.log and x == 0.0:
            return -math.inf
        return math.nan
    except OverflowError:
        return math.inf


def _host(fn, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """fn (math.sin, math.cos or math.log: glibc's double functions) on the host for the elements in mask; NaN
    elsewhere. One host round trip."""
    out = torch.full_like(x, math.nan)
    if bool(mask.any()):
        vals = [_c_call(fn, v) for v in x[mask].tolist()]
        out[mask] = torch.tensor(vals, dtype=F64, device=x.device)
    return out


def _host_sincos(x: torch.Tensor, mask: torch.Tensor) -> tuple:
    """glibc's sincos (exact_ballpark.modes._sincos: the sine and cosine of one argument as compiled Destiny takes them,
    the finding F3) on the host for the elements in mask; NaN elsewhere. One host round trip."""
    s, c = torch.full_like(x, math.nan), torch.full_like(x, math.nan)
    if bool(mask.any()):
        vals = [MO._sincos(v) for v in x[mask].tolist()]
        s[mask] = torch.tensor([v[0] for v in vals], dtype=F64, device=x.device)
        c[mask] = torch.tensor([v[1] for v in vals], dtype=F64, device=x.device)
    return s, c


def _trunc7(x: torch.Tensor) -> torch.Tensor:
    """`(double)((int64_t)(x * 10000000)) / 10000000` (modes.py's _trunc7; the integer conversion makes a value in
    (-1e-7, 0] a +0)."""
    return _div(_to_int64(x * float(TRUNC7)).to(F64), float(TRUNC7))


def q_rotation_ypr(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> tuple:
    """RotationQuaternion(yaw, pitch, roll) (exact_ballpark.full.q_rotation_ypr: DirectXMath's scalar path) on floats
    held in float64: half angles, cosf and sinf, four sums of float products left to right."""
    hp, hy, hr = f32(pitch * 0.5), f32(yaw * 0.5), f32(roll * 0.5)
    cp, sp, cy, sy, cr, sr = L.cosf(hp), L.sinf(hp), L.cosf(hy), L.sinf(hy), L.cosf(hr), L.sinf(hr)
    return (f32(f32(f32(cr * sp) * cy) + f32(f32(sr * cp) * sy)), f32(f32(f32(cr * cp) * sy) - f32(f32(sr * sp) * cy)),
            f32(f32(f32(sr * cp) * cy) - f32(f32(cr * sp) * sy)), f32(f32(f32(cr * cp) * cy) + f32(f32(sr * sp) * sy)))


class VecModesPark(VecOrientPark):
    """B balls, one per row, in independent worlds, with Destiny's flight modes; evolve is one CUDA kernel a tick on a
    CUDA device and the same algorithm on tensors elsewhere (`evolve_torch`)."""

    disable_do_for_missiles = False         # no missiles here (VecCombatPark's option; `_phase1`)

    def __init__(self, n: int, friction: float = 1.0e6, tick_interval_ms: int = 1000, device="cuda",
                 dynamical_orientation: bool = False, use_new_orbit: bool = False, block: int = 128,
                 ypr: str = "host"):
        super().__init__(n, friction, tick_interval_ms, device, dynamical_orientation, ypr=ypr)
        if self.dev.type == "cuda" and self.dev.index is None:      # "cuda" as the tensors report it: cuda:N
            self.dev = torch.device("cuda", torch.cuda.current_device())
        self.use_new_orbit = bool(use_new_orbit)
        self.block = int(block)
        z = lambda *s: torch.zeros(*s, dtype=F64, device=self.dev)
        zi = lambda: torch.zeros(n, dtype=I64, device=self.dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.last_collision, self.follow_range = z(n) - 1.0, z(n) + 10.0
        self.ids, self.world, self.clock = torch.arange(1, n + 1, dtype=I64, device=self.dev), zi(), zi()
        self.effect_stamp, self.follow_id, self.owner_id = zi(), zi(), zi()
        self.follow_ptr = zi() - 1
        self.cloaked, self.bubble = zi(), zi() - 1
        self.free, self.moribund, self.massive, self.interactive, self.sensor = zb(), zb(), zb(), zb(), zb()
        self.act = torch.zeros(n, dtype=torch.uint8, device=self.dev)
        self.pos0, self.vel0, self.vy_after = z(n, 3), z(n, 3), z(n)

    # ---- loading: the Python AddBall of new balls (Ballpark.cpp:3303; MOV-711) ----

    def load(self, rows: list[dict]) -> None:
        """Every ball as the Python AddBall adds a new one (Ballpark.cpp:3303; MOV-711): AddBall's clamps, the decay
        factor, mode STOP or an inert mode, Ball::Ball's defaults (follow range 10, warp length -1, no follow fields,
        the sensor off); with dynamical orientation the turning values and the rotation at rest snapped to the velocity
        (AddDynamicallyOrientedBall, SnapOrientation), as VecOrientPark.load. Keys: id, world, time (the world's
        clock), mass, radius, max_vel, pos, vel, agility, speed_fraction, free, massive, interactive, cloaked, bubble,
        moribund, mode, max_ang_vel (default 1), rot_agility (default 0.9)."""
        n = len(rows)
        if n != self.n:
            raise ValueError(f"{n} rows for a park of {self.n}")
        for r in rows:
            if r.get("mode", STOP) not in LOADABLE:
                raise ValueError(f"a ball starts in STOP or an inert mode, not {r.get('mode')}")
        t = lambda xs, dt=F64: torch.tensor(xs, dtype=dt, device=self.dev)
        ff = lambda key, default=None: f32(t([float(r[key] if default is None else r.get(key, default)) for r in rows]))
        self.mass = t([float(r["mass"]) for r in rows])
        self.mass = torch.where(self.mass < 0.0, 0.0, self.mass)
        self.radius = ff("radius")
        self.radius = torch.where(self.radius < 0.0, 0.0, self.radius)
        self.max_vel = ff("max_vel")
        self.max_vel = torch.where(self.max_vel < 0.0, 0.0, self.max_vel)
        self.agility = ff("agility")
        self.agility = torch.where(self.agility <= 0.0, 1.0, self.agility)
        self.speed_fraction = ff("speed_fraction")
        self.speed_fraction = torch.where(self.speed_fraction < 0.0, 0.0, self.speed_fraction)
        self.pos = t([[float(c) for c in r["pos"]] for r in rows])
        self.vel = t([[float(c) for c in r["vel"]] for r in rows])
        self.old_pos = self.pos - self.vel * self.dt
        self.old_vel = self.vel.clone()
        self.goto, self.last_g = torch.zeros_like(self.pos), torch.zeros_like(self.pos)
        self.time_factor = self._decay(self.mass * self.agility)
        z = torch.zeros(n, dtype=F64, device=self.dev)
        self.rot, self.old_rot = _stack((z, z, z, z + 1.0)), _stack((z, z, z, z + 1.0))
        self.ang_vel, self.old_ang_vel = torch.zeros_like(self.pos), torch.zeros_like(self.pos)
        self.torque, self.rot_roll = torch.zeros_like(self.pos), z.clone()
        if self.dynamical_orientation:                  # only AddDynamicallyOrientedBall takes the turning values
            self.max_ang_vel = ff("max_ang_vel", 1.0)
            self.max_ang_vel = torch.where(self.max_ang_vel < 0.0, 0.0, self.max_ang_vel)
            self.rot_agility = ff("rot_agility", 0.9)
            self.rot_agility = torch.where(self.rot_agility <= 0.0, 1.0, self.rot_agility)
            w = _cols(self.ang_vel)
            back = q_rotation(w, f32(-self.fdt * v3_len(w)))            # mOldRot of a new ball: rot x RotationQ
            self.old_rot = _stack(q_mul(_cols(self.rot), back))           # (angVel, -dt |angVel|)
            self._snap_orientation(torch.ones(n, dtype=torch.bool, device=self.dev))
        else:                                           # Ball::Ball's defaults: 1 rad/s, 0.9 (as floats)
            self.max_ang_vel, self.rot_agility = z + 1.0, f32(z + 0.9)
        self.ids = t([int(r["id"]) for r in rows], I64)
        self.world = t([int(r.get("world", 0)) for r in rows], I64)
        self.clock = t([int(r.get("time", 0)) for r in rows], I64)
        self.mode = t([int(r.get("mode", STOP)) for r in rows], I64)
        self.free = t([bool(r.get("free", True)) for r in rows], torch.bool)
        self.massive = t([bool(r.get("massive", False)) for r in rows], torch.bool)
        self.interactive = t([bool(r.get("interactive", False)) for r in rows], torch.bool)
        self.moribund = t([bool(r.get("moribund", False)) for r in rows], torch.bool)
        self.cloaked = t([int(r.get("cloaked", 0)) for r in rows], I64)
        self.bubble = t([int(r.get("bubble", -1)) for r in rows], I64)
        self.sensor = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.effect_stamp, self.follow_id, self.owner_id = (torch.zeros(n, dtype=I64, device=self.dev)
                                                            for _ in range(3))
        self.follow_ptr = torch.full((n,), -1, dtype=I64, device=self.dev)
        self.follow_range = torch.full((n,), 10.0, dtype=F64, device=self.dev)
        self.last_collision = torch.full((n,), -1.0, dtype=F64, device=self.dev)

    # ---- the follower bookkeeping: Stop(Ball*) (Ballpark.cpp:4578) ----

    def _following(self) -> torch.Tensor:
        return (self.mode == FOLLOW) | (self.mode == ORBIT)

    def _stop_rows(self, mask: torch.Tensor) -> None:
        """Stop(Ball*): a FOLLOW or ORBIT ball leaves its target's (implicit) follower set and its follow fields are
        cleared; then STOP."""
        follower = mask & self._following()
        self.effect_stamp = torch.where(follower, 0, self.effect_stamp)
        self.follow_ptr = torch.where(follower, -1, self.follow_ptr)
        self.follow_id = torch.where(follower, 0, self.follow_id)
        self.owner_id = torch.where(follower, 0, self.owner_id)
        self.follow_range = torch.where(follower, 0.0, self.follow_range)
        self.mode = torch.where(mask, STOP, self.mode)

    def _members(self, rows: torch.Tensor) -> torch.Tensor:
        """[N] bool: the balls in the follower set of a row in `rows` ([N] bool) as StopAllFollowers acts on it
        (Ballpark.cpp:5433-5450): follow_ptr names the row and the follow id is its id. In FOLLOW or ORBIT that is
        follow_ptr alone; a ball SetBoid took out of a follow mode keeps both (the module's docstring)."""
        safe = torch.where(self.follow_ptr >= 0, self.follow_ptr, 0)
        return (self.follow_ptr >= 0) & (self.follow_id == self.ids[safe]) & rows[safe]

    def followers(self, row: int) -> list:
        """The follower set of the ball in `row` (Destiny's mFollowers less its stale entries, the module's docstring):
        the ids of the balls whose follow_ptr is that row and whose follow id is its id, ascending."""
        sel = (self.follow_ptr == row) & (self.follow_id == self.ids[row])
        return sorted(self.ids[sel].tolist())

    # ---- commands (MOV-708, MOV-709, MOV-715, TRV-701 to TRV-704, TRV-715) ----

    def stop(self, mask: torch.Tensor) -> None:
        """Stop(const ID&) (Ballpark.cpp:4556): nothing for a ball already in STOP."""
        self._stop_rows(mask & (self.mode != STOP))

    def goto_point(self, mask: torch.Tensor, point: torch.Tensor) -> None:
        """GotoPoint (Ballpark.cpp:4529): refused for a non-finite point; Stop, the point, a zero speed fraction made 1,
        then GOTO."""
        point = point.to(F64)
        ok = mask & torch.isfinite(point).all(dim=1)
        self._stop_rows(ok)
        self.goto = torch.where(ok[:, None], point, self.goto)
        self.speed_fraction = torch.where(ok & (self.speed_fraction == 0.0), 1.0, self.speed_fraction)
        self.mode = torch.where(ok, GOTO, self.mode)

    def _unit(self, d: torch.Tensor) -> torch.Tensor:
        """ref.py's _normalized: times the reciprocal of the length; a zero vector stays as it is."""
        norm = torch.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        inv = torch.ones_like(norm) / torch.where(norm == 0.0, 1.0, norm)
        return torch.where((norm == 0.0)[:, None], d, d * inv[:, None])

    def goto_direction(self, mask: torch.Tensor, direction: torch.Tensor) -> None:
        """GotoDirection (Ballpark.cpp:4483): refused when not finite; a point 1e17 m along the unit direction."""
        d = direction.to(F64)
        ok = mask & torch.isfinite(d).all(dim=1)
        self.goto_point(ok, self.pos + self._unit(d) * 1.0e17)

    def _follow(self, mask: torch.Tensor, target: torch.Tensor, range_: torch.Tensor, mode: int) -> None:
        """FollowBall and Orbit (Ballpark.cpp:3879, 4007; MOV-715): the float range refused unless finite; the ball
        itself, a missing (row -1), moribund, cloaked or other-bubble target refused; else Stop, the follow fields, the
        mode."""
        rng = f32(range_.to(F64))
        target = target.to(I64)
        safe = torch.where((target >= 0) & (target < self.n), target, 0)
        rows = torch.arange(self.n, device=self.dev)
        ok = (mask & torch.isfinite(rng) & (target >= 0) & (target < self.n) & (safe != rows)
              & (self.world[safe] == self.world) & ~self.moribund[safe] & (self.bubble[safe] == self.bubble)
              & (self.cloaked[safe] == 0))
        self._stop_rows(ok)
        self.follow_id = torch.where(ok, self.ids[safe], self.follow_id)
        self.follow_ptr = torch.where(ok, safe, self.follow_ptr)
        self.follow_range = torch.where(ok, rng, self.follow_range)
        self.mode = torch.where(ok, mode, self.mode)

    def follow(self, mask: torch.Tensor, target: torch.Tensor, range_: torch.Tensor) -> None:
        """FollowBall: target is each row's target row (-1: missing); range_ the follow range (Destiny's default 1.0
        is the caller's)."""
        self._follow(mask, target, range_, FOLLOW)

    def orbit(self, mask: torch.Tensor, target: torch.Tensor, range_: torch.Tensor) -> None:
        """Orbit: as FollowBall, in ORBIT."""
        self._follow(mask, target, range_, ORBIT)

    def warp_to(self, mask: torch.Tensor, point: torch.Tensor, min_range: torch.Tensor,
                warp_factor: torch.Tensor) -> None:
        """WarpTo (Ballpark.cpp:4072; TRV-701 to TRV-704): refused for a non-finite point; a negative minimum range is
        0 and a warp factor below 1 is 1; within 1e5 m a GotoPoint; the same point while warping is a no-op (the align
        counter kept); else Stop, the point, aligning (-1), the minimum range's bits and the warp factor, WARP."""
        point, mr, wf = point.to(F64), min_range.to(F64), warp_factor.to(I64)
        ok = mask & torch.isfinite(point).all(dim=1)
        mr = torch.where(mr < 0.0, 0.0, mr)
        wf = torch.where(wf <= 0, 1, wf)
        d = self.pos - point
        short = ok & ((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]) < MIN_WARP_DISTANCE_SQ)
        self.goto_point(short, point)
        ok = ok & ~short & ~((self.mode == WARP) & (self.goto == point).all(dim=1))
        self._stop_rows(ok)
        self.goto = torch.where(ok[:, None], point, self.goto)
        self.effect_stamp = torch.where(ok, -1, self.effect_stamp)
        self.follow_range = torch.where(ok, 0.0, self.follow_range)
        self.follow_id = torch.where(ok, mr.contiguous().view(I64), self.follow_id)
        self.owner_id = torch.where(ok, wf, self.owner_id)
        self.mode = torch.where(ok, WARP, self.mode)

    def _release_followers(self, warping: torch.Tensor) -> None:
        """StopAllFollowers (Ballpark.cpp:5416; MOV-730) of the rows in `warping`: an interactive orbiter flies on by
        GotoDirection along its velocity, any other member stops (Stop(Ball*): a member out of the follow modes keeps
        its follow fields, so it stays a member)."""
        f = self._members(warping)
        orbiter = f & self.interactive & (self.mode == ORBIT)
        self._stop_rows(f & ~orbiter)
        self.goto_direction(orbiter, self.vel.clone())

    def entity_warp_in(self, mask: torch.Tensor, point: torch.Tensor, warp_factor: torch.Tensor) -> None:
        """EntityWarpIn (Ballpark.cpp:4370; TRV-715): WarpTo with no minimum range, then, unless that made a GOTO,
        warp proper at once (RealWarp, its followers released), begun five ticks ago, at warp speed along the
        point's direction from the origin, with dynamical orientation the rotation snapped to it."""
        point = point.to(F64)
        self.warp_to(mask, point, torch.zeros_like(point[:, 0]), warp_factor)
        go = mask & (self.mode != GOTO)
        self._release_followers(go)
        delta = self._unit(self.pos - self.goto)                     # RealWarp: the exit point, min range short
        mr = self.follow_id.contiguous().view(F64)
        dst = self.goto + delta * mr[:, None]
        self.goto = torch.where(go[:, None], dst, self.goto)
        self.massive = torch.where(go, False, self.massive)
        self.sensor = torch.where(go, False, self.sensor)
        self.vel = torch.where(go[:, None], self._unit(point) * AU, self.vel)
        if self.dynamical_orientation:                  # SnapOrientation to the new velocity (Ballpark.cpp:4392)
            self._snap_orientation(go)
        self.effect_stamp = torch.where(go, torch.clamp(self.clock - 5, min=0), self.effect_stamp)
        back = self.pos - self.goto
        length = torch.sqrt(back[:, 0] * back[:, 0] + back[:, 1] * back[:, 1] + back[:, 2] * back[:, 2])
        self.last_collision = torch.where(go, length, self.last_collision)

    # ---- the setters: VecOrientPark's (MOV-711, MOV-713), SetBallVelocity's snap only with dynamical orientation ----

    def set_velocity(self, mask: torch.Tensor, v: torch.Tensor) -> None:
        """SetBallVelocity (MOV-713): the new and old velocity; with dynamical orientation the rotation snapped (the
        yaw, pitch and roll without it are not kept here)."""
        v = v.to(F64)
        self.vel = torch.where(mask[:, None], v, self.vel)
        self.old_vel = torch.where(mask[:, None], v, self.old_vel)
        if self.dynamical_orientation:
            self._snap_orientation(mask)

    # ---- the commands: SetBallFree, SetBallRigid, SetBoid, SetBallTroll (MOV-732, 734, 736) ----

    def _g11(self, name: str) -> None:
        """Refuse a command whose tick this park's subclass does not run: VecCombatPark and VecComposedPark tick with
        their own kernels, which have neither the troll's petrification nor the follower sets of SetBoid."""
        if type(self).evolve is not VecModesPark.evolve or type(self).evolve_torch is not VecModesPark.evolve_torch:
            raise NotImplementedError(f"{name} is batched for VecModesPark's own tick only, not {type(self).__name__}")

    def set_free(self, mask: torch.Tensor, flag: torch.Tensor) -> None:
        """SetBallFree (Ballpark.cpp:4887; MOV-732) for the rows in mask, flag [N] (nonzero: free): nothing where the
        flag is the ball's already. Freed: in the free set, so it evolves from the next tick (its minis removed and
        mNewTime set a tick back: neither is kept here). Fixed: Stop(Ball*), SetBallVelocity(0, 0, 0) (the new and old
        velocity; with dynamical orientation SnapOrientation, which leaves a ball at rest as it is), out of the free
        set, mLastG cleared (InsertBallInBoxes and AddMiniBalls: no partition or minis here)."""
        self._g11("SetBallFree")
        flag = flag.to(self.dev) != 0
        change = mask & (flag != self.free)
        fix = change & ~flag
        self.free = torch.where(change, flag, self.free)
        self._stop_rows(fix)
        self.set_velocity(fix, torch.zeros_like(self.vel))
        self.last_g = torch.where(fix[:, None], 0.0, self.last_g)

    def set_ball_rigid(self, mask: torch.Tensor) -> None:
        """SetBallRigid (Ballpark.cpp:6278; MOV-734): Stop(Ball*), then RIGID (no thrust; a free ball coasts)."""
        self._g11("SetBallRigid")
        self._stop_rows(mask)
        self.mode = torch.where(mask, RIGID, self.mode)

    def set_boid(self, mask: torch.Tensor, on: torch.Tensor) -> None:
        """The Python SetBoid (PySetBoid, Thunkers.cpp:1003; MOV-734): turnOn nonzero sets BOID without a Stop (a
        follower keeps its follow fields and stays in its target's follower set: the module's docstring); zero is
        Stop(Ball*)."""
        self._g11("SetBoid")
        on = on.to(self.dev) != 0
        self.mode = torch.where(mask & on, BOID, self.mode)
        self._stop_rows(mask & ~on)

    def set_interactive(self, mask: torch.Tensor, flag: torch.Tensor) -> None:
        """SetBallInteractive (Ballpark.cpp:6339): the flag (the bubble's interactive count is not kept here)."""
        self.interactive = torch.where(mask, flag.to(self.dev) != 0, self.interactive)

    def set_ball_troll(self, mask: torch.Tensor, delay: torch.Tensor) -> None:
        """PySetBallTroll ("Li": the delay a C int) and SetBallTroll (Ballpark.cpp:6290; MOV-736): a delay below 1 made
        1; Stop(Ball*); SetBallFree(true); SetBallInteractive(true); due at the world's clock + the delay; TROLL. The
        kernel petrifies it at the end of the tick in which it is due (PetrifyTroll, Ballpark.cpp:6327)."""
        self._g11("SetBallTroll")
        d = delay.to(device=self.dev, dtype=I64)
        d = torch.where(d < TROLL_MIN_DELAY, TROLL_MIN_DELAY, d)
        self._stop_rows(mask)
        self.set_free(mask, torch.ones_like(mask))
        self.set_interactive(mask, torch.ones_like(mask))
        self.effect_stamp = torch.where(mask, self.clock + d, self.effect_stamp)
        self.mode = torch.where(mask, TROLL, self.mode)

    # ---- the tick on tensors: VecCombatPark's thrust, integrator and rotated
    # vector, and VecComposedPark's phase 1, warp path and drop-out, moved here verbatim; the combat parks' own parts
    # of phase 1 are hooks (`_existing`, `_phase1_combat`, `_clear_contacts`) ----

    def _thrust_to(self, target: torch.Tensor, taper: torch.Tensor) -> torch.Tensor:
        """GotoThrust (Ballpark.cpp:1398) toward target for every row: tapered by (L^2 / D^2)^2 inside one tick's
        distance where `taper` (no missile flag), at maxThrust otherwise; C's division where the mass may be zero."""
        a = target - self.pos
        length2 = a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1] + a[:, 2] * a[:, 2]
        dist = self.speed_fraction * self.max_vel * self.dt
        max_thrust = self.friction * self.speed_fraction * self.max_vel / (self.mass * self.agility)
        unit = self._unit(a)
        dd = dist * dist
        coff = length2 / torch.where(dd == 0.0, torch.ones_like(dd), dd)
        scale = torch.where(taper & (length2 < dd), max_thrust * coff * coff, max_thrust)
        return unit * scale[:, None]

    def _integrate(self, p, v, a, m, tf) -> tuple:
        """Ballpark::Integrate over the tick with its cached decay factor (MOV-701 to MOV-703), m a column."""
        k, t = self.friction, self.dt
        if t == 0.0:                    # Integrate returns at once for no time (ref.py, the kernels)
            return p, v
        k2 = k * k
        ook2, ook = 1.0 / k2, 1.0 / k
        exact = (m * (m * a * (tf - 1.0) + k * (a * t + v - tf * v)) + p * k2) * ook2
        taylor = (m * (a * (-k * t) + k * (a * t + v * k * (1.0 / m) * t)) + p * k2) * ook2
        small = (k < 1e-10 * m * t)
        return torch.where(small, taylor, exact), (m * a - (m * a - v * k) * tf) * ook

    def rotated_vector(self, rows: torch.Tensor, v: tuple) -> torch.Tensor:
        """Ball::GetRotatedVector (Ball.cpp:2191) of the balls in `rows`: the float vector v turned by the rotation
        (dynamical orientation) or by RotationQuaternion(yaw, pitch, roll) of the floats; [R, 3] floats as doubles."""
        if self.dynamical_orientation:
            q = _cols(self.rot[rows])
        else:
            q = q_rotation_ypr(f32(self.yaw[rows]), f32(self.pitch[rows]), f32(self.roll[rows]))
        return _stack(transform_coord(v, m_rotation_q(q)))

    def _activates(self, live: torch.Tensor) -> torch.Tensor:
        """Starts warp proper at its own turn this tick (IsAlignedForWarp of an aligning ball, TRV-706, on its start
        state, which no other ball changes before its turn)."""
        d = _unit(self.goto - self.pos)
        u = _unit(self.vel)
        cos = u[:, 0] * d[:, 0] + u[:, 1] * d[:, 1] + u[:, 2] * d[:, 2]
        fast = _dot(self.vel, self.vel) > ALIGN_SPEED2 * self.max_vel * self.max_vel
        aligned = ((1.0 - cos).abs() < ALIGN_COS) & fast
        forced = self.effect_stamp.abs() > MAX_ALIGN_TICKS
        return live & (self.mode == WARP) & (self.effect_stamp < 0) & (aligned | forced)

    def _released_mode(self) -> torch.Tensor:
        """The mode StopAllFollowers leaves each row in (combat_release with apply false): a missile or an interactive
        orbiter GOTO, unless its velocity or its point is not finite (then its mode stays); any other follower STOP."""
        flies = (self.mode == MISSILE) | (self.interactive & (self.mode == ORBIT))
        point = self.pos + _unit(self.vel) * DIR_DIST
        ok = torch.isfinite(self.vel).all(dim=1) & torch.isfinite(point).all(dim=1)
        return torch.where(flies, torch.where(ok, GOTO, self.mode), STOP)

    def _release_rows(self, f: torch.Tensor) -> None:
        """StopAllFollowers on the follower rows in f (MOV-730): a missile made non-massive, a missile or an
        interactive orbiter flies on by GotoDirection along its velocity, any other follower stops (on the master a
        formation follower's slot freed: VecCombatPark._stop_rows)."""
        if not bool(f.any()):
            return
        missile = f & (self.mode == MISSILE)
        flies = missile | (f & self.interactive & (self.mode == ORBIT))
        self.massive = torch.where(missile, False, self.massive)
        self._stop_rows(f & ~flies)
        self.goto_direction(flies, self.vel.clone())

    def _release_turns(self, hit: torch.Tensor, ts: torch.Tensor) -> tuple:
        """The followers released this tick (`hit`: their target starts warp proper) split by turn (MOV-730): before
        their own turn when the target's id is lower, after it otherwise."""
        return hit & (self.ids[ts] < self.ids), hit & (self.ids[ts] > self.ids)

    def _precessing_normal(self, rows: torch.Tensor) -> torch.Tensor:
        """DefaultOrbitalNormal (Ballpark.cpp:735): the precessing w vector of MOV-717 truncated to 7 decimals, not
        normalised; cos and sin on the host for `rows`."""
        t = self.clock.to(F64)
        phi1 = t * ORBITAL_PRECESSION
        phi2 = (self.ids & 0x000000000000FFFF).to(F64) + t * ORBITAL_PRECESSION
        (s1, c1), (s2, c2) = _host_sincos(phi1, rows), _host_sincos(phi2, rows)
        return torch.stack((_trunc7(c1 * c2), _trunc7(s2), _trunc7(s1 * c2)), dim=1)

    def _old_orbit(self, rows: torch.Tensor, op: torch.Tensor, orad: torch.Tensor) -> tuple:
        """EvolveOldStyleOrbit (Ballpark.cpp:1246; MOV-717, MOV-718): (thrust, goto point)."""
        cruise = self.speed_fraction * self.max_vel
        max_thrust = (self.friction * cruise) / (self.mass * self.agility)
        r = (self.follow_range + self.radius) + orad
        to = op - self.pos
        dist = _len(to)
        to = _unit(to)
        w = _unit(_cross(self._precessing_normal(rows), to))
        to_comp = dist * dist - r * r
        tangent = to_comp >= 0.0
        rad_comp = (r * torch.sqrt(torch.where(tangent, to_comp, 0.0))) / dist
        s = to_comp / dist
        to = torch.where(tangent[:, None], _unit(to * s[:, None] + w * rad_comp[:, None]), to)
        dr = r - dist
        rf = _trunc7(vec_exp.exp(_div((-dr) * dr, RF_SCALE)))
        phi = ((-to[:, 0]) * w[:, 0] + (-to[:, 1]) * w[:, 1]) + (-to[:, 2]) * w[:, 2]
        tf = 1.0 + (rf * rf) * (phi * phi - 1.0)
        tf = torch.where(tf > 0.0, rf * phi + torch.sqrt(torch.where(tf > 0.0, tf, 0.0)), rf * phi)
        tf = tf * torch.where(dist - r >= 0.0, 1.0, -1.0)
        a = (w * rf[:, None] + to * tf[:, None]) * max_thrust[:, None]
        return a, self.pos + a * GOTO_FAR

    def _orbit_thrust(self, target: torch.Tensor) -> torch.Tensor:
        """OrbitThrust (Ballpark.cpp:1428; MOV-722): brake toward the point; zero below a tenth of max acceleration."""
        fom = _div(self.friction, self.mass * self.agility)
        max_accel = (fom * self.speed_fraction) * self.max_vel
        a = (target - self.pos) * fom[:, None] - self.vel
        tacc = _len(a) / max_accel
        inv = torch.ones_like(tacc) / tacc
        a = torch.where((tacc > 1.0)[:, None], a * inv[:, None], a)
        a = torch.where((tacc < THRUST_OFF)[:, None], 0.0, a)
        return torch.where((max_accel <= 0.0)[:, None], 0.0, a)

    def _goto_thrust_follow(self, target: torch.Tensor, tv: torch.Tensor) -> torch.Tensor:
        """GotoThrustFollow (Ballpark.cpp:1463; MOV-721) with weight 1, for the new orbit (its only caller): reach
        the point and match the velocity, capped; OrbitThrust for a slow target."""
        speed = _len(tv)
        slow = speed < (SLOW_TARGET * self.max_vel) * self.speed_fraction
        fom = _div(self.friction, self.mass * self.agility)
        max_accel = (fom * self.speed_fraction) * self.max_vel
        one_minus = 1.0 - self.time_factor
        inv = torch.ones_like(fom) / (self.dt * fom - one_minus)
        a = (((target - self.pos) * fom[:, None] - self.vel * one_minus[:, None]) * fom[:, None]) * inv[:, None]
        inv = torch.ones_like(fom) / (1.0 - self.time_factor)
        avel = ((tv - self.vel * self.time_factor[:, None]) * fom[:, None]) * inv[:, None]
        weight = 1.0
        inv_w = 1.0 / (1.0 + weight)                    # a Python constant: 0.5, exactly
        a = (a + avel * weight) * inv_w
        accel = _dot(a, a)
        capped = accel > max_accel * max_accel
        s = max_accel / torch.sqrt(accel)
        a = torch.where(capped[:, None], a * s[:, None], a)
        a = torch.where((max_accel <= 0.0)[:, None], 0.0, a)
        return torch.where(slow[:, None], self._orbit_thrust(target), a)

    def _new_orbit(self, rows: torch.Tensor, op: torch.Tensor, ov: torch.Tensor, orad: torch.Tensor) -> tuple:
        """EvolveNewStyleOrbit (Ballpark.cpp:1134; MOV-720) with GetOrbitalNormal (MOV-719; with dynamical orientation
        the heading's fallback, FullBallpark._orbital_normal): (thrust, goto point). ov is the target's velocity as
        the ball sees it at its own turn."""
        do = self.dynamical_orientation
        cruise = self.speed_fraction * self.max_vel
        tau = _div(self.mass * self.agility, self.friction)
        max_thrust = cruise / tau
        r = (self.follow_range + self.radius) + orad
        to = op - self.pos
        to = to + ov * self.dt
        dist = _len(to)
        on = dist > 0.0
        inv = torch.ones_like(dist) / torch.where(on, dist, 1.0)
        to = torch.where(on[:, None], to * inv[:, None], torch.tensor([0.0, 0.0, 1.0], dtype=F64, device=self.dev))
        o = _cross(to, self.vel)
        length2 = _dot(o, o)
        par = length2 > NORMAL_MIN
        inv = torch.ones_like(length2) / torch.sqrt(torch.where(par, length2, 1.0))
        normal = o * inv[:, None]
        if do:
            heading = self.rotated_vector(torch.arange(self.n, device=self.dev), (0.0, 0.0, 1.0))
            o2 = _cross(to, heading)
            l2 = _dot(o2, o2)
            use_h = ~par & ~(l2 <= NORMAL_MIN)
            inv2 = torch.ones_like(l2) / torch.sqrt(torch.where(use_h, l2, 1.0))
            normal = torch.where(use_h[:, None], o2 * inv2[:, None], normal)
            default = ~par & ~use_h
        else:
            default = ~par
        default = default & rows
        if bool(default.any()):
            normal = torch.where(default[:, None], self._precessing_normal(default), normal)
        radial = _unit(_cross(normal, to))
        speed = _len(self.vel)
        max_speed = torch.where(cruise < speed, speed, cruise)
        angle = _div(ANGLE_SCALE * (r - dist), max_speed)
        angle = torch.where(angle < -1.0, -1.0, angle)
        angle = torch.where(angle > 1.0, 1.0, angle)
        dist = torch.where(dist < ANGLE_SCALE, ANGLE_SCALE, dist)
        sq = torch.sqrt(dist * max_thrust)
        est_v = torch.where(sq < max_speed, sq, max_speed)
        est_d = est_v * self.dt
        circle = angle.abs() < ANGLE_ZERO
        # the circle: t = est_d / dist
        t_c = est_d / dist
        # the spiral: t from the factor's three branches
        factor = ((est_d * angle) / dist) / torch.sqrt(angle * angle + 1.0)
        floor = ~circle & (factor < FACTOR_FLOOR)
        logb = ~circle & ~floor & (factor.abs() > ANGLE_ZERO)
        lin = ~circle & ~floor & ~logb
        log_floor = _c_call(math.log, LOG_FLOOR)
        lg = _host(math.log, 1.0 + factor, logb & rows)
        t_s = torch.where(floor, _div(torch.full_like(angle, log_floor), angle),
                          torch.where(logb, lg / angle, factor / angle))
        t = torch.where(circle, t_c, t_s)
        need = rows & (circle | floor | logb | lin)
        st, ct = _host_sincos(t, need)
        gp_c = to * (dist * (1.0 - ct))[:, None] + radial * (dist * st)[:, None]
        gv_c = radial * (ct * est_v)[:, None] + to * (st * est_v)[:, None]
        dexp = dist * vec_exp.exp(angle * t)
        gp_s = to * (dist - dexp * ct)[:, None] + radial * (dexp * st)[:, None]
        gv_s = _unit(to * ((-angle) * ct + st)[:, None] + radial * (angle * st + ct)[:, None]) * est_v[:, None]
        gp = torch.where(circle[:, None], gp_c, gp_s)
        gv = torch.where(circle[:, None], gv_c, gv_s)
        goto = self.pos + gp
        gv = gv + ov
        return self._goto_thrust_follow(goto, gv), goto

    def _real_warp(self, act: torch.Tensor) -> None:
        """RealWarp (Ballpark.cpp:4143; TRV-707), the warper's own part (its followers are released by their rows):
        the exit point min range short of the goto point, warp proper from this tick, non-massive, sensor off."""
        if not bool(act.any()):
            return
        dst = self.goto
        delta = _unit(self.pos - dst)
        mr = self.follow_id.contiguous().view(F64)
        dst = dst + delta * mr[:, None]
        delta = dst - self.pos
        self.goto = torch.where(act[:, None], dst, self.goto)
        self.effect_stamp = torch.where(act, self.clock, self.effect_stamp)
        self.last_collision = torch.where(act, _len(delta), self.last_collision)
        self.massive = torch.where(act, False, self.massive)
        self.sensor = torch.where(act, False, self.sensor)

    def _orient_composed(self, a: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        """EvolveBehaviorForBall's dynamical-orientation branch (Ballpark.cpp:840-890; MOV-728; FullBallpark._orient)
        for the rows in `steer`: VecOrientPark._orient, with a ball in warp proper that coasts turning toward its exit
        point (the torque searched toward mGoto - pos, as floats)."""
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
        accel = f32(torch.sqrt(ax * ax + ay * ay + az * az))
        max_accel = _fdiv(f32(f32(torch.full_like(ax, self.ffriction) * self.speed_fraction) * self.max_vel),
                          f32(f32(self.mass) * self.agility))
        thrust = accel > 1e-2 * max_accel
        warping = (self.mode == WARP) & (self.effect_stamp >= 0)
        coast_warp = ~thrust & warping
        t_scale = self._t_scale()
        tf = L.expf(f32(-self.fdt * t_scale))
        kap = self._kappas(tf)
        a32 = _cols(f32(a))
        g32 = _cols(f32(self.goto - self.pos))
        direction = _where(thrust, a32, g32)
        search = steer & (thrust | coast_warp)
        torque = self._goto_torque_breaks(direction, tf, kap, search)
        torque = _where(search, torque, (torch.zeros_like(ax),) * 3)
        torque0, roll0 = self.torque, self.rot_roll
        self.torque = torch.where(steer[:, None], _stack(torque), torque0)
        roll = self._goto_roll_breaks(torque, tf, kap)
        self.rot_roll = torch.where(steer, roll, roll0)
        rotation, angvel = self._apply_torque(torque, roll, self._tau())
        heading = transform_coord((0.0, 0.0, 1.0), m_rotation_q(rotation))
        cosangle = _fmin(torch.ones_like(accel), _fdiv(v3_dot(heading, a32), accel))
        h = _stack(v3_mul(heading, f32(f32(cosangle * cosangle) * accel)))
        steered = torch.where((cosangle <= 0.0)[:, None], 0.0, h)
        return torch.where((steer & thrust)[:, None], steered, a)

    def _phase1(self, live: torch.Tensor) -> None:
        """Phase 1 (EvolveBehaviorForBall, Ballpark.cpp:789) for every live ball in ascending id, as tensors:
        vec_combat_cuda's prepass and phase-1 kernels. A member of a ball that starts warp proper this tick (Destiny's
        test, follow_ptr and the follow id: a ball SetBoid took out of a follow mode too) is released
        before its own turn when the target's id is lower (it then behaves in its new mode), after it otherwise; a ball
        reading its target's velocity sees it after the target's STOP damping when the target's turn came first."""
        do = self.dynamical_orientation
        n = self.n
        t = self.follow_ptr
        t_ok = (t >= 0) & (t < n)
        ts = torch.where(t_ok, t, 0)
        act = self._activates(live)
        pos0, vel0 = self.pos.clone(), self.vel.clone()
        hit = self._members(act) & self._existing()  # StopAllFollowers' members (Destiny's test)
        before, after = self._release_turns(hit, ts)
        # the prepass: the vertical velocity each ball shows after its own turn (its mode at its turn)
        m_turn = torch.where(before, self._released_mode(), self.mode)
        vy = vel0[:, 1]
        damp = live & (m_turn == STOP) & (not do)
        vy_after = torch.where(damp, (vy - 0.07 * vy) * STOP_DAMP, vy)
        vy_seen = torch.where(self.ids[ts] < self.ids, vy_after[ts], vel0[ts, 1])
        # the releases before the follower's own turn
        self._release_rows(before)
        # the behaviours
        mode = self.mode
        t = self.follow_ptr
        t_ok = (t >= 0) & (t < n)
        ts = torch.where(t_ok, t, 0)
        a = torch.zeros_like(self.pos)
        warp = live & (mode == WARP)
        proper = warp & (self.effect_stamp >= 0)
        activate = warp & ~proper & act
        align = warp & ~proper & ~act
        self.effect_stamp = torch.where(align, self.effect_stamp - 1, self.effect_stamp)
        self._real_warp(activate)
        go = live & (mode == GOTO)
        a = torch.where((go | align)[:, None], self._thrust_to(self.goto, torch.ones_like(go)), a)
        follow = live & (mode == FOLLOW) & t_ok
        if bool(follow.any()):
            op = pos0[ts]
            delta = self.pos - op
            dist = _len(delta)
            r = (self.follow_range + self.radius) + self.radius[ts]
            inv = torch.ones_like(dist) / torch.where(dist == 0.0, 1.0, dist)
            on = torch.stack((op[:, 0] + 1.0 * r, op[:, 1] + 0.0 * r, op[:, 2] + 0.0 * r), dim=1)
            target = torch.where((dist == 0.0)[:, None], on, op + (delta * r[:, None]) * inv[:, None])
            self.goto = torch.where(follow[:, None], target, self.goto)
            a = torch.where(follow[:, None], self._thrust_to(self.goto, torch.ones_like(go)), a)
        orbit = live & (mode == ORBIT) & t_ok
        if bool(orbit.any()):
            op = pos0[ts]
            if self.use_new_orbit:
                ov = vel0[ts].clone()
                ov[:, 1] = vy_seen
                th, gp = self._new_orbit(orbit, op, ov, self.radius[ts])
            else:
                th, gp = self._old_orbit(orbit, op, self.radius[ts])
            self.goto = torch.where(orbit[:, None], gp, self.goto)
            a = torch.where(orbit[:, None], th, a)
        a = self._phase1_combat(a, live, mode, t_ok, ts, pos0, vel0, vy_seen)
        stop = live & (mode == STOP)
        if not do:
            v = self.vel.clone()
            v[:, 1] = torch.where(stop, (vy - 0.07 * vy) * STOP_DAMP, v[:, 1])
            self.vel = v
        else:
            steer = live & ~(self.disable_do_for_missiles & (mode == MISSILE))
            a = self._orient_composed(a, steer)
        self.last_g = torch.where(live[:, None], a, self.last_g)
        self._clear_contacts(live)
        # the releases after the follower's own turn (its thrust stands)
        self._release_rows(after)

    def _warp_path(self, rows: torch.Tensor) -> tuple:
        """WarpDistance (Ballpark.cpp:4260; TRV-708 to TRV-713) with SetupWarpConstants, for the rows in warp proper:
        (position, velocity, dropped out). The logs on the host; the exps glibc's (vec_exp)."""
        wf = self.owner_id.to(F64)
        wd = self.last_collision
        s = (wf * WARP_FACTOR_TO_AU_PER_SECOND) * AU
        acc = wf * WARP_FACTOR_TO_ACCELERATION
        d0 = wf * WARP_FACTOR_TO_DECELERATION
        dec = torch.where(2.0 < d0, 2.0, d0)
        cap = (((wd + 1.0) * acc) * dec) / (acc + dec)
        s = torch.where(cap < s, cap, s)
        t_a = _host(math.log, s / acc, rows) / acc
        t_c = (((wd / s) - (torch.ones_like(acc) / acc)) - (torch.ones_like(dec) / dec)) + (torch.ones_like(s) / s)
        t_d = _host(math.log, s / dec, rows) / dec
        d_a, d_c, d_d = s / acc, s * t_c, s / dec - 1.0
        tt = (self.clock - self.effect_stamp).to(F64) * self.dt
        p, v = self.pos, self.vel
        d = _unit(self.goto - p)
        total = (d_a + d_c) + d_d
        b1 = tt < t_a
        b2 = ~b1 & ((tt - t_a) < t_c)
        b3 = ~b1 & ~b2 & (((tt - t_c) - t_a) < (t_d + 1.0))
        b4 = ~b1 & ~b2 & ~b3
        e1 = vec_exp.exp(acc * tt)
        speed1 = acc * e1
        cur = _len(v)
        speed1 = torch.where(cur > speed1, cur, speed1)
        v1 = d * speed1[:, None]
        p1 = self.goto - d * (total - e1)[:, None]
        v2 = d * s[:, None]
        p2 = self.goto - d * (total - (s * (tt - t_a) + d_a))[:, None]
        e3 = vec_exp.exp((-dec) * ((tt - t_c) - t_a))
        speed3 = s * e3
        v3 = d * speed3[:, None]
        sd = s / dec
        p3 = self.goto - d * (total - (((sd - sd * e3) + d_a) + d_c))[:, None]
        half = _div(self.max_vel, 2.0)
        drop3 = speed3 < torch.where(DROP_SPEED < half, DROP_SPEED, half)
        pw = torch.where(b1[:, None], p1, torch.where(b2[:, None], p2, torch.where(b3[:, None], p3, self.goto)))
        vw = torch.where(b1[:, None], v1, torch.where(b2[:, None], v2, torch.where(b3[:, None], v3, 0.0)))
        drop = rows & ((b3 & drop3) | b4)
        return pw, vw, drop

    def _integrate_rows(self, p, v, a) -> tuple:
        m = (self.mass * self.agility)[:, None]
        return self._integrate(p, v, a, m, self.time_factor[:, None])

    def _at_rest(self, dropped: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
        """The drop-outs re-filed at their drop-out point: at rest, |v|^2 < 0.1 (modes.py:711)."""
        return dropped & ~(_dot(wv, wv) >= REST_SQ)

    def _existing(self) -> torch.Tensor:
        """The rows holding a ball (the hook in `_phase1`; VecCombatPark: `exists`): every row here."""
        return torch.ones(self.n, dtype=torch.bool, device=self.dev)

    def _phase1_combat(self, a, live, mode, t_ok, ts, pos0, vel0, vy_seen) -> torch.Tensor:
        """Phase 1's MISSILE and FORMATION thrusts (the hook in `_phase1`; VecCombatPark's): none here."""
        return a

    def _clear_contacts(self, live: torch.Tensor) -> None:
        """Phase 1's clearing of the collision lists (the hook in `_phase1`; VecComposedPark's): no lists
        here."""

    def evolve_torch(self, active: torch.Tensor | None = None) -> None:
        """One tick of every world on tensors: the main kernel's algorithm
        (vec_modes_cuda.cu's main_kernel), the tick on a CPU device and a cross-check of the kernel on a CUDA one.
        Phase 1 (`_phase1`: a warper's followers released before or after their own turn by their ids, every mode's
        behaviour, dynamical orientation with a warper in warp proper turning toward its exit point), mLastCollision of
        a massive ball (the gradient without neighbours), the warp paths and the drop-outs (massive and sensing again,
        Stop, their state the drop-out point's), phases 3 and 4 (the warp path or the integrator with mLastG + 0, the
        rotation a tick on, the commit: the new rotation and angular velocity where finite), PetrifyTroll of each troll
        due, and every row's clock. The prepass's scratch (pos0, vel0, vy_after, act) is the kernel's own and is left as
        it is. `active` [n]: OnTick's per-world mask, as `evolve`'s (vec_orient.masked_tick, in
        place)."""
        if active is not None:
            masked_tick(self, active, lambda: VecModesPark.evolve_torch(self), in_place=True)
            return
        live = self.free & ~self.moribund
        do = self.dynamical_orientation
        self._phase1(live)
        warping = live & (self.mode == WARP) & (self.effect_stamp >= 0)
        self.last_collision = torch.where(live & self.massive, -1.0, self.last_collision)
        wp, wv, dropped = self._warp_path(warping)
        # the drop-out: massive and sensing again, Stop (no follower: a warper is none), the drop-out point's state
        self.massive = torch.where(dropped, True, self.massive)
        self.sensor = torch.where(dropped, True, self.sensor)
        self._stop_rows(dropped)
        self.pos = torch.where(dropped[:, None], wp, self.pos)
        self.vel = torch.where(dropped[:, None], wv, self.vel)
        # phase 3: the end-of-tick states (mLastG + mLastC, zero without collisions: -0 becomes +0); the rotation
        pn, vn = self._integrate_rows(self.pos, self.vel, self.last_g + 0.0)
        coast = warping & ~dropped
        pn = torch.where(coast[:, None], wp, pn)
        vn = torch.where(coast[:, None], wv, vn)
        if do:
            rot1, w1 = self._apply_torque(_cols(f32(self.torque)), f32(self.rot_roll), self._tau())
            rot1, w1 = _stack(rot1), _stack(w1)
        # phase 4: the commit of every live ball
        sel = live[:, None]
        self.old_pos = torch.where(sel, self.pos, self.old_pos)
        self.old_vel = torch.where(sel, self.vel, self.old_vel)
        self.pos = torch.where(sel, pn, self.pos)
        self.vel = torch.where(sel, vn, self.vel)
        if do:
            q = rot1
            rot_ok = torch.isfinite(f32(f32(f32(f32(q[:, 0] * q[:, 0]) + f32(q[:, 1] * q[:, 1]))
                                            + f32(q[:, 2] * q[:, 2])) + f32(q[:, 3] * q[:, 3])))
            w_ok = torch.isfinite(f32(f32(f32(w1[:, 0] * w1[:, 0]) + f32(w1[:, 1] * w1[:, 1]))
                                      + f32(w1[:, 2] * w1[:, 2])))
            r0, w0 = self.rot, self.ang_vel
            self.rot = torch.where(sel & rot_ok[:, None], rot1, r0)
            self.old_rot = torch.where(sel & rot_ok[:, None], r0, torch.where(sel, rot1, self.old_rot))
            self.ang_vel = torch.where(sel & w_ok[:, None], w1, w0)
            self.old_ang_vel = torch.where(sel & w_ok[:, None], w0, torch.where(sel, w1, self.old_ang_vel))
        # PetrifyTroll (Ballpark.cpp:6327; MOV-736) of each live troll due (TrollReady, 6315: not effect_stamp > clock),
        # after the commit, as the kernel does it: SetBallFree(false) (Stop(Ball*), a troll following nothing;
        # SetBallVelocity(0, 0, 0), the new and old velocity, whose rotation snap leaves a ball at rest as it is; out of
        # the free set; mLastG cleared), SetBallInteractive(false), RIGID
        due = live & (self.mode == TROLL) & ~(self.effect_stamp > self.clock)
        self.vel = torch.where(due[:, None], 0.0, self.vel)
        self.old_vel = torch.where(due[:, None], 0.0, self.old_vel)
        self.last_g = torch.where(due[:, None], 0.0, self.last_g)
        self.free = self.free & ~due
        self.interactive = self.interactive & ~due
        self.mode = torch.where(due, RIGID, self.mode)
        self.clock = self.clock + 1
        self.current_time += 1

    # ---- one tick (Ballpark::Evolve, TIME-706, TIME-707) ----

    def evolve(self, lib: dict | None = None, active: torch.Tensor | None = None) -> None:
        """One tick of every world, in place: on a CUDA device the prepass and the main kernel (vec_modes_cuda.tick), no
        host sync; on any other device the same tick on tensors (`evolve_torch`; `lib`, a kernel
        library, is refused there).
        `active` [n] (OnTick's per-world mask, a world's rows together): only those rows tick; every
        other row keeps every per-row tensor, its world's clock included, bit for bit (vec_orient.masked_tick: the
        kernel runs for every row, whose reads stay in its world, and the other rows are put back in place; no host
        sync, so a CUDA graph captures it)."""
        if self.dev.type != "cuda":                 # the tick on tensors off a CUDA device
            if lib is not None:
                raise ValueError("lib is a kernel library: a park on the CPU ticks on tensors (evolve_torch)")
            self.evolve_torch(active)
            return
        if active is not None:
            masked_tick(self, active, lambda: VecModesPark.evolve(self, lib), in_place=True)
            return
        K.tick(self, self.block, lib)
        self.current_time += 1

    def _world_split(self, x: torch.Tensor) -> torch.Tensor:
        """Whether some world's rows hold different values of x [n] (a 0-dim bool tensor, no host
        synchronisation): the rows sorted by world, neighbours of one world compared."""
        order = torch.argsort(self.world, stable=True)
        w, v = self.world[order], x[order]
        return ((w[1:] == w[:-1]) & (v[1:] != v[:-1])).any()

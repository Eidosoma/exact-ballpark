# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's flight batched on the GPU: GOTO, STOP, FOLLOW, ORBIT (both laws) and WARP for B balls, with the modes'
transitions, the setters and dynamical orientation, one CUDA kernel a tick (exact_ballpark.vec_modes_cuda).

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
tensors: only correctly rounded operations in the reference's order, tensor divisors.

Not here: collisions (exact_ballpark.vec_collide), the partition, bubbles and sensors (every bubble live, no events),
missiles, formations, mushrooms, the yaw, pitch and roll without dynamical orientation (MOV-729: display only, never
fed back into these modes' flight; exact_ballpark.vec_ypr_cuda has them), OnTick's time stamps (so SetBallFree's
mNewTime is not kept). VecCombatPark and VecComposedPark batch the
commands for their own ticks too (their kernels and torch ticks hold the troll's petrification and the follower sets);
`_g11` refuses them only for a park whose tick is none of the three.
"""

from __future__ import annotations

import torch

from . import vec_libm as L
from . import vec_modes_cuda as K
from .modes import AU, BOID, FOLLOW, INERT_MODES, MIN_WARP_DISTANCE_SQ, ORBIT, RIGID, TROLL, WARP
from .ref import GOTO, STOP
from .vec_orient import VecOrientPark, _cols, _stack, masked_tick, q_mul, q_rotation, v3_len

F64, I64 = torch.float64, torch.int64
f32 = L.f32
LOADABLE = (STOP,) + tuple(INERT_MODES)            # AddBall's STOP, or an inert mode (ModeBallpark.add_ball)
TROLL_MIN_DELAY = 1                                 # SetBallTroll (Ballpark.cpp:6300); a test holds it to full.py's


class VecModesPark(VecOrientPark):
    """B balls, one per row, in independent worlds, with Destiny's flight modes; evolve is one CUDA kernel a tick."""

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
        if type(self).evolve is not VecModesPark.evolve:
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

    # ---- one tick (Ballpark::Evolve, TIME-706, TIME-707) ----

    def evolve(self, lib: dict | None = None, active: torch.Tensor | None = None) -> None:
        """One tick of every world, in place: the prepass and the main kernel (vec_modes_cuda.tick), no host sync.
        `active` [n] (OnTick's per-world mask, a world's rows together): only those rows tick; every
        other row keeps every per-row tensor, its world's clock included, bit for bit (vec_orient.masked_tick: the
        kernel runs for every row, whose reads stay in its world, and the other rows are put back in place; no host
        sync, so a CUDA graph captures it)."""
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

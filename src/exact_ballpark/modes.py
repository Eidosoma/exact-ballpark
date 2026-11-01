# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's FOLLOW, ORBIT and WARP modes, added to the scalar movement reference (ref.py), one ball at a time.

A transcription of Carbon's `destiny` (github.com/carbonengine/destiny at 114e89fa, MIT, (c) CCP Games), as ref.py. `ModeBallpark` extends ref.py's `Ballpark` without changing it: GOTO and STOP keep
ref.py's code (`goto_thrust`, `integrate`, the STOP damping), and a tick follows `Ballpark::Evolve` (Ballpark.cpp:421):
every free ball's behaviour first (`EvolveBehaviorForBall`, Ballpark.cpp:789), then its end-of-tick state into
temporaries (the warp path in warp proper, the closed-form integrator otherwise), then the commit, balls in ascending
id. A behaviour reads other balls' state as it stands at that moment of the tick, so a follower released in the same
tick, or a target whose STOP damping already ran, is seen as Destiny sees it. Rules implemented, by rule id (all
but MOV-730 and TRV-716 marked passing against the compiled code):

- MOV-714: FOLLOW aims at the point at centre distance r = followRange + both radii on the line from the target, and
  thrusts by the goto law (`EvolveFollow`, Ballpark.cpp:1073).
- MOV-715: `FollowBall` and `Orbit` (Ballpark.cpp:3879, 4007): ignored for a non-finite range, src == dst, a missing
  ball, a moribund, cloaked or other-bubble target; otherwise Stop, the follow fields, the mode and the follower set.
  The range defaults to 1.0, the Python thunkers' default (Thunkers.cpp:250, 336).
- MOV-716: the orbit law is chosen by the process-global useNewOrbit (`use_new_orbit`, default False).
- MOV-717, MOV-718: the old-style orbit (`EvolveOldStyleOrbit`, Ballpark.cpp:1246).
- MOV-719, MOV-720: the new-style orbit (`GetOrbitalNormal`, `DefaultOrbitalNormal`, `EvolveNewStyleOrbit`;
  Ballpark.cpp:664, 735, 1134), dynamical orientation off.
- MOV-721, MOV-722: `GotoThrustFollow` and `OrbitThrust` (Ballpark.cpp:1463, 1428).
- MOV-730, in part: `StopAllFollowers` (Ballpark.cpp:5416) when a ball starts warp proper (cloaking and removal, the
  rule's other triggers, are later packages; its missile branch is coded but unreachable without MISSILE).
- MOV-735: `Ball::SetMode`'s events, DoModeChange and OnExitWarp (Ball.cpp:905), kept in `events` with the other
  events Destiny posts in these paths.
- TRV-701 to TRV-704: `WarpTo` (Ballpark.cpp:4072), with the thunker's defaults (minimum range 20 km, warp factor 20;
  Thunkers.cpp:366-367).
- TRV-705, TRV-706: aligning and the activation test (`EvolveWarp`, Ballpark.cpp:915; `Ball::IsAlignedForWarp`,
  Ball.cpp:1900).
- TRV-707: `RealWarp` (Ballpark.cpp:4143).
- TRV-708 to TRV-713: the warp path and dropping out (`SetupWarpConstants`, `WarpDistance`; Ballpark.cpp:4167, 4260).
- TRV-714: the drop-out tick integrates one step more from the drop-out point (Ballpark.cpp:478-496).
- TRV-715: `EntityWarpIn` (Ballpark.cpp:4370).
- TRV-716 (inferred): no ballpark-wide warp speed enters; a warp's speed comes from the ball's own warp factor.

Carried for these rules (MOV-733, MOV-734 in part): the flags is_free (only free balls are evolved), is_massive and the
proximity sensor's `active` (a warp clears and restores them), is_interactive (MOV-730), and the target conditions the
guards read (is_cloaked, bubble, is_moribund, set by the caller: cloaking, bubbles and removal are later packages).
The modes BOID, MINIBALL, FIELD and RIGID get zero thrust, as Destiny's default branch; MISSILE, FORMATION, MUSHROOM and
TROLL are refused (not in this slice). Where Destiny re-inserts a ball into its partition, an event "partition_insert"
marks the instant (no partition is modelled here). Collisions find no neighbours (the gradient leaves mLastC zero),
every bubble is live, and the dynamical-orientation and iterative-collision switches are off, as Destiny's defaults.
Those are the tick's hooks (`_free_ids`, `_in_dead_bubble`, `_gradient`, `_insert_ball_in_boxes`, `_commit`,
`_handle_proximities`, `_follower_order`, `_collision_responses`, `_position_velocity`), which `exact_ballpark.space`
fills with Destiny's partition, bubbles, collisions and sensors.

Python raises on a zero divisor where C gives an infinity or NaN; the one such division a legal command reaches, the
new orbit's angle for a ball at rest with zero cruise speed, follows IEEE (`_ieee_div`). Degenerate inputs raise here: a
massless ball, and an old-style orbiter exactly on its target's centre with r = 0, where Destiny divides 0 by 0
(Ballpark.cpp:1290) and on the next tick converts the NaN to an integer (Ballpark.cpp:1301), which C leaves undefined.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import struct
from collections import Counter
from dataclasses import dataclass, field, fields

from .ref import GOTO, STOP, STOP_DAMP, Ball, Ballpark, _normalized, f32

# DSTBALLMODE (IDstConstants.h:11-26); GOTO 0 and STOP 2 come from ref.py
FOLLOW, WARP, ORBIT, MISSILE, MUSHROOM, BOID = 1, 3, 4, 5, 6, 7
TROLL, MINIBALL, FIELD, RIGID, FORMATION = 8, 9, 10, 11, 12
FOLLOW_MODES = (FOLLOW, ORBIT, MISSILE, FORMATION)          # followModes (Ballpark.cpp:68)
INERT_MODES = (BOID, MINIBALL, FIELD, RIGID)                # zero thrust, plain integration (MOV-734)
UNSUPPORTED_MODES = (MISSILE, FORMATION, MUSHROOM, TROLL)   # later packages

AU = .1495978707e12                                         # Ballpark.h:31 (TRV-704)
ORBITAL_PRECESSION = 0.001                                  # Ballpark.cpp:70 (MOV-717), per tick
WARP_FACTOR_TO_AU_PER_SECOND = 0.001                        # Ballpark.cpp:61-63 (TRV-708)
WARP_FACTOR_TO_ACCELERATION = 1.0 / 1000
WARP_FACTOR_TO_DECELERATION = 1.0 / 3000
MAX_ALIGN_TICKS = 180                                       # Ball.h:23 (TRV-706)
MIN_WARP_DISTANCE_SQ = 10000000000.0                        # Ballpark.cpp:4104 (TRV-701)


def _lensq(v) -> float:
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def _length(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _cross(u, v) -> list:
    """Vector3d::Cross (Vector3d.h:74): u x v, components in Destiny's order."""
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def _trunc7(x: float) -> float:
    """`(double)((int64_t)(x*10000000))/10000000`: truncation toward zero to 7 decimals (MOV-717, MOV-718)."""
    return float(int(x * 10000000)) / 10000000


def _ieee_div(a: float, b: float) -> float:
    """C's double division, which gives an infinity or NaN where Python raises on a zero divisor."""
    if b == 0.0:
        if a == 0.0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _float32(x: float) -> float:
    """A double converted to a C++ float parameter; out-of-range values become infinities, as the hardware does."""
    try:
        return f32(x)
    except OverflowError:
        return math.copysign(math.inf, x)


_LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
_LIBM.sincos.restype = None
_LIBM.sincos.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]


def _sincos(x: float) -> tuple:
    """The C library's sincos(x): a sine and a cosine of one argument as Destiny's compiled code takes them (g++ -O2
    makes one sincos call of sin(x) and cos(x)); glibc 2.34's differs from its sin at +-0x1.ad1fb54442d18p+0 (exact_ballpark.dlibm.sincos is its transcription). An infinity gives NaNs, as C does."""
    s, c = ctypes.c_double(), ctypes.c_double()
    _LIBM.sincos(float(x), ctypes.byref(s), ctypes.byref(c))
    return s.value, c.value


def _bits_of(x: float) -> int:
    """`*((int64_t *)&x)`: a double's bits as a signed 64-bit integer (WarpTo keeps the minimum range so, TRV-703)."""
    return struct.unpack("<q", struct.pack("<d", x))[0]


def _double_of(i: int) -> float:
    """`*((double *)&i)`: the inverse of `_bits_of` (RealWarp reads the minimum range back, TRV-707)."""
    return struct.unpack("<d", struct.pack("<q", i))[0]


@dataclass
class ModeBall(Ball):
    """ref.py's Ball with the fields these modes use, with Destiny's types and the Ball constructor's defaults."""
    follow_id: int = 0                  # ID; in WARP the bits of the minimum range (TRV-703)
    follow_ptr: int | None = None       # mFollowPtr, by id
    follow_range: float = 10.0          # float (MOV-712); Ball.cpp: mFollowRange(10.0)
    owner_id: int = 0                   # ID; in WARP the warp factor (TRV-703)
    effect_stamp: int = 0               # long; the aligning counter, then the warp start tick
    last_collision: float = -1.0        # double; in warp proper the warp length (TRV-707)
    last_c: list = field(default_factory=lambda: [0.0, 0.0, 0.0])   # the collision acceleration: zero here
    is_free: bool = True
    is_massive: bool = False
    is_interactive: bool = False
    is_cloaked: int = 0
    is_moribund: bool = False
    bubble: int = -1                    # Partitionable: mNewBubble(-1)
    sensor_active: bool = False         # mSensor.active
    followers: set = field(default_factory=set)


class ModeBallpark(Ballpark):
    """ref.py's Ballpark with FOLLOW, ORBIT and WARP; the server's view (is_master) unless told otherwise."""

    def __init__(self, friction: float = 1.0e6, tick_interval_ms: int = 1000, use_new_orbit: bool = False,
                 is_master: bool = True):
        super().__init__(friction, tick_interval_ms)
        self.use_new_orbit = use_new_orbit      # g_useNewOrbit (Settings.h:13; MOV-716)
        self.iterative_collision = False        # g_useIterativeCollision (Settings.h:10): off, Destiny's default
        self.is_master = is_master
        self.events: list[tuple] = []           # (current_time, name, ball id, a, b)
        self.trace: Counter | None = None       # branch counts, for the tests' coverage checks

    def _hit(self, name: str) -> None:
        if self.trace is not None:
            self.trace[name] += 1

    def _event(self, name: str, bid: int, a: int = 0, b: int = 0) -> None:
        self.events.append((self.current_time, name, bid, a, b))

    # ---- balls ----

    def add_ball(self, bid: int, mass: float, radius: float, max_vel: float, pos, vel, agility: float,
                 speed_fraction: float, *, is_free: bool = True, is_massive: bool = False,
                 is_interactive: bool = False, is_cloaked: int = 0, bubble: int = -1, is_moribund: bool = False,
                 mode: int = STOP) -> ModeBall:
        """AddBall's clamps (ref.py, MOV-711), then the flags; `mode` may be STOP or an inert mode."""
        if mode != STOP and mode not in INERT_MODES:
            raise ValueError(f"a ball starts in STOP or an inert mode, not {mode}")
        b = super().add_ball(bid, mass, radius, max_vel, pos, vel, agility, speed_fraction)
        mb = ModeBall(**{f.name: getattr(b, f.name) for f in fields(Ball)}, is_free=is_free, is_massive=is_massive,
                      is_interactive=is_interactive, is_cloaked=is_cloaked, bubble=bubble, is_moribund=is_moribund)
        mb.mode = mode
        self.balls[bid] = mb
        return mb

    # ---- mode changes (MOV-735) ----

    def _set_mode(self, b: ModeBall, mode: int) -> None:
        """Ball::SetMode (Ball.cpp:905): OnExitWarp(id, 0) when leaving WARP, then DoModeChange(old, new)."""
        if mode != b.mode:
            if mode == WARP:
                pass
            elif b.mode == WARP:
                self._event("OnExitWarp", b.id, 0)
            self._event("DoModeChange", b.id, b.mode, mode)
        b.mode = mode

    def _stop(self, b: ModeBall) -> None:
        """Stop(Ball*) (Ballpark.cpp:4578): leave the target's follower set and clear the follow fields, then STOP."""
        if b.mode in FOLLOW_MODES:
            if self.is_master:
                self.balls[b.follow_ptr].followers.discard(b.id)
            else:                                    # the client checks that pointer and id agree first
                by_id = self.balls.get(b.follow_id)
                if by_id is not None and b.follow_ptr == b.follow_id:
                    by_id.followers.discard(b.id)
            b.effect_stamp = 0
            b.follow_ptr = None
            b.follow_id = 0
            b.owner_id = 0
            b.follow_range = 0.0
        self._set_mode(b, STOP)

    # ---- commands (MOV-708, MOV-709 as ref.py, now with the follower bookkeeping and events) ----

    def stop(self, bid: int) -> None:
        """Stop(const ID&) (Ballpark.cpp:4556)."""
        b = self.balls.get(bid)
        if b is None or b.mode == STOP:
            return
        self._stop(b)

    def goto_point(self, bid: int, x: float, y: float, z: float) -> None:
        """GotoPoint(const ID&, ...) (Ballpark.cpp:4511)."""
        b = self.balls.get(bid)
        if b is None:
            return
        self._goto_point(b, [float(x), float(y), float(z)])

    def _goto_point(self, b: ModeBall, p: list) -> None:
        """GotoPoint(Ball*, ...) (Ballpark.cpp:4529)."""
        if not all(math.isfinite(c) for c in p):
            return
        self._stop(b)
        b.goto = list(p)
        if b.speed_fraction == 0.0:
            b.speed_fraction = 1.0
        self._set_mode(b, GOTO)

    def goto_direction(self, bid: int, x: float, y: float, z: float) -> None:
        """GotoDirection(const ID&, ...) (Ballpark.cpp:4471)."""
        self._goto_direction(self.balls.get(bid), [float(x), float(y), float(z)])

    def _goto_direction(self, b: ModeBall | None, d: list) -> None:
        """GotoDirection(Ball*, ...) (Ballpark.cpp:4483): a point 1e17 m along the unit direction."""
        if b is None or not all(math.isfinite(c) for c in d):
            return
        d = _normalized(d)
        self._goto_point(b, [b.pos[i] + d[i] * 1.0e17 for i in range(3)])

    def follow_ball(self, src: int, dst: int, range_: float = 1.0) -> None:
        """FollowBall (Ballpark.cpp:3879; MOV-715)."""
        self._follow(src, dst, range_, FOLLOW)

    def orbit(self, src: int, dst: int, range_: float = 1.0) -> None:
        """Orbit (Ballpark.cpp:4007; MOV-715): the same checks and bookkeeping as FollowBall, in ORBIT."""
        self._follow(src, dst, range_, ORBIT)

    def _follow(self, src: int, dst: int, range_: float, mode: int) -> None:
        rng = _float32(range_)                     # `float range`
        if not math.isfinite(rng):
            return
        ball = self.balls.get(src)
        if ball is None or src == dst:
            return
        dest = self.balls.get(dst)
        if dest is None or dest.is_moribund or ball.bubble != dest.bubble or dest.is_cloaked:
            return
        self._stop(ball)
        ball.follow_id = dst
        ball.follow_ptr = dst
        ball.follow_range = rng
        self._set_mode(ball, mode)
        dest.followers.add(ball.id)

    def warp_to(self, bid: int, x: float, y: float, z: float, min_range: float = 20000.0,
                warp_factor: int = 20) -> None:
        """WarpTo (Ballpark.cpp:4072; TRV-701 to TRV-704)."""
        x, y, z, min_range, warp_factor = float(x), float(y), float(z), float(min_range), int(warp_factor)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return
        if min_range < 0.0:
            min_range = 0.0
        if warp_factor <= 0:
            warp_factor = 1
        b = self.balls.get(bid)
        if b is None:
            return
        dst = [x, y, z]
        if _lensq([b.pos[0] - x, b.pos[1] - y, b.pos[2] - z]) < MIN_WARP_DISTANCE_SQ:
            self._hit("warp.short_goto")
            self._goto_point(b, dst)
            self._event("OnExitWarp", b.id, 1)
            return
        if b.mode == WARP and dst == b.goto:        # TRV-702: the align counter is kept
            self._hit("warp.repeat_noop")
            return
        self._stop(b)
        b.goto = dst
        b.effect_stamp = -1
        b.follow_range = 0.0
        b.follow_id = _bits_of(min_range)
        b.owner_id = warp_factor
        self._set_mode(b, WARP)

    def entity_warp_in(self, bid: int, x: float, y: float, z: float, warp_factor: int) -> None:
        """EntityWarpIn (Ballpark.cpp:4370; TRV-715): WarpTo, then warp proper at once, begun five ticks ago."""
        self.warp_to(bid, x, y, z, 0.0, warp_factor)
        b = self.balls.get(bid)
        if b is None:
            return
        if b.mode == GOTO:
            self._hit("warpin.goto")
            return
        self._hit("warpin.warp")
        self._real_warp(b)
        d = _normalized([float(x), float(y), float(z)])
        b.vel = [d[0] * AU, d[1] * AU, d[2] * AU]
        b.effect_stamp = max(self.current_time - 5, 0)
        b.last_collision = _length([b.pos[i] - b.goto[i] for i in range(3)])
        if self.is_master:
            self._event("OnActivatingWarp", b.id, self.current_time)

    # ---- warp state (Ball.cpp:1885-1925) ----

    @staticmethod
    def _is_aligning(b: ModeBall) -> bool:
        return b.mode == WARP and b.effect_stamp < 0

    @staticmethod
    def _is_warping(b: ModeBall) -> bool:
        return b.mode == WARP and not b.effect_stamp < 0

    def _is_aligned_for_warp(self, b: ModeBall) -> bool:
        """TRV-706: within |1 - cos| < 0.01 of the goto direction above 75% of maxVel, or past 180 aligning ticks."""
        if not self._is_aligning(b):
            return False
        d = _normalized([b.goto[i] - b.pos[i] for i in range(3)])
        u = _normalized(list(b.vel))
        if abs(1.0 - (u[0] * d[0] + u[1] * d[1] + u[2] * d[2])) < 0.01 and \
                _lensq(b.vel) > 0.5625 * b.max_vel * b.max_vel:
            self._hit("warp.align_ok")
            return True
        if abs(b.effect_stamp) > MAX_ALIGN_TICKS:
            self._hit("warp.align_forced")
            return True
        return False

    # ---- behaviours (phase 1) ----

    def _behaviour(self, b: ModeBall) -> None:
        """EvolveBehaviorForBall (Ballpark.cpp:789); CapAcceleration returns at once (MOV-727)."""
        b.last_g = [0.0, 0.0, 0.0]
        b.last_c = [0.0, 0.0, 0.0]
        a = [0.0, 0.0, 0.0]
        m = b.mode
        if m == WARP:
            a = self._evolve_warp(b)
        elif m == GOTO:
            a = self.goto_thrust(b)
        elif m == FOLLOW:
            a = self._evolve_follow(b)
        elif m == ORBIT:
            a = self._evolve_new_orbit(b) if self.use_new_orbit else self._evolve_old_orbit(b, self.current_time)
        elif m == STOP:
            b.vel[1] = (b.vel[1] - 0.07 * b.vel[1]) * STOP_DAMP      # EvolveStop (MOV-710)
        elif m in UNSUPPORTED_MODES:
            raise NotImplementedError(f"mode {m} is not in this slice")
        b.last_g = a

    def _evolve_warp(self, b: ModeBall) -> list:
        """EvolveWarp (Ballpark.cpp:915; TRV-705, TRV-707): zero in warp proper; activate, or align like a goto."""
        if self._is_warping(b):
            return [0.0, 0.0, 0.0]
        if self._is_aligned_for_warp(b):
            self._event("OnActivatingWarp", b.id, self.current_time)
            self._real_warp(b)
            return [0.0, 0.0, 0.0]
        self._hit("warp.align_wait")
        b.effect_stamp -= 1
        return self.goto_thrust(b)                 # EvolveGoto: GotoThrust toward mGoto

    def _real_warp(self, b: ModeBall) -> None:
        """RealWarp (Ballpark.cpp:4143; TRV-707): release the followers, set the exit point, start warp proper."""
        self._stop_all_followers(b)
        dst = list(b.goto)
        delta = _normalized([b.pos[i] - dst[i] for i in range(3)])
        mr = _double_of(b.follow_id)
        dst = [dst[i] + delta[i] * mr for i in range(3)]
        delta = [dst[i] - b.pos[i] for i in range(3)]
        b.goto = dst
        b.effect_stamp = self.current_time
        b.last_collision = _length(delta)
        b.is_massive = False
        b.sensor_active = False

    def _follower_order(self, b: ModeBall) -> list:
        """The order StopAllFollowers walks a copy of the follower set: an unordered_set in Destiny, whose order
        exact_ballpark.space keeps; ascending id here (each follower is handled alone)."""
        return sorted(b.followers)

    def _stop_all_followers(self, b: ModeBall) -> None:
        """StopAllFollowers (Ballpark.cpp:5416; MOV-730): missiles and interactive orbiters fly on, others stop."""
        if not b.followers:
            return
        for fid in self._follower_order(b):
            f = self.balls.get(fid)
            if f is None or f.follow_id != b.id or f.follow_ptr != b.id:
                b.followers.discard(fid)
                continue
            if f.mode == MISSILE or (f.is_interactive and f.mode == ORBIT):
                self._hit("release.goto_direction")
                if f.mode == MISSILE:
                    f.is_massive = False
                self._goto_direction(f, list(f.vel))
            else:
                self._hit("release.stop")
                self._stop(f)

    def _evolve_follow(self, b: ModeBall) -> list:
        """EvolveFollow (Ballpark.cpp:1073; MOV-714)."""
        other = self.balls[b.follow_ptr]
        op = list(other.pos)
        delta = [b.pos[0] - op[0], b.pos[1] - op[1], b.pos[2] - op[2]]
        dist = _length(delta)
        r = b.follow_range + b.radius + other.radius
        if dist == 0.0:
            self._hit("follow.coincident")
            target = [op[0] + 1.0 * r, op[1] + 0.0 * r, op[2] + 0.0 * r]
        else:
            inv = 1.0 / dist                       # Vector3d's operator/ multiplies by the reciprocal
            target = [op[i] + delta[i] * r * inv for i in range(3)]
        b.goto = target
        return self.goto_thrust(b)                 # GotoThrust, not in missile mode

    def _evolve_old_orbit(self, b: ModeBall, current_time: int) -> list:
        """EvolveOldStyleOrbit (Ballpark.cpp:1246; MOV-717, MOV-718)."""
        cruise = b.speed_fraction * b.max_vel
        max_thrust = self.friction * cruise / (b.mass * b.agility)
        other = self.balls[b.follow_ptr]
        op = list(other.pos)
        r = b.follow_range + b.radius + other.radius
        to = [op[0] - b.pos[0], op[1] - b.pos[1], op[2] - b.pos[2]]
        dist = _length(to)
        to = _normalized(to)
        phi1 = current_time * ORBITAL_PRECESSION
        phi2 = (b.id & 0x000000000000FFFF) + current_time * ORBITAL_PRECESSION
        (s1, c1), (s2, c2) = _sincos(phi1), _sincos(phi2)
        w = [c1 * c2, s2, s1 * c2]
        w = _normalized(_cross([_trunc7(c) for c in w], to))
        to_comp = dist * dist - r * r
        if to_comp >= 0.0:
            self._hit("orbit_old.tangent")
            rad_comp = r * math.sqrt(to_comp) / dist
            s = to_comp / dist
            to = _normalized([to[i] * s + w[i] * rad_comp for i in range(3)])
        else:
            self._hit("orbit_old.inside")
        rf = _trunc7(math.exp(-(r - dist) * (r - dist) / 40000.0))
        phi1 = (-to[0]) * w[0] + (-to[1]) * w[1] + (-to[2]) * w[2]
        tf = 1.0 + rf * rf * (phi1 * phi1 - 1.0)
        if tf > 0.0:
            tf = rf * phi1 + math.sqrt(tf)
        else:
            self._hit("orbit_old.real_part")
            tf = rf * phi1
        tf *= 1.0 if dist - r >= 0.0 else -1.0     # SIGNUM (Ballpark.cpp:50)
        a = [(w[i] * rf + to[i] * tf) * max_thrust for i in range(3)]
        b.goto = [b.pos[i] + a[i] * (10.0 * AU) for i in range(3)]
        return a

    def _default_orbital_normal(self, b: ModeBall) -> list:
        """DefaultOrbitalNormal (Ballpark.cpp:735): the precessing w vector of MOV-717, truncated, not normalised."""
        phi1 = self.current_time * ORBITAL_PRECESSION
        phi2 = (b.id & 0x000000000000FFFF) + self.current_time * ORBITAL_PRECESSION
        (s1, c1), (s2, c2) = _sincos(phi1), _sincos(phi2)
        w = [c1 * c2, s2, s1 * c2]
        return [_trunc7(c) for c in w]

    def _orbital_normal(self, b: ModeBall) -> tuple:
        """GetOrbitalNormal (Ballpark.cpp:664) with useNewOrbit on (its only caller) and dynamical orientation off
        (MOV-719): (unit vector to the predicted target, distance to it, the orbit's normal)."""
        other = self.balls[b.follow_ptr]
        to = [other.pos[i] - b.pos[i] for i in range(3)]
        to = [to[i] + other.vel[i] * self.dt for i in range(3)]
        dist = _length(to)
        if dist > 0:
            inv = 1.0 / dist
            to = [c * inv for c in to]
        else:
            self._hit("orbit_new.on_target")
            to = [0.0, 0.0, 1.0]
        o = _cross(to, b.vel)
        length2 = _lensq(o)
        if length2 > 1e-12:
            inv = 1.0 / math.sqrt(length2)
            return to, dist, [c * inv for c in o]
        self._hit("orbit_new.default_normal")
        return to, dist, self._default_orbital_normal(b)

    def _evolve_new_orbit(self, b: ModeBall) -> list:
        """EvolveNewStyleOrbit (Ballpark.cpp:1134; MOV-720): a logarithmic spiral toward the orbit radius."""
        cruise = b.speed_fraction * b.max_vel
        tau = (b.mass * b.agility) / self.friction
        max_thrust = cruise / tau
        other = self.balls[b.follow_ptr]
        r = b.follow_range + b.radius + other.radius
        to, dist, normal = self._orbital_normal(b)
        radial = _normalized(_cross(normal, to))
        max_speed = max(cruise, _length(b.vel))
        angle = _ieee_div(0.1 * (r - dist), max_speed)
        if angle < -1:
            self._hit("orbit_new.angle_clamp_low")
            angle = -1.0
        if angle > 1:
            self._hit("orbit_new.angle_clamp_high")
            angle = 1.0
        if dist < 0.1:
            self._hit("orbit_new.dist_floor")
            dist = 0.1
        est_v = min(max_speed, math.sqrt(dist * max_thrust))
        est_d = est_v * self.dt
        if abs(angle) < 1e-10:
            self._hit("orbit_new.circle")
            t = est_d / dist
            st, ct = _sincos(t)
            gp = [to[i] * (dist * (1.0 - ct)) + radial[i] * (dist * st) for i in range(3)]
            gv = [radial[i] * (ct * est_v) + to[i] * (st * est_v) for i in range(3)]
        else:
            factor = est_d * angle / dist / math.sqrt(angle * angle + 1)
            if factor < -0.999:
                self._hit("orbit_new.factor_floor")
                t = math.log(0.001) / angle
            elif abs(factor) > 1e-10:
                self._hit("orbit_new.factor_log")
                t = math.log(1 + factor) / angle
            else:
                self._hit("orbit_new.factor_linear")
                t = factor / angle
            dexp = dist * math.exp(angle * t)
            st, ct = _sincos(t)
            gp = [to[i] * (dist - dexp * ct) + radial[i] * (dexp * st) for i in range(3)]
            gv = _normalized([to[i] * (-angle * ct + st) + radial[i] * (angle * st + ct) for i in range(3)])
            gv = [c * est_v for c in gv]
        b.goto = [b.pos[i] + gp[i] for i in range(3)]
        gv = [gv[i] + other.vel[i] for i in range(3)]
        return self._goto_thrust_follow(b, b.goto, gv, 1.0)

    def _goto_thrust_follow(self, b: ModeBall, target: list, target_vel: list, weight: float) -> list:
        """GotoThrustFollow (Ballpark.cpp:1463; MOV-721): reach the point and match the velocity, capped."""
        speed = _length(target_vel)
        if speed < 5e-2 * b.max_vel * b.speed_fraction:
            if self.use_new_orbit and b.mode == ORBIT:
                self._hit("gtf.orbit_thrust")
                return self._orbit_thrust(b, target)
            return self.goto_thrust(b)             # unreachable: the new orbit (target = mGoto) is the only caller
        fom = self.friction / (b.mass * b.agility)
        max_accel = fom * b.speed_fraction * b.max_vel
        if max_accel <= 0:
            self._hit("gtf.no_accel")
            return [0.0, 0.0, 0.0]
        one_minus = 1 - b.time_factor
        inv = 1.0 / (self.dt * fom - one_minus)
        a = [((target[i] - b.pos[i]) * fom - b.vel[i] * one_minus) * fom * inv for i in range(3)]
        inv = 1.0 / (1.0 - b.time_factor)
        avel = [(target_vel[i] - b.vel[i] * b.time_factor) * fom * inv for i in range(3)]
        inv = 1.0 / (1.0 + weight)
        a = [(a[i] + avel[i] * weight) * inv for i in range(3)]
        accel = _lensq(a)
        if accel > max_accel * max_accel:
            self._hit("gtf.capped")
            s = max_accel / math.sqrt(accel)
            a = [c * s for c in a]
        else:
            self._hit("gtf.free")
        return a

    def _orbit_thrust(self, b: ModeBall, target: list) -> list:
        """OrbitThrust (Ballpark.cpp:1428; MOV-722): brake toward the point; zero below a tenth of max acceleration."""
        fom = self.friction / (b.mass * b.agility)
        max_accel = fom * b.speed_fraction * b.max_vel
        if max_accel <= 0:
            return [0.0, 0.0, 0.0]
        a = [(target[i] - b.pos[i]) * fom - b.vel[i] for i in range(3)]
        tacc = _length(a) / max_accel
        if tacc > 1:
            self._hit("orbit_thrust.capped")
            inv = 1.0 / tacc
            a = [c * inv for c in a]
        if tacc < 0.1:
            self._hit("orbit_thrust.off")
            return [0.0, 0.0, 0.0]
        return a

    # ---- the warp path (phase 3; TRV-708 to TRV-713) ----

    @staticmethod
    def warp_constants(warp_factor: float, warp_distance: float) -> tuple:
        """SetupWarpConstants (Ballpark.cpp:4167): (tA, tC, tD, dA, dC, dD, ACC, DEC, S), with S lowered for short
        warps so the cruise lasts zero (TRV-709)."""
        s = warp_factor * WARP_FACTOR_TO_AU_PER_SECOND * AU
        acc = warp_factor * WARP_FACTOR_TO_ACCELERATION
        dec = min(warp_factor * WARP_FACTOR_TO_DECELERATION, 2.0)
        s = min(s, (warp_distance + 1) * acc * dec / (acc + dec))
        t_a = math.log(s / acc) / acc
        t_c = (warp_distance / s) - (1.0 / acc) - (1.0 / dec) + (1.0 / s)
        t_d = math.log(s / dec) / dec
        return t_a, t_c, t_d, s / acc, s * t_c, s / dec - 1, acc, dec, s

    def _warp_distance(self, b: ModeBall, p: list, v: list, t: float) -> tuple:
        """WarpDistance (Ballpark.cpp:4260), not interpolating: the new position and velocity on the warp line, aimed
        from the current position at the exit point; drops the ball out of warp when it is slow enough (TRV-713)."""
        warp_factor = float(b.owner_id)
        t_a, t_c, t_d, d_a, d_c, d_d, acc, dec, s = self.warp_constants(warp_factor, b.last_collision)
        if self.trace is not None and s < warp_factor * WARP_FACTOR_TO_AU_PER_SECOND * AU:
            self._hit("warp.speed_limited")
        d = _normalized([b.goto[i] - p[i] for i in range(3)])
        if t < t_a:
            self._hit("warp.accel")
            speed = acc * math.exp(acc * t)
            cur = _length(v)
            if cur > speed:                        # the reported speed is kept continuous
                self._hit("warp.speed_kept")
                speed = cur
            v = [d[i] * speed for i in range(3)]
            x = math.exp(acc * t)
            rest = d_a + d_c + d_d - x
            p = [b.goto[i] - d[i] * rest for i in range(3)]
        elif (t - t_a) < t_c:
            self._hit("warp.cruise")
            v = [d[i] * s for i in range(3)]
            x = s * (t - t_a) + d_a
            rest = d_a + d_c + d_d - x
            p = [b.goto[i] - d[i] * rest for i in range(3)]
        elif (t - t_c - t_a) < (t_d + 1):
            self._hit("warp.decel")
            speed = s * math.exp(-dec * (t - t_c - t_a))
            v = [d[i] * speed for i in range(3)]
            x = s / dec - (s / dec) * math.exp(-dec * (t - t_c - t_a)) + d_a + d_c
            rest = d_a + d_c + d_d - x
            p = [b.goto[i] - d[i] * rest for i in range(3)]
            if speed < min(b.max_vel / 2.0, 100.0):
                self._hit("warp.drop_out")
                if not self.is_master:
                    self._event("OnDeactivatingWarp", b.id, self.current_time)
                b.is_massive = True
                b.sensor_active = True
                self._stop(b)
        else:                                      # past the deceleration phase by more than 1 s
            self._hit("warp.overrun")
            p = list(b.goto)
            v = [0.0, 0.0, 0.0]
            if not self.is_master:
                self._event("OnDeactivatingWarp", b.id, self.current_time)
            b.is_massive = True
            b.sensor_active = True
            self._stop(b)
        return p, v

    # ---- the tick's hooks: no partition, bubbles, collisions or sensors here (exact_ballpark.space has them) ----

    def _free_ids(self) -> list:
        """mFreeBalls in ascending id (TIME-707): here every ball flagged free."""
        return [bid for bid in sorted(self.balls) if self.balls[bid].is_free]

    def _in_dead_bubble(self, b: ModeBall) -> bool:
        """InDeadBubble (TIME-714): every bubble is live here."""
        return False

    def _gradient(self, b: ModeBall) -> None:
        """Gradient (COL-702 to COL-711): no neighbours here, so mLastC stays zero."""

    def _insert_ball_in_boxes(self, b: ModeBall) -> None:
        """InsertBallInBoxes: no partition here; the instant is kept as an event."""
        self._event("partition_insert", b.id)

    def _commit(self, b: ModeBall, p: list, v: list) -> None:
        """Phase 4's swap: the end-of-tick state becomes the ball's state."""
        b.pos, b.vel = p, v

    def _handle_proximities(self) -> None:
        """HandleProximities (SEN-701 to SEN-704): no sensors here."""

    def _collision_responses(self) -> None:
        """CalculateIterativeCollisionResponses (COL-714): reached only with the iterative method on."""

    def _position_velocity(self, b: ModeBall, p: list, v: list) -> tuple:
        """CalculateBallPositionVelocity (Ballpark.cpp:2715): the closed-form step from the tick's accelerations."""
        a = [b.last_g[i] + b.last_c[i] for i in range(3)]
        return self.integrate(p, v, a, b.mass * b.agility, self.friction, b.time_factor, self.dt)

    # ---- one tick (TIME-706, TIME-707) ----

    def evolve(self) -> None:
        order = self._free_ids()                             # mFreeBalls, ascending id
        dead = set()
        for bid in order:                                    # (1) behaviours
            b = self.balls[bid]
            if b.is_moribund:
                continue
            if self._in_dead_bubble(b):                      # mInDeadBubble: skipped for the whole tick
                dead.add(bid)
                continue
            self._behaviour(b)
        if self.iterative_collision:                         # (2)
            self._collision_responses()
        new = {}
        for bid in order:                                    # (3) end-of-tick states into temporaries
            b = self.balls[bid]
            if b.is_moribund or bid in dead:
                continue
            if b.is_massive and not self.iterative_collision:
                b.last_collision = -1.0
                self._gradient(b)
            p, v = b.pos, b.vel
            if self._is_warping(b):
                p, v = self._warp_distance(b, p, v, (self.current_time - b.effect_stamp) * self.dt)
                if b.mode == STOP:                           # TRV-714: the drop-out point becomes the old state
                    b.pos, b.vel = p, v
                    a = [b.last_g[i] + b.last_c[i] for i in range(3)]
                    p, v = self.integrate(p, v, a, b.mass * b.agility, self.friction, b.time_factor, self.dt)
                    if _lensq(b.vel) < 0.1:
                        self._hit("warp.drop_out_insert")
                        self._insert_ball_in_boxes(b)
            else:
                p, v = self._position_velocity(b, p, v)
            new[bid] = (p, v)
        for bid in order:                                    # (4) commit
            b = self.balls[bid]
            if b.is_moribund or bid in dead:
                continue
            old = b.pos
            self._commit(b, *new[bid])
            if not _lensq([b.pos[0] - old[0], b.pos[1] - old[1], b.pos[2] - old[2]]) < 0.1:
                self._insert_ball_in_boxes(b)
        self._handle_proximities()                           # (6)
        self.current_time += 1                               # (7)

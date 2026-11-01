# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's missiles, static collidables, cloaking, dynamical orientation and the orientation itself, mushrooms,
trolls, formations, the accuracy helpers and OnTick, added to the scalar movement reference (space.py), one ball at a
time.

A transcription of Carbon's `destiny` (github.com/carbonengine/destiny at 114e89fa, MIT, (c) CCP Games), as ref.py, modes.py and space.py, with the float vector, quaternion and matrix functions it calls
from carbonengine/math (CcpMath.h at 8273322, MIT, (c) CCP Games). `FullBallpark` extends `SpaceBallpark`; its tick is
`Ballpark::Evolve` (Ballpark.cpp:421) whole, with every branch the earlier slices refused. Rules implemented, by rule id:

- MOV-723, MOV-724, MOV-725: MISSILE mode (`EvolveMissile`, Ballpark.cpp:972), `MissileFollow` (3798) and the Python
  `LaunchMissile` (Thunkers.cpp:857), with GotoThrust's missile flag (1398).
- COL-719: missile collision events (`Ball::DispatchCollisions`, Ball.cpp:468), only for missiles that moved at least
  sqrt(0.1) m, with the master's self-collision for an owner gone, cloaked or in another bubble (Ballpark.cpp:610-639);
  in the iterative method the colliders of the missile's impact points.
- COL-703, COL-704 (their missile and static parts): the candidates of `Partition::GetCollisionCandidates` (a normal
  missile sees only its target, a defender missile everything but cloaked balls) and of `GetNearbyBalls` with static
  collidables (collected unless fixed balls are excluded; other bubbles excluded on the master; a moribund static
  collidable still collides); missiles and mushrooms ignore static collidables (Ballpark.cpp:2778, 2100).
- COL-712, COL-713: capsules (Capsule.cpp), oriented boxes (OrientedBox.cpp, BoxShape.cpp, Triangle.cpp, Plane.cpp,
  AABB.cpp, Collision.cpp), miniballs, minicapsules and miniboxes (`AddMiniball`, `AddCapsule`, `AddOrientedBox`,
  Ball.cpp:1634-1883) in the partition (their own InsertInBoxes) and in both collision methods (`CollideWithBall` in
  the gradient; `CheckCollision` and `CalculateCollisionImpactStatic` in the iterative method). COL-711, COL-715,
  COL-716 and COL-718 with them, MOV-732 (fixing a ball spawns its minis, freeing it removes them).
- SEN-711 and MOV-730's cloak trigger: `CloakBall`, `UncloakBall` (Ballpark.cpp:5223, 5257).
- MOV-727: `CapAcceleration` returns at once (1350); with dynamical orientation it is not called.
- MOV-728: dynamical orientation (`GotoTorqueBreaks`, `GotoRollBreaks`, `ApplyTorque`, `CalculateN2Alpha2`,
  `CalculateBallRotationVelocity`, `CollisionBallProperties::CalculateRotation`, `SnapOrientation`, the rotation
  setters, `AddDynamicallyOrientedBall`; the rotation terms of the iterative impacts and `ApproximateEllipsePointOfImpact`;
  the missile exclusion), in C++ float arithmetic: every float operation rounded to binary32 (`_f`), and the float
  transcendentals (expf, logf, acosf, asinf, atan2f, sinf, cosf, powf) taken from the C library itself through ctypes,
  as Destiny's compiled code takes them (glibc 2.34 here).
- MOV-729: the orientation without it (`CalculateYawPitchRoll`, `Modulo2pi`; Ball.cpp:335, 777), and the Python
  AddBall's snap (`PyAddBall` -> `AddOldStyleOrientedBall`, Thunkers.cpp:150, Ballpark.cpp:3467), which MOV-711 needs.
- MOV-734's remaining modes: MISSILE, FORMATION, MUSHROOM (a growing ball, removed when grown; Ballpark.cpp:540-556,
  3719), TROLL (MOV-736: `SetBallTroll`, `TrollReady`, `PetrifyTroll`; 6290-6337), RIGID, BOID (`PySetBoid`).
- MOV-726: formations (`EvolveFormation`, `FormationFollow`, `SetBallFormation`, `AddFormation`, `PyLoadFormations`,
  `Ball::ReserveFormationSlot`, `Ball::FreeFormationSlot`). Thunkers.cpp:1886-1897 calls this "old abandoned code".
- CMB-701, CMB-702: `Ball::CalculateRelativeVelocities` (Ball.cpp:161) and `GetAccuracy` (Ballpark.cpp:5156), with the
  Python `GetCenterDist` and `GetSurfaceDist`.
- TIME-702 to TIME-705: `OnTick` (Ballpark.cpp:219): moribund balls out first, whole intervals only, the master's
  resynchronisation above 5 intervals, the first call evolving once without tick events, DoPreTick and DoPostTick
  around each Evolve with the interval's timestamp (the balls' mNewTime and mOldTime), the remainder carried in mTime.
  `pre_tick_handlers` and `post_tick_handlers` are called where Destiny sends the two events.
- `ClearAll` (Ballpark.cpp:5896), which the server layer's adapter and Destiny's tests call.
- TIME-708, in part: `ClientBall::GetShiftedTime` (Ball.cpp:1318); the interpolation that uses it is display code and
  is not transcribed.

One function the code calls comes from DirectXMath (microsoft/DirectXMath at f5d82574, MIT, (c) Microsoft
Corporation; a public source since 22 October 2026): carbonengine/math's RotationQuaternion(yaw, pitch, roll)
(src/Quaternion.cpp:60) is XMQuaternionRotationRollPitchYaw, transcribed from DirectXMath's scalar path
(`q_rotation_ypr`; the oracle compiles that path, `_XM_NO_INTRINSICS_`). It is reached by Ball::GetRotatedVector
without dynamical orientation: a formation leader's slot offsets (MOV-726) and a stationary missile launcher's
direction (MOV-725), once that ball has turned. Destiny's Windows build likely takes DirectXMath's SSE path, whose
sines and cosines are its own minimax polynomials (or Intel's SVML): there the last bits may differ.

Where Destiny's result is undefined the reference refuses (`DestinyUndefined`), as space.py: a negative formation slot
or formation index read from a vector, a formation index past the end after a smaller LoadFormations (SetBallFormation,
Ballpark.cpp:6696), a follower id missing from the park while a formation shrinks, AddCapsule or AddOrientedBox under
an id whose map entry a stale removal left null (RemoveCapsule and RemoveOrientedBox look the id up with `operator[]`,
Ballpark.cpp:5599, 5650, which inserts that null entry), and a capsule whose diameter is below half a float ulp of its
length, whose walk through the boxes never ends (Capsule.cpp:499, 578). BringOutDeadCapsules walks an
unordered_set of pointers (heap order, Ballpark.cpp:5772), as BringOutDeadBalls: ascending id here. Oriented boxes
removed with a delay are never brought out (BringOutTheDeads calls no BringOutDeadOrientedBoxes, 5699-5706).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import struct
from dataclasses import dataclass, field

from .modes import (BOID, FIELD, FOLLOW, FORMATION, MINIBALL, MISSILE, MUSHROOM, ORBIT, RIGID, TROLL, WARP, _cross,
                    _float32, _ieee_div, _lensq, _length)
from .ref import GOTO, STOP, STOP_DAMP, _normalized, f32
from .space import (LEVEL_WIDTH, LOCAL_BALLS, NUM_LEVELS, BallLocation, Box, CollisionBallProperties,
                    CollisionImpact, CollisionItem, DestinyUndefined, SpaceBall, SpaceBallpark, StdHashMap, _MASK, _dot,
                    _sub, _to_int64)

FLT_EPS_ZERO = 0.0
DSTNORMALCLOAK = 1                                          # IDstConstants.h:77
MISSILE_STRAIGHT_MS = 800                                   # Ballpark.cpp:974 (MOV-723)
MISSILE_LAUNCH_SPEED = 150.0                                # Thunkers.cpp:925 (MOV-725)
TROLL_MIN_DELAY = 1                                         # Ballpark.cpp:6300
FORMATION_MAX_SLOTS = 16                                    # Ball.cpp:2288 (MOV-726)
MUSHROOM_MASS = 1.0e20                                      # Ballpark.cpp:3742: the float literal 1.0e20f, below
ROLL_SPEED_ACCELERATION, ROLL_SPEED_DECAY = -3.0, -0.9      # Ballpark.cpp:115-118 (MOV-729)
ROLL_ACCELERATION, ROLL_DECAY = 1.0, -0.5
CLOAK_SENSOR_PERIOD = 2.0                                   # Ballpark.cpp:5246 (SEN-711)
MAX_CATCH_UP_TICKS = 5                                      # Ballpark.cpp:248 (TIME-703)
DISPATCH_MIN_MOVE_SQ = 0.1                                  # Ballpark.cpp:602: moved at least sqrt(0.1) m (COL-719)
EJECT_MARGIN = 1.0                                          # Capsule.cpp:427, OrientedBox.cpp:106 (COL-712)
SPIN_LIMIT = 4                                              # Ballpark.cpp:2610: 4 * maxAngVel^2 (COL-718)
HALF_PI, TWO_PI = 1.57079632679489655, 6.2831853071795862   # Ball.cpp:13-15
DBL_MAX = 1.7976931348623157e308


class DestinyUnavailable(Exception):
    """The value needs code the snapshot does not have. It was raised for DirectXMath's
    XMQuaternionRotationRollPitchYaw away from zero until 23 October 2026, when DirectXMath was transcribed
    (`q_rotation_ypr`); nothing raises it now."""


# ================================================================================================ C++ float arithmetic

_F32 = struct.Struct("<f")


def _f(x: float) -> float:
    """A double rounded to binary32, as a C++ float operation's result (overflow to an infinity). For +, -, *, / and
    sqrt of float operands the double result rounded once is the correctly rounded float result (53 >= 2 x 24 + 2)."""
    try:
        return _F32.unpack(_F32.pack(x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


_MUSHROOM_MASS = _f(MUSHROOM_MASS)
_LIBM = ctypes.CDLL(ctypes.util.find_library("m"))


def _libm(name: str, nargs: int):
    fn = getattr(_LIBM, name)
    fn.restype = ctypes.c_float
    fn.argtypes = [ctypes.c_float] * nargs
    return fn


_expf, _logf, _acosf, _asinf = _libm("expf", 1), _libm("logf", 1), _libm("acosf", 1), _libm("asinf", 1)
_sinf, _cosf = _libm("sinf", 1), _libm("cosf", 1)
_atan2f, _powf = _libm("atan2f", 2), _libm("powf", 2)


def _fdiv(a: float, b: float) -> float:
    """float / float."""
    return _f(_ieee_div(a, b))


def _fmax(a: float, b: float) -> float:
    """std::max(a, b): (a < b) ? b : a."""
    return b if a < b else a


def _fmin(a: float, b: float) -> float:
    """std::min(a, b): (b < a) ? b : a."""
    return b if b < a else a


def _ceil(x: float) -> float:
    return x if not math.isfinite(x) else float(math.ceil(x))


def _to_int32(x: float) -> int:
    """`int(x)` of a float: truncation toward zero; outside the range C++ leaves it undefined and aarch64 (fcvtzs)
    saturates, NaN giving 0, as the compiled code does on this machine (space.py's _to_int64 for int32)."""
    if x != x:
        return 0
    if x >= 2147483648.0:
        return 2147483647
    if x < -2147483648.0:
        return -2147483648
    return int(x)


def _sqrt(x: float) -> float:
    """C's sqrt: NaN for a negative argument where Python raises."""
    return math.sqrt(x) if x >= 0.0 or x != x else math.nan


def _char(x: int) -> int:
    """A conversion to (signed) char."""
    return ((int(x) + 128) & 0xFF) - 128


# ---- carbonengine/math: Vector3, Quaternion, Matrix (Vector3_inline.h, Quaternion_inline.h, Matrix_inline.h) ----

def v3(x: float, y: float, z: float) -> tuple:
    """Vector3(float, float, float) from doubles (each narrowed)."""
    return (_f(x), _f(y), _f(z))


def v3_of(v) -> tuple:
    """Vector3d::AsVector3 (Vector3d.h:23)."""
    return (_f(v[0]), _f(v[1]), _f(v[2]))


def v3_add(a, b) -> tuple:
    return (_f(a[0] + b[0]), _f(a[1] + b[1]), _f(a[2] + b[2]))


def v3_sub(a, b) -> tuple:
    return (_f(a[0] - b[0]), _f(a[1] - b[1]), _f(a[2] - b[2]))


def v3_mul(a, f: float) -> tuple:
    """Vector3 * float, Vector3 *= float and float * Vector3 (each other.c * f)."""
    return (_f(a[0] * f), _f(a[1] * f), _f(a[2] * f))


def v3_div(a, f: float) -> tuple:
    """Vector3 / float: multiplied by fDiv = 1.0f / f."""
    return v3_mul(a, _fdiv(1.0, f))


def v3_dot(a, b) -> float:
    return _f(_f(_f(a[0] * b[0]) + _f(a[1] * b[1])) + _f(a[2] * b[2]))


def v3_len(a) -> float:
    return _f(_sqrt(v3_dot(a, a)))


def v3_normalize(v) -> tuple:
    """Normalize(Vector3): divided by the largest magnitude first (overflow guard), then by its length."""
    mx = _fmax(_fmax(abs(v[0]), abs(v[1])), abs(v[2]))
    minified = v3_div(v, mx if mx else 1.0)
    length = v3_len(minified)
    if length:
        length = _fdiv(1.0, length)
    return v3_mul(minified, length)


def v3_cross(a, b) -> tuple:
    return (_f(_f(a[1] * b[2]) - _f(a[2] * b[1])), _f(_f(a[2] * b[0]) - _f(a[0] * b[2])),
            _f(_f(a[0] * b[1]) - _f(a[1] * b[0])))


QUAT_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def q_mul(q, o) -> tuple:
    """Quaternion::operator*(other): q * o (Quaternion_inline.h:118)."""
    x, y, z, w = q
    ox, oy, oz, ow = o
    return (_f(_f(_f(_f(ow * x) + _f(ox * w)) + _f(oy * z)) - _f(oz * y)),
            _f(_f(_f(_f(ow * y) - _f(ox * z)) + _f(oy * w)) + _f(oz * x)),
            _f(_f(_f(_f(ow * z) + _f(ox * y)) - _f(oy * x)) + _f(oz * w)),
            _f(_f(_f(_f(ow * w) - _f(ox * x)) - _f(oy * y)) - _f(oz * z)))


def q_lensq(q) -> float:
    return _f(_f(_f(_f(q[0] * q[0]) + _f(q[1] * q[1])) + _f(q[2] * q[2])) + _f(q[3] * q[3]))


def q_len(q) -> float:
    return _f(_sqrt(q_lensq(q)))


def q_div(q, f: float) -> tuple:
    d = _fdiv(1.0, f)
    return (_f(q[0] * d), _f(q[1] * d), _f(q[2] * d), _f(q[3] * d))


def q_conj(q) -> tuple:
    return (-q[0], -q[1], -q[2], q[3])


def q_rotation(axis, angle: float) -> tuple:
    """RotationQuaternion(axis, angle) (Quaternion_inline.h:232): sin and cos of float are sinf and cosf."""
    t = v3_normalize(axis)
    h = _fdiv(angle, 2.0)
    s = _sinf(h)
    return (_f(s * t[0]), _f(s * t[1]), _f(s * t[2]), _cosf(h))


def q_rotation_ypr(yaw: float, pitch: float, roll: float) -> tuple:
    """RotationQuaternion(yaw, pitch, roll) (carbonengine/math src/Quaternion.cpp:60): DirectXMath's
    XMQuaternionRotationRollPitchYaw(pitch, yaw, roll), its scalar path (_XM_NO_INTRINSICS_, DirectXMathMisc.inl:510
    at f5d82574): each half angle, the C library's cosf and sinf of it (g++ calls sincosf, equal to them for every
    float here), and the quaternion's four sums of float products, left to right."""
    halfpitch = _f(pitch * 0.5)
    cp, sp = _cosf(halfpitch), _sinf(halfpitch)
    halfyaw = _f(yaw * 0.5)
    cy, sy = _cosf(halfyaw), _sinf(halfyaw)
    halfroll = _f(roll * 0.5)
    cr, sr = _cosf(halfroll), _sinf(halfroll)
    return (_f(_f(_f(cr * sp) * cy) + _f(_f(sr * cp) * sy)), _f(_f(_f(cr * cp) * sy) - _f(_f(sr * sp) * cy)),
            _f(_f(_f(sr * cp) * cy) - _f(_f(cr * sp) * sy)), _f(_f(_f(cr * cp) * cy) + _f(_f(sr * sp) * sy)))


def m_rotation_q(q) -> tuple:
    """RotationMatrix(Quaternion) (Matrix_inline.h:365), rows m[0] to m[3]."""
    x, y, z, w = q
    return ((_f(1.0 - _f(2.0 * _f(_f(y * y) + _f(z * z)))), _f(2.0 * _f(_f(x * y) + _f(z * w))),
             _f(2.0 * _f(_f(x * z) - _f(y * w))), 0.0),
            (_f(2.0 * _f(_f(x * y) - _f(z * w))), _f(1.0 - _f(2.0 * _f(_f(x * x) + _f(z * z)))),
             _f(2.0 * _f(_f(y * z) + _f(x * w))), 0.0),
            (_f(2.0 * _f(_f(x * z) + _f(y * w))), _f(2.0 * _f(_f(y * z) - _f(x * w))),
             _f(1.0 - _f(2.0 * _f(_f(x * x) + _f(y * y)))), 0.0),
            (0.0, 0.0, 0.0, 1.0))


def m_rotation_axis(axis, angle: float) -> tuple:
    """RotationMatrix(axis, angle) (Matrix_inline.h:388)."""
    n = v3_normalize(axis)
    s, c = _sinf(angle), _cosf(angle)
    omc = _f(1.0 - c)
    m = [[0.0] * 4 for _ in range(4)]
    m[0][0] = _f(_f(_f(omc * n[0]) * n[0]) + c)
    m[1][0] = _f(_f(_f(omc * n[0]) * n[1]) - _f(s * n[2]))
    m[2][0] = _f(_f(_f(omc * n[0]) * n[2]) + _f(s * n[1]))
    m[0][1] = _f(_f(_f(omc * n[1]) * n[0]) + _f(s * n[2]))
    m[1][1] = _f(_f(_f(omc * n[1]) * n[1]) + c)
    m[2][1] = _f(_f(_f(omc * n[1]) * n[2]) - _f(s * n[0]))
    m[0][2] = _f(_f(_f(omc * n[2]) * n[0]) - _f(s * n[1]))
    m[1][2] = _f(_f(_f(omc * n[2]) * n[1]) + _f(s * n[0]))
    m[2][2] = _f(_f(_f(omc * n[2]) * n[2]) + c)
    m[3][3] = 1.0
    return tuple(tuple(r) for r in m)


def transform_coord(v, m) -> tuple:
    """TransformCoord (Matrix_inline.h:560): the row vector times the matrix, divided by the homogeneous norm."""
    x, y, z = v
    norm = _f(_f(_f(_f(m[0][3] * x) + _f(m[1][3] * y)) + _f(m[2][3] * z)) + m[3][3])
    if norm != 0:
        return (_fdiv(_f(_f(_f(_f(x * m[0][0]) + _f(y * m[1][0])) + _f(z * m[2][0])) + m[3][0]), norm),
                _fdiv(_f(_f(_f(_f(x * m[0][1]) + _f(y * m[1][1])) + _f(z * m[2][1])) + m[3][1]), norm),
                _fdiv(_f(_f(_f(_f(x * m[0][2]) + _f(y * m[1][2])) + _f(z * m[2][2])) + m[3][2]), norm))
    return (0.0, 0.0, 0.0)


# ---- Vector3d (Vector3d.h): doubles ----

def d_add(a, b) -> list:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def d_scale(a, c: float) -> list:
    """Vector3d * double and double * Vector3d (v.c * c)."""
    return [a[0] * c, a[1] * c, a[2] * c]


def d_div(a, c: float) -> list:
    """Vector3d / double: multiplied by the reciprocal."""
    return d_scale(a, _ieee_div(1.0, c))


def d_neg(a) -> list:
    return [-a[0], -a[1], -a[2]]


# ================================================================================ the static collidables' geometry

def _quadratic(a: float, b: float, c: float) -> tuple:
    """Quadratic (Collision.cpp:137): (found, v1, v2). Its linear branch is overwritten (dead code, as coded)."""
    det = b * b - 4.0 * a * c
    if det < 0.0:
        return False, None, None
    det = _sqrt(det)
    return True, _ieee_div((-b + det) * 0.5, a), _ieee_div((-b - det) * 0.5, a)


def collide_two_spheres(p0, p1, q0, q1, coll_radius: float) -> float:
    """CollideTwoSpheres (Collision.cpp:108)."""
    return SpaceBallpark.collide_two_spheres(p0, p1, q0, q1, coll_radius)


def _point_on_ball(ball_pos, radius: float, normal) -> list:
    """GetPointOnBallThatFirstIntersectsPlane (Collision.cpp:6)."""
    return _sub(ball_pos, d_scale(normal, radius))


def _segment_plane(p0, p1, plane_d: float, plane_n) -> tuple:
    """IntersectSegmentPlane (Collision.cpp:11): (found, point, time); the time is written even when not found."""
    p0p1 = _sub(p1, p0)
    ndab = _dot(plane_n, p0p1)
    if ndab == 0.0:
        return False, None, None
    t = _ieee_div(plane_d - _dot(plane_n, p0), ndab)
    if 0.0 <= t <= 1.0:
        return True, d_add(p0, d_scale(p0p1, t)), t
    return False, None, t


def _segment_sphere(p0, p1, sphere_pos, sphere_rad: float) -> tuple:
    """IntersectSegmentSphere (Collision.cpp:29): (found, point, time)."""
    direction = _sub(p1, p0)
    nd = _normalized(direction)
    m = _sub(p0, sphere_pos)
    b = _dot(m, nd)
    c = _dot(m, m) - sphere_rad * sphere_rad
    if c > 0.0 and b > 0.0:
        return False, None, None
    disc = b * b - c
    if disc < 0.0:
        return False, None, None
    t = -b - _sqrt(disc)
    if t < 0.0:
        t = 0.0
    else:
        t = _ieee_div(t, _length(direction))
    if t > 1.0:
        return False, None, t
    return True, d_add(p0, d_scale(direction, t)), t


class Triangle:
    """Triangle (Triangle.cpp)."""
    __slots__ = ("a", "b", "c")

    def __init__(self, a, b, c):
        self.a, self.b, self.c = list(a), list(b), list(c)

    def normal(self) -> list:
        return _normalized(_cross(_sub(self.b, self.a), _sub(self.c, self.a)))

    def closest_point(self, p) -> list:
        a, b, c = self.a, self.b, self.c
        ab, ac = _sub(b, a), _sub(c, a)
        ap = _sub(p, a)
        d1, d2 = _dot(ab, ap), _dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return list(a)
        bp = _sub(p, b)
        d3, d4 = _dot(ab, bp), _dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return list(b)
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = _ieee_div(d1, d1 - d3)
            return d_add(a, d_scale(ab, v))
        cp = _sub(p, c)
        d5, d6 = _dot(ab, cp), _dot(ac, cp)
        if d6 > 0.0 and d5 <= d6:
            return list(c)
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = _ieee_div(d2, d2 - d6)
            return d_add(a, d_scale(ac, w))
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and d5 - d6 >= 0.0:
            w = _ieee_div(d4 - d3, (d4 - d3) + (d5 - d6))
            return d_add(b, d_scale(_sub(c, b), w))
        denom = _ieee_div(1.0, va + vb + vc)
        v, w = vb * denom, vc * denom
        return d_add(d_add(a, d_scale(ab, v)), d_scale(ac, w))

    def contains(self, p) -> bool:
        v0, v1, v2 = _sub(self.c, self.a), _sub(self.b, self.a), _sub(p, self.a)
        dot00, dot01, dot02 = _dot(v0, v0), _dot(v0, v1), _dot(v0, v2)
        dot11, dot12 = _dot(v1, v1), _dot(v1, v2)
        inv = _ieee_div(1.0, dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        return u >= 0 and v >= 0 and u + v < 1


def _sphere_triangle(sphere_pos, radius: float, vel, tri: Triangle, out: dict) -> bool:
    """IntersectSphereTriangle (Collision.cpp:67); the outputs (point, time, normal) go into `out` as written."""
    n = tri.normal()
    p0 = _point_on_ball(sphere_pos, radius, n)
    p1 = d_add(p0, vel)
    found, point, t = _segment_plane(p0, p1, _dot(tri.a, n), n)
    if t is not None:
        out["time"] = t
    if not found:
        return False
    out["point"] = point
    if tri.contains(point):
        out["normal"] = n
        return True
    point = tri.closest_point(point)
    out["point"] = point
    out["normal"] = d_neg(vel)
    found, impact, t = _segment_sphere(point, _sub(point, vel), sphere_pos, radius)
    if t is not None:
        out["time"] = t
    if not found:
        return False
    out["normal"] = _normalized(_sub(sphere_pos, impact))
    return True


CORNER_A, CORNER_B, CORNER_C, CORNER_D, CORNER_E, CORNER_F, CORNER_G, CORNER_H = range(8)
SIDE_FRONT, SIDE_BACK, SIDE_LEFT, SIDE_RIGHT, SIDE_TOP, SIDE_BOTTOM = range(6)
CONNECTED_SIDES = ((SIDE_BACK, SIDE_LEFT, SIDE_BOTTOM), (SIDE_BACK, SIDE_RIGHT, SIDE_BOTTOM),      # BoxShape.cpp:189
                   (SIDE_BACK, SIDE_LEFT, SIDE_TOP), (SIDE_FRONT, SIDE_LEFT, SIDE_BOTTOM),
                   (SIDE_BACK, SIDE_RIGHT, SIDE_TOP), (SIDE_FRONT, SIDE_LEFT, SIDE_TOP),
                   (SIDE_FRONT, SIDE_RIGHT, SIDE_BOTTOM), (SIDE_FRONT, SIDE_RIGHT, SIDE_TOP))
SIDE_MAPPING = ((SIDE_LEFT, SIDE_RIGHT), (SIDE_BOTTOM, SIDE_TOP), (SIDE_BACK, SIDE_FRONT))           # BoxShape.cpp:137


class RectangleShape:
    """RectangleShape (BoxShape.cpp:8): corners a, b, c, d and the normal of (b - a) x (c - a)."""
    __slots__ = ("a", "b", "c", "d", "normal")

    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = list(a), list(b), list(c), list(d)
        self.normal = _normalized(_cross(_sub(b, a), _sub(c, a)))

    def collide_with_sphere(self, sphere_pos, radius: float, vel, out: dict) -> bool:
        pa, pc = _sub(self.a, sphere_pos), _sub(self.c, sphere_pos)
        m = _cross(pc, pa)
        tri = Triangle(self.a, self.c, self.d)
        if _dot(vel, m) <= 0.0:
            tri = Triangle(self.a, self.b, self.c)
        return _sphere_triangle(sphere_pos, radius, vel, tri, out)


class BoxShape:
    """BoxShape (BoxShape.cpp:34): the corners A to H from a corner and three edge vectors, and the six sides."""
    __slots__ = ("corners", "sides", "local")

    def __init__(self, corner, lx, ly, lz):
        A = list(corner)
        B, C, D = d_add(A, lx), d_add(A, ly), d_add(A, lz)
        E, F, G = d_add(B, ly), d_add(C, lz), d_add(B, lz)
        H = d_add(E, lz)
        self.corners = (A, B, C, D, E, F, G, H)
        sides = [None] * 6
        sides[SIDE_FRONT] = RectangleShape(H, F, D, G)
        sides[SIDE_BACK] = RectangleShape(C, E, B, A)
        sides[SIDE_LEFT] = RectangleShape(F, C, A, D)
        sides[SIDE_RIGHT] = RectangleShape(E, H, G, B)
        sides[SIDE_TOP] = RectangleShape(C, F, H, E)
        sides[SIDE_BOTTOM] = RectangleShape(B, G, D, A)
        self.sides = tuple(sides)
        self.local = (list(lx), list(ly), list(lz))

    def center(self) -> list:
        return d_add(self.corners[CORNER_A], d_div(_sub(self.corners[CORNER_H], self.corners[CORNER_A]), 2.0))

    def bounding_radius(self) -> float:
        return _length(_sub(self.corners[CORNER_H], self.corners[CORNER_A])) / 2.0

    def closest_corner(self, p) -> int:
        best, min_d = CORNER_A, _lensq(_sub(self.corners[CORNER_A], p))
        for i in range(CORNER_B, 8):
            d = _lensq(_sub(self.corners[i], p))
            if d < min_d:
                min_d, best = d, i
        return best

    def closest_point(self, p) -> list:
        q = self.center()
        d = _sub(p, q)
        for axis in self.local:
            half = _length(axis) / 2.0
            u = _normalized(axis)
            dist = _dot(d, u)
            if dist > half:
                dist = half
            elif dist < -half:
                dist = -half
            q = d_add(q, d_scale(u, dist))
        return q

    def ray_intersect(self, origin, ray) -> tuple:
        """RayIntersect (BoxShape.cpp:128): the slabs method; (found, side1, d1, side2, d2), side2 the opposite face
        on the entry axis (as coded)."""
        tmin, tmax, revert = [0.0] * 3, [0.0] * 3, [0] * 3
        o = _sub(origin, self.center())
        for i in range(3):
            arm = self.local[i]
            arm_len_sq, arm_origin, arm_ray = _lensq(arm), _dot(arm, o), _dot(arm, ray)
            if arm_ray < 0:
                revert[i] = 1
                tmin[i] = _ieee_div(0.5 * arm_len_sq - arm_origin, arm_ray)
                tmax[i] = _ieee_div(-0.5 * arm_len_sq - arm_origin, arm_ray)
            elif arm_ray > 0:
                tmin[i] = _ieee_div(-0.5 * arm_len_sq - arm_origin, arm_ray)
                tmax[i] = _ieee_div(0.5 * arm_len_sq - arm_origin, arm_ray)
            else:
                tmin[i], tmax[i] = -DBL_MAX, DBL_MAX
        i_min = 0                                           # std::max_element: the first largest
        for i in (1, 2):
            if tmin[i_min] < tmin[i]:
                i_min = i
        i_max = 0                                           # std::min_element: the first smallest
        for i in (1, 2):
            if tmax[i] < tmax[i_max]:
                i_max = i
        d1, d2 = tmin[i_min], tmax[i_max]
        if d1 < d2:
            return True, SIDE_MAPPING[i_min][revert[i_min]], d1, SIDE_MAPPING[i_min][1 - revert[i_min]], d2
        return False, None, d1, None, d2

    def collide_with_sphere(self, sphere_pos, radius: float, vel, out: dict) -> bool:
        """CollideWithSphere (BoxShape.cpp:200); the outputs go into `out` as written."""
        closest = self.closest_corner(sphere_pos)
        could_be_inside, shortest, closest_half = True, -1.0, None
        for i in range(3):
            side = CONNECTED_SIDES[closest][i]
            point = _point_on_ball(sphere_pos, radius, self.sides[side].normal)
            hs = _dot(self.sides[side].normal, _sub(self.sides[side].a, point))
            if hs >= 0.0:
                if hs < shortest or shortest == -1.0:
                    shortest, closest_half = hs, side
            else:
                could_be_inside = False
                break
        if could_be_inside:
            cp = self.closest_point(sphere_pos)
            normal = _sub(sphere_pos, cp)
            out["normal"] = normal
            n_len = _length(normal)
            if n_len < radius:
                if n_len < 1e-10:
                    speed = _length(vel)
                    if speed < 1e-10:
                        normal = d_scale(self.sides[closest_half].normal, _fmax(shortest, 1e-5))
                    else:
                        found, side1, d1, side2, d2 = self.ray_intersect(sphere_pos, vel)
                        if found:
                            if abs(d1) < abs(d2):
                                normal = d_scale(self.sides[side1].normal, _fmax(abs(d1) * speed, 1e-5))
                            else:
                                normal = d_scale(self.sides[side2].normal, _fmax(abs(d2) * speed, 1e-5))
                        else:
                            normal = d_scale(self.sides[closest_half].normal, shortest)
                else:
                    extrication = abs(radius - n_len)
                    normal = d_scale(_normalized(normal), extrication)
                out["normal"], out["time"] = normal, 0.0
                return True
        for i in range(3):
            if self.sides[CONNECTED_SIDES[closest][i]].collide_with_sphere(sphere_pos, radius, vel, out):
                return True
        return False


class AABB:
    """AABB (AABB.cpp): grown corner by corner; a straight move can be excluded by one slab."""
    __slots__ = ("low", "high")

    def __init__(self, low=(0.0, 0.0, 0.0), high=(0.0, 0.0, 0.0)):
        self.low, self.high = list(low), list(high)               # AABB() is at the origin (AABB.cpp)

    def update(self, v) -> None:
        for q in range(3):
            if v[q] > self.high[q]:
                self.high[q] = v[q]
            elif v[q] < self.low[q]:
                self.low[q] = v[q]

    def can_exclude(self, p0, p1, r: float) -> bool:
        lo, hi = self.low, self.high
        return ((p0[0] - r > hi[0] and p1[0] - r > hi[0]) or (p0[0] + r < lo[0] and p1[0] + r < lo[0])
                or (p0[1] - r > hi[1] and p1[1] - r > hi[1]) or (p0[1] + r < lo[1] and p1[1] + r < lo[1])
                or (p0[2] - r > hi[2] and p1[2] - r > hi[2]) or (p0[2] + r < lo[2] and p1[2] + r < lo[2]))

    def longest_width(self) -> float:
        dx, dy, dz = (self.high[0] - self.low[0], self.high[1] - self.low[1], self.high[2] - self.low[2])
        return _fmax(dx, _fmax(dy, dz))


class StaticCollidable:
    """Partitionable and StaticCollidable (Partitionable.cpp, StaticCollidable.cpp): a fixed collision shape in the
    partition and the bubbles, sorted in a box by its parent ball's id, then its own."""
    kind = -1

    def __init__(self):
        self.id, self.parent_id, self.handled = 0, 0, False
        self.pos, self.old_pos = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        self.bubble, self.old_bubble, self.is_interactive = -1, -1, False
        self.boxes: dict = {}
        self.my_box = None
        self.park = None
        self.effect_stamp, self.is_moribund = 0, False

    def sort_key(self) -> tuple:
        return (self.parent_id, self.id)

    def sorted_boxes(self) -> list:
        return [self.boxes[k] for k in sorted(self.boxes)]


class Capsule(StaticCollidable):
    """Capsule (Capsule.cpp): the segment A-B swept by a sphere of radius r (a float)."""
    kind = 0

    def initialize(self, the_id: int, parent: int, a, b, radius: float) -> None:
        self.id, self.parent_id = the_id, parent
        self.a, self.b, self.radius = list(a), list(b), radius
        self.ab = _sub(self.b, self.a)
        self.pos = d_scale(d_add(self.a, self.b), 0.5)
        self.old_pos = list(self.pos)

    def bounding_radius(self) -> float:
        """(float)|AB| * 0.5f + r."""
        return _f(_f(_f(_length(self.ab)) * 0.5) + self.radius)


class OrientedBox(StaticCollidable):
    """OrientedBox (OrientedBox.cpp): a BoxShape with its axis-aligned bounding box."""
    kind = 1

    def initialize(self, the_id: int, parent: int, corner, lx, ly, lz) -> None:
        self.id, self.parent_id = the_id, parent
        self.shape = BoxShape(corner, lx, ly, lz)
        self.bounding_radius_f = _f(self.shape.bounding_radius())
        self.pos = self.shape.center()
        self.old_pos = list(self.pos)
        self.aabb = AABB(self.shape.corners[CORNER_A], self.shape.corners[CORNER_A])
        for i in range(CORNER_B, 8):
            self.aabb.update(self.shape.corners[i])
        self.mod = [0, 0, 0]

    def bounding_radius(self) -> float:
        return _f(self.shape.bounding_radius())


@dataclass
class MiniBallRec:
    """MiniBall (MiniBall.h): a parent's miniball, offset from its position; `id` is the ball made from it."""
    id: int = 0
    radius: float = 0.0
    pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class MiniCapsuleRec:
    """MiniCapsule (MiniCapsule.h)."""
    id: int = 0
    parent: int = 0
    a: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    b: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius: float = 0.0


@dataclass
class MiniBoxRec:
    """MiniBox (MiniBox.h)."""
    id: int = 0
    parent: int = 0
    corner: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    lx: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ly: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    lz: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class FullBall(SpaceBall):
    """space.py's ball with Ball.h's remaining members at Ball::Ball's defaults (Ball.cpp:59): the formation, the
    rotation state (floats), the orientation (doubles), the time stamps and the minis."""
    formation_id: int = -1                                  # char mFormationID
    formation_slots: int = 0                                # std::bitset<16>
    max_ang_vel: float = 1.0                                # float
    rot_roll: float = 0.0                                   # float mRoll
    rot_agility: float = f32(0.9)                           # float mRotAgility
    rot: tuple = QUAT_IDENTITY                              # Quaternion mNewRot
    old_rot: tuple = QUAT_IDENTITY
    ang_vel: tuple = (0.0, 0.0, 0.0)                        # Vector3 mNewAngVel
    old_ang_vel: tuple = (0.0, 0.0, 0.0)
    torque: tuple = (0.0, 0.0, 0.0)                         # Vector3 mTorque
    yaw: float = 0.0                                        # double mNewYaw
    old_yaw: float = 0.0
    pitch: float = 0.0
    old_pitch: float = 0.0
    roll_angle: float = 0.0                                 # double mNewRoll
    old_roll: float = 0.0
    roll_speed: float = 0.0
    old_roll_speed: float = 0.0
    yaw_delta: float = 0.0
    new_time: int = 0                                       # Be::Time mNewTime
    old_time: int = 0
    mini_balls: list = field(default_factory=list)
    mini_capsules: list = field(default_factory=list)
    mini_boxes: list = field(default_factory=list)


class FullBallLocation(BallLocation):
    """BallLocation with its rotation and angular velocity (CollisionStructures.h:40)."""
    __slots__ = ("rot", "omega")

    def __init__(self, time: float = -1.0, collider: int = -1, p=None, v=None):
        super().__init__(time, collider, p, v)
        self.rot, self.omega = QUAT_IDENTITY, (0.0, 0.0, 0.0)


class FullCollisionImpact(CollisionImpact):
    """CollisionImpact with its angular velocity change (value-initialised: zero)."""
    __slots__ = ("dw",)

    def __init__(self, time: float):
        super().__init__(time)
        self.dw = (0.0, 0.0, 0.0)


class FullCollisionBallProperties(CollisionBallProperties):
    """CollisionBallProperties with the rotation of dynamical orientation (CollisionBallProperties.cpp:15-260)."""

    def __init__(self, park):
        super().__init__(park)
        self.tau = -1.0                                     # float m_tau

    def _resize(self, n: int) -> None:
        if n > self.capacity:
            self.capacity = len(self.locations) + max(len(self.locations), n - len(self.locations))
        del self.locations[n:]
        while len(self.locations) < n:
            self.locations.append(FullBallLocation())

    def _torque(self, rot, omega, dt: float) -> tuple:
        b = self.ball
        return self.park._apply_torque(rot, omega, b.torque, b.rot_roll, self.tau, _f(dt))

    def initialize(self, ball, t_end: float) -> None:
        do = self.park.dynamical_orientation
        self.ball, self.max_time, self.t_start, self.t_end = ball, t_end, 0.0, t_end
        if do:
            self.tau = self.park._tau(ball)
        self._resize(2)
        l0, l1 = self.locations
        l0.time, l0.collider, l0.p, l0.v = 0.0, -1, list(ball.pos), list(ball.vel)
        l1.time, l1.collider = t_end, -1
        l1.p, l1.v = self._integrate(list(ball.pos), list(ball.vel), t_end)
        if do:
            l0.rot, l0.omega = ball.rot, ball.ang_vel
            l1.rot, l1.omega = self._torque(l0.rot, l0.omega, self.t_end)

    def add_impact(self, impact) -> list:
        invalidated = []
        if self.t_end <= impact.time:
            return invalidated
        index = len(self.locations)
        while index > 0:
            if impact.time >= self.locations[index - 1].time:
                break
            if self.locations[index - 1].collider != -1:
                invalidated.append((self.locations[index - 1].time, self.locations[index - 1].collider))
            index -= 1
        self._resize(index + 1)
        loc = self.locations[index]
        loc.collider, loc.time = impact.collider, impact.time
        do = self.park.dynamical_orientation
        if index == 0:
            loc.p = list(self.ball.pos)
            loc.v = [self.ball.vel[i] + impact.dv[i] for i in range(3)]
            if do:
                loc.rot, loc.omega = self.ball.rot, v3_add(self.ball.ang_vel, impact.dw)
        else:
            prev = self.locations[index - 1]
            p, v = self._integrate(list(prev.p), list(prev.v), loc.time - prev.time)
            loc.p, loc.v = p, [v[i] + impact.dv[i] for i in range(3)]
            if do:
                rot, omega = self._torque(prev.rot, prev.omega, loc.time - prev.time)
                loc.rot, loc.omega = rot, v3_add(omega, impact.dw)
        self.t_end = impact.time
        return invalidated

    def remove_impact(self, time: float, collider: int) -> list:
        from . import space as _space
        invalidated = []
        index = 0                                           # `found` starts true: a missing impact means index 0
        for i, loc in enumerate(self.locations):
            if loc.collider == collider and abs(loc.time - time) < _space.FUZZY_DELTA:
                index = i
                break
        if index == 0:
            self.locations[0].collider = -1
            self.locations[0].v = list(self.ball.vel)
            if self.park.dynamical_orientation:
                self.locations[0].omega = self.ball.ang_vel
        for loc in self.locations[index + 1:]:
            if loc.collider != -1:
                invalidated.append((loc.time, loc.collider))
        self._resize(index if index > 0 else 1)
        self.calculate_end()
        return invalidated

    def calculate_end(self) -> None:
        if self.max_time > self.locations[-1].time:
            if len(self.locations) == self.capacity:        # push_back reallocates under a held reference
                raise DestinyUndefined("CalculateEnd reads a point of a reallocated vector "
                                       "(CollisionBallProperties.cpp:173)")
            prev = self.locations[-1]
            p, v = self._integrate(list(prev.p), list(prev.v), self.max_time - prev.time)
            loc = FullBallLocation(self.max_time, -1, p, v)
            if self.park.dynamical_orientation:
                loc.rot, loc.omega = self._torque(prev.rot, prev.omega, self.max_time - prev.time)
            self.locations.append(loc)
            self.t_end, self.t_start = self.max_time, prev.time

    def rotation(self, time: float) -> tuple:
        """CalculateRotation (CollisionBallProperties.cpp:246): note > where CalculatePosition has >=."""
        from . import space as _space
        loc = self.locations[self.find_index(time)]
        rot, omega = loc.rot, loc.omega
        if abs(loc.time - time) > _space.FUZZY_DELTA:
            rot, omega = self._torque(rot, omega, time - loc.time)
        return rot, omega


def _ball_key(b) -> tuple:
    """BallSortComparer (Ball.cpp:2528): two local (negative) ids by their owner first, otherwise by id."""
    return (0, b.owner_id, b.id) if b.id < 0 else (1, 0, b.id)


def _modulo2pi(a: float, b: float) -> tuple:
    """Modulo2pi (Ball.cpp:777): both angles reduced by fmod, then b moved by 2 pi toward a."""
    b, a = math.fmod(b, TWO_PI), math.fmod(a, TWO_PI)

    def ab(x):
        return -x if x < 0.0 else x
    if ab(b - a) < ab(b + TWO_PI - a):
        if ab(b - a) < ab(b - TWO_PI - a):
            return a, b
        return a, b - TWO_PI
    if ab(b + TWO_PI - a) < ab(b - TWO_PI - a):
        return a, b + TWO_PI
    return a, b - TWO_PI


class FullBallpark(SpaceBallpark):
    """space.py's ballpark with Destiny's missiles, static collidables, cloaking, orientation (dynamical or not),
    mushrooms, trolls, formations, the accuracy helpers and OnTick. Balls are added as the Python AddBall adds them
    (`PyAddBall`). A client's view unless told otherwise, as Destiny's Ballpark()."""

    def __init__(self, friction: float = 1.0e6, tick_interval_ms: int = 1000, use_new_orbit: bool = False,
                 is_master: bool = False, seed: int = 1, iterative: bool = False, dynamical_orientation: bool = False,
                 disable_do_for_missiles: bool = False):
        super().__init__(friction, tick_interval_ms, use_new_orbit, is_master, seed, iterative)
        self.dynamical_orientation = dynamical_orientation          # g_useDynamicalOrientation (MOV-728)
        self.disable_do_for_missiles = disable_do_for_missiles      # g_disableDynamicalOrientationForMissiles
        self.time = 0                                               # Be::Time mTime (TIME-702)
        self.first_time = True                                      # mFirstTime (TIME-704)
        self.formations: list = []                                  # mFormations
        self.capsules: dict = {}                                    # mCapsules (a removed id may map to None)
        self.oriented_boxes: dict = {}                              # mOrientedBoxes (likewise)
        self.static_collidables: dict = {}                          # mStaticCollidables
        self.moribund_capsules: set = set()                         # moribundCapsules (objects)
        self.moribund_boxes: set = set()                            # moribundOrientedBoxes (never brought out)
        self.potential_collidables: dict = {}                       # mPotentialCollidables
        self.pre_tick_handlers: list = []                           # called where Destiny sends DoPreTick
        self.post_tick_handlers: list = []                          # called where Destiny sends DoPostTick

    def _event(self, name: str, bid: int, *args) -> None:
        self.events.append((self.current_time, name, bid, *args))

    # ===================================================================================== partition (statics)

    def _release(self, box: Box) -> None:
        """Box::ReleaseBox, as space.py, with a box's static collidables keeping its parent (Box.cpp:149)."""
        from . import space as _space
        self._hit("box.release")
        keep = self.free_boxes < _space.BOX_CHUNK_SIZE
        if box.parent is not None:
            p = box.parent
            p.children.pop(box.key, None)
            if not p.children and not p.balls and not p.statics:
                self._release(p)
        del self.levels[box.level][box.key]
        box.released = True
        if keep:
            self.free_boxes += 1
        else:
            self._hit("box.deleted")
            box.deleted = True

    def _box_remove_ball(self, box: Box, b) -> None:
        """Box::RemoveBall (Box.cpp:165)."""
        box.balls.pop(b.id, None)
        if not box.children and not box.balls and not box.statics:
            self._release(box)

    def _box_remove_static(self, box: Box, s: StaticCollidable) -> None:
        """Box::RemoveStaticCollidable (Box.cpp:187): found by its sort key."""
        box.statics.pop(s.sort_key(), None)
        if not box.children and not box.balls and not box.statics:
            self._release(box)

    def _static_insert_in_box(self, s: StaticCollidable, box: Box | None) -> None:
        """StaticCollidable::InsertInBox: a std::set insert keeps an equivalent element already there."""
        if box is None:
            return
        box.statics.setdefault(s.sort_key(), s)
        s.boxes[box.key] = box

    def _delete_from_boxes(self, p) -> None:
        """Partitionable::DeleteFromBoxes: Ball's (space.py) or StaticCollidable's (StaticCollidable.cpp:24), which
        clears the boxes and mMyBox only."""
        if not isinstance(p, StaticCollidable):
            super()._delete_from_boxes(p)
            return
        for box in p.sorted_boxes():
            self._box_remove_static(box, p)
        p.boxes, p.my_box = {}, None

    def inflated_radius(self, p) -> float:
        """Partitionable::GetInflatedRadius: a static collidable's is its bounding radius (a float)."""
        if isinstance(p, StaticCollidable):
            return p.bounding_radius()
        return super().inflated_radius(p)

    def _ball_insert_in_boxes(self, p, box1: Box, top, new_bubble: int) -> None:
        """Partitionable::InsertInBoxes: Ball's (space.py), Capsule's or OrientedBox's."""
        if isinstance(p, Capsule):
            self._capsule_insert_in_boxes(p, box1, new_bubble)
        elif isinstance(p, OrientedBox):
            self._obox_insert_in_boxes(p, new_bubble)
        else:
            super()._ball_insert_in_boxes(p, box1, top, new_bubble)

    def _capsule_insert_in_boxes(self, c: Capsule, box1: Box, new_bubble: int) -> None:
        """Capsule::InsertInBoxes (Capsule.cpp:449): the boxes at its radius's level along the segment, sampled every
        2r from A while the float length lasts, each with the adjacent boxes its radius reaches."""
        if c.my_box is not None:
            if box1.released:
                self._hit("insert.stale_box")
                self._stale(box1, "the key")
            if box1.level == c.my_box.level and box1.key == c.my_box.key:
                self._hit("capsule.unchanged")
                return
        if c.radius <= 0.0:
            self._hit("capsule.no_volume")
            return
        self._delete_from_boxes(c)
        box1 = self._get_box_r(c.radius, c.pos)
        box_a = self._get_box_r(c.radius, c.a)
        level = box_a.level
        if self.is_master:
            self._topmost(box1).bubble = new_bubble
        c.my_box = box1
        interp = list(c.a)
        direction = _sub(c.b, c.a)
        length = _f(_length(direction))
        direction = [x * float(c.radius) * 2.0 for x in _normalized(direction)]
        step = _f(c.radius * 2.0)
        while length >= 0.0:
            box_a = self._get_box_r(c.radius, interp)
            ix, iy, iz = box_a.key
            mod = [0, 0, 0]
            for q in range(3):
                if box_a.lo[q] + float(c.radius) - interp[q] > 0.0:
                    mod[q] = -1
                elif box_a.hi[q] - float(c.radius) - interp[q] < 0.0:
                    mod[q] = 1
            self._static_insert_in_box(c, self._get_box(ix, iy, iz, level))
            if mod[0]:
                self._static_insert_in_box(c, self._get_box(ix + mod[0], iy, iz, level))
            if mod[1]:
                self._static_insert_in_box(c, self._get_box(ix, iy + mod[1], iz, level))
            if mod[2]:
                self._static_insert_in_box(c, self._get_box(ix, iy, iz + mod[2], level))
            if mod[0] and mod[1]:
                self._static_insert_in_box(c, self._get_box(ix + mod[0], iy + mod[1], iz, level))
            if mod[0] and mod[2]:
                self._static_insert_in_box(c, self._get_box(ix + mod[0], iy, iz + mod[2], level))
            if mod[1] and mod[2]:
                self._static_insert_in_box(c, self._get_box(ix, iy + mod[1], iz + mod[2], level))
            if mod[0] and mod[1] and mod[2]:
                self._static_insert_in_box(c, self._get_box(ix + mod[0], iy + mod[1], iz + mod[2], level))
            interp = d_add(interp, direction)
            shorter = _f(length - step)
            if shorter == length:
                raise DestinyUndefined("Capsule::InsertInBoxes never ends: twice the radius is below half a float ulp "
                                       "of the length left (Capsule.cpp:499, 578), a loop without forward progress")
            length = shorter

    def _obox_insert_in_boxes(self, ob: OrientedBox, new_bubble: int) -> None:
        """OrientedBox::InsertInBoxes (OrientedBox.cpp:127): the box at half its longest AABB width, and the adjacent
        boxes its AABB reaches (all 26 at the top level when its bounding diameter exceeds the top width)."""
        r = ob.aabb.longest_width() / 2.0
        box1 = self._get_box_r(r, ob.pos)
        mod = [0, 0, 0]
        for q in range(3):
            if box1.lo[q] - ob.aabb.low[q] > 0.0:
                mod[q] = -1
            elif box1.hi[q] - ob.aabb.high[q] < 0.0:
                mod[q] = 1
        if ob.my_box is not None and box1.level == ob.my_box.level and box1.key == ob.my_box.key:
            if mod == ob.mod:
                self._hit("obox.unchanged")
                return
        self._delete_from_boxes(ob)
        ob.mod = list(mod)
        box1 = self._get_box_r(r, ob.pos)
        ix, iy, iz = box1.key
        level = box1.level
        self._static_insert_in_box(ob, box1)
        if self.is_master:
            self._topmost(box1).bubble = new_bubble
        ob.my_box = box1
        if level == 0 and _f(ob.bounding_radius_f * 2.0) > LEVEL_WIDTH[0]:
            self._hit("obox.huge")
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    for z in (-1, 0, 1):
                        if x == 0 and y == 0 and z == 0:
                            continue
                        self._static_insert_in_box(ob, self._get_box(ix + x, iy + y, iz + z, 0))
            return
        if mod[0]:
            self._static_insert_in_box(ob, self._get_box(ix + mod[0], iy, iz, level))
        if mod[1]:
            self._static_insert_in_box(ob, self._get_box(ix, iy + mod[1], iz, level))
        if mod[2]:
            self._static_insert_in_box(ob, self._get_box(ix, iy, iz + mod[2], level))
        if mod[0] and mod[1]:
            self._static_insert_in_box(ob, self._get_box(ix + mod[0], iy + mod[1], iz, level))
        if mod[0] and mod[2]:
            self._static_insert_in_box(ob, self._get_box(ix + mod[0], iy, iz + mod[2], level))
        if mod[1] and mod[2]:
            self._static_insert_in_box(ob, self._get_box(ix, iy + mod[1], iz + mod[2], level))
        if mod[0] and mod[1] and mod[2]:
            self._static_insert_in_box(ob, self._get_box(ix + mod[0], iy + mod[1], iz + mod[2], level))

    def _insert_static_in_boxes(self, s: StaticCollidable) -> None:
        """InsertStaticCollidableInBoxes (Ballpark.cpp:3224)."""
        self._insert_in_boxes(s)
        if not self.in_evolve:
            self.static_collidables[s.id] = s

    def _explode_children(self, box: Box, out: list) -> None:
        """Partition::ExplodeChildren: every descendant with balls or static collidables (Partition.cpp:268)."""
        for child in box.sorted_children():
            self._explode_children(child, out)
            if child.balls or child.statics:
                out.append(child)

    def _nearby(self, b, exclude_fixed: bool, exclude_inert: bool, exclude_missiles: bool, exclude_cloaked: bool,
                exclude_harmonics: bool) -> tuple:
        """Partition::GetNearbyBalls (Partition.cpp:344) with its static collidables: (balls, static collidables)."""
        if exclude_inert and self.interactive_cnt(b) == 0:
            self._hit("nearby.no_interactives")
            return [], []
        s_boxes: list = []
        for box in b.sorted_boxes():
            if box.handled:
                continue
            p = box
            while p.parent is not None:
                p = p.parent
                if p.handled:
                    break
                if p.balls or p.statics:
                    s_boxes.append(p)
                    p.handled = True
            self._explode_children(box, s_boxes)
            if len(box.balls) > 1 or box.statics:
                s_boxes.append(box)
                box.handled = True
        uni: list = []
        uni_stat: list = []
        for box in s_boxes:
            box.handled = False
            for other in box.sorted_balls():
                if other.handled:
                    continue
                if other is b:
                    continue
                if other.is_moribund:
                    self._hit("nearby.moribund")
                    continue
                if exclude_cloaked and (other.is_cloaked or not other.is_massive):
                    continue
                if exclude_inert:
                    if not other.is_interactive:
                        continue
                else:
                    if self.is_master and other.bubble != b.bubble and not other.is_global:
                        self._hit("nearby.other_bubble")
                        continue
                    if self.is_master and other.bubble != b.bubble:
                        self._hit("nearby.global_other_bubble")
                    if exclude_fixed and not other.is_free:
                        continue
                    if other.mode == FIELD and not exclude_harmonics:
                        if b.harmonic == -2:
                            self._hit("nearby.field_master_pass")
                            continue
                        if other.harmonic != -1 and other.harmonic == b.harmonic:
                            self._hit("nearby.field_harmonic")
                            continue
                        if other.corporation_id != -1 and other.corporation_id == b.corporation_id:
                            self._hit("nearby.field_corporation")
                            continue
                        if other.alliance_id != -1 and other.alliance_id == b.alliance_id:
                            self._hit("nearby.field_alliance")
                            continue
                    if exclude_missiles and other.mode == MISSILE:
                        continue
                other.handled = True
                uni.append(other)
            if not exclude_fixed:
                for s in box.sorted_statics():
                    if s.handled:
                        continue
                    if self.is_master and s.bubble != b.bubble:
                        self._hit("nearby.static_other_bubble")
                        continue
                    s.handled = True
                    uni_stat.append(s)
        for other in uni:
            other.handled = False
        for s in uni_stat:
            s.handled = False
        return uni, uni_stat

    def _nearby_balls(self, b, exclude_fixed: bool, exclude_inert: bool, exclude_missiles: bool,
                      exclude_cloaked: bool, exclude_harmonics: bool) -> list:
        """GetNearbyBalls' balls (the proximity sensors' candidates, GetProximityCandidates)."""
        return self._nearby(b, exclude_fixed, exclude_inert, exclude_missiles, exclude_cloaked, exclude_harmonics)[0]

    def _collision_candidates(self, b) -> tuple:
        """Partition::GetCollisionCandidates (Partition.cpp:301; COL-703): a normal missile sees only its target (in
        its bubble, unfiltered), a defender missile (negative owner) everything but cloaked and non-massive balls."""
        if b.mode == MISSILE:
            if b.owner_id < 0:
                self._hit("candidates.defender")
                return self._nearby(b, False, False, False, True, False)
            target = self.balls[b.follow_ptr]
            if b.bubble == target.bubble:
                self._hit("candidates.missile_target")
                return [target], []
            return [], []
        return self._nearby(b, False, False, True, True, False)

    def box_hash(self, box: Box) -> int:
        """The harness's per-box checksum, with the balls in BallSortComparer's order and the static collidables."""
        h = 14695981039346656037

        def mix(h: int, v: int) -> int:
            return ((h ^ (v & _MASK)) * 1099511628211) & _MASK
        for v in (box.level, *box.key, box.bubble, int(box.handled), len(box.balls),
                  *(b.id for b in box.sorted_balls()), len(box.statics), *(s.id for s in box.sorted_statics()),
                  len(box.children)):
            h = mix(h, v)
        for k in sorted(box.children):
            for c in k:
                h = mix(h, c)
        return mix(h, box.parent.level if box.parent is not None else 99)

    # =============================================================================================== adding balls

    def _add_ball_core(self, object_id: int, mass: float, radius: float, max_vel: float, is_free: bool,
                       is_global: bool, is_massive: bool, is_interactive: bool, is_space_junk: bool, x: float,
                       y: float, z: float, vx: float, vy: float, vz: float, agility: float,
                       speed_fraction: float) -> FullBall:
        """Ballpark::AddBall (Ballpark.cpp:3303; MOV-711): as space.py's, with local ids and mFormationID reset."""
        the_id = self._id_for_collision_object(int(object_id))
        ball = self.balls.get(the_id)
        created = ball is None
        if created:
            ball = FullBall(id=the_id, mass=0.0, radius=0.0, max_vel=0.0, agility=1.0, speed_fraction=1.0,
                            time_factor=1.0, pos=[1e10, 1e10, 1e10])
        else:
            self._hit("add.existing")
            self._delete_from_boxes(ball)
        ball.pos = [float(x), float(y), float(z)]
        ball.vel = [float(vx), float(vy), float(vz)]
        if created:
            ball.old_pos = [ball.pos[i] - ball.vel[i] * self.dt for i in range(3)]
            ball.old_vel = list(ball.vel)
        else:
            ball.formation_id = -1
            ball.effect_stamp = 0
            ball.owner_id = 0
            ball.follow_id = 0
            ball.follow_range = 10.0
            ball.corporation_id = -1
            ball.alliance_id = -1
            ball.harmonic = -1
            if ball.is_moribund:
                ball.is_moribund = False
                self.moribund.discard(the_id)
        ball.radius = 0.0 if radius < 0.0 else radius
        mass = float(mass)
        ball.mass = 0.0 if mass < 0.0 else mass
        ball.is_free, ball.is_global, ball.is_massive = bool(is_free), bool(is_global), bool(is_massive)
        ball.is_interactive, ball.is_space_junk = bool(is_interactive), bool(is_space_junk)
        ball.max_vel = 0.0 if max_vel < 0.0 else max_vel
        ball.agility = 1.0 if agility <= 0.0 else agility
        ball.speed_fraction = 0.0 if speed_fraction < 0.0 else speed_fraction
        self._refresh_time_factor(ball)
        if not created and not is_free:
            self.free_balls.discard(the_id)
        elif ball.is_free:
            self.free_balls.add(the_id)
        self._set_mode(ball, STOP)
        if not created and ball.followers:
            for fid in ball.followers.ordered():
                f = self.balls.get(fid)
                if f is None or f.follow_ptr != ball.id:
                    ball.followers.discard(fid)
        self._insert_ball_in_boxes(ball)
        return ball

    def _add_old_style_ball(self, *args) -> FullBall:
        """AddOldStyleOrientedBall (Ballpark.cpp:3467): AddBall, and a free ball's orientation snapped."""
        ball = self._add_ball_core(*args)
        if ball.is_free:
            self._yaw_pitch_roll(ball, snap=True)
        return ball

    def _add_dynamic_ball(self, object_id, mass, radius, max_vel, max_ang_vel, is_free, is_global, is_massive,
                          is_interactive, is_space_junk, x, y, z, vx, vy, vz, rx, ry, rz, rw, wx, wy, wz, agility,
                          rot_agility, speed_fraction) -> FullBall:
        """AddDynamicallyOrientedBall (Ballpark.cpp:3500). It takes an id from GetIDForCollisionObject before AddBall
        takes another, so a local ball consumes two local ids (as coded)."""
        the_id = self._id_for_collision_object(int(object_id))
        is_new = self.balls.get(the_id) is None
        ball = self._add_ball_core(object_id, mass, radius, max_vel, is_free, is_global, is_massive, is_interactive,
                                   is_space_junk, x, y, z, vx, vy, vz, agility, speed_fraction)
        ball.rot = (rx, ry, rz, rw)
        ball.ang_vel = (wx, wy, wz)
        if is_new:
            ball.old_rot = q_mul(ball.rot, q_rotation(ball.ang_vel, _f(-_f(self.dt) * v3_len(ball.ang_vel))))
            ball.old_ang_vel = (wx, wy, wz)
        ball.max_ang_vel = 0.0 if max_ang_vel < 0.0 else max_ang_vel
        ball.rot_agility = 1.0 if rot_agility <= 0.0 else rot_agility
        return ball

    def add_ball(self, object_id: int, mass: float, radius: float, max_vel: float, is_free: bool = False,
                 is_global: bool = False, is_massive: bool = False, is_interactive: bool = False,
                 is_space_junk: bool = False, x: float = 0.0, y: float = 0.0, z: float = 0.0, vx: float = 0.0,
                 vy: float = 0.0, vz: float = 0.0, agility: float = 0.0, speed_fraction: float = 0.0,
                 max_ang_vel: float = 1.0, rot_agility: float = 0.9) -> FullBall | None:
        """The Python AddBall (PyAddBall, Thunkers.cpp:150): AddDynamicallyOrientedBall then SnapOrientation with
        dynamical orientation (its last two arguments), AddOldStyleOrientedBall without; refused during Evolve."""
        if self.in_evolve:
            return None
        radius, max_vel = _float32(radius), _float32(max_vel)
        agility, speed_fraction = _float32(agility), _float32(speed_fraction)
        flags = (int(is_free) != 0, int(is_global) != 0, int(is_massive) != 0, int(is_interactive) != 0,
                 int(is_space_junk) != 0)
        pos = (float(x), float(y), float(z), float(vx), float(vy), float(vz))
        if self.dynamical_orientation:
            ball = self._add_dynamic_ball(object_id, float(mass), radius, max_vel, _float32(max_ang_vel), *flags,
                                          *pos, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, agility, _float32(rot_agility),
                                          speed_fraction)
            self._snap_orientation(ball)
            return ball
        return self._add_old_style_ball(object_id, float(mass), radius, max_vel, *flags, *pos, agility,
                                        speed_fraction)

    def _new_ball(self, the_id: int) -> FullBall:
        """`new OBall()` / `new OClientBall()`: Ball::Ball's defaults."""
        return FullBall(id=the_id, mass=0.0, radius=0.0, max_vel=0.0, agility=1.0, speed_fraction=1.0,
                        time_factor=1.0, pos=[1e10, 1e10, 1e10])

    def _add_miniball(self, parent_id: int, mass: float, radius: float, x: float, y: float, z: float) -> FullBall:
        """Ballpark::AddMiniball (Ballpark.cpp:3559; COL-713): a fixed, massive, non-interactive ball with the parent's
        mass, mode MINIBALL; registered in the park only outside Evolve (InsertBallInBoxes)."""
        the_id = self._id_for_collision_object(-1)
        ball = self._new_ball(the_id)
        ball.pos = [x, y, z]
        ball.old_pos = list(ball.pos)
        ball.radius = 0.0 if radius < 0.0 else radius
        ball.mass = 0.0 if mass < 0.0 else mass
        ball.owner_id = parent_id
        ball.vel, ball.old_vel = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        ball.max_vel, ball.agility, ball.speed_fraction = 0.0, 0.0, 0.0
        ball.is_free, ball.is_global, ball.is_massive, ball.is_interactive, ball.is_space_junk = (False, False, True,
                                                                                                  False, False)
        ball.rot, ball.old_rot = QUAT_IDENTITY, QUAT_IDENTITY
        ball.ang_vel, ball.old_ang_vel = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        self._refresh_time_factor(ball)
        self._set_mode(ball, MINIBALL)
        if self.in_evolve:
            self._hit("miniball.ghost")
        self._insert_ball_in_boxes(ball)
        return ball

    # ===================================================================================== static collidables

    def add_capsule(self, object_id: int, parent_id: int, ax: float, ay: float, az: float, bx: float, by: float,
                    bz: float, radius: float) -> Capsule:
        """Ballpark::AddCapsule (Ballpark.cpp:3639), as the Python AddCapsule calls it."""
        the_id = self._id_for_collision_object(int(object_id))
        if the_id in self.capsules:
            cap = self.capsules[the_id]
            if cap is None:
                raise DestinyUndefined("AddCapsule dereferences the null entry a stale RemoveCapsule left "
                                       "(Ballpark.cpp:3667, 5599)")
            self._hit("capsule.existing")
        else:
            cap = Capsule()
            cap.initialize(the_id, int(parent_id), [float(ax), float(ay), float(az)],
                           [float(bx), float(by), float(bz)], _float32(radius))
            self.capsules[the_id] = cap
        cap.park = self
        self._insert_static_in_boxes(cap)
        return cap

    def add_oriented_box(self, object_id: int, parent_id: int, c0: float, c1: float, c2: float, x0: float,
                         x1: float, x2: float, y0: float, y1: float, y2: float, z0: float, z1: float,
                         z2: float) -> OrientedBox:
        """Ballpark::AddOrientedBox (Ballpark.cpp:3675)."""
        the_id = self._id_for_collision_object(int(object_id))
        if the_id in self.oriented_boxes:
            ob = self.oriented_boxes[the_id]
            if ob is None:
                raise DestinyUndefined("AddOrientedBox dereferences the null entry a stale RemoveOrientedBox left "
                                       "(Ballpark.cpp:3706, 5650)")
            self._hit("obox.existing")
        else:
            ob = OrientedBox()
            ob.initialize(the_id, int(parent_id), [float(c0), float(c1), float(c2)], [float(x0), float(x1), float(x2)],
                          [float(y0), float(y1), float(y2)], [float(z0), float(z1), float(z2)])
            self.oriented_boxes[the_id] = ob
        ob.park = self
        self._insert_static_in_boxes(ob)
        return ob

    def _remove_static(self, table: dict, moribund: set, sid: int, delay: int) -> None:
        if sid not in table:                                # operator[] inserts a null entry
            self._hit("static.null_entry")
            table[sid] = None
        s = table[sid]
        if s is None:
            return
        s.is_moribund = True
        if self.in_evolve or delay > 0:
            self._hit("static.delayed")
            s.effect_stamp = self.current_time + delay
            moribund.add(s)
            return
        self._hit("static.removed")
        self._event("DoFinalCleanup", s.id)
        moribund.discard(s)
        self._delete_from_boxes(s)
        s.park = None
        s.old_bubble, s.bubble = s.bubble, -1
        self._add_to_bubble(s)
        del table[sid]

    def remove_capsule(self, cid: int, delay: int = 0) -> None:
        """Ballpark::RemoveCapsule (Ballpark.cpp:5594)."""
        self._remove_static(self.capsules, self.moribund_capsules, int(cid), int(delay))

    def remove_oriented_box(self, oid: int, delay: int = 0) -> None:
        """Ballpark::RemoveOrientedBox (Ballpark.cpp:5645)."""
        self._remove_static(self.oriented_boxes, self.moribund_boxes, int(oid), int(delay))

    def clear_all(self) -> None:
        """Ballpark::ClearAll (Ballpark.cpp:5896): every ball and static collidable out of its boxes (the partition
        empties) and out of the maps, the iterative method's maps, the bubble dictionaries (ClearBubbles, 5872), the
        free and proximity balls cleared; the local id counters reset (5948-5949). The clock, the formations, the
        bubble counter and the moribund sets stay (Destiny keeps pointers to the cleared balls there)."""
        for b in list(self.balls.values()):                 # BallIterator: mBalls, a Python dict, in insertion order
            self._delete_from_boxes(b)
        self.balls.clear()
        self.globals.clear()
        if self.iterative_collision:
            self.collision_locations.clear()
            self.potential_balls.clear()
            self.potential_collidables.clear()
        for s in list(self.static_collidables.values()):
            self._delete_from_boxes(s)
            s.park = None
        self.static_collidables.clear()
        self.capsules.clear()
        self.oriented_boxes.clear()
        self.bubbles.clear()
        self.bubble_interactives.clear()
        self.bubble_keep_alives.clear()
        self.bubble_keep_alive_balls.clear()
        self.free_balls.clear()
        self.proximity_balls.clear()
        self.local_cnt, self.local_hi_cnt = -1, LOCAL_BALLS

    def bring_out_the_deads(self) -> None:
        """BringOutTheDeads (Ballpark.cpp:5699): the balls (space.py), then the capsules; never the oriented boxes."""
        if self.in_evolve:
            return
        super().bring_out_the_deads()
        kill, due = 0, []
        for c in sorted(self.moribund_capsules, key=lambda c: c.id):   # heap order in Destiny: ascending id here
            if c.effect_stamp - self.current_time < 0:
                due.append(c)
            elif c.effect_stamp - self.current_time - 2 < 0 and kill < 7:
                self._hit("static.buffer")
                due.append(c)
                kill += 1
        for c in due:
            self.remove_capsule(c.id)
            self.moribund_capsules.discard(c)

    # ---- the minis (Ball.cpp:1634-1883) ----

    def add_mini_ball(self, bid: int, x: float, y: float, z: float, r: float) -> None:
        """Ball::AddMiniBall through the Python thunker ("dddf"): kept by the parent, made a ball if it is fixed."""
        b = self.balls.get(bid)
        if b is None:
            return
        rec = MiniBallRec(0, _float32(r), [float(x), float(y), float(z)])
        b.mini_balls.append(rec)
        if not b.is_free:
            self._add_actual_miniball(b, rec)

    def _add_actual_miniball(self, b: FullBall, rec: MiniBallRec) -> None:
        if b.is_free:
            return
        ball = self._add_miniball(b.id, b.mass, rec.radius, b.pos[0] + rec.pos[0], b.pos[1] + rec.pos[1],
                                  b.pos[2] + rec.pos[2])
        rec.id = ball.id

    def add_mini_capsule(self, bid: int, ax: float, ay: float, az: float, bx: float, by: float, bz: float,
                         r: float) -> None:
        """Ball::PyAddMiniCapsule ("ddddddf"; a radius <= 0 is a ValueError) and Ball::AddMiniCapsule."""
        b = self.balls.get(bid)
        if b is None:
            return
        r = _float32(r)
        if r <= 0.0:
            raise ValueError("Radius must be positive")         # Ball.cpp:2156-2160
        rec = MiniCapsuleRec(0, b.id, [float(ax), float(ay), float(az)], [float(bx), float(by), float(bz)], r)
        b.mini_capsules.append(rec)
        if not b.is_free:
            self._add_actual_minicapsule(b, rec)

    def _add_actual_minicapsule(self, b: FullBall, rec: MiniCapsuleRec) -> None:
        if b.is_free:
            return
        cap = self.add_capsule(-1, rec.parent, b.pos[0] + rec.a[0], b.pos[1] + rec.a[1], b.pos[2] + rec.a[2],
                               b.pos[0] + rec.b[0], b.pos[1] + rec.b[1], b.pos[2] + rec.b[2], rec.radius)
        rec.id = cap.id

    def add_mini_box(self, bid: int, c0: float, c1: float, c2: float, x0: float, x1: float, x2: float, y0: float,
                     y1: float, y2: float, z0: float, z1: float, z2: float) -> None:
        """Ball::PyAddMiniBox and Ball::AddMiniBox."""
        b = self.balls.get(bid)
        if b is None:
            return
        rec = MiniBoxRec(0, b.id, [float(c0), float(c1), float(c2)], [float(x0), float(x1), float(x2)],
                         [float(y0), float(y1), float(y2)], [float(z0), float(z1), float(z2)])
        b.mini_boxes.append(rec)
        if not b.is_free:
            self._add_actual_minibox(b, rec)

    def _add_actual_minibox(self, b: FullBall, rec: MiniBoxRec) -> None:
        if b.is_free:
            return
        ob = self.add_oriented_box(-1, rec.parent, rec.corner[0] + b.pos[0], rec.corner[1] + b.pos[1],
                                   rec.corner[2] + b.pos[2], *rec.lx, *rec.ly, *rec.lz)
        rec.id = ob.id

    def _add_minis(self, b: FullBall) -> None:
        """AddMiniBalls, AddMiniCapsules, AddMiniBoxes: only for a fixed ball."""
        if b.is_free:
            return
        for rec in b.mini_balls:
            self._add_actual_miniball(b, rec)
        for rec in b.mini_capsules:
            self._add_actual_minicapsule(b, rec)
        for rec in b.mini_boxes:
            self._add_actual_minibox(b, rec)

    def _remove_minis(self, b: FullBall) -> None:
        """RemoveMiniBalls, RemoveMiniCapsules, RemoveMiniBoxes: every record's last id, which may be stale."""
        for rec in b.mini_balls:
            self.remove_ball(rec.id)
        for rec in b.mini_capsules:
            self.remove_capsule(rec.id)
        for rec in b.mini_boxes:
            self.remove_oriented_box(rec.id)

    # ================================================================================== setters and removal

    def set_free(self, bid: int, flag: bool) -> None:
        """SetBallFree (Ballpark.cpp:4887; MOV-732): freeing removes the minis and sets mNewTime a tick back; fixing
        stops, zeroes the velocity, re-files, clears mLastG and makes the minis."""
        b = self.balls.get(bid)
        if b is None or bool(flag) == b.is_free:
            return
        b.is_free = bool(flag)
        if b.is_free:
            self.free_balls.add(bid)
            self._remove_minis(b)
            b.new_time = self.time - self.tick_interval * 10000
        else:
            self._stop(b)
            self.set_velocity(bid, 0.0, 0.0, 0.0)
            self.free_balls.discard(bid)
            self._insert_ball_in_boxes(b)
            b.last_g = [0.0, 0.0, 0.0]
            self._add_minis(b)

    def entity_warp_in(self, bid: int, x: float, y: float, z: float, warp_factor: int) -> None:
        """EntityWarpIn (Ballpark.cpp:4370; TRV-715): modes.py's, then with dynamical orientation the rotation snapped
        to the new velocity (Ballpark.cpp:4392-4395; added 23 October 2026: the compiled code snaps
        there, the reference did not, and EntityWarpIn with dynamical orientation had not been generated before)."""
        super().entity_warp_in(bid, x, y, z, warp_factor)
        b = self.balls.get(bid)
        if b is not None and b.mode != GOTO and self.dynamical_orientation:
            self._snap_orientation(b)

    def set_velocity(self, bid: int, x: float, y: float, z: float) -> None:
        """SetBallVelocity (Ballpark.cpp:4770; MOV-713): the new and old velocity, then the orientation snapped."""
        b = self.balls.get(bid)
        if b is None:
            return
        b.vel = [float(x), float(y), float(z)]
        b.old_vel = list(b.vel)
        if self.dynamical_orientation:
            self._snap_orientation(b)
        else:
            self._yaw_pitch_roll(b, snap=True)

    def set_max_angular_speed(self, bid: int, w: float) -> None:
        """SetMaxAngularSpeed (Ballpark.cpp:4721)."""
        w = _float32(w)
        if w < 0.0:
            return
        b = self.balls.get(bid)
        if b is not None:
            b.max_ang_vel = w

    def set_ball_angular_velocity(self, bid: int, wx: float, wy: float, wz: float) -> None:
        """SetBallAngularVelocity (Ballpark.cpp:4802)."""
        b = self.balls.get(bid)
        if b is None:
            return
        b.ang_vel = v3(wx, wy, wz)
        b.old_ang_vel = b.ang_vel

    def set_ball_rotation(self, bid: int, rx: float, ry: float, rz: float, rw: float) -> None:
        """SetBallRotation (Ballpark.cpp:4819): normalised if longer than 1e-20; the rotation stopped."""
        b = self.balls.get(bid)
        if b is None:
            return
        rot = (_f(rx), _f(ry), _f(rz), _f(rw))
        length = q_len(rot)
        if length > 1e-20:
            rot = q_div(rot, length)
            b.rot, b.old_rot = rot, rot
            b.ang_vel = (0.0, 0.0, 0.0)
            b.old_ang_vel = b.ang_vel

    def set_ball_angular_agility(self, bid: int, a: float) -> None:
        """SetBallAngularAgility (Ballpark.cpp:4958)."""
        a = _float32(a)
        b = self.balls.get(bid)
        if b is None or a <= 0.0:
            return
        b.rot_agility = a

    def remove_ball(self, bid: int, delay: int = 0) -> None:
        """Ballpark::RemoveBall (Ballpark.cpp:5471): as space.py, removing a fixed ball's minis first."""
        b = self.balls.get(bid)
        if b is None:
            return
        self._stop(b)
        self._stop_all_followers(b)
        b.is_moribund = True
        if b.mini_balls and not b.is_free:
            for rec in reversed(b.mini_balls):
                self.remove_ball(rec.id)
            b.mini_balls = []
        if b.mini_capsules and not b.is_free:
            for rec in reversed(b.mini_capsules):
                self.remove_capsule(rec.id)
            b.mini_capsules = []
        if b.mini_boxes and not b.is_free:
            for rec in reversed(b.mini_boxes):
                self.remove_oriented_box(rec.id)
            b.mini_boxes = []
        if self.in_evolve or delay > 0:
            self._hit("remove.delayed")
            b.is_massive = False
            b.effect_stamp = self.current_time + delay
            self.moribund.add(bid)
            return
        self._hit("remove.now")
        self._event("DoFinalCleanup", b.id)
        self.moribund.discard(bid)
        if self.iterative_collision:
            self.collision_locations.pop(bid, None)
            self.potential_collidables.pop(bid, None)
            self.potential_balls.pop(bid, None)
        self._delete_from_boxes(b)
        b.old_bubble, b.bubble = b.bubble, -1
        self._add_to_bubble(b)
        if b.is_free:
            self.free_balls.discard(bid)
        if b.has_proximity:
            self.proximity_balls.erase(bid)
        if b.is_global:
            self.globals.pop(bid, None)
        del self.balls[bid]

    # ================================================================================================ commands

    def missile_follow(self, src: int, dst: int, owner: int) -> None:
        """PyMissileFollow (Thunkers.cpp:275: a negative target is an error) and MissileFollow (Ballpark.cpp:3798;
        MOV-724): the missile is stopped, then aimed at the target's centre (follow range minus both radii, a float),
        straight on for 800 ms toward 1e16 m along its velocity, at full speed fraction."""
        src, dst, owner = int(src), int(dst), int(owner)
        if dst < 0:
            return
        if src == dst or src == owner or dst == owner:
            return
        missile, target = self.balls.get(src), self.balls.get(dst)
        if missile is None or target is None or target.is_moribund:
            return
        self._stop(missile)
        if missile.bubble != target.bubble:
            self._hit("missile.other_bubble")
            return
        if target.is_cloaked:
            self._hit("missile.cloaked_target")
            return
        missile.follow_id = dst
        missile.follow_ptr = dst
        missile.follow_range = -_f(missile.radius + target.radius)
        missile.owner_id = owner
        missile.effect_stamp = self.current_time
        missile.goto = d_scale(_normalized(list(missile.vel)), 1.e16)
        missile.speed_fraction = 1.0
        self._set_mode(missile, MISSILE)
        target.followers.add(missile.id)

    def launch_missile(self, src: int, dst: int, owner: int, aimed: int, massive: int) -> None:
        """PyLaunchMissile (Thunkers.cpp:857; MOV-725)."""
        src, dst, owner = int(src), int(dst), int(owner)
        aimed, massive = _char(aimed), _char(massive)
        if dst < 0:
            return
        launcher = self.balls.get(-owner if owner < 0 else owner)
        if launcher is None:
            return
        target = self.balls.get(dst) if dst != owner else None
        max_velocity = 0.0
        ps = list(launcher.pos)
        if target is not None and aimed == 1 and not target.is_cloaked:
            self._hit("launch.aimed")
            v0 = _normalized(_sub(target.pos, ps))
        else:
            vs = list(launcher.vel)
            max_velocity = float(launcher.max_vel)
            direction = _normalized(vs)
            if _lensq(direction) == 0.0:
                self._hit("launch.stationary")
                direction = [float(c) for c in self._rotated_vector(launcher, (0.0, 0.0, -1.0))]
            v0 = d_add(vs, d_scale(direction, _fmax(MISSILE_LAUNCH_SPEED, max_velocity)))
        if massive:
            self.set_massive(src, True)
        self.set_position(src, *ps)
        self.set_velocity(src, *v0)
        if target is not None and target.is_cloaked:
            return
        if target is None:
            if max_velocity > 0.0:
                self._hit("launch.straight")
                self.goto_direction(src, *v0)
            return
        self.missile_follow(src, dst, owner)

    def cloak_ball(self, bid: int, mode: int, range_: float = 2000.0) -> None:
        """PyCloakBall ("Lb|f") and CloakBall (Ballpark.cpp:5223; SEN-711, MOV-730): on the master a normal cloak adds
        a 2 s proximity sensor; the followers are released; the ball is non-massive with isCloaked = mode."""
        mode, range_ = _char(mode), _float32(range_)
        b = self.balls.get(bid)
        if b is None or mode <= 0:
            self._hit("cloak.refused")
            return
        self._hit("cloak.applied")
        if self.is_master and mode == DSTNORMALCLOAK:
            self._hit("cloak.sensor")
            self.add_proximity_sensor(bid, range_, CLOAK_SENSOR_PERIOD, 0)
        self._stop_all_followers(b)
        b.is_cloaked = mode & 0xFF
        b.is_massive = False

    def uncloak_ball(self, bid: int) -> None:
        """UncloakBall (Ballpark.cpp:5257): the normal cloak's sensor removed; massive again unless in warp proper, and
        then dropped from the sensors of its bubble."""
        b = self.balls.get(bid)
        if b is None:
            return
        if self.is_master and b.is_cloaked == DSTNORMALCLOAK:
            self.remove_proximity_sensor(bid)
        b.is_cloaked = 0
        if not self._is_warping(b):
            self._hit("uncloak.massive")
            b.is_massive = True
            for sid in self.proximity_balls.keys():
                s = self.balls[sid]
                if s.bubble != b.bubble or s.is_moribund or not s.has_proximity:
                    continue
                if bid in s.sensor_balls:
                    self._hit("uncloak.sensor_erase")
                s.sensor_balls.erase(bid)
        else:
            self._hit("uncloak.warping")

    def set_ball_troll(self, bid: int, delay: int) -> None:
        """PySetBallTroll ("Li") and SetBallTroll (Ballpark.cpp:6290; MOV-736)."""
        b = self.balls.get(bid)
        if b is None:
            return
        delay = int(delay)
        if delay < TROLL_MIN_DELAY:
            delay = TROLL_MIN_DELAY
        self._stop(b)
        self.set_free(bid, True)
        self.set_interactive(bid, True)
        b.effect_stamp = self.current_time + delay
        self._set_mode(b, TROLL)

    def _troll_ready(self, b) -> bool:
        """TrollReady (Ballpark.cpp:6315)."""
        return b.mode == TROLL and not b.effect_stamp > self.current_time

    def _petrify_troll(self, b) -> None:
        """PetrifyTroll (Ballpark.cpp:6327): fixed, non-interactive, RIGID."""
        if not self._troll_ready(b):
            return
        self._hit("troll.petrified")
        self.set_free(b.id, False)
        self.set_interactive(b.id, False)
        self._set_mode(b, RIGID)

    def set_ball_rigid(self, bid: int) -> None:
        """SetBallRigid (Ballpark.cpp:6278)."""
        b = self.balls.get(bid)
        if b is not None:
            self._stop(b)
            self._set_mode(b, RIGID)

    def set_boid(self, bid: int, on: int) -> None:
        """PySetBoid (Thunkers.cpp:1003): BOID mode, or a Stop."""
        b = self.balls.get(bid)
        if b is None:
            return
        if on:
            self._set_mode(b, BOID)
        else:
            self._stop(b)

    def add_mushroom(self, owner_id: int, range_: float, time: float) -> None:
        """PyAddMushroom ("Lfd") and AddMushroom (Ballpark.cpp:3719): a fixed massive ball at the owner (mass 1e20f),
        id 0 (a local id on the master), its range and lifetime in mFollowRange and mGoto[0]."""
        range_, time = _float32(range_), float(time)
        if range_ <= 0.0 or time <= 0:
            return
        owner = self.balls.get(owner_id)
        if owner is None:
            return
        if self.dynamical_orientation:
            m = self._add_dynamic_ball(0, _MUSHROOM_MASS, 0.0, 0.0, 0.0, False, False, True, False, False,
                                       owner.pos[0], owner.pos[1], owner.pos[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                       0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        else:
            m = self._add_old_style_ball(0, _MUSHROOM_MASS, 0.0, 0.0, False, False, True, False, False,
                                         owner.pos[0], owner.pos[1], owner.pos[2], 0.0, 0.0, 0.0, 1.0, 1.0)
        m.follow_range = range_
        m.goto[0] = time
        m.owner_id = int(owner_id)
        m.effect_stamp = self.current_time
        self._set_mode(m, MUSHROOM)

    # ---- formations (MOV-726) ----

    def _stop(self, b) -> None:
        """Stop(Ball*) (Ballpark.cpp:4578): modes.py's, and on the master a FORMATION follower's slot freed in its
        leader first (Ballpark.cpp:4629-4633: FreeFormationSlot(mEffectStamp), a size_t). Added 25 October 2026 by
        the compiled code frees it and the reference did not; found by running the combat scenarios
        (tests/destiny_combat_cases.py) on the harness, which the slice's own scenarios had not reached."""
        if self.is_master and b.mode == FORMATION and b.follow_ptr is not None:
            self.free_formation_slot(b.follow_ptr, b.effect_stamp)
        super()._stop(b)

    def load_formations(self, formations: list) -> None:
        """PyLoadFormations (Thunkers.cpp:1921) and AddFormation (Ballpark.cpp:6651): the slot offsets, as doubles."""
        self.formations = [[[float(c) for c in v] for v in vectors] for vectors in formations]

    def set_ball_formation(self, bid: int, formation_id: int) -> None:
        """PySetBallFormation ("Li", then a char) and SetBallFormation (Ballpark.cpp:6673). Leaving a formation stops
        every follower and frees the slots; a smaller formation stops the followers in the slots it lacks but never
        frees those slots' bits (its loop runs from the old size up to the new, smaller one)."""
        fid = _char(formation_id)
        b = self.balls.get(bid)
        if b is None or fid > len(self.formations) - 1:
            return
        if b.formation_id >= 0:
            if fid == -1:
                self._hit("formation.leave")
                self._stop_all_followers(b)
                b.formation_slots = 0
            elif fid != b.formation_id:
                if fid < 0:
                    raise DestinyUndefined("SetBallFormation reads mFormations at a negative index (Ballpark.cpp:6696)")
                new_len = len(self.formations[fid])
                if b.formation_id > len(self.formations) - 1:
                    raise DestinyUndefined("SetBallFormation reads mFormations past its end: the ball's formation is "
                                           "gone after a smaller LoadFormations (Ballpark.cpp:6696)")
                if len(self.formations[b.formation_id]) > new_len:
                    self._hit("formation.shrink")
                    stop = []
                    for fol in b.followers.ordered():
                        f = self.balls.get(fol)
                        if f is None:
                            raise DestinyUndefined("SetBallFormation dereferences a follower missing from the park "
                                                   "(Ballpark.cpp:6705)")
                        if f.mode != FORMATION:
                            continue
                        if f.effect_stamp > new_len - 1:
                            stop.append(f)
                    for f in stop:
                        self._stop(f)
        b.formation_id = fid

    def formation_follow(self, src: int, dst: int, slot: int) -> None:
        """PyFormationFollow ("LLi", then a char) and FormationFollow (Ballpark.cpp:3947)."""
        slot = _char(slot)
        ball = self.balls.get(src)
        if ball is None or src == dst:
            return
        dest = self.balls.get(dst)
        if dest is None or dest.formation_id < 0 or dest.formation_id > len(self.formations) - 1:
            return
        if slot > len(self.formations[dest.formation_id]) - 1:
            return
        self._stop(ball)
        if ball.bubble != dest.bubble or dest.is_cloaked:
            self._hit("formation.refused")
            return
        ball.follow_id = dst
        ball.follow_ptr = dst
        ball.effect_stamp = slot
        self._set_mode(ball, FORMATION)
        dest.followers.add(ball.id)

    def reserve_formation_slot(self, bid: int) -> int:
        """Ball::ReserveFormationSlot (Ball.cpp:2275): the first free slot of at most 16, or -1."""
        b = self.balls[bid]
        fid = b.formation_id
        if fid == -1 or fid > len(self.formations) - 1:
            return -1
        if fid < 0:
            raise DestinyUndefined("ReserveFormationSlot reads mFormations at a negative index (Ball.cpp:2286)")
        n = len(self.formations[fid])
        if n > FORMATION_MAX_SLOTS:
            return -1
        for i in range(n):
            if b.formation_slots >> i & 1:
                continue
            b.formation_slots |= 1 << i
            return i
        return -1

    def free_formation_slot(self, bid: int, slot: int) -> None:
        """Ball::FreeFormationSlot (Ball.cpp:2324), with a size_t slot."""
        b = self.balls.get(bid)
        if b is None:
            return
        fid = b.formation_id
        if fid == -1 or fid > len(self.formations) - 1:
            return
        if fid < 0:
            raise DestinyUndefined("FreeFormationSlot reads mFormations at a negative index (Ball.cpp:2340)")
        n = len(self.formations[fid])
        slot = int(slot) & 0xFFFFFFFFFFFFFFFF
        if slot > ((n - 1) & 0xFFFFFFFFFFFFFFFF):
            return
        if slot >= FORMATION_MAX_SLOTS:
            raise DestinyUndefined("std::bitset<16>::reset out of range (Ball.cpp:2351)")
        b.formation_slots &= ~(1 << slot)

    # ---- the accuracy helpers (CMB-701, CMB-702) ----

    def get_accuracy(self, src: int, dst: int, optimal: float, falloff: float, tracking: float, signature: float,
                     optimal_signature: float) -> tuple:
        """PyGetAccuracy and GetAccuracy (Ballpark.cpp:5156): (2^E, the surface distance, the angular speed); the
        early returns leave the last two as given."""
        optimal, tracking = float(optimal), float(tracking)
        falloff, signature, optimal_signature = float(falloff), float(signature), float(optimal_signature)
        if src == dst:
            return 0.0, optimal, tracking
        s, d = self.balls.get(src), self.balls.get(dst)
        if s is None or d is None or optimal < 0.0:
            return 0.0, optimal, tracking
        if tracking == 0.0 or signature == 0.0 or falloff == 0.0:
            return 0.0, optimal, tracking
        radial = _sub(d.pos, s.pos)
        dist = _length(radial)
        short = (dist - d.radius) - s.radius
        if short < 0.0:
            short = 0.0
        if dist == 0.0:
            return 0.0, optimal, tracking
        combined = _sub(d.vel, s.vel)
        radial_speed = _dot(combined, radial) / dist
        speed2 = (_lensq(combined) - radial_speed * radial_speed) / (dist * dist)
        exponent = _ieee_div(optimal_signature, tracking * signature)
        exponent *= -exponent * speed2
        if short > optimal:
            self._hit("accuracy.falloff")
            exponent -= _ieee_div((short - optimal) * (short - optimal), falloff * falloff)
        try:
            return math.pow(2.0, exponent), short, _sqrt(speed2)
        except OverflowError:                       # C's pow returns +inf where Python's raises
            return math.inf, short, _sqrt(speed2)

    def relative_velocities(self, src: int, dst: int, flags: int) -> tuple:
        """Ball::CalculateRelativeVelocities (Ball.cpp:161): (relative velocity, radial speed, surface distance,
        transversal speed, angular speed), each left at -7777 where it is not computed, as the harness's driver."""
        rel, rad, short, trans, ang = [-7777.0] * 3, -7777.0, -7777.0, -7777.0, -7777.0
        s, target = self.balls.get(src), self.balls.get(dst)
        if s is None or target is None:
            return rel, rad, short, trans, ang
        c_rel, c_rad, c_trans, c_ang = bool(flags & 1), bool(flags & 2), bool(flags & 4), bool(flags & 8)
        if c_ang:
            c_trans = True
        if c_trans:
            c_rad = True
        if c_rad:
            c_rel = True
        if c_rel:
            rel = _sub(target.vel, s.vel)
            if c_rad:
                radial = _sub(target.pos, s.pos)
                dist = _length(radial)
                if dist == 0.0:
                    return rel, rad, short, trans, ang
                short = (dist - target.radius) - s.radius
                if short < 0.0:
                    short = 0.0
                radial = d_div(radial, dist)
                rad = _dot(rel, radial)
                if c_trans:
                    tv = _sub(rel, d_scale(radial, rad))
                    trans = _length(tv)
                    if c_ang:
                        if self.dynamical_orientation:
                            omega = _cross([float(c) for c in s.ang_vel], radial)
                            tv = _sub(tv, d_scale(omega, dist))
                            ang = _length(tv) / dist
                        else:
                            ang = trans / dist
        return rel, rad, short, trans, ang

    def center_dist(self, a: int, b: int):
        """PyGetCenterDist and PyGetSurfaceDist (Thunkers.cpp:1774, 1798), or None for a missing ball."""
        ba, bb = self.balls.get(a), self.balls.get(b)
        if ba is None or bb is None:
            return None
        d = _length(_sub(ba.pos, bb.pos))
        return d, d - ba.radius - bb.radius

    def shifted_time(self, bid: int, time: int) -> int:
        """ClientBall::GetShiftedTime (Ball.cpp:1318; TIME-708): the render time less two tick intervals."""
        return int(time) - 2 * self.tick_interval * 10000

    # ================================================================================================ orientation

    def _yaw_pitch_roll(self, b, snap: bool = False) -> None:
        """Ball::CalculateYawPitchRoll (Ball.cpp:335; MOV-729): yaw and pitch of the velocity (the goto direction at
        rest; mixed with it for non-missiles), smoothed by the mass; roll from the yaw changes."""
        self._hit("ypr.snap" if snap else "ypr.evolve")
        b.old_pitch, b.old_yaw, b.old_roll, b.old_roll_speed = b.pitch, b.yaw, b.roll_angle, b.roll_speed
        mm = b.mass / self.friction
        mm = 100.0 / mm if mm != 0.0 else 100.0
        vel2 = _lensq(b.vel)
        if vel2 == 0.0:
            direction = _sub(b.goto, b.pos)
        else:
            direction = [0.0, 0.0, 0.0]
            if b.mode == WARP:
                if b.effect_stamp < 0:
                    direction = _sub(b.goto, b.pos) if not b.collisions else list(b.vel)
                elif mm < 1.0 and self.current_time != b.effect_stamp:
                    self._hit("ypr.warp_mass")
                    inner = (self.current_time - b.effect_stamp) * 2
                    inner = 15.0 if 15.0 < inner else float(inner)
                    x = mm * inner
                    mm = 1.0 if 1.0 < x else x
            else:
                direction = list(b.vel)
            if b.mode != MISSILE and not snap:
                offset = _normalized(_sub(b.goto, b.pos))
                direction = d_add(direction, d_scale(offset, math.sqrt(vel2) * 0.3))
        b.yaw = math.atan2(direction[0], direction[2])
        r = _length(direction)
        if r > 0.0:
            b.pitch = -math.asin(_fmin(1.0, _fmax(-1.0, direction[1] / r)))
        else:
            b.pitch = 0.0
        b.old_yaw, b.yaw = _modulo2pi(b.old_yaw, b.yaw)
        b.old_pitch, b.pitch = _modulo2pi(b.old_pitch, b.pitch)
        if snap:
            b.roll_angle = b.old_roll = b.roll_speed = b.old_roll_speed = b.yaw_delta = 0.0
            return
        tmp = 1.0 / (1.0 + mm * 0.5)
        b.yaw = (b.old_yaw + mm * 0.5 * b.yaw) * tmp
        b.pitch = (b.old_pitch + mm * 0.5 * b.pitch) * tmp
        b.yaw_delta = math.asin(math.sin(b.yaw - b.old_yaw))
        b.roll_speed += b.yaw_delta * ROLL_SPEED_ACCELERATION
        b.roll_speed += b.old_roll_speed * ROLL_SPEED_DECAY * self.dt
        b.roll_speed = _fmax(-HALF_PI, _fmin(HALF_PI, b.roll_speed))
        b.roll_angle += b.roll_speed * ROLL_ACCELERATION * self.dt
        b.roll_angle += b.old_roll * ROLL_DECAY * self.dt
        b.roll_angle = _fmax(-HALF_PI, _fmin(HALF_PI, b.roll_angle))

    def _snap_orientation(self, b) -> None:
        """Ball::SnapOrientation (Ball.cpp:290): the rotation turned to the velocity and rolled up; the rotation
        stopped. Nothing changes below a speed of 1e-6 m/s."""
        if not _lensq(b.vel) > 1e-12:
            self._hit("snap.at_rest")
            return
        direction = v3_of(b.vel)
        len_dir = v3_len(direction)
        if len_dir < 1e-12:
            b.rot = QUAT_IDENTITY
        else:
            vec = v3_cross(v3(0.0, 0.0, 1.0), direction)
            angle = _acosf(_fmax(-1.0, _fmin(1.0, _fdiv(direction[2], len_dir))))
            b.rot = q_rotation(vec, angle)
            up = transform_coord((0.0, 1.0, 0.0), m_rotation_q(q_conj(b.rot)))
            up_roll = _atan2f(-up[0], up[1]) if up[2] < _f(0.99) else 0.0
            heading = transform_coord((0.0, 0.0, 1.0), m_rotation_q(b.rot))
            b.rot = q_mul(b.rot, q_rotation(heading, up_roll))
        b.old_rot = b.rot
        b.ang_vel = b.old_ang_vel = (0.0, 0.0, 0.0)

    def _rotated_vector(self, b, vec) -> tuple:
        """Ball::GetRotatedVector (Ball.cpp:2191): by the rotation with dynamical orientation, otherwise by the yaw,
        pitch and roll (as floats) through DirectXMath's RotationQuaternion (`q_rotation_ypr`)."""
        if self.dynamical_orientation:
            return transform_coord(vec, m_rotation_q(b.rot))
        return transform_coord(vec, m_rotation_q(q_rotation_ypr(_f(b.yaw), _f(b.pitch), _f(b.roll_angle))))

    # ---- dynamical orientation (MOV-728): Ballpark.cpp:1580-1983, in float ----

    def _tau(self, b) -> float:
        """(float(mass) * agility * rotAgility) / float(friction)."""
        return _fdiv(_f(_f(_f(b.mass) * b.agility) * b.rot_agility), _f(self.friction))

    def _t_scale(self, b) -> float:
        """float(friction) / (float(mass) * agility * rotAgility)."""
        return _fdiv(_f(self.friction), _f(_f(_f(b.mass) * b.agility) * b.rot_agility))

    def _do_disabled(self, b) -> bool:
        """DynamicalOrientationDisabledForBall (Ballpark.cpp:408)."""
        return self.disable_do_for_missiles and b.mode == MISSILE

    @staticmethod
    def _n2_alpha2_v(omega0, len_omega0: float, b, tf: float, dt: float, t_scale: float) -> tuple:
        """CalculateN2Alpha2, the vector version (Ballpark.cpp:1580): (n2, kappa2, alpha2)."""
        if len_omega0 > 1e-10:
            factor = _fdiv(b.max_ang_vel, len_omega0)
            n2 = _to_int32(_ceil(_fdiv(_logf(_fdiv(factor, _f(1.0 + factor))), _f(-dt * t_scale))))
            kappa2 = _f(math.pow(tf, n2))
            alpha2 = v3_mul((-omega0[0], -omega0[1], -omega0[2]), _fdiv(_f(t_scale * kappa2), _f(1 - kappa2)))
            return n2, kappa2, alpha2
        return 0, 1.0, (0.0, 0.0, 0.0)

    @staticmethod
    def _n2_alpha2_f(omega0: float, b, tf: float, dt: float, t_scale: float) -> tuple:
        """CalculateN2Alpha2, the float version (Ballpark.cpp:1598)."""
        if abs(omega0) > 1e-10:
            factor = _fdiv(b.max_ang_vel, abs(omega0))
            n2 = _to_int32(_ceil(_fdiv(_logf(_fdiv(factor, _f(1.0 + factor))), _f(-dt * t_scale))))
            kappa2 = _f(math.pow(tf, n2))
            alpha2 = _f(-omega0 * _fdiv(_f(t_scale * kappa2), _f(1 - kappa2)))
            return n2, kappa2, alpha2
        return 0, 1.0, 0.0

    def _goto_torque_breaks(self, b, direction) -> tuple:
        """GotoTorqueBreaks (Ballpark.cpp:1616): the heading torque, by a search for the accelerating (n1) and braking
        (n2) tick counts, at most 5 x 100 steps."""
        t_scale = self._t_scale(b)
        max_alpha = _f(b.max_ang_vel * t_scale)
        if max_alpha <= 0.0:
            return (0.0, 0.0, 0.0)
        len_dir = v3_len(direction)
        if len_dir == 0:
            return (0.0, 0.0, 0.0)
        heading = v3_normalize(transform_coord((0.0, 0.0, 1.0), m_rotation_q(b.rot)))
        rot = v3_div(v3_cross(heading, direction), len_dir)
        rot_len = v3_len(rot)
        rot_angle = _acosf(_fmax(-1.0, _fmin(1.0, _fdiv(v3_dot(direction, heading), len_dir))))
        if rot_len > 1e-12:
            rot = v3_mul(rot, _fdiv(rot_angle, rot_len))
        elif rot_angle > 1:
            self._hit("torque.flip")
            rot = (-rot_angle, 0.0, 0.0)
        omega0 = v3_sub(b.ang_vel, v3_mul(heading, v3_dot(heading, b.ang_vel)))
        len_omega0 = v3_len(omega0)
        dt = self.dt
        if len_omega0 * dt < 1e-4 and rot_angle < 1e-4:
            self._hit("torque.none")
            return (0.0, 0.0, 0.0)
        fdt = _f(dt)
        tf = _expf(_f(-fdt * t_scale))
        tau = _fdiv(1.0, t_scale)
        n1 = 0
        alpha1 = (0.0, 0.0, 0.0)
        kappa1 = 1.0
        n2, kappa2, alpha2 = self._n2_alpha2_v(omega0, len_omega0, b, tf, fdt, t_scale)
        for _ in range(5):
            if n2 > 0:
                theta2 = v3_add(v3_mul(v3_mul(alpha2, tau), _f(_f(n2 * dt) - _f(tau * _f(1.0 - kappa2)))),
                                v3_mul(v3_mul(omega0, tau), _f(1 - kappa2)))
                if v3_len(v3_sub(rot, theta2)) < 1e-3:
                    self._hit("torque.brake_only")
                    n1 = 0
                    break
            kappa1 = 1.0
            n1 = 1
            while n1 < 100:
                kappa1 = _f(kappa1 * tf)
                if n2 == 0:
                    alpha1 = v3_div(v3_sub(rot, v3_mul(omega0, _f(tau * _f(1.0 - kappa1)))),
                                    _f(tau * _f(_f(n1 * dt) - _f(tau * _f(1.0 - kappa1)))))
                else:
                    num = _f(tau - _fdiv(_f(_f(kappa1 * kappa2) * _f(n2 * dt)), _f(1 - kappa2)))
                    den = _f(tau * _f(_f(n1 * dt) - _fdiv(_f(_f(_f(n2 * dt) * kappa2) * _f(1 - kappa1)),
                                                          _f(1 - kappa2))))
                    alpha1 = v3_div(v3_sub(rot, v3_mul(omega0, num)), den)
                if v3_len(alpha1) < max_alpha:
                    break
                n1 += 1
            n2_old = n2
            omega1 = v3_add(v3_mul(alpha1, _f(tau * _f(1 - kappa1))), v3_mul(omega0, kappa1))
            n2, kappa2, alpha2 = self._n2_alpha2_v(omega1, v3_len(omega1), b, tf, fdt, t_scale)
            if n2_old == n2:
                break
        if n1 > 0:
            if n1 > 1:
                self._hit("torque.full")
                alpha1 = v3_mul(v3_normalize(alpha1), max_alpha)
            return alpha1
        if n2 > 0:
            return alpha2
        return (0.0, 0.0, 0.0)

    def _goto_roll_breaks(self, b, heading_torque) -> float:
        """GotoRollBreaks (Ballpark.cpp:1751): the roll torque, rolling into strong turns, otherwise upright."""
        t_scale = self._t_scale(b)
        max_alpha = _f(t_scale * b.max_ang_vel)
        if max_alpha <= 0:
            return 0.0
        mat = m_rotation_q(q_conj(b.rot))
        ship_omega = transform_coord(b.ang_vel, mat)
        omega_roll = ship_omega[2]
        len_torque = v3_len(heading_torque)
        turn_roll, weight = 0.0, 0.0
        if len_torque > 0.05 * max_alpha:
            ship_torque = transform_coord(heading_torque, mat)
            ship_omega = (ship_omega[0], ship_omega[1], 0.0)
            if v3_dot(ship_omega, ship_torque) >= 0 or v3_len(ship_omega) < _f(_f(0.01) * b.max_ang_vel):
                self._hit("roll.turn")
                turn_roll = _atan2f(-ship_torque[1], -ship_torque[0])
                weight = _fdiv(len_torque, t_scale)
                if weight > 1.0:
                    weight = 1.0
        up = transform_coord((0.0, 1.0, 0.0), mat)
        up_roll = _atan2f(-up[0], up[1]) if up[2] < _f(0.95) else 0.0
        desired = up_roll if weight < _f(0.05) else _f(weight * turn_roll)
        if abs(omega_roll) < 1e-4 and abs(desired) < 1e-4:
            self._hit("roll.none")
            return 0.0
        dt = self.dt
        fdt = _f(dt)
        tf = _expf(_f(-fdt * t_scale))
        tau = _fdiv(1.0, t_scale)
        n1, alpha1, kappa1 = 0, 0.0, 1.0
        n2, kappa2, alpha2 = self._n2_alpha2_f(omega_roll, b, tf, fdt, t_scale)
        for _ in range(5):
            if n2 > 0:
                theta2 = _f(_f(_f(tau * alpha2) * _f(_f(n2 * dt) - _f(tau * _f(1.0 - kappa2))))
                            + _f(_f(tau * omega_roll) * _f(1 - kappa2)))
                if abs(_f(desired - theta2)) < 1e-3:
                    self._hit("roll.brake_only")
                    n1 = 0
                    break
            kappa1 = 1.0
            n1 = 1
            while n1 < 100:
                kappa1 = _f(kappa1 * tf)
                if n2 == 0:
                    alpha1 = _fdiv(_f(desired - _f(omega_roll * _f(tau * _f(1.0 - kappa1)))),
                                   _f(tau * _f(_f(n1 * dt) - _f(tau * _f(1.0 - kappa1)))))
                else:
                    num = _f(tau - _fdiv(_f(_f(kappa1 * kappa2) * _f(n2 * dt)), _f(1 - kappa2)))
                    den = _f(tau * _f(_f(n1 * dt) - _fdiv(_f(_f(_f(n2 * dt) * kappa2) * _f(1 - kappa1)),
                                                          _f(1 - kappa2))))
                    alpha1 = _fdiv(_f(desired - _f(omega_roll * num)), den)
                if abs(alpha1) < max_alpha:
                    break
                n1 += 1
            n2_old = n2
            omega1 = _f(_f(_f(alpha1 * tau) * _f(1 - kappa1)) + _f(omega_roll * kappa1))
            n2, kappa2, alpha2 = self._n2_alpha2_f(omega1, b, tf, fdt, t_scale)
            if n2_old == n2:
                break
        if n1 > 0:
            if n1 > 1:
                alpha1 = _f((1.0 if alpha1 >= 0.0 else -1.0) * max_alpha)
            return alpha1
        if n2 > 0:
            return alpha2
        return 0.0

    def _apply_torque(self, rotation, angvel, torque, roll: float, tau: float, fraction: float) -> tuple:
        """ApplyTorque (Ballpark.cpp:1936): the rotation and angular velocity a time step fraction on, heading and
        roll separately; (rotation, angular velocity)."""
        fdt = _f(self.dt)
        tf = _expf(_fdiv(_f(-fdt * fraction), tau))
        heading = transform_coord((0.0, 0.0, 1.0), m_rotation_q(rotation))
        omega_roll = v3_dot(angvel, heading)
        omega_heading = v3_sub(angvel, v3_mul(heading, omega_roll))
        span = _f(_f(fraction * fdt) - _f(tau * _f(1.0 - tf)))
        rot_roll = _f(tau * _f(_f(roll * span) + _f(omega_roll * _f(1.0 - tf))))
        rot_heading = v3_mul(v3_add(v3_mul(torque, span), v3_mul(omega_heading, _f(1.0 - tf))), tau)
        angle_heading = v3_len(rot_heading)
        apply_heading = angle_heading > 1e-10
        quat_heading = q_rotation(rot_heading, v3_len(rot_heading)) if apply_heading else QUAT_IDENTITY
        heading_rotated = (transform_coord(heading, m_rotation_q(quat_heading)) if apply_heading else heading)
        angvel = v3_add(v3_mul(torque, _f(tau * _f(1.0 - tf))), v3_mul(omega_heading, tf))
        angvel = v3_add(angvel, v3_mul(heading_rotated,
                                       _f(_f(tau * roll) - _f(_f(_f(tau * roll) - omega_roll) * tf))))
        if apply_heading:
            rotation = q_mul(rotation, quat_heading)
        rotation = q_mul(rotation, q_rotation(heading_rotated, rot_roll))
        norm = q_len(rotation)
        if norm > 0:
            rotation = q_div(rotation, norm)
        return rotation, angvel

    # ================================================================================================== behaviours

    def _goto_thrust(self, b, target, missile: bool = False) -> list:
        """GotoThrust (Ballpark.cpp:1398; MOV-706, MOV-723): toward the target at maxThrust, tapered near it unless
        a missile; C's division where the mass may be zero."""
        a = _sub(target, b.pos)
        length2 = _lensq(a)
        dist = float(b.speed_fraction) * b.max_vel * self.dt
        max_thrust = _ieee_div(self.friction * b.speed_fraction * float(b.max_vel), b.mass * b.agility)
        unit = _normalized(a)
        if not missile and length2 < dist * dist:
            coff = _ieee_div(length2, dist * dist)
            return [c * (max_thrust * coff * coff) for c in unit]
        return [c * max_thrust for c in unit]

    def goto_thrust(self, b) -> list:
        """EvolveGoto: GotoThrust toward mGoto (every caller in ref.py and modes.py)."""
        return self._goto_thrust(b, b.goto)

    def _evolve_missile(self, b) -> list:
        """EvolveMissile (Ballpark.cpp:972; MOV-723): straight on while (t - launch) x tick <= 800 ms, then at full
        thrust toward the target's position predicted dist / maxVel ahead, offset by the follow range."""
        if (self.current_time - b.effect_stamp) * self.tick_interval <= MISSILE_STRAIGHT_MS:
            self._hit("missile.straight")
            return self._goto_thrust(b, b.goto)
        self._hit("missile.pursue")
        other = self.balls[b.follow_ptr]
        op = list(other.pos)
        delta = _sub(b.pos, op)
        dist = _length(delta)
        ct = _ieee_div(dist, b.max_vel)
        target = d_add(op, d_scale(other.vel, ct))
        r = float(b.follow_range) + float(b.radius) + float(other.radius)
        if dist == 0.0:
            self._hit("missile.on_target")
            target = [op[0] + 1.0 * r, op[1] + 0.0 * r, op[2] + 0.0 * r]
        else:
            target = d_add(target, d_div(d_scale(delta, r), dist))
        b.goto = target
        return self._goto_thrust(b, target, True)

    def _evolve_formation(self, b) -> list:
        """EvolveFormation (Ballpark.cpp:1023; MOV-726): the leader's slot offset, scaled to both radii plus its length,
        turned by the leader's rotation (through a float vector)."""
        other = self.balls[b.follow_ptr]
        if other.formation_id < 0 or other.formation_id > len(self.formations) - 1:
            self._hit("formation.no_formation")
            return [0.0, 0.0, 0.0]
        slot = _char(b.effect_stamp)
        formation = self.formations[other.formation_id]
        if slot > len(formation) - 1:
            self._hit("formation.no_slot")
            return [0.0, 0.0, 0.0]
        if slot < 0:
            raise DestinyUndefined("EvolveFormation reads a formation at a negative slot (Ballpark.cpp:1044)")
        offset = formation[slot]
        length = _length(offset)
        if length == 0.0:
            self._hit("formation.zero_offset")
            length = 1.0
        vec = v3_of(d_div(d_scale(offset, float(b.radius) + float(other.radius) + length), length))
        vec = self._rotated_vector(other, vec)
        op = [other.pos[i] + vec[i] for i in range(3)]
        b.goto = op
        self._hit("formation.slot")
        return self._goto_thrust(b, op)

    def _orbital_normal(self, b) -> tuple:
        """GetOrbitalNormal (Ballpark.cpp:664) with useNewOrbit on: modes.py's, and with dynamical orientation the
        fallback from the ship's heading."""
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
        if not self.dynamical_orientation:
            self._hit("orbit_new.default_normal")
            return to, dist, self._default_orbital_normal(b)
        heading = self._rotated_vector(b, (0.0, 0.0, 1.0))
        o = _cross(to, [float(c) for c in heading])
        length2 = _lensq(o)
        if length2 <= 1e-12:
            self._hit("orbit_new.default_normal")
            return to, dist, self._default_orbital_normal(b)
        self._hit("orbit_new.heading_normal")
        inv = 1.0 / math.sqrt(length2)
        return to, dist, [c * inv for c in o]

    def _behaviour(self, b) -> None:
        """EvolveBehaviorForBall (Ballpark.cpp:789): the mode's acceleration (STOP damps unless orientation is
        dynamical; BOID, TROLL, MINIBALL, FIELD, RIGID and MUSHROOM get none), then dynamical orientation, or
        CapAcceleration, which returns at once (MOV-727)."""
        b.last_g = [0.0, 0.0, 0.0]
        b.last_c = [0.0, 0.0, 0.0]
        b.collisions = []
        a = [0.0, 0.0, 0.0]
        m = b.mode
        if m == WARP:
            a = self._evolve_warp(b)
        elif m == GOTO:
            a = self.goto_thrust(b)
        elif m == MISSILE:
            a = self._evolve_missile(b)
        elif m == FORMATION:
            a = self._evolve_formation(b)
        elif m == FOLLOW:
            a = self._evolve_follow(b)
        elif m == ORBIT:
            a = self._evolve_new_orbit(b) if self.use_new_orbit else self._evolve_old_orbit(b, self.current_time)
        elif m == STOP:
            if not self.dynamical_orientation:
                b.vel[1] = (b.vel[1] - 0.07 * b.vel[1]) * STOP_DAMP
        else:
            self._hit("mode.inert")
        if self.dynamical_orientation and not self._do_disabled(b):
            a = self._orient(b, a)
        else:
            a = self._cap_acceleration(b, a)
        b.last_g = a

    def _cap_acceleration(self, b, a) -> list:
        """CapAcceleration (Ballpark.cpp:1350; MOV-727): returns at once, so the thrust keeps its direction."""
        self._hit("cap_acceleration")
        return a

    def _orphaned(self, b) -> bool:
        """The master's test for a missile's self-collision (Ballpark.cpp:614-638; COL-719): its owner (the absolute
        id, mBalls[ABS(mOwnerId)] at 633) is gone, cloaked, or in another bubble while the missile has one."""
        owner = self.balls.get(abs(b.owner_id))
        return owner is None or (owner.bubble != b.bubble and b.bubble != -1) or bool(owner.is_cloaked)

    def _orient(self, b, a) -> list:
        """EvolveBehaviorForBall's dynamical-orientation branch (Ballpark.cpp:840-890; MOV-728): above 1 % of the
        maximum acceleration, the torque and roll are found, the rotation a tick on is predicted, and the thrust is
        along that heading, scaled by cos^2 of its angle to the wanted thrust (zero at 90 degrees or more)."""
        accel = _f(_length(a))
        max_accel = _fdiv(_f(_f(_f(self.friction) * b.speed_fraction) * b.max_vel), _f(_f(b.mass) * b.agility))
        if accel > 1e-2 * max_accel:
            self._hit("orient.thrust")
            b.torque = self._goto_torque_breaks(b, v3_of(a))
            b.rot_roll = self._goto_roll_breaks(b, b.torque)
            rotation, _ = self._apply_torque(b.rot, b.ang_vel, b.torque, b.rot_roll, self._tau(b), 1.0)
            heading = transform_coord((0.0, 0.0, 1.0), m_rotation_q(rotation))
            cosangle = _fmin(1.0, _fdiv(v3_dot(heading, v3_of(a)), accel))
            if cosangle <= 0.0:
                self._hit("orient.behind")
                return [0.0, 0.0, 0.0]
            h = v3_mul(heading, _f(_f(cosangle * cosangle) * accel))
            return [h[0], h[1], h[2]]
        self._hit("orient.coast")
        if self._is_warping(b):
            b.torque = self._goto_torque_breaks(b, v3_of(_sub(b.goto, b.pos)))
        else:
            b.torque = (0.0, 0.0, 0.0)
        b.rot_roll = self._goto_roll_breaks(b, b.torque)
        return a

    # ===================================================================================================== the tick

    def evolve(self, timestamp: int = 0) -> None:
        """Ballpark::Evolve (Ballpark.cpp:421; TIME-706) whole: behaviours; the iterative responses; collision gradient
        and end-of-tick states (and the rotation a tick on) into the old members; then for each live ball the time
        stamps, a mushroom's growth or removal, the swap, the orientation, the re-filing and a missile's collision
        events; the trolls due are petrified; HandleProximities; the clock."""
        self.in_evolve = True
        order = sorted(self.free_balls)
        do = self.dynamical_orientation
        for bid in order:
            b = self.balls[bid]
            if b.is_moribund:
                continue
            if self.in_dead_bubble(b):
                self._hit("tick.dead_bubble")
                b.in_dead_bubble = True
                continue
            self._behaviour(b)
        if self.iterative_collision:
            self._collision_responses()
        for bid in order:
            b = self.balls[bid]
            if b.is_moribund or b.in_dead_bubble:
                continue
            if not self.iterative_collision and b.is_massive:
                b.last_collision = -1.0
                self._gradient(b)
            p, v = list(b.pos), list(b.vel)
            if self._is_warping(b):
                p, v = self._warp_distance(b, p, v, (self.current_time - b.effect_stamp) * self.dt)
                if b.mode == STOP:
                    b.pos, b.vel = list(p), list(v)
                    a = [b.last_g[i] + b.last_c[i] for i in range(3)]
                    p, v = self.integrate(p, v, a, b.mass * b.agility, self.friction, b.time_factor, self.dt)
                    if _lensq(b.vel) < 0.1:
                        self._hit("warp.drop_out_insert")
                        self._insert_ball_in_boxes(b)
            else:
                p, v = self._position_velocity(b, p, v)
            if do and not self._do_disabled(b):
                if self.iterative_collision:
                    rot, w = self._rotation_velocity(b, 1.0, b.rot, b.ang_vel)
                else:
                    rot, w = self._apply_torque(b.rot, b.ang_vel, b.torque, b.rot_roll, self._tau(b), 1.0)
                b.old_rot, b.old_ang_vel = rot, w
            b.old_pos, b.old_vel = p, v
        trolls = []
        for bid in order:
            b = self.balls[bid]
            if b.is_moribund:
                continue
            if b.in_dead_bubble:
                b.in_dead_bubble = False
                continue
            if b.mode == TROLL and self._troll_ready(b):
                trolls.append(b)
            if timestamp != 0:
                b.old_time = timestamp - self.tick_interval * 10000 if b.new_time == 0 else b.new_time
                b.new_time = timestamp
            if b.mode == MUSHROOM:
                fraction = _fdiv(_f(_f(float((self.current_time - b.effect_stamp) * self.tick_interval))
                                    * _f(0.001)), _f(b.goto[0]))
                if fraction < 1.0:
                    self._hit("mushroom.grow")
                    b.radius = _f(b.follow_range * _powf(fraction, 0.25))
                    self._insert_ball_in_boxes(b)
                else:
                    self._hit("mushroom.done")
                    self.remove_ball(b.id)
                continue
            b.pos, b.old_pos = b.old_pos, b.pos
            b.vel, b.old_vel = b.old_vel, b.vel
            if do:
                if math.isfinite(q_lensq(b.old_rot)):
                    b.rot, b.old_rot = b.old_rot, b.rot
                if math.isfinite(v3_dot(b.old_ang_vel, b.old_ang_vel)):
                    b.ang_vel, b.old_ang_vel = b.old_ang_vel, b.ang_vel
                if self._do_disabled(b):
                    self._hit("orient.missile_snap")
                    self._snap_orientation(b)
            else:
                self._yaw_pitch_roll(b)
            if _lensq(_sub(b.pos, b.old_pos)) < DISPATCH_MIN_MOVE_SQ:
                continue
            self._insert_ball_in_boxes(b)
            if b.mode == MISSILE:
                if self.iterative_collision:
                    prop = self.collision_locations.get(b.id)
                    if prop is not None:
                        for loc in prop.locations:
                            if loc.collider != -1:
                                b.collisions.append(loc.collider)
                if self.is_master and self._orphaned(b):
                    self._hit("missile.self_collision")
                    b.collisions.append(b.id)
                for c in b.collisions:                      # Ball::DispatchCollisions (COL-719)
                    self._hit("missile.collision_event")
                    self._event("DoCollision", b.id, c, b.last_c[0], b.last_c[1], b.last_c[2])
        for t in trolls:
            self._petrify_troll(t)
        self._handle_proximities()
        self.current_time += 1
        self.in_evolve = False

    def on_tick(self, sim_time: int, real_time: int = 0) -> None:
        """Ballpark::OnTick (Ballpark.cpp:219; TIME-702 to TIME-705): moribund balls out; whole tick intervals only
        (100 ns units, mTickInterval x 10,000 each); the master evolves one interval instead of more than 5; the first
        call evolves once, without tick events; each interval DoPreTick, Evolve at its timestamp, DoPostTick; the
        remainder carried in mTime."""
        sim_time = int(sim_time)
        last_delta = sim_time - self.time
        self.bring_out_the_deads()
        interval = self.tick_interval * 10000
        if last_delta < interval:
            self._hit("tick.too_early")
            return
        self.dt = self.tick_interval * 0.001
        if self.is_master and last_delta // interval > MAX_CATCH_UP_TICKS:
            self._hit("tick.resync")
            last_delta = interval
        if self.first_time:
            self._hit("tick.first")
            self.evolve(sim_time)
            last_delta = 0
            self.first_time = False
        else:
            intervals = last_delta // interval
            if intervals > 1:
                self._hit("tick.catch_up")
            for _ in range(intervals):
                self._event("DoPreTick", -2, self.current_time)
                for handler in self.pre_tick_handlers:
                    handler(self.current_time)
                self.evolve(sim_time - last_delta)
                self._event("DoPostTick", -2, self.current_time)
                for handler in self.post_tick_handlers:
                    handler(self.current_time)
                last_delta -= interval
        self.time = sim_time - last_delta

    # ================================================================================ collisions with static shapes

    def _gradient(self, b) -> None:
        """Ballpark::Gradient (Ballpark.cpp:2745): Potential with each ball candidate (missiles see only their target or,
        defending, every visible ball), then each static collidable's CollideWithBall unless a missile or mushroom."""
        uni, stat = self._collision_candidates(b)
        for n in uni:
            if ((b.mode == MISSILE and (n.id == b.owner_id or (n.mode == MINIBALL and n.owner_id == b.owner_id)))
                    or ((n.mode == MISSILE or n.mode == MUSHROOM) and b.id == n.owner_id)
                    or (n.mode == MISSILE and b.id == n.follow_id)
                    or (n.is_space_junk and not b.is_space_junk)):
                self._hit("gradient.skip")
                continue
            self._potential(b, n, 0)
        for s in stat:
            if b.mode == MISSILE or b.mode == MUSHROOM:
                self._hit("gradient.static_skip")
                continue
            self._collide_with_ball(s, b)

    def _collide_with_ball(self, s, b) -> None:
        """Capsule::CollideWithBall (Capsule.cpp:365) and OrientedBox::CollideWithBall (OrientedBox.cpp:44): the free
        ball's straight path over the tick (predicted with mLastG)."""
        if not b.is_free:
            return
        m1 = b.mass * b.agility
        p0 = list(b.pos)
        p1, vp1 = self.integrate(list(b.pos), list(b.vel), b.last_g, m1, self.friction, b.time_factor, self.dt)
        if isinstance(s, Capsule):
            if self.iterative_collision:                    # never in the tick: Gradient runs only without it
                found, normal, toi = self._capsule_check(s, p0, p1, b.radius)
                if found:
                    self._capsule_react(s, b, p1, vp1, m1, normal, toi)
            else:
                self._capsule_non_iterative(s, b, p0, p1, m1, vp1)
            return
        if s.aabb.can_exclude(p0, p1, b.radius):
            self._hit("obox.excluded")
            return
        out: dict = {}
        if s.shape.collide_with_sphere(p0, float(b.radius), _sub(p1, p0), out):
            self._obox_react(s, b, p0, list(b.vel), m1, out["normal"], out["time"])

    def _capsule_non_iterative(self, c: Capsule, b, p0, p1, m1: float, vp1) -> None:
        """Capsule::HandleCollisionNonIteratively (Capsule.cpp:206), with its parameter along the end velocity, the
        end position and velocity passed on when that velocity is parallel to the axis, and the far cap's normal taken
        from p1 for the second root (as coded)."""
        rad = float(b.radius) + float(c.radius)
        d = list(vp1)
        ao = _sub(p0, c.a)
        abab = _dot(c.ab, c.ab)
        m = _ieee_div(_dot(c.ab, d), abab)
        n = _ieee_div(_dot(c.ab, ao), abab)
        q = _sub(d, d_scale(c.ab, m))
        r = _sub(ao, d_scale(c.ab, n))
        a = _dot(q, q)
        bq = 2.0 * _dot(q, r)
        cq = _dot(r, r) - rad * rad
        rad_sq = rad * rad
        s, toi, normal = -1.0, -1.0, [0.0, 0.0, 0.0]
        if a == 0.0:
            self._hit("capsule.parallel")
            sa = self.collide_two_spheres(p0, p1, c.a, c.a, rad)
            if sa != -1.0 and (toi == -1.0 or sa < toi):
                toi, normal = sa, _sub(p0, c.a)
            sb = self.collide_two_spheres(p0, p1, c.b, c.b, rad)
            if sb != -1.0 and (toi == -1.0 or sb < toi):
                toi, normal = sb, _sub(p0, c.b)
            self._capsule_react(c, b, p1, vp1, m1, normal, toi)
            return
        found, s1, s2 = _quadratic(a, bq, cq)
        if not found:
            return
        if 0.0 <= s1 <= 1.0:
            s = s1
        if 0.0 <= s2 <= 1.0 and s2 < s1:
            s = s2
        elif s == -1.0:
            t = _ieee_div(_dot(_sub(p0, c.a), c.ab), _lensq(c.ab))
            if 0.0 <= t <= 1.0:
                proj = d_add(c.a, d_scale(c.ab, t))
                dist = _sub(p0, proj)
                if _lensq(dist) < rad_sq:
                    self._hit("capsule.inside_cylinder")
                    self._capsule_react(c, b, p0, list(b.vel), m1, dist, 0.0)
                    return
        tk1, tk2 = (s1 * m) + n, (s2 * m) + n
        for tk, far in ((tk1, p0), (tk2, p1)):
            if tk < 0.0:
                sa = self.collide_two_spheres(p0, p1, c.a, c.a, rad)
                if sa != -1.0 and (toi == -1.0 or sa < toi):
                    toi, normal = sa, _sub(p0, c.a)
            elif tk > 1.0:
                sb = self.collide_two_spheres(p0, p1, c.b, c.b, rad)
                if sb != -1.0 and (toi == -1.0 or sb < toi):
                    toi, normal = sb, _sub(far, c.b)
            else:
                if s != -1.0 and (toi == -1.0 or s < toi):
                    toi = s
                    proj = d_scale(c.ab, _ieee_div(_dot(ao, c.ab), _lensq(c.ab)))
                    normal = _sub(ao, proj)
                    if _lensq(normal) < rad_sq:
                        self._hit("capsule.cylinder_contact")
                        toi = 0.0
        self._capsule_react(c, b, p0, list(b.vel), m1, normal, toi)

    def _static_response(self, b, s_id: int, last_c: list, toi: float) -> None:
        """The end of ReactToCollision: the stronger response at an equal time, else the latest; the collision kept
        (COL-711, COL-712; no s >= mLastCollision test)."""
        if toi == b.last_collision:
            self._hit("static.equal_time")
            if _lensq(last_c) > _lensq(b.last_c):
                b.last_c = last_c
        else:
            b.last_c = last_c
            b.last_collision = toi
        b.collisions.append(s_id)

    def _reflect(self, b, pos, vel, m1: float, normal, toi: float) -> list:
        """ReactToCollision's s > 0 branch (Capsule.cpp:409, OrientedBox.cpp:89): integrate to the impact, reverse the
        normal velocity, integrate the rest; the acceleration that gets there from the tick's start."""
        k, tf, g, dt = self.friction, b.time_factor, b.last_g, self.dt
        pos, vel = self.integrate(pos, vel, g, m1, k, tf, toi * dt)
        v1 = _dot(vel, normal)
        vel = _sub(vel, d_scale(normal, 2.0 * v1))
        pos, vel = self.integrate(pos, vel, g, m1, k, tf, (1.0 - toi) * dt)
        inv_m, inv_t = _ieee_div(1.0, m1), _ieee_div(1.0, tf - 1.0)
        return [-((((g[i] * -m1) + g[i] * (tf * m1)) - (b.vel[i] * tf) * k) + vel[i] * k) * inv_m * inv_t
                for i in range(3)]

    def _eject(self, b, m1: float, normal, normal_comp: float, dist: float) -> list:
        """ReactToCollision's s = 0 branch: out by `dist` along the normal within the tick, against mLastG's normal
        part."""
        dt = self.dt
        tmp = _ieee_div(1.0, m1 + dt * self.friction)
        c = _ieee_div(dist, dt * tmp * m1)
        inv_dt = _ieee_div(1.0, dt)
        return [(normal[i] * c - b.vel[i]) * inv_dt - normal[i] * normal_comp for i in range(3)]

    def _capsule_react(self, c: Capsule, b, pos, vel, m1: float, normal, toi: float) -> None:
        """Capsule::ReactToCollision (Capsule.cpp:395): ignored when mLastG points out of the capsule (G.n > 0)."""
        if toi == -1.0:
            return
        center_distance = _length(normal)
        normal = _normalized(normal)
        normal_comp = _dot(b.last_g, normal)
        if toi > 0.0:
            if normal_comp > 0.0:
                self._hit("capsule.outward")
                return
            self._hit("capsule.reflect")
            acc = self._reflect(b, list(pos), list(vel), m1, normal, toi)
        else:
            self._hit("capsule.eject")
            dist = float(c.radius) + float(b.radius) - center_distance + EJECT_MARGIN
            acc = self._eject(b, m1, normal, normal_comp, dist)
        from . import space as _space
        self._static_response(b, c.id, [x * _space.RESPONSE_SCALE for x in acc], toi)

    def _obox_react(self, ob: OrientedBox, b, pos, vel, m1: float, normal, toi: float) -> None:
        """OrientedBox::ReactToCollision (OrientedBox.cpp:76): the normal's length is the penetration depth."""
        if toi == -1.0:
            return
        depth = _length(normal)
        normal = _normalized(normal)
        if toi > 0.0:
            self._hit("obox.reflect")
            acc = self._reflect(b, list(pos), list(vel), m1, normal, toi)
        else:
            self._hit("obox.eject")
            acc = self._eject(b, m1, normal, _dot(b.last_g, normal), depth + EJECT_MARGIN)
        from . import space as _space
        self._static_response(b, ob.id, [x * _space.RESPONSE_SCALE for x in acc], toi)

    def _capsule_check(self, c: Capsule, p0, p1, radius: float) -> tuple:
        """Capsule::CheckCollision (Capsule.cpp:37): (found, normal toward the ball, time as a fraction)."""
        bounding = float(c.bounding_radius()) + float(radius)
        path = _sub(p1, p0)
        pathsq = _dot(path, path)
        separation, separation_end = _sub(c.pos, p0), _sub(c.pos, p1)
        to_closest = _dot(separation, path)
        if ((to_closest < 0 and bounding * bounding < _dot(separation, separation))
                or (to_closest > pathsq and bounding * bounding < _dot(separation_end, separation_end))):
            return False, None, None
        distsq = pathsq * _lensq(separation) - to_closest * to_closest
        if distsq > pathsq * bounding * bounding:
            return False, None, None
        sq_ab = _lensq(c.ab)
        if sq_ab < 1e-10:
            self._hit("capsule.sphere")
            s = self.collide_two_spheres(p0, p1, c.pos, c.pos, bounding)
            if s >= 0.0:
                normal = _sub(d_add(p0, d_scale(path, s)), c.pos)
                n_len = _length(normal)
                normal = [0.0, 0.0, 1.0] if n_len < 1e-10 else d_div(normal, n_len)
                return True, normal, s
            return False, None, None
        cr = float(radius) + float(c.radius)
        inv = 1.0 / sq_ab
        length_ab = math.sqrt(sq_ab)
        ao = _sub(p0, c.a)
        para_ao = _dot(ao, c.ab) / length_ab
        perp_path = _sub(path, d_scale(d_scale(c.ab, _dot(path, c.ab)), inv))
        perp_ao = _sub(ao, d_scale(d_scale(c.ab, _dot(ao, c.ab)), inv))
        cq = _dot(perp_ao, perp_ao) - cr * cr
        if cq <= 0.0:
            if 0.0 <= para_ao <= length_ab:
                self._hit("capsule.check_inside")
                return True, _normalized(perp_ao), 0.0
            center = c.a if para_ao < 0.0 else c.b
            s = self.collide_two_spheres(p0, p1, center, center, cr)
            if s >= 0.0:
                return True, _normalized(_sub(d_add(p0, d_scale(path, s)), center)), s
            return False, None, None
        a = _dot(perp_path, perp_path)
        if a < 1e-10:
            return False, None, None
        found, s1, s2 = _quadratic(a, 2.0 * _dot(perp_path, perp_ao), cq)
        if not found:
            return False, None, None
        if 0.0 <= s2 <= 1.0:
            pc = _sub(d_add(p0, d_scale(path, s2)), c.a)
            pc_height = _dot(pc, c.ab) / length_ab
            if 0.0 <= pc_height <= length_ab:
                self._hit("capsule.check_cylinder")
                return True, _normalized(_sub(pc, d_scale(d_scale(c.ab, _dot(pc, c.ab)), inv))), s2
            center = c.a if pc_height < 0 else c.b
            s = self.collide_two_spheres(p0, p1, center, center, cr)
            if s >= 0.0:
                self._hit("capsule.check_cap")
                return True, _normalized(_sub(d_add(p0, d_scale(path, s)), center)), s
        return False, None, None

    def _check_collision(self, s, p0, p1, radius: float) -> tuple:
        """StaticCollidable::CheckCollision: (found, normal, time as a fraction)."""
        if isinstance(s, Capsule):
            return self._capsule_check(s, p0, p1, radius)
        if s.aabb.can_exclude(p0, p1, radius):
            return False, None, None
        out: dict = {}
        found = s.shape.collide_with_sphere(p0, float(radius), _sub(p1, p0), out)
        if not found:
            return False, None, None
        normal = out["normal"]
        length = _length(normal)
        return True, (d_div(normal, length) if length > 1e-10 else [0.0, 0.0, 1.0]), out["time"]

    # ========================================================================= the iterative method, with statics

    def _properties(self, b):
        prop = self.collision_locations.get(b.id)
        if prop is None:
            prop = self.collision_locations[b.id] = FullCollisionBallProperties(self)
            prop.initialize(b, self.dt)
        return prop

    def _collision_responses(self) -> None:
        """CalculateIterativeCollisionResponses (Ballpark.cpp:332), the static collidables' candidates cleared too."""
        from . import space as _space
        for prop in self.collision_locations.values():
            prop.initialize(prop.ball, self.dt)
        for candidates in self.potential_balls.values():
            candidates.clear()
        for candidates in self.potential_collidables.values():
            candidates.clear()
        current = sorted(self.free_balls)
        iteration = 0
        while True:
            following: set = set()
            for bid in current:
                b = self.balls[bid]
                if b.is_moribund or b.in_dead_bubble or not b.is_massive:
                    continue
                if self._collision_response(b, iteration):
                    self._hit("iterative.found")
                    for n in self.potential_balls[bid]:
                        if n.is_free:
                            following.add(n.id)
                    following.add(bid)
            if not following:
                break
            iteration += 1
            if iteration >= _space.COLLISION_MAX_ITERATIONS:
                self._hit("iterative.cap")
                break
            current = sorted(following)

    def _collision_response(self, b, iteration: int) -> bool:
        """CalculateCollisionResponse (Ballpark.cpp:1988), with the static collidables."""
        prop = self._properties(b)
        if iteration == 0:
            uni, stat = self._collision_candidates(b)
            self.potential_balls.setdefault(b.id, []).extend(uni)
            self.potential_collidables.setdefault(b.id, []).extend(stat)
        if prop.t_start >= prop.t_end:
            prop.calculate_end()
            return False
        if prop.locations[0].collider != -1:
            return False
        balls = self.potential_balls.setdefault(b.id, [])
        collidables = self.potential_collidables.setdefault(b.id, [])
        if not balls and not collidables:
            return False
        item, neighbour = CollisionItem(), CollisionItem()
        self._next_collision_static(prop, item, balls, collidables)
        if item.time != prop.t_start:
            self._next_collision_free(prop, item, balls)
        if item.collider is not None:
            if item.mb2 == -2:
                self._impact_static(prop, item)
            else:
                neighbour = self._impact_free(prop, self.collision_locations[item.collider.id], item)
            if neighbour.impact is not None and neighbour.impact.collider != -1:
                self._add_impact(neighbour.impact, neighbour.collider)
            if item.impact is not None and item.impact.collider != -1:
                self._add_impact(item.impact, b)
                prop.calculate_end()
                return True
        prop.calculate_end()
        return False

    def _next_collision_static(self, prop, item, balls: list, collidables: list = ()) -> None:
        """GetNextCollisionStatic (Ballpark.cpp:2080; COL-715): the static collidables first (not for missiles and
        mushrooms), then, unless an impact at tStart was found, the fixed balls (space.py's loop)."""
        b = prop.ball
        if collidables and (b.mode == MISSILE or b.mode == MUSHROOM):
            self._hit("iterative.static_skip")
        if b.mode != MISSILE and b.mode != MUSHROOM and collidables:
            p0, _ = prop.position(prop.t_start)
            p1, _ = prop.position(prop.t_end)
            for s in collidables:
                found, normal, toi = self._check_collision(s, p0, p1, b.radius)
                if found:
                    if prop.ignore_collision(s.id, toi):
                        self._hit("iterative.ignored")
                        continue
                    t = prop.t_start + (prop.t_end - prop.t_start) * toi
                    if item.time < 0.0 or t < item.time:
                        self._hit("iterative.static")
                        item.normal, item.collider, item.time, item.mb1, item.impact = normal, s, t, -1, None
                if item.time == prop.t_start:
                    break
        super()._next_collision_static(prop, item, balls)

    def _ellipse_point(self, b, rot, normal) -> list:
        """ApproximateEllipsePointOfImpact (Ballpark.cpp:2323): -n r, or with dynamical orientation the normal turned
        toward a fake ellipse of ellipticity 0.3 about the heading."""
        if not self.dynamical_orientation:
            return d_scale(d_neg(normal), float(b.radius))
        fake = 0.3
        ship = [float(c) for c in transform_coord((0.0, 0.0, 1.0), m_rotation_q(rot))]
        cos_t = _dot(d_neg(normal), ship)
        poi = d_neg(normal)
        if abs(cos_t) < 1.0 and abs(cos_t) > 0.0:
            self._hit("ellipse.turned")
            sin_t = _sqrt(1 - cos_t * cos_t) if abs(cos_t) < 1.0 else 0.0
            sin_a = 2 * fake * cos_t * sin_t / _sqrt(
                (math.pow(cos_t, 2) + math.pow((1 - fake) * sin_t, 2))
                * (math.pow(cos_t, 2) + math.pow((1 + fake) * sin_t, 2)))
            ship = _cross(ship, normal)
            alpha = _asinf(_f(sin_a))
            turned = v3_mul(transform_coord(v3_of(poi), m_rotation_axis(v3_of(ship), alpha)), b.radius)
            poi = [float(c) for c in turned]
            poi = d_scale(poi, (1 - fake) * _sqrt(1. / (sin_t * sin_t + (1 - fake) * cos_t * cos_t)))
        return poi

    def _impact_static(self, prop, item) -> None:
        """CalculateCollisionImpactStatic (Ballpark.cpp:2370; COL-716), with the rotation terms of dynamical
        orientation (I = 0.5 m r^2)."""
        if item.impact is not None or item.mb2 != -2:
            return
        b = prop.ball
        do = self.dynamical_orientation
        item.impact = FullCollisionImpact(item.time)
        if prop.t_start == item.time:
            self._hit("iterative.static_contact")
            anormal = _dot(b.last_g, item.normal)
            max_acc = _ieee_div(b.max_vel * self.friction, b.mass * b.agility) - anormal
            b.last_c = [item.normal[i] * max_acc for i in range(3)]
        _, v = prop.position(item.time)
        rot, omega = QUAT_IDENTITY, (0.0, 0.0, 0.0)
        if do:
            rot, omega = prop.rotation(item.time)
        poi = self._ellipse_point(b, rot, item.normal)
        rel = list(v)
        vrot = [float(c) for c in omega]
        if do:
            vrot = _cross(vrot, poi)
            rel = d_add(rel, vrot)
        impulse = _dot([x * -2.0 for x in rel], item.normal)
        if impulse < 0.0:
            self._hit("iterative.no_impulse")
            return
        if do:
            rcrossn = _cross(poi, item.normal)
            vrot = _cross(rcrossn, poi)
            inertia = 0.5 * b.radius * b.radius
            impulse = _ieee_div(impulse, 1.0 + _ieee_div(_dot(vrot, item.normal), inertia))
            item.impact.dw = v3_mul(v3_of(rcrossn), _f(_ieee_div(impulse, inertia)))
        item.impact.dv = [item.normal[i] * impulse for i in range(3)]
        item.impact.collider = item.collider.id

    def _impact_free(self, prop, oprop, item):
        """CalculateCollisionImpactFree (Ballpark.cpp:2442; COL-717), with the rotation terms."""
        if item.impact is not None or item.mb2 == -2:
            return CollisionItem()
        do = self.dynamical_orientation
        other = CollisionItem()
        other.time, other.collider, other.mb1, other.mb2 = item.time, item.collider, item.mb2, item.mb1
        other.impact, item.impact = FullCollisionImpact(item.time), FullCollisionImpact(item.time)
        bp, bv = prop.position(item.time)
        op, ov = oprop.position(item.time)
        brot, bomega, orot, oomega = QUAT_IDENTITY, (0.0, 0.0, 0.0), QUAT_IDENTITY, (0.0, 0.0, 0.0)
        if do:
            brot, bomega = prop.rotation(item.time)
            orot, oomega = oprop.rotation(item.time)
        normal = _sub(bp, op)
        length = math.sqrt(_lensq(normal))
        if length > 0.0:
            c = 1.0 / length
            normal = [x * c for x in normal]
        else:
            self._hit("iterative.coincident")
            normal[2] = 1.0
        item.normal, other.normal = normal, [-x for x in normal]
        b, o = prop.ball, oprop.ball
        if 0.0 == item.time:
            self._hit("iterative.free_contact")
            anormal = _dot(b.last_g, item.normal)
            max_acc = _ieee_div(b.max_vel * self.friction, b.mass * b.agility) - anormal
            b.last_c = [item.normal[i] * max_acc for i in range(3)]
            anormal = _dot(o.last_g, other.normal)
            max_acc = _ieee_div(o.max_vel * self.friction, o.mass * o.agility) - anormal
            o.last_c = [other.normal[i] * max_acc for i in range(3)]
        poi1 = self._ellipse_point(b, brot, item.normal)
        poi2 = self._ellipse_point(o, orot, other.normal)
        vrot1, vrot2 = [float(c) for c in bomega], [float(c) for c in oomega]
        if do:
            vrot1, vrot2 = _cross(vrot1, poi1), _cross(vrot2, poi2)
            rel = d_add(_sub(ov, bv), _sub(vrot2, vrot1))
        else:
            rel = _sub(ov, bv)
        v_impulse = -2.0 * _dot(rel, other.normal)
        if v_impulse < 0.0:
            self._hit("iterative.no_impulse")
            return other
        rcn1, rcn2 = list(poi1), list(poi2)
        if do:
            rcn1 = _cross(rcn1, item.normal)
            vrot1 = _cross(rcn1, poi1)
            rcn2 = _cross(rcn2, other.normal)
            vrot2 = _cross(rcn2, poi2)
        m1, m2 = b.mass, o.mass
        r1, r2 = float(b.radius), float(o.radius)
        if do:
            t1, t2 = d_div(d_scale(vrot1, m2), r1 * r1), d_div(d_scale(vrot2, m1), r2 * r2)
            impulse = _ieee_div(m1 * m2 * v_impulse, m1 + m2 + _dot(d_scale(_sub(t1, t2), 2.0), item.normal))
        else:
            impulse = _ieee_div(m1 * m2 * v_impulse, m1 + m2)
        c1, c2 = _ieee_div(impulse, m1), _ieee_div(impulse, m2)
        item.impact.dv = [item.normal[i] * c1 for i in range(3)]
        other.impact.dv = [other.normal[i] * c2 for i in range(3)]
        if do:
            item.impact.dw = v3_mul(v3_of(rcn1), _f(_ieee_div(impulse, 0.5 * m1 * r1 * r1)))
            other.impact.dw = v3_mul(v3_of(rcn2), _f(_ieee_div(impulse, 0.5 * m2 * r2 * r2)))
        item.impact.collider, other.impact.collider = o.id, b.id
        return other

    def _add_impact(self, impact, b) -> None:
        """Ballpark::AddImpact (Ballpark.cpp:2577; COL-718): space.py's, with the angular change scaled by
        4 maxAngVel^2 / |dw|^2 below 1 (floats)."""
        from . import space as _space
        prop = self.collision_locations.get(b.id)
        if prop is None or prop.t_end <= impact.time:
            return
        fraction = _ieee_div(f32(f32(_space.VELOCITY_LIMIT * b.max_vel) * b.max_vel), _lensq(impact.dv))
        if fraction < 1:
            self._hit("iterative.limited")
            impact.dv = [x * fraction for x in impact.dv]
        if self.dynamical_orientation:
            frac = _fdiv(_f(_f(SPIN_LIMIT * b.max_ang_vel) * b.max_ang_vel), v3_dot(impact.dw, impact.dw))
            if frac < 1:
                self._hit("iterative.spin_limited")
                impact.dw = v3_mul(impact.dw, frac)
        for t, cid in prop.add_impact(impact):
            if cid in self.free_balls:
                self._hit("iterative.invalidated")
                self._remove_impact_from_free(t, b.id, self.balls[cid])

    def _rotation_velocity(self, b, fraction: float, rot, omega) -> tuple:
        """CalculateBallRotationVelocity (Ballpark.cpp:2685): from the collision path if the ball has one (at the time
        `fraction`, which Evolve passes as 1.0 whatever the tick), else ApplyTorque."""
        if fraction == 0.0:
            return rot, omega
        prop = self.collision_locations.get(b.id)
        if prop is None:
            return self._apply_torque(rot, omega, b.torque, b.rot_roll, self._tau(b), _f(fraction))
        return prop.rotation(fraction)

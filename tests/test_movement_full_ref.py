# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against published test vectors for the full ballpark: the reference
(exact_ballpark.full) against the expectations of Destiny's own tests.

The expected values are copied from carbonengine/destiny at 114e89fa (MIT licence, (c) CCP Games):
tests/TestAABB.cpp, TestBoxShape.cpp, TestCollision.cpp, TestPlane.cpp, TestTriangle.cpp and TestVector3d.cpp (the
geometry of the static collidables, exact as their ASSERT_EQ); python/destiny/test/ballpark/evolve/
test_simple_collision.py (TestSimpleCapsuleCollision, TestSimpleBoxCollision: whole trajectories, which Destiny checks
to four decimals and these check bit for bit), test_iterative_collision.py (TestIterativeCapsuleCollision,
TestIterativeBoxCollision and the miniball case: bounds, kept with their tolerances), test_missile.py, test_minis.py and
test_follow.py (TestFormation); ballpark/test_movement_controls.py (MissileFollow, LaunchMissile, Stop of a missile and
of a formation follower, FormationFollow, Orbit of a cloaked ball), test_boxes_and_bubbles.py
(GetStaticCollidableBox, the active boxes of capsules and oriented boxes), test_getters_and_setters.py (the
orientation set by SetBallVelocity, SetBoid, SetBallTroll, SetBallRigid, GetAccuracy, SetBallFormation),
test_visibility.py (cloaked occluders, TestCloaking), test_lifecycle_management.py (TestAddMushroom, TestCapsule,
TestIdGeneration) and python/destiny/test/test_ball.py (minis, GetRotatedVector, formation slots, default values), with
the balls of helpers.py (add_ball_to_park: the POPO defaults, not free, at 1e10 m on each axis; create_space_ball: then
SetBallFree, max velocity 10, agility 0.9, mass 13,000,000, radius 2, massive, speed fraction 0.95; ids from 1 here,
from a process-wide counter in Destiny's suite). Destiny's tests run on `destiny.Ballpark()`, a client's view. A Python
attribute write is Blue's: the member is written, then Ball::OnModified calls the park's setter (Ball.cpp:1507), for
maxAngularVelocity (SetMaxAngularSpeed) and RotationalAgility (SetBallAngularAgility) too; isCloaked has no setter.
Blue shows the char mFormationID unsigned (255 for -1).

Two of Destiny's expectations need DirectXMath (carbonengine/math's RotationQuaternion(yaw, pitch, roll),
Quaternion.cpp:60-82): TestFormation turns the slot offsets by the leader's orientation once it has one. Until 23
October 2026 the reference computed only the first tick (the leader still at zero angles) and refused the second;
with DirectXMath's scalar path transcribed it computes all ten, within Destiny's four decimals of its expectations
(those differ from this platform's code by up to 9e-7 m, from the first tick of the moving case on: Destiny's values
were produced elsewhere), and the first tick bit for bit. In TestIterativeCapsuleCollision's zero-length capsule case
Destiny's test measures from miniCapsules[0], which is setUp's capsule (AddBall of id 0 again returns the same ball):
kept as written.
"""

from __future__ import annotations

import math

import pytest

from exact_ballpark.full import (AABB, CORNER_A, CORNER_B, CORNER_C, CORNER_D, CORNER_E, CORNER_F, CORNER_G, CORNER_H,
                               BoxShape, FullBallpark, RectangleShape, Triangle, _point_on_ball,
                               _segment_plane, _segment_sphere, _sphere_triangle)
from exact_ballpark.modes import BOID, FORMATION, MISSILE, MUSHROOM, ORBIT, RIGID, TROLL, WARP, _cross
from exact_ballpark.ref import STOP, f32
from exact_ballpark.space import _dot, _sub

TEN_BILLION = 10000000000.0
AU = .1495978707e12
MIN_BOX_SIZE = 480.0
DSTNORMALCLOAK = 1
MAX_FORMATION_SLOTS_FOR_BALLS = 16
FORMATIONS = (                                              # helpers.FORMATIONS: Diamond, Arrow, FormationOfOne
    ((100.0, 0.0, 0.0), (0.0, 100.0, 0.0), (-100.0, 0.0, 0.0), (0.0, -100.0, 0.0)),
    ((100.0, 0.0, -50.0), (50.0, 0.0, 0.0), (-100.0, 0.0, -50.0), (-50.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0),),
)


# ---- helpers.py ----

def add(park: FullBallpark, object_id: int = 0, **kw):
    """helpers.add_ball_to_park with the POPO defaults."""
    b = {"mass": 0.0, "radius": 0.0, "maxVel": 0.0, "isFree": False, "isGlobal": False, "isMassive": False,
         "isInteractive": False, "isSpaceJunk": False, "x": TEN_BILLION, "y": TEN_BILLION, "z": TEN_BILLION,
         "vx": 0.0, "vy": 0.0, "vz": 0.0, "agility": 0.0, "speedFraction": 0.0}
    b.update(kw)
    return park.add_ball(object_id, b["mass"], b["radius"], b["maxVel"], b["isFree"], b["isGlobal"], b["isMassive"],
                         b["isInteractive"], b["isSpaceJunk"], b["x"], b["y"], b["z"], b["vx"], b["vy"], b["vz"],
                         b["agility"], b["speedFraction"])


def add_balls(park: FullBallpark, n: int) -> list:
    """BallparkTestCase.add_balls: ids 1 to n."""
    return [add(park, i + 1) for i in range(n)]


def put(park: FullBallpark, ball, name: str, value) -> None:
    """A Python attribute write: the member, then OnModified's setter (Ball.cpp:1507)."""
    if name in ("x", "y", "z"):
        pos = list(ball.pos)
        pos["xyz".index(name)] = float(value)
        ball.pos = pos
        park.set_position(ball.id, *pos)
    elif name in ("vx", "vy", "vz"):
        vel = list(ball.vel)
        vel["xyz".index(name[1])] = float(value)
        ball.vel = vel
        park.set_velocity(ball.id, *vel)
    elif name == "radius":
        ball.radius = f32(value)
        park.set_radius(ball.id, ball.radius)
    elif name == "maxVelocity":
        ball.max_vel = f32(value)
        park.set_max_velocity(ball.id, ball.max_vel)
    elif name == "Agility":
        ball.agility = f32(value)
        park.set_agility(ball.id, ball.agility)
    elif name == "mass":
        ball.mass = float(value)
        park.set_mass(ball.id, ball.mass)
    elif name == "speedFraction":
        ball.speed_fraction = f32(value)
        park.set_speed_fraction(ball.id, ball.speed_fraction)
    elif name == "isFree":
        ball.is_free = bool(value)
        park.set_free(ball.id, bool(value))
    elif name == "isMassive":
        ball.is_massive = bool(value)
        park.set_massive(ball.id, bool(value))
    elif name == "isCloaked":
        ball.is_cloaked = int(value)                        # no setter runs for isCloaked
    elif name == "maxAngularVelocity":
        ball.max_ang_vel = f32(value)
        park.set_max_angular_speed(ball.id, ball.max_ang_vel)
    elif name == "RotationalAgility":
        ball.rot_agility = f32(value)
        park.set_ball_angular_agility(ball.id, ball.rot_agility)
    else:
        raise ValueError(name)


def space_ball(park: FullBallpark, bid: int, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, mass=None,
               max_velocity=None):
    """helpers.create_space_ball."""
    ball = add(park, bid, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    park.set_free(bid, True)
    put(park, ball, "maxVelocity", max_velocity if max_velocity is not None else 10.0)
    put(park, ball, "Agility", 0.9)
    put(park, ball, "mass", mass if mass is not None else 13000000.0)
    put(park, ball, "radius", 2.0)
    put(park, ball, "isMassive", True)
    put(park, ball, "speedFraction", 0.95)
    return ball


def almost(a: float, b: float, places: int = 7) -> bool:
    """unittest's assertAlmostEqual: the difference rounded to `places` decimals is zero."""
    return round(abs(a - b), places) == 0


def distance(p, q) -> float:
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)


def coordinates(park: FullBallpark, ball, steps: int) -> list:
    """BallparkTestCase.evolve_ball_and_get_coordinates."""
    out = []
    for _ in range(steps):
        park.evolve()
        out.append(tuple(ball.pos))
    return out


def formation_view(ball) -> int:
    """Blue's formationID: the char read unsigned."""
    return ball.formation_id & 0xFF


# ================================================================ tests/*.cpp: the geometry, exact (ASSERT_EQ)

def test_aabb():
    b = AABB()
    assert (b.low, b.high) == ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    b = AABB((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert (b.low, b.high) == ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    b = AABB()
    b.update((1.0, 2.0, 3.0))
    assert b.high == [1.0, 2.0, 3.0]
    b = AABB()
    b.update((-1.0, -2.0, -3.0))
    assert b.low == [-1.0, -2.0, -3.0]
    b = AABB((0.0, 0.0, 0.0), (5.0, 5.0, 5.0))
    b.update((1.0, 2.0, 3.0))
    assert b.high == [5.0, 5.0, 5.0]
    b = AABB((-5.0, -5.0, -5.0), (0.0, 0.0, 0.0))
    b.update((-1.0, -2.0, -3.0))
    assert b.low == [-5.0, -5.0, -5.0]
    unit = AABB((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert not unit.can_exclude((-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), 1.0)
    for p0, p1 in (((-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)), ((2.0, -1.0, 0.0), (2.0, 1.0, 0.0)),
                   ((-1.0, 2.0, 0.0), (1.0, 2.0, 0.0)), ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0)),
                   ((-1.0, 0.0, 2.0), (1.0, 0.0, 2.0)), ((-1.0, 0.0, -1.0), (1.0, 0.0, -1.0))):
        assert unit.can_exclude(p0, p1, 0.5)
    assert not unit.can_exclude((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 0.5)
    assert AABB().longest_width() == 0.0
    assert AABB((-0.5, -0.5, -0.5), (1.0, 0.5, 0.5)).longest_width() == 1.5
    assert AABB((-0.5, -0.5, -0.5), (0.5, 1.0, 0.5)).longest_width() == 1.5
    assert AABB((-0.5, -0.5, -0.5), (0.5, 0.5, 1.0)).longest_width() == 1.5


def test_plane_and_vectors():
    point, normal = (1.0, 2.0, 3.0), (0.0, 1.0, 0.0)
    assert _dot(point, normal) == 2.0                       # Planed(point, normal).d
    assert _sub((1.0, 2.0, 3.0), (0.0, 1.0, 2.0)) == [1.0, 1.0, 1.0]
    assert list(_cross([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])) == [0.0, 0.0, 0.0]
    assert list(_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])) == [0.0, 0.0, 1.0]


def test_box_shape():
    box = BoxShape((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert box.local == ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    want = {CORNER_A: [0.0, 0.0, 0.0], CORNER_B: [1.0, 0.0, 0.0], CORNER_C: [0.0, 1.0, 0.0],
            CORNER_D: [0.0, 0.0, 1.0], CORNER_E: [1.0, 1.0, 0.0], CORNER_F: [0.0, 1.0, 1.0],
            CORNER_G: [1.0, 0.0, 1.0], CORNER_H: [1.0, 1.0, 1.0]}
    assert {k: box.corners[k] for k in want} == want
    assert box.closest_corner((2.0, 2.0, 0.0)) == CORNER_E
    assert box.closest_corner((2.0, 2.0, 2.0)) == CORNER_H
    assert box.center() == [0.5, 0.5, 0.5]
    assert box.bounding_radius() == math.sqrt(3.0) / 2.0
    side = RectangleShape((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0))
    assert (side.a, side.b, side.c, side.d, side.normal) == ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0],
                                                             [0.0, 10.0, 0.0], [0.0, 0.0, 1.0])
    for pos, point, normal in (((5.0, 5.0, 6.0), [5.0, 5.0, 0.0], [0.0, 0.0, 1.0]),
                               ((-1.0, 5.0, 5.0), [0.0, 5.0, 0.0], [-1.0, 0.0, 0.0]),
                               ((11.0, 5.0, 5.0), [10.0, 5.0, 0.0], [1.0, 0.0, 0.0]),
                               ((5.0, 11.0, 5.0), [5.0, 10.0, 0.0], [0.0, 1.0, 0.0]),
                               ((5.0, -1.0, 5.0), [5.0, 0.0, 0.0], [0.0, -1.0, 0.0])):
        out = {}
        assert side.collide_with_sphere(pos, 1.0, (0.0, 0.0, -10.0), out)
        assert (out["time"], out["point"], out["normal"]) == (0.5, point, normal), pos
    cube = BoxShape((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
    for pos, vel, point, normal in (((5.0, 5.0, 16.0), (0.0, 0.0, -10.0), [5.0, 5.0, 10.0], [0.0, 0.0, 1.0]),
                                    ((5.0, 5.0, -6.0), (0.0, 0.0, 10.0), [5.0, 5.0, 0.0], [0.0, 0.0, -1.0]),
                                    ((-6.0, 5.0, 5.0), (10.0, 0.0, 0.0), [0.0, 5.0, 5.0], [-1.0, 0.0, 0.0]),
                                    ((16.0, 5.0, 5.0), (-10.0, 0.0, 0.0), [10.0, 5.0, 5.0], [1.0, 0.0, 0.0]),
                                    ((5.0, 16.0, 5.0), (0.0, -10.0, 0.0), [5.0, 10.0, 5.0], [0.0, 1.0, 0.0]),
                                    ((5.0, -6.0, 5.0), (0.0, 10.0, 0.0), [5.0, 0.0, 5.0], [0.0, -1.0, 0.0])):
        out = {}
        assert cube.collide_with_sphere(pos, 1.0, vel, out)
        assert (out["time"], out["point"], out["normal"]) == (0.5, point, normal), pos
    for p, want in (((5.0, 5.0, 5.0), [5.0, 5.0, 5.0]), ((11.0, 11.0, 11.0), [10.0, 10.0, 10.0]),
                    ((-1.0, 5.0, 5.0), [0.0, 5.0, 5.0]), ((11.0, 5.0, 5.0), [10.0, 5.0, 5.0]),
                    ((5.0, 11.0, 5.0), [5.0, 10.0, 5.0]), ((5.0, -1.0, 5.0), [5.0, 0.0, 5.0]),
                    ((5.0, 5.0, -1.0), [5.0, 5.0, 0.0]), ((5.0, 5.0, 11.0), [5.0, 5.0, 10.0])):
        assert cube.closest_point(p) == want, p


def test_collision_primitives():
    assert _point_on_ball((0.0, 3.0, 0.0), 1.0, (0.0, 1.0, 0.0)) == [0.0, 2.0, 0.0]
    n = (0.0, 1.0, 0.0)
    found, point, t = _segment_plane((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), _dot((0.0, 0.0, 0.0), n), n)
    assert found and point == [0.0, 0.0, 0.0] and t == 0.5
    assert not _segment_plane((0.0, 3.0, 0.0), (0.0, 2.0, 0.0), 0.0, n)[0]
    assert not _segment_plane((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), 0.0, (0.0, 0.0, 1.0))[0]
    found, point, t = _segment_sphere((10.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
    assert found and point == [1.0, 0.0, 0.0] and t == 0.9
    found, point, t = _segment_sphere((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
    assert found and point == [0.0, 0.0, 0.0] and t == 0.0
    assert not _segment_sphere((0.0, 0.0, 7.0), (0.0, 0.0, 8.0), (0.0, 0.0, 0.0), 1.0)[0]
    tri = Triangle((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 2.0, 0.0))
    out = {}
    assert _sphere_triangle((1.0, 1.0, 2.0), 1.0, (0.0, 0.0, -2.0), tri, out)
    assert (out["point"], out["time"], out["normal"]) == ([1.0, 1.0, 0.0], 0.5, [0.0, 0.0, 1.0])
    out = {}
    assert _sphere_triangle((0.0, -1.0, 2.0), 1.0, (0.0, 0.0, -2.0), tri, out)
    assert (out["point"], out["time"], out["normal"]) == ([0.0, 0.0, 0.0], 1.0, [0.0, -1.0, 0.0])
    assert not _sphere_triangle((1.0, 1.0, 2.0), 1.0, (0.0, 0.0, -2.0),
                                Triangle((1.0, 2.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)), {})
    assert not _sphere_triangle((1.0, 1.0, 2.0), 1.0, (0.0, 100.0, 0.0), tri, {})
    assert not _sphere_triangle((100.0, 1.0, 1.0), 1.0, (0.0, 0.0, -2.0), tri, {})


def test_triangle():
    t = Triangle((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    assert (t.a, t.b, t.c) == ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
    t = Triangle((0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.0, 1.0))
    assert t.normal() == [0.0, 1.0, 0.0]
    assert t.contains((1.0, 0.0, 1.0))
    assert not t.contains((1.5, 0.0, 2.0)) and not t.contains((1.0, 0.0, 0.5)) and not t.contains((2.5, 2.0, 0.0))
    t = Triangle((0.0, 1.0, 0.0), (1.0, 2.0, 0.0), (2.0, 1.0, 0.0))
    assert t.closest_point((1.0, 1.5, 1.0)) == [1.0, 1.5, 0.0]
    assert t.closest_point((1.0, 1.5, -1.0)) == [1.0, 1.5, 0.0]
    assert Triangle((10.0, 10.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 0.0)).closest_point((-1.0, 5.0, 0.0)) == \
        [0.0, 5.0, 0.0]
    right = Triangle((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0))
    assert right.closest_point((5.0, -1.0, 0.0)) == [5.0, 0.0, 0.0]
    assert right.closest_point((11.0, 5.0, 0.0)) == [10.0, 5.0, 0.0]
    assert right.closest_point((0.0, 10.0, 0.0)) == [5.0, 5.0, 0.0]
    assert Triangle((0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (10.0, 10.0, 0.0)).closest_point((5.0, 11.0, 0.0)) == \
        [5.0, 10.0, 0.0]


# ================================================= test_simple_collision.py: the gradient method, bit for bit

CAPSULE_LEFT = [
    (90.39465942803726, 100.0, 0.0), (91.53522284165427, 100.0, 0.0), (93.36058623726207, 100.0, 0.0),
    (95.81465121430743, 100.0, 0.0), (96.19428373412633, 100.0, 0.0), (94.74389513390018, 100.0, 0.0),
    (94.19055573837095, 100.0, 0.0), (94.46077981896904, 100.0, 0.0), (95.48710155358613, 100.0, 0.0),
    (95.26270051345809, 100.0, 0.0), (93.94466009572491, 100.0, 0.0), (93.5128269995852, 100.0, 0.0),
    (93.89460365697916, 100.0, 0.0), (95.02333964873613, 100.0, 0.0), (96.8378445177328, 100.0, 0.0),
    (97.3081193532411, 100.0, 0.0), (96.15310072010158, 100.0, 0.0), (95.43685904668816, 100.0, 0.0),
    (95.55752570552956, 100.0, 0.0), (96.44654167197064, 100.0, 0.0)]
CAPSULE_RIGHT = [
    (209.60534057196273, 100.0, 0.0), (208.46477715834573, 100.0, 0.0), (206.63941376273795, 100.0, 0.0),
    (204.18534878569255, 100.0, 0.0), (203.80571626587366, 100.0, 0.0), (205.25610486609978, 100.0, 0.0),
    (205.80944426162898, 100.0, 0.0), (205.53922018103088, 100.0, 0.0), (204.51289844641377, 100.0, 0.0),
    (204.73729948654176, 100.0, 0.0), (206.05533990427483, 100.0, 0.0), (206.48717300041446, 100.0, 0.0),
    (206.10539634302043, 100.0, 0.0), (204.9766603512634, 100.0, 0.0), (203.16215548226666, 100.0, 0.0),
    (202.45322116730054, 100.0, 0.0), (203.45863658621496, 100.0, 0.0), (204.33052522945047, 100.0, 0.0),
    (204.35275503475415, 100.0, 0.0), (203.59492954049838, 100.0, 0.0)]
CAPSULE_TOP = [
    (150.0, 109.60534057196273, 0.0), (150.0, 108.46477715834571, 0.0), (150.0, 106.63941376273792, 0.0),
    (150.0, 104.18534878569254, 0.0), (150.0, 103.56551380937034, 0.0), (150.0, 104.56192118699425, 0.0),
    (150.0, 104.69846922723862, 0.0), (150.0, 104.04559708722422, 0.0), (150.0, 103.88074751531305, 0.0),
    (150.0, 104.12987940232064, 0.0), (150.0, 103.58036830840118, 0.0), (150.0, 103.26349609203893, 0.0),
    (150.0, 103.2443564826648, 0.0), (150.0, 102.94843147069491, 0.0), (150.0, 103.20462217476765, 0.0),
    (150.0, 103.56098365081996, 0.0), (150.0, 103.60239882749173, 0.0), (150.0, 103.8500067373003, 0.0),
    (150.0, 103.79396656673549, 0.0), (150.0, 103.76288953683304, 0.0)]
CAPSULE_BOTTOM = [
    (150.0, 90.39465942803726, 0.0), (150.0, 91.53522284165427, 0.0), (150.0, 93.36058623726207, 0.0),
    (150.0, 95.81465121430743, 0.0), (150.0, 96.43448619062963, 0.0), (150.0, 95.43807881300572, 0.0),
    (150.0, 95.30153077276132, 0.0), (150.0, 95.95440291277573, 0.0), (150.0, 96.11925248468687, 0.0),
    (150.0, 95.87012059767925, 0.0), (150.0, 96.41963169159868, 0.0), (150.0, 96.73650390796085, 0.0),
    (150.0, 96.75564351733463, 0.0), (150.0, 97.05156852930398, 0.0), (150.0, 96.79537782523148, 0.0),
    (150.0, 96.43901634918006, 0.0), (150.0, 96.3976011725089, 0.0), (150.0, 96.14999326270136, 0.0),
    (150.0, 96.20603343326782, 0.0), (150.0, 96.23711046317295, 0.0)]
BOX_LEFT_X = [-9.605340571962731, -8.46477715834572, -6.639413762737928, -4.185348785692558, -3.809252356728292,
              -5.266324147762671, -5.825799251149228, -5.561208244929905, -4.540058126643755, -2.824325761400378,
              -2.3169360278041253, -2.86705381202771, -2.5938720325746143, -2.46888358211965, -2.4545569738496824,
              -2.3691238030650834, -2.1985812857514397, -1.847589525055808, -2.2264577298771377,
              -2.6823664830129617]
BOX_RIGHT_X = [109.60534057196273, 108.46477715834571, 106.63941376273792, 104.18534878569254, 103.8092523567283,
               105.26632414776267, 105.82579925114923, 105.5612082449299, 104.54005812664376, 102.82432576140037,
               102.31693602780412, 102.8670538120277, 102.59387203257461, 102.46888358211964, 102.45455697384968,
               102.36912380306507, 102.19858128575139, 101.84758952505571, 102.22645772987704, 102.68236648301296]
BOX_BOTTOM_Y = [-49.605340571962735, -48.464777158345726, -46.63941376273794, -44.18534878569257,
                -41.154085080701805, -37.59290641863395, -33.54522311243153, -29.050889328442953, -24.1464944038018,
                -18.865630299262452, -13.239137142461129, -7.295328656421343, -6.449286068036103,
                -10.132042794594934, -12.73487606904647, -14.346252567758015, -15.047391830448245,
                -14.912859941240473, -14.0111145757229, -12.405005398083203]


def _capsule_park(iterative: bool = False, radius: float = 1.0) -> FullBallpark:
    park = FullBallpark(iterative=iterative)
    parent = add(park, 0, x=0.0, y=0.0, z=0.0, radius=500.0)
    park.add_mini_capsule(parent.id, 100.0, 100.0, 0.0, 200.0, 100.0, 0.0, radius)
    return park


@pytest.mark.parametrize("start, goal, want", [
    ((90.0, 100.0, 0.0), (300.0, 100.0, 0.0), CAPSULE_LEFT),
    ((210.0, 100.0, 0.0), (0.0, 100.0, 0.0), CAPSULE_RIGHT),
    ((150.0, 110.0, 0.0), (150.0, 0.0, 0.0), CAPSULE_TOP),
    ((150.0, 90.0, 0.0), (150.0, 300.0, 0.0), CAPSULE_BOTTOM)], ids=["left", "right", "top", "bottom"])
def test_simple_capsule_collision(start, goal, want):
    park = _capsule_park()
    ball = space_ball(park, 1, *start)
    park.goto_point(ball.id, *goal)
    assert coordinates(park, ball, 20) == want


def _box_park(iterative: bool = False, radius: float = 500.0) -> FullBallpark:
    park = FullBallpark(iterative=iterative)
    parent = add(park, 0, x=0.0, y=0.0, z=0.0, radius=radius)
    park.add_mini_box(parent.id, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)
    return park


@pytest.mark.parametrize("axis, start, want", [
    (0, -10.0, BOX_LEFT_X), (0, 110.0, BOX_RIGHT_X), (1, 110.0, BOX_RIGHT_X), (1, -50.0, BOX_BOTTOM_Y),
    (2, 110.0, BOX_RIGHT_X), (2, -10.0, BOX_LEFT_X)], ids=["left", "right", "top", "bottom", "front", "back"])
def test_simple_box_collision(axis, start, want):
    park = _box_park()
    pos = [50.0, 50.0, 50.0]
    pos[axis] = start
    ball = space_ball(park, 1, *pos)
    park.goto_point(ball.id, 50.0, 50.0, 50.0)
    got = coordinates(park, ball, 20)
    expected = []
    for v in want:
        p = [50.0, 50.0, 50.0]
        p[axis] = v
        expected.append(tuple(p))
    assert got == expected


# ============================================ test_iterative_collision.py: static collidables, as bounds

@pytest.mark.parametrize("start, goal, rotation, end", [
    ((80.0, 100.0, 0.0), (300.0, 100.0, 0.0), (0.0, 0.5 * math.sqrt(2), 0.0, 0.5 * math.sqrt(2)), "a"),
    ((220.0, 100.0, 0.0), (0.0, 100.0, 0.0), (0.0, math.sin(-math.radians(45)), 0.0, math.cos(-math.radians(45))),
     "b")], ids=["left", "right"])
def test_iterative_capsule_collision_along_the_axis(start, goal, rotation, end):
    park = _capsule_park(iterative=True, radius=10.0)
    cap = park.balls[0].mini_capsules[0]
    ball = space_ball(park, 1, *start)
    if end == "a":
        park.goto_point(ball.id, *goal)
        park.set_ball_rotation(ball.id, *rotation)
    else:
        park.set_ball_rotation(ball.id, *rotation)
        park.goto_point(ball.id, *goal)
    for c in coordinates(park, ball, 20):
        assert 1.11 * distance(c, cap.a if end == "a" else cap.b) > ball.radius + cap.radius
        assert almost(c[1], 100.0, 4) and almost(c[2], 0.0, 2)


def test_iterative_capsule_collision_across_the_axis():
    for start, rotation, goto in (((150.0, 120.0, 0.0), (math.sin(math.radians(45)), 0.0, 0.0,
                                                          math.cos(math.radians(45))), ("point", 150.0, 0.0, 0.0)),
                                  ((150.0, 80.0, 0.0), (-1.0, 0.0, 0.0, 1.0), ("direction", 0.0, 1.0, 0.0))):
        park = _capsule_park(iterative=True, radius=10.0)
        cap = park.balls[0].mini_capsules[0]
        ball = space_ball(park, 1, *start)
        park.set_ball_rotation(ball.id, *rotation)
        if goto[0] == "point":
            park.goto_point(ball.id, *goto[1:])
        else:
            park.goto_direction(ball.id, *goto[1:])
        for c in coordinates(park, ball, 20):
            assert 1.11 * distance(c, (150.0, cap.b[1], cap.b[2])) > ball.radius + cap.radius
            assert almost(c[0], 150.0, 2 if goto[0] == "point" else 4) and almost(c[2], 0.0, 2)


def test_iterative_collision_with_zero_length_capsule():
    park = _capsule_park(iterative=True, radius=10.0)
    parent = add(park, 0, x=0.0, y=0.0, z=0.0, radius=500.0)          # AddBall of id 0 again: the same ball
    park.add_mini_capsule(parent.id, 100.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0)
    capsule = parent.mini_capsules[0]                                   # setUp's capsule, as Destiny's test reads
    ball = space_ball(park, 1, 100.0, 0.0, -10.0)
    park.goto_direction(ball.id, 0.0, 0.0, 1.0)
    for c in coordinates(park, ball, 20):
        assert 1.11 * distance(c, capsule.b) > ball.radius + capsule.radius
        assert almost(c[0], 100.0, 4) and almost(c[1], 0.0, 2)
    zero = parent.mini_capsules[1]                                      # and the zero-length one is never entered
    for c in [tuple(ball.pos)]:
        assert distance(c, zero.a) > ball.radius + zero.radius


def test_iterative_capsule_collision_at_angle():
    park = _capsule_park(iterative=True, radius=10.0)
    cap = park.balls[0].mini_capsules[0]
    ball = space_ball(park, 1, 170.0, 80.0, 0.0, vx=-6.7, vy=6.7)
    put(park, ball, "Agility", 0.3)
    park.goto_direction(ball.id, -1.0, 1.0, 0.0)
    park.set_velocity(ball.id, -6.7, 6.7, 0.0)
    for _ in range(10):
        park.evolve()
        assert ball.pos[1] + 0.9 * (ball.radius + cap.radius) < cap.b[1]
        assert almost(ball.pos[2], 0.0, 1)


def test_iterative_ball_starting_within_the_capsule():
    for start, axis in (((160.0, 95.0, 0.0), 1), ((95.0, 100.0, 0.0), 0)):
        park = _capsule_park(iterative=True, radius=10.0)
        cap = park.balls[0].mini_capsules[0]
        ball = space_ball(park, 1, *start)
        for _ in range(20):
            park.evolve()
            assert ball.vel[axis] < 0.0
            assert all(almost(ball.vel[i], 0.0, 2) for i in range(3) if i != axis)
        if axis == 1:
            assert ball.pos[1] + 0.9 * (ball.radius + cap.radius) < cap.b[1]
        else:
            assert ball.pos[0] + 0.9 * (ball.radius + cap.radius) < cap.a[0]


@pytest.mark.parametrize("axis, start, rotation, places, steps", [
    (0, -10.0, (0.0, 0.5 * math.sqrt(2), 0.0, 0.5 * math.sqrt(2)), 4, 20),
    (0, 110.0, (0.0, -0.5 * math.sqrt(2), 0.0, 0.5 * math.sqrt(2)), 4, 20),
    (1, 110.0, (math.sin(math.radians(45)), 0.0, 0.0, math.cos(math.radians(45))), 2, 20),
    (1, -50.0, (math.sin(math.radians(-45)), 0.0, 0.0, math.cos(math.radians(-45))), 3, 20),
    (2, 110.0, (1.0, 0.0, 0.0, 0.0), 4, 100),
    (2, -10.0, None, 4, 20)], ids=["left", "right", "top", "bottom", "front", "back"])
def test_iterative_box_collision(axis, start, rotation, places, steps):
    park = _box_park(iterative=True, radius=0.0)
    pos = [50.0, 50.0, 50.0]
    pos[axis] = start
    ball = space_ball(park, 1, *pos)
    if rotation is not None:
        park.set_ball_rotation(ball.id, *rotation)
    park.goto_point(ball.id, 50.0, 50.0, 50.0)
    face = [50.0, 50.0, 50.0]
    face[axis] = 100.0 if start > 50.0 else 0.0
    for c in coordinates(park, ball, steps):
        assert 1.11 * distance(c, face) > ball.radius
        assert all(almost(c[i], 50.0, places) for i in range(3) if i != axis)


def test_iterative_box_collision_at_angle():
    park = _box_park(iterative=True, radius=0.0)
    ball = space_ball(park, 1)
    xp, yp, zp = [0.0, 50, -50], [50, -50, 0.0], [-50, 0.0, 50]
    xv, yv, zv = [0.1, 0.0, 90], [0.0, 90, 0.1], [90, 0.1, 0.0]
    for i in range(3):
        for name, value in (("x", xp[i]), ("y", yp[i]), ("z", zp[i]), ("vx", xv[i]), ("vy", yv[i]), ("vz", zv[i]),
                            ("maxVelocity", 100), ("Agility", 0.5)):
            put(park, ball, name, value)
        park.goto_direction(ball.id, xv[i], yv[i], zv[i])
        for _ in range(40):
            park.evolve()
            if (i == 0 and ball.pos[0] > 100) or (i == 1 and ball.pos[2] > 100) or (i == 2 and ball.pos[1] > 100):
                break
            if i == 0:
                assert ball.pos[2] + 0.9 * ball.radius < 0.0 and almost(ball.vel[1], 0.0, 2)
            if i == 1:
                assert ball.pos[1] + 0.9 * ball.radius < 0.0 and almost(ball.vel[0], 0.0, 2)
            if i == 2:
                assert ball.pos[0] + 0.9 * ball.radius < 0.0 and almost(ball.vel[2], 0.0, 2)


def test_iterative_starting_within_box():
    park = _box_park(iterative=True, radius=0.0)
    ball = space_ball(park, 1, 20.0, 60.0, 30.0, vx=-10.0)
    put(park, ball, "Agility", 0.01)
    for _ in range(100):
        park.evolve()
        assert ball.vel[0] < 0.0 and almost(ball.vel[1], 0.0) and almost(ball.vel[2], 0.0)
    assert ball.pos[0] + 0.9 * ball.radius < 0.0 and almost(ball.vel[0], 0.0, 3)


def test_iterative_two_aligning_boxes():
    park = _box_park(iterative=True, radius=0.0)
    ball = space_ball(park, 1, 0.0, 50.0, -10.0, vz=1000.0)
    put(park, ball, "maxVelocity", 1000)
    put(park, ball, "Agility", 0.5)
    parent = add(park, 0, x=0.0, y=0.0, z=0.0)
    park.add_mini_box(parent.id, -100.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)
    park.goto_direction(ball.id, 0.0, 0.0, 1.0)
    for _ in range(100):
        park.evolve()
        assert almost(ball.vel[0], 0.0) and almost(ball.vel[1], 0.0)
        assert ball.pos[2] < -0.8 * ball.radius


def test_iterative_balls_with_mini_balls_are_collision_targets():
    park = FullBallpark(iterative=True)
    a = space_ball(park, 1, 0.0, 0.0, -4.0)
    b = space_ball(park, 2, 0.0, 0.0, 4.0)
    put(park, a, "radius", 2.0)
    put(park, b, "radius", 2.0)
    park.set_free(b.id, False)
    park.add_mini_ball(b.id, 0.0, 0.0, 10.0, 2.0)
    park.goto_direction(a.id, 0.0, 0.0, 1.0)
    coords = coordinates(park, a, 50)
    assert almost(b.pos[0], 0.0) and almost(b.pos[1], 0.0) and almost(b.pos[2], 4.0)
    for c in coords:
        assert almost(c[0], 0.0) and almost(c[1], 0.0) and c[2] < 0.5


# ================================================================ test_minis.py and TestCapsule, TestIdGeneration

def test_minis_are_kept_as_given():
    park = FullBallpark()
    parent, = add_balls(park, 1)
    park.add_mini_capsule(parent.id, -1, -2, -3, 1, 2, 3, 10)
    cap = parent.mini_capsules[0]
    assert (cap.a, cap.b) == ([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
    for name, value in (("x", 100), ("y", 200), ("z", 300)):
        put(park, parent, name, value)
    park.add_mini_box(parent.id, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)
    box = parent.mini_boxes[0]
    assert (box.corner, box.lx, box.ly, box.lz) == ([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0],
                                                    [0.0, 0.0, 100.0])
    park.add_mini_box(parent.id, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, -2.0, 1.0)
    box = parent.mini_boxes[1]
    assert (box.lx, box.ly, box.lz) == ([1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, -2.0, 1.0])
    park.add_mini_ball(parent.id, 1, 2, 3, 10)
    assert parent.mini_balls[0].pos == [1.0, 2.0, 3.0]
    park.add_mini_capsule(parent.id, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 42.0)          # TestCapsule
    cap = parent.mini_capsules[-1]
    assert (cap.a, cap.b, cap.radius) == ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 42.0)


def test_minicapsule_radius_must_be_positive():
    park = FullBallpark()
    b = add(park, 0)
    park.add_mini_ball(b.id, 1.0, 2.0, 3.0, 1.0)
    assert len(b.mini_balls) == 1
    park.add_mini_capsule(b.id, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1)
    assert len(b.mini_capsules) == 1
    park2 = FullBallpark()
    b2 = add(park2, 0)
    with pytest.raises(ValueError):
        park2.add_mini_capsule(b2.id, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1)
    assert len(b2.mini_capsules) == 0


@pytest.mark.parametrize("with_parent_minis", [False, True], ids=["one", "several"])
def test_miniball_collision(with_parent_minis):
    park = FullBallpark()
    parent, = add_balls(park, 1)
    for name, value in (("x", 100), ("y", 0), ("z", 0)):
        put(park, parent, name, value)
    park.add_mini_ball(parent.id, 100, 0, 0, 10)
    src = space_ball(park, 2, 200.0, 0.0, -30.0)
    put(park, src, "radius", 10.0)
    if with_parent_minis:
        for args in ((4, 0, 4, 3), (0, 0, 5, 2), (-4, 0, 4, 3)):
            park.add_mini_ball(src.id, *args)
    park.goto_point(src.id, 200.0, 0.0, 0.0)
    for c in coordinates(park, src, 20):
        assert almost(c[0], 200.0) and almost(c[1], 0.0)
        assert (1.01 * c[2] < -17) if with_parent_minis else (1.1 * c[2] < -20)


def test_local_ids_are_generated_and_shared():
    first = -2 ** 30 - 1
    park = FullBallpark()

    def ball_id():
        return park.add_ball(-1, 1, 1, 1, True, False, True, True, False, 1, 1, 1, 1, 1, 1, 1, 1).id

    def capsule_id():
        parent = add(park)
        park.add_mini_capsule(parent.id, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 42.0)
        return parent.mini_capsules[-1].id

    def box_id():
        parent = add(park)
        park.add_mini_box(parent.id, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        return parent.mini_boxes[-1].id

    assert ball_id() == first
    for make in (ball_id, capsule_id, box_id):
        park = FullBallpark()
        assert make() == first
        last = make()
        assert last == first - 1
    park = FullBallpark()
    first_id = ball_id()
    assert capsule_id() == first_id - 1


# ======================================================= test_boxes_and_bubbles.py: static collidables' boxes

def _static_box(park: FullBallpark, sid: int):
    """PyGetStaticCollidableBox (Thunkers.cpp:2302): (width, centre) of its own box."""
    box = park.static_collidables[sid].my_box
    return box.hi[0] - box.lo[0], tuple(box.lo[i] + (box.hi[i] - box.lo[i]) * 0.5 for i in range(3))


def _active(park: FullBallpark, level: int) -> list:
    """GetActiveBoxes (Partition.cpp:490): the low corners of a level's boxes."""
    return [tuple(box.lo) for box in park.levels[level].values()]


def test_static_collidable_box():
    m = MIN_BOX_SIZE
    for a, b, want in (((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), m / 2.0),
                       ((1.0, 1.0, m + 1.0), (1.0, 1.0, m + 3.0), m * 1.5)):
        park = FullBallpark()
        parent = add(park, 0, x=0, y=0, z=0)
        park.add_mini_capsule(parent.id, *a, *b, 1.0)
        width, center = _static_box(park, parent.mini_capsules[-1].id)
        assert width == m and almost(center[0], m / 2.0) and almost(center[1], m / 2.0) and almost(center[2], want)
    for corner, want in (((m / 4.0, m / 4.0, m / 4.0), m / 2.0), ((m / 4.0, m / 4.0, m * 1.5), m * 1.5)):
        park = FullBallpark()
        parent = add(park, 0, x=0, y=0, z=0)
        park.add_mini_box(parent.id, *corner, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        width, center = _static_box(park, parent.mini_boxes[-1].id)
        assert width == m and almost(center[0], m / 2.0) and almost(center[1], m / 2.0) and almost(center[2], want)


def test_active_boxes_of_capsules():
    m = MIN_BOX_SIZE
    for a, b, r, want in (((2.0, 2.0, 2.0), (3.0, 3.0, 3.0), 1.0, [(0.0, 0.0, 0.0)]),
                          ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 2.0,
                           [(-m, -m, -m), (-m, -m, 0.0), (-m, 0.0, -m), (-m, 0.0, 0.0), (0.0, -m, -m), (0.0, -m, 0.0),
                            (0.0, 0.0, -m), (0.0, 0.0, 0.0)]),
                          ((m / 2.0, m / 2.0, m / 2.0), (m * 6.0, m / 2.0, m / 2.0), m / 3.0,
                           [(m * k, 0.0, 0.0) for k in range(7)])):
        park = FullBallpark()
        parent = add(park, 0, x=0, y=0, z=0)
        park.add_mini_capsule(parent.id, *a, *b, r)
        assert sorted(_active(park, 7)) == want


@pytest.mark.parametrize("corner, edge, level, want", [
    ((2.0, 2.0, 2.0), 1.0, 7, [(0.0, 0.0, 0.0)]),
    ((0.0, 0.0, 0.0), 10.0, 7, [(0.0, 0.0, 0.0)]),
    ((0.0, 0.0, 0.0), 10.1, 6, [(0.0, 0.0, 0.0)]),
    ((-1.0, 0.0, 0.0), 5.0, 7, [(0.0, 0.0, 0.0), (-MIN_BOX_SIZE, 0.0, 0.0)]),
    ((MIN_BOX_SIZE - 4.0, 0.0, 0.0), 5.0, 7, [(0.0, 0.0, 0.0), (MIN_BOX_SIZE, 0.0, 0.0)]),
    ((0.0, -1.0, 0.0), 5.0, 7, [(0.0, 0.0, 0.0), (0.0, -MIN_BOX_SIZE, 0.0)]),
    ((0.0, MIN_BOX_SIZE - 4.0, 0.0), 5.0, 7, [(0.0, 0.0, 0.0), (0.0, MIN_BOX_SIZE, 0.0)]),
    ((0.0, 0.0, -1.0), 5.0, 7, [(0.0, 0.0, 0.0), (0.0, 0.0, -MIN_BOX_SIZE)]),
    ((0.0, 0.0, MIN_BOX_SIZE - 4.0), 5.0, 7, [(0.0, 0.0, 0.0), (0.0, 0.0, MIN_BOX_SIZE)]),
])
def test_active_boxes_of_oriented_boxes(corner, edge, level, want):
    park = FullBallpark()
    parent = add(park, 0, x=0.0, y=0.0, z=0.0, radius=500.0 if edge == 1.0 else 0.0)
    park.add_mini_box(parent.id, *corner, edge, 0.0, 0.0, 0.0, edge, 0.0, 0.0, 0.0, edge)
    assert set(_active(park, level)) == set(want)


def test_active_boxes_of_an_oriented_box_in_a_corner():
    m = MIN_BOX_SIZE
    park = FullBallpark()
    parent = add(park, 0, x=0, y=0, z=0)
    park.add_mini_box(parent.id, m - 4.0, m - 4.0, m - 4.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0)
    assert set(_active(park, 7)) == {(x, y, z) for x in (0.0, m) for y in (0.0, m) for z in (0.0, m)}


# ================================================================ test_movement_controls.py: missiles, formations

def test_missile_follow():
    park = FullBallpark()
    src, dst, owner = add_balls(park, 3)
    park.missile_follow(src.id, dst.id, owner.id)
    assert (src.follow_id, src.owner_id, src.mode) == (dst.id, owner.id, MISSILE)
    park = FullBallpark()
    src, dst, owner = add_balls(park, 3)
    put(park, src, "radius", 1.0)
    put(park, dst, "radius", 2.0)
    park.missile_follow(src.id, dst.id, owner.id)
    assert src.follow_range == -3.0
    park = FullBallpark()
    src, dst, owner = add_balls(park, 3)
    park.remove_ball(dst.id)
    park.missile_follow(src.id, dst.id, owner.id)
    assert src.follow_id != dst.id
    park = FullBallpark()
    src, dst, owner = add_balls(park, 3)
    park.follow_ball(src.id, dst.id)
    park.remove_ball(dst.id)
    assert src.follow_id != dst.id


def test_stop_a_missile_and_a_formation_follower():
    park = FullBallpark()
    src, dst, owner = add_balls(park, 3)
    park.missile_follow(src.id, dst.id, owner.id)
    park.stop(src.id)
    assert src.mode == STOP and src.id not in dst.followers and src.follow_id == 0 and src.follow_range == 0
    park = FullBallpark()
    src, dst = add_balls(park, 2)
    park.load_formations(FORMATIONS)
    park.set_ball_formation(dst.id, 1)
    park.formation_follow(src.id, dst.id, 1)
    assert src.mode == FORMATION and src.id in dst.followers                    # TestFormationFollow
    park.stop(src.id)
    assert src.mode == STOP and src.id not in dst.followers and src.follow_id == 0 and src.follow_range == 0


def test_launch_missile():
    def launch(aimed, massive, setup=None):
        park = FullBallpark()
        src, dst, owner = add_balls(park, 3)
        if setup:
            setup(park, src, dst, owner)
        park.launch_missile(src.id, dst.id, owner.id, aimed, massive)
        return park, src, dst, owner

    park, src, dst, owner = launch(False, False)
    assert src.mode == MISSILE and src.id in dst.followers and src.owner_id == owner.id
    assert launch(False, True)[1].is_massive is True
    _, src, _, _ = launch(False, False, lambda p, s, d, o: put(p, d, "x", d.pos[0] + 10000.0))
    assert src.vel[0] == 0.0 and src.vel[1] == 0.0 and almost(src.vel[2], -150.0, 4)

    def faster(p, s, d, o):
        put(p, d, "x", d.pos[0] + 10000.0)
        p.set_max_velocity(o.id, 1234.0)
    _, src, _, _ = launch(False, False, faster)
    assert src.vel[0] == 0.0 and src.vel[1] == 0.0 and almost(src.vel[2], -1234.0, 3)

    def moving(p, s, d, o):
        put(p, d, "x", d.pos[0] + 10000.0)
        put(p, o, "vx", 100.0)
    _, src, _, _ = launch(False, False, moving)
    assert tuple(src.vel) == (250.0, 0.0, 0.0)
    _, src, _, _ = launch(True, False, lambda p, s, d, o: put(p, d, "x", d.pos[0] + 10000.0))
    assert tuple(src.vel) == (1.0, 0.0, 0.0)


def test_can_not_orbit_a_cloaked_ball():
    park = FullBallpark()
    first, second = add_balls(park, 2)
    park.cloak_ball(second.id, DSTNORMALCLOAK)
    park.orbit(first.id, second.id)
    assert first.follow_id != second.id and first.follow_range != 1.0 and first.mode != ORBIT
    assert first.id not in second.followers


# ======================================================================================= test_missile.py

def _missile_park():
    park = FullBallpark()
    dst = space_ball(park, 1, x=1e6)
    owner = space_ball(park, 2)
    collider = space_ball(park, 3, x=2e3)
    put(park, collider, "radius", 1.9e3)
    missile = space_ball(park, 4)
    put(park, missile, "maxVelocity", 300)
    put(park, missile, "Agility", 0.01)
    put(park, missile, "maxAngularVelocity", 2.0)
    put(park, missile, "RotationalAgility", 0.5)
    put(park, missile, "mass", 1e4)
    return park, dst, owner, collider, missile


def _collider_still(collider) -> bool:
    return almost(collider.pos[0], 2e3) and almost(collider.pos[1], 0.0) and almost(collider.pos[2], 0.0)


def test_missile_not_aimed_from_stationary():
    park, dst, owner, collider, missile = _missile_park()
    park.launch_missile(missile.id, dst.id, owner.id, False, False)
    assert almost(missile.vel[0], 0.0) and almost(missile.vel[1], 0.0) and almost(missile.vel[2], -150)
    for _ in range(20):
        park.evolve()
    assert almost(missile.vel[0], missile.max_vel * missile.speed_fraction, 2)
    assert abs(missile.vel[1]) <= 0.5 and abs(missile.vel[2]) <= 0.5
    assert _collider_still(collider)


def test_missile_not_aimed_from_moving():
    park, dst, owner, collider, missile = _missile_park()
    put(park, owner, "vz", 10)
    park.launch_missile(missile.id, dst.id, owner.id, False, False)
    put(park, owner, "vz", 0)
    assert almost(missile.vel[0], 0.0) and almost(missile.vel[1], 0.0) and almost(missile.vel[2], 160)
    for _ in range(20):
        park.evolve()
    assert almost(missile.vel[0], missile.max_vel * missile.speed_fraction, 4)
    assert almost(missile.vel[1], 0.0, 0) and almost(missile.vel[2], 0.0, 0)
    assert _collider_still(collider)


@pytest.mark.parametrize("massive", [False, True])
def test_missile_aimed(massive):
    park, dst, owner, collider, missile = _missile_park()
    park.launch_missile(missile.id, dst.id, owner.id, True, massive)
    if not massive:
        assert almost(missile.vel[0], 1.0) and almost(missile.vel[1], 0.0) and almost(missile.vel[2], 0.0)
    for _ in range(20):
        park.evolve()
    assert almost(missile.vel[0], missile.max_vel * missile.speed_fraction, 4)
    assert almost(missile.vel[1], 0.0, 4) and almost(missile.vel[2], 0.0, 4)
    assert _collider_still(collider)
    if not massive:
        assert missile.pos[0] > collider.pos[0]


# ============================================================ test_getters_and_setters.py, test_visibility.py

def test_velocity_sets_the_orientation():
    park = FullBallpark()
    ball, = add_balls(park, 1)
    park.set_velocity(ball.id, 1.0, 2.0, 3.0)
    assert (ball.yaw, ball.pitch, ball.roll_angle) == (0.3217505543966422, -0.5639426413606289, 0.0)   # bit for bit


def test_boid_troll_rigid():
    park = FullBallpark()
    ball, = add_balls(park, 1)
    park.set_boid(ball.id, True)
    assert ball.mode == BOID
    park.set_boid(ball.id, False)
    assert ball.mode == STOP
    park = FullBallpark()
    ball, = add_balls(park, 1)
    park.set_ball_troll(ball.id, 0)
    assert ball.mode == TROLL
    park = FullBallpark()
    ball, = add_balls(park, 1)
    park.set_ball_troll(ball.id, 3)
    assert ball.mode == TROLL and ball.effect_stamp == 3
    park = FullBallpark()
    ball, = add_balls(park, 1)
    park.set_ball_rigid(ball.id)
    assert ball.mode == RIGID


@pytest.mark.parametrize("dx, args, want", [
    (0.0, (1.0, 1.0, 1.0, 1.0, 1.0), 0.0),                         # zero distance
    (1.0, (1.0, 1.0, 1.0, 1.0, 1.0), 1.0),                         # full accuracy
    (2.0, (1.0, 1.0, 1.0, 1.0, 1.0), 0.5),                         # no falloff
    (2.0, (1.0, 0.95, 1.0, 1.0, 1.0), 0.46392604883716854),        # some falloff (Destiny: 7 places; here exact)
    (2.0, (0.5, 0.95, 0.6, 0.2, 0.7), 0.17762729659881724),        # combination (likewise)
])
def test_get_accuracy(dx, args, want):
    park = FullBallpark()
    src, dst = add_balls(park, 2)
    if dx:
        put(park, dst, "x", dst.pos[0] + dx)
    accuracy, _, _ = park.get_accuracy(src.id, dst.id, *args)
    assert accuracy == want


def test_get_accuracy_returns_and_refuses():
    park = FullBallpark()
    src, dst = add_balls(park, 2)
    assert park.get_accuracy(src.id, src.id, 1.0, 1.0, 1.0, 1.0, 1.0)[0] == 0
    accuracy, optimal, tracking = park.get_accuracy(src.id, dst.id, 42.0, 1.0, 3.14, 1.0, 1.0)
    assert (optimal, tracking) == (42.0, 3.14)


def test_set_ball_formation():
    park = FullBallpark()
    ball, = add_balls(park, 1)
    park.load_formations(FORMATIONS)
    park.set_ball_formation(ball.id, 0)
    assert formation_view(ball) == 0
    park = FullBallpark()
    ball, = add_balls(park, 1)
    park.load_formations(FORMATIONS)
    park.set_ball_formation(ball.id, len(FORMATIONS))
    assert formation_view(ball) == 255
    park = FullBallpark()
    src, dst = add_balls(park, 2)
    park.load_formations(FORMATIONS)
    park.set_ball_formation(dst.id, 1)
    park.formation_follow(src.id, dst.id, 1)
    park.set_ball_formation(dst.id, 2)
    assert src.mode == STOP


def test_cloaking():
    park = FullBallpark()
    src, dst, occluder = add_balls(park, 3)
    put(park, occluder, "radius", 25.0)
    put(park, occluder, "x", occluder.pos[0] + 50.0)
    put(park, occluder, "isMassive", True)
    put(park, occluder, "isCloaked", True)
    put(park, dst, "x", dst.pos[0] + 100.0)
    assert park.check_visibility(src.id, dst.id) != occluder.id
    park = FullBallpark()
    ball, = add_balls(park, 1)
    put(park, ball, "isMassive", True)
    park.cloak_ball(ball.id, DSTNORMALCLOAK)
    assert ball.is_cloaked and not ball.is_massive
    park = FullBallpark()
    ball, = add_balls(park, 1)
    put(park, ball, "isCloaked", True)
    put(park, ball, "isMassive", False)
    park.uncloak_ball(ball.id)
    assert not ball.is_cloaked and ball.is_massive


def test_uncloaking_in_warp_leaves_a_ball_non_massive():
    park = FullBallpark()
    ball, = add_balls(park, 1)
    put(park, ball, "isMassive", False)
    for name in "xyz":
        put(park, ball, name, 0)
    put(park, ball, "maxVelocity", 1.0)
    put(park, ball, "vx", 1.0)
    park.set_free(ball.id, True)
    park.warp_to(ball.id, TEN_BILLION, 0, 0)
    park.evolve()
    assert ball.mode == WARP
    park.uncloak_ball(ball.id)
    assert not ball.is_massive


# ============================================================================== test_ball.py: slots, rotation

def test_formation_slots():
    park = FullBallpark()
    b = add(park, 0)
    assert park.reserve_formation_slot(b.id) == -1                       # no formation
    assert formation_view(b) == 255 and (b.yaw, b.pitch, b.roll_angle) == (0.0, 0.0, 0.0)
    assert list(park._rotated_vector(b, (1.0, 2.0, 3.0))) == [1.0, 2.0, 3.0]
    park.load_formations(FORMATIONS)
    park.set_ball_formation(b.id, 1)
    assert [park.reserve_formation_slot(b.id) for _ in FORMATIONS[1]] == [0, 1, 2, 3]
    assert park.reserve_formation_slot(b.id) == -1
    park = FullBallpark()
    park.load_formations(FORMATIONS)
    b = add(park, 0)
    park.set_ball_formation(b.id, 1)
    for _ in range(MAX_FORMATION_SLOTS_FOR_BALLS):
        park.reserve_formation_slot(b.id)
    park.free_formation_slot(b.id, 2)
    assert park.reserve_formation_slot(b.id) == 2


# ============================================================================== test_follow.py: TestFormation

FORMATION_STOPPED = [                                       # test_formation_follow_stopped_ball
    (0.34937874984850925, 0.0, -0.18354986595744605), (0.8278427298581066, -0.3459978313028692, -0.584942714251052),
    (0.8983700252647063, -1.344611204434962, -1.0160119154083649),
    (0.5844941405575219, -2.8926128310477566, -1.5975939889522262),
    (-0.08489523178080371, -4.784247012129277, -2.593859202902083),
    (-1.0834106420909224, -6.787529292534025, -4.122561424197542),
    (-2.3834969832762254, -8.744771342586718, -6.190397079773381),
    (-3.958565430928698, -10.569491075166201, -8.76574805927726),
    (-5.783982011387288, -12.21826200033105, -11.808792734080857),
    (-7.837195912453384, -13.670582854838065, -15.280267553697655)]
FORMATION_MOVING = [                                        # test_formation_follow_moving_ball
    (0.34937874956616805, 0.0, -0.18354986649486965), (1.2522700311290196, -0.2984878365701446, -0.442015514428003),
    (2.6428549859837505, -1.083812623793305, -0.568440079311451),
    (4.525369455660191, -2.2133606150700387, -0.8465910315244277),
    (6.698838038625631, -3.6099869836316993, -1.5982896364292383),
    (8.916087445091419, -5.209637520522771, -2.9541942809113397),
    (11.011849619430125, -6.955575729117526, -4.920658718422647),
    (12.897996631839323, -8.79576410566157, -7.465035973264715),
    (14.532033823223955, -10.680120838194423, -10.547372880541618),
    (15.895577914276311, -12.55839407965528, -14.128291692594965)]


def points_almost_equal(first: list, second: list, places: int = 4) -> bool:
    """helpers.assertListOfPointsAlmostEqual: equal points, or every coordinate equal when rounded to `places`."""
    return len(first) == len(second) and all(p == q or all(round(p[j], places) == round(q[j], places)
                                                           for j in range(3)) for p, q in zip(first, second))


@pytest.mark.parametrize("leader_moves, expected", [(False, FORMATION_STOPPED), (True, FORMATION_MOVING)],
                         ids=["stopped", "moving"])
def test_formation_follow_turns_the_slot_by_the_leader(leader_moves, expected):
    """TestFormation: ten ticks, the slot offset turned by the leader's yaw, pitch and roll (DirectXMath) from the
    second tick on, within Destiny's four decimals; the first tick (zero angles: the identity) bit for bit."""
    park = FullBallpark()
    park.load_formations(FORMATIONS)
    follower = space_ball(park, 1)
    leader = space_ball(park, 2, x=-5.0)
    park.set_ball_formation(leader.id, 1)
    slot = park.reserve_formation_slot(leader.id)
    park.formation_follow(follower.id, leader.id, slot)
    if leader_moves:
        park.goto_point(leader.id, 0.0, 100.0, 0.0)
    got = []
    for i in range(10):
        park.evolve()
        got.append(tuple(follower.pos))
        if i == 0:
            assert (leader.yaw, leader.pitch) != (0.0, 0.0)     # turned toward its goto point at the first tick
    assert points_almost_equal(got, expected)
    assert got[0] == FORMATION_STOPPED[0]                       # bit for bit (Destiny's moving vector differs in the
                                                                # 10th decimal: its code compiled here agrees)


# ================================================================ test_lifecycle_management.py: TestAddMushroom

def test_mushroom_is_created_at_id_zero():
    park = FullBallpark()
    owner, = add_balls(park, 1)
    park.add_mushroom(owner.id, 10.0, 1.0)
    m = park.balls[0]
    assert (m.mode, m.owner_id, m.follow_range, m.goto[0], m.is_moribund) == (MUSHROOM, owner.id, 10.0, 1.0, False)
    park = FullBallpark()
    park.add_mushroom(1, 10.0, 1.0)
    assert 0 not in park.balls


def test_mushroom_expands_only_when_free():
    park = FullBallpark()
    owner, = add_balls(park, 1)
    park.add_mushroom(owner.id, 10.0, 5.0)
    m = park.balls[0]
    park.evolve()
    park.evolve()
    assert almost(m.radius, 0.0)
    park = FullBallpark()
    owner, = add_balls(park, 1)
    park.add_mushroom(owner.id, 10.0, 5.0)
    m = park.balls[0]
    park.set_free(m.id, True)
    assert m.radius == 0.0
    radii = []
    for _ in range(6):
        park.evolve()
        radii.append(m.radius)
    assert radii == [0.0, 6.687403202056885, 7.952707290649414, 8.801117897033691, 9.457415580749512,
                     9.457415580749512]


def test_mushroom_dies_when_fully_expanded():
    park = FullBallpark()
    owner, = add_balls(park, 1)
    park.add_mushroom(owner.id, 10.0, 1.0)
    m = park.balls[0]
    park.set_free(m.id, True)
    park.evolve()
    park.evolve()
    assert m.is_moribund

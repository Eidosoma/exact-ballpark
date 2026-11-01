# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against published test vectors for the space layer: the reference
(exact_ballpark.space) against the expectations of Destiny's own tests.

The expected values are copied from carbonengine/destiny at 114e89fa (MIT licence, (c) CCP Games): python/destiny/test/
ballpark/evolve/test_simple_collision.py (TestSimpleCollision), ballpark/test_boxes_and_bubbles.py (InitializeBubbles,
GetBallIdsInRange, GetBallIdsAndDistInRange, GetBallBox, GetActiveBoxes, GetBoxCenter, GetBubbleAtCoordinates),
ballpark/test_visibility.py (TestCheckVisibility), ballpark/test_callbacks.py (TestProximitySensors,
TestNotificationRange, TestSetTargetTracking), ballpark/test_getters_and_setters.py (TestGetRemoteFollowers),
ballpark/evolve/test_iterative_collision.py (TestIterativeCollision: the ball-ball cases, whose checks are bounds, kept
with their tolerances) and tests/TestPartition.cpp, with the balls of helpers.py (add_ball_to_park: the POPO defaults,
not free, at 1e10 m on each axis; create_space_ball: then SetBallFree, max velocity 10, agility 0.9, mass 13,000,000,
radius 2, massive, speed fraction 0.95). Destiny's tests run on `destiny.Ballpark()`, a client's view (isMaster false).
A Python attribute write (`ball.x = ...`) is Blue's: the member is written, then Ball::OnModified calls the park's
setter (Ball.cpp:1507), so `ball.isFree = True` leaves the ball out of the free set (SetBallFree returns at once,
Ballpark.cpp:4897). Destiny's trajectory tests compare to four decimal places; these ask for bit equality. The tests of
capsules, oriented boxes, miniballs, cones, triangles, scan cones, cloaking (CloakBall) and dynamical orientation are
not in this slice.
"""

from __future__ import annotations

from exact_ballpark.ref import f32
from exact_ballpark.space import SpaceBallpark, box_center, key_hash

TEN_BILLION = 10000000000.0
MIN_BOX_SIZE = 480.0


def add(park: SpaceBallpark, object_id: int, **kw):
    """helpers.add_ball_to_park with the POPO defaults."""
    b = {"mass": 0.0, "radius": 0.0, "maxVel": 0.0, "isFree": False, "isGlobal": False, "isMassive": False,
         "isInteractive": False, "isSpaceJunk": False, "x": TEN_BILLION, "y": TEN_BILLION, "z": TEN_BILLION,
         "vx": 0.0, "vy": 0.0, "vz": 0.0, "agility": 0.0, "speedFraction": 0.0}
    b.update(kw)
    return park.add_ball(object_id, b["mass"], b["radius"], b["maxVel"], b["isFree"], b["isGlobal"], b["isMassive"],
                         b["isInteractive"], b["isSpaceJunk"], b["x"], b["y"], b["z"], b["vx"], b["vy"], b["vz"],
                         b["agility"], b["speedFraction"])


def put(park: SpaceBallpark, ball, name: str, value) -> None:
    """A Python attribute write: the member, then OnModified's setter (Ball.cpp:1507)."""
    if name in ("x", "y", "z"):
        pos = list(ball.pos)
        pos["xyz".index(name)] = float(value)
        ball.pos = pos
        park.set_position(ball.id, *pos)
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
        park.set_free(ball.id, bool(value))               # returns at once: the flag is already the member's
    elif name == "isMassive":
        ball.is_massive = bool(value)
        park.set_massive(ball.id, bool(value))
    elif name == "isCloaked":
        ball.is_cloaked = int(value)                        # no setter runs for isCloaked
    else:
        raise ValueError(name)


def space_ball(park: SpaceBallpark, bid: int, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, mass=None,
               max_velocity=None):
    """helpers.create_space_ball (ids increase with each call, as next_object_id)."""
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
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2) ** 0.5


def evolve(park: SpaceBallpark, balls: list, ticks: int) -> list:
    out = []
    for _ in range(ticks):
        park.evolve()
        out.append([tuple(b.pos) for b in balls])
    return out


# ---- test_simple_collision.py, TestSimpleCollision (COL-705 to COL-711), bit for bit ----

SAME_LOCATION_A = [
    (1.121144335565424, 0.0, 0.0), (3.2401005010556143, 0.0, 0.0), (5.185473058408457, 0.0, 0.0),
    (6.9714818718687654, 0.0, 0.0), (8.61118192299853, 0.0, 0.0), (10.116558737162107, 0.0, 0.0),
    (11.498615992731432, 0.0, 0.0), (12.76745595339809, 0.0, 0.0), (13.93235331151902, 0.0, 0.0),
    (15.001822982259961, 0.0, 0.0)]
INTERSECTING_A = [
    (0.0, 2.5951094880505643, 0.0), (0.0, 3.736969417977334, 0.0), (0.0, 4.648125579572035, 0.0),
    (0.0, 5.375189844683108, 0.0), (0.0, 5.955356525086511, 0.0), (0.0, 6.418305116018683, 0.0),
    (0.0, 6.787718605426691, 0.0), (0.0, 7.082495020842102, 0.0), (0.0, 7.317714192790605, 0.0),
    (0.0, 7.505409191300473, 0.0)]
INTERSECTING_B = [
    (0.0, -0.6951094880505645, 0.0), (0.0, -1.836969417977334, 0.0), (0.0, -2.7481255795720347, 0.0),
    (0.0, -3.4751898446831078, 0.0), (0.0, -4.055356525086512, 0.0), (0.0, -4.5183051160186825, 0.0),
    (0.0, -4.8877186054266915, 0.0), (0.0, -5.182495020842103, 0.0), (0.0, -5.417714192790607, 0.0),
    (0.0, -5.605409191300473, 0.0)]
GOTO_A = [0.22785672701553972, 0.8863613208951594, 1.9402353687182086, 3.3570904438241858, 3.745389410116848,
          3.2276392358401793, 3.2016165950513056, 3.627039465463305, 3.8640182292625984, 3.9449236847646176,
          4.014836947038442, 4.087400730922078, 4.152795682608074, 4.2242631173879985, 4.289057014884843,
          4.360366727534607, 4.425074016382081, 4.496360773851808, 4.561055454563248, 4.624809527703686]
GOTO_B = [9.772143272984462, 9.113638679104842, 8.059764631281793, 6.642909556175815, 6.254610589883153,
          6.772360764159822, 6.7983834049486935, 6.372960534536694, 6.1359817707374, 6.055076315235381,
          5.985163052961557, 5.912599269077921, 5.847204317391924, 5.775736882612, 5.710942985115155,
          5.639633272465392, 5.574925983617919, 5.503639226148192, 5.438944545436752, 5.375190472296313]


def test_stopped_balls_with_same_location():
    park = SpaceBallpark()
    a, b = space_ball(park, 1), space_ball(park, 2)
    got = evolve(park, [a, b], 10)
    assert [g[0] for g in got] == SAME_LOCATION_A
    assert [g[1] for g in got] == [(-x, y, z) for (x, y, z) in SAME_LOCATION_A]


def test_stopped_balls_intersecting():
    park = SpaceBallpark()
    a, b = space_ball(park, 1, y=1.9), space_ball(park, 2)
    got = evolve(park, [a, b], 10)
    assert [g[0] for g in got] == INTERSECTING_A
    assert [g[1] for g in got] == INTERSECTING_B


def test_goto_collision():
    park = SpaceBallpark()
    a, b = space_ball(park, 1), space_ball(park, 2, x=10, y=10, z=10)
    park.goto_point(a.id, 10, 10, 10)
    park.goto_point(b.id, 0, 0, 0)
    got = evolve(park, [a, b], 20)
    assert [g[0] for g in got] == [(x, x, x) for x in GOTO_A]
    assert [g[1] for g in got] == [(x, x, x) for x in GOTO_B]


def test_balls_in_warp_should_not_collide():                 # COL-702: warp proper is non-massive
    park = SpaceBallpark()
    a = space_ball(park, 1, x=-4, vx=1000.0, max_velocity=1000.0, mass=1.0)
    b = space_ball(park, 2, x=4, vx=-1000.0, max_velocity=1000.0, mass=1.0)
    put(park, a, "radius", 2.0)
    put(park, b, "radius", 2.0)
    park.warp_to(a.id, 100000.0, 0, 0)
    park.warp_to(b.id, -100000.0, 0, 0)
    for _ in range(80):
        park.evolve()
    assert a.pos[0] > b.pos[0]
    assert (a.pos[1], a.pos[2], b.pos[1], b.pos[2]) == (0.0, 0.0, 0.0, 0.0)


# ---- test_boxes_and_bubbles.py (SEN-705, SEN-706, SEN-707, SEN-710) ----

def test_can_initialize_bubbles():
    park = SpaceBallpark()
    ball = add(park, 1)
    assert ball.bubble == -1
    park.initialize_bubbles()
    assert ball.bubble != -1 and ball.old_bubble == -1


def test_can_initialize_bubble_for_specific_ball():
    park = SpaceBallpark()
    ball, other = add(park, 1), add(park, 2)
    park.initialize_bubbles(ball.id)
    assert ball.bubble != -1 and ball.old_bubble == -1 and other.bubble == -1


def _two(dx: float = 0.0):
    park = SpaceBallpark()
    ball, other = add(park, 1), add(park, 2)
    if dx:
        put(park, other, "x", other.pos[0] + dx)
    park.initialize_bubbles()
    return park, ball, other


def test_get_ball_ids_in_range():
    park, ball, other = _two()
    assert park.ball_ids_in_range(ball.id, 100) == [other.id]
    assert _two(100.0)[0].ball_ids_in_range(1, 100.0) == [2]                 # on the edge: included
    assert _two(100.1)[0].ball_ids_in_range(1, 100) == []
    assert park.ball_ids_in_range(*ball.pos, 100) == [ball.id, other.id]     # from a location


def test_get_ball_ids_and_dist_in_range():
    park, ball, other = _two()
    assert park.ball_ids_in_range(ball.id, 100, include_dist=True) == [(0.0, 2)]
    assert _two(2.0)[0].ball_ids_in_range(1, 100.0, include_dist=True) == [(4.0, 2)]
    assert _two(100.0)[0].ball_ids_in_range(1, 100.0, include_dist=True) == [(10000.0, 2)]
    assert _two(100.1)[0].ball_ids_in_range(1, 100, include_dist=True) == []
    assert park.ball_ids_in_range(*ball.pos, 100, include_dist=True) == [(0.0, ball.id), (0.0, other.id)]


def _ball_box(park, ball):
    """PyGetBallBox (Thunkers.cpp:2260): (width, centre) of the ball's own box."""
    box = ball.my_box
    return box.hi[0] - box.lo[0], tuple(box.lo[i] + (box.hi[i] - box.lo[i]) * 0.5 for i in range(3))


def test_ball_box():
    for z, want in ((1.0, MIN_BOX_SIZE / 2.0), (MIN_BOX_SIZE + 1.0, MIN_BOX_SIZE * 1.5)):
        park = SpaceBallpark()
        ball = add(park, 1)
        put(park, ball, "x", 1.0)
        put(park, ball, "y", 1.0)
        put(park, ball, "z", z)
        assert _ball_box(park, ball) == (MIN_BOX_SIZE, (MIN_BOX_SIZE / 2.0, MIN_BOX_SIZE / 2.0, want))


def test_active_boxes():
    park = SpaceBallpark()
    assert not park.levels[0]                                # GetActiveBoxes(0) is None: no box
    ball = add(park, 1)
    for name in "xyz":
        put(park, ball, name, 1)
    boxes = park.levels[7]
    assert [tuple(box.lo) for box in boxes.values()] == [(0.0, 0.0, 0.0)]
    assert next(iter(boxes.values())).hi[0] - next(iter(boxes.values())).lo[0] == MIN_BOX_SIZE


def test_box_center():
    assert box_center(7, (0.0, 0.0, 0.0)) == [MIN_BOX_SIZE / 2.0] * 3


def test_bubble_at_coordinates():
    park = SpaceBallpark()
    assert park.bubble_at(0.0, 0.0, 0.0) is None
    ball = add(park, 1, x=0.0, y=0.0, z=0.0)
    assert park.bubble_at(0.0, 0.0, 0.0) == ball.bubble


# ---- test_visibility.py, TestCheckVisibility (SEN-709) ----

def _visibility(massive: bool | None, cloaked: bool = False):
    park = SpaceBallpark()
    src, dst = add(park, 1), add(park, 2)
    occluder = add(park, 3) if massive is not None else None
    if occluder is not None:
        put(park, occluder, "radius", 25.0)
        put(park, occluder, "x", occluder.pos[0] + 50.0)
        put(park, occluder, "isMassive", massive)
        if cloaked:
            put(park, occluder, "isCloaked", True)
    put(park, dst, "x", dst.pos[0] + 100.0)
    return park.check_visibility(src.id, dst.id)


def test_check_visibility():
    assert _visibility(None) == 0                            # test_returns_zero_when_there_is_no_occlusion
    assert _visibility(True) == 3                            # ..._returns_the_id_of_the_occluding_ball_...
    assert _visibility(False) != 3                           # test_ball_that_is_not_massive_is_not_an_occlusion
    assert _visibility(True, cloaked=True) != 3              # test_ball_that_is_cloaked_is_not_an_occlusion


# ---- test_callbacks.py (SEN-701 to SEN-704) ----

def _proximity_events(park, bid):
    return [(e[3], e[4]) for e in park.events if e[1] == "DoProximity" and e[2] == bid]


def _sensor(period: float, interactives: bool = False, remove: bool = False, regular_free: bool = True,
            dx: float = 0.0):
    park = SpaceBallpark()
    proximity_ball = add(park, 3)
    park.add_proximity_sensor(3, 10.0, period, False, interactives)
    if remove:
        park.remove_proximity_sensor(3)
    regular = add(park, 4, x=proximity_ball.pos[0] + dx) if dx else add(park, 4)
    put(park, regular, "isFree", regular_free)
    return park, regular


def test_proximity_callback_after_the_period():
    park, regular = _sensor(2.0)
    park.evolve()
    assert _proximity_events(park, 3) == []
    park.evolve()
    assert _proximity_events(park, 3) == [(regular.id, 1)]


def test_proximity_callbacks():
    park, regular = _sensor(0.0)
    park.evolve()
    assert _proximity_events(park, 3) == [(regular.id, 1)]                     # test_callback_gets_called
    for kw in ({"regular_free": False}, {"interactives": True}, {"remove": True}, {"dx": 11.0}):
        park, regular = _sensor(0.0, **kw)
        park.evolve()
        assert _proximity_events(park, 3) == [], kw
    park, regular = _sensor(0.0, dx=10.0)                                       # on the edge of the range
    park.evolve()
    assert _proximity_events(park, 3) == [(regular.id, 1)]


def test_leave_proximity():
    park, regular = _sensor(0.0)
    park.evolve()
    put(park, regular, "x", regular.pos[0] + 11.0)
    park.evolve()
    assert _proximity_events(park, 3) == [(regular.id, 1), (regular.id, 0)]


def _range_events(park, bid, name="DoRange"):
    return [e[3] for e in park.events if e[1] == name and e[2] == bid]


def test_notification_range():
    park = SpaceBallpark()                                   # test_detects_ball_in_range_when_in_goto_mode
    ball = add(park, 3)
    other = add(park, 4, x=ball.pos[0] + 1.0)
    park.goto_point(ball.id, *other.pos)
    park.set_notification_range(ball.id, 10.0)
    park.evolve()
    assert _range_events(park, 3) == [1]
    park = SpaceBallpark()                                   # test_ignores_ball_on_edge_of_range
    ball = add(park, 3, radius=1.0)
    other = add(park, 4, x=ball.pos[0] + 11.0)
    park.goto_point(ball.id, *other.pos)
    park.set_notification_range(ball.id, 10.0)
    park.evolve()
    assert _range_events(park, 3) == []
    park = SpaceBallpark()                                   # test_ignores_ball_that_should_be_out_of_range
    ball = add(park, 3)
    other = add(park, 4, y=ball.pos[1] + 100.0)
    park.goto_point(ball.id, *other.pos)
    park.set_notification_range(ball.id, 10.0)
    park.evolve()
    assert _range_events(park, 3) == []
    park = SpaceBallpark()                                   # test_ignores_ball_in_range_if_stopped
    ball = add(park, 3)
    add(park, 4)
    park.set_notification_range(ball.id, 10.0)
    park.evolve()
    assert _range_events(park, 3) == []
    park = SpaceBallpark()                                   # test_detects_ball_leaving_range
    ball = add(park, 3)
    park.set_notification_range(ball.id, 10.0)
    add(park, 4)
    park.goto_point(ball.id, ball.pos[0] + 1.0, ball.pos[1], ball.pos[2])
    park.evolve()
    put(park, ball, "x", ball.pos[0] + 100.0)
    park.evolve()
    assert _range_events(park, 3) == [1, 0]


def test_target_tracking():
    for dx, want in ((0.0, [1]), (10.0, []), (11.0, [])):    # test_foo, _on_edge_of_range, _out_of_range
        park = SpaceBallpark()
        ball = add(park, 3)
        other = add(park, 4, x=ball.pos[0] + dx) if dx else add(park, 4)
        park.set_target_tracking(ball.id, other.id, 10.0)
        park.evolve()
        assert _range_events(park, 3, "DoTargetTracking") == want, dx
    park = SpaceBallpark()                                   # test_detects_target_leaving_range
    ball, other = add(park, 3), add(park, 4)
    park.set_target_tracking(ball.id, other.id, 10.0)
    park.evolve()
    put(park, other, "x", other.pos[0] + 100.0)
    park.evolve()
    assert _range_events(park, 3, "DoTargetTracking") == [1, 0]


# ---- test_getters_and_setters.py, TestGetRemoteFollowers (MOV-731, SEN-708) ----

def _remote(free: bool | None, follow: bool, move: bool) -> list:
    park = SpaceBallpark()
    src, dst = add(park, 1), add(park, 2)
    if free is not None:
        park.set_free(src.id, free)
    if follow:
        park.follow_ball(src.id, dst.id)
    if move:
        put(park, dst, "x", 0)
    park.initialize_bubbles()
    return park.remote_followers()


def test_get_remote_followers():
    assert SpaceBallpark().remote_followers() == []                   # test_returns_empty_list_when_there_are_none
    assert _remote(False, True, True) == []                           # test_ignores_ball_that_is_not_free
    assert _remote(True, True, True) == [1]                           # test_finds_following_ball
    assert _remote(True, False, True) == []                           # test_ignores_balls_that_are_not_following
    assert _remote(True, True, False) == []                           # test_ignores_balls_that_are_in_the_same_bubble


# ---- tests/TestPartition.cpp ----

def test_partition_key_hash_separates_coordinates_that_previously_collided():
    assert key_hash(1, 0, 0, 16) != key_hash(0, 2, 0, 16)


# ---- test_iterative_collision.py, TestIterativeCollision (COL-714 to COL-718), as bounds ----

def iterative_park() -> SpaceBallpark:
    return SpaceBallpark(iterative=True)                     # settings.useIterativeCollision = True


def test_iterative_stopped_balls_with_same_location():
    park = iterative_park()
    a, b = space_ball(park, 1), space_ball(park, 2)
    path = []
    for _ in range(10):
        park.evolve()
        path.append((tuple(a.pos), tuple(b.pos)))
        assert all(almost(a.vel[i], -b.vel[i]) for i in range(3))
    assert distance(*path[2]) > a.radius + b.radius


def test_iterative_multiple_overlapping_balls():
    park = iterative_park()
    a, b = space_ball(park, 1, x=1.0), space_ball(park, 2, x=-1.0)
    c, d = space_ball(park, 3, z=1.0), space_ball(park, 4, z=-1.0)
    path = []
    for _ in range(10):
        park.evolve()
        path.append([tuple(x.pos) for x in (a, b, c, d)])
        assert all(almost(x.vel[1], 0.0) for x in (a, b, c, d))
    four = path[4]
    assert all(distance(four[i], four[j]) > a.radius * 2 for i in range(4) for j in range(i + 1, 4))


def test_iterative_stopped_balls_intersecting():
    park = iterative_park()
    a, b = space_ball(park, 1, y=1.9), space_ball(park, 2)
    path = []
    for _ in range(10):
        park.evolve()
        path.append((tuple(a.pos), tuple(b.pos)))
        assert a.vel[1] > 0.0 and b.vel[1] < 0.0 and almost(a.vel[1], -b.vel[1])
        assert all(almost(v, 0.0) for v in (a.vel[0], a.vel[2], b.vel[0], b.vel[2]))
    assert distance(*path[1]) > a.radius + b.radius
    for pa, pb in path:
        assert almost(pa[0], path[0][0][0], 4) and almost(pa[2], path[0][0][2], 4)
        assert almost(pb[0], path[0][1][0], 4) and almost(pb[2], path[0][1][2], 4)


def test_iterative_ball_within_static_ball():
    park = iterative_park()
    ball = space_ball(park, 1, z=8, vz=2)
    asteroid = space_ball(park, 2)
    put(park, asteroid, "radius", 10)
    park.set_free(asteroid.id, False)
    path = evolve(park, [ball], 10)
    last = path[-1][0]
    assert last[2] > 12 and almost(last[0], 0.0, 4) and almost(last[1], 0.0, 4)


def test_iterative_balls_intersecting_with_goto():
    park = iterative_park()
    a, b = space_ball(park, 1, vx=3), space_ball(park, 2, x=10, vx=-3)
    put(park, a, "radius", 10)
    put(park, b, "radius", 10)
    park.goto_direction(a.id, 1.0, 0.0, 0.0)
    park.goto_direction(b.id, -1.0, 0.0, 0.0)
    pa, pb, va, vb = [], [], [], []
    for _ in range(20):
        park.evolve()
        pa.append(list(a.pos))
        pb.append(list(b.pos))
        va.append(list(a.vel))
        vb.append(list(b.vel))
    dist = [distance(pb[i], pa[i]) for i in range(20)]
    inside = True
    for i in range(20):
        assert all(almost(x, 0.0, 1) for x in (pa[i][1], pa[i][2], pb[i][1], pb[i][2], va[i][1], va[i][2], vb[i][1],
                                               vb[i][2]))
        if dist[i] > a.radius + b.radius:
            inside = False
        if inside:
            assert va[i][0] < 0.0 and vb[i][0] > 0.0
            if i > 0:
                assert dist[i] > dist[i - 1]
        elif i + 1 < 20:
            if dist[i] + (vb[i][0] - va[i][0]) > a.radius + b.radius:
                assert vb[i + 1][0] < vb[i][0] and va[i + 1][0] > va[i][0]
            else:
                assert vb[i + 1][0] > vb[i][0] and va[i + 1][0] < va[i][0]
    assert not inside


def test_iterative_collision_small_with_massive():           # base masses: the large ball gets 0.04 m/s
    park = iterative_park()
    small = space_ball(park, 1, x=-2, vx=20, mass=1e6, max_velocity=20)
    put(park, small, "speedFraction", 1.0)
    put(park, small, "Agility", 1000)
    put(park, small, "radius", 2.0)
    large = space_ball(park, 2, x=22.01, mass=1e9, max_velocity=10)
    put(park, large, "speedFraction", 1.0)
    put(park, large, "radius", 20)
    park.goto_direction(small.id, -1.0, 0.0, 0.0)
    park.evolve()
    assert almost(large.vel[0], 0.04, 2) and almost(large.vel[1], 0.0) and almost(large.vel[2], 0.0)
    assert almost(small.vel[0], -19.96, 1) and almost(small.vel[1], 0.0, 4) and almost(small.vel[2], 0.0, 4)


def test_iterative_collision_massive_with_small():
    park = iterative_park()
    small = space_ball(park, 1, mass=1e6)
    large = space_ball(park, 2, x=30, vx=-20, max_velocity=20, mass=1e9)
    put(park, large, "radius", 20.0)
    put(park, large, "speedFraction", 1.0)
    put(park, small, "Agility", 3.0)
    park.goto_direction(large.id, -1.0, 0.0, 0.0)
    for _ in range(20):
        park.evolve()
        assert distance(small.pos, large.pos) >= 0.9 * (large.radius + small.radius)
        assert large.vel[0] < -19.0
        assert all(almost(v, 0.0, 2) for v in (small.vel[1], small.vel[2], large.vel[1], large.vel[2]))


def test_iterative_goto_collision():
    park = iterative_park()
    a, b = space_ball(park, 1), space_ball(park, 2, z=10)
    park.goto_point(a.id, 0, 0, 10)
    park.goto_point(b.id, 0, 0, 0)
    path = evolve(park, [a, b], 20)
    for pa, pb in path:
        assert 1.11 * distance(pa, pb) > a.radius + b.radius
        assert almost(pa[0], path[0][0][0]) and almost(pa[1], path[0][0][1])
        assert almost(pb[0], path[0][1][0]) and almost(pb[1], path[0][1][1])


def test_iterative_balls_in_warp_should_not_collide():
    park = iterative_park()
    a = space_ball(park, 1, x=-40, vx=10.0, max_velocity=10.0)
    b = space_ball(park, 2, x=40, vx=-10.0, max_velocity=10.0)
    put(park, a, "radius", 2.0)
    put(park, b, "radius", 2.0)
    park.warp_to(a.id, 10000000.0, 0, 0, 0.0, 1000)
    park.warp_to(b.id, -10000000.0, 0, 0, 0.0, 1000)
    for _ in range(80):
        park.evolve()
    assert a.pos[0] > b.pos[0] and (a.pos[1], a.pos[2], b.pos[1], b.pos[2]) == (0.0, 0.0, 0.0, 0.0)


def test_iterative_non_massive_balls_should_not_collide():
    park = iterative_park()
    a, b = space_ball(park, 1, z=-4), space_ball(park, 2, z=4)
    put(park, a, "radius", 2.0)
    put(park, b, "radius", 2.0)
    park.set_massive(b.id, False)
    park.goto_direction(a.id, 0, 0, 1)
    for _ in range(80):
        park.evolve()
    assert a.pos[2] > b.pos[2] and almost(a.pos[0], 0.0) and almost(a.pos[1], 0.0)
    assert almost(b.pos[0], 0.0) and almost(b.pos[1], 0.0) and almost(b.pos[2], 4.0)


def test_iterative_correct_collision_order():
    park = iterative_park()
    b = space_ball(park, 1, z=4)
    c = space_ball(park, 2, z=10, mass=1e6)
    a = space_ball(park, 3, z=-4, vz=20, max_velocity=100.0)
    for x in (a, b, c):
        put(park, x, "radius", 2.0)
    park.set_free(b.id, False)
    park.evolve()
    assert almost(c.pos[0], 0.0) and almost(c.pos[1], 0.0) and almost(c.pos[2], 10.0)
    assert b.pos[2] > a.pos[2]


def test_iterative_substep_collision():                      # the velocity passes down a chain: c ends at 28 m
    park = SpaceBallpark(friction=1e-10, iterative=True)
    a = space_ball(park, 1, vz=20, max_velocity=100)
    c = space_ball(park, 2, z=10, max_velocity=100)
    b = space_ball(park, 3, z=5, max_velocity=100)
    park.evolve()
    assert almost(a.vel[2], 0.0, 0) and almost(a.pos[2], 1.0, 0)
    assert almost(b.vel[2], 0.0, 0) and almost(b.pos[2], 6.0, 0)
    assert almost(c.vel[2], 20.0, 0) and almost(c.pos[2], 28.0, 0)


def test_iterative_substep_collision_static():
    park = SpaceBallpark(friction=1e-10, iterative=True)
    a = space_ball(park, 1, z=-14, vz=50, max_velocity=100)
    b = space_ball(park, 2, x=-10, vx=50, max_velocity=100)
    c = space_ball(park, 3, z=20)
    d = space_ball(park, 4, x=20)
    park.set_free(c.id, False)
    park.set_free(d.id, False)
    park.evolve()
    assert almost(a.vel[2], 0.0, 2) and almost(a.vel[0], 0.0, 2) and almost(a.pos[2], -4.0, 2)
    assert almost(b.vel[2], 50.0, 2) and almost(b.vel[0], 50.0, 2)

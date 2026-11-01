# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against published test vectors for the flight modes: the FOLLOW, ORBIT and WARP
reference (exact_ballpark.modes) against the expectations of Destiny's own Python tests.

The expected values are copied from carbonengine/destiny at 114e89fa (MIT licence, (c) CCP Games),
python/destiny/test/: ballpark/evolve/test_follow.py, test_orbit.py and test_warp.py, and
ballpark/test_movement_controls.py, with the balls of helpers.py (create_space_ball: max velocity 10, agility 0.9, mass
13,000,000, radius 2, speed fraction 0.95, massive; add_ball_to_park: the POPO defaults, not free, at 1e10 m on each
axis). Destiny's trajectory tests compare to four decimal places; these ask for bit equality. Its new-orbit tests give
bounds, not values; they are kept as bounds. Where Destiny's test reaches a state through a call outside this slice
(RemoveBall, CloakBall, InitializeBubbles), the flag that call sets is set directly, as noted. Beside them: the
reference keeps ref.py's GOTO and STOP bit for bit, and its events follow Ball::SetMode.
"""

from __future__ import annotations

import math
import struct

import pytest

from movement_cases import cases as m2_cases
from movement_cases import reference as m2_reference
from exact_ballpark.modes import AU, FOLLOW, ORBIT, WARP, ModeBallpark
from exact_ballpark.ref import GOTO, STOP

TEN_BILLION = 10000000000.0

FOLLOW_STOPPED = [            # test_follow.py, TestFollow.test_follow_stopped_ball
    (0.10547716886968704, 0.21095433773937408, 0.3164315066090625),
    (0.41030556327284373, 0.8206111265456875, 1.2309166898185306),
    (0.8981544513244621, 1.7963089026489243, 2.694463353973384),
    (1.5540309048233925, 3.108061809646785, 4.662092714470175),
    (2.364170207183283, 4.728340414366566, 7.092510621549848),
    (3.3159352390795633, 6.631870478159127, 9.94780571723868),
    (4.397724106363411, 8.795448212726821, 13.193172319090223),
    (5.59888533504116, 11.19777067008232, 16.796656005123477),
    (6.909640013429743, 13.819280026859486, 20.72892004028922),
    (8.321010312379672, 16.642020624759343, 24.963030937139006),
]
FOLLOW_MOVING = [             # test_follow.py, TestFollow.test_follow_moving_ball
    (0.3946594280372685, 0.0, 0.0),
    (1.5352202516956626, 0.0006394210607791958, 0.0012788421215583917),
    (3.360538351919784, 0.00437328037960026, 0.00874656075920052),
    (5.814319316187089, 0.01591817574944254, 0.03183635149888508),
    (8.844475669456193, 0.042139697272792834, 0.08427939454558567),
    (12.402376916827746, 0.09207177010911027, 0.18414354021822055),
    (16.441932909387862, 0.17705502464375017, 0.35411004928750034),
    (20.918410162782422, 0.31098291710272036, 0.6219658342054407),
    (25.786827618186656, 0.5106319284615736, 1.0212638569231471),
    (30.99971869544917, 0.796013087027414, 1.592026174054828),
]
OLD_ORBIT = [                 # test_orbit.py, TestOldOrbit.test_orbit_ball
    (5.0, 0.0, 0.3946594280372685),
    (4.968965087530276, -0.014060509698578017, 1.533749375235321),
    (4.7948197558341885, -0.09342506603769896, 3.3353413772468783),
    (4.303462228867957, -0.3182911817228117, 5.664861650309143),
    (3.341010762290748, -0.7601346870799338, 8.325917825178466),
    (1.826888038965309, -1.4570571575293851, 11.103680716692217),
    (-0.25072581197636695, -2.415534207332392, 13.808614450861453),
    (-2.860769892110699, -3.6221094985835713, 16.289471089613752),
    (-5.948838546575331, -5.052297043579673, 18.42935795458875),
    (-9.447930869309866, -6.675507070626672, 20.138672333159825),
]
WARP_TO = [                   # test_warp.py, TestWarp.test_warpto
    (1.0639340706602571, 2.1278681413205143, 3.1918022119807685),
    (2.248703156860341, 4.497406313720682, 6.746109470581021),
    (3.544408527173739, 7.088817054347478, 10.633225581521215),
    (4.941962348268478, 9.883924696536956, 14.825887044805434),
    (6.433021256625409, 12.866042513250818, 19.299063769876224),
    (8.00992537202119, 16.01985074404238, 24.029776116063566),
    (9.665642306989863, 19.331284613979726, 28.99692692096958),
    (11.393715762995505, 22.78743152599101, 34.18114728898651),
    (13.188218337575332, 26.376436675150664, 39.56465501272599),
    (15.043708197493105, 30.08741639498621, 45.13112459247932),
]


def space_ball(park: ModeBallpark, bid: int, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)) -> None:
    """helpers.create_space_ball: AddBall with the POPO defaults, free, then the setters in the helper's order."""
    park.add_ball(bid, 0.0, 0.0, 0.0, pos, vel, 0.0, 0.0, is_free=True, is_massive=True)
    park.set_max_velocity(bid, 10.0)
    park.set_agility(bid, 0.9)
    park.set_mass(bid, 13000000.0)
    park.set_radius(bid, 2.0)
    park.set_speed_fraction(bid, 0.95)


def popo_balls(park: ModeBallpark, n: int, pos=(TEN_BILLION, TEN_BILLION, TEN_BILLION)) -> list:
    """helpers.add_ball_to_park with the POPO defaults (BallparkTestCase.add_balls): ids 1..n, not free."""
    for i in range(n):
        park.add_ball(i + 1, 0.0, 0.0, 0.0, pos, (0.0, 0.0, 0.0), 0.0, 0.0, is_free=False)
    return [park.balls[i + 1] for i in range(n)]


def trajectory(park: ModeBallpark, bid: int, ticks: int) -> list:
    out = []
    for _ in range(ticks):
        park.evolve()
        out.append(tuple(park.balls[bid].pos))
    return out


# ---- trajectories, bit for bit ----

def test_follow_of_a_stopped_ball_reproduces_destiny():          # MOV-714, MOV-715 (the default range 1.0)
    park = ModeBallpark()
    space_ball(park, 1)
    space_ball(park, 2, pos=(100.0, 200.0, 300.0))
    park.follow_ball(1, 2)
    assert trajectory(park, 1, 10) == FOLLOW_STOPPED


def test_follow_of_a_moving_ball_reproduces_destiny():
    park = ModeBallpark()
    space_ball(park, 1)
    space_ball(park, 2, pos=(100.0, 0.0, 0.0))
    park.follow_ball(1, 2)
    park.goto_point(2, 0.0, 100.0, 200.0)
    assert trajectory(park, 1, 10) == FOLLOW_MOVING


def test_old_style_orbit_reproduces_destiny():                   # MOV-716 (default), MOV-717, MOV-718
    park = ModeBallpark()                                        # helpers.create_space_ball.next_object_id = 1
    assert park.current_time == 0 and park.use_new_orbit is False
    space_ball(park, 1)
    space_ball(park, 2, pos=(5.0, 0.0, 0.0))
    park.orbit(2, 1, 1.0)
    assert trajectory(park, 2, 10) == OLD_ORBIT


def test_warp_alignment_reproduces_destiny():                    # TRV-703, TRV-704 (defaults), TRV-705
    park = ModeBallpark()
    space_ball(park, 1)
    park.balls[1].vel = [1.0, 2.0, 3.0]                          # ball.vx = 1.0, ... (the velocity attributes)
    park.warp_to(1, 100000.0, 200000.0, 300000.0)
    got = []
    for _ in range(10):
        park.evolve()
        assert park.balls[1].mode == WARP
        got.append(tuple(park.balls[1].pos))
    assert got == WARP_TO


# ---- the new orbit: Destiny's bounds (test_orbit.py, TestNewOrbit) ----

def new_orbit_park(orbiter_x: float = 10.0) -> ModeBallpark:
    park = ModeBallpark(use_new_orbit=True)                      # enable_new_orbit(); next_object_id = 1
    space_ball(park, 1)
    space_ball(park, 2, pos=(orbiter_x, 0.0, 0.0))
    return park


def geometry(park: ModeBallpark, relative: bool = False) -> tuple:
    a, b = park.balls[2], park.balls[1]
    d = [a.pos[i] - b.pos[i] for i in range(3)]
    v = [a.vel[i] - (b.vel[i] if relative else 0.0) for i in range(3)]
    return math.sqrt(sum(x * x for x in d)), math.sqrt(sum(x * x for x in v)), abs(sum(v[i] * d[i] for i in range(3)))


def test_new_orbit_of_a_stopped_ball_from_the_right_distance():  # test_orbit_stopped_ball_starting_right_distance
    park = new_orbit_park()
    for orbit in [100.0, 500.0, 1000.0, 2000.0]:
        expected = orbit + 2.0 + 2.0
        park.balls[2].pos, park.balls[2].vel = [expected, 0.0, 0.0], [0.0, 0.0, 1.0]   # SetBallPosition, -Velocity
        park.orbit(2, 1, orbit)
        for _ in range(100):
            park.evolve()
            distance, speed, radial = geometry(park)
            assert abs(distance - expected) <= 20
            assert radial < 0.2 * speed * distance


@pytest.mark.parametrize("start,vel,band,cone", [(1.5, (-1.0, 0.0, 1.0), 10, 0.8), (0.5, (0.0, 0.0, 1.0), 20, 0.85)])
def test_new_orbit_of_a_stopped_ball_from_outside_and_inside(start, vel, band, cone):
    """test_orbit_stopped_ball_starting_outside and _inside: close in at an angle, then hold the distance."""
    orbit = 1000.0
    expected = orbit + 2.0 + 2.0
    park = new_orbit_park()
    park.balls[2].pos, park.balls[2].vel = [start * expected, 0.0, 0.0], list(vel)
    park.orbit(2, 1, orbit)
    previous, iorbit = park.balls[2].pos[0], 2000
    for i in range(2000):
        park.evolve()
        distance, speed, radial = geometry(park)
        approaching = distance > expected + band if start > 1 else distance < expected - band
        if approaching:
            assert (distance < previous) if start > 1 else (distance > previous)
            previous = distance
            assert radial < cone * speed * distance
        else:
            iorbit = min(iorbit, i)
            assert abs(distance - expected) <= band
    assert iorbit < 2000


def test_new_orbit_of_a_moving_ball():                           # test_orbit_moving_ball_starting_right_distance
    orbit = 2000.0
    expected = orbit + 2.0 + 2.0
    park = new_orbit_park()
    park.set_max_velocity(2, 200.0)
    park.set_max_velocity(1, 50.0)
    park.goto_direction(1, 1.0, 1.0, 1.0)
    park.balls[2].pos, park.balls[2].vel = [expected, 0.0, 0.0], [0.0, 0.0, 1.0]
    park.orbit(2, 1, orbit)
    for _ in range(200):
        park.evolve()
        distance, speed, radial = geometry(park, relative=True)
        assert abs(distance - expected) <= 700
        assert radial < 0.4 * speed * distance


# ---- commands (test_movement_controls.py) ----

@pytest.mark.parametrize("command", ["follow_ball", "orbit"])
def test_follow_and_orbit_set_the_target_and_the_range(command):  # TestFollow.test_follow_ball(_with_range), TestOrbit
    for args, want in (((), 1.0), ((42.0,), 42.0)):
        park = ModeBallpark()
        src, dst = popo_balls(park, 2)
        getattr(park, command)(src.id, dst.id, *args)
        assert (src.follow_id, src.follow_range, src.mode) == (dst.id, want, FOLLOW if command == "follow_ball"
                                                                else ORBIT)
        assert src.id in dst.followers


@pytest.mark.parametrize("command", ["follow_ball", "orbit"])
@pytest.mark.parametrize("case", ["self", "nan", "moribund", "cloaked", "bubble"])
def test_follow_and_orbit_refuse(command, case):
    """test_ball_can_not_follow_self, _moribund_ball; test_can_not_orbit_self, _at_nan_range, _cloaked_ball,
    _ball_in_different_bubble. RemoveBall, CloakBall and InitializeBubbles are later packages: the flag each sets is
    set here directly."""
    park = ModeBallpark()
    src, dst = popo_balls(park, 2)
    rng = 1.0
    if case == "self":
        dst = src
    elif case == "nan":
        rng = float("nan")
    elif case == "moribund":
        dst.is_moribund = True
    elif case == "cloaked":
        dst.is_cloaked = 1
    else:
        dst.bubble = 3
    getattr(park, command)(src.id, dst.id, rng)
    assert src.follow_id != dst.id or case == "self"
    assert (src.follow_id, src.follow_range, src.mode) == (0, 10.0, STOP)
    assert src.id not in dst.followers


@pytest.mark.parametrize("command", ["follow_ball", "orbit"])
def test_stop_ends_following(command):                           # TestStop.test_stop_following_ball, _orbiting_ball
    park = ModeBallpark()
    first, second = popo_balls(park, 2)
    getattr(park, command)(first.id, second.id)
    assert first.effect_stamp == 0
    park.stop(first.id)
    assert (first.mode, first.follow_id, first.follow_range) == (STOP, 0, 0.0)
    assert first.id not in second.followers


def test_warp_to_stores_the_destination_and_its_strange_places():  # TestWarpTo (TRV-703, TRV-704)
    park = ModeBallpark()
    src, = popo_balls(park, 1)
    park.warp_to(src.id, 1, 2, 3)
    assert src.mode == WARP and src.goto == [1.0, 2.0, 3.0]
    assert src.follow_id == struct.unpack("!q", struct.pack("!d", 20000.0))[0] and src.owner_id == 20
    park.warp_to(src.id, 1, 2, 3, 42.0, 10000)                   # the same point while warping: a no-op (TRV-702)
    assert src.owner_id == 20
    park2 = ModeBallpark()
    src2, = popo_balls(park2, 1)
    park2.warp_to(src2.id, 1, 2, 3, 42.0, 10000)                 # test_minrange_and_warpfactor_are_stored_in_...
    assert src2.follow_id == struct.unpack("!q", struct.pack("!d", 42.0))[0] and src2.owner_id == 10000
    assert (src2.effect_stamp, src2.follow_range) == (-1, 0.0)


def test_a_warp_to_a_point_too_close_is_a_goto():               # test_if_warp_location_is_too_close_we_goto_it_instead
    park = ModeBallpark()
    src, = popo_balls(park, 1, pos=(0.0, 0.0, 0.0))
    park.warp_to(src.id, 1, 2, 3)
    assert src.mode == GOTO and src.goto == [1.0, 2.0, 3.0]
    assert park.events[-1] == (0, "OnExitWarp", src.id, 1, 0)   # TRV-701


def test_entity_warp_in():                                       # TestEntityWarpIn (TRV-715)
    park = ModeBallpark()
    src, = popo_balls(park, 1)
    park.entity_warp_in(src.id, 0.0, 0.0, 0.0, 1)                # test_warps_immediately
    assert src.mode == WARP and src.effect_stamp == 0
    park = ModeBallpark()
    src, = popo_balls(park, 1, pos=(0.0, 0.0, 0.0))
    park.entity_warp_in(src.id, TEN_BILLION, 0.0, 0.0, 1)        # test_velocity_is_approximately_one_au
    assert src.vel == [AU, 0.0, 0.0]
    park = ModeBallpark()
    src, = popo_balls(park, 1, pos=(1.0, 0.0, 0.0))
    park.goto_point(src.id, 0.0, 0.0, 0.0)
    park.entity_warp_in(src.id, 0.0, 0.0, 0.0, 1)                # test_ball_in_goto_mode_does_not_warp
    assert src.mode != WARP


@pytest.mark.parametrize("d,want", [((1, 0, 0), (1e17, 0, 0)), ((-1, 0, 0), (-1e17, 0, 0)), ((0, 1, 0), (0, 1e17, 0)),
                                    ((0, -1, 0), (0, -1e17, 0)), ((0, 0, 1), (0, 0, 1e17)),
                                    ((0, 0, -1), (0, 0, -1e17))])
def test_goto_direction_and_point(d, want):      # GotoDirection.test_go_right to test_go_backward; GotoPoint too
    park = ModeBallpark()
    src, = popo_balls(park, 1, pos=(0.0, 0.0, 0.0))
    park.goto_direction(src.id, *d)
    assert src.mode == GOTO and src.goto == list(map(float, want)) and src.speed_fraction == 1.0


# ---- GOTO and STOP are ref.py's, bit for bit; events follow Ball::SetMode ----

@pytest.mark.parametrize("seed", [1, 2])
def test_goto_and_stop_are_ref_py_bit_for_bit(seed):
    """M2's generated cases (tests/movement_cases.py) through the mode ballpark give ref.py's rows exactly."""
    for c in m2_cases(seed):
        cid, friction, tick, steps, mode, mass, agility, maxvel, sf = c[:9]
        park = ModeBallpark(friction=friction, tick_interval_ms=tick)
        b = park.add_ball(1, mass, 0.0, maxvel, c[9:12], c[12:15], agility, sf)
        b.mode, b.goto = mode, list(c[15:18])
        got = []
        for s in range(1, steps + 1):
            park.evolve()
            got.append((cid, s, *b.pos, *b.vel))
        assert got == m2_reference(c)


def test_mode_changes_raise_events():                            # MOV-735
    park = ModeBallpark()
    space_ball(park, 1)
    space_ball(park, 2, pos=(1e3, 0.0, 0.0))
    park.warp_to(1, 1e9, 0.0, 0.0)
    park.stop(1)
    park.orbit(1, 2)
    park.follow_ball(1, 2)
    names = [(e[1], e[3], e[4]) for e in park.events if e[2] == 1]
    assert names == [("DoModeChange", STOP, WARP), ("OnExitWarp", 0, 0), ("DoModeChange", WARP, STOP),
                     ("DoModeChange", STOP, ORBIT), ("DoModeChange", ORBIT, STOP), ("DoModeChange", STOP, FOLLOW)]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance row 1: the scalar movement reference against Destiny's own
published test expectations.

The expected values are copied from carbonengine/destiny at 114e89fa (MIT licence, (c) CCP Games):
python/destiny/test/ballpark/evolve/test_goto.py and test_stop.py, with the ball of python/destiny/test/helpers.py
(create_space_ball: max velocity 10, agility 0.9, mass 13,000,000, radius 2, speed fraction 0.95). Destiny's own test
compares to four decimal places; this one asks for bit equality, which the transcription meets.
"""

from __future__ import annotations

import pytest

from exact_ballpark.ref import Ballpark

GOTO_DIRECTION = [
    (0.10547716886968704, 0.21095433773937408, 0.31643150660906116),
    (0.4103055632728434, 0.8206111265456868, 1.2309166898185293),
    (0.8981544513244618, 1.7963089026489236, 2.6944633539733838),
    (1.5540309048233936, 3.108061809646787, 4.662092714470173),
    (2.364170207183284, 4.728340414366568, 7.0925106215498435),
    (3.315935239079564, 6.631870478159128, 9.947805717238685),
    (4.397724106363411, 8.795448212726821, 13.19317231909023),
    (5.598885335041164, 11.197770670082328, 16.796656005123484),
    (6.909640013429747, 13.819280026859493, 20.728920040289236),
    (8.321010312379675, 16.64202062475935, 24.96303093713902),
]
GOTO_POINT = [
    (0.10547716886968671, 0.21095433773937342, 0.31643150660906116),
    (0.4103055632728427, 0.8206111265456854, 1.2309166898185306),
    (0.898154451324462, 1.796308902648924, 2.694463353973384),
    (1.5540309048233925, 3.108061809646785, 4.662092714470175),
    (2.364170207183283, 4.728340414366566, 7.092510621549848),
    (3.3159352390795633, 6.631870478159127, 9.947805717238683),
    (4.397724106363411, 8.795448212726821, 13.193172319090227),
    (5.598885335041161, 11.197770670082322, 16.79665600512348),
    (6.909640013429744, 13.819280026859488, 20.728920040289225),
    (8.321010312379672, 16.642020624759343, 24.96303093713901),
    (9.739446987259553, 19.478893974519107, 29.218340961778658),
    (11.061301666724546, 22.122603333449092, 33.18390500017364),
    (12.271662033494039, 24.543324066988077, 36.814986100482116),
    (13.312148751580072, 26.624297503160143, 39.93644625474022),
    (14.096228724360666, 28.19245744872133, 42.288686173082),
    (14.608085488313941, 29.216170976627883, 43.82425646494182),
    (14.870019404144168, 29.740038808288336, 44.610058212432506),
    (14.902504000487617, 29.805008000975235, 44.70751200146285),
]
STOP_VELOCITIES = [
    (9.180806045144438, 15.959158171933325, 27.542418135433316),
    (8.428719963856068, 12.73473647783931, 25.286159891568197),
    (7.738244319699939, 10.161783686386327, 23.21473295909981),
    (7.104332022910581, 8.108675658000553, 21.312996068731735),
    (6.522349438265067, 6.470381869546817, 19.567048314795194),
    (5.9880425151368355, 5.163092384445365, 17.9641275454105),
    (5.497505692155016, 4.119930400983397, 16.49251707646504),
    (5.047153349175273, 3.2875310463325356, 15.141460047525813),
    (4.633693597887935, 2.6233113981781195, 13.901080793663798),
    (4.254104219483663, 2.0932920768880083, 12.762312658450982),
]


def space_ball(park: Ballpark, bid: int = 1, vel=(0.0, 0.0, 0.0)) -> None:
    """helpers.create_space_ball: AddBall with the POPO defaults, then the property setters in the helper's order."""
    park.add_ball(bid, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), vel, 0.0, 0.0)
    park.set_max_velocity(bid, 10.0)
    park.set_agility(bid, 0.9)
    park.set_mass(bid, 13000000.0)
    park.set_radius(bid, 2.0)
    park.set_speed_fraction(bid, 0.95)


def run(command, ticks: int, read) -> list:
    park = Ballpark()
    space_ball(park)
    command(park)
    out = []
    for _ in range(ticks):
        park.evolve()
        out.append(tuple(read(park.balls[1])))
    return out


@pytest.mark.parametrize("name,command,ticks,read,expected", [
    ("goto_direction", lambda p: p.goto_direction(1, 10, 20, 30), 10, lambda b: b.pos, GOTO_DIRECTION),
    ("goto_point", lambda p: p.goto_point(1, 10, 20, 30), 18, lambda b: b.pos, GOTO_POINT),
])
def test_goto_reproduces_destiny_bit_for_bit(name, command, ticks, read, expected):
    assert run(command, ticks, read) == expected


def test_stop_reproduces_destiny_bit_for_bit():
    park = Ballpark()
    space_ball(park, vel=(10.0, 20.0, 30.0))
    park.stop(1)
    got = []
    for _ in range(10):
        park.evolve()
        got.append(tuple(park.balls[1].vel))
    assert got == STOP_VELOCITIES


def test_a_stopped_ball_at_rest_stays_at_the_origin():
    park = Ballpark()
    space_ball(park)
    for _ in range(10):
        park.evolve()
        assert park.balls[1].pos == [0.0, 0.0, 0.0]


def test_goto_overshoots_its_point():        # MOV-707
    pos = run(lambda p: p.goto_point(1, 10, 20, 30), 18, lambda b: b.pos)
    assert pos[10][0] < 10 < pos[11][0] and pos[17][0] > 14.9

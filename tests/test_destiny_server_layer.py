# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against published test vectors for Destiny's server layer: this port's
transcription (exact_ballpark.server) against Destiny's own tests, transcribed.

Copied from carbonengine/destiny at 114e89fa (MIT licence, (c) CCP Games), python/destiny/test/:
- net/server/helpers.py (add_ball_to_park, TestBallInfo, TestNetwork, TestCharacterInterests, TestClientInterests,
  DestinyTestCase), net/server/test_actions.py (TestActions, 75 tests), net/server/test_ticker.py (TestTicker, 19),
  net/server/test_updatebatcher.py (TestUpdateBatcher, 14) and net/server/test_bubbleupdater.py (TestBubbleUpdater, 4):
  every server test, in Destiny's order;
- the ballpark tests of the C++ thunkers transcribed in exact_ballpark.server: ballpark/test_lifecycle_management.py's
  TestAdditionsAndDeletions (4) and TestBubbleTransitions (3), and ballpark/test_stream_read_write.py's
  TestWriteBallsToStream (5) and TestWriteFullStateToStream (2), with test/helpers.py's BallPOPO, add_ball_to_park (here
  popo_add_ball_to_park), BallparkTestCase, BallparkWrapped and BallparkWrappedTestCase.
Each keeps Destiny's values, expected tuples, byte strings, orders and signals, and names its source. Destiny's tests
run on `destiny.Ballpark()` (the C++ ballpark; here `DestinyPark` over this port's ballpark) with `blue.MemStream` and
`blue.marshal` (here `MemStream` and `marshal_save` / `marshal_load`). Changes from Destiny's text, all mechanical and
checked by comparing the syntax trees: the imports point at this port; the helper classes named Test* carry
`__test__ = False`; the server tests' BALL_ID_n of test_ticker.py, test_updatebatcher.py and test_bubbleupdater.py are
T_/U_/B_BALL_ID_n here (same values). Where a test needs something this port's ballpark does not have (ClientBall's
members centerDist and surfaceDist), it is skipped with the reason
(unittest.SkipTest), never dropped: add_ball_to_park turns this port's NotImplementedError into that skip, and a
guard line names the missing piece where a test starts.

After Destiny's tests, this port's checks (test_lane_*) cover what Destiny's tests do not: the pre-tick's first step (the
follower clean-up queues its Stops before the stamp moves, behind earlier commands: TIME-712, MOV-731), events of a ball
in no known bubble and of a dead bubble being dropped at the first gate (SEN-712), the ball stream's round trip for
every mode it writes, Actions' read of park._currentTime, and SetMaxAngularVelocity never being applied. Then planted
faults, one per clause, each of which must be caught by the Destiny tests or this port's checks named beside it. This port
checks and faults check the transcription; they are not test vectors.
"""

from __future__ import annotations

import unittest
import unittest.mock as mock
from collections import defaultdict

import pytest

from exact_ballpark import server as srv
from exact_ballpark.server import (Actions, BallInfoInterface, BubbleUpdater, CharacterInterestsInterface,
                                 ClientInterestsInterface, ClientUpdateCountThisTick, NetworkInterface,
                                 ParkUpdateBatcher, Ticker, disable_dynamical_orientation, enable_dynamical_orientation,
                                 is_dynamical_orientation_enabled, reset_settings_to_default)


class destiny:                                              # the names the tests take from the destiny module
    Ballpark = srv.DestinyPark
    DSTNORMALCLOAK = srv.DSTNORMALCLOAK
    DSTBALL_GOTO = srv.DSTBALL_GOTO
    DSTBALL_STOP = srv.DSTBALL_STOP


class blue:                                                 # blue.MemStream and blue.marshal
    MemStream = srv.MemStream

    class marshal:
        Save = staticmethod(srv.marshal_save)
        Load = staticmethod(srv.marshal_load)


def _require(park, method: str) -> None:
    """Skip when this port's ballpark lacks a method the test needs."""
    if not hasattr(park.lane, method):
        raise unittest.SkipTest(f"this port's ballpark ({type(park.lane).__name__}) has no {method}()")


def _require_ball_field(park, field: str) -> None:
    for b in park.lane.balls.values():
        if not hasattr(b, field):
            raise unittest.SkipTest(f"this port's ball ({type(b).__name__}) has no {field!r}")


# ======================================================================== helpers.py (test/net/server/helpers.py)

def add_ball_to_park(
        park,
        ball_id=-1,
        mass=1,
        radius=1.0,
        max_velocity=1.0,
        is_free=True,
        is_global=False,
        is_massive=True,
        is_interactive=True,
        is_space_junk=False,
        x=0,
        y=0,
        z=0,
        vx=0,
        vy=0,
        vz=0,
        agility=1.0,
        speed_fraction=1.0,
        maxAngularSpeed=1.0,
        angularAgility=0.9
):
    """helpers.py:18-58."""
    args = (ball_id,
            mass,
            radius,
            max_velocity,
            is_free,
            is_global,
            is_massive,
            is_interactive,
            is_space_junk,
            x,
            y,
            z,
            vx,
            vy,
            vz,
            agility,
            speed_fraction,)
    dynamical_args = args + (maxAngularSpeed, angularAgility)
    try:
        if is_dynamical_orientation_enabled():
            return park.AddBall(*dynamical_args)
        return park.AddBall(*args)
    except NotImplementedError as exc:                      # (lane) e.g. local ball ids (-1) on SpaceBallpark
        raise unittest.SkipTest(f"this port's ballpark cannot add this ball: {exc}")


class TestBallInfo(BallInfoInterface):
    """helpers.py:61-73."""
    __test__ = False

    def __init__(self):
        self._characters_for_ball = defaultdict(list)

    def get_character_for_ball(self, ball_id):
        return self._characters_for_ball[ball_id][0]

    def get_characters_for_ball(self, ball_id):
        return self._characters_for_ball[ball_id]

    def set_characters_for_ball(self, ball_id, characters):
        self._characters_for_ball[ball_id] = characters


class TestNetwork(NetworkInterface):
    """helpers.py:76-90."""
    __test__ = False

    def __init__(self):
        self.singlecasts = []
        self.narrowcasts = []

    def narrowcast(self, updates):
        self.narrowcasts += updates

    def singlecast(self, updates):
        self.singlecasts += updates

    def clear(self):
        self.singlecasts = []
        self.narrowcasts = []


class TestCharacterInterests(CharacterInterestsInterface):
    """helpers.py:93-111."""
    __test__ = False

    def __init__(self):
        self._ball_to_char = defaultdict(list)
        self._char_to_client = {}

    def add_interest(self, ball_id, char_id):
        self._ball_to_char[ball_id].append(char_id)

    def map_to_client(self, char_id, client_id):
        self._char_to_client[char_id] = client_id

    def get_interested_character_ids_for_ball(self, ball_id):
        return self._ball_to_char[ball_id]

    def remove_character_interest(self, ball_id, character_id):
        self._ball_to_char[ball_id].remove(character_id)

    def get_client_id_for_character(self, character_id):
        return self._char_to_client.get(character_id, None)


class TestClientInterests(ClientInterestsInterface):
    """helpers.py:114-133 (its get_all_interested_ball_ids returns nothing and its remove_client_interest calls
    .remove on a defaultdict: kept as written; no test uses them)."""
    __test__ = False

    def __init__(self):
        self._ball_id_to_client_ids = defaultdict(list)

    def add_client_interest(self, ball_id, client_id):
        self._ball_id_to_client_ids[ball_id].append(client_id)

    def get_all_interested_ball_ids(self):
        self._ball_id_to_client_ids.keys()

    def get_interested_client_ids_for_ball(self, ball_id):
        return self._ball_id_to_client_ids[ball_id]

    def remove_client_interest(self, ball_id, client_id):
        self._ball_id_to_client_ids.remove(ball_id)

    def has_client_interest(self, ball_id):
        return bool(self._ball_id_to_client_ids[ball_id])


class DestinyTestCase(unittest.TestCase):
    """helpers.py:137-230."""

    def assertBallsEqual(self, first, second, ignore_bubble_id=False):
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.mass, second.mass)
        self.assertEqual(first.radius, second.radius)
        self.assertEqual(first.maxVelocity, second.maxVelocity)
        self.assertEqual(first.isFree, second.isFree)
        self.assertEqual(first.isGlobal, second.isGlobal)
        self.assertEqual(first.isMassive, second.isMassive)
        self.assertEqual(first.isInteractive, second.isInteractive)
        self.assertEqual(first.isCloaked, second.isCloaked)
        self.assertEqual(first.isMoribund, second.isMoribund)
        self.assertEqual(first.harmonic, second.harmonic)
        self.assertEqual(first.corporationID, second.corporationID)
        self.assertEqual(first.allianceID, second.allianceID)
        self.assertEqual(first.x, second.x)
        self.assertEqual(first.y, second.y)
        self.assertEqual(first.z, second.z)
        self.assertEqual(first.vx, second.vx)
        self.assertEqual(first.vy, second.vy)
        self.assertEqual(first.vz, second.vz)
        self.assertEqual(first.Agility, second.Agility)
        self.assertEqual(first.speedFraction, second.speedFraction)
        self.assertEqual(first.mode, second.mode)
        self.assertEqual(first.gotoX, second.gotoX)
        self.assertEqual(first.gotoY, second.gotoY)
        self.assertEqual(first.gotoZ, second.gotoZ)
        self.assertEqual(first.followId, second.followId)
        self.assertEqual(first.followRange, second.followRange)
        self.assertEqual(first.ownerId, second.ownerId)
        self.assertEqual(first.effectStamp, second.effectStamp)
        if not ignore_bubble_id:
            self.assertEqual(first.newBubbleId, second.newBubbleId)
            self.assertEqual(first.oldBubbleId, second.oldBubbleId)
        self.assertEqual(first.formationID, second.formationID)
        self.assertListEqual(list(first.miniBalls), list(second.miniBalls))
        self.assertListEqual(list(first.miniCapsules), list(second.miniCapsules))
        self.assertListEqual(list(first.miniBoxes), list(second.miniBoxes))

    def ballsAreEqual(self, first, second, ignore_bubble_id=False):
        """helpers.py:184-224, with Destiny's slips kept: maxVelocity is compared with radius and speedFraction with
        mass (both hold for add_ball_to_park's defaults)."""
        if not ignore_bubble_id and (
                first.newBubbleId != second.newBubbleId or
                first.oldBubbleId != second.oldBubbleId):
            return False

        return (
            first.id == second.id and
            first.mass == second.mass and
            first.radius == second.radius and
            first.maxVelocity == second.radius and
            first.isFree == second.isFree and
            first.isGlobal == second.isGlobal and
            first.isMassive == second.isMassive and
            first.isInteractive == second.isInteractive and
            first.isCloaked == second.isCloaked and
            first.isMoribund == second.isMoribund and
            first.harmonic == second.harmonic and
            first.corporationID == second.corporationID and
            first.allianceID == second.allianceID and
            first.x == second.x and
            first.y == second.y and
            first.z == second.z and
            first.vx == second.vx and
            first.vy == second.vy and
            first.vz == second.vz and
            first.yaw == second.yaw and
            first.pitch == second.pitch and
            first.roll == second.roll and
            first.Agility == second.Agility and
            first.speedFraction == second.mass and
            first.speedFraction == second.speedFraction and
            first.mode == second.mode and
            first.gotoX == second.gotoX and
            first.gotoY == second.gotoY and
            first.gotoZ == second.gotoZ and
            first.followId == second.followId and
            first.followRange == second.followRange and
            first.ownerId == second.ownerId and
            first.effectStamp == second.effectStamp and
            first.formationID == second.formationID and
            list(first.miniBalls) == list(second.miniBalls) and
            list(first.miniCapsules) == list(second.miniCapsules) and
            list(first.miniBoxes) == list(second.miniBoxes)
        )

    def read_balls_from_stream_into_park(self, ball_stream):
        stream = blue.MemStream()
        stream.Write(ball_stream)
        park = destiny.Ballpark()
        park.ReadFullStateFromStream(stream)
        return park


# ============================================================================================== test_actions.py

BALL_ID_1 = 1000000000001
BALL_ID_2 = 1000000000002
BALL_ID_3 = 1000000000003
INITIAL_TIMESTAMP = 0

CORP_ID_1 = 98000001
ALLIANCE_ID_1 = 99000001


class TestActions(DestinyTestCase):
    """test_actions.py:21-555."""

    def setUp(self):
        self._park = destiny.Ballpark()
        self._park.isMaster = True
        self._actions = Actions(self._park)

    def tearDown(self):
        self._park.ClearAll()
        del self._park
        del self._actions
        reset_settings_to_default()

    def assertHistoryEmpty(self):
        history = self._actions.flush_history()
        self.assertListEqual(
            [],
            history
        )

    def assertSingleActionHistory(self, ball_id, expected_action):
        history = self._actions.flush_history()
        self.assertListEqual(
            [(ball_id, INITIAL_TIMESTAMP, expected_action)],
            history
        )

    def assertMultiActionHistory(self, ball_action_list):
        history = self._actions.flush_history()
        self.assertListEqual(ball_action_list, history)

    def assertBallAddedToClientParks(self, ball, timestamp):
        history = self._actions.flush_history()
        for entry in history:
            ball_id, entry_timestamp, action = entry
            if ball_id != ball.id:
                continue
            if entry_timestamp != timestamp:
                continue
            actionName, parameters = action
            if actionName != "AddBallsToPark":
                continue
            self.assertEqual(1, len(parameters))
            bytes = parameters[0]
            park = self.read_balls_from_stream_into_park(bytes)
            loaded_ball = park.balls[ball.id]
            if self.ballsAreEqual(ball, loaded_ball, ignore_bubble_id=True):
                return
        self.assertFalse(True, "Ball not added to park (or was added at the wrong time).")

    def assertActionInHistory(self, ball_id, action, timestamp=INITIAL_TIMESTAMP):
        history = self._actions.flush_history()
        for entry in history:
            entry_ball_id, entry_timestamp, entry_action = entry
            if entry_ball_id != ball_id:
                continue
            if entry_action != action:
                continue
            if entry_timestamp != timestamp:
                continue
            return
        self.assertFalse(
            "Entry for action %s not found in history for ball %s at time %s" % (action, ball_id, timestamp)
        )

    def test_go_to_point(self):
        x, y, z = 100, 200, 300
        self._actions.go_to_point(BALL_ID_1, x, y, z)
        expected_action = ('GotoPoint', (BALL_ID_1, 100, 200, 300))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_duplicate_subsequent_actions_are_ignored(self):
        x, y, z = 100, 200, 300
        self._actions.go_to_point(BALL_ID_1, x, y, z)
        self._actions.go_to_point(BALL_ID_1, x, y, z)
        expected_action = ('GotoPoint', (BALL_ID_1, 100, 200, 300))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_duplicate_subsequent_actions_with_different_timestamps_are_not_ignored(self):
        x, y, z = 100, 200, 300
        self._actions.go_to_point(BALL_ID_1, x, y, z)
        self._actions.set_stamp_for_system(INITIAL_TIMESTAMP + 1)
        self._actions.go_to_point(BALL_ID_1, x, y, z)
        expected_action = ('GotoPoint', (BALL_ID_1, 100, 200, 300))
        self.assertMultiActionHistory(
            [
                (BALL_ID_1, INITIAL_TIMESTAMP, expected_action),
                (BALL_ID_1, INITIAL_TIMESTAMP + 1, expected_action)
            ]
        )

    def test_duplicate_non_subsequent_actions_are_not_ignored(self):
        x, y, z = 100, 200, 300
        self._actions.go_to_point(BALL_ID_1, x, y, z)
        self._actions.stop(BALL_ID_1)
        self._actions.go_to_point(BALL_ID_1, x, y, z)
        goto_action = ('GotoPoint', (BALL_ID_1, x, y, z))
        stop_action = ('Stop', (BALL_ID_1,))
        self.assertMultiActionHistory(
            [
                (BALL_ID_1, INITIAL_TIMESTAMP, goto_action),
                (BALL_ID_1, INITIAL_TIMESTAMP, stop_action),
                (BALL_ID_1, INITIAL_TIMESTAMP, goto_action)
            ]
        )

    def test_go_to_direction(self):
        x, y, z = 100, 200, 300
        self._actions.go_to_direction(BALL_ID_1, x, y, z)
        expected_action = ('GotoDirection', (BALL_ID_1, x, y, z))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_orbit(self):
        range_meters = 123
        self._actions.orbit(BALL_ID_1, BALL_ID_2, range_meters)
        expected_action = ("Orbit", (BALL_ID_1, BALL_ID_2, range_meters))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_troll(self):
        tick_delay = 42
        self._actions.set_ball_troll(BALL_ID_1, tick_delay)
        expected_action = ("SetBallTroll", (BALL_ID_1, tick_delay))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_velocity(self):
        vx, vy, vz = (333.3, 666.6, 999.9)
        self._actions.set_ball_velocity(BALL_ID_1, vx, vy, vz)
        expected_action = ("SetBallVelocity", (BALL_ID_1, vx, vy, vz))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_angular_velocity(self):
        enable_dynamical_orientation()
        wx, wy, wz = 1.0, 2.0, 3.0
        self._actions.set_ball_angular_velocity(BALL_ID_1, wx, wy, wz)
        expected_action = ("SetBallAngularVelocity", (BALL_ID_1, wx, wy, wz))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_angular_velocity_raises_when_dynamical_orientation_disabled(self):
        disable_dynamical_orientation()
        wx, wy, wz = 1.0, 2.0, 3.0
        self.assertRaises(RuntimeError, self._actions.set_ball_angular_velocity, BALL_ID_1, wx, wy, wz)

    def test_set_max_angular_velocity(self):
        enable_dynamical_orientation()
        wx, wy, wz = 1.0, 2.0, 3.0
        self._actions.set_max_angular_velocity(BALL_ID_1, wx, wy, wz)
        expected_action = ("SetMaxAngularVelocity", (BALL_ID_1, wx, wy, wz))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_max_angular_velocity_raises_when_dynamical_orientation_disabled(self):
        disable_dynamical_orientation()
        wx, wy, wz = 1.0, 2.0, 3.0
        self.assertRaises(RuntimeError, self._actions.set_max_angular_velocity, BALL_ID_1, wx, wy, wz)

    def test_set_ball_rotation(self):
        enable_dynamical_orientation()
        rx, ry, rz, rw = 0.0, 0.1, 0.2, 0.3
        self._actions.set_ball_rotation(BALL_ID_1, rx, ry, rz, rw)
        expected_action = ("SetBallRotation", (BALL_ID_1, rx, ry, rz, rw))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_rotation_raises_when_dynamical_orientation_disabled(self):
        disable_dynamical_orientation()
        rx, ry, rz, rw = 0.0, 0.1, 0.2, 0.3
        self.assertRaises(RuntimeError, self._actions.set_ball_rotation, BALL_ID_1, rx, ry, rz, rw)

    def test_set_ball_massive(self):
        self._actions.set_ball_massive(BALL_ID_1, False)
        expected_action = ("SetBallMassive", (BALL_ID_1, False))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_stop(self):
        self._actions.stop(BALL_ID_1)
        expected_action = ("Stop", (BALL_ID_1,))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_mass(self):
        mass_kg = 72.6
        self._actions.set_ball_mass(BALL_ID_1, mass_kg)
        expected_action = ("SetBallMass", (BALL_ID_1, mass_kg))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_agility(self):
        agility = 3.141592
        self._actions.set_ball_agility(BALL_ID_1, agility)
        expected_action = ("SetBallAgility", (BALL_ID_1, agility))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_angular_agility(self):
        enable_dynamical_orientation()
        angular_agility = 0.3
        self._actions.set_ball_angular_agility(BALL_ID_1, angular_agility)
        expected_action = ("SetBallAngularAgility", (BALL_ID_1, angular_agility))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_angular_agility_raises_when_dynamical_orientation_disabled(self):
        disable_dynamical_orientation()
        angular_agility = 0.3
        self.assertRaises(RuntimeError, self._actions.set_ball_angular_agility, BALL_ID_1, angular_agility)

    def test_add_ball_to_client_parks(self):
        ball = add_ball_to_park(self._park)
        _require_ball_field(self._park, "yaw")         # (lane) ballsAreEqual compares yaw, pitch and roll
        self._actions.add_ball_to_client_parks(ball.id)
        self.assertBallAddedToClientParks(ball, INITIAL_TIMESTAMP)

    def test_make_ball_local(self):
        ball = add_ball_to_park(self._park)
        self._actions.make_ball_local(ball.id)
        expected_action = ("BallNotGlobal", (ball.newBubbleId,))
        self.assertSingleActionHistory(ball.id, expected_action)

    def test_make_ball_global_makes_ball_global(self):
        ball = add_ball_to_park(self._park, is_global=False)
        self._actions.make_ball_global(ball.id)
        self.assertTrue(ball.isGlobal)

    def test_make_ball_global_adds_ball_to_client_parks(self):
        ball = add_ball_to_park(self._park, is_global=False)
        _require_ball_field(self._park, "yaw")         # (lane) ballsAreEqual compares yaw, pitch and roll
        self._actions.make_ball_global(ball.id)
        self.assertBallAddedToClientParks(ball, INITIAL_TIMESTAMP)

    def test_make_ball_global_sends_out_signal_with_src_id(self):
        callback_parameters = []

        def callback(src_id):
            callback_parameters.append(src_id)

        self._actions.on_ball_made_global.connect(callback)
        ball = add_ball_to_park(self._park, is_global=False)
        self._actions.make_ball_global(ball.id)
        self.assertListEqual([ball.id], callback_parameters)

    def test_remove_ball_removes_ball_right_away(self):
        ball = add_ball_to_park(self._park, is_global=False)
        self._actions.remove_ball(ball.id)
        self.assertNotIn(ball.id, self._park.balls)

    def test_remove_ball_does_not_note_removal_of_non_global_ball(self):
        ball = add_ball_to_park(self._park, is_global=False)
        self._actions.remove_ball(ball.id)
        self.assertHistoryEmpty()

    def test_remove_ball_notifies_removal_of_global_ball(self):
        ball = add_ball_to_park(self._park, is_global=True)
        self._actions.remove_ball(ball.id)
        expected_action = ("RemoveGlobalBall", (ball.id,))
        self.assertSingleActionHistory(ball.id, expected_action)

    def test_set_speed_fraction(self):
        speed_fraction = 0.5
        self._actions.set_speed_fraction(BALL_ID_1, speed_fraction)
        expected_action = ("SetSpeedFraction", (BALL_ID_1, speed_fraction))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_warp_to(self):
        x, y, z = 123.0, 456.0, 789.0
        minimum_range = 1000.0
        warp_speed = 20
        self._actions.warp_to(BALL_ID_1, x, y, z, minimum_range, warp_speed)
        expected_action = ("WarpTo", (BALL_ID_1, x, y, z, minimum_range, warp_speed))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_entity_warp_in(self):
        x, y, z = 123.0, 456.0, 789.0
        warp_speed = 20
        self._actions.entity_warp_in(BALL_ID_1, x, y, z, warp_speed)
        expected_action = ("EntityWarpIn", (BALL_ID_1, x, y, z, warp_speed))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_position_local_ball(self):
        x, y, z = 123.0, 456.0, 789.0
        self._actions.set_ball_position(BALL_ID_1, x, y, z)
        expected_action = ("SetBallPosition", (BALL_ID_1, x, y, z), False)
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_position_not_local_ball(self):
        x, y, z = 123.0, 456.0, 789.0
        self._actions.set_ball_position(BALL_ID_1, x, y, z, is_local_ball=True)
        expected_action = ("SetBallPosition", (BALL_ID_1, x, y, z), True)
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_harmonic_on_forcefield_with_all_negative_one_values_makes_field_non_massive(self):
        harmonic_value = -1
        corporation_id = -1
        alliance_id = -1
        self._actions.set_ball_harmonic(BALL_ID_1, harmonic_value, corporation_id, alliance_id, is_forcefield=True)
        expected_action = ("SetBallMassive", (BALL_ID_1, False))
        self.assertActionInHistory(BALL_ID_1, expected_action)

    def test_set_ball_harmonic_on_forcefield_with_harmonic_value_makes_field_massive(self):
        harmonic_value = 0
        corporation_id = -1
        alliance_id = -1
        self._actions.set_ball_harmonic(BALL_ID_1, harmonic_value, corporation_id, alliance_id, is_forcefield=True)
        expected_action = ("SetBallMassive", (BALL_ID_1, True))
        self.assertActionInHistory(BALL_ID_1, expected_action)

    def test_set_ball_harmonic_on_forcefield_with_corporation_id_makes_field_massive(self):
        harmonic_value = -1
        corporation_id = CORP_ID_1
        alliance_id = -1
        self._actions.set_ball_harmonic(BALL_ID_1, harmonic_value, corporation_id, alliance_id, is_forcefield=True)
        expected_action = ("SetBallMassive", (BALL_ID_1, True))
        self.assertActionInHistory(BALL_ID_1, expected_action)

    def test_set_ball_harmonic_on_forcefield_with_alliance_id_makes_field_massive(self):
        harmonic_value = -1
        corporation_id = -1
        alliance_id = ALLIANCE_ID_1
        self._actions.set_ball_harmonic(BALL_ID_1, harmonic_value, corporation_id, alliance_id, is_forcefield=True)
        expected_action = ("SetBallMassive", (BALL_ID_1, True))
        self.assertActionInHistory(BALL_ID_1, expected_action)

    def test_set_ball_harmonic_on_forcefield_sets_harmonic_values(self):
        harmonic_value = 12345
        corporation_id = CORP_ID_1
        alliance_id = ALLIANCE_ID_1
        self._actions.set_ball_harmonic(BALL_ID_1, harmonic_value, corporation_id, alliance_id, is_forcefield=True)
        expected_action = ("SetBallHarmonic", (BALL_ID_1, harmonic_value, corporation_id, alliance_id, True))
        self.assertActionInHistory(BALL_ID_1, expected_action)

    def test_set_ball_harmonic_on_non_forcefield_sets_harmonic_values(self):
        harmonic_value = 12345
        corporation_id = CORP_ID_1
        alliance_id = ALLIANCE_ID_1
        self._actions.set_ball_harmonic(BALL_ID_1, harmonic_value, corporation_id, alliance_id, is_forcefield=False)
        expected_action = ("SetBallHarmonic", (BALL_ID_1, harmonic_value, corporation_id, alliance_id, False))
        self.assertActionInHistory(BALL_ID_1, expected_action)

    def test_set_ball_radius(self):
        radius_meters = 42.0
        self._actions.set_ball_radius(BALL_ID_1, radius_meters)
        expected_action = ("SetBallRadius", (BALL_ID_1, radius_meters))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_free(self):
        self._actions.set_ball_free(BALL_ID_1)
        expected_action = ("SetBallFree", (BALL_ID_1, True))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_not_free(self):
        self._actions.set_ball_free(BALL_ID_1, False)
        expected_action = ("SetBallFree", (BALL_ID_1, False))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_interactive(self):
        self._actions.set_ball_interactive(BALL_ID_1, True)
        expected_action = ("SetBallInteractive", (BALL_ID_1, True))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_not_interactive(self):
        self._actions.set_ball_interactive(BALL_ID_1, False)
        expected_action = ("SetBallInteractive", (BALL_ID_1, False))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_follow_ball(self):
        range_meters = 1000.0
        self._actions.follow_ball(BALL_ID_1, BALL_ID_2, range_meters)
        expected_action = ("FollowBall", (BALL_ID_1, BALL_ID_2, range_meters))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_max_speed(self):
        max_meters_per_second = 500.0
        self._actions.set_max_speed(BALL_ID_1, max_meters_per_second)
        expected_action = ("SetMaxSpeed", (BALL_ID_1, max_meters_per_second))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_max_angular_speed(self):
        enable_dynamical_orientation()
        max_angular_speed = 6.0
        self._actions.set_max_angular_speed(BALL_ID_1, max_angular_speed)
        expected_action = ("SetMaxAngularSpeed", (BALL_ID_1, max_angular_speed,))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_set_ball_angular_speed_raises_when_dynamical_orientation_disabled(self):
        disable_dynamical_orientation()
        wx, wy, wz = 1.0, 2.0, 3.0
        self.assertRaises(RuntimeError, self._actions.set_ball_angular_velocity, BALL_ID_1, wx, wy, wz)

    def test_cloak_nonexistent_ball_should_fail(self):
        uncloak_range_meters = 1000.0
        success = self._actions.cloak_ball(BALL_ID_1, destiny.DSTNORMALCLOAK, uncloak_range_meters)
        self.assertFalse(success)
        self.assertHistoryEmpty()

    def test_cloak_existing_ball(self):
        ball = add_ball_to_park(self._park)
        uncloak_range_meters = 1000.0
        cloak_mode = destiny.DSTNORMALCLOAK
        success = self._actions.cloak_ball(ball.id, cloak_mode, uncloak_range_meters)
        self.assertTrue(success)
        expected_action = ("CloakBall", (ball.id, cloak_mode, uncloak_range_meters), ball.oldBubbleId)
        self.assertSingleActionHistory(ball.id, expected_action)

    def test_uncloak_ball(self):
        self._actions.uncloak_ball(BALL_ID_1)
        expected_action = ("UncloakBall", (BALL_ID_1,))
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_launch_missile(self):
        src_id = BALL_ID_1
        dst_id = BALL_ID_2
        owner_id = BALL_ID_3
        is_aimed_launch = True
        is_missile_massive = False
        self._actions.launch_missile(src_id, dst_id, owner_id, is_aimed_launch, is_missile_massive)
        expected_action = ("LaunchMissile", (src_id, dst_id, owner_id, is_aimed_launch, is_missile_massive))
        self.assertSingleActionHistory(src_id, expected_action)

    def test_clean_up_followers(self):
        follower = add_ball_to_park(self._park, ball_id=1)
        followee = add_ball_to_park(self._park, ball_id=2)
        self._park.FollowBall(follower.id, followee.id, 1000)
        self._park.SetBallPosition(follower.id, 500000000, 0, 0)
        remote = self._park.GetRemoteFollowers()
        self.assertEqual(1, len(remote))
        self._actions.clean_up_followers()
        expected_action = ("Stop", (follower.id,))
        self.assertSingleActionHistory(follower.id, expected_action)

    def test_undo_pending_cloak(self):
        ball = add_ball_to_park(self._park)
        self._actions.cloak_ball(ball.id, destiny.DSTNORMALCLOAK, 1000.0)
        self._actions.undo_pending_cloak(ball.id)
        self.assertHistoryEmpty()

    def test_undo_pending_cloak_ignores_other_balls(self):
        ball = add_ball_to_park(self._park)
        uncloak_range_meters = 1000.0
        cloak_mode = destiny.DSTNORMALCLOAK
        self._actions.cloak_ball(ball.id, cloak_mode, uncloak_range_meters)
        self._actions.undo_pending_cloak(ball.id + 1)
        expected_action = ("CloakBall", (ball.id, cloak_mode, uncloak_range_meters), ball.oldBubbleId)
        self.assertSingleActionHistory(ball.id, expected_action)

    def test_undo_pending_cloak_ignores_other_actions(self):
        radius_meters = 42.0
        self._actions.set_ball_radius(BALL_ID_1, radius_meters)
        expected_action = ("SetBallRadius", (BALL_ID_1, radius_meters))
        self._actions.undo_pending_cloak(BALL_ID_1)
        self.assertSingleActionHistory(BALL_ID_1, expected_action)

    def test_has_pending_cloak_returns_false_when_no_pending_cloak(self):
        has_pending_cloak = self._actions.has_pending_cloak(BALL_ID_1)
        self.assertFalse(has_pending_cloak)

    def test_has_pending_cloak_returns_true_when_pending_cloak(self):
        ball = add_ball_to_park(self._park)
        self._actions.cloak_ball(ball.id, destiny.DSTNORMALCLOAK, 1000.0)
        has_pending_cloak = self._actions.has_pending_cloak(ball.id)
        self.assertTrue(has_pending_cloak)

    def test_has_pending_uncloak_returns_false_when_no_pending_uncloak(self):
        has_pending_uncloak = self._actions.has_pending_uncloak(BALL_ID_1)
        self.assertFalse(has_pending_uncloak)

    def test_has_pending_uncloak_returns_true_when_pending_uncloak(self):
        self._actions.uncloak_ball(BALL_ID_1)
        has_pending_uncloak = self._actions.has_pending_uncloak(BALL_ID_1)
        self.assertTrue(has_pending_uncloak)

    def test_get_pending_cloak_mode_returns_none_when_ball_does_not_exist(self):
        cloak_mode = self._actions.get_pending_cloak_mode(BALL_ID_1)
        self.assertIsNone(cloak_mode)

    def test_get_pending_cloak_mode_returns_existing_mode_when_no_cloak_pending(self):
        ball = add_ball_to_park(self._park)
        cloak_mode = self._actions.get_pending_cloak_mode(ball.id)
        self.assertEqual(0, cloak_mode)

    def test_get_pending_cloak_mode_returns_pending_cloak_mode_when_no_cloak_pending(self):
        ball = add_ball_to_park(self._park)
        self._actions.cloak_ball(ball.id, destiny.DSTNORMALCLOAK, 1000.0)
        cloak_mode = self._actions.get_pending_cloak_mode(ball.id)
        self.assertEqual(destiny.DSTNORMALCLOAK, cloak_mode)

    def test_has_pending_set_free_returns_false_when_freedom_not_pending(self):
        freedom_pending = self._actions.has_pending_set_free(BALL_ID_1)
        self.assertFalse(freedom_pending)

    def test_has_pending_set_free_returns_false_when_freedom_not_pending_and_ball_already_free(self):
        ball = add_ball_to_park(self._park)
        self._park.SetBallFree(ball.id, True)
        freedom_pending = self._actions.has_pending_set_free(ball.id)
        self.assertFalse(freedom_pending)

    def test_has_pending_set_free_returns_true_when_freedom_pending(self):
        self._actions.set_ball_free(BALL_ID_1, True)
        freedom_pending = self._actions.has_pending_set_free(BALL_ID_1)
        self.assertTrue(freedom_pending)

    def test_has_pending_set_not_free_returns_false_when_freedom_removal_not_pending(self):
        freedom_removal_pending = self._actions.has_pending_set_not_free(BALL_ID_1)
        self.assertFalse(freedom_removal_pending)

    def test_has_pending_set_not_free_returns_false_when_freedom_removal_not_pending_and_ball_already_not_free(self):
        ball = add_ball_to_park(self._park)
        self._park.SetBallFree(ball.id, False)
        freedom_removal_pending = self._actions.has_pending_set_not_free(ball.id)
        self.assertFalse(freedom_removal_pending)

    def test_has_pending_set_not_free_returns_true_when_freedom_removal_pending(self):
        self._actions.set_ball_free(BALL_ID_1, False)
        freedom_removal_pending = self._actions.has_pending_set_not_free(BALL_ID_1)
        self.assertTrue(freedom_removal_pending)

    def test_get_pending_interactive_state_returns_none_when_ball_does_not_exist(self):
        state = self._actions.get_pending_interactive_state(BALL_ID_1)
        self.assertIsNone(state)

    def test_get_pending_interactive_state_returns_current_state_when_ball_exists(self):
        ball = add_ball_to_park(self._park, is_interactive=True)
        state = self._actions.get_pending_interactive_state(ball.id)
        self.assertEqual(True, state)

    def test_get_pending_interactive_state_returns_pending_state_when_available(self):
        ball = add_ball_to_park(self._park, is_interactive=True)
        self._actions.set_ball_interactive(ball.id, False)
        state = self._actions.get_pending_interactive_state(ball.id)
        self.assertEqual(False, state)

    def test_get_pending_speed_fraction_returns_none_when_ball_does_not_exist(self):
        speed_fraction = self._actions.get_pending_speed_fraction(BALL_ID_1)
        self.assertIsNone(speed_fraction)

    def test_get_pending_speed_fraction_returns_current_state_when_ball_exists(self):
        ball = add_ball_to_park(self._park, speed_fraction=0.5)
        speed_fraction = self._actions.get_pending_speed_fraction(ball.id)
        self.assertEqual(0.5, speed_fraction)

    def test_get_pending_speed_fraction_returns_pending_state_when_available(self):
        ball = add_ball_to_park(self._park, speed_fraction=0.5)
        self._actions.set_speed_fraction(ball.id, 1.0)
        speed_fraction = self._actions.get_pending_speed_fraction(ball.id)
        self.assertEqual(1.0, speed_fraction)


# =============================================================================================== test_ticker.py

T_BALL_ID_1 = 1001
T_BALL_ID_2 = 1002
T_BALL_ID_3 = 1003

CHAR_1 = 90000128
CHAR_2 = 90000256
CHAR_3 = 90000512

CLIENT_1 = 13579
CLIENT_2 = 24680
CLIENT_3 = 80808


class TestTicker(DestinyTestCase):
    """test_ticker.py:25-350 (its BALL_ID_n are T_BALL_ID_n here)."""

    def setUp(self):
        self._park = destiny.Ballpark()
        self._park.isMaster = True
        self._ball_info = TestBallInfo()
        self._actions = Actions(self._park)
        self._network = TestNetwork()
        self._character_interests = TestCharacterInterests()
        self._client_interests = TestClientInterests()
        self._bubble_updater = BubbleUpdater(self._park)
        self._update_batcher = ParkUpdateBatcher(
            self._park,
            self._network,
            self._character_interests,
            self._client_interests,
            self._bubble_updater
        )
        self._ticker = Ticker(
            self._park,
            self._ball_info,
            self._actions,
            self._update_batcher
        )

    def tearDown(self):
        del self._ticker
        del self._update_batcher
        del self._bubble_updater
        del self._actions
        del self._park

    def test_applies_action(self):
        ball = add_ball_to_park(self._park, ball_id=T_BALL_ID_1)
        self._actions.go_to_point(T_BALL_ID_1, 10000.0, 0.0, 0.0)
        self._ticker.tick()
        self.assertEqual(destiny.DSTBALL_GOTO, ball.mode)

    def test_cloak_ball_cloaks_ball(self):
        _require(self._park, "cloak_ball")
        ball = add_ball_to_park(self._park, ball_id=T_BALL_ID_1)
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, 1000.0)
        self._ticker.tick()
        self.assertEqual(destiny.DSTNORMALCLOAK, ball.isCloaked)

    def test_cloak_ball_emits_signal(self):
        _require(self._park, "cloak_ball")
        callback = mock.Mock()
        self._ticker.on_ball_cloaking.connect(callback)
        add_ball_to_park(self._park, ball_id=T_BALL_ID_1)
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, 1000.0)
        self._ticker.tick()
        callback.assert_called_once_with(T_BALL_ID_1)

    def assertUpdateNarrowcast(self, client_ids, action, timestamp=None):
        if timestamp is None:
            timestamp = self._ticker.stamp_for_system - 1
        for entry in self._network.narrowcasts:
            clients, method_name, actions, wait_for_bubble, casts_in_batch = entry
            if method_name != "DoDestinyUpdate":
                continue
            if set(clients) != set(client_ids):
                continue
            if (timestamp, action) in actions:
                return
        self.assertTrue(False, "Could not find specified action in narrowcasts")

    def assertUpdateSinglecast(self, client_id, action, timestamp=None):
        if timestamp is None:
            timestamp = self._ticker.stamp_for_system - 1
        for entry in self._network.singlecasts:
            entry_client_id, method_name, actions, wait_for_bubble, casts_in_batch = entry
            if method_name != "DoDestinyUpdate":
                continue
            if entry_client_id != client_id:
                continue
            if (timestamp, action) in actions:
                return
        self.assertTrue(False, "Could not find specified action in singlecasts")

    def _set_up_ball(self, ball_id, char_id, client_id):
        ball = add_ball_to_park(self._park, ball_id=ball_id)
        self._client_interests.add_client_interest(ball_id, client_id)
        self._character_interests.add_interest(ball_id, char_id)
        self._character_interests.map_to_client(char_id, client_id)
        self._ball_info.set_characters_for_ball(ball_id, [char_id])
        return ball

    def _set_up_single_ball(self):
        return self._set_up_ball(T_BALL_ID_1, CHAR_1, CLIENT_1)

    def _set_up_two_balls_in_same_bubble(self):
        ball_one = self._set_up_ball(T_BALL_ID_1, CHAR_1, CLIENT_1)
        ball_two = self._set_up_ball(T_BALL_ID_2, CHAR_2, CLIENT_2)
        return ball_one, ball_two

    def _set_up_three_balls_in_same_bubble(self):
        ball_one, ball_two = self._set_up_two_balls_in_same_bubble()
        ball_three = self._set_up_ball(T_BALL_ID_3, CHAR_3, CLIENT_3)
        return ball_one, ball_two, ball_three

    def test_cloak_ball_event_narrowcast_to_clients_in_bubble(self):
        _require(self._park, "cloak_ball")
        self._set_up_two_balls_in_same_bubble()
        self._perform_tick()
        uncloak_range_meters = 1000.0
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, uncloak_range_meters)
        self._perform_tick()
        self.assertUpdateNarrowcast(
            [CLIENT_1, CLIENT_2],
            ('CloakBall', (T_BALL_ID_1, destiny.DSTNORMALCLOAK, uncloak_range_meters))
        )

    def test_cloaking_and_uncloaking_in_same_tick_does_not_uncloak_ball(self):
        _require(self._park, "cloak_ball")
        self._set_up_single_ball()
        ball = add_ball_to_park(self._park, ball_id=T_BALL_ID_1)
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, 1000.0)
        self._actions.uncloak_ball(T_BALL_ID_1)
        self._ticker.tick()
        self.assertTrue(ball.isCloaked)

    def test_uncloak_ball_uncloaks_ball(self):
        _require(self._park, "cloak_ball")
        self._set_up_single_ball()
        ball = add_ball_to_park(self._park, ball_id=T_BALL_ID_1)
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, 1000.0)
        self._ticker.tick()
        self._actions.uncloak_ball(T_BALL_ID_1)
        self._ticker.tick()
        self.assertFalse(ball.isCloaked)

    def test_uncloak_does_not_error_when_ball_missing(self):
        self._actions.uncloak_ball(T_BALL_ID_1)
        self._ticker.tick()

    def test_uncloak_ball_emits_signal(self):
        _require(self._park, "cloak_ball")
        self._set_up_single_ball()
        callback = mock.Mock()
        self._ticker.on_ball_uncloaking.connect(callback)
        add_ball_to_park(self._park, ball_id=T_BALL_ID_1)
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, 1000.0)
        self._ticker.tick()
        self._actions.uncloak_ball(T_BALL_ID_1)
        self._ticker.tick()
        callback.assert_called_once_with(T_BALL_ID_1)

    def _perform_tick(self, skip_evolve=False):
        self._ticker.tick()
        if not skip_evolve:
            self._park.Evolve()
        self._update_batcher.send_batch()

    def test_uncloak_event_sent_to_uncloaking_ball(self):
        _require(self._park, "cloak_ball")
        self._set_up_single_ball()
        add_ball_to_park(self._park, ball_id=T_BALL_ID_1)
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, 1000.0)
        self._perform_tick()
        self._actions.uncloak_ball(T_BALL_ID_1)
        self._network.clear()
        self._perform_tick()
        self.assertUpdateSinglecast(CLIENT_1, ('UncloakBall', (T_BALL_ID_1,)))

    def _countMatchingAddBallsToParkActions(self, actions, ball_id):
        count = 0
        for timestamp, action in actions:
            action_name, action_parameters = action
            if action_name == "AddBallsToPark":
                ball_data = action_parameters[0]
                park = self.read_balls_from_stream_into_park(ball_data)
                self.assertIn(ball_id, park.balls, "Ball %s not in AddBallsToPark update stream" % ball_id)
                loaded_ball = park.balls[ball_id]
                self.assertBallsEqual(self._park.balls[ball_id], loaded_ball, ignore_bubble_id=True)
                count += 1
        return count

    def assertAddBallsToParkSinglecastExistsForBall(self, ball_id, client_id):
        found = 0
        for entry in self._network.singlecasts:
            entry_client_id, method_name, actions, wait_for_bubble, casts_in_batch = entry
            if method_name != "DoDestinyUpdate":
                continue
            if entry_client_id != client_id:
                continue

            found += self._countMatchingAddBallsToParkActions(actions, ball_id)
        self.assertEqual(1, found)

    def assertAddBallsToParkNarrowcastExistsForBall(self, ball_id, client_ids):
        found = 0
        for entry in self._network.narrowcasts:
            entry_client_id, method_name, actions, wait_for_bubble, casts_in_batch = entry
            if method_name != "DoDestinyUpdate":
                continue
            if set(entry_client_id) != set(client_ids):
                continue
            found += self._countMatchingAddBallsToParkActions(actions, ball_id)
        self.assertEqual(1, found)

    def test_add_ball_event_sent_to_balls_in_same_bubble_as_uncloaking_ball(self):
        _require(self._park, "cloak_ball")
        self._set_up_two_balls_in_same_bubble()
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, 1000.0)
        self._perform_tick()
        self._actions.uncloak_ball(T_BALL_ID_1)
        self._network.clear()
        self._perform_tick(skip_evolve=True)
        self.assertAddBallsToParkSinglecastExistsForBall(T_BALL_ID_1, CLIENT_2)

    def test_set_ball_position_emits_signal(self):
        self._set_up_single_ball()
        callback = mock.Mock()
        self._ticker.on_set_ball_position.connect(callback)
        is_local_ball = False
        self._actions.set_ball_position(
            T_BALL_ID_1,
            1234.5, 678.9, 987654321.0,
            is_local_ball=is_local_ball
        )
        self._perform_tick()
        callback.assert_called_once_with(T_BALL_ID_1, is_local_ball)

    def test_set_ball_position_sets_ball_position(self):
        ball = self._set_up_single_ball()
        x, y, z = 1234.5, 678.9, 987654321.0
        self._actions.set_ball_position(
            T_BALL_ID_1,
            1234.5, 678.9, 987654321.0,
            is_local_ball=False
        )
        self._perform_tick()
        self.assertEqual(
            (x, y, z),
            (ball.x, ball.y, ball.z)
        )

    def test_set_ball_position_event_sent_to_bubble(self):
        self._set_up_two_balls_in_same_bubble()
        self._perform_tick()
        x, y, z = 1.5, 2.9, 3.0
        self._actions.set_ball_position(
            T_BALL_ID_1,
            1.5, 2.9, 3.0,
            is_local_ball=False
        )
        self._perform_tick()
        self.assertUpdateNarrowcast(
            [CLIENT_1, CLIENT_2],
            ('SetBallPosition', (T_BALL_ID_1, x, y, z))
        )

    def test_set_ball_position_event_not_sent_to_bubble_for_local_balls(self):
        self._set_up_two_balls_in_same_bubble()
        self._perform_tick()
        self._actions.set_ball_position(
            T_BALL_ID_1,
            1.5, 2.9, 3.0,
            is_local_ball=True
        )
        self._perform_tick()
        self.assertEqual([], self._network.narrowcasts)

    def test_add_balls_to_park(self):
        self._set_up_two_balls_in_same_bubble()
        self._actions.set_ball_position(T_BALL_ID_2, 1000000000.0, 0.0, 0.0)
        self._perform_tick()
        self._network.clear()
        self._actions.add_ball_to_client_parks(T_BALL_ID_2)
        self._perform_tick(skip_evolve=True)
        self.assertAddBallsToParkNarrowcastExistsForBall(T_BALL_ID_2, [CLIENT_1])

    def test_ball_not_global_sends_remove_ball_update_to_other_bubbles(self):
        self._set_up_three_balls_in_same_bubble()
        self._actions.set_ball_position(T_BALL_ID_2, 1000000000.0, 0.0, 0.0)
        self._actions.set_ball_position(T_BALL_ID_3, 1000000000.0, 0.0, 1000.0)
        self._actions.make_ball_global(T_BALL_ID_2)
        self._perform_tick()
        self._network.clear()
        self._actions.make_ball_local(T_BALL_ID_2)
        self._perform_tick()
        # Ball 3 should not be included in the narrowcast as it is in the same bubble.
        self.assertUpdateNarrowcast([CLIENT_1], ('RemoveBall', (T_BALL_ID_2,)))

    def test_handle_remove_global_ball_sends_remove_ball_event_to_each_bubble_containing_interactives(self):
        self._set_up_three_balls_in_same_bubble()
        self._actions.set_ball_position(T_BALL_ID_2, 1000000000.0, 0.0, 0.0)
        self._actions.set_ball_position(T_BALL_ID_3, 1000000000.0, 0.0, 1000.0)
        self._actions.make_ball_global(T_BALL_ID_2)
        self._perform_tick()
        self._network.clear()
        self._actions.remove_ball(T_BALL_ID_2)
        self._perform_tick()
        self.assertUpdateNarrowcast([CLIENT_1], ('RemoveBall', (T_BALL_ID_2,)))
        self.assertUpdateNarrowcast([CLIENT_3], ('RemoveBall', (T_BALL_ID_2,)))

    def test_cloaked_ball_updates_only_get_sent_to_cloaked_balls_client(self):
        _require(self._park, "cloak_ball")
        self._set_up_two_balls_in_same_bubble()
        uncloak_range_meters = 1000.0
        self._actions.cloak_ball(T_BALL_ID_1, destiny.DSTNORMALCLOAK, uncloak_range_meters)
        self._perform_tick()
        self._network.clear()
        x, y, z = 1.5, 2.9, 3.0
        self._actions.go_to_point(
            T_BALL_ID_1,
            x, y, z
        )
        self._perform_tick()
        self.assertEqual([], self._network.narrowcasts)
        self.assertUpdateSinglecast(CLIENT_1, ("GotoPoint", (T_BALL_ID_1, x, y, z)))

    def test_no_updates_are_sent_out_for_bubbles_not_containing_interactive_balls(self):
        self._set_up_two_balls_in_same_bubble()
        self._actions.set_ball_interactive(T_BALL_ID_1, False)
        self._actions.set_ball_interactive(T_BALL_ID_2, False)
        self._perform_tick()
        self._network.clear()
        self._actions.go_to_point(T_BALL_ID_1, 1000.0, 2000.0, 3000.0)
        self._actions.set_speed_fraction(T_BALL_ID_2, 0.5)
        self._perform_tick()
        self.assertEqual([], self._network.narrowcasts)
        self.assertEqual([], self._network.singlecasts)


# ========================================================================================= test_updatebatcher.py

CHAR_ID_1 = 90000128
U_BALL_ID_1 = 1001
U_BALL_ID_2 = 1002
CLIENT_ID_1 = 13579

BUBBLE_1 = 21

GOTO_ACTION = (0, ("GotoPoint", (U_BALL_ID_1, 1.0, 2.0, 3.0)))
STOP_ACTION = (0, ("Stop", (U_BALL_ID_1)))                 # (sic) not a tuple: Destiny's text


class TestUpdateBatcher(DestinyTestCase):
    """test_updatebatcher.py:27-257 (its BALL_ID_n are U_BALL_ID_n here)."""

    def setUp(self):
        self._park = destiny.Ballpark()
        self._park.isMaster = True
        self._ball_info = TestBallInfo()
        self._network = TestNetwork()
        self._character_interests = TestCharacterInterests()
        self._client_interests = TestClientInterests()
        self._bubble_updater = BubbleUpdater(self._park)
        self._update_batcher = ParkUpdateBatcher(
            self._park,
            self._network,
            self._character_interests,
            self._client_interests,
            self._bubble_updater
        )

    def tearDown(self):
        del self._bubble_updater
        del self._update_batcher
        del self._park

    def test_add_to_character_history(self):
        self._update_batcher.add_to_character_history(CHAR_ID_1, GOTO_ACTION)
        self.assertListEqual(
            [GOTO_ACTION],
            self._update_batcher.get_character_history(CHAR_ID_1)
        )

    def test_add_to_character_history_adds_to_back(self):
        self._update_batcher.add_to_character_history(CHAR_ID_1, GOTO_ACTION)
        self._update_batcher.add_to_character_history(CHAR_ID_1, STOP_ACTION)
        self.assertListEqual(
            [GOTO_ACTION, STOP_ACTION],
            self._update_batcher.get_character_history(CHAR_ID_1)
        )

    def test_add_to_character_history_can_insert_in_front(self):
        self._update_batcher.add_to_character_history(CHAR_ID_1, GOTO_ACTION)
        self._update_batcher.add_to_character_history(CHAR_ID_1, STOP_ACTION, in_front=True)
        self.assertListEqual(
            [STOP_ACTION, GOTO_ACTION],
            self._update_batcher.get_character_history(CHAR_ID_1)
        )

    def test_clear_character_history(self):
        self._update_batcher.add_to_character_history(CHAR_ID_1, GOTO_ACTION)
        self._update_batcher.clear_character_history(CHAR_ID_1)
        self.assertListEqual(
            [],
            self._update_batcher.get_character_history(CHAR_ID_1)
        )

    def test_add_to_bubble_history(self):
        self._update_batcher.add_to_bubble_history(BUBBLE_1, GOTO_ACTION)
        self.assertListEqual(
            [GOTO_ACTION],
            self._update_batcher.get_bubble_history(BUBBLE_1)
        )

    def asssertCastsInBatchSinglecast(self, client_id, expected, cast_list):
        found = False
        for cast in cast_list:
            cast_client_id, name, _, _, casts_in_batch = cast
            if name == "DoDestinyUpdate" and cast_client_id == client_id:
                self.assertEqual(expected, casts_in_batch)
                found = True
        self.assertTrue(found)

    def asssertCastsInBatchNarrowcast(self, client_id, expected, cast_list):
        found = False
        for cast in cast_list:
            cast_client_id_list, name, _, _, casts_in_batch = cast
            if name == "DoDestinyUpdate" and client_id in cast_client_id_list:
                self.assertEqual(expected, casts_in_batch)
                found = True
        self.assertTrue(found)

    def test_character_history_only_has_cast_count_one(self):
        self._set_up_ball(U_BALL_ID_1, CHAR_ID_1, CLIENT_ID_1)
        self._update_batcher.add_to_character_history(CHAR_ID_1, GOTO_ACTION)
        self._update_batcher.send_batch()
        self.assertEqual(1, len(self._network.singlecasts))
        self.assertEqual(0, len(self._network.narrowcasts))
        self.asssertCastsInBatchSinglecast(
            CLIENT_ID_1,
            ClientUpdateCountThisTick.ONE,
            self._network.singlecasts
        )

    def test_clients_with_character_history_are_returned_as_a_set(self):
        self._set_up_ball(U_BALL_ID_1, CHAR_ID_1, CLIENT_ID_1)
        self._update_batcher.add_to_character_history(CHAR_ID_1, GOTO_ACTION)

        client_ids = self._update_batcher.get_clients_with_character_history()

        self.assertSetEqual({CLIENT_ID_1}, client_ids)

    def test_bubble_history_only_has_cast_count_one(self):
        ball = self._set_up_ball(U_BALL_ID_1, CHAR_ID_1, CLIENT_ID_1)
        self._update_batcher.add_to_bubble_history(ball.newBubbleId, GOTO_ACTION)
        self._update_batcher.send_batch()
        self.assertEqual(0, len(self._network.singlecasts))
        self.assertEqual(1, len(self._network.narrowcasts))
        self.asssertCastsInBatchNarrowcast(
            CLIENT_ID_1,
            ClientUpdateCountThisTick.ONE,
            self._network.narrowcasts
        )

    def test_character_and_bubble_history_has_cast_count_two(self):
        ball = self._set_up_ball(U_BALL_ID_1, CHAR_ID_1, CLIENT_ID_1)
        self._update_batcher.add_to_character_history(CHAR_ID_1, GOTO_ACTION)
        self._update_batcher.add_to_bubble_history(ball.newBubbleId, GOTO_ACTION)
        self._update_batcher.send_batch()
        self.asssertCastsInBatchNarrowcast(
            CLIENT_ID_1,
            ClientUpdateCountThisTick.TWO,
            self._network.narrowcasts
        )
        self.asssertCastsInBatchSinglecast(
            CLIENT_ID_1,
            ClientUpdateCountThisTick.TWO,
            self._network.singlecasts
        )

    def _set_up_ball(self, ball_id, char_id, client_id):
        ball = add_ball_to_park(self._park, ball_id=ball_id)
        self._client_interests.add_client_interest(ball_id, client_id)
        self._character_interests.add_interest(ball_id, char_id)
        self._character_interests.map_to_client(char_id, client_id)
        self._ball_info.set_characters_for_ball(ball_id, [char_id])
        return ball

    def _count_matching_set_state_actions(self, actions, ball_id):
        count = 0
        for timestamp, action in actions:
            action_name, action_parameters = action
            if action_name == "SetState":
                ball_data = action_parameters[0]
                park = self.read_balls_from_stream_into_park(ball_data)
                self.assertIn(ball_id, park.balls, "Ball %s not in SetState update stream" % ball_id)
                loaded_ball = park.balls[ball_id]
                self.assertBallsEqual(self._park.balls[ball_id], loaded_ball, ignore_bubble_id=True)
                count += 1
        return count

    def test_send_full_state_update(self):
        self._set_up_ball(U_BALL_ID_1, CHAR_ID_1, CLIENT_ID_1)
        self._update_batcher.send_full_state_update(CHAR_ID_1, U_BALL_ID_1)
        character_history = self._update_batcher.get_character_history(CHAR_ID_1)
        count = self._count_matching_set_state_actions(character_history, U_BALL_ID_1)
        self.assertEqual(1, count, "Could not find set state action including ball %s" % U_BALL_ID_1)

    def assertBallsInAddBallsAction(self, add_balls_action, ball_list):
        action_name, action_parameters = add_balls_action
        self.assertEqual("AddBallsToPark", action_name)
        self.assertEqual(1, len(action_parameters))
        ball_stream, = action_parameters
        park = self.read_balls_from_stream_into_park(ball_stream)
        for ball_id in ball_list:
            self.assertBallsEqual(self._park.balls[ball_id], park.balls[ball_id], ignore_bubble_id=True)

    def assertRemoveBallsActionRemovesBalls(self, remove_balls_action, ball_list):
        action_name, action_parameters = remove_balls_action
        self.assertEqual("RemoveBalls", action_name)
        self.assertEqual(1, len(action_parameters))
        ball_id_list, = action_parameters
        self.assertListEqual(ball_id_list, ball_list)

    def assertBallsInPackagedAddBallsAction(self, packaged_add_balls_action, ball_list):
        action_name, action_parameters = packaged_add_balls_action
        unpackaged_action_list = blue.marshal.Load(action_parameters)
        self.assertEqual(1, len(unpackaged_action_list))
        timestamp, add_balls_action = unpackaged_action_list[0]
        self.assertEqual(self._park.currentTime, timestamp)
        self.assertBallsInAddBallsAction(add_balls_action, ball_list)

    def assertPackagedRemoveBallsActionRemovesBalls(self, packaged_add_balls_action, ball_list):
        action_name, action_parameters = packaged_add_balls_action
        unpackaged_action_list = blue.marshal.Load(action_parameters)
        self.assertEqual(1, len(unpackaged_action_list))
        timestamp, add_balls_action = unpackaged_action_list[0]
        self.assertEqual(self._park.currentTime, timestamp)
        self.assertRemoveBallsActionRemovesBalls(add_balls_action, ball_list)

    def test_update_bubbles_inserts_add_balls_actions_to_character_history(self):
        self._set_up_ball(U_BALL_ID_1, CHAR_ID_1, CLIENT_ID_1)
        self._update_batcher.update_bubbles()
        self._park.Evolve()
        self._update_batcher.clear_character_history(CHAR_ID_1)
        add_ball_to_park(self._park, ball_id=U_BALL_ID_2)
        self._update_batcher.update_bubbles()

        char_history = self._update_batcher.get_character_history(CHAR_ID_1)
        self.assertEqual(1, len(char_history))
        timestamp, action = char_history[0]
        self.assertEqual(self._park.currentTime, timestamp)
        self.assertBallsInPackagedAddBallsAction(action, [U_BALL_ID_2])

    def test_update_bubbles_inserts_remove_balls_actions_to_character_history(self):
        self._set_up_ball(U_BALL_ID_1, CHAR_ID_1, CLIENT_ID_1)
        add_ball_to_park(self._park, ball_id=U_BALL_ID_2)
        self._update_batcher.update_bubbles()
        self._update_batcher.clear_character_history(CHAR_ID_1)
        self._park.RemoveBall(U_BALL_ID_2)
        self._park.Evolve()
        self._update_batcher.update_bubbles()

        char_history = self._update_batcher.get_character_history(CHAR_ID_1)
        self.assertEqual(1, len(char_history))
        timestamp, action = char_history[0]
        self.assertEqual(self._park.currentTime, timestamp)
        self.assertPackagedRemoveBallsActionRemovesBalls(action, [U_BALL_ID_2])

    def test_character_has_history(self):
        self.assertFalse(
            self._update_batcher.character_has_history(CHAR_ID_1)
        )
        self._update_batcher.add_to_character_history(CHAR_ID_1, GOTO_ACTION)
        self.assertTrue(
            self._update_batcher.character_has_history(CHAR_ID_1)
        )

    def test_bubble_has_history(self):
        self.assertFalse(
            self._update_batcher.bubble_has_history(BUBBLE_1)
        )
        self._update_batcher.add_to_bubble_history(BUBBLE_1, GOTO_ACTION)
        self.assertTrue(self._update_batcher.bubble_has_history(BUBBLE_1))


# ======================================================================================== test_bubbleupdater.py

B_BALL_ID_1 = 1001
B_BALL_ID_2 = 1002


class TestBubbleUpdater(unittest.TestCase):
    """test_bubbleupdater.py:13-82 (its BALL_ID_n are B_BALL_ID_n here)."""

    def setUp(self):
        self._park = destiny.Ballpark()
        self._park.isMaster = True
        self._bubble_updater = BubbleUpdater(self._park)

    def tearDown(self):
        del self._park
        del self._bubble_updater

    def test_additions_and_deletions(self):
        add_ball_to_park(self._park, ball_id=B_BALL_ID_1)
        add_ball_to_park(self._park, ball_id=B_BALL_ID_2, x=500000000.0)
        self._bubble_updater.additions_and_deletions()
        self.assertEqual({B_BALL_ID_1: [B_BALL_ID_1], B_BALL_ID_2: [B_BALL_ID_2]},
                         self._bubble_updater.additions_per_player)
        self.assertEqual({B_BALL_ID_1: [], B_BALL_ID_2: []}, self._bubble_updater.deletions_per_player)
        self.assertEqual({0: [B_BALL_ID_1], 1: [B_BALL_ID_2]}, self._bubble_updater.additions_per_bubble)
        self.assertEqual({0: [], 1: []}, self._bubble_updater.deletions_per_bubble)

    def test_additions_and_deletions_updates_bubbles_dict(self):
        add_ball_to_park(self._park, ball_id=B_BALL_ID_1)
        add_ball_to_park(self._park, ball_id=B_BALL_ID_2, x=500000000.0)
        self._bubble_updater.additions_and_deletions()
        self.assertEqual({0: {B_BALL_ID_1: 0}, 1: {B_BALL_ID_2: 0}}, self._park.bubbles)

    def test_additions_and_deletions_handles_removal(self):
        add_ball_to_park(self._park, ball_id=B_BALL_ID_1)
        add_ball_to_park(self._park, ball_id=B_BALL_ID_2)
        self._bubble_updater.additions_and_deletions()
        self._park.Evolve()
        self._park.RemoveBall(B_BALL_ID_1)
        self._bubble_updater.additions_and_deletions()
        self.assertEqual(
            {B_BALL_ID_2: 0},
            self._bubble_updater.additions_per_player
        )
        self.assertEqual(
            {B_BALL_ID_2: 0},
            self._bubble_updater.deletions_per_player
        )
        self.assertEqual(
            {0: []},
            self._bubble_updater.additions_per_bubble
        )
        self.assertEqual(
            {0: [B_BALL_ID_1]},
            self._bubble_updater.deletions_per_bubble
        )

    def test_additions_and_deletions_multiple_calls_same_tick(self):
        add_ball_to_park(self._park, ball_id=B_BALL_ID_1)
        add_ball_to_park(self._park, ball_id=B_BALL_ID_2, x=500000000.0)
        for _ in range(3):
            self._bubble_updater.additions_and_deletions()
        self.assertEqual(
            {B_BALL_ID_1: [B_BALL_ID_1], B_BALL_ID_2: [B_BALL_ID_2]},
            self._bubble_updater.additions_per_player
        )
        self.assertEqual(
            {B_BALL_ID_1: [], B_BALL_ID_2: []},
            self._bubble_updater.deletions_per_player
        )
        self.assertEqual(
            {0: [B_BALL_ID_1], 1: [B_BALL_ID_2]},
            self._bubble_updater.additions_per_bubble
        )
        self.assertEqual(
            {0: [], 1: []},
            self._bubble_updater.deletions_per_bubble
        )


# ============================= test/helpers.py with test_lifecycle_management.py and test_stream_read_write.py
# (the ballpark tests that exercise the C++ thunkers transcribed in exact_ballpark.server: AdditionsAndDeletions and the
# bubble transitions, and the ball streams)

TEN_BILLION = 10000000000.0


class BallPOPO(object):
    """test/helpers.py:159-176."""
    objectID = 0
    mass = 0.0
    radius = 0.0
    maxVel = 0.0
    isFree = False
    isGlobal = False
    isMassive = False
    isInteractive = False
    isSpaceJunk = False
    x = TEN_BILLION
    y = TEN_BILLION
    z = TEN_BILLION
    vx = 0.0
    vy = 0.0
    vz = 0.0
    agility = 0.0
    speedFraction = 0.0


def popo_add_ball_to_park(park, **kwargs):
    """test/helpers.py:179-200 (add_ball_to_park there; renamed here beside the server helpers' one)."""
    ball = BallPOPO()
    for i in kwargs:
        ball.__setattr__(i, kwargs[i])
    return park.AddBall(
        ball.objectID,
        ball.mass,
        ball.radius,
        ball.maxVel,
        ball.isFree,
        ball.isGlobal,
        ball.isMassive,
        ball.isInteractive,
        ball.isSpaceJunk,
        ball.x,
        ball.y,
        ball.z,
        ball.vx,
        ball.vy,
        ball.vz,
        ball.agility,
        ball.speedFraction,
    )


class BallparkTestCase(unittest.TestCase):
    """test/helpers.py:40-122 (the methods these tests use)."""

    def setUp(self):
        self.park = destiny.Ballpark()

    def tearDown(self):
        self.park.ClearAll()
        del self.park

    def add_balls(self, n):
        for i in range(n):
            yield popo_add_ball_to_park(self.park, objectID=i+1)

    def assertMiniBallEqual(self, first, second):
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.x, second.x)
        self.assertEqual(first.y, second.y)
        self.assertEqual(first.z, second.z)

    def assertMiniCapsuleEqual(self, first, second):
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.ax, second.ax)
        self.assertEqual(first.ay, second.ay)
        self.assertEqual(first.az, second.az)
        self.assertEqual(first.bx, second.bx)
        self.assertEqual(first.by, second.by)
        self.assertEqual(first.bz, second.bz)

    def assertMiniBoxEqual(self, first, second):
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.c0, second.c0)
        self.assertEqual(first.c1, second.c1)
        self.assertEqual(first.c2, second.c2)
        self.assertEqual(first.x0, second.x0)
        self.assertEqual(first.x1, second.x1)
        self.assertEqual(first.x2, second.x2)
        self.assertEqual(first.y0, second.y0)
        self.assertEqual(first.y1, second.y1)
        self.assertEqual(first.y2, second.y2)
        self.assertEqual(first.z0, second.z0)
        self.assertEqual(first.z1, second.z1)
        self.assertEqual(first.z2, second.z2)

    def assertBallEqual(self, first, second):
        for name in ("centerDist", "surfaceDist", "showBoxes"):            # (lane) ClientBall's members
            if not hasattr(first, name):
                raise unittest.SkipTest(f"ClientBall's {name} is not in this port")
        self.assertEqual(first.Agility, second.Agility)
        self.assertEqual(first.allianceID, second.allianceID)
        self.assertEqual(first.centerDist, second.centerDist)
        self.assertEqual(first.corporationID, second.corporationID)
        self.assertEqual(first.effectStamp, second.effectStamp)
        self.assertEqual(first.followId, second.followId)
        self.assertEqual(first.followRange, second.followRange)
        self.assertEqual(first.formationID, second.formationID)
        self.assertEqual(first.gotoX, second.gotoX)
        self.assertEqual(first.gotoY, second.gotoY)
        self.assertEqual(first.gotoZ, second.gotoZ)
        self.assertEqual(first.harmonic, second.harmonic)
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.isCloaked, second.isCloaked)
        self.assertEqual(first.isFree, second.isFree)
        self.assertEqual(first.isGlobal, second.isGlobal)
        self.assertEqual(first.isInteractive, second.isInteractive)
        self.assertEqual(first.isMassive, second.isMassive)
        self.assertEqual(first.isMoribund, second.isMoribund)
        self.assertEqual(first.mass, second.mass)
        self.assertEqual(first.maxVelocity, second.maxVelocity)
        self.assertEqual(len(first.miniBalls), len(second.miniBalls))
        self.assertEqual(first.mode, second.mode)
        self.assertEqual(first.newBubbleId, second.newBubbleId)
        self.assertEqual(first.oldBubbleId, second.oldBubbleId)
        self.assertEqual(first.ownerId, second.ownerId)
        self.assertEqual(first.pitch, second.pitch)
        self.assertEqual(first.radius, second.radius)
        self.assertEqual(first.roll, second.roll)
        self.assertEqual(first.showBoxes, second.showBoxes)
        self.assertEqual(first.speedFraction, second.speedFraction)
        self.assertEqual(first.surfaceDist, second.surfaceDist)
        self.assertEqual(first.vx, second.vx)
        self.assertEqual(first.vy, second.vy)
        self.assertEqual(first.vz, second.vz)
        self.assertEqual(first.x, second.x)
        self.assertEqual(first.y, second.y)
        self.assertEqual(first.yaw, second.yaw)
        self.assertEqual(first.z, second.z)


class BallparkWrapped(srv.DestinyPark):
    """test/helpers.py:269-278: a Python ballpark keeping the DoBubbleTransitions it receives. (Destiny's metaclass
    creates the C++ park before this __init__ runs: here the super().__init__ call.)"""
    __test__ = False

    def __init__(self):
        super().__init__()
        self.transitions = []

    def DoBubbleTransitions(self, transitions):
        self.transitions = transitions

    def GetTransitions(self):
        return self.transitions


class BallparkWrappedTestCase(BallparkTestCase):
    """test/helpers.py:281-288."""

    def setUp(self):
        self.park = BallparkWrapped()

    def check_transitions(self, expected_transitions):
        actual_transitions = self.park.GetTransitions()
        self.assertEqual(expected_transitions, actual_transitions)


class TestAdditionsAndDeletions(BallparkTestCase):
    """test_lifecycle_management.py:44-114."""

    def test_addition_of_invalid_ball_id_raises_typeerror(self):
        ball, = self.add_balls(1)
        ball.isInteractive = True
        additions_per_player = {}
        deletions_per_player = {}
        additions_per_bubble = {}
        deletions_per_bubble = {}
        user_ships = [None]
        self.park.InitializeBubbles()
        self.assertRaises(
            TypeError,
            self.park.AdditionsAndDeletions,
            additions_per_player,
            deletions_per_player,
            additions_per_bubble,
            deletions_per_bubble,
            user_ships
        )

    def test_addition(self):
        ball, = self.add_balls(1)
        ball.isInteractive = True
        additions_per_player = {}
        deletions_per_player = {}
        additions_per_bubble = {}
        deletions_per_bubble = {}
        user_ships = [ball.id]
        self.park.InitializeBubbles()
        self.park.AdditionsAndDeletions(additions_per_player, deletions_per_player, additions_per_bubble,
                                        deletions_per_bubble, user_ships)
        self.assertEqual(additions_per_player, {ball.id: [ball.id]})
        self.assertEqual(deletions_per_player, {ball.id: []})
        self.assertEqual(additions_per_bubble, {ball.newBubbleId: [ball.id]})
        self.assertEqual(deletions_per_bubble, {ball.newBubbleId: []})

    def test_no_change(self):
        ball, = self.add_balls(1)
        ball.isInteractive = True
        additions_per_player = {}
        deletions_per_player = {}
        additions_per_bubble = {}
        deletions_per_bubble = {}
        user_ships = [ball.id]
        self.park.InitializeBubbles()
        self.park.AdditionsAndDeletions(additions_per_player, deletions_per_player, additions_per_bubble,
                                        deletions_per_bubble, user_ships)
        self.park.CopyBubbles()
        self.park.AdditionsAndDeletions(additions_per_player, deletions_per_player, additions_per_bubble,
                                        deletions_per_bubble, user_ships)
        self.assertEqual(additions_per_player, {ball.id: ball.newBubbleId})
        self.assertEqual(deletions_per_player, {ball.id: ball.newBubbleId})
        self.assertEqual(additions_per_bubble, {ball.newBubbleId: []})
        self.assertEqual(deletions_per_bubble, {ball.newBubbleId: []})

    def test_removal(self):
        ball, = self.add_balls(1)
        ball.isInteractive = True
        additions_per_player = {}
        deletions_per_player = {}
        additions_per_bubble = {}
        deletions_per_bubble = {}
        user_ships = [ball.id]

        self.park.InitializeBubbles()
        self.park.AdditionsAndDeletions(additions_per_player, deletions_per_player, additions_per_bubble,
                                        deletions_per_bubble, user_ships)
        self.park.RemoveBall(ball.id)
        self.park.CopyBubbles()
        self.park.AdditionsAndDeletions(additions_per_player, deletions_per_player, additions_per_bubble,
                                        deletions_per_bubble, user_ships)

        self.assertEqual(additions_per_player, {})
        self.assertEqual(deletions_per_player, {})
        self.assertEqual(additions_per_bubble, {})
        self.assertEqual(deletions_per_bubble, {})


class TestBubbleTransitions(BallparkWrappedTestCase):
    """test_lifecycle_management.py:117-184."""

    def test_empty_bubble_transitions_list_is_sent_as_empty(self):
        additions_per_player = {}
        deletions_per_player = {}
        additions_per_bubble = {}
        deletions_per_bubble = {}
        user_ships = []
        self.park.InitializeBubbles()
        self.park.AdditionsAndDeletions(
            additions_per_player, deletions_per_player, additions_per_bubble, deletions_per_bubble, user_ships,
        )

        expected_transitions = []
        self.check_transitions(expected_transitions)

    def test_adding_ball_transitions_it_to_bubble(self):
        ball, = self.add_balls(1)
        ball.isInteractive = True

        ball_id = ball.id
        old_bubble_id = ball.newBubbleId

        additions_per_player = {}
        deletions_per_player = {}
        additions_per_bubble = {}
        deletions_per_bubble = {}
        user_ships = [ball_id]
        self.park.InitializeBubbles()
        self.park.AdditionsAndDeletions(
            additions_per_player, deletions_per_player, additions_per_bubble, deletions_per_bubble, user_ships,
        )
        new_bubble_id = ball.newBubbleId

        self.assertNotEqual(old_bubble_id, new_bubble_id)

        expected_transitions = [(ball_id, old_bubble_id, new_bubble_id)]
        self.check_transitions(expected_transitions)

    def test_adding_two_balls_transition_them_to_bubble(self):
        ball_1, ball_2 = self.add_balls(2)
        ball_1.isInteractive = True
        ball_2.isInteractive = True

        old_bubble_id_1 = ball_1.newBubbleId
        old_bubble_id_2 = ball_2.newBubbleId

        additions_per_player = {}
        deletions_per_player = {}
        additions_per_bubble = {}
        deletions_per_bubble = {}
        user_ships = [ball_1.id, ball_2.id]
        self.park.InitializeBubbles()
        self.park.AdditionsAndDeletions(
            additions_per_player, deletions_per_player, additions_per_bubble, deletions_per_bubble, user_ships,
        )

        new_bubble_id_1 = ball_1.newBubbleId
        new_bubble_id_2 = ball_2.newBubbleId

        self.assertNotEqual(old_bubble_id_1, new_bubble_id_1)
        self.assertNotEqual(old_bubble_id_2, new_bubble_id_2)

        expected_transitions = [
            (ball_1.id, old_bubble_id_1, new_bubble_id_1),
            (ball_2.id, old_bubble_id_2, new_bubble_id_2),
        ]
        self.check_transitions(expected_transitions)


class TestWriteBallsToStream(BallparkTestCase):
    """test_stream_read_write.py:8-58."""

    def test_write_no_ball(self):
        stream = blue.MemStream()
        self.park.WriteBallsToStream([], stream)
        self.assertEqual(stream.Read(), b"\x01\x00\x00\x00\x00")

    def test_write_ball_then_read_ball(self):
        stream = blue.MemStream()
        saved_ball, = self.add_balls(1)
        self.park.WriteBallsToStream([saved_ball.id], stream)
        reader_park = destiny.Ballpark()
        reader_park.ReadFullStateFromStream(stream)
        self.assertIn(saved_ball.id, reader_park.balls)
        loaded_ball = reader_park.balls[saved_ball.id]
        self.assertBallEqual(saved_ball, loaded_ball)

    def test_write_read_miniball(self):
        stream = blue.MemStream()
        saved_ball, = self.add_balls(1)
        saved_ball.AddMiniBall(1.0, 2.0, 3.0, 4.0)
        self.park.WriteBallsToStream([saved_ball.id], stream)
        self.park.ClearAll()
        self.park.ReadFullStateFromStream(stream)
        self.assertIn(saved_ball.id, self.park.balls)
        loaded_ball = self.park.balls[saved_ball.id]
        self.assertMiniBallEqual(saved_ball.miniBalls[0], loaded_ball.miniBalls[0])

    def test_write_read_minicapsule(self):
        stream = blue.MemStream()
        saved_ball, = self.add_balls(1)
        saved_ball.AddMiniCapsule(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        self.park.WriteBallsToStream([saved_ball.id], stream)
        self.park.ClearAll()
        self.park.ReadFullStateFromStream(stream)
        self.assertIn(saved_ball.id, self.park.balls)
        loaded_ball = self.park.balls[saved_ball.id]
        self.assertMiniCapsuleEqual(saved_ball.miniCapsules[0], loaded_ball.miniCapsules[0])

    def test_write_read_minibox(self):
        stream = blue.MemStream()
        saved_ball, = self.add_balls(1)
        saved_ball.AddMiniBox(1.0, 2.0, 3.0,
                              1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0)
        self.park.WriteBallsToStream([saved_ball.id], stream)
        self.park.ClearAll()
        self.park.ReadFullStateFromStream(stream)
        self.assertIn(saved_ball.id, self.park.balls)
        loaded_ball = self.park.balls[saved_ball.id]
        self.assertMiniBoxEqual(saved_ball.miniBoxes[0], loaded_ball.miniBoxes[0])


class TestWriteFullStateToStream(BallparkTestCase):
    """test_stream_read_write.py:61-75."""

    def test_write_no_ball(self):
        stream = blue.MemStream()
        self.park.WriteFullStateToStream(stream)
        self.assertEqual(stream.Read(), b"\x00\x00\x00\x00\x00")

    def test_write_ball_then_read_ball(self):
        stream = blue.MemStream()
        saved_ball, = self.add_balls(1)
        self.park.WriteFullStateToStream(stream)
        reader_park = destiny.Ballpark()
        reader_park.ReadFullStateFromStream(stream)
        self.assertIn(saved_ball.id, reader_park.balls)
        loaded_ball = reader_park.balls[saved_ball.id]
        self.assertBallEqual(saved_ball, loaded_ball)


# ==================================================== this port's checks: what Destiny's tests do not cover (not vectors)

def _server_park():
    park = destiny.Ballpark()
    park.isMaster = True
    network, chars, clients, info = TestNetwork(), TestCharacterInterests(), TestClientInterests(), TestBallInfo()
    actions = Actions(park)
    batcher = ParkUpdateBatcher(park, network, chars, clients, BubbleUpdater(park))
    ticker = Ticker(park, info, actions, batcher)
    return park, actions, ticker, batcher, network, chars, clients, info


def test_lane_pre_tick_cleanup_queues_stops_before_the_stamp_moves_and_behind_earlier_commands():
    """TIME-712 step 1 and MOV-731 (_baseticker.py:23-32, _ticker.py:56-69): the clean-up runs first, its Stop takes the
    stamp of the tick before, joins the queue behind a command already queued for the same ball, and is applied after
    it, in the same pre-tick."""
    park, actions, ticker, *_ = _server_park()
    add_ball_to_park(park, ball_id=1)
    add_ball_to_park(park, ball_id=2)
    park.FollowBall(1, 2, 1000.0)
    park.SetBallPosition(1, 500000000.0, 0.0, 0.0)          # the follower in another bubble
    assert park.GetRemoteFollowers() == [1]
    actions.go_to_point(1, 10.0, 20.0, 30.0)                # queued at stamp 0
    seen = []

    def record(name):
        seen.append(name)
    actions.on_add_to_system_history.connect(record)
    queued = []
    original_flush = actions.flush_history

    def flush():
        out = original_flush()
        queued.extend(out)
        return out
    actions.flush_history = flush
    ticker.tick()
    assert seen == ["Stop"]
    assert queued == [(1, 0, ('GotoPoint', (1, 10.0, 20.0, 30.0))), (1, 0, ('Stop', (1,)))]
    assert ticker.stamp_for_system == 1 and actions._stamp_for_system == 1
    assert park.balls[1].mode == destiny.DSTBALL_STOP     # the Stop, applied last, wins


def test_lane_events_of_a_ball_in_no_known_bubble_are_dropped():
    """SEN-712's last clause (_baseticker.py:99-115): an applied action of a ball the park does not hold has bubble -1,
    which is not a bubble: nothing is sent."""
    park, actions, ticker, batcher, network, chars, clients, info = _server_park()
    add_ball_to_park(park, ball_id=1001)
    clients.add_client_interest(1001, 7)
    chars.add_interest(1001, 70)
    chars.map_to_client(70, 7)
    info.set_characters_for_ball(1001, [70])
    ticker.tick()                                           # the first tick: the ball enters its bubble
    park.Evolve()
    batcher.send_batch()
    network.clear()
    actions.go_to_point(4242, 1.0, 2.0, 3.0)                # no such ball (stamp 1)
    actions.set_speed_fraction(1001, 0.5)
    ticker.tick()
    park.Evolve()
    batcher.send_batch()
    assert network.singlecasts == []
    assert len(network.narrowcasts) == 1
    clients_of, name, state, wait, count = network.narrowcasts[0]
    assert (clients_of, name, wait, count) == ([7], 'DoDestinyUpdate', False, ClientUpdateCountThisTick.ONE)
    assert state == [(1, ('SetSpeedFraction', (1001, 0.5)))]


def test_lane_events_in_dead_bubbles_never_enter_the_bubble_history():
    """SEN-712's dead-bubble clause at its first gate (_baseticker.py:108-115): an event of a ball whose bubble has no
    interactive ball is not queued for the bubble. (Destiny's test_no_updates_are_sent_out_for_bubbles_not_containing_
    interactive_balls cannot see this gate: send_batch addresses only the interactive balls' clients, so a dead
    bubble's history would reach no one either way, _parkupdatebatcher.py:72-76.)"""
    park, actions, ticker, batcher, network, chars, clients, info = _server_park()
    for bid, cid in ((1001, 7), (1002, 8)):
        add_ball_to_park(park, ball_id=bid)
        clients.add_client_interest(bid, cid)
    actions.set_ball_interactive(1001, False)
    actions.set_ball_interactive(1002, False)
    ticker.tick()
    park.Evolve()
    batcher.send_batch()
    assert 0 in park.bubbles and 0 not in park.bubbleInteractives
    actions.go_to_point(1001, 1000.0, 2000.0, 3000.0)
    ticker.tick()
    assert not batcher.bubble_has_history(0)


def test_lane_ball_stream_round_trip_for_every_mode_it_writes():
    """WriteBallToStream / ReadBallFromStream (Thunkers.cpp:3180-3376, 2897-3178) through the adapter: each mode's
    fields come back as written (the transcription's own consistency; Destiny's tests cover STOP)."""
    park = destiny.Ballpark()
    for bid, x in ((11, 0.0), (12, 50.0), (13, -80.0), (14, 30.0)):
        add_ball_to_park(park, ball_id=bid, x=x, vx=3.0)
    park.GotoPoint(11, 1.0e3, 2.0e3, 3.0e3)
    park.FollowBall(12, 11, 250.0)
    park.Orbit(13, 11, 900.0)
    park.WarpTo(14, 1.0e9, 0.0, 0.0, 15000.0, 7)
    for _ in range(2):
        park.Evolve()
    dump = blue.MemStream()
    park.WriteBallsToStream([11, 12, 13, 14, 999], dump)
    data = dump.Read()
    assert data[0] == srv.DESTINY_BALLS and int.from_bytes(data[1:5], "little", signed=True) == park.currentTime
    loaded = DestinyTestCase().read_balls_from_stream_into_park(data)
    assert list(loaded.balls) == [11, 12, 13, 14]
    for bid in (11, 12, 13, 14):
        a, b = park.balls[bid], loaded.balls[bid]
        for name in ("id", "mass", "radius", "maxVelocity", "isFree", "isMassive", "isInteractive", "x", "y", "z",
                     "vx", "vy", "vz", "Agility", "speedFraction", "mode", "followId", "ownerId"):
            assert getattr(a, name) == getattr(b, name), (bid, name)
    for bid in (12, 13):                                    # FOLLOW and ORBIT carry the range
        assert park.balls[bid].followRange == loaded.balls[bid].followRange
    assert (park.balls[14].followRange, loaded.balls[14].followRange) == (0.0, 10.0)   # WARP does not: the default
    assert [loaded.balls[11].gotoX, loaded.balls[11].gotoY, loaded.balls[11].gotoZ] == [1.0e3, 2.0e3, 3.0e3]
    w, lw = park.balls[14], loaded.balls[14]
    assert (lw.effectStamp, lw.gotoX) == (w.effectStamp, w.gotoX)
    assert loaded.lane.balls[14].last_collision == park.lane.balls[14].last_collision
    assert loaded.lane.balls[12].follow_ptr == 11 and 12 in loaded.lane.balls[11].followers


def test_lane_actions_read_a_private_time_attribute_after_time_zero():
    """_actions.py:31-34: with the park's time past zero, Actions reads park._currentTime, which the C++ park does not
    expose: the adapter has none either."""
    park = destiny.Ballpark()
    park.Evolve()
    try:
        Actions(park)
    except AttributeError as exc:
        assert "_currentTime" in str(exc)
    else:
        raise AssertionError("Actions found a _currentTime")


def test_lane_ticker_inside_on_tick_skips_the_first_tick():
    """The wiring the layer's docstrings ask for (_baseticker.py:23-26: tick in DoPreTick; _parkupdatebatcher.py:53-56:
    send_batch in DoPostTick) through this port's OnTick (Ballpark.cpp:219-330): the first OnTick evolves once and sends
    no DoPreTick (260-273), so an action queued before it is applied only at the second OnTick."""
    park, actions, ticker, batcher, network, chars, clients, info = _server_park()
    _require(park, "on_tick")
    park.lane.pre_tick_handlers.append(lambda current_time: ticker.tick())
    park.lane.post_tick_handlers.append(lambda current_time: batcher.send_batch())
    ball = add_ball_to_park(park, ball_id=1001)
    actions.go_to_point(1001, 10000.0, 0.0, 0.0)
    interval = park.tickInterval * 10000
    park.lane.on_tick(interval)                             # the first OnTick: Evolve once, no pre-tick
    assert (park.currentTime, ball.mode) == (1, destiny.DSTBALL_STOP)
    park.lane.on_tick(2 * interval)                         # one whole interval: DoPreTick, Evolve, DoPostTick
    assert (park.currentTime, ball.mode) == (2, destiny.DSTBALL_GOTO)


def test_lane_set_max_angular_velocity_is_never_applied_to_the_park():
    """_actions.py:190-209 with _ticker.py:106-122: the park exposes no SetMaxAngularVelocity (Ballpark_Blue.cpp), so
    the Ticker has no _parent_ method to apply it with."""
    park, actions, ticker, *_ = _server_park()
    assert "_parent_SetMaxAngularVelocity" not in ticker._get_parent_methods()
    assert "_parent_SetMaxAngularSpeed" in ticker._get_parent_methods()
    assert "_parent_GotoPoint" in ticker._get_parent_methods()


# ================================= planted faults: each must be caught by the tests named beside it (this port's checks)


def _run(cases) -> tuple:
    """Run Destiny's transcribed test methods (TestCase class, names) and this port's checks (plain functions): (failed,
    skipped, run)."""
    failed = skipped = run = 0
    for case in cases:
        if callable(case) and not isinstance(case, tuple):
            run += 1
            try:
                case()
            except unittest.SkipTest:
                skipped += 1
            except Exception:
                failed += 1
            continue
        cls, names = case
        result = unittest.TestResult()
        unittest.TestSuite(cls(n) for n in names).run(result)
        failed += len(result.failures) + len(result.errors)
        skipped += len(result.skipped)
        run += result.testsRun
    return failed, skipped, run


def _dup_any(self, ball_id, eventName, *args, **kwargs):
    """TIME-710 broken: an action equal to ANY queued action is dropped."""
    if kwargs.get("insert_ball_id_into_args", True):
        args = (ball_id,) + args
    local_args = kwargs.get("local_args", None)
    event = (eventName, args) if local_args is None else (eventName, args, local_args)
    action = ball_id, self._stamp_for_system, event
    if action not in self._system_history:
        self._system_history.append(action)
    self.on_add_to_system_history(eventName)


def _dup_without_stamp(self, ball_id, eventName, *args, **kwargs):
    """TIME-710 broken: the duplicate test ignores the stamp."""
    if kwargs.get("insert_ball_id_into_args", True):
        args = (ball_id,) + args
    local_args = kwargs.get("local_args", None)
    event = (eventName, args) if local_args is None else (eventName, args, local_args)
    action = ball_id, self._stamp_for_system, event
    if not self._system_history or (self._system_history[-1][0], self._system_history[-1][2]) != (ball_id, event):
        self._system_history.append(action)
    self.on_add_to_system_history(eventName)


def _stamp_frozen(self):
    """TIME-709 broken: the stamp is not moved at the pre-tick."""
    self._actions.set_stamp_for_system(self.stamp_for_system)


def _tick_stamp_first(self):
    """TIME-712 broken: the stamp moves before the follower clean-up."""
    self._increment_timestamp()
    self._finalize_tick()
    self._flush_history()
    self._pre_update_bubbles()
    self._update_bubbles()
    self._post_update_bubbles()


def _apply_cloak_with_others(self):
    """TIME-712 broken: CloakBall applied with the other park methods, before the bubbles are updated."""
    for entry in self.current_system_history:
        event_ball_id, stamp, event = entry
        action, args = event[0], event[1]
        if action in srv.PARK_METHODS_THAT_GET_CALLED_DIRECTLY:
            continue
        if '_parent_' + action in self._get_parent_methods():
            try:
                getattr(self._park, '_parent_' + action)(*args)
            except Exception:
                continue


def _handle_cloak_without_applying(self, args, event, event_ball_id, local_args, stamp):
    """TIME-712 broken (with _apply_cloak_with_others): the CloakBall handler only distributes the event."""
    ball_id = args[0]
    if ball_id in self._park.balls:
        self.on_ball_cloaking(ball_id)
    self._update_batcher.add_to_bubble_history(local_args, (stamp, event))
    return True


def _cloak_in_step_four(mp):
    mp.setattr(Ticker, "_apply_park_parent_methods_except_cloak_ball", _apply_cloak_with_others)
    mp.setattr(Ticker, "_handle_cloak_ball_action", _handle_cloak_without_applying)


def _valid_without_interactives(self, event_bubble_id, event_ball_id):
    """SEN-712 broken: dead bubbles (no interactives) are not dropped."""
    if event_bubble_id not in self._park.bubbles:
        return False
    if event_ball_id <= 0 or event_ball_id not in self._park.balls:
        return False
    return True


def _valid_any_bubble(self, event_bubble_id, event_ball_id):
    """SEN-712 broken: events of a ball in no known bubble are not dropped."""
    return True


_ORIGINAL_ADD_DEL = srv.DestinyPark.AdditionsAndDeletions


def _add_del_keeping_old_bubbles(self, apl, dpl, apb, dpb, users):
    """AdditionsAndDeletions broken: the user balls' old bubbles are not moved on (Thunkers.cpp:2887)."""
    old = {b.id: b.old_bubble for b in self.lane.balls.values()}
    _ORIGINAL_ADD_DEL(self, apl, dpl, apb, dpb, users)
    for b in self.lane.balls.values():
        b.old_bubble = old[b.id]


def _additions_without_copy(self):
    """BubbleUpdater broken: CopyBubbles skipped."""
    if self._tick != self._park.currentTime:
        self._process_additions_and_deletions_for_tick()
        self._tick = self._park.currentTime


def _remove_queued(self, src_id):
    """TIME-711 broken: removal queued like the other actions."""
    ball = self._park.balls[src_id]
    if ball.isGlobal:
        self._notify_remove_global_ball(src_id)
    self._add_to_system_history(src_id, 'RemoveBall')


_CLEANUP_CHECK = test_lane_pre_tick_cleanup_queues_stops_before_the_stamp_moves_and_behind_earlier_commands

FAULTS = {
    "duplicate_any_queued": ("TIME-710", lambda mp: mp.setattr(Actions, "_add_to_system_history", _dup_any),
                             [(TestActions, ["test_duplicate_non_subsequent_actions_are_not_ignored"])]),
    "duplicate_ignores_stamp": ("TIME-710", lambda mp: mp.setattr(Actions, "_add_to_system_history",
                                                                   _dup_without_stamp),
                                [(TestActions, ["test_duplicate_subsequent_actions_with_different_timestamps_are_not_"
                                                "ignored"])]),
    "stamp_frozen": ("TIME-709", lambda mp: mp.setattr(Ticker, "_increment_timestamp", _stamp_frozen),
                     [(TestTicker, ["test_set_ball_position_event_sent_to_bubble",
                                    "test_handle_remove_global_ball_sends_remove_ball_event_to_each_bubble_containing_"
                                    "interactives"])]),
    "actions_applied_at_once": ("TIME-709", lambda mp: mp.setattr(
        Actions, "go_to_point", lambda self, src_id, x, y, z: self._park.GotoPoint(src_id, x, y, z)),
        [(TestTicker, ["test_applies_action"]), (TestActions, ["test_go_to_point"])]),
    "remove_queued": ("TIME-711", lambda mp: mp.setattr(Actions, "remove_ball", _remove_queued),
                      [(TestActions, ["test_remove_ball_removes_ball_right_away",
                                      "test_remove_ball_does_not_note_removal_of_non_global_ball"])]),
    "stamp_before_cleanup": ("TIME-712", lambda mp: mp.setattr(srv.BaseTicker, "tick", _tick_stamp_first),
                             [_CLEANUP_CHECK]),
    "cloak_applied_with_others": ("TIME-712", _cloak_in_step_four,
                                  [(TestTicker, ["test_cloaking_and_uncloaking_in_same_tick_does_not_uncloak_ball"])]),
    "dead_bubbles_kept": ("SEN-712", lambda mp: mp.setattr(srv.BaseTicker, "_is_valid_event_bubble",
                                                           _valid_without_interactives),
                          [test_lane_events_in_dead_bubbles_never_enter_the_bubble_history]),
    "unknown_bubbles_kept": ("SEN-712", lambda mp: mp.setattr(srv.BaseTicker, "_is_valid_event_bubble",
                                                              _valid_any_bubble),
                             [test_lane_events_of_a_ball_in_no_known_bubble_are_dropped]),
    "cleanup_skipped": ("MOV-731", lambda mp: mp.setattr(Ticker, "_finalize_tick", lambda self: None),
                        [_CLEANUP_CHECK]),
    "remote_followers_not_stopped": ("MOV-731", lambda mp: mp.setattr(Actions, "clean_up_followers",
                                                                      lambda self: None),
                                     [(TestActions, ["test_clean_up_followers"])]),
    "old_bubbles_kept": ("AdditionsAndDeletions", lambda mp: mp.setattr(srv.DestinyPark, "AdditionsAndDeletions",
                                                                        _add_del_keeping_old_bubbles),
                         [(TestBubbleUpdater, ["test_additions_and_deletions_handles_removal"])]),
    "copy_bubbles_skipped": ("BubbleUpdater", lambda mp: mp.setattr(BubbleUpdater, "additions_and_deletions",
                                                                    _additions_without_copy),
                             [(TestBubbleUpdater, ["test_additions_and_deletions_updates_bubbles_dict"])]),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_lane_planted_faults_are_caught(fault, monkeypatch):
    rule, patch, cases = FAULTS[fault]
    clean = _run(cases)
    if clean[1] == clean[2]:
        pytest.skip(f"{fault} ({rule}): every catching test is skipped on this port's ballpark")
    assert clean[0] == 0, f"{fault}: the catching tests fail without the fault"
    patch(monkeypatch)
    failed, skipped, run = _run(cases)
    assert failed > 0, f"the planted fault {fault} ({rule}) went unnoticed"

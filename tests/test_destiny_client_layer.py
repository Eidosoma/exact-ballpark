# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against published test vectors for Destiny's client layer: this port's
transcription (exact_ballpark.client) against Destiny's own client tests, transcribed; then this port's checks of the
session harness (what a pilot's client knows at each tick), which are not test vectors.

Copied from carbonengine/destiny at 114e89fa (MIT licence, (c) CCP Games), python/destiny/test/net/client/:
helpers.py (add_ball_to_park, TestErrorHandler, TestClientTicker), test_ticker.py (TestTicker, 9 tests) and
test_mergestateintohistory.py (TestMergeStateIntoHistory, 11 tests): every client test, in Destiny's order, keeping
Destiny's values, stamps, structures and orders, each naming its source. Destiny's tests run on `destiny.Ballpark()`
(here `ClientPark`, exact_ballpark.server's DestinyPark over this port's FullBallpark with the ticker's `ego`, `isRunning`
and `Start`) with `blue.MemStream` (here exact_ballpark.server's MemStream). Changes from Destiny's text, all mechanical:
the imports point at this port; the helper classes named Test* carry `__test__ = False`; test_ticker.py's BALL_ID_n are
C_BALL_ID_n here (same values).

After Destiny's tests, this port's checks (test_lane_*) of the session (exact_ballpark.client.Session): every client's park
equals the server's park at the client's time, field by field, on generated command streams, for downlink delays of 0
to 2 ticks; the client's clock runs `down` ticks behind; a command first moves its ship in the server's Evolve
n + up -> n + up + 1 and its pilot sees it `down` ticks later; a late update makes the client rewind and its clock slip;
cloaked ships leave the other clients' parks and come back on uncloaking; each client gets 0 to 2 updates a tick, its
user update first; and the start-up quirk (a command before the first pre-tick). Then planted faults, each of which
must be caught by the tests named beside it.
"""

from __future__ import annotations

import math
import random
import unittest
import unittest.mock as mock

import pytest

from exact_ballpark import client as cl
from exact_ballpark import server as srv
from exact_ballpark.client import ClientTickerInterface, TickErrorHandlerInterface, Ticker, merge_state_into_history
from exact_ballpark.modes import FOLLOW, ORBIT, WARP
from exact_ballpark.ref import GOTO, STOP


class destiny:                                              # the names the tests take from the destiny module
    Ballpark = cl.ClientPark
    DSTBALL_ORBIT = srv.DSTBALL_ORBIT
    DSTBALL_STOP = srv.DSTBALL_STOP


class blue:                                                 # blue.MemStream
    MemStream = srv.MemStream


# =================================================================== helpers.py (test/net/client/helpers.py)

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
        speed_fraction=1.0
):
    """helpers.py:6-44."""
    return park.AddBall(
        ball_id,
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
        speed_fraction,
    )


class TestErrorHandler(TickErrorHandlerInterface):
    """helpers.py:47-64."""
    __test__ = False

    def __init__(self):
        self._fatal_desync_occured = False
        self._recoverable_desync_occurred = False

    def on_fatal_desync(self):
        self._fatal_desync_occured = True

    def on_recoverable_desync(self):
        self._recoverable_desync_occurred = True

    @property
    def fatal_desync_occurred(self):
        return self._fatal_desync_occured

    @property
    def recoverable_desync_occurred(self):
        return self._recoverable_desync_occurred


class TestClientTicker(ClientTickerInterface):
    """helpers.py:67-90."""
    __test__ = False

    def on_set_state(self):
        pass

    def on_ballpark_local_action(self, func_name, args):
        pass

    def should_log_actions(self):
        return True

    def get_ball_destruction_delay(self, ball, is_terminal=False):
        return 0

    def get_ball_destruction_delays(self, ball_ids, is_release=False):
        return {
            ball_id: 0
            for ball_id in ball_ids
        }

    def clean_up_after_ball_removal(self, ball_id, ball, is_terminal=False):
        pass

    def clean_up_after_multiple_ball_removal(self, ball_ids, is_release=False):
        pass


# ============================================================================================== test_ticker.py

C_BALL_ID_1 = 1
C_BALL_ID_2 = 2
C_BALL_ID_3 = 3

TICKS_ELAPSED = 10


class TestTicker(unittest.TestCase):
    """test_ticker.py:18-124 (its BALL_ID_n are C_BALL_ID_n here)."""

    def setUp(self):
        self._error_handler = TestErrorHandler()
        self._client_ticker = TestClientTicker()
        self._ticker = Ticker(self._error_handler, self._client_ticker)
        self._ballpark = destiny.Ballpark()
        self._ticker.set_ballpark(self._ballpark)
        self._do_setstate()
        self._tick(TICKS_ELAPSED)

    def tearDown(self):
        del self._ballpark
        del self._ticker
        del self._client_ticker
        del self._error_handler

    def _get_park_with_state(self):
        bp = destiny.Ballpark()
        ball_1 = add_ball_to_park(bp, ball_id=1, x=0, y=0, z=0)
        ball_2 = add_ball_to_park(bp, ball_id=2, x=1000, y=0, z=0)
        ball_3 = add_ball_to_park(bp, ball_id=3, x=-1000, y=0, z=0)
        bp.Orbit(ball_2.id, ball_1.id)
        bp.GotoPoint(ball_3.id, 0, 0, 0)
        bp.GotoDirection(ball_1.id, 0.0, 1.0, 0.0)
        for _ in range(TICKS_ELAPSED):
            bp.Evolve()
        return bp

    def _do_setstate(self, current_time=0):
        park_with_state = self._get_park_with_state()
        s = blue.MemStream()
        park_with_state.WriteFullStateToStream(s)
        self._ticker.update([[current_time, ("SetState", (s.Read(), C_BALL_ID_1))]], False)

    def test_setstate(self):
        self.assertEqual(self._ballpark.currentTime, TICKS_ELAPSED + 1)
        ball_two = self._ballpark.balls[C_BALL_ID_2]
        self.assertEqual(destiny.DSTBALL_ORBIT, ball_two.mode)

    def test_stopball(self):
        self._ticker.update([[TICKS_ELAPSED + 1, ("Stop", (C_BALL_ID_2,))]], False)
        self._ticker.do_pre_tick()
        self._ticker.do_post_tick(TICKS_ELAPSED)
        ball_two = self._ballpark.balls[C_BALL_ID_2]
        self.assertEqual(destiny.DSTBALL_STOP, ball_two.mode)

    def test_future_state_gets_applied_at_correct_time(self):
        self._ticker.update([[TICKS_ELAPSED + 2, ("Stop", (C_BALL_ID_2,))]], False)
        ball_two = self._ballpark.balls[C_BALL_ID_2]
        self._tick(TICKS_ELAPSED + 1)
        self.assertEqual(destiny.DSTBALL_ORBIT, ball_two.mode)
        self._tick(TICKS_ELAPSED + 2)
        self.assertEqual(destiny.DSTBALL_STOP, ball_two.mode)

    def test_late_arrival(self):
        ball_two = self._ballpark.balls[C_BALL_ID_2]
        self._tick(TICKS_ELAPSED + 1)
        self._tick(TICKS_ELAPSED + 2)
        self.assertEqual(destiny.DSTBALL_ORBIT, ball_two.mode)
        self._ticker.update([[TICKS_ELAPSED + 2, ("Stop", (C_BALL_ID_2,))]], False)
        self._tick(TICKS_ELAPSED + 3)
        self.assertEqual(destiny.DSTBALL_STOP, ball_two.mode)

    def _tick(self, time_stamp):
        self._ticker.do_pre_tick()
        self._ballpark._parent_Evolve()
        self._ticker.do_post_tick(time_stamp)

    def test_synchronize_to_simulation_time_can_not_rewind_before_first_tick(self):
        success = self._ticker.synchronize_to_simulation_time(TICKS_ELAPSED - 1)
        self.assertFalse(success)

    def test_synchronize_to_simulation_time_can_rewind_to_last_tick(self):
        self._tick(TICKS_ELAPSED + 1)
        self._ballpark.RemoveBall(C_BALL_ID_1)
        self.assertNotIn(C_BALL_ID_1, self._ballpark.balls)
        self.assertEqual(TICKS_ELAPSED + 2, self._ballpark.currentTime)
        success = self._ticker.synchronize_to_simulation_time(TICKS_ELAPSED + 1)
        self.assertTrue(success)
        self.assertEqual(TICKS_ELAPSED + 1, self._ballpark.currentTime)
        self.assertIn(C_BALL_ID_1, self._ballpark.balls)

    def test_synchronize_to_simulation_time_can_rewind_two_ticks(self):
        self._tick(TICKS_ELAPSED + 1)
        self._ballpark.RemoveBall(C_BALL_ID_1)
        self.assertNotIn(C_BALL_ID_1, self._ballpark.balls)
        self._tick(TICKS_ELAPSED + 2)
        self.assertEqual(TICKS_ELAPSED + 3, self._ballpark.currentTime)
        success = self._ticker.synchronize_to_simulation_time(TICKS_ELAPSED + 1)
        self.assertTrue(success)
        self.assertEqual(TICKS_ELAPSED + 1, self._ballpark.currentTime)

    def test_no_desync_reported_for_successfuly_handled_update(self):
        self._ticker.update([[TICKS_ELAPSED + 1, ("Stop", (C_BALL_ID_2,))]], False)
        self._ticker.do_pre_tick()
        self.assertFalse(self._error_handler.recoverable_desync_occurred)
        self.assertFalse(self._error_handler.fatal_desync_occurred)

    def test_fatal_desync_reported_when_update_has_multiple_timestamps(self):
        self._ticker.update([[TICKS_ELAPSED + 1, ("Stop", (C_BALL_ID_2,))], [TICKS_ELAPSED + 2, ("Stop",
                                                                                                (C_BALL_ID_2,))]],
                            False)
        self.assertTrue(self._error_handler.fatal_desync_occurred)

    def test_recoverable_desync_reported_when_fail_to_sync_to_timestamp(self):
        self._ticker.update([[TICKS_ELAPSED + 1, ("Stop", (C_BALL_ID_2,))]], False)
        self._ticker.synchronize_to_simulation_time = mock.Mock(return_value=False)
        self._ticker.do_pre_tick()
        self.assertTrue(self._error_handler.recoverable_desync_occurred)


# ================================================================================ test_mergestateintohistory.py

SHIP_ID_1 = 1000000176143
GOTO_DIRECTION = ('GotoDirection', (SHIP_ID_1, 1.0, 0.0, 0.0))
STOP_EVENT = ('Stop', (SHIP_ID_1,))                         # Destiny's STOP (the name STOP is this port's mode here)


class TestMergeStateIntoHistory(unittest.TestCase):
    """test_mergestateintohistory.py:13-135 (its STOP is STOP_EVENT here)."""

    def test_merging_empty_state_list_has_no_effect(self):
        state = []
        history_list = []
        merge_state_into_history(state, history_list, True)
        merge_state_into_history(state, history_list, False)
        self.assertListEqual([], history_list)

    def test_merging_empty_state_list_with_wait_for_bubble(self):
        state = [(101, GOTO_DIRECTION)]
        history_list = []
        merge_state_into_history(state, history_list, True)
        self.assertListEqual(
            [[state, True]],
            history_list
        )

    def test_merging_empty_state_list_without_wait_for_bubble(self):
        state = [(101, GOTO_DIRECTION)]
        history_list = []
        merge_state_into_history(state, history_list, False)
        self.assertListEqual(
            [[state, False]],
            history_list
        )

    def test_merge_two_states_same_tick_wait_for_bubble_should_clear(self):
        state_one = [(101, GOTO_DIRECTION)]
        state_two = [(101, STOP_EVENT)]
        history_list = []
        merge_state_into_history(state_one, history_list, True)
        merge_state_into_history(state_two, history_list, False)
        self.assertListEqual(
            [[state_one + state_two, False]],
            history_list
        )

    def test_merge_two_states_same_tick_wait_for_bubble_should_clear_if_wait_in_second_state(self):
        state_one = [(101, GOTO_DIRECTION)]
        state_two = [(101, STOP_EVENT)]
        history_list = []
        merge_state_into_history(state_one, history_list, False)
        merge_state_into_history(state_two, history_list, True)
        self.assertListEqual(
            [[state_one + state_two, False]],
            history_list
        )

    def test_merge_two_states_same_tick_no_wait_for_bubble(self):
        state_one = [(101, GOTO_DIRECTION)]
        state_two = [(101, STOP_EVENT)]
        history_list = []
        merge_state_into_history(state_one, history_list, False)
        merge_state_into_history(state_two, history_list, False)
        self.assertListEqual(
            [[state_one + state_two, False]],
            history_list
        )

    def test_merge_newer_state_wait_for_bubble_persists_for_older_state(self):
        state_one = [(101, GOTO_DIRECTION)]
        state_two = [(102, STOP_EVENT)]
        history_list = []
        merge_state_into_history(state_one, history_list, True)
        merge_state_into_history(state_two, history_list, False)
        self.assertListEqual(
            [
                [state_one, True],
                [state_two, False]],
            history_list
        )

    def test_merge_newer_state_wait_for_bubble_persists_for_newer_state(self):
        state_one = [(101, GOTO_DIRECTION)]
        state_two = [(102, STOP_EVENT)]
        history_list = []
        merge_state_into_history(state_one, history_list, True)
        merge_state_into_history(state_two, history_list, False)
        self.assertListEqual(
            [
                [state_one, True],
                [state_two, False]],
            history_list
        )

    def test_merge_older_state(self):
        state_newer = [(101, GOTO_DIRECTION)]
        state_older = [(100, STOP_EVENT)]
        history_list = []
        merge_state_into_history(state_newer, history_list, False)
        merge_state_into_history(state_older, history_list, False)
        self.assertListEqual(
            [
                [state_older, False],
                [state_newer, False]],
            history_list
        )

    def test_merge_older_state_older_wait_for_bubble(self):
        state_newer = [(101, GOTO_DIRECTION)]
        state_older = [(100, STOP_EVENT)]
        history_list = []
        merge_state_into_history(state_newer, history_list, False)
        merge_state_into_history(state_older, history_list, True)
        self.assertListEqual(
            [
                [state_older, True],
                [state_newer, False]],
            history_list
        )

    def test_merge_older_state_newer_wait_for_bubble(self):
        state_newer = [(101, GOTO_DIRECTION)]
        state_older = [(100, STOP_EVENT)]
        history_list = []
        merge_state_into_history(state_newer, history_list, True)
        merge_state_into_history(state_older, history_list, False)
        self.assertListEqual(
            [
                [state_older, False],
                [state_newer, True]],
            history_list
        )


# ============================================================ this port's checks: the session (not vectors)

def _rows(park) -> dict:
    """{ball id: the fields a client's park must share with the server's}: the flags, the movement state and every
    mode's own fields while it is in that mode (the ball stream carries a mode's fields only in that mode, and AddBall
    gives the rest Ball::Ball's defaults, Thunkers.cpp:3180-3377; a STOP ball's old goto point, a WARP ball's follow
    range and the sensor flag are not in it, nor read by the flight). Doubles by their bits."""
    out = {}
    for bid, b in park.lane.balls.items():
        h = lambda x: float(x).hex()                        # noqa: E731
        row = [b.mode, int(b.is_free), int(b.is_massive), int(b.is_interactive), int(b.is_cloaked),
               *map(h, b.pos), *map(h, b.vel), *map(h, b.last_g), h(b.mass), h(b.radius), h(b.max_vel), h(b.agility),
               h(b.speed_fraction), h(b.time_factor)]
        if b.mode in (GOTO, WARP):
            row += [*map(h, b.goto)]
        if b.mode in (FOLLOW, ORBIT):
            row += [b.follow_id, h(b.follow_range)]
        if b.mode == WARP:
            row += [b.effect_stamp, h(b.last_collision), b.follow_id, b.owner_id]
        out[bid] = tuple(row)
    return out


def _seen_by(rows: dict, ego: int) -> dict:
    """The server's rows as a pilot's client holds them: another ship cloaked in that state is not in its park (the
    CloakBall event removed it; the UncloakBall's AddBallsToPark brings it back), its own ship is."""
    return {bid: row for bid, row in rows.items() if bid == ego or row[4] == 0}


def _server_park(balls: list) -> srv.DestinyPark:
    park = srv.DestinyPark(is_master=True)
    for bid, x, y, mass, max_vel, inter in balls:
        park.AddBall(bid, mass, 40.0, max_vel, 1, 0, 0, inter, 0, x, y, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    return park


SHIPS = ((101, 0.0, 0.0, 2.0e6, 300.0, 1), (102, 6000.0, 0.0, 1.0e7, 200.0, 1), (103, -4000.0, 3000.0, 5.0e6,
                                                                                    250.0, 1),
         (104, 2000.0, -7000.0, 3.0e6, 150.0, 0))           # three pilots' ships and a ship without a pilot
PILOTS = {101: 0, 102: 1, 103: 2}


def _random_command(r: random.Random, ships: list) -> tuple:
    """One Actions call: (name, args), commands and setters the session's clients apply as the server does; cloaking
    only by pilots' ships (a ball without a pilot is not interactive: its cloak never reaches the clients, the first
    finding below)."""
    s = r.choice(ships)
    other = r.choice([b for b in ships if b != s])
    k = r.random()
    if k < 0.22:
        return "go_to_direction", (s, r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1))
    if k < 0.36:
        return "go_to_point", (s, r.uniform(-2e4, 2e4), r.uniform(-2e4, 2e4), r.uniform(-2e3, 2e3))
    if k < 0.50:
        return "set_speed_fraction", (s, r.choice([0.0, 0.25, r.uniform(0, 1), 1.0, 1.3]))
    if k < 0.58:
        return "stop", (s,)
    if k < 0.66:
        return "orbit", (s, other, r.uniform(500.0, 5000.0))
    if k < 0.72:
        return "follow_ball", (s, other, r.uniform(100.0, 3000.0))
    if k < 0.77:
        return "set_max_speed", (s, r.uniform(50.0, 400.0))
    if k < 0.82:
        return "set_ball_mass", (s, 10 ** r.uniform(6.0, 7.5))
    if k < 0.86:
        return "set_ball_agility", (s, r.uniform(0.3, 2.0))
    if k < 0.90:
        return "set_ball_velocity", (s, r.uniform(-100, 100), r.uniform(-100, 100), r.uniform(-100, 100))
    if k < 0.94:
        return "warp_to", (s, r.uniform(1.5e5, 4e5) * r.choice([-1, 1]), r.uniform(-1e4, 1e4), 0.0,
                           r.uniform(0.0, 3e4), r.choice([1, 3, 5]))
    if k < 0.96:
        return "set_ball_radius", (s, r.uniform(10.0, 200.0))
    pilot_ship = r.choice(sorted(PILOTS))                  # cloaks by pilots' ships: see the findings below
    if k < 0.98:
        return "cloak_ball", (pilot_ship, r.choice([srv.DSTNORMALCLOAK, srv.DSTRESTORECLOAK, srv.DSTGMCLOAK]),
                              r.uniform(500.0, 3000.0))
    return "uncloak_ball", (pilot_ship,)


def _run_session(seed: int, down: int, up: int, ticks: int = 26, per_tick: float = 1.6) -> tuple:
    """A session on SHIPS with random commands after each tick, a tick's uncloaks first and its cloaks last; every
    tick, each client's park against the server's park at the client's time (_seen_by). Returns (mismatches, the
    clients' desync counts, client-ticks compared)."""
    r = random.Random(seed)
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS, down=down, up=up)
    s.start()
    history = {park.currentTime: _rows(park)}
    bad, compared = [], 0
    ships = [b[0] for b in SHIPS]
    for _ in range(ticks):
        cmds = [_random_command(r, ships) for _ in range(r.choice([0, 1, 1, 2, 3]) if per_tick else 0)]
        # a tick's uncloaks first and cloaks last, the order every client applies them in (the findings below)
        cmds = sorted(cmds, key=lambda c: {"uncloak_ball": 0, "cloak_ball": 2}.get(c[0], 1))
        for name, args in cmds:
            s.command(name, *args)
        s.tick()
        history[park.currentTime] = _rows(park)
        for p, (cpark, ticker, errors) in s.clients.items():
            if not ticker._state_is_valid:                  # no SetState yet (the first `down` ticks)
                continue
            want = history.get(cpark.currentTime)
            if want is None:
                bad.append((s.tick_no, p, "no server state at", cpark.currentTime))
                continue
            want = _seen_by(want, ships[p])
            got = _rows(cpark)
            compared += 1
            if got != want:
                bad.append((s.tick_no, p, cpark.currentTime, sorted(k for k in set(want) | set(got)
                                                                    if want.get(k) != got.get(k))))
    desyncs = [(e.fatal, e.recoverable) for _, _, e in s.clients.values()]
    return bad, desyncs, compared


@pytest.mark.parametrize("down,up", [(0, 0), (1, 0), (2, 0), (0, 1), (1, 2)])
def test_lane_session_clients_hold_the_servers_state_at_their_time(down, up):
    """Every client's park equals the server's at the client's time, every tick, field by field (_rows), on random
    commands and setters from several pilots (follow, orbit and short warps among them), and no desync is reported."""
    for seed in (1, 2, 3):
        bad, desyncs, compared = _run_session(seed, down, up)
        assert compared > 0
        assert bad == [], (seed, bad[:3])
        assert desyncs == [(0, 0)] * len(PILOTS), desyncs


@pytest.mark.parametrize("down", [0, 1, 2, 3])
def test_lane_session_client_clock_runs_down_ticks_behind(down):
    """After session tick n the server's park is at time n and each client's at n - down (start-up included: the
    first update is applied at its stamp)."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS, down=down)
    s.start()
    for _ in range(8):
        s.tick()
        assert park.currentTime == s.tick_no
        for p in PILOTS.values():
            c = s.client_park(p).currentTime
            if s.tick_no - down >= 2:
                assert c == s.tick_no - down, (down, s.tick_no, c)


@pytest.mark.parametrize("down,up", [(0, 0), (1, 0), (0, 1), (2, 1), (1, 3)])
def test_lane_session_command_lag_is_down_plus_up(down, up):
    """A command issued after tick n (seeing the client's state at n - down) is stamped n + up and first moves the ship
    in the server's Evolve n + up -> n + up + 1; its pilot first sees it after tick n + up + 1 + down: L = down + up
    ticks later than a command applied at once."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS, down=down, up=up)
    s.start()
    for _ in range(3):
        s.tick()
    n = s.tick_no
    s.command("go_to_direction", 101, 1.0, 0.0, 0.0)
    seen_server, seen_client = None, None
    x0 = park.balls[101].x
    for _ in range(down + up + 6):
        s.tick()
        if seen_server is None and park.balls[101].x != x0:
            seen_server = park.currentTime
        c = s.client_park(0).balls[101]
        if seen_client is None and c.x != x0:
            seen_client = s.tick_no
    stamps = [e[0] for _, sent in s.sent_log for kind, to, m, st, w, cnt in sent for e in st
              if e[1][0] == "GotoDirection"]
    assert stamps and set(stamps) == {n + up}, stamps
    assert seen_server == n + up + 1, (seen_server, n, up)
    assert seen_client == n + up + 1 + down, (seen_client, n, up, down)


def test_lane_session_late_update_rewinds_and_the_clock_slips():
    """An update one tick later than the rest: the client applies it at its stamp by rewinding to a stored snapshot
    (_ticker.py:265-305), its park left at that stamp; its clock then runs one tick further behind, and its park still
    equals the server's at the client's time."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS, down=0)
    s.start()
    for _ in range(3):
        s.tick()
    s.command("go_to_direction", 102, 0.0, 1.0, 0.0)
    real_tick = s.tick

    def tick_holding_pilot_1():
        s.tick_no += 1
        n = s.tick_no
        for name, args in s.uplink.pop(n, []):
            getattr(s.actions, name)(*args)
        s.server.lane.on_tick(n * s.interval)
        for kind, to, method, state, wait, count in s.network.take():
            for client_id in ([to] if kind == "single" else to):
                late = client_id == s.interests.client_base + 1
                s.downlink[n + (1 if late else 0)].append((client_id, state, wait))
        for client_id, state, wait in s.downlink.pop(n, []):
            s.clients[client_id - s.interests.client_base][1].update(state, wait)
        for p in sorted(s.clients):
            s.clients[p][0].lane.on_tick(n * s.interval)
    tick_holding_pilot_1()                                  # pilot 1's update of this tick arrives one tick late
    c1 = s.client_park(1)
    assert c1.currentTime == park.currentTime               # it ran on without the command: a prediction
    assert _rows(c1) != _rows(park)
    want = _rows(park)
    real_tick()                                             # the late update arrives: stamp < the client's time
    assert c1.currentTime == park.currentTime - 1           # rewound to the stamp, then one tick: the clock slipped
    assert _rows(c1) == want                                # and it holds the server's state at its time
    assert s.clients[1][2].recoverable == 0
    for _ in range(4):
        prev = _rows(park)
        s.tick()
        assert c1.currentTime == park.currentTime - 1
        assert _rows(c1) == prev
    assert _rows(s.client_park(0)) == _rows(park)           # the others stayed in step


def test_lane_session_cloaked_ships_leave_the_other_clients_parks():
    """A CloakBall event removes the cloaked ship from every other client's park (not its own: the ego,
    _ticker.py:213-216); UncloakBall sends the others an AddBallsToPark of it (_ticker.py:139-163 on the server) and it
    comes back; the cloaked ship's own commands reach only its own client meanwhile."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS)
    s.start()
    s.tick()
    s.command("cloak_ball", 102, srv.DSTRESTORECLOAK, 1000.0)
    s.tick()
    assert 102 in s.client_park(1).balls and s.client_park(1).balls[102].isCloaked == srv.DSTRESTORECLOAK
    assert 102 not in s.client_park(0).balls and 102 not in s.client_park(2).balls
    s.command("go_to_direction", 102, 0.0, 0.0, 1.0)
    s.tick()
    assert s.client_park(1).balls[102].mode == GOTO and park.balls[102].mode == GOTO
    s.command("uncloak_ball", 102)
    s.tick()
    for p in (0, 1, 2):
        assert 102 in s.client_park(p).balls
        assert _rows(s.client_park(p))[102] == _rows(park)[102]


def test_lane_session_updates_per_client_per_tick():
    """Each client gets at most two updates a tick, its user (singlecast) update before the bubble's narrowcast, with
    wait_for_bubble and the count TWO when it gets both (_parkupdatebatcher.py:52-119, net/__init__.py:14-22)."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS)
    s.start()
    s.command("cloak_ball", 101, srv.DSTRESTORECLOAK, 1000.0)
    s.tick()
    s.command("go_to_direction", 101, 1.0, 0.0, 0.0)        # a cloaked ship: to its own character only
    s.command("set_speed_fraction", 102, 0.5)               # to the bubble
    s.tick()
    n, sent = s.sent_log[-1]
    per_client = {}
    for kind, to, method, state, wait, count in sent:
        for c in ([to] if kind == "single" else to):
            per_client.setdefault(c, []).append((kind, wait, int(count)))
    base = s.interests.client_base
    assert per_client[base + 0] == [("single", True, 2), ("narrow", False, 2)]
    assert per_client[base + 1] == [("narrow", False, 1)] and per_client[base + 2] == [("narrow", False, 1)]
    assert all(len(v) <= 2 for v in per_client.values())
    assert _rows(s.client_park(0)) == _seen_by(_rows(park), 101)    # both parts merged and applied this tick


def test_lane_session_start_up_command_before_the_first_pre_tick_blocks_its_client():
    """A command issued before the server's first pre-tick (the first OnTick has none, TIME-704) carries Actions' first
    stamp, 0 (_actions.py:31-34), while the park is at 1. The pilots' SetState is stamped 1, so the user update and the
    bubble update of that tick have different stamps: the SetState's entry keeps waiting for its bubble part for ever
    (_util.py:41-68, _ticker.py:93-96) and the client never applies anything."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS)
    s.tick()
    for ship, p in sorted(PILOTS.items(), key=lambda kv: kv[1]):
        s.batcher.send_full_state_update(s.interests.char_of[ship], ship)
    s.actions.go_to_direction(101, 1.0, 0.0, 0.0)           # before the first pre-tick (session tick 2)
    s.tick()
    assert s.sent_log[-1][1][0][3][0][0] == 1               # the SetState's stamp
    bubble = [st for kind, to, m, st, w, c in s.sent_log[-1][1] if kind == "narrow"]
    assert bubble and bubble[0][0][0] == 0                  # the command's stamp
    for _ in range(5):
        s.tick()
    for p in PILOTS.values():
        head = s.clients[p][1]._history[0]
        assert head[1] is True and head[0][0][1][0] == "SetState"
        assert len(s.client_park(p).balls) == 0


def test_lane_entry_two_ticks_ahead_waits_three_ahead_runs_on():
    """_ticker.py:100-102: an entry stamped one or two ticks ahead of the client's park waits for its stamp; one three
    or more ahead is applied at once, the park run forward to its stamp (_ticker.py:265-305). (Destiny's
    test_future_state_gets_applied_at_correct_time checks one tick ahead.)"""
    t = TestTicker("test_setstate")
    t.setUp()
    park = t._ballpark
    now = park.currentTime
    t._ticker.update([[now + 2, ("Stop", (C_BALL_ID_2,))]], False)
    t._tick(now)
    assert (park.currentTime, park.balls[C_BALL_ID_2].mode) == (now + 1, ORBIT)   # waiting
    t._tick(now + 1)
    assert (park.currentTime, park.balls[C_BALL_ID_2].mode) == (now + 2, ORBIT)   # still waiting
    t._tick(now + 2)
    assert (park.currentTime, park.balls[C_BALL_ID_2].mode) == (now + 3, STOP)    # applied at now + 2, then a tick
    now = park.currentTime
    t._ticker.update([[now + 3, ("Orbit", (C_BALL_ID_2, C_BALL_ID_1, 500.0))]], False)
    t._ticker.do_pre_tick()
    assert (park.currentTime, park.balls[C_BALL_ID_2].mode) == (now + 3, ORBIT)   # run forward three ticks


def test_lane_finding_a_non_interactive_balls_cloak_never_reaches_the_clients():
    """Finding (Destiny's code as written): Actions.cloak_ball sends the CloakBall event to the ball's OLD bubble when
    its old and new bubbles differ (_actions.py:628-638), and AdditionsAndDeletions moves the old bubble on only for
    user (interactive) balls (Thunkers.cpp:2887). A ball without a pilot keeps old bubble -1, so its CloakBall event goes
    to bubble -1, which no client is sent (_parkupdatebatcher.py:72-76): the clients keep the ball uncloaked, and its
    later commands, sent only to its (absent) character while it is cloaked (_ticker.py:96-104), never reach them."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS)
    s.start()
    s.tick()
    s.command("go_to_direction", 104, 0.0, 1.0, 0.0)
    s.tick()
    s.command("cloak_ball", 104, srv.DSTRESTORECLOAK, 1000.0)
    s.tick()
    b = park.lane.balls[104]
    assert (b.old_bubble, b.bubble, b.is_cloaked) == (-1, 0, srv.DSTRESTORECLOAK)
    s.command("stop", 104)
    s.tick()
    s.tick()
    assert park.balls[104].mode == STOP
    for p in PILOTS.values():
        c = s.client_park(p).balls[104]
        assert (c.isCloaked, c.mode) == (0, GOTO)           # never told: flying on its last command


def test_lane_finding_a_command_refused_for_a_cloaked_target_is_accepted_by_the_clients():
    """Finding (Destiny's code as written): Orbit(101 -> 103) queued before UncloakBall(103) in one tick is refused on
    the server, where 103 is still cloaked when the pre-tick applies it (_ticker.py:106-122; MOV-715's guard), and is
    sent to the bubble all the same. Each client gets the uncloak's AddBallsToPark(103) in its user update, applied
    before the bubble's update (net/__init__.py:14-22; merge_state_into_history appends the bubble's entries after the
    user's), so its Orbit finds 103 and is accepted: every client flies 101 in ORBIT, the server keeps it in STOP."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS)
    s.start()
    s.tick()
    s.command("cloak_ball", 103, srv.DSTRESTORECLOAK, 1000.0)
    s.tick()
    s.command("orbit", 101, 103, 2000.0)
    s.command("uncloak_ball", 103)
    s.tick()
    assert park.balls[101].mode == STOP
    for p in PILOTS.values():
        assert s.client_park(p).balls[101].mode == ORBIT


def test_lane_finding_a_cloak_before_a_follow_in_one_tick_parts_server_and_clients():
    """Finding (Destiny's code as written): the server applies CloakBall after the tick's other actions
    (_ticker.py:106-131) but every client applies the tick's events in their queue order. [CloakBall(101),
    FollowBall(104 -> 101)]: on the server the follow is accepted (101 not yet cloaked) and the cloak then releases the
    new follower (StopAllFollowers: 104 stops); each client cloaks 101 first and refuses the follow (MOV-715's guard),
    so 104 flies on in GOTO there."""
    park = _server_park(list(SHIPS))
    s = cl.Session(park, PILOTS)
    s.start()
    s.command("go_to_direction", 104, 1.0, 0.0, 0.0)
    s.tick()
    s.command("cloak_ball", 101, srv.DSTRESTORECLOAK, 1000.0)
    s.command("follow_ball", 104, 101, 500.0)
    s.tick()
    assert park.balls[104].mode == STOP
    for p in PILOTS.values():
        assert s.client_park(p).balls[104].mode == GOTO


# ================================== planted faults: each must be caught by the tests named beside it (this port's checks)

def _run(cases) -> int:
    """Run Destiny's transcribed test methods (TestCase class, names) and this port's checks (callables): the failures."""
    failed = 0
    for case in cases:
        if callable(case) and not isinstance(case, tuple):
            try:
                case()
            except Exception:
                failed += 1
            continue
        cls, names = case
        result = unittest.TestResult()
        unittest.TestSuite(cls(n) for n in names).run(result)
        failed += len(result.failures) + len(result.errors)
    return failed


def _wait_two(self):
    """do_pre_tick broken: an entry two ticks ahead is applied at once (the window is 1, not 2)."""
    while len(self._history) > 0:
        state, wait_for_bubble = self._history[0]
        if wait_for_bubble:
            return
        event_stamp = state[0][0]
        if event_stamp > self._current_time and event_stamp - self._current_time < 2:
            break
        self._real_flush_state(state)
        del self._history[0]
        if self._state_is_valid and self._should_rebase:
            self.store_state(mid_tick=True)
        if len(self._history) > 1:
            if self._history[0][1]:
                return
            self._ballpark._parent_Evolve()


def _merge_keeps_wait(state, history_list, wait_for_bubble):
    """merge broken: a same-stamp merge does not clear the entry's wait_for_bubble."""
    from collections import defaultdict
    entries_by_time = defaultdict(list)
    for entry in state:
        entries_by_time[entry[0]].append(entry)
    time_list = sorted(entries_by_time.keys())
    time_list_idx = 0
    for history_idx, history_time_entries in enumerate(history_list):
        history_time = history_time_entries[0][0][0]
        entry_time = time_list[time_list_idx]
        if history_time < entry_time:
            wait_for_bubble = False
            continue
        if history_time == entry_time:
            history_time_entries[0].extend(entries_by_time[entry_time])
        else:
            history_list.insert(history_idx, [entries_by_time[entry_time], wait_for_bubble])
            wait_for_bubble = False
        time_list_idx += 1
        if time_list_idx >= len(time_list):
            break
    for tl_idx in range(time_list_idx, len(time_list)):
        history_list.append([entries_by_time[time_list[tl_idx]], wait_for_bubble])


def _sync_without_rewind(self, stamp):
    """synchronize broken: an older stamp is applied where the park is, without the snapshot."""
    for _ in range(max(0, stamp - self._current_time)):
        self._ballpark._parent_Evolve()
    return 1


def _cloak_removes_ego_too(self, state):
    """_real_flush_state broken: a CloakBall removes the ball from every park, the ego's included."""
    orig = _ORIGINAL_REAL_FLUSH
    orig(self, state)
    for _, (name, args) in state:
        if name == "CloakBall" and args[0] in self._ballpark.balls:
            self.RemoveBall(args[0])


_ORIGINAL_REAL_FLUSH = Ticker._real_flush_state


def _rewind_without_evolving(self, stamp):
    """synchronize broken: a rewind reads the snapshot but does not evolve up to the stamp."""
    if stamp < self._current_time:
        last_state = None
        for item in self._states:
            if item[1] <= stamp:
                last_state = item[0]
        if not last_state:
            return 0
        self._ballpark._parent_ReadFullStateFromStream(last_state, 1)
        return 1
    for _ in range(stamp - self._current_time):
        self._ballpark._parent_Evolve()
    return 1


def _bubble_first(self):
    """Session broken: the bubble updates delivered before the user updates (Destiny requires the opposite)."""
    self.tick_no += 1
    n = self.tick_no
    for name, args in self.uplink.pop(n, []):
        getattr(self.actions, name)(*args)
    self.server.lane.on_tick(n * self.interval)
    sent = self.network.take()
    self.sent_log.append((n, sent))
    sent = [x for x in sent if x[0] == "narrow"] + [x for x in sent if x[0] == "single"]
    for kind, to, method, state, wait, count in sent:
        for client_id in ([to] if kind == "single" else to):
            self.downlink[n + self.down].append((client_id, state, wait))
    for client_id, state, wait in self.downlink.pop(n, []):
        self.clients[client_id - self.interests.client_base][1].update(state, wait)
    for p in sorted(self.clients):
        self.clients[p][0].lane.on_tick(n * self.interval)


def _session_check(down, up):
    def check():
        for seed in (1, 2):
            bad, desyncs, compared = _run_session(seed, down, up, ticks=16)
            assert bad == [] and desyncs == [(0, 0)] * len(PILOTS)
    return check


def _lag_check():
    test_lane_session_command_lag_is_down_plus_up(1, 1)


def _cloak_check():
    test_lane_session_cloaked_ships_leave_the_other_clients_parks()


FAULTS = {
    "wait_window_one_tick": (lambda mp: mp.setattr(Ticker, "do_pre_tick", _wait_two),
                             [test_lane_entry_two_ticks_ahead_waits_three_ahead_runs_on]),
    "merge_keeps_waiting": (lambda mp: mp.setattr(cl, "merge_state_into_history", _merge_keeps_wait),
                            [test_lane_session_updates_per_client_per_tick]),
    "no_rewind": (lambda mp: mp.setattr(Ticker, "synchronize_to_simulation_time", _sync_without_rewind),
                  [(TestTicker, ["test_synchronize_to_simulation_time_can_rewind_to_last_tick",
                                 "test_synchronize_to_simulation_time_can_not_rewind_before_first_tick"]),
                   test_lane_session_late_update_rewinds_and_the_clock_slips]),
    "rewind_without_evolving": (lambda mp: mp.setattr(Ticker, "synchronize_to_simulation_time",
                                                      _rewind_without_evolving),
                                [(TestTicker, ["test_synchronize_to_simulation_time_can_rewind_to_last_tick"]),
                                 test_lane_session_late_update_rewinds_and_the_clock_slips]),
    "cloak_removes_the_ego": (lambda mp: mp.setattr(Ticker, "_real_flush_state", _cloak_removes_ego_too),
                              [_cloak_check]),
    "bubble_before_user": (lambda mp: mp.setattr(cl.Session, "tick", _bubble_first),
                           [test_lane_finding_a_command_refused_for_a_cloaked_target_is_accepted_by_the_clients]),
    "uplink_one_tick_late": (lambda mp: mp.setattr(cl.Session, "command",
                                                   lambda self, name, *args: self.uplink[self.tick_no + 2 + self.up]
                                                   .append((name, args))),
                             [_lag_check]),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_lane_planted_faults_are_caught(fault, monkeypatch):
    patch, cases = FAULTS[fault]
    assert _run(cases) == 0, f"{fault}: the catching tests fail without the fault"
    patch(monkeypatch)
    assert _run(cases) > 0, f"the planted fault {fault} went unnoticed"

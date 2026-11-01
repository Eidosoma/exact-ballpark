# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer over the bubble partition (exact_ballpark.vec_server_space) against the scalar server layer (exact_ballpark.server on a master DestinyPark over FullBallpark, held to
Destiny's published server tests), bit for bit, every tick: every client's update list (its singlecast, then its
ship's bubble's narrowcast; stamps, events, argument types, AddBallsToPark streams byte for byte, RemoveBalls,
PackagedActions, wait_for_bubble, counts) and every ship's state and bubble, over worlds of several bubbles
(tests/server_space_cases.py); and the client's view (SpaceClientView) against exact_ballpark.client.Session's client
parks (what each client's park holds and its state, four delay pairs; tests/space_session.py).

On the CPU (VecOrientPark): development seed 1 and held-out seed 7, the update kinds reached, the client's view, two
findings pinned (a refused cloak hides the ship from every other client; a ship made non-interactive leaves its client
on its own), planted faults (each must be caught). On the GPU (skipped without one; VecModesPark): follows, orbits and
warps, the follower clean-up's Stops reached, its planted fault.
"""

from __future__ import annotations

import logging

import pytest
import torch

import server_space_cases as SS
import space_session as SSN
from exact_ballpark import client as cl
from exact_ballpark import server as srv
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_space as VSS

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
logging.getLogger("exact_ballpark.server").setLevel(logging.CRITICAL)
logging.getLogger("exact_ballpark.client").setLevel(logging.CRITICAL)
_RUNS: dict = {}
HELD_OUT = 7                                                # first run after the generators were frozen


def run(seed: int, n: int = 300, park: str = "orient", ticks: int = 16, device: str = "cpu") -> dict:
    k = (seed, n, park, ticks, device)
    if k not in _RUNS:
        _RUNS[k] = SS.run(SS.cases(seed, n, park, ticks), park, device)
    return _RUNS[k]


MINIMUM = {"single:RemoveBalls (own lists)": 50, "single:PackagedRemoveBalls": 100,
           "single:PackagedAddBallsToPark": 400, "single:AddBallsToPark": 100, "narrow:CloakBall": 100,
           "ticks_with_several_bubbles_narrowcasting": 100, "clients_with_two_updates": 400,
           "narrow:SetBallPosition": 200, "narrow:SetBallInteractive": 100}


def assert_exact(res: dict, minimum: dict | None = None, worlds: int = 200) -> None:
    assert not res["flagged"], res["flagged"][:10]
    assert not res["sends"], [str(d)[:600] for d in res["sends"][:3]]
    assert not res["rows"], res["rows"][:5]
    assert res["worlds_with_several_bubbles"] >= worlds
    for key, low in (minimum or {}).items():
        assert res["coverage"].get(key, 0) >= low, (key, res["coverage"].get(key, 0))


# ==================================================================================================== on the CPU

def test_the_server_layer_matches_the_reference_development_worlds():
    """Seed 1: 300 worlds x 16 ticks over VecOrientPark, most of them several bubbles: no update list or ship differs,
    and every update kind is reached (own-list removals, the bubble's packaged lists, uncloak notices, cloaks sent to
    a bubble, several bubbles narrowcasting in one tick, clients with two updates)."""
    assert_exact(run(1), MINIMUM)


def test_the_server_layer_matches_the_reference_held_out_worlds():
    """The held-out seed, first run after the generators were frozen."""
    assert_exact(run(HELD_OUT), MINIMUM)


@pytest.mark.parametrize("down,up", [(0, 0), (1, 0), (2, 1), (0, 2)])
def test_the_client_view_equals_the_sessions_client_parks(down, up):
    """After every tick, for every client that has heard its bubble throughout (synced): the balls its park holds
    equal the view's, and the state (time, positions, velocities, modes, goto points, speed fractions) of every ball
    it holds fresh; held balls that are not fresh (the finding 1: a non-interactive ball's cloak never reaches the
    clients) are counted, and occur."""
    checked, stale = [], []
    for seed in range(1, 13):
        SSN.session_and_view(seed, down, up, check=SSN.holdings_check(down, checked, stale))
    assert len(checked) >= 400 and sum(c[2] for c in checked) >= 1000
    assert stale


def _two_ships_session(calls_after_start: list, down: int = 0) -> tuple:
    """One world: ships 101 and 102 (both interactive, pilots 0 and 1) in one bubble; the Session and the batched
    server with its view, the calls issued after the start."""
    w = {"id": 0, "park": "orient", "do": False, "anchors": [(0.0, 0.0, 0.0)], "special": 101, "row": None,
         "pilots": {101: 0, 102: 1}, "ticks": [[]],
         "balls": [{"id": bid, "mass": 1e7, "radius": 50.0, "max_vel": 200.0, "agility": 0.5, "sf": 1.0,
                    "pos": [1000.0 * k, 2000.0, 3000.0], "vel": [0.0, 0.0, 0.0], "interactive": True,
                    "massive": False, "max_ang_vel": 1.0, "rot_agility": 0.9} for k, bid in ((1, 101), (2, 102))]}
    park = srv.DestinyPark(is_master=True)
    for b in w["balls"]:
        park.AddBall(b["id"], b["mass"], b["radius"], b["max_vel"], 1, 0, 0, 1, 0, *b["pos"], *b["vel"], b["agility"],
                     b["sf"])
    session = cl.Session(park, w["pilots"], down=down)
    server, index = SS.build([w], "orient", "cpu")
    view = VSS.SpaceClientView(server, down)
    session.start()
    server.evolve()
    view.record(int(server.a.clock()[0]))
    server.pre_tick()
    server.evolve()
    server.post_tick()
    view.record(int(server.a.clock()[0]))
    import server_cases as S
    for name, args in calls_after_start:
        session.command(name, *args)
        server.push(*S.pushes([dict(w, ticks=[[(name, args)]])], 0, "cpu")[0])
    for _ in range(3):
        session.tick()
        server.pre_tick()
        server.evolve()
        server.post_tick()
        view.record(int(server.a.clock()[0]))
    return session, server, view, index


def test_finding_a_refused_cloak_hides_the_ship_from_every_other_client():
    """New: Actions.cloak_ball queues any mode, the Ticker sends the CloakBall event to the bubble
    before the park refuses it (a mode of 0, or 128 and above as a signed char), and every client but the ship's own
    removes the ship on a CloakBall (net/client _ticker.py:213-216): the ship stays uncloaked on the server while the
    other pilots' clients no longer hold it (it reappears only with an AddBallsToPark: an uncloak, a bubble change)."""
    session, server, view, index = _two_ships_session([("cloak_ball", (101, 0, 1000.0))])
    assert session.server.lane.balls[101].is_cloaked == 0
    assert 101 not in session.client_park(1).lane.balls and 101 in session.client_park(0).lane.balls
    t, fields, present, synced, fresh = view.view()
    assert not bool(present[index[(0, 101)], index[(0, 102)]]) and bool(present[index[(0, 101)], index[(0, 101)]])


def test_finding_a_ship_made_non_interactive_leaves_its_client_on_its_own():
    """New: a pilot's ship made non-interactive leaves its bubble's interactive set, so its client
    is sent nothing more (no narrowcast, no bubble update); its park runs on alone and keeps the other ship flying on
    its last command while the server's turned it: the view marks that client not synced."""
    session, server, view, index = _two_ships_session(
        [("set_ball_interactive", (101, 0)), ("go_to_direction", (102, 1.0, 0.0, 0.0))])
    t, fields, present, synced, fresh = view.view()
    assert not bool(synced[index[(0, 101)]]) and bool(synced[index[(0, 102)]])
    server_mode = session.server.lane.balls[102].mode
    client_mode = session.client_park(0).lane.balls[102].mode
    assert server_mode != client_mode


# ---- planted faults: each changes the batched layer and must be caught by the comparison named ----

def _decode_narrow_all(server, batch=None, client_base=10000):
    """decode with every bubble's events in every pilot's narrowcast."""
    out = _ORIGINAL_DECODE(server, batch, client_base)
    b = batch or server.batch
    for w, per in enumerate(out):
        merged = []
        for j, s in enumerate(b.slots):
            if bool(s.ok[w]) and int(b.route[w, j]) == VS.BUBBLE:
                merged.append((int(b.stamp[w]), VS.event_of(s, w, server.a.ids.tolist())))
        for c, ups in per.items():
            per[c] = [(k, m, merged if k == "narrow" else st, wt, n) for k, m, st, wt, n in ups]
    return out


def _decode_own_packaged(server, batch=None, client_base=10000):
    """decode with an own-list RemoveBalls packaged as the bubble's."""
    out = _ORIGINAL_DECODE(server, batch, client_base)
    for per in out:
        for c, ups in per.items():
            fixed = []
            for k, m, st, wt, n in ups:
                st = [(t, ("PackagedAction", srv.Marshalled(((t, ev),)))) if ev[0] == "RemoveBalls" else (t, ev)
                      for t, ev in st]
                fixed.append((k, m, st, wt, n))
            per[c] = fixed
    return out


_ORIGINAL_DECODE = VSS.decode
_ORIGINAL_PUSH = VSS.SpaceServer.push
_ORIGINAL_TAPE_ONE = VSS.SpaceServer._tape_one
_ORIGINAL_IA_BUBBLE = VSS.SpaceServer.ia_bubble_of_rows
_ORIGINAL_IN_IA_OF = VSS.SpaceServer.in_ia_of
_ORIGINAL_APPLY_BATCH = VSS.SpaceClientView.apply_batch


def _push_cloak_current(self, kind, mask, ball, target=None, f=None, n=None, loc=None):
    k = VS.CODE[kind] if isinstance(kind, str) else int(kind)
    if k == VS.CLOAK:
        row = self.a.lookup(ball.to(torch.int64))
        return self._push_with_loc(k, mask, ball, target, f, n, self.bubbles_of(row.clamp(min=0)))
    return _ORIGINAL_PUSH(self, kind, mask, ball, target, f, n, loc)


def _tape_no_refile(self, name, slot, n=None, f=None):
    if name == "REFILE":
        return None
    return _ORIGINAL_TAPE_ONE(self, name, slot, n, f)


def _ia_by_bubble(self):
    return self.bubbles_of(self.a.arange)


def _notices_to_the_world(self, bubble_w):
    return torch.ones(self.a.R, dtype=torch.bool, device=self.dev) & (bubble_w[self.a.world] != -1)


def _apply_batch_keeping_cloaked(self, b):
    saved = list(b.slots)
    b.slots = [s for s in b.slots if s.kind != VS.CLOAK]
    try:
        _ORIGINAL_APPLY_BATCH(self, b)
    finally:
        b.slots = saved


def _server_caught(seed=1, n=150, ticks=12):
    def check():
        res = SS.run(SS.cases(seed, n, "orient", ticks), "orient", "cpu")
        return bool(res["sends"] or res["rows"])
    return check


def _view_caught():
    def check():
        try:
            for seed in range(1, 9):
                SSN.session_and_view(seed, 0, 0, check=SSN.holdings_check(0, [], []))
        except AssertionError:
            return True
        return False
    return check


FAULTS = {
    "narrowcast_of_every_bubble": ("SEN-712 (a bubble's events to its own pilots)",
                                   lambda mp: mp.setattr(VSS, "decode", _decode_narrow_all), _server_caught()),
    "own_list_removals_packaged": ("_parkupdatebatcher.py:285-289 (own lists sent raw)",
                                   lambda mp: mp.setattr(VSS, "decode", _decode_own_packaged), _server_caught()),
    "cloak_sent_to_its_current_bubble": ("_actions.py:628-638 (the old bubble while they differ)",
                                         lambda mp: mp.setattr(VSS.SpaceServer, "push", _push_cloak_current),
                                         _server_caught(seed=2, n=300, ticks=16)),
    "setters_not_refiled": ("SEN-706, SEN-707 (SetBallPosition re-files)",
                            lambda mp: mp.setattr(VSS.SpaceServer, "_tape_one", _tape_no_refile), _server_caught()),
    "narrowcast_by_the_ships_bubble": ("send_batch (the interactive set, not the bubble)",
                                       lambda mp: mp.setattr(VSS.SpaceServer, "ia_bubble_of_rows", _ia_by_bubble),
                                       _server_caught()),
    "uncloak_notices_to_the_whole_world": ("_ticker.py:139-163 (its bubble's interactive balls)",
                                           lambda mp: mp.setattr(VSS.SpaceServer, "in_ia_of",
                                                                 _notices_to_the_world), _server_caught()),
    "view_keeps_a_cloaked_ship": ("net/client _ticker.py:213-216 (a CloakBall removes the ball)",
                                  lambda mp: mp.setattr(VSS.SpaceClientView, "apply_batch",
                                                        _apply_batch_keeping_cloaked), _view_caught()),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_a_planted_fault_is_caught(fault, monkeypatch):
    rule, plant, caught = FAULTS[fault]
    with monkeypatch.context() as mp:
        plant(mp)
        assert caught(), (fault, rule)
    assert not caught(), "the unplanted layer must pass the same check"


# ==================================================================================================== on the GPU

@CUDA
def test_the_server_layer_over_vec_modes_park_matches_the_reference():
    """VecModesPark (follows and orbits refused across bubbles, warps to far top boxes), development and held-out
    seeds: no update list or ship differs; the follower clean-up's Stops occur."""
    for seed in (1, HELD_OUT):
        res = run(seed, 200, "modes", 16, "cuda")
        assert_exact(res, {"narrow:FollowBall": 50, "narrow:Orbit": 50, "narrow:WarpTo": 20,
                           "single:RemoveBalls (own lists)": 20}, worlds=100)
        assert res["coverage"]["follower_clean_up_stops"] > 0


def _no_cleanup(self, kind, a=None, b=None, f=None):
    out, outf = _ORIGINAL_QUERY(self, kind, a, b, f)
    if kind == "REMOTE_FOLLOWERS":
        out = out.clone()
        out[:, 0] = 0
    return out, outf


from exact_ballpark import vec_space as _VSP                  # noqa: E402
_ORIGINAL_QUERY = _VSP.VecSpace.query


@CUDA
def test_the_gpu_planted_fault_is_caught(monkeypatch):
    """The follower clean-up skipped (MOV-731): a follower left in another bubble flies on; caught."""
    monkeypatch.setattr(_VSP.VecSpace, "query", _no_cleanup)
    res = SS.run(SS.cases(1, 200, "modes", 16), "modes", "cuda")
    assert res["sends"] or res["rows"]

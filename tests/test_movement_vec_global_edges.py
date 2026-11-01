# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Global balls where this port left them refused: the client's view of the global notices, the empty
stream of make_ball_global of a ball not in the park, and the OnTick duplicate-global case made common.

- **The client's view** (exact_ballpark.vec_server_space.SpaceClientView over SpaceServer on a VecCollidePark made with
  global balls) against exact_ballpark.client.Session's client parks (tests/global_session.py: Destiny's client Ticker
  applying every update): after every tick, for every synced client, the balls its park holds and the state of those
  it holds fresh, four delay pairs; development seeds 1 to 12, held-out seeds 101 to 112 (first run after the
  generator's freeze); constructed worlds pin three of Destiny's behaviours; the planted faults must be caught.
- **The empty stream** (SpaceServer against the scalar server layer, tests/global_edge_cases.py's worlds with
  make_ball_global and make_ball_local of an id no world holds): every client's update list byte for byte and every
  ship's row; development seeds 1 and 7, held-out seed 29; planted faults.
- **The OnTick duplicate-global case** (tests/global_edge_cases.py's OnTick worlds, most with a moving ship made global
  just before and just after its world's first evolving OnTick): SpaceServer.on_tick against the scalar layer through
  its park's OnTick; development seeds 1 and 7, held-out seed 31; the planted fault on the duplicate test (the
  stream's bytes ignored), which its generator caught in one world, caught in most.

- **Global balls over the modes and orient parks** (the flag held by the partition layer's view): SpaceServer over
  VecOrientPark (the CPU) and VecModesPark (the card) against the scalar server layer with AddBall's isGlobal
  (tests/global_edge_cases.py's space_cases: server_space_cases' worlds with global flags, make_ball_global and
  make_ball_local); the space server's kernel (CudaSpaceServer) with AddBall's flags (the kinds it refuses left out);
  development seeds 1 and 7, held-out seed 41; planted faults.

The CPU cases run everywhere; the rest need a GPU (the batched server's park on the card: its torch path and the
kernel; VecModesPark)."""

import inspect
import logging
import textwrap

import pytest
import torch

import global_cases as G
import global_edge_cases as E
import global_session as GS
import ontick_cases as O
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_space as VSS

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
logging.getLogger("exact_ballpark.server").setLevel(logging.CRITICAL)
logging.getLogger("exact_ballpark.client").setLevel(logging.CRITICAL)
VIEW_DEV, VIEW_HELD = range(1, 13), range(101, 113)
DEV, DEV2, HELD_SERVER, HELD_ONTICK, HELD_SPACE = 1, 7, 29, 31, 41
PAIRS = ((0, 0), (1, 0), (2, 1), (0, 2))
_RUNS: dict = {}
T = GS.TOP

# what the development seeds reached (seeds 1 to 12, four delay pairs; the minimums are about half of it)
VIEW_MIN = {"held: a ball of another bubble, fresh": 300, "held: a ball of another bubble, stale": 300,
            "narrow: AddBallsToPark": 200, "narrow: AddBallsToPark, empty stream": 25, "narrow: RemoveBall": 25}


def view(seeds, device: str = "cpu", kernel: str = "torch") -> dict:
    k = ("view", tuple(seeds), device, kernel)
    if k not in _RUNS:
        _RUNS[k] = GS.run(seeds, PAIRS, device=device, kernel=kernel)
    return _RUNS[k]


def assert_view(res: dict, seeds: int = 12, minimum: bool = True) -> None:
    assert res["checked"] >= 60 * seeds and res["held"] >= 200 * seeds, res     # (seeds 1 to 12: 144 and 503 a seed)
    if minimum:
        for key, low in VIEW_MIN.items():
            assert res["coverage"].get(key, 0) >= low, (key, res["coverage"].get(key, 0))


def server(seed: int, device: str = "cpu", kernel: str = "torch") -> dict:
    k = ("server", seed, device, kernel)
    if k not in _RUNS:
        cs = E.server_cases(seed, 60, 14)
        res = G.run_server(cs, device, kernel)
        res.pop("server")
        res["unknown_calls"] = sum(1 for w in cs for tk in w["ticks"] for c in tk if c[1][0] == E.UNKNOWN)
        _RUNS[k] = res
    return _RUNS[k]


def assert_server(res: dict) -> None:
    assert res["sends"] == [] and res["rows"] == [], (res["sends"][:2], res["rows"][:2])
    assert len(res["flagged"]) <= 2 and res["updates"] > 1000 and res["unknown_calls"] >= 100
    assert res["coverage"]["global: AddBallsToPark narrowcast entries"] > \
        res["coverage"]["global: AddBallsToPark entries of a global ball"] + 100      # the empty streams among them


def ontick(seed: int, device: str = "cpu", kernel: str = "torch") -> dict:
    k = ("ontick", seed, device, kernel)
    if k not in _RUNS:
        cs = E.ontick_cases(seed, 60, 14)
        res = O.run(cs, device, kernel)
        res.pop("server")
        res["dup_worlds"] = sum(1 for w in cs if w["dup_global"] is not None)
        _RUNS[k] = res
    return _RUNS[k]


def assert_ontick(res: dict) -> None:
    assert res["sends"] == [] and res["rows"] == [] and res["flagged"] == [], (res["sends"][:2], res["rows"][:2])
    assert res["dup_worlds"] >= 30 and res["updates"] > 1000


def space(seed: int, park: str, device: str, kernel_server: bool = False) -> dict:
    k = ("space", seed, park, device, kernel_server)
    if k not in _RUNS:
        cs = E.space_cases(seed, 200, park, 14)
        cls = None
        if kernel_server:                       # the kernel refuses the global kinds: AddBall's flags alone
            from exact_ballpark import vec_server_space_cuda as Q
            cls = Q.CudaSpaceServer
            for w in cs:
                w["ticks"] = [[c for c in tk if c[0] not in ("make_ball_global", "make_ball_local")]
                              for tk in w["ticks"]]
        res = E.run_space(cs, park, device, server_cls=cls)
        res.pop("server")
        _RUNS[k] = res
    return _RUNS[k]


def assert_space(res: dict, calls: bool = True) -> None:
    assert res["sends"] == [] and res["rows"] == [], (res["sends"][:2], res["rows"][:2])
    assert len(res["flagged"]) <= 4 and res["updates"] > 2000 and res["worlds_with_several_bubbles"] >= 100
    cov = res["coverage"]
    assert cov["global_ship_ticks"] > 1000 and cov.get("single:PackagedAddBallsToPark", 0) >= 200
    if calls:                                   # (seed 1: 3,003 and 444 over VecOrientPark; 3,279 and 596 VecModesPark)
        assert cov["global: AddBallsToPark narrowcast entries"] >= 1000
        assert cov["global: RemoveBall narrowcast entries"] >= 150


# ==================================================================================================== on the CPU

def test_the_client_view_equals_the_sessions_client_parks_with_global_balls():
    """Development seeds 1 to 12, four delay pairs: every synced client's holdings equal its park's, and every ball it
    holds fresh equals its park's (time, positions, velocities, modes, goto points, speed fractions); clients hold
    balls of other bubbles (global ones), fresh and stale; the notices reach them (AddBallsToPark, its empty stream,
    RemoveBall)."""
    assert_view(view(VIEW_DEV))


def test_held_out_client_view_equals_the_sessions_client_parks():
    """Held-out seeds 101 to 112 (first run after the generator's freeze)."""
    assert_view(view(VIEW_HELD))


def test_finding_make_ball_local_and_a_jump_remove_the_ship_from_its_own_client():
    """New: make_ball_local queues BallNotGlobal with the ship's bubble at the call; a jump queued in the
    same tick re-files it into another bubble before the history is distributed, so the RemoveBall goes to the new
    bubble, whose clients (its own pilot's among them) remove the ship: its own client no longer holds it."""
    s, srv_, view_, rows_of, checked, stale = GS.two_ships(
        [("make_ball_local", (101,)), ("set_ball_position", (101, 3 * T + 1000.0, -2 * T, T, False))], far=True,
        glob=(True, False))
    assert 101 in s.server.lane.balls and 101 not in s.client_park(0).lane.balls
    t, fields, present, synced, fresh = view_.view()
    assert not bool(present[rows_of[101], rows_of[101]]) and bool(synced[rows_of[101]]) and checked


def test_finding_both_routes_in_one_tick_leave_the_own_ship_behind():
    """New: a ship's command before its cloak goes to its bubble, one after it to its own character;
    the client applies its singlecast first, so its own ship ends on the earlier command (the server's on the later):
    the view marks it stale (before this change it called it fresh)."""
    s, srv_, view_, rows_of, checked, stale = GS.two_ships(
        [("go_to_point", (101, 9000.0, 2000.0, 3000.0)), ("cloak_ball", (101, 1, 1000.0)),
         ("go_to_point", (101, -9000.0, 2000.0, 3000.0))], far=False)
    assert s.server.lane.balls[101].goto[0] == -9000.0 and s.client_park(0).lane.balls[101].goto[0] == 9000.0
    assert (3, 0, 101) in stale


def test_finding_a_cloaked_ship_made_global_is_revealed():
    """New: make_ball_global writes the ship's stream, its cloak among the flags, and every bubble with
    an interactive ball hears it, so the other pilot's client adds the cloaked ship (stale: it is cloaked)."""
    s, srv_, view_, rows_of, checked, stale = GS.two_ships(
        [("cloak_ball", (101, 1, 1000.0))], far=False,
        then=([("make_ball_global", (101,)), ("go_to_point", (101, 9000.0, 2000.0, 3000.0))],))
    assert s.server.lane.balls[101].is_cloaked and 101 in s.client_park(1).lane.balls
    t, fields, present, synced, fresh = view_.view()
    assert bool(present[rows_of[101], rows_of[102]]) and not bool(fresh[rows_of[101], rows_of[102]])


def test_a_batch_over_global_balls_needs_the_set_state():
    cs = G.server_cases(DEV, 2, 1)
    gs, _ = G.build_server(cs, "cpu")
    gs.push("AddBallsToPark", torch.tensor([True, True]), torch.tensor([cs[0]["balls"][0]["id"],
                                                                         cs[1]["balls"][0]["id"]]))
    gs.pre_tick()
    gs.evolve()
    gs.post_tick()
    v = VSS.SpaceClientView(gs, 0)
    with pytest.raises(ValueError, match="set_state"):
        v.record(0)
    v.set_state()
    v.record(int(gs.a.clock()[0]))


def test_the_empty_stream_matches_the_reference_development_worlds():
    """make_ball_global of a ball not in the park: its empty stream to every bubble with an interactive ball, byte for
    byte (seeds 1 and 7)."""
    for seed in (DEV, DEV2):
        assert_server(server(seed))


def test_held_out_empty_stream_matches_the_reference():
    assert_server(server(HELD_SERVER))


def test_on_tick_duplicate_global_matches_the_reference_development_worlds():
    for seed in (DEV, DEV2):
        assert_ontick(ontick(seed))


def test_held_out_on_tick_duplicate_global_matches_the_reference():
    assert_ontick(ontick(HELD_ONTICK))


def test_global_balls_over_the_orient_park_match_the_reference_development_worlds():
    """The flags held by the layer's view over VecOrientPark (seeds 1 and 7, 200 worlds x 14 ticks): every update list
    and every ship's row and flag equal."""
    for seed in (DEV, DEV2):
        assert_space(space(seed, "orient", "cpu"))


def test_held_out_global_balls_over_the_orient_park():
    assert_space(space(HELD_SPACE, "orient", "cpu"))


def test_view_flag_refusals():
    from exact_ballpark import vec_space as SP
    cs = E.space_cases(DEV, 2, "orient", 1)
    srv_, _ = E.SS.build(cs, "orient", "cpu")
    with pytest.raises(ValueError, match="global_balls"):           # a view without the flags
        srv_.space.view.set_global(torch.ones(srv_.a.R, dtype=torch.bool), True)
    gs, _ = G.build_server(G.server_cases(DEV, 2, 1), "cpu")
    with pytest.raises(ValueError, match="itself"):                 # a VecCollidePark holds its own
        VSS.space_for(gs.a, global_balls=[False] * gs.a.R)
    assert SP.ModesView.global_flags(SP.ParkView.__new__(SP.ParkView)) is None


# ---- planted faults (each must make its comparison differ) ----

VIEW_FAULTS = {                                  # (method, the line's text, its replacement): each occurs once
    "global_add_ignored": ("apply_batch", "m[ball[got], u[got]] = True", "pass"),
    "global_remove_ignored": ("apply_batch", "m[ball[got], u[got]] = False", "pass"),
    "remove_to_its_own_bubble_too": ("apply_batch", "& (narrow != b.rbubble[:, j][world])", ""),
    "set_state_without_global_balls": ("set_state", "| self._global_rows()[:, None]", ""),
    "global_add_always_fresh": ("apply_batch", "fr[ball[got], u[got]] = cur[got]", "fr[ball[got], u[got]] = True"),
    "dead_balls_kept_fresh": ("record", "self.fresh &= ~dead.to(BOOL).to(self.fresh.device).reshape(-1)[:, None]",
                              "pass"),
    "missed_action_kept_fresh": ("apply_batch", "miss = valid & ~hd", "miss = valid & ~valid"),
    "both_routes_kept_fresh": ("apply_batch", "fr &= ~(by_char & by_narrow)", "pass"),
    "own_ship_kept_on_remove": ("apply_batch", "m[ball[got], u[got]] = False",
                                "own = got & (ball != u); m[ball[own], u[own]] = False"),
    "actions_before_the_add_ignored": ("apply_batch", "& ~touched[safe]", ""),
    "character_events_before_the_add_ignored": ("apply_batch", "& ~by_char[safe, u])", ")"),
}

CONSTRUCTED = {
    "own_ship_removed": dict(calls=[("make_ball_local", (101,)),
                                    ("set_ball_position", (101, 3 * T + 1000.0, -2 * T, T, False))],
                             far=True, glob=(True, False)),
    "both_routes": dict(calls=[("go_to_point", (101, 9000.0, 2000.0, 3000.0)), ("cloak_ball", (101, 1, 1000.0)),
                               ("go_to_point", (101, -9000.0, 2000.0, 3000.0))], far=False),
    "character_after_add": dict(calls=[("cloak_ball", (101, 1, 1000.0))], far=False,
                                then=([("make_ball_global", (101,)), ("go_to_point", (101, 9000.0, 2000.0, 3000.0))],)),
}


def _plant_view(mp, name: str) -> None:
    meth, old, new = VIEW_FAULTS[name]
    src = textwrap.dedent(inspect.getsource(getattr(VSS.SpaceClientView, meth)))
    assert src.count(old) == 1, (name, src.count(old))
    src = src.replace(old, new).replace("super().", "super(SpaceClientView, self).")
    ns: dict = {}
    exec(compile(src, f"<{name}>", "exec"), VSS.__dict__, ns)
    mp.setattr(VSS.SpaceClientView, meth, ns[meth])


def _view_caught() -> bool:
    for kw in CONSTRUCTED.values():
        try:
            GS.two_ships(**kw)
        except AssertionError:
            return True
    for seed in VIEW_DEV:
        for down, up in PAIRS:
            try:
                GS.session_and_view(seed, down, up, check=GS.global_check(down, [], [], GS.Counter()))
            except AssertionError:
                return True
    return False


@pytest.mark.parametrize("fault", sorted(VIEW_FAULTS))
def test_a_planted_view_fault_is_caught(fault, monkeypatch):
    """The global AddBallsToPark or RemoveBall ignored, the RemoveBall sent to its own bubble too, the SetState without
    the global balls, a global stream always fresh, the dead bubbles' balls, a missed action or both routes in one tick
    kept fresh, the own ship kept on RemoveBall, the actions or the character events before a global stream ignored:
    each is seen by the constructed worlds or the development seeds."""
    with monkeypatch.context() as mp:
        _plant_view(mp, fault)
        assert _view_caught(), fault
    assert not _view_caught(), "the unplanted view must pass the same check"


def _e1(monkeypatch):
    push = VSS.SpaceServer._push_with_loc

    def faulty(self, k, mask, ball, target, f, n, loc):           # the empty stream not queued
        row = self.a.lookup(ball.to(torch.int64).to(self.dev))
        return push(self, k, mask.to(torch.bool).to(self.dev) & (row >= 0), ball, target, f, n, loc)
    monkeypatch.setattr(VSS.SpaceServer, "_push_with_loc", faulty)


def _decode_with(monkeypatch, change):
    dec = VSS.decode

    def faulty(server, batch=None, client_base=10000):
        b = batch or server.batch
        for s in b.slots:
            if s.kind == VS.GLOBAL_ON:
                change(server, s)
        return dec(server, batch, client_base)
    monkeypatch.setattr(VSS, "decode", faulty)


def _e2(monkeypatch):                                           # written as the last row's stream
    def change(server, s):
        s.row = torch.where(s.row < 0, torch.full_like(s.row, server.a.R - 1), s.row)
    _decode_with(monkeypatch, change)


def _e3(monkeypatch):                                           # the empty stream at time 0
    def change(server, s):
        s.fields_clock = torch.where(s.row < 0, torch.zeros_like(s.fields_clock), s.fields_clock)
    _decode_with(monkeypatch, change)


def _stream_bytes_ignored(monkeypatch):                         # the fault on its duplicate test
    push = VSS.SpaceServer._push_with_loc

    def faulty(self, k, mask, ball, target, f, n, loc):
        return push(self, k, mask, ball, target, f, None if k == VS.GLOBAL_ON else n, loc)
    monkeypatch.setattr(VSS.SpaceServer, "_push_with_loc", faulty)


@pytest.mark.parametrize("fault", ["empty_stream_not_queued", "empty_stream_as_a_ball", "empty_stream_at_time_zero"])
def test_a_planted_empty_stream_fault_is_caught(fault, monkeypatch):
    {"empty_stream_not_queued": _e1, "empty_stream_as_a_ball": _e2, "empty_stream_at_time_zero": _e3}[fault](monkeypatch)
    res = G.run_server(E.server_cases(DEV, 60, 14), "cpu")
    assert len(res["sends"]) >= 20


def test_the_duplicate_tests_fault_is_caught_in_most_worlds(monkeypatch):
    """the planted fault (AddBallsToPark's stream left out of the duplicate test) differs in most worlds here."""
    _stream_bytes_ignored(monkeypatch)
    res = O.run(E.ontick_cases(DEV, 60, 14), "cpu")
    assert len({x[1] for x in res["sends"]}) >= 30


def _m1(monkeypatch):                                           # the layer reads no flag (the Park's pointer null)
    from exact_ballpark import vec_space as SP
    orig = SP.VecSpace._park_struct

    def faulty(self):
        st, keep = orig(self)
        st.global_ = None
        return st, keep
    monkeypatch.setattr(SP.VecSpace, "_park_struct", faulty)


def _m2(monkeypatch):                                           # SetBallGlobal not applied by the view
    from exact_ballpark import vec_space as SP
    monkeypatch.setattr(SP.ParkView, "set_global", lambda self, rows, flag: None)


@pytest.mark.parametrize("fault", ["layer_reads_no_flag", "set_global_ignored"])
def test_a_planted_view_flag_fault_is_caught(fault, monkeypatch):
    {"layer_reads_no_flag": _m1, "set_global_ignored": _m2}[fault](monkeypatch)
    res = E.run_space(E.space_cases(DEV, 200, "orient", 14), "orient", "cpu")
    assert len(res["sends"]) >= 20


# ==================================================================================================== on the GPU

@CUDA
@pytest.mark.parametrize("kernel", ["torch", "cuda"])
def test_the_client_view_on_the_card_equals_the_sessions_client_parks(kernel):
    """The batched server on the card (the park's torch path; the kernel), development seeds 1 to 6."""
    assert_view(view(range(1, 7), "cuda", kernel), 6, minimum=False)


@CUDA
def test_held_out_client_view_on_the_card():
    """Held-out seeds 101 to 106 on the card (the kernel; first card run after the freeze)."""
    assert_view(view(range(101, 107), "cuda", "cuda"), 6, minimum=False)


@CUDA
@pytest.mark.parametrize("kernel", ["torch", "cuda"])
def test_the_empty_stream_on_the_card_matches_the_reference(kernel):
    assert_server(server(DEV, "cuda", kernel))


@CUDA
def test_held_out_edges_on_the_card():
    """Held-out seeds on the card (the kernel): the empty stream's 29 and OnTick's 31."""
    assert_server(server(HELD_SERVER, "cuda", "cuda"))
    assert_ontick(ontick(HELD_ONTICK, "cuda", "cuda"))


@CUDA
@pytest.mark.parametrize("kernel", ["torch", "cuda"])
def test_on_tick_duplicate_global_on_the_card(kernel):
    assert_ontick(ontick(DEV, "cuda", kernel))


@CUDA
def test_global_balls_over_the_modes_park_on_the_card_match_the_reference():
    """The flags held by the layer's view over VecModesPark (SpaceServer, the torch path; follows, orbits, warps)."""
    assert_space(space(DEV, "modes", "cuda"))


@CUDA
def test_the_space_servers_kernel_reads_addballs_global_flags():
    """CudaSpaceServer with AddBall's flags held by the view (its partition operations read them, its snapshots carry
    them); the global kinds stay refused."""
    assert_space(space(DEV, "modes", "cuda", kernel_server=True), calls=False)
    from exact_ballpark import vec_server_space_cuda as Q
    cs = E.space_cases(DEV, 2, "modes", 1)
    ks, _ = E.build_space(cs, "modes", "cuda", server_cls=Q.CudaSpaceServer)
    with pytest.raises(ValueError, match="not applied"):
        ks.push("AddBallsToPark", torch.tensor([True, True]), torch.tensor([cs[0]["balls"][0]["id"],
                                                                             cs[1]["balls"][0]["id"]]))


@CUDA
def test_held_out_global_balls_over_the_modes_park_and_the_kernel():
    """Held-out seed 41 on the card: the torch path with the calls, the kernel with AddBall's flags."""
    assert_space(space(HELD_SPACE, "modes", "cuda"))
    assert_space(space(HELD_SPACE, "modes", "cuda", kernel_server=True), calls=False)


@CUDA
def test_the_kernels_snapshot_fault_is_caught(monkeypatch):
    """The kernel path's streams written from the adapter's fields alone (no global flag: the fault found
    in development): the AddBall flags' streams differ."""
    from exact_ballpark import vec_server_space_cuda as Q
    monkeypatch.setattr(Q.CudaSpaceServer, "_stream_fields", lambda self, copy=False: self.a.stream_fields())
    cs = E.space_cases(DEV, 200, "modes", 14)
    for w in cs:
        w["ticks"] = [[c for c in tk if c[0] not in ("make_ball_global", "make_ball_local")] for tk in w["ticks"]]
    res = E.run_space(cs, "modes", "cuda", server_cls=Q.CudaSpaceServer)
    assert len(res["sends"]) >= 10

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer batched (exact_ballpark.vec_server) against the scalar server layer (exact_ballpark.server, Destiny's
net/server transcribed, on a master DestinyPark over FullBallpark), bit for bit, every tick: the park's state of every
ball (doubles by their bits) and the update batches each client receives (stamps, events and their argument types,
the AddBallsToPark streams byte for byte, wait_for_bubble and the counts); on the CPU and (when one is visible) the GPU.

Generated command streams (tests/server_cases.py): several calls a tick, conflicting commands, a Stop in the same tick,
the same call twice in a row (dropped) and with another between (kept), cloaks and uncloaks in both orders, local and
non-local SetBallPosition, follows and orbits of a cloaking ship, warps and warp-ins with followers, every setter, with
and without dynamical orientation, refused values now and then; on VecModesPark (every flight mode; its tick a CUDA
kernel, so on the CPU the pre-tick alone, as Destiny's own tests drive the Ticker), VecOrientPark and VecCollidePark
(Destiny's collisions). Development worlds are seed 1, held-out worlds seed 2.

Then: the paths the streams must reach; the refusals; a world tiled to many worlds; a CUDA graph's ticks against eager
ticks; the client's view (exact_ballpark.vec_server.ClientView) against exact_ballpark.client.Session's client parks;
planted faults, each caught by the tests named beside it.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter

import pytest
import torch

import server_cases as S
from exact_ballpark import client as cl
from exact_ballpark import server as srv
from exact_ballpark import vec_server as VS

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
logging.getLogger("exact_ballpark.server").setLevel(logging.CRITICAL)   # the batcher's stamp check logs every tick

# (park, dynamical orientation, evolve) on the CPU: VecModesPark's whole tick too (its tick on tensors)
CPU_RUNS = [("orient", False, True), ("orient", True, True), ("orient", False, False), ("modes", False, False),
            ("modes", True, False), ("modes", False, True), ("modes", True, True), ("collide", False, True),
            ("collide", False, False)]
GPU_RUNS = [("modes", False, True), ("modes", True, True), ("orient", False, True), ("orient", True, True),
            ("collide", False, True)]


def _check(res: dict) -> None:
    assert res["compared"] > 0 and res["updates"] > 0
    assert res["rows"] == [], f"{len(res['rows'])} ball states differ, first {res['rows'][:3]}"
    assert res["sends"] == [], f"{len(res['sends'])} update lists differ, first {str(res['sends'][:1])[:800]}"


@pytest.mark.parametrize("park,do,evolve", CPU_RUNS)
@pytest.mark.parametrize("seed", [1, 2])
def test_matches_the_scalar_server_layer_on_the_cpu(park, do, evolve, seed):
    _check(S.run(S.cases(seed, 40, park, do=do, ticks=14), park, "cpu", do=do, evolve=evolve))


@CUDA
@pytest.mark.parametrize("park,do,evolve", GPU_RUNS)
@pytest.mark.parametrize("seed", [1, 2])
def test_matches_the_scalar_server_layer_on_the_gpu(park, do, evolve, seed):
    kw = {"contact_slots": 12} if park == "collide" else {}
    _check(S.run(S.cases(seed, 40, park, do=do, ticks=14), park, "cuda", do=do, evolve=evolve, **kw))


@CUDA
@pytest.mark.parametrize("park,do,kernel", [("orient", True, "cuda"), ("collide", False, "cuda")])
def test_matches_on_the_parks_cuda_kernels(park, do, kernel):
    kw = {"contact_slots": 12} if park == "collide" else {}
    _check(S.run(S.cases(1, 40, park, do=do, ticks=14), park, "cuda", do=do, kernel=kernel, **kw))


def _coverage(park: str, do: bool, evolve: bool) -> Counter:
    cs = S.cases(1, 40, park, do=do, ticks=14)
    server, index = S.build(cs, park, "cpu", do=do)
    c = Counter()
    for t in range(14):
        for kind, mask, ball, target, f, n, loc in S.pushes(cs, t, "cpu"):
            sl = server.push(kind, mask, ball, target, f, n, loc)
            c["dropped"] += int((mask & ~sl.ok).sum())
        server.pre_tick(snapshot=False)
        b = server.batch
        for j, sl in enumerate(b.slots):
            for code in (VS.NONE, VS.BUBBLE, VS.CHARACTER):
                c[(VS.KINDS[sl.kind], code)] += int(((b.route[:, j] == code) & sl.ok).sum())
            if sl.kind == VS.CLOAK:
                c["cloak_to_old_bubble"] += int((sl.ok & (sl.loc == -1)).sum())
            if sl.kind == VS.SETPOS:
                c["local_move"] += int((sl.ok & (sl.loc == 1)).sum())
        c["notices"] += sum(int(rec.sum()) for _, rec, _ in b.notices)
        c["additions"] += int(b.additions[0].sum())
        if evolve:
            server.evolve()
        server.post_tick()
        c["two_updates"] += int((b.has_character & b.gets_bubble).sum())
    return c


@pytest.mark.parametrize("park,do,evolve", [("modes", False, False), ("orient", True, True),
                                            ("collide", False, True)])
def test_the_streams_reach_every_path(park, do, evolve):
    """Floors on the development worlds: duplicates dropped, every kind routed to the bubble, cloaked ships' events to
    their characters, CloakBall to the old bubble (-1: a ship not yet a user ball), local moves, uncloak notices, the
    first tick's additions, clients with two updates in a tick."""
    c = _coverage(park, do, evolve)
    for key in ("dropped", "notices", "additions", "two_updates", "cloak_to_old_bubble"):
        assert c[key] >= 5, (key, c[key])
    applied = S.build(S.cases(1, 2, park, do=do, ticks=1), park, "cpu", do=do)[0].a.kinds
    for k, code in VS.CODE.items():
        if code not in applied or k in ("SetMaxAngularVelocity", "SetBallTroll", "SetBallFree"):
            continue                    # (the two kinds: these frozen worlds have none; their routes and
                                        # updates are held to the scalar layer by tests/test_movement_vec_server_g11.py)
        want = VS.CHARACTER if k == "UncloakBall" else VS.BUBBLE
        assert c[(k, want)] >= 3, (k, want, c[(k, want)])
    assert sum(v for key, v in c.items() if isinstance(key, tuple) and key[1] == VS.CHARACTER) >= 20
    if park != "collide":
        assert c["local_move"] >= 5


def test_refusals():
    """A kind the park does not apply, a turning setter without dynamical orientation (Actions raises there), and a
    world whose bubble died (no interactive ball at an evolve: TIME-714 on the master) are refused."""
    cs = S.cases(1, 3, "orient", ticks=1)
    server, index = S.build(cs, "orient", "cpu")
    W = server.W
    z = torch.zeros(W, dtype=torch.int64)
    with pytest.raises(ValueError, match="FollowBall is not applied"):
        server.push("FollowBall", torch.ones(W, dtype=torch.bool), z + 101, z + 102)
    with pytest.raises(ValueError, match="SetBallAngularVelocity is not applied"):
        server.push("SetBallAngularVelocity", torch.ones(W, dtype=torch.bool), z + 101)
    anchor = torch.tensor([w["anchor"] for w in cs])
    others = [b["id"] for b in cs[0]["balls"] if b["id"] != cs[0]["anchor"]]
    mask = torch.tensor([True] + [False] * (W - 1))
    server.push("SetBallInteractive", mask, anchor, n=z)
    for k, bid in enumerate(others):
        server.push("SetBallInteractive", mask, torch.full((W,), bid), n=z)
    server.tick()
    with pytest.raises(RuntimeError, match="no interactive ball"):
        server.check()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_a_world_alone_among_many_and_reversed(device):
    """A world's ticks do not depend on the other worlds: 30 worlds together, each alone, and in reverse order give
    the same states and updates (the pre-tick only on the CPU for VecModesPark; orient with evolving)."""
    park = "orient" if device == "cpu" else "modes"
    cs = S.cases(3, 30, park, ticks=10)
    together = _final(cs, park, device)
    alone = {}
    for w in cs:
        alone.update(_final([w], park, device))
    rev = _final(list(reversed(cs)), park, device)
    assert together == alone == rev


def _final(cs: list, park: str, device: str) -> dict:
    """{(world, ball): the last tick's fields, and per world the last tick's update lists (canonical)}."""
    server, index = S.build(cs, park, device)
    for t in range(len(cs[0]["ticks"])):
        for kind, *data in S.pushes(cs, t, server.dev):
            server.push(kind, *data)
        server.pre_tick(snapshot=True)
        server.evolve()
        server.post_tick()
    out = {k: S.canon(list(v)) for k, v in S.batched_rows(server, cs, index, park, False).items()}
    for w, sends in zip(cs, VS.decode(server)):
        out[("sends", w["id"])] = S.canon(S._listify(sends))
    return out


@CUDA
@pytest.mark.parametrize("park", ["modes", "orient", "collide"])
def test_tiled_to_many_worlds(park):
    """40 worlds tiled to 100, 1,000 and 10,000 worlds in one park: after the last tick every copy's states and updates
    equal its original's, and the originals equal the scalar layer's."""
    cs = S.cases(1, 40, park, ticks=8)
    kw = {"contact_slots": 12} if park == "collide" else {}
    _check(S.run(cs, park, "cuda", record_every=False, **kw))
    first = _final_kw(cs, park, "cuda", **kw)
    for copies in (100, 1000, 10000):
        tiled = [dict(cs[k % len(cs)], id=k) for k in range(copies)]
        got = _final_kw(tiled, park, "cuda", **kw)
        assert len(got) == sum(len(cs[k % len(cs)]["balls"]) + 1 for k in range(copies))
        for key, v in got.items():
            o = (key[0], key[1] % len(cs)) if key[0] == "sends" else (key[0] % len(cs), key[1])
            assert v == first[o], (copies, key)


def _final_kw(cs, park, device, **kw):
    server, index = S.build(cs, park, device, **kw)
    for t in range(len(cs[0]["ticks"])):
        for kind, *data in S.pushes(cs, t, server.dev):
            server.push(kind, *data)
        server.pre_tick(snapshot=True)
        server.evolve()
        server.post_tick()
    out = {k: S.canon(list(v)) for k, v in S.batched_rows(server, cs, index, park, False).items()}
    for w, sends in zip(cs, VS.decode(server)):
        out[("sends", w["id"])] = S.canon(S._listify(sends))
    return out


# ============================================================================== a CUDA graph against eager ticks

def _fixed_schedule_case(seed: int, worlds: int, park: str, do: bool, ticks: int) -> tuple:
    """A fixed schedule of pushes every tick (a game's shape: each pilot's ship GotoDirection then SetSpeedFraction,
    then a mixed tail of a Stop, a cloak and an uncloak of the special ship and, on VecModesPark, an Orbit) with random
    arguments and masks, as per-tick input tensors."""
    r = random.Random(f"graph-{park}-{seed}")
    cs = S.cases(seed, worlds, park, do=do, ticks=1)
    sched = []
    for k in range(3):
        sched += [("GotoDirection", k), ("SetSpeedFraction", k)]
    sched += [("Stop", 0), ("CloakBall", "special"), ("UncloakBall", "special")]
    if park == "modes":
        sched += [("Orbit", 1)]
    inputs = []
    for t in range(ticks):
        per = []
        for kind, who in sched:
            W = len(cs)
            mask = torch.tensor([r.random() < 0.8 for _ in range(W)])
            ball = torch.zeros(W, dtype=torch.int64)
            target = torch.zeros(W, dtype=torch.int64)
            f = torch.zeros(W, 4, dtype=torch.float64)
            n = torch.zeros(W, dtype=torch.int64)
            for wi, w in enumerate(cs):
                ids = [b["id"] for b in w["balls"]]
                b = w["special"] if who == "special" else ids[who % len(ids)]
                ball[wi] = b
                if kind == "GotoDirection":
                    f[wi, :3] = torch.tensor([r.uniform(-1, 1) for _ in range(3)], dtype=torch.float64)
                elif kind == "SetSpeedFraction":
                    f[wi, 0] = r.choice([0.0, 1.0, r.random()])
                elif kind == "CloakBall":
                    n[wi] = r.choice([1, 2] if park != "collide" else [2, 3])
                    f[wi, 0] = r.uniform(100, 3000)
                elif kind == "Orbit":
                    target[wi] = r.choice(ids)
                    f[wi, 0] = r.uniform(300, 5000)
            per.append((kind, mask, ball, target, f, n, torch.zeros(W, dtype=torch.int64)))
        inputs.append(per)
    return cs, [k for k, _ in sched], inputs


@CUDA
@pytest.mark.parametrize("park,do,kw", [("modes", False, {}), ("modes", True, {}),
                                        ("orient", True, {"kernel": "cuda"}),
                                        ("collide", False, {"contact_slots": 12}),
                                        ("collide", False, {"kernel": "cuda", "contact_slots": 12})])
def test_a_graphs_ticks_equal_eager_ticks(park, do, kw):
    """The same fixed schedule ticked eagerly and replayed from a ServerGraph: every tick the same states, routes and
    delivery flags; the graph also survives an eager setter between ticks (its buffers synced)."""
    cs, sched, inputs = _fixed_schedule_case(1, 64, park, do, 10)
    eager, index = S.build(cs, park, "cuda", do=do, **kw)
    graphed, _ = S.build(cs, park, "cuda", do=do, **kw)
    g = VS.ServerGraph(graphed, sched)
    for t, per in enumerate(inputs):
        for kind, *data in per:
            eager.push(kind, *(x.cuda() for x in data))
        eager.pre_tick(snapshot=False)
        eager.evolve()
        eager.post_tick()
        for slot_in, (kind, *data) in zip(g.inputs, per):
            for key, x in zip(("mask", "ball", "target", "f", "n", "loc"), data):
                slot_in[key].copy_(x.cuda())
        out = g.tick()
        torch.cuda.synchronize()
        a = S.batched_rows(eager, cs, index, park, do)
        b = S.batched_rows(graphed, cs, index, park, do)
        assert all(S.same_rows(a[k], b[k]) for k in a), (t, [k for k in a if not S.same_rows(a[k], b[k])][:3])
        assert torch.equal(eager.batch.route, out.route), t
        assert torch.equal(eager.batch.has_character, out.has_character), t
        assert torch.equal(eager.batch.gets_bubble, out.gets_bubble), t
        if t == 4 and park != "collide":                    # an eager setter between ticks, on both
            m = torch.zeros(eager.R, dtype=torch.bool, device="cuda")
            m[::3] = True
            v = torch.full((eager.R,), 0.5, dtype=torch.float64, device="cuda")
            eager.a.park.set_speed_fraction(m, v)
            graphed.a.park.set_speed_fraction(m, v)


# ======================================================================================= the client's view

def _session_and_view(seed: int, down: int, up: int, ticks: int = 18, check=None) -> tuple:
    """A scalar Session and a batched server with a ClientView on the same worlds and commands (VecOrientPark, one
    world: the session's), driven alike: the first tick evolves only (the first OnTick), then pre-tick, evolve and
    post-tick; commands issued after tick n reach both servers at tick n + 1 + up."""
    r = random.Random(f"view-{seed}")
    cs = S.cases(seed, 1, "orient", ticks=1)
    w = cs[0]
    park = srv.DestinyPark(is_master=True)
    for b in w["balls"]:
        park.AddBall(b["id"], b["mass"], b["radius"], b["max_vel"], 1, 0, 0, int(b["interactive"]), 0, *b["pos"],
                     *b["vel"], b["agility"], b["sf"])
    session = cl.Session(park, w["pilots"], down=down, up=up)
    server, index = S.build(cs, "orient", "cpu")
    view = VS.ClientView(server, down, fields=("pos", "vel", "mode", "goto", "speed_fraction"))
    pending = {}
    rows_of = {bid: index[(w["id"], bid)] for bid in (b["id"] for b in w["balls"])}

    def batched_tick(n):
        for name, args in pending.pop(n, []):
            one = S.pushes([dict(w, ticks=[[(name, args)]])], 0, "cpu")[0]
            server.push(*one)
        if n == 1:
            server.evolve()                                 # the first OnTick: no pre-tick, no post-tick
        else:
            server.pre_tick()
            server.evolve()
            server.post_tick()
        view.record(int(server.a.clock()[0]))

    session.tick()
    batched_tick(1)
    for ship, p in sorted(w["pilots"].items(), key=lambda kv: kv[1]):
        session.batcher.send_full_state_update(session.interests.char_of[ship], ship)
    session.tick()
    batched_tick(2)
    ids = [b["id"] for b in w["balls"]]
    special = next(b for b in w["balls"] if b["id"] == w["special"])
    for _ in range(ticks):
        for _ in range(r.choice([0, 1, 2])):
            s = r.choice(ids)
            k = r.random()
            if k < 0.5:
                call = ("go_to_direction", (s, r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1)))
            elif k < 0.8:
                call = ("set_speed_fraction", (s, r.random()))
            elif k < 0.9 or not special["interactive"]:     # (a non-interactive ship's cloak: the first finding)
                call = ("stop", (s,))
            else:
                call = ("cloak_ball", (w["special"], 2, 1000.0)) if r.random() < 0.5 else \
                    ("uncloak_ball", (w["special"],))
            session.command(*call[:1], *call[1])
            pending.setdefault(session.tick_no + 1 + up, []).append(call)
        session.tick()
        batched_tick(session.tick_no)
        if check is not None:
            check(session, view, rows_of, w)
    return session, view, rows_of, w


@pytest.mark.parametrize("down,up", [(0, 0), (1, 0), (2, 1), (0, 2)])
def test_client_view_equals_the_sessions_client_parks(down, up):
    """After every tick (once the clients hold their SetState), ClientView's state (time, positions, velocities,
    modes, goto points, speed fractions, and who is in whose park) equals each scalar client park's, for pilots' and
    other ships alike."""
    def check(session, view, rows_of, w):
        t, fields, present = view.view()
        assert t == session.tick_no - down
        for ship, p in w["pilots"].items():
            if not session.clients[p][1]._state_is_valid:   # its SetState has not arrived yet (the first `down` ticks)
                continue
            cpark = session.client_park(p)
            assert cpark.currentTime == t
            mine = rows_of[ship]
            held = {bid for bid in rows_of if bool(present[rows_of[bid], mine])}
            assert held == set(cpark.lane.balls), (session.tick_no, p, held, set(cpark.lane.balls))
            for bid in held:
                b, r = cpark.lane.balls[bid], rows_of[bid]
                assert S.same_rows(tuple(fields["pos"][r].tolist()), tuple(b.pos)), (session.tick_no, p, bid)
                assert S.same_rows(tuple(fields["vel"][r].tolist()), tuple(b.vel))
                assert int(fields["mode"][r]) == b.mode
                assert S.same_rows((float(fields["speed_fraction"][r]),), (b.speed_fraction,))
                if b.mode in (0, 3):                         # GOTO, WARP: the goto point is the client's too
                    assert S.same_rows(tuple(fields["goto"][r].tolist()), tuple(b.goto))
            checked.append((session.tick_no, p))

    checked = []
    for seed in (1, 2, 3):
        _session_and_view(seed, down, up, check=check)
    assert len(checked) >= 3 * (18 - down)


# ================================== planted faults: each must be caught by the tests named beside it

def _runs(park, do, evolve, seed=1, n=24, ticks=10):
    def run():
        _check(S.run(S.cases(seed, n, park, do=do, ticks=ticks), park, "cpu", do=do, evolve=evolve))
    return run


def _dup_any_queued(self, kind, mask, ball, target=None, f=None, n=None, loc=None):
    """TIME-710 broken: no duplicate test (every call queued)."""
    s = _ORIGINAL_PUSH(self, kind, mask, ball, target, f, n, loc)
    s.ok = mask.to(torch.bool).to(self.dev) & ((s.row >= 0) | (s.kind != VS.CLOAK))
    return s


_ORIGINAL_PUSH = VS.VecServer.push
_ORIGINAL_PRE = VS.VecServer.pre_tick


def _cloak_with_the_others(self, snapshot=None):
    """TIME-712 broken: CloakBall applied with the other actions, before the bubble update."""
    cloaks = [s for s in self.slots if s.kind == VS.CLOAK]
    for s in cloaks:
        m, _, f, n = self._rows(s)
        self.a.cloak(m, n, f[:, 0])
    orig_cloak = self.a.cloak
    self.a.cloak = lambda *args: None
    try:
        _ORIGINAL_PRE(self, snapshot)
    finally:
        self.a.cloak = orig_cloak


def _stamp_is_the_clock(self, snapshot=None):
    """The stamp broken: the actions stamped with the clock at the pre-tick, not the stamp they were queued with."""
    _ORIGINAL_PRE(self, snapshot)
    self.batch.stamp = self.batch.clock.clone()


def _cloaked_events_to_the_bubble(self, snapshot=None):
    """SEN-712 broken: a cloaked ship's events go to the bubble like the others."""
    _ORIGINAL_PRE(self, snapshot)
    b = self.batch
    b.route = torch.where(b.route == VS.CHARACTER, torch.tensor(VS.BUBBLE, dtype=torch.int8), b.route)


def _bubble_lists_every_time(self, snapshot=None):
    """BubbleUpdater broken: the additions recomputed at every pre-tick, not once per park time."""
    self._bubble_time = torch.full_like(self._bubble_time, -7)
    _ORIGINAL_PRE(self, snapshot)


def _no_release_on_cloak(self, m, mode, rng):
    """MOV-730 broken: a cloak does not release the followers."""
    orig = self.release_followers
    self.release_followers = lambda m: None
    try:
        _ORIGINAL_CLOAK(self, m, mode, rng)
    finally:
        del self.release_followers


_ORIGINAL_CLOAK = VS._Adapter.cloak


def _uncloak_massive_while_warping(self, m):
    """SEN-711 broken: an uncloaked ship massive again even in warp proper."""
    ok = m & self.exists
    if self.master:
        self.clear_sensor(ok & (self.cloaked == VS.DSTNORMALCLOAK))
    self.cloaked = torch.where(ok, 0, self.cloaked)
    self.massive = torch.where(ok, True, self.massive)


def _old_bubble_ignored(self, kind, mask, ball, target=None, f=None, n=None, loc=None):
    """_actions.py:628-638 broken: CloakBall always sent to the ship's bubble, as if its old bubble were its bubble."""
    s = _ORIGINAL_PUSH(self, kind, mask, ball, target, f, n, loc)
    if s.kind == VS.CLOAK:
        s.loc = torch.zeros_like(s.loc)
    return s


def _local_moves_sent(self, kind, mask, ball, target=None, f=None, n=None, loc=None):
    """_ticker.py:165-171 broken: a local SetBallPosition is sent to the bubble."""
    s = _ORIGINAL_PUSH(self, kind, mask, ball, target, f, n, loc)
    if s.kind == VS.SETPOS:
        s.loc = torch.zeros_like(s.loc)
    return s


def _notices_at_the_back(server, batch=None, client_base=10000):
    """_ticker.py:139-163 broken (in decode): the uncloak's AddBallsToPark appended, not inserted in front."""
    b = batch or server.batch
    out = _ORIGINAL_DECODE(server, b, client_base)
    for per in out:
        for c, lst in per.items():
            for i, (kind, method, state, wait, count) in enumerate(lst):
                if kind == "single":
                    front = [e for e in state if e[1][0] == "AddBallsToPark"]
                    rest = [e for e in state if e[1][0] != "AddBallsToPark"]
                    lst[i] = (kind, method, rest + front, wait, count)
    return out


_ORIGINAL_DECODE = VS.decode


def _wait_never(self):
    """send_batch broken: wait_for_bubble never set (the client applies its user update before the bubble's)."""
    _ORIGINAL_POST(self)
    self.batch.gets_bubble = self.batch.gets_bubble & ~self.batch.has_character


_ORIGINAL_POST = VS.VecServer.post_tick


def _view_without_delay(self):
    """ClientView broken: the latest recorded state, not the one `down` ticks back."""
    down = self.down
    self.down = 0
    try:
        return _ORIGINAL_VIEW(self)
    finally:
        self.down = down


_ORIGINAL_VIEW = VS.ClientView.view


def _view_check():
    test_client_view_equals_the_sessions_client_parks(1, 0)
    test_client_view_equals_the_sessions_client_parks(2, 1)

FAULTS = {
    "no_duplicate_test": ("TIME-710", lambda mp: mp.setattr(VS.VecServer, "push", _dup_any_queued),
                          [_runs("orient", False, True)]),
    "cloak_with_the_others": ("TIME-712", lambda mp: mp.setattr(VS.VecServer, "pre_tick", _cloak_with_the_others),
                              [_runs("modes", False, False)]),
    "stamp_is_the_clock": ("TIME-709", lambda mp: mp.setattr(VS.VecServer, "pre_tick", _stamp_is_the_clock),
                           [_runs("orient", False, False)]),
    "cloaked_events_to_the_bubble": ("SEN-712", lambda mp: mp.setattr(VS.VecServer, "pre_tick",
                                                                      _cloaked_events_to_the_bubble),
                                     [_runs("collide", False, True)]),
    "bubble_lists_every_time": ("BubbleUpdater", lambda mp: mp.setattr(VS.VecServer, "pre_tick",
                                                                       _bubble_lists_every_time),
                                [_runs("modes", False, False)]),
    "no_release_on_cloak": ("MOV-730", lambda mp: mp.setattr(VS._Adapter, "cloak", _no_release_on_cloak),
                            [_runs("modes", False, False)]),
    "uncloak_massive_while_warping": ("SEN-711", lambda mp: mp.setattr(VS._Adapter, "uncloak",
                                                                       _uncloak_massive_while_warping),
                                      [_runs("modes", False, False, n=60, ticks=14)]),
    "old_bubble_ignored": ("_actions.py:628-638", lambda mp: mp.setattr(VS.VecServer, "push", _old_bubble_ignored),
                           [_runs("orient", False, True)]),
    "local_moves_sent": ("_ticker.py:165-171", lambda mp: mp.setattr(VS.VecServer, "push", _local_moves_sent),
                         [_runs("orient", False, True)]),
    "notices_at_the_back": ("_ticker.py:139-163", lambda mp: mp.setattr(VS, "decode", _notices_at_the_back),
                            [_runs("collide", False, True)]),
    "wait_for_bubble_never": ("_parkupdatebatcher.py:52-119", lambda mp: mp.setattr(VS.VecServer, "post_tick",
                                                                                    _wait_never),
                              [_runs("orient", False, True)]),
    "view_without_delay": ("the client's delay", lambda mp: mp.setattr(VS.ClientView, "view", _view_without_delay),
                           [_view_check]),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    rule, patch, cases = FAULTS[fault]
    for case in cases:
        case()                                              # clean: passes
    patch(monkeypatch)
    caught = 0
    for case in cases:
        try:
            case()
        except AssertionError:
            caught += 1
    assert caught > 0, f"the planted fault {fault} ({rule}) went unnoticed"

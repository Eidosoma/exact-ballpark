# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Global balls in what each client holds: exact_ballpark.client.Session (the scalar server layer on a
master DestinyPark over FullBallpark, one client park per pilot, Destiny's client Ticker) against the batched
SpaceServer over SpaceCollideAdapter on a VecCollidePark made with global balls, with a SpaceClientView, on the same
world and commands, by tests/space_session.py's protocol (the first tick evolves only; then each pilot's SetState,
`set_state` on the view; then pre-tick, evolve and post-tick; a command issued after tick n reaches both servers at
tick n + 1 + up): after every tick, for every synced client, the balls its park holds and the state of those it holds
fresh, both ways (space_session.holdings_check).

**The worlds** are tests/server_space_cases.py's (2 to 6 non-massive ships around one to three anchors or a row of three
top boxes; several bubbles: a world of one anchor and no row is drawn again), each ball global at its AddBall with
probability 0.3. **The commands** (0 to 3 a tick) are space_session's (flights, jumps across bubbles, the speed
fraction, Stop, interactive changes, the special ship's cloaks and uncloaks) and make_ball_global and make_ball_local
of any ship (about a third of the calls; a tenth of those twice in a row: the second dropped by the duplicate test) and
make_ball_global of a ball no world holds (its empty stream), drawn from this module's own random stream."""

from __future__ import annotations

import random
from collections import Counter

import global_cases as G
import server_cases as S
import server_space_cases as SS
import space_session as SSN
from exact_ballpark import client as cl
from exact_ballpark import server as srv
from exact_ballpark import vec_server_space as VSS
from exact_ballpark.space import LEVEL_WIDTH

TOP = LEVEL_WIDTH[0]
UNKNOWN = 4242                                      # an id no world holds (server_space_cases draws 101 to 998)
FIELDS = ("pos", "vel", "mode", "goto", "speed_fraction")


def world(r: random.Random) -> dict:
    """One world of several bubbles (server_space_cases' generator, drawn again while it has one anchor and no
    row), each ball global at AddBall with probability 0.3."""
    while True:
        w = SS._world(r, 0, "collide")
        if w["row"] is not None or len(w["anchors"]) > 1:
            break
    for b in w["balls"]:
        b["global"] = r.random() < 0.3
    return w


def command(r: random.Random, w: dict) -> list:
    """One tick's call (or the same global call twice in a row)."""
    ids = [b["id"] for b in w["balls"]]
    s = r.choice(ids)
    sp = w["special"]
    a = r.choice(w["anchors"])
    k = r.random()
    if k < 0.12:
        return [("go_to_direction", (s, r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1)))]
    if k < 0.20:
        return [("go_to_point", (s, a[0] + r.choice([-1, 1]) * 6e3, a[1] + r.uniform(-4e3, 4e3), a[2]))]
    if k < 0.36:
        row = w.get("row")
        if row and r.random() < 0.5:                            # across the face between two bubbles
            dst = (row[0] + r.choice([-1.0, 1.0]) * r.uniform(200.0, 4000.0), row[1] + r.uniform(-2e3, 2e3), row[2])
        else:
            dst = r.choice(w["anchors"] + [(r.randint(-4, 4) * TOP + 0.5 * TOP, 0.5 * TOP, 0.5 * TOP)])
            dst = (dst[0] + r.uniform(-5e3, 5e3), dst[1] + r.uniform(-5e3, 5e3), dst[2])
        return [("set_ball_position", (s, *dst, False))]
    if k < 0.42:
        return [("set_speed_fraction", (s, r.random()))]
    if k < 0.46:
        return [("stop", (s,))]
    if k < 0.52:
        return [("set_ball_interactive", (s, r.choice([0, 1])))]
    if k < 0.58:
        return [("cloak_ball", (sp, r.choice([1, 2, 0]), 1000.0))]
    if k < 0.62:
        return [("uncloak_ball", (sp,))]
    if k < 0.96:
        call = (r.choice(["make_ball_global", "make_ball_global", "make_ball_local"]), (s,))
    else:
        call = ("make_ball_global", (UNKNOWN,))                 # a ball not in the park: its empty stream
    return [call, call] if r.random() < 0.1 else [call]


def session_and_view(seed: int, down: int, up: int, ticks: int = 18, check=None, view_cls=None, device: str = "cpu",
                     kernel: str = "torch") -> tuple:
    """Both sides driven alike (tests/space_session.py's protocol with global balls): (session, view, rows_of, w).
    `device`, `kernel`: the batched server's (the collision park's torch path, or the kernel on a CUDA device)."""
    r = random.Random(f"global-view-{seed}")
    w = world(r)
    old = srv.is_dynamical_orientation_enabled()
    srv.disable_dynamical_orientation()
    try:
        park = srv.DestinyPark(is_master=True)
        for b in w["balls"]:
            park.AddBall(b["id"], b["mass"], b["radius"], b["max_vel"], 1, int(b["global"]), 0, int(b["interactive"]),
                         0, *b["pos"], *b["vel"], b["agility"], b["sf"])
        session = cl.Session(park, w["pilots"], down=down, up=up)
        server, index = G.build_server([w], device, kernel)
        view = (view_cls or VSS.SpaceClientView)(server, down, fields=FIELDS)
        pending: dict = {}
        rows_of = {b["id"]: index[(w["id"], b["id"])] for b in w["balls"]}

        def batched_tick(n):
            for name, args in pending.pop(n, []):
                one = S.pushes([dict(w, ticks=[[(name, args)]])], 0, "cpu")[0]
                server.push(*one)
            if n == 1:
                server.evolve()
            else:
                server.pre_tick()
                server.evolve()
                server.post_tick()
            view.record(int(server.a.clock()[0]))

        session.tick()
        batched_tick(1)
        for ship, p in sorted(w["pilots"].items(), key=lambda kv: kv[1]):
            session.batcher.send_full_state_update(session.interests.char_of[ship], ship)
        if hasattr(view, "set_state"):
            view.set_state()
        session.tick()
        batched_tick(2)
        for _ in range(ticks):
            for _ in range(r.choice([0, 1, 2, 3])):
                for call in command(r, w):
                    session.command(call[0], *call[1])
                    pending.setdefault(session.tick_no + 1 + up, []).append(call)
            session.tick()
            batched_tick(session.tick_no)
            if check is not None:
                check(session, view, rows_of, w)
        return session, view, rows_of, w
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()


def global_check(down: int, checked: list, stale: list, cov: Counter):
    """space_session.holdings_check, and what it reached: the balls synced clients hold in another bubble than their
    ship's (global ones; fresh and stale), the global notices the clients were sent (AddBallsToPark, RemoveBall and
    the empty streams in narrowcasts), a client's own ship removed."""
    inner = SSN.holdings_check(down, checked, stale)

    def check(session, view, rows_of, w):
        inner(session, view, rows_of, w)
        t, fields, present, synced, fresh = view.view()
        lane = session.server.lane
        for ship, p in w["pilots"].items():
            if not session.clients[p][1]._state_is_valid or not bool(synced[rows_of[ship]]):
                continue
            cpark = session.client_park(p)
            mine = rows_of[ship]
            own = lane.balls.get(ship)
            for bid in cpark.lane.balls:
                b = lane.balls.get(bid)
                if b is None or own is None or b.bubble == own.bubble:
                    continue
                cov["held: a ball of another bubble"] += 1
                cov["held: a global ball of another bubble"] += int(b.is_global)
                key = "fresh" if bool(fresh[rows_of[bid], mine]) else "stale"
                cov[f"held: a ball of another bubble, {key}"] += 1
            if ship not in cpark.lane.balls:
                cov["a client without its own ship"] += 1
        n, sent = session.sent_log[-1]
        for kind, to, method, state, wait, count in sent:
            if kind != "narrow":
                continue
            for stamp, (name, args) in state:
                if name == "AddBallsToPark":
                    cov["narrow: AddBallsToPark"] += 1
                    cov["narrow: AddBallsToPark, empty stream"] += int(len(args[0]) == 5)
                elif name == "RemoveBall":
                    cov["narrow: RemoveBall"] += 1
    return check


def run(seeds, pairs=((0, 0), (1, 0), (2, 1), (0, 2)), ticks: int = 18, device: str = "cpu",
        kernel: str = "torch") -> dict:
    """Every seed with every (down, up) pair: what the checks covered (an AssertionError propagates)."""
    checked, stale, cov = [], [], Counter()
    for seed in seeds:
        for down, up in pairs:
            session_and_view(seed, down, up, ticks, check=global_check(down, checked, stale, cov), device=device,
                             kernel=kernel)
    return {"checked": len(checked), "held": sum(c[2] for c in checked), "stale": len(stale), "coverage": dict(cov)}


def two_ships(calls: list, far: bool, glob: tuple = (False, False), down: int = 0, ticks: int = 3,
              view_cls=None, then: tuple = ()) -> tuple:
    """A constructed world: ships 101 and 102 (interactive, non-massive, pilots 0 and 1; 102 1 km from 101, or at a far
    anchor in another bubble when `far`), global at AddBall as `glob` says; the Session and the batched server with
    its view driven alike (the start with each SetState and `set_state`), `calls` issued after the start and each list
    of `then` after one more tick, `ticks` ticks in all, space_session.holdings_check after each: (session, server,
    view, rows_of, checked, stale)."""
    pos = {101: [1000.0, 2000.0, 3000.0], 102: ([3 * TOP, -2 * TOP, TOP] if far else [2000.0, 2000.0, 3000.0])}
    w = {"id": 0, "park": "collide", "do": False, "anchors": [(0.0, 0.0, 0.0)], "special": 101, "row": None,
         "pilots": {101: 0, 102: 1}, "ticks": [[]],
         "balls": [{"id": bid, "mass": 1e7, "radius": 50.0, "max_vel": 200.0, "agility": 0.5, "sf": 1.0,
                    "pos": list(pos[bid]), "vel": [0.0, 0.0, 0.0], "interactive": True, "massive": False,
                    "global": bool(g), "max_ang_vel": 1.0, "rot_agility": 0.9} for bid, g in zip((101, 102), glob)]}
    old = srv.is_dynamical_orientation_enabled()
    srv.disable_dynamical_orientation()
    try:
        park = srv.DestinyPark(is_master=True)
        for b in w["balls"]:
            park.AddBall(b["id"], b["mass"], b["radius"], b["max_vel"], 1, int(b["global"]), 0, 1, 0, *b["pos"],
                         *b["vel"], b["agility"], b["sf"])
        session = cl.Session(park, w["pilots"], down=down)
        server, index = G.build_server([w], "cpu")
        view = (view_cls or VSS.SpaceClientView)(server, down, fields=FIELDS)
        rows_of = {bid: index[(0, bid)] for bid in (101, 102)}
        checked, stale = [], []
        check = SSN.holdings_check(down, checked, stale)
        session.tick()
        server.evolve()
        view.record(int(server.a.clock()[0]))
        for ship, p in sorted(w["pilots"].items(), key=lambda kv: kv[1]):
            session.batcher.send_full_state_update(session.interests.char_of[ship], ship)
        view.set_state()
        session.tick()
        server.pre_tick()
        server.evolve()
        server.post_tick()
        view.record(int(server.a.clock()[0]))
        plan = [list(calls)] + [list(c) for c in then]
        for k in range(ticks):
            for name, args in (plan[k] if k < len(plan) else []):
                session.command(name, *args)
                server.push(*S.pushes([dict(w, ticks=[[(name, args)]])], 0, "cpu")[0])
            session.tick()
            server.pre_tick()
            server.evolve()
            server.post_tick()
            view.record(int(server.a.clock()[0]))
            check(session, view, rows_of, w)
        return session, server, view, rows_of, checked, stale
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""A scalar Session (exact_ballpark.client: the server layer on a master DestinyPark and one client park per pilot) and
the batched SpaceServer with a SpaceClientView on the same world and commands, over several bubbles:
what each client's park holds and its state, both ways, after every tick."""

from __future__ import annotations

import random

import server_cases as S
import server_space_cases as SS
from exact_ballpark import client as cl
from exact_ballpark import server as srv
from exact_ballpark import vec_server_space as VSS
from exact_ballpark.space import LEVEL_WIDTH

TOP = LEVEL_WIDTH[0]


def session_and_view(seed: int, down: int, up: int, ticks: int = 18, check=None, view_cls=None) -> tuple:
    """Both sides driven alike: the first tick evolves only (the first OnTick), then pre-tick, evolve and post-tick;
    commands issued after tick n reach both servers at tick n + 1 + up (the harness, several bubbles here)."""
    r = random.Random(f"space-view-{seed}")
    cs = SS.cases(seed, 1, "orient", ticks=1)
    w = cs[0]
    park = srv.DestinyPark(is_master=True)
    for b in w["balls"]:
        park.AddBall(b["id"], b["mass"], b["radius"], b["max_vel"], 1, 0, 0, int(b["interactive"]), 0, *b["pos"],
                     *b["vel"], b["agility"], b["sf"])
    session = cl.Session(park, w["pilots"], down=down, up=up)
    server, index = SS.build(cs, "orient", "cpu")
    view = (view_cls or VSS.SpaceClientView)(server, down, fields=("pos", "vel", "mode", "goto", "speed_fraction"))
    pending = {}
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
    session.tick()
    batched_tick(2)
    ids = [b["id"] for b in w["balls"]]
    sp = w["special"]
    for _ in range(ticks):
        for _ in range(r.choice([0, 1, 2])):
            s = r.choice(ids)
            a = r.choice(w["anchors"])
            k = r.random()
            if k < 0.25:
                call = ("go_to_direction", (s, r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1)))
            elif k < 0.40:
                call = ("go_to_point", (s, a[0] + r.choice([-1, 1]) * 6e3, a[1] + r.uniform(-4e3, 4e3), a[2]))
            elif k < 0.55:
                dst = r.choice(w["anchors"] + [(r.randint(-4, 4) * TOP + 0.5 * TOP, 0.5 * TOP, 0.5 * TOP)])
                call = ("set_ball_position", (s, dst[0] + r.uniform(-5e3, 5e3), dst[1] + r.uniform(-5e3, 5e3),
                                              dst[2], False))
            elif k < 0.65:
                call = ("set_speed_fraction", (s, r.random()))
            elif k < 0.72:
                call = ("stop", (s,))
            elif k < 0.80:
                call = ("set_ball_interactive", (s, r.choice([0, 1])))
            elif k < 0.92:
                call = ("cloak_ball", (sp, r.choice([1, 2, 0]), 1000.0))
            else:
                call = ("uncloak_ball", (sp,))
            session.command(*call[:1], *call[1])
            pending.setdefault(session.tick_no + 1 + up, []).append(call)
        session.tick()
        batched_tick(session.tick_no)
        if check is not None:
            check(session, view, rows_of, w)
    return session, view, rows_of, w


def holdings_check(down: int, checked: list, stale: list | None = None):
    """A check for session_and_view: after every tick, each client's park (its time, the balls it holds, their
    positions, velocities, modes, goto points, speed fractions) equals the view's (the fields of the balls it holds
    fresh; the others counted in `stale`)."""
    stale = [] if stale is None else stale

    def check(session, view, rows_of, w):
        t, fields, present, synced, fresh = view.view()
        assert t == session.tick_no - down
        for ship, p in w["pilots"].items():
            if not session.clients[p][1]._state_is_valid or not bool(synced[rows_of[ship]]):
                continue
            cpark = session.client_park(p)
            assert cpark.currentTime == t
            mine = rows_of[ship]
            held = {bid for bid in rows_of if bool(present[rows_of[bid], mine])}
            assert held == set(cpark.lane.balls), (session.tick_no, p, sorted(held), sorted(cpark.lane.balls))
            for bid in held:
                b, r = cpark.lane.balls[bid], rows_of[bid]
                if not bool(fresh[r, mine]):
                    stale.append((session.tick_no, p, bid))
                    continue
                assert S.same_rows(tuple(fields["pos"][r].tolist()), tuple(b.pos)), (session.tick_no, p, bid)
                assert S.same_rows(tuple(fields["vel"][r].tolist()), tuple(b.vel))
                assert int(fields["mode"][r]) == b.mode
                assert S.same_rows((float(fields["speed_fraction"][r]),), (b.speed_fraction,))
                if b.mode in (0, 3):
                    assert S.same_rows(tuple(fields["goto"][r].tolist()), tuple(b.goto))
            checked.append((session.tick_no, p, len(held)))
    return check

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""OnTick's intervals on Destiny's master server: each world driven by its own simulation
clock, generated from a seed, and the two runners compared bit for bit: the scalar server layer (tests/global_cases.py's
ServerWorld: Actions, Ticker, BubbleUpdater and ParkUpdateBatcher on a master DestinyPark over FullBallpark), its
park's OnTick (exact_ballpark.full.FullBallpark.on_tick, Ballpark.cpp:219-330, TIME-702 to TIME-705) calling the Ticker's
tick at DoPreTick and send_batch at DoPostTick, as Destiny's server hangs them (_baseticker.py:23-26,
_parkupdatebatcher.py:16-18), against exact_ballpark.vec_server_space.SpaceServer.on_tick over SpaceCollideAdapter (one
call for every world, each world's clock its own), compared after every call: each world's sends in order (one list a
DoPostTick: the canonical form) and every ship's row, extra fields, bubble and global flag.

**The clocks.** Each call advances each world's simulation time (100 ns units; an interval is the tick interval x
10,000) by its own step: nothing, under one interval (too early: no tick, the time kept), one interval and a little
more (the remainder carried to the next call), two or three intervals (as many ticks in one call), or more than five
(the master's resynchronisation: one tick, the remainder dropped); a world's first call that holds an interval evolves
once without the tick events (no pre-tick, no send). **The calls** between two OnTicks are tests/global_cases.py's
(an earlier generator's, with make_ball_global and make_ball_local), drawn per call; the worlds are an earlier generator's, as global_cases makes
them for the server."""

from __future__ import annotations

import random
from collections import Counter

import torch

import global_cases as G
import server_cases as S
import space_collide_cases as C
from exact_ballpark import server as srv
from exact_ballpark import vec_server_space as VSS

INTERVAL = 1000 * 10000                                  # the tick interval (ms) x 10,000: 100 ns units (Be::Time)
STEPS = (0.0, 0.4, 0.9999, 1.0, 1.0, 1.0, 1.0, 1.25, 1.6, 2.0, 2.3, 3.0, 4.2, 5.5, 6.0, 7.4)


def _step(r: random.Random) -> int:
    """One call's advance of a world's clock (100 ns units)."""
    k = r.choice(STEPS)
    return int(k * INTERVAL) + (r.randint(0, 999) if k else 0)


def cases(seed: int, n: int, calls: int = 14) -> list:
    """n worlds, each with `calls` OnTick calls: the Actions calls before each and the simulation time it passes."""
    r = random.Random(f"ontick-{seed}")
    out = []
    for wid in range(n):
        w = C._world(r, wid)
        for b in w["balls"]:
            b["free"] = True                                # the server's AddBall: every ship free
        G._globalise(r, w)
        t = 0
        w["ticks"], w["clock"] = [], []
        for _ in range(calls):
            t += _step(r)
            w["clock"].append(t)
            w["ticks"].append(G._tick_calls(r, w))
        out.append(w)
    return out


class OnTickWorld(G.ServerWorld):
    """The scalar server world driven through its park's OnTick: the Ticker at DoPreTick, send_batch at DoPostTick;
    each DoPostTick's sends kept apart."""

    def __init__(self, w: dict):
        super().__init__(w)
        lane = self.park.lane
        self.sends: list = []
        lane.pre_tick_handlers.append(lambda time: self.ticker.tick())
        lane.post_tick_handlers.append(self._post)

    def _post(self, time) -> None:
        self.batcher.send_batch()
        self.sends.append(S.per_client(self.network.take()))

    def call(self, calls: list, sim_time: int) -> list:
        for name, args in calls:
            getattr(self.actions, name)(*args)
        self.sends = []
        self.park.lane.on_tick(sim_time)
        return self.sends


def run(cs: list, device: str, kernel: str = "torch", calls: int | None = None) -> dict:
    """Both sides on the worlds cs, call by call: the worlds whose sends differ (their DoPostTick lists in order), the
    ship rows that differ, the volumes and what the clocks reached (tick counts per call, too-early calls, first
    calls, resynchronisations, the remainders carried)."""
    T = len(cs[0]["clock"]) if calls is None else int(calls)
    old = srv.is_dynamical_orientation_enabled()
    try:
        scal = [OnTickWorld(w) for w in cs]
        server, index = G.build_server(cs, device, kernel)
        bad_sends, bad_rows, compared, sends, cov = [], [], 0, 0, Counter()
        for t in range(T):
            want = [sw.call(w["ticks"][t], w["clock"][t]) for sw, w in zip(scal, cs)]
            for kind, *data in S.pushes(cs, t, server.dev):
                server.push(kind, *data)
            batches = server.on_tick(torch.tensor([w["clock"][t] for w in cs], dtype=torch.int64))
            decoded = [(b.ticked.cpu().tolist(), VSS.decode(server, b)) for b in batches]
            rows_b = C.collide_rows(server, index)
            p = server.a.park
            R = server.a.R
            ext = {k: getattr(p, k).reshape(R).cpu().tolist() for k in C.EXTRA}
            bub = server.bubbles_of(torch.arange(R, device=server.dev)).cpu().tolist()
            glob = p.glob.reshape(R).cpu().tolist()
            flags = server.space.st["flags"].cpu().tolist()
            for wi, w in enumerate(cs):
                if flags[wi]:
                    continue
                got = [d[wi] for ticked, d in decoded if ticked[wi]]
                cov[f"ticks in a call: {len(want[wi])}"] += 1
                if len(want[wi]) != len(got) or any(S.canon(S._listify(a)) != S.canon(S._listify(b))
                                                    for a, b in zip(want[wi], got)):
                    bad_sends.append((t, w["id"], len(want[wi]), len(got)))
                sends += sum(sum(len(v) for v in x.values()) for x in want[wi])
                lane = scal[wi].park.lane
                srows = S.scalar_rows(scal[wi], "collide", False)
                for b in lane.balls.values():
                    r = index[(w["id"], b.id)]
                    compared += 1
                    mine = (rows_b[(w["id"], b.id)], tuple(ext[k][r] for k in C.EXTRA), bub[r], bool(glob[r]))
                    theirs = (srows[b.id], tuple(getattr(b, k) for k in C.EXTRA), b.bubble, bool(b.is_global))
                    if not (S.same_rows(mine[0], theirs[0]) and S.same_rows(mine[1], theirs[1])
                            and mine[2:] == theirs[2:]):
                        bad_rows.append((t, w["id"], b.id))
                if int(server.time[wi]) != lane.time or bool(server.first_time[wi]) != bool(lane.first_time):
                    bad_rows.append((t, w["id"], "clock", lane.time, int(server.time[wi])))
            for k, v in server.tick_counts.items():
                cov[k] += v
        flagged = [wi for wi, f in enumerate(server.space.st["flags"].cpu().tolist()) if f]
        return {"sends": bad_sends, "rows": bad_rows, "updates": sends, "compared": compared, "flagged": flagged,
                "calls": T, "coverage": dict(cov), "server": server}
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()

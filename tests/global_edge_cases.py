# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Global balls' edge cases on Destiny's master server, generated from a seed, for tests/global_cases.py's
and tests/ontick_cases.py's runners (which this module draws from and does not change):

- **make_ball_global of a ball not in the park** (`server_cases`): tests/global_cases.py's worlds and calls with
  make_ball_global and make_ball_local of an id no world holds (6 % of the calls each): Actions' SetBallGlobal finds no
  ball, make_ball_global writes the empty stream (WriteBallsToStream skips the id: the header and the park's time) and
  queues AddBallsToPark under the duplicate test, which every bubble with an interactive ball hears; make_ball_local
  queues nothing. Compared by global_cases.run_server (every client's update list byte for byte, every ship's row).
- **The OnTick duplicate-global case** (`ontick_cases`): tests/ontick_cases.py's clocks and calls, and in most worlds
  (probability 0.8) a moving interactive ship made global as the last call before the world's first OnTick that holds
  an interval (its single Evolve without a pre-tick) and again as the first call after it: one queue then holds two
  AddBallsToPark of one ship whose streams differ (it moved in that Evolve), and Destiny's duplicate test, which
  compares the stream's bytes, keeps both. Compared by ontick_cases.run.
- **Global balls over the modes and orient parks** (`space_cases`, `run_space`): tests/server_space_cases.py's worlds
  (several bubbles, non-massive ships) with each ball global at AddBall with probability 0.25 and its calls with
  make_ball_global and make_ball_local of any ship (0 to 2 a tick, at random places among them) and of the unknown id,
  over SpaceServer on VecOrientPark (the CPU) or VecModesPark (a CUDA device; follows, orbits and warps) with the flags
  held by the partition layer's view (`space_for(adapter, global_balls)`), against the scalar server layer with
  AddBall's isGlobal: every client's update list byte for byte, every ship's position, velocity, bubble and flag.
"""

from __future__ import annotations

import random

import torch

import global_cases as G
import ontick_cases as O
import server_cases as S
import server_space_cases as SS
import space_collide_cases as C
from exact_ballpark import server as srv
from exact_ballpark import vec_server_space as VSS

UNKNOWN = 4242                                      # an id no world holds (space_collide_cases' ids stay below it)


def _call(r: random.Random, w: dict) -> tuple:
    k = r.random()
    if k < 0.12:
        return (r.choice(["make_ball_global", "make_ball_local"]), (UNKNOWN,))
    return G._call(r, w)


def _tick_calls(r: random.Random, w: dict) -> list:
    calls = []
    for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4])):
        c = _call(r, w)
        calls.append(c)
        x = r.random()
        nan = any(isinstance(v, float) and v != v for v in c[1])
        if x < 0.10 and not nan:
            calls.append((c[0], tuple(c[1])))           # the same call twice in a row: the second dropped
        elif x < 0.16 and c[1][0] != UNKNOWN:
            calls.append(("stop", (c[1][0],)))
    return calls


def server_cases(seed: int, n: int, ticks: int = 14) -> list:
    """n worlds with `ticks` ticks of Actions calls each (global_cases.server_cases' with the unknown id's calls)."""
    r = random.Random(f"server-global-edge-{seed}")
    out = []
    for wid in range(n):
        w = C._world(r, wid)
        for b in w["balls"]:
            b["free"] = True                            # the server's AddBall: every ship free
        G._globalise(r, w)
        w["ticks"] = [_tick_calls(r, w) for _ in range(ticks)]
        out.append(w)
    return out


def ontick_cases(seed: int, n: int, calls: int = 14) -> list:
    """n worlds, each with `calls` OnTick calls (ontick_cases.cases'), most with a moving interactive ship made global
    just before and just after the world's first evolving OnTick (w["dup_global"]: the ship, else None)."""
    r = random.Random(f"ontick-global-edge-{seed}")
    out = []
    for wid in range(n):
        w = C._world(r, wid)
        for b in w["balls"]:
            b["free"] = True
        G._globalise(r, w)
        t = 0
        w["ticks"], w["clock"] = [], []
        for _ in range(calls):
            t += O._step(r)
            w["clock"].append(t)
            w["ticks"].append(G._tick_calls(r, w))
        w["dup_global"] = None
        first = next((k for k, c in enumerate(w["clock"]) if c >= O.INTERVAL), None)
        ships = [b for b in w["balls"] if b["interactive"]]
        if first is not None and first + 1 < calls and ships and r.random() < 0.8:
            b = r.choice(ships)
            if not any(b["vel"]):
                b["vel"] = [r.uniform(20.0, 80.0), r.uniform(-30.0, 30.0), 0.0]
            w["ticks"][first].append(("make_ball_global", (b["id"],)))
            w["ticks"][first + 1].insert(0, ("make_ball_global", (b["id"],)))
            w["dup_global"] = b["id"]
        out.append(w)
    return out


# ======================================================= global balls over the modes and orient parks

def _space_tick_calls(r: random.Random, w: dict, park: str) -> list:
    calls = SS._tick_calls(r, w, park)
    ids = [b["id"] for b in w["balls"]]
    for _ in range(r.choice([0, 0, 1, 1, 2])):
        bid = UNKNOWN if r.random() < 0.15 else r.choice(ids)
        c = (r.choice(["make_ball_global", "make_ball_global", "make_ball_local"]), (bid,))
        k = r.randint(0, len(calls))
        calls[k:k] = [c, c] if r.random() < 0.1 else [c]
    return calls


def space_cases(seed: int, n: int, park: str = "orient", ticks: int = 14) -> list:
    """n worlds of server_space_cases (park "orient" or "modes") with global flags and calls."""
    r = random.Random(f"server-space-global-{park}-{seed}")
    out = []
    for wid in range(n):
        w = SS._world(r, wid, park)
        for b in w["balls"]:
            b["global"] = r.random() < 0.25
        w["ticks"] = [_space_tick_calls(r, w, park) for _ in range(ticks)]
        w.pop("warped", None)
        out.append(w)
    return out


def build_space(cs: list, park: str, device: str, server_cls=None, **kw) -> tuple:
    """server_space_cases.build's SpaceServer with each row's AddBall flag held by the layer's view."""
    srv_, index = SS.build(cs, park, device)
    flags = [False] * srv_.a.R
    for w in cs:
        for b in w["balls"]:
            flags[index[(w["id"], b["id"])]] = bool(b["global"])
    space = VSS.space_for(srv_.a, global_balls=flags)
    return (server_cls or VSS.SpaceServer)(srv_.a, srv_.pilot.tolist(), space=space, **kw), index


def run_space(cs: list, park: str, device: str, ticks: int | None = None, server_cls=None, **kw) -> dict:
    """server_space_cases.run with global balls: the scalar layer with AddBall's isGlobal (global_cases.ServerWorld)
    against build_space's server, tick by tick; the ships' rows with their global flags."""
    T = len(cs[0]["ticks"]) if ticks is None else int(ticks)
    old = srv.is_dynamical_orientation_enabled()
    try:
        scal = [G.ServerWorld(w) for w in cs]
        server, index = build_space(cs, park, device, server_cls, **kw)
        bad_sends, bad_rows, sends, compared, bubbles, cov = [], [], 0, 0, set(), {}
        for t in range(T):
            want = [sw.tick(w["ticks"][t]) for sw, w in zip(scal, cs)]
            for kind, *data in S.pushes(cs, t, server.dev):
                server.push(kind, *data)
            server.pre_tick(snapshot=True)
            server.evolve()
            server.post_tick()
            got = VSS.decode(server)
            flags = server.space.st["flags"].cpu().tolist()
            bub_rows = server.bubbles_of(torch.arange(server.a.R, device=server.dev)).cpu().tolist()
            pos, vel = server.a.park.pos.cpu(), server.a.park.vel.cpu()
            glob = server.space.view.global_flags().cpu().tolist()
            for wi, w in enumerate(cs):
                if flags[wi]:
                    continue
                if S.canon(S._listify(want[wi])) != S.canon(S._listify(got[wi])):
                    bad_sends.append((t, w["id"], S._first_diff(want[wi], got[wi])))
                sends += sum(len(v) for v in want[wi].values())
                SS.coverage(want[wi], cov)
                gc = __import__("collections").Counter()
                G._global_coverage(want[wi], gc)
                for k, v in gc.items():
                    cov[k] = cov.get(k, 0) + v
                for b in scal[wi].park.lane.balls.values():
                    r = index[(w["id"], b.id)]
                    compared += 1
                    bubbles.add((w["id"], b.bubble))
                    cov["global_ship_ticks"] = cov.get("global_ship_ticks", 0) + int(b.is_global)
                    mine = (bub_rows[r], tuple(SS._bits(x) for x in pos[r].tolist()),
                            tuple(SS._bits(x) for x in vel[r].tolist()), bool(glob[r]))
                    theirs = (b.bubble, tuple(SS._bits(x) for x in b.pos), tuple(SS._bits(x) for x in b.vel),
                              bool(b.is_global))
                    if mine != theirs:
                        bad_rows.append((t, w["id"], b.id, theirs[0], mine[0], theirs[3], mine[3]))
        flagged = [wi for wi, f in enumerate(server.space.st["flags"].cpu().tolist()) if f]
        per_world: dict = {}
        for wid, b in bubbles:
            per_world.setdefault(wid, set()).add(b)
        cov["follower_clean_up_stops"] = server.cleanup_stops
        return {"sends": bad_sends, "rows": bad_rows, "updates": sends, "compared": compared, "flagged": flagged,
                "ticks": T, "worlds_with_several_bubbles": sum(1 for v in per_world.values() if len(v) > 1),
                "coverage": cov, "server": server}
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()

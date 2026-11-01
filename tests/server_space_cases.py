# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Command streams over worlds of several bubbles for Destiny's server layer on the partition, generated from a seed, and the two runners compared bit for bit: the scalar server layer
(tests/server_cases.py's ScalarWorld: Actions, Ticker, BubbleUpdater, ParkUpdateBatcher on a master DestinyPark over
FullBallpark) and the batched layer (exact_ballpark.vec_server_space.SpaceServer over VecOrientPark on the CPU or
VecModesPark on a CUDA device, with exact_ballpark.vec_space's partition).

A world is 2 to 6 ships around one to three anchors (the origin, a top box's face, a far point, a row of three top
boxes whose middle and right tops are adjacent with different bubbles), each pilot flying one interactive ship. Each
tick 0 to 4 Actions calls: flights across top faces, jumps (SetBallPosition: across a face, to another anchor, to a far
point: a new bubble), the re-filing setters (refused values now and then), interactive changes (a bubble with no
interactive ball dies: its balls freeze, its pilots hear nothing), the special ship's cloaks and uncloaks (it is the
only ship that can turn massive), duplicates and a Stop after a command; on a CUDA device (VecModesPark) follows and
orbits (refused across bubbles; a follower left in another bubble stopped at the pre-tick) and the special ship's
warps to far top boxes (a new bubble each tick in warp proper). Compared after every tick: every client's update list
(its singlecast, then its bubble's narrowcast: stamps, events, argument types, AddBallsToPark streams byte for byte,
RemoveBalls, PackagedActions, wait_for_bubble, counts) and every ship's state and bubble.
"""

from __future__ import annotations

import math
import random
import struct

import torch

import server_cases as S
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_space as VSS
from exact_ballpark.space import LEVEL_WIDTH

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
TOP = LEVEL_WIDTH[0]


def _bits(x: float) -> str:
    return "nan" if x != x else struct.pack("<d", float(x)).hex()


def _f(r: random.Random, lo: float, hi: float) -> float:
    return float(r.uniform(lo, hi))


def _world(r: random.Random, wid: int, park: str) -> dict:
    n = r.randint(2, 6)
    ids = r.sample(range(101, 999), n)
    y, z = 0.3 * TOP + _f(r, -2e3, 2e3), -0.4 * TOP + _f(r, -2e3, 2e3)
    row = r.random() < 0.35
    if row:                                                 # left and right tops first, then the middle, then near the face
        places = [(-0.5 * TOP, y, z), (1.5 * TOP, y, z), (TOP - _f(r, 500.0, 4000.0), y, z),
                  (TOP + _f(r, 500.0, 4000.0), y, z)]
        while len(places) < n:
            places.append((TOP + _f(r, -6e3, 6e3), y + _f(r, -3e3, 3e3), z + _f(r, -3e3, 3e3)))
        anchors = [(TOP, y, z), (1.5 * TOP, y, z), (-0.5 * TOP, y, z)]
    else:
        pool = [(0.0, 0.0, 0.0), (TOP, y, z), (3 * TOP, -2 * TOP, TOP)]
        anchors = r.sample(pool, r.randint(1, 3))
        places = []
        for _ in range(n):
            a = r.choice(anchors)
            places.append(tuple(c + _f(r, -8e3, 8e3) for c in a))
    balls, pilots, p = [], {}, 0
    for k, bid in enumerate(ids):
        inter = k == 0 or r.random() < 0.6
        b = {"id": bid, "mass": 10 ** r.uniform(6.0, 9.0), "radius": r.choice([_f(r, 5.0, 60.0), _f(r, 60.0, 300.0)]),
             "max_vel": r.choice([_f(r, 50.0, 500.0), _f(r, 5.0, 150.0)]), "agility": _f(r, 0.1, 2.0),
             "sf": r.choice([1.0, _f(r, 0.1, 1.0), 0.0]), "pos": list(places[k]),
             "vel": r.choice([[0.0, 0.0, 0.0], [_f(r, -150.0, 150.0) for _ in range(3)]]),
             "interactive": inter, "massive": False, "max_ang_vel": 1.0, "rot_agility": 0.9}
        balls.append(b)
        if inter and (k == 0 or r.random() < 0.7):
            pilots[bid] = p
            p += 1
    return {"id": wid, "park": park, "do": False, "balls": balls, "anchors": anchors, "special": ids[-1],
            "pilots": pilots, "ticks": [], "row": (TOP, y, z) if row else None}


_INTERRUPTS = S._INTERRUPTS


def _call(r: random.Random, w: dict, park: str) -> tuple:
    ids = [b["id"] for b in w["balls"]]
    sp = w["special"]
    movers = [i for i in ids if not (i == sp and w.get("warped"))]
    s = r.choice(movers or ids)
    other = r.choice([i for i in ids if i != s])
    refuse = r.random() < 0.08
    a = r.choice(w["anchors"])
    near = lambda c, d: [x + _f(r, -d, d) for x in c]       # noqa: E731
    k = r.random()
    if k < 0.14:
        return "go_to_point", (s, *near((a[0] + r.choice([-1.0, 1.0]) * 6e3, a[1], a[2]), 4e3))
    if k < 0.22:
        return "go_to_direction", (s, *([0.0, S._nan(), 0.0] if refuse else [_f(r, -1.0, 1.0) for _ in range(3)]))
    if k < 0.26:
        return "stop", (s,)
    if k < 0.31:
        return "set_speed_fraction", (s, r.choice([-0.5, 1.5, S._nan()]) if refuse else r.choice([0.0, 1.0,
                                                                                                    _f(r, 0, 1)]))
    if k < 0.34:
        return "set_ball_velocity", (s, *[_f(r, -200.0, 200.0) for _ in range(3)])
    if k < 0.44:
        row = w.get("row")
        if row and r.random() < 0.5:                       # across the face between two bubbles
            dst = (row[0] + r.choice([-1.0, 1.0]) * _f(r, 200.0, 4000.0), row[1] + _f(r, -2e3, 2e3), row[2])
        else:
            dst = r.choice(w["anchors"] + [(r.randint(-6, 6) * TOP + 0.5 * TOP, r.randint(-6, 6) * TOP + 0.5 * TOP,
                                            0.5 * TOP)])
            dst = near(dst, 8e3)
        return "set_ball_position", (s, *dst, r.random() < 0.15)
    if k < 0.48:
        return "set_ball_radius", (s, -5.0 if refuse else _f(r, 5.0, 300.0))
    if k < 0.52:
        return "set_max_speed", (s, -5.0 if refuse else _f(r, 10.0, 500.0))
    if k < 0.55:
        return "set_ball_mass", (s, -1.0 if refuse else 10 ** r.uniform(6.0, 9.0))
    if k < 0.57:
        return "set_ball_agility", (s, 0.0 if refuse else _f(r, 0.1, 2.0))
    if k < 0.64:
        return "set_ball_interactive", (s, r.choice([0, 1]))
    if k < 0.70 and not w.get("warped"):
        return "cloak_ball", (sp, r.choice([1, 1, 2, 3, 0, 200]), _f(r, 100.0, 6000.0))
    if k < 0.75 and not w.get("warped"):
        return "uncloak_ball", (sp,)
    if park == "modes" and k < 0.82:
        return r.choice(["follow_ball", "orbit"]), (s, r.choice([other, sp, 4242]), _f(r, 100.0, 4000.0))
    if park == "modes" and k < 0.86 and not w.get("warped"):
        w["warped"] = True                                  # the caller adds the alignment first (_tick_calls)
        b = next(x for x in w["balls"] if x["id"] == sp)
        dist = 10 ** r.uniform(8.0, 10.0)
        u = [r.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(sum(c * c for c in u)) or 1.0
        u = [c / n for c in u]
        dest = [b["pos"][i] + dist * u[i] for i in range(3)]
        return "warp_to", (sp, *dest, r.choice([0.0, 2e4]), r.randint(1500, 10000))
    return "go_to_direction", (s, *[_f(r, -1.0, 1.0) for _ in range(3)])


def _tick_calls(r: random.Random, w: dict, park: str) -> list:
    calls = []
    for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4])):
        c = _call(r, w, park)
        if c[0] == "warp_to":                               # aligned at 90 % of its top speed: activation at once
            b = next(x for x in w["balls"] if x["id"] == c[1][0])
            mv = r.choice([0.5, b["max_vel"]])
            d = [c[1][1 + i] - b["pos"][i] for i in range(3)]
            n = math.sqrt(sum(x * x for x in d))
            calls.append(("set_max_speed", (c[1][0], mv)))
            calls.append(("set_ball_velocity", (c[1][0], *[d[i] / n * 0.9 * mv for i in range(3)])))
        calls.append(c)
        x = r.random()
        nan = any(isinstance(v, float) and v != v for v in c[1])
        if x < 0.10 and not nan:
            calls.append((c[0], tuple(c[1])))               # the same call twice in a row: the second dropped
        elif x < 0.16 and not (c[1][0] == w["special"] and w.get("warped")):
            calls.append(("stop", (c[1][0],)))
    return calls


def cases(seed: int, n: int, park: str = "orient", ticks: int = 14) -> list:
    r = random.Random(f"server-space-{park}-{seed}")
    out = []
    for wid in range(n):
        w = _world(r, wid, park)
        w["ticks"] = [_tick_calls(r, w, park) for _ in range(ticks)]
        w.pop("warped", None)
        out.append(w)
    return out


def build(cs: list, park: str, device: str) -> tuple:
    """(SpaceServer, index {(world, ball id): row})."""
    from exact_ballpark.vec_modes import VecModesPark
    from exact_ballpark.vec_orient import VecOrientPark
    rows = [(w, b) for w in cs for b in w["balls"]]
    if park == "modes":
        p = VecModesPark(len(rows), device=device)
        p.load([{"id": b["id"], "world": w["id"], "time": 0, "mass": b["mass"], "radius": b["radius"],
                 "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                 "speed_fraction": b["sf"], "free": True, "massive": b["massive"], "interactive": b["interactive"],
                 "cloaked": 0, "bubble": -1, "moribund": False} for w, b in rows])
        a = VSS.SpaceModesAdapter(p)
    else:
        p = VecOrientPark(len(rows), device=device)
        p.load([{"mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"],
                 "agility": b["agility"], "speed_fraction": b["sf"]} for w, b in rows])
        a = VSS.SpaceOrientAdapter(p, world=[w["id"] for w, b in rows], ids=[b["id"] for w, b in rows],
                                   interactive=[b["interactive"] for w, b in rows],
                                   massive=[b["massive"] for w, b in rows])
    index = {(w["id"], b["id"]): i for i, (w, b) in enumerate(rows)}
    pilots = [-1] * a.R
    for w in cs:
        for bid, pp in w["pilots"].items():
            pilots[index[(w["id"], bid)]] = pp
    return VSS.SpaceServer(a, pilots), index


def coverage(per: dict, cov: dict) -> None:
    """Counts what a tick's sends hold (one world's {client: updates}): the update kinds the comparison reached."""
    from exact_ballpark.server import Marshalled
    narrow_states = set()
    for c, ups in per.items():
        if len(ups) == 2:
            cov["clients_with_two_updates"] = cov.get("clients_with_two_updates", 0) + 1
        for kind, method, state, wait, count in ups:
            if kind == "narrow":
                narrow_states.add(id(state))
            for stamp, (name, args) in state:
                key = name
                if name == "PackagedAction":
                    inner = args.value if isinstance(args, Marshalled) else args
                    key = "Packaged" + inner[0][1][0]
                elif name == "RemoveBalls":
                    key = "RemoveBalls (own lists)"
                cov[f"{kind}:{key}"] = cov.get(f"{kind}:{key}", 0) + 1
    if len(narrow_states) > 1:
        cov["ticks_with_several_bubbles_narrowcasting"] = cov.get("ticks_with_several_bubbles_narrowcasting", 0) + 1


def run(cs: list, park: str, device: str, ticks: int | None = None) -> dict:
    """Both sides on the worlds cs, tick by tick: the client update lists that differ, the ship rows that differ (the
    position, velocity, mode, flags and bubble), the volumes, the flagged worlds."""
    T = len(cs[0]["ticks"]) if ticks is None else int(ticks)
    scal = [S.ScalarWorld(w) for w in cs]
    server, index = build(cs, park, device)
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
        pos = server.a.park.pos.cpu()
        vel = server.a.park.vel.cpu()
        for wi, w in enumerate(cs):
            if flags[wi]:
                continue
            if S.canon(S._listify(want[wi])) != S.canon(S._listify(got[wi])):
                bad_sends.append((t, w["id"], S._first_diff(want[wi], got[wi])))
            sends += sum(len(v) for v in want[wi].values())
            coverage(want[wi], cov)
            lane = scal[wi].park.lane
            for b in lane.balls.values():
                r = index[(w["id"], b.id)]
                compared += 1
                bubbles.add((w["id"], b.bubble))
                mine = (bub_rows[r], tuple(_bits(x) for x in pos[r].tolist()), tuple(_bits(x) for x in vel[r].tolist()))
                theirs = (b.bubble, tuple(_bits(x) for x in b.pos), tuple(_bits(x) for x in b.vel))
                if mine != theirs:
                    bad_rows.append((t, w["id"], b.id, theirs[0], mine[0]))
    flagged = [wi for wi, f in enumerate(server.space.st["flags"].cpu().tolist()) if f]
    per_world = {}
    for wid, b in bubbles:
        per_world.setdefault(wid, set()).add(b)
    cov["follower_clean_up_stops"] = server.cleanup_stops
    return {"sends": bad_sends, "rows": bad_rows, "updates": sends, "compared": compared, "flagged": flagged,
            "ticks": T, "worlds_with_several_bubbles": sum(1 for v in per_world.values() if len(v) > 1),
            "coverage": cov}

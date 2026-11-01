# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Worlds of massive ships on Destiny's master whose collisions keep within their bubbles,
generated from a seed, and the runners the tests compare bit for bit, at two levels:

- **the partition layer over the collision park**: the scalar reference (exact_ballpark.full.FullBallpark with
  is_master=True, one park a world: its partition, bubbles, sensors and collisions, the master's GetNearbyBalls
  skipping another bubble's ball, a dead bubble's balls left out of the tick) against
  exact_ballpark.vec_space.VecSpace on its CollideView over exact_ballpark.vec_collide.VecCollidePark (on the CPU the
  layer's host path and the park's torch path; on a CUDA device the layer's kernel with the park's torch path or its
  kernel), compared after every step: tests/space_batch.py's whole state comparison (bubbles, filings, sets,
  dictionaries, sensors, events, query answers) and each ball's collision state as the park holds it (position and
  velocity bits, mLastC, mLastCollision, the collision list in order, the park's own filing);
- **the server layer**: the scalar server layer (tests/server_cases.py's ScalarWorld: Actions, Ticker, BubbleUpdater,
  ParkUpdateBatcher on a master DestinyPark over FullBallpark) against exact_ballpark.vec_server_space.SpaceServer over
  SpaceCollideAdapter, compared after every tick: every client's update list (the canonical form) and every ship's
  state (tests/server_cases.py's collision fields, the park's filing and collision list included), its radius, top
  speed, mass, agility and time factor, and its bubble;
- **Destiny's compiled code** (the conformance kit's space harness, tests/space_cases.py): the worlds as space_cases scenarios
  (`scenarios`), the master SpaceBallpark against the harness on every row kind, and the batched layer's ball rows
  against the harness's directly (`compare_harness`).

**The worlds** (2 to 6 balls, massive but for a few, free but for a rare fixed one):
- a **row** of three top boxes along x: a seed ball in the left and one in the right top filed first (bubbles 0 and
  1, not adjacent), then two ships facing each other across the face between the middle top (which takes the left
  top's bubble, its first neighbour in i-j-k order) and the right top, overlapping or about to meet: **a client's view
  would collide them, the master never does** (other bubbles); and ships beside either of them in its own top, which
  collide in their bubble;
- a **face** of two adjacent top boxes, ships on both sides in contact (one bubble: the second top takes the first's);
- the **corner** of eight top boxes at the origin, ships in contact around it (one bubble);
- a **split** world: two clusters a few top boxes apart (two bubbles), ships in contact in each.

**The records** (the layer, each step 0 to 4 a world, then the tick): GotoPoint toward another ship or across a face,
GotoDirection, Stop, the speed fraction, the velocity, jumps (SetBallPosition: next to another ship, overlapping;
across the row's face, beside the ship there; to another anchor; far: a new bubble), the re-filing setters
(SetBallRadius, SetMaxSpeed; refused values now and then), the mass and agility, interactive changes (a bubble left
without an interactive ball dies: its balls freeze, and a live ship never meets them, being in another bubble),
keep-alive balls, InitializeBubbles (rare; its quirk), CopyBubbles, a raw cloak flag (no longer a candidate), proximity
sensors (each check re-files its ball twice: the park's filing must follow the layer's), notification ranges, target
tracking, and the queries. No removal (CollideView does not model one).

**The calls** (the server, each tick 0 to 4 Actions calls a world): flights, jumps (local and not), the setters
(refused values now and then), SetBallInteractive, SetBallMassive, the special ship's cloaks (modes 1, 2, 3 and refused
ones: the normal cloak's 2 s sensor re-files it) and uncloaks, duplicates, a Stop after a command.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import torch

import server_cases as S
import space_batch as SB
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_space as VSS
from exact_ballpark import vec_space as SP
from exact_ballpark.space import LEVEL_WIDTH

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
TOP = LEVEL_WIDTH[0]                                        # 7,864,320 m
KINDS = ("row", "row", "row", "face", "corner", "split")


# ============================================================================================== the generator

def _f(r: random.Random, lo: float, hi: float) -> float:
    return float(r.uniform(lo, hi))


def _ship(r: random.Random, bid: int, pos: list, radius: float | None = None) -> dict:
    return {"id": bid, "mass": 10 ** r.uniform(6.0, 9.0), "radius": radius if radius is not None else _f(r, 30.0, 400.0),
            "max_vel": r.choice([_f(r, 30.0, 300.0), _f(r, 5.0, 60.0)]), "agility": _f(r, 0.1, 2.0),
            "sf": r.choice([1.0, _f(r, 0.2, 1.0)]), "pos": [float(c) for c in pos],
            "vel": [0.0, 0.0, 0.0], "interactive": r.random() < 0.7, "massive": r.random() < 0.92, "free": True,
            "max_ang_vel": 1.0, "rot_agility": 0.9}


def _beside(r: random.Random, other: dict, radius: float, side=None) -> list:
    """A centre at contact distance from `other` (overlapping a little, touching, or a few hundred metres apart), on
    the given x side of it (+1, -1) or anywhere."""
    while True:
        u = SB._unit(r)
        if side is not None:
            u[0] = side * abs(u[0])
        gap = r.choice([_f(r, -0.5, -0.05) * min(radius, other["radius"]), _f(r, 0.0, 50.0), _f(r, 50.0, 500.0)])
        d = other["radius"] + radius + gap
        return [other["pos"][i] + d * u[i] for i in range(3)]


def _toward(r: random.Random, a: dict, b: dict) -> list:
    """A's velocity toward B, at 20 to 100 % of its top speed."""
    d = [b["pos"][i] - a["pos"][i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in d)) or 1.0
    s = a["max_vel"] * _f(r, 0.2, 1.0)
    return [d[i] / n * s for i in range(3)]


def _world(r: random.Random, wid: int) -> dict:
    kind = r.choice(KINDS)
    ids = SB._ids(r, 6)
    y, z = 0.3 * TOP + _f(r, -2e3, 2e3), -0.4 * TOP + _f(r, -2e3, 2e3)
    balls: list = []
    if kind == "row":
        a = _ship(r, ids[0], (-0.5 * TOP, y, z), _f(r, 5.0, 50.0))         # the seeds: bubbles 0 and 1
        b = _ship(r, ids[1], (1.5 * TOP, y, z), _f(r, 5.0, 50.0))
        c = _ship(r, ids[2], (TOP - _f(r, 1.0, 300.0), y + _f(r, -200.0, 200.0), z + _f(r, -200.0, 200.0)))
        d = _ship(r, ids[3], [0.0, 0.0, 0.0])
        while True:                                                      # across the face, in contact
            d["pos"] = _beside(r, c, d["radius"], side=1.0)
            if d["pos"][0] > TOP + 1.0:
                break
        c["massive"] = d["massive"] = True
        if r.random() < 0.7:
            c["vel"], d["vel"] = _toward(r, c, d), _toward(r, d, c)
        rest = []
        for k in range(r.randint(0, 2)):                                  # beside C (middle top) or D (right top)
            host, side = (c, -1.0) if r.random() < 0.5 else (d, 1.0)
            e = _ship(r, ids[4 + k], [0.0, 0.0, 0.0])
            for _ in range(50):
                e["pos"] = _beside(r, host, e["radius"], side=side)
                if (e["pos"][0] < TOP - 1.0) if side < 0 else (e["pos"][0] > TOP + 1.0):
                    break
            if r.random() < 0.6:
                e["vel"] = _toward(r, e, host)
            rest.append(e)
        balls = [a, b] + r.sample([c, d] + rest, 2 + len(rest))
        anchors = [(TOP, y, z), (1.5 * TOP, y, z), (-0.5 * TOP, y, z)]
        row = (TOP, y, z)
        pair = (c["id"], d["id"])
    else:
        if kind == "face":
            centres = [(TOP, y, z)]
        elif kind == "corner":
            centres = [(0.0, 0.0, 0.0)]
        else:
            centres = [(0.5 * TOP, 0.5 * TOP, 0.5 * TOP), (3.5 * TOP, -1.5 * TOP, 1.5 * TOP)]
        n = r.randint(2, 5) if kind != "split" else r.randint(4, 6)
        for k in range(n):
            ctr = centres[k % len(centres)]
            mine = [x for x in balls if x["cluster"] == k % len(centres)]
            if not mine:
                pos = [ctr[i] + _f(r, -1500.0, 1500.0) for i in range(3)]
                s = _ship(r, ids[k], pos)
            else:
                host = r.choice(mine)
                s = _ship(r, ids[k], [0.0, 0.0, 0.0])
                s["pos"] = _beside(r, host, s["radius"])
                if r.random() < 0.6:
                    s["vel"] = _toward(r, s, host)
            s["cluster"] = k % len(centres)
            if k > 0 and r.random() < 0.08:
                s["free"] = False                                        # a fixed ball (a static sphere)
                s["vel"] = [0.0, 0.0, 0.0]
            balls.append(s)
        for s in balls:
            s.pop("cluster")
        anchors = list(centres)
        row, pair = None, None
    if not any(x["interactive"] for x in balls if x["free"]):
        next(x for x in balls if x["free"])["interactive"] = True
    pilots, p = {}, 0
    for x in balls:
        if x["interactive"] and x["free"] and r.random() < 0.8:
            pilots[x["id"]] = p
            p += 1
    if not pilots:
        x = next(x for x in balls if x["interactive"] and x["free"])
        pilots[x["id"]] = 0
    movers = [x["id"] for x in balls if x["free"]]
    return {"id": wid, "kind": kind, "park": "collide", "do": False, "balls": balls, "anchors": anchors, "row": row,
            "pair": pair, "special": r.choice(movers), "pilots": pilots, "steps": [], "ticks": []}


def _target_near(r: random.Random, w: dict, s: int) -> list:
    """Where a command or a jump sends ship s: beside another ship (from its start), across the row's face, near an
    anchor."""
    others = [b for b in w["balls"] if b["id"] != s]
    k = r.random()
    if w["row"] and k < 0.3:
        side = r.choice([-1.0, 1.0])
        return [w["row"][0] + side * _f(r, 1.0, 600.0), w["row"][1] + _f(r, -300.0, 300.0),
                w["row"][2] + _f(r, -300.0, 300.0)]
    if k < 0.75 and others:
        o = r.choice(others)
        me = next(b for b in w["balls"] if b["id"] == s)
        return _beside(r, o, me["radius"])
    a = r.choice(w["anchors"])
    return [c + _f(r, -3000.0, 3000.0) for c in a]


def _record(r: random.Random, w: dict) -> tuple | list:
    """One layer record of a world (tests/space_batch.py's kinds, but no removal, follows or warps)."""
    movers = [b for b in w["balls"] if b["free"]]
    b = r.choice(movers)
    s = b["id"]
    every = w["balls"]
    k = r.random()
    if k < 0.16:
        return ("goto", s, *_target_near(r, w, s))
    if k < 0.21:
        return ("dir", s, *SB._unit(r))
    if k < 0.25:
        return ("stop", s)
    if k < 0.28:
        return ("sf", s, r.choice([0.0, 1.0, _f(r, 0.1, 1.0), -0.5, 1.5]))
    if k < 0.32:
        o = r.choice(every)
        return ("vel", s, *(_toward(r, b, o) if o["id"] != s else [_f(r, -100.0, 100.0) for _ in range(3)]))
    if k < 0.41:
        if r.random() < 0.12:
            return ("pos", s, *[r.randint(-6, 6) * TOP + 0.5 * TOP for _ in range(3)])
        return ("pos", s, *_target_near(r, w, s))
    if k < 0.45:
        return ("radius", s, r.choice([_f(r, 30.0, 400.0), _f(r, 400.0, 1500.0), -5.0]))
    if k < 0.48:
        return ("maxvel", s, r.choice([_f(r, 10.0, 400.0), -1.0, 0.0]))
    if k < 0.50:
        return ("mass", s, r.choice([10 ** r.uniform(6.0, 9.0), -1.0]))
    if k < 0.52:
        return ("agility", s, r.choice([_f(r, 0.1, 2.0), 0.0]))
    if k < 0.59:
        return ("interactive", s, r.choice([0, 1]))
    if k < 0.61:
        return ("keepalive", r.choice(every)["id"])
    if k < 0.618:
        return ("init", r.choice([0, 0, s]))
    if k < 0.66:
        return ("copy",)
    if k < 0.68:
        return ("cloaked", r.choice(every)["id"], r.choice([0, 0, 1, 2]))
    if k < 0.75:
        return ("sensor", r.choice(every)["id"], r.choice([_f(r, 100.0, 2000.0), _f(r, 2000.0, 20000.0)]),
                r.choice([1.0, 1.0, 2.0, 0.5]), r.choice([0, 0, 1]))
    if k < 0.77:
        return ("unsensor", r.choice(every)["id"])
    if k < 0.80:
        return ("notify", s, r.choice([_f(r, 300.0, 5000.0), 0.0]))
    if k < 0.83:
        o = r.choice(every)
        return ("track", s, r.choice([o["id"], 0]), _f(r, 500.0, 20000.0))
    q = r.random()
    o = r.choice(every)
    if q < 0.3:
        pair = w.get("pair")
        if pair and r.random() < 0.6:
            return ("vis",) + (pair if r.random() < 0.5 else pair[::-1])
        return ("vis", s, o["id"])
    if q < 0.5:
        return ("range_ball", s, r.choice([_f(r, 300.0, 3000.0), _f(r, 3e3, 3e4)]), r.choice([0, 1]))
    if q < 0.6:
        return ("range_dist", s, _f(r, 300.0, 3e4), r.choice([0, 1]))
    if q < 0.75:
        a = r.choice(w["anchors"])
        return ("range_point", *[c + _f(r, -2e3, 2e3) for c in a], _f(r, 1e3, 3e4), r.choice([0, 1]))
    if q < 0.9:
        a = r.choice(w["anchors"])
        return ("bubble_at", *[c + _f(r, -2e3, 2e3) for c in a])
    return ("remote",)


def cases(seed: int, n: int, steps: int = 16) -> list:
    """n worlds, each with `steps` steps of layer records (0 to 4 a world a step)."""
    r = random.Random(f"space-collide-{seed}")
    out = []
    for wid in range(n):
        w = _world(r, wid)
        for _ in range(steps):
            w["steps"].append([_record(r, w) for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4]))])
        out.append(w)
    return out


def _call(r: random.Random, w: dict) -> tuple:
    """One Actions call of a world (the kinds SpaceCollideAdapter applies)."""
    movers = [b["id"] for b in w["balls"] if b["free"]]
    s = r.choice(movers)
    sp = w["special"]
    refuse = r.random() < 0.08
    k = r.random()
    if k < 0.18:
        return "go_to_point", (s, *_target_near(r, w, s))
    if k < 0.25:
        return "go_to_direction", (s, *([0.0, S._nan(), 0.0] if refuse else SB._unit(r)))
    if k < 0.29:
        return "stop", (s,)
    if k < 0.33:
        return "set_speed_fraction", (s, r.choice([-0.5, 1.5, S._nan()]) if refuse else r.choice([0.0, 1.0,
                                                                                                    _f(r, 0, 1)]))
    if k < 0.37:
        me = next(b for b in w["balls"] if b["id"] == s)
        o = r.choice(w["balls"])
        return "set_ball_velocity", (s, *(_toward(r, me, o) if o["id"] != s else [_f(r, -80.0, 80.0)
                                                                                   for _ in range(3)]))
    if k < 0.47:
        if r.random() < 0.12:
            dst = [r.randint(-6, 6) * TOP + 0.5 * TOP for _ in range(3)]
        else:
            dst = _target_near(r, w, s)
        return "set_ball_position", (s, *dst, r.random() < 0.15)
    if k < 0.51:
        return "set_ball_radius", (s, -5.0 if refuse else r.choice([_f(r, 30.0, 400.0), _f(r, 400.0, 1500.0)]))
    if k < 0.54:
        return "set_max_speed", (s, -5.0 if refuse else _f(r, 10.0, 400.0))
    if k < 0.56:
        return "set_ball_mass", (s, -1.0 if refuse else 10 ** r.uniform(6.0, 9.0))
    if k < 0.58:
        return "set_ball_agility", (s, 0.0 if refuse else _f(r, 0.1, 2.0))
    if k < 0.65:
        return "set_ball_interactive", (s, r.choice([0, 1]))
    if k < 0.69:
        return "set_ball_massive", (s, r.choice([0, 1, 1]))
    if k < 0.78:
        return "cloak_ball", (sp, r.choice([1, 1, 1, 2, 3, 0, 200]), _f(r, 100.0, 6000.0))
    if k < 0.84:
        return "uncloak_ball", (sp,)
    return "go_to_point", (s, *_target_near(r, w, s))


def _tick_calls(r: random.Random, w: dict) -> list:
    calls = []
    for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4])):
        c = _call(r, w)
        calls.append(c)
        x = r.random()
        nan = any(isinstance(v, float) and v != v for v in c[1])
        if x < 0.10 and not nan:
            calls.append((c[0], tuple(c[1])))               # the same call twice in a row: the second dropped
        elif x < 0.16:
            calls.append(("stop", (c[1][0],)))
    return calls


def server_cases(seed: int, n: int, ticks: int = 14) -> list:
    """n worlds with `ticks` ticks of Actions calls each."""
    r = random.Random(f"server-space-collide-{seed}")
    out = []
    for wid in range(n):
        w = _world(r, wid)
        for b in w["balls"]:
            b["free"] = True                                # the server's AddBall: every ship free
        w["ticks"] = [_tick_calls(r, w) for _ in range(ticks)]
        out.append(w)
    return out


# ================================================================== the layer's side: VecSpace over VecCollidePark

class _FlatPark:
    """VecCollidePark's commands and setters with masks and arguments by row (w x N + slot), its position as a flat
    view: what tests/space_batch.py's BatchedWorlds._apply calls on a park."""

    def __init__(self, park):
        self.p = park
        self.n = park.W * park.N

    def _m(self, m):
        return m.reshape(self.p.W, self.p.N)

    def _x(self, x, k=0):
        return x.reshape(self.p.W, self.p.N, k) if k else x.reshape(self.p.W, self.p.N)

    pos = property(lambda s: s.p.pos.reshape(s.n, 3),
                   lambda s, v: setattr(s.p, "pos", v.reshape(s.p.W, s.p.N, 3).contiguous()))
    old_pos = property(lambda s: s.p.old_pos.reshape(s.n, 3),
                       lambda s, v: setattr(s.p, "old_pos", v.reshape(s.p.W, s.p.N, 3).contiguous()))

    def goto_point(self, m, xyz):
        self.p.goto_point(self._m(m), self._x(xyz, 3))

    def goto_direction(self, m, xyz):
        self.p.goto_direction(self._m(m), self._x(xyz, 3))

    def stop(self, m):
        self.p.stop(self._m(m))

    def set_speed_fraction(self, m, v):
        self.p.set_speed_fraction(self._m(m), self._x(v))

    def set_velocity(self, m, xyz):
        self.p.set_velocity(self._m(m), self._x(xyz, 3))

    def set_radius(self, m, v):
        self.p.set_radius(self._m(m), self._x(v))

    def set_max_velocity(self, m, v):
        self.p.set_max_velocity(self._m(m), self._x(v))

    def set_mass(self, m, v):
        self.p.set_mass(self._m(m), self._x(v))

    def set_agility(self, m, v):
        self.p.set_agility(self._m(m), self._x(v))


class CollideWorlds(SB.BatchedWorlds):
    """The worlds on one VecCollidePark (W worlds of N slots, each world's balls in ascending id, as the park loads
    them) with the VecSpace layer on its CollideView (each world's slots in AddBall's order)."""

    def __init__(self, cs: list, device: str, kernel: str = "torch", contact_slots: int | None = None,
                 iterative: bool = False, hits: bool = True, lib: dict | None = None, friction: float = 1.0e6,
                 tick_ms: int = 1000):
        from exact_ballpark.vec_collide import VecCollidePark
        self.cs = cs
        self.dev = torch.device(device)
        self.kind = "collide"
        N = max(len(w["balls"]) for w in cs)
        W = len(cs)
        self.real = p = VecCollidePark(W, N, friction=friction, tick_interval_ms=tick_ms, device=self.dev,
                                       contact_slots=contact_slots, kernel=kernel, iterative=iterative)
        p.load([[{"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                  "agility": b["agility"], "speed_fraction": b["sf"], "pos": b["pos"], "vel": b["vel"],
                  "free": b["free"], "massive": b["massive"]} for b in w["balls"]] for w in cs])
        rows, index, inter = [], {}, [False] * (W * N)
        self.slot_of = {}
        for wi, w in enumerate(cs):
            order = sorted(b["id"] for b in w["balls"])                 # the park's slots
            rr = []
            for j, b in enumerate(w["balls"]):                          # the layer's slots: AddBall's order
                r = wi * N + order.index(b["id"])
                index[(wi, b["id"])] = (r, j)
                inter[r] = bool(b["interactive"])
                rr.append(r)
            rows.append(rr + [-1] * (N - len(rr)))
        self.index = index
        self.flat = [(r // N, None) for r in range(W * N)]
        self.view = SP.CollideView(p, interactive=inter)
        self.park = _FlatPark(p)
        self.space = SP.VecSpace(self.view, rows, hits=hits, lib=lib)
        self.space.load()
        self.answers = [[] for _ in cs]
        self.W = W

    def _evolve(self) -> None:
        self.space.tick(self.real)


def scalar_collisions(sw: SB.ScalarWorld) -> dict:
    """{ball id: (mLastC bits, mLastCollision bits, the collision list, the filing)} of a scalar world."""
    out = {}
    for b in sw.park.balls.values():
        out[b.id] = (tuple(SB._bits(x) for x in b.last_c), SB._bits(b.last_collision), tuple(b.collisions),
                     None if b.my_box is None else (b.my_box.level, tuple(b.my_box.key), tuple(b.mod)))
    return out


def batched_collisions(bw: CollideWorlds, wi: int) -> dict:
    """The same of world wi as the park holds it (its own filing: the layer's, copied in)."""
    p = bw.real
    N = p.N
    lc, lcol = p.last_c[wi].cpu().tolist(), p.last_collision[wi].cpu().tolist()
    hits, hn = p.hits[wi].cpu().tolist(), p.hits_n[wi].cpu().tolist()
    lvl, key, mod = p.level[wi].cpu().tolist(), p.key[wi].cpu().tolist(), p.mod[wi].cpu().tolist()
    out = {}
    for (w, bid), (r, j) in bw.index.items():
        if w != wi:
            continue
        s = r - wi * N
        lst = tuple(int(o) for o in hits[s][:hn[s]]) if s < p.R else ()   # the list holds the other balls' ids
        out[bid] = (tuple(SB._bits(x) for x in lc[s]), SB._bits(lcol[s]), lst, (lvl[s], tuple(key[s]), tuple(mod[s])))
    return out


def compare_layer(cs: list, device: str, kernel: str = "torch", steps: int | None = None, hits: bool = True,
                  lib: dict | None = None, contact_slots: int | None = None, iterative: bool = False,
                  graph: bool = False) -> dict:
    """Both sides on the worlds cs, step by step (tests/space_batch.py's compare, with the collision state): the
    differences, the volumes compared, the flagged worlds and the branch counts. `graph`: the layer's tick replayed as
    a CUDA graph (SpaceGraph, captured after the first step's eager tick)."""
    T = len(cs[0]["steps"]) if steps is None else int(steps)
    scal = [SB.ScalarWorld(w) for w in cs]
    if iterative:
        for sw in scal:
            sw.park.iterative_collision = True
    bw = CollideWorlds(cs, device, kernel=kernel, hits=hits, lib=lib, contact_slots=contact_slots,
                       iterative=iterative)
    g = None
    diffs, compared = [], Counter()
    for sw in scal:
        SB.scalar_events(sw)
    for t in range(T):
        for wi, w in enumerate(cs):
            for rec in w["steps"][t]:
                scal[wi].apply(rec)
        for wi in range(len(cs)):
            scal[wi].evolve()
        bw.space.clear_events()
        width = max((len(w["steps"][t]) for w in cs), default=0)
        for j in range(width):
            bw._apply([w["steps"][t][j] if j < len(w["steps"][t]) else None for w in cs])
        if graph and t >= 1:
            if g is None:
                g = SP.SpaceGraph(bw.space, bw.real)
            g.tick()
        else:
            bw._evolve()
        flags = bw.space.st["flags"].cpu().tolist()
        for wi in range(len(cs)):
            if flags[wi]:
                continue
            want, got = SB.scalar_state(scal[wi]), SB.batched_state(bw, wi)
            if set(want["balls"]) != set(got["balls"]):
                diffs.append((t, wi, "balls", sorted(set(want["balls"]) ^ set(got["balls"]))))
            for bid in want["balls"]:
                if bid not in got["balls"]:
                    continue
                for key, val in want["balls"][bid].items():
                    compared["ball_fields"] += 1
                    if got["balls"][bid][key] != val:
                        diffs.append((t, wi, bid, key, val, got["balls"][bid][key]))
            for key in ("dicts", "interactives", "keep_alives", "bubble_id", "tops", "proximity"):
                compared["world_fields"] += 1
                if want["world"][key] != got["world"][key]:
                    diffs.append((t, wi, key, want["world"][key], got["world"][key]))
            cw, cg = scalar_collisions(scal[wi]), batched_collisions(bw, wi)
            for bid, val in cw.items():
                compared["collision_fields"] += 4
                compared["collision_list_entries"] += len(val[2])
                if cg.get(bid) != val:
                    diffs.append((t, wi, bid, "collision", val, cg.get(bid)))
            ev_w = SB.scalar_events(scal[wi])
            ev_g = bw.space.events(wi)
            compared["events"] += len(ev_w)
            if ev_w != ev_g:
                diffs.append((t, wi, "events", ev_w, ev_g))
            aw, ag = scal[wi].answers, bw.answers[wi]
            compared["answers"] += len(aw)
            if aw != ag:
                diffs.append((t, wi, "answers", aw, ag))
            scal[wi].answers, bw.answers[wi] = [], []
            compared["ball_steps"] += len(want["balls"])
            compared["ball_steps_with_collisions"] += sum(1 for v in cw.values() if v[2])
    flagged = [wi for wi, f in enumerate(bw.space.st["flags"].cpu().tolist()) if f]
    ref_hits = Counter()
    for sw in scal:
        ref_hits.update(sw.park.trace)
    return {"diffs": diffs, "compared": dict(compared), "flagged": flagged, "hits": bw.space.hit_counts(),
            "ref_hits": dict(ref_hits), "steps": T, "graph": g is not None}


# ============================================================================= the server's side: SpaceServer

EXTRA = ("radius", "max_vel", "mass", "agility", "time_factor")


def collide_rows(server, index: dict) -> dict:
    """{(world, ball id): tests/server_cases.py's FIELDS_COLLIDE} of the server's VecCollidePark, as
    server_cases.batched_rows' collide branch reads them, but for the collision list, which holds the other balls'
    ids (VecCollidePark._update and the kernel's update append the other's id; batched_rows maps them as slots)."""
    a, p = server.a, server.a.park
    R = a.R
    flat = lambda t: t.reshape(R, *t.shape[2:]).cpu()          # noqa: E731
    cols = {}
    for pre, attr in (("p", "pos"), ("v", "vel"), ("op", "old_pos"), ("ov", "old_vel"), ("g", "goto"),
                      ("a", "last_g"), ("c", "last_c")):
        t = flat(getattr(p, attr))
        for i, ax in enumerate("xyz"):
            cols[pre + ax] = t[:, i].tolist()
    for attr in ("last_collision", "speed_fraction"):
        cols[attr] = flat(getattr(p, attr)).tolist()
    cols["mode"] = flat(p.mode).tolist()
    cols["massive"] = a.massive.to(I64).cpu().tolist()
    cols["cloaked"] = a.cloaked.cpu().tolist()
    cols["interactive"] = a.interactive.to(I64).cpu().tolist()
    cols["box_level"] = flat(p.level).tolist()
    key, mod = flat(p.key), flat(p.mod)
    for i, ax in enumerate("xyz"):
        cols["box_i" + ax] = key[:, i].tolist()
        cols["mod_" + ax] = mod[:, i].tolist()
    hits, hn = p.hits.cpu(), p.hits_n.cpu()
    cols["collisions"] = []
    for w in range(p.W):
        for s in range(p.N):
            cols["collisions"].append(tuple(int(o) for o in hits[w, s, :int(hn[w, s])].tolist()) if s < p.R else ())
    return {k: tuple(cols[f][r] for f in S.FIELDS_COLLIDE) for k, r in index.items()}


def build_server(cs: list, device: str, kernel: str = "torch", contact_slots: int | None = None,
                 added: bool = True) -> tuple:
    """(SpaceServer over SpaceCollideAdapter, index {(world, ball id): row}); `added` False lists each world's rows
    in the park's slot order (ascending id) instead of AddBall's (a planted fault)."""
    from exact_ballpark.vec_collide import VecCollidePark
    slots = max(len(w["balls"]) for w in cs)
    p = VecCollidePark(len(cs), slots, device=device, contact_slots=contact_slots, kernel=kernel)
    p.load([[{"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
              "agility": b["agility"], "speed_fraction": b["sf"], "pos": b["pos"], "vel": b["vel"], "free": True,
              "massive": b["massive"]} for b in w["balls"]] for w in cs])
    index, inter = {}, []
    for wi, w in enumerate(cs):
        by_id = {b["id"]: b for b in w["balls"]}
        row_inter = [False] * slots
        for s, bid in enumerate(sorted(by_id)):
            index[(w["id"], bid)] = wi * slots + s
            row_inter[s] = bool(by_id[bid]["interactive"])
        inter += row_inter
    a = VSS.SpaceCollideAdapter(p, interactive=inter,
                                added=[[b["id"] for b in w["balls"]] for w in cs] if added else None)
    pilots = [-1] * a.R
    for w in cs:
        for bid, pp in w["pilots"].items():
            pilots[index[(w["id"], bid)]] = pp
    return VSS.SpaceServer(a, pilots), index


def run_server(cs: list, device: str, kernel: str = "torch", contact_slots: int | None = None,
               ticks: int | None = None, added: bool = True) -> dict:
    """Both sides on the worlds cs, tick by tick: the client update lists that differ, the ship rows that differ
    (tests/server_cases.py's collision fields, EXTRA, the bubble), the volumes, the flagged worlds, what was reached."""
    T = len(cs[0]["ticks"]) if ticks is None else int(ticks)
    old = S.srv.is_dynamical_orientation_enabled()
    try:
        scal = [S.ScalarWorld(w) for w in cs]
        server, index = build_server(cs, device, kernel, contact_slots, added)
        bad_sends, bad_rows, sends, compared, cov = [], [], 0, 0, Counter()
        bubbles: set = set()
        for t in range(T):
            want = [sw.tick(w["ticks"][t]) for sw, w in zip(scal, cs)]
            for kind, *data in S.pushes(cs, t, server.dev):
                server.push(kind, *data)
            server.pre_tick(snapshot=True)
            server.evolve()
            server.post_tick()
            got = VSS.decode(server)
            flags = server.space.st["flags"].cpu().tolist()
            rows_b = collide_rows(server, index)
            p = server.a.park
            R = server.a.R
            ext = {k: getattr(p, k).reshape(R).cpu().tolist() for k in EXTRA}
            bub = server.bubbles_of(torch.arange(R, device=server.dev)).cpu().tolist()
            for wi, w in enumerate(cs):
                if flags[wi]:
                    continue
                if S.canon(S._listify(want[wi])) != S.canon(S._listify(got[wi])):
                    bad_sends.append((t, w["id"], S._first_diff(want[wi], got[wi])))
                sends += sum(len(v) for v in want[wi].values())
                SSC_coverage(want[wi], cov)
                lane = scal[wi].park.lane
                srows = S.scalar_rows(scal[wi], "collide", False)
                for b in lane.balls.values():
                    r = index[(w["id"], b.id)]
                    compared += 1
                    bubbles.add((w["id"], b.bubble))
                    mine = (rows_b[(w["id"], b.id)], tuple(ext[k][r] for k in EXTRA), bub[r])
                    theirs = (srows[b.id], tuple(getattr(b, k) for k in EXTRA), b.bubble)
                    if not (S.same_rows(mine[0], theirs[0]) and S.same_rows(mine[1], theirs[1])
                            and mine[2] == theirs[2]):
                        names = S.FIELDS_COLLIDE
                        diff = [f for f, x, y in zip(names, theirs[0], mine[0]) if not S.fc.same(x, y)]
                        diff += [f for f, x, y in zip(EXTRA, theirs[1], mine[1]) if not S.fc.same(x, y)]
                        if mine[2] != theirs[2]:
                            diff.append("bubble")
                        bad_rows.append((t, w["id"], b.id, diff))
                    if b.collisions:
                        cov["ship_ticks_with_collisions"] += 1
        flagged = [wi for wi, f in enumerate(server.space.st["flags"].cpu().tolist()) if f]
        per_world: dict = {}
        for wid, b in bubbles:
            per_world.setdefault(wid, set()).add(b)
        ref = Counter()
        for sw in scal:
            ref.update(getattr(sw.park.lane, "trace", None) or {})
        cov["follower_clean_up_stops"] = server.cleanup_stops
        return {"sends": bad_sends, "rows": bad_rows, "updates": sends, "compared": compared, "flagged": flagged,
                "ticks": T, "worlds_with_several_bubbles": sum(1 for v in per_world.values() if len(v) > 1),
                "coverage": dict(cov), "server": server}
    finally:
        if old:
            S.srv.enable_dynamical_orientation()
        else:
            S.srv.disable_dynamical_orientation()


def SSC_coverage(per: dict, cov: Counter) -> None:
    """tests/server_space_cases.py's coverage of a tick's sends (the update kinds reached)."""
    import server_space_cases as SSC
    d: dict = {}
    SSC.coverage(per, d)
    cov.update(d)


# ========================================================== the park's kernel against its torch path, in lockstep

PARK_FIELDS = ("pos", "vel", "old_pos", "old_vel", "goto", "last_g", "last_c", "last_collision", "mode", "speed_fraction",
               "mass", "agility", "radius", "max_vel", "time_factor", "massive", "cloaked", "level", "key", "mod",
               "bubble", "overflow")


def _park_state(bw: CollideWorlds) -> dict:
    """Every compared tensor of the park (the collision lists to their lengths) and of the layer, on the host."""
    p = bw.real
    out = {k: getattr(p, k).cpu() for k in PARK_FIELDS}
    hn = p.hits_n.cpu()
    hits = p.hits.cpu()
    out["hits"] = [[tuple(hits[w, s, :int(hn[w, s])].tolist()) for s in range(p.R)] for w in range(p.W)]
    out["hits_n"] = hn
    for k, v in bw.space.st.items():
        if v is not None:
            out["layer." + k] = v.cpu()
    return out


def _same_tensor(a, b) -> bool:
    if isinstance(a, list):
        return a == b
    if a.dtype.is_floating_point:
        return a.shape == b.shape and bool((a.view(torch.int64) == b.view(torch.int64)).all())
    return a.shape == b.shape and bool((a == b).all())


def lockstep(cs: list, device: str, kernel_a: str = "torch", kernel_b: str = "cuda", steps: int | None = None,
             contact_slots: int | None = None) -> dict:
    """The worlds cs on two CollideWorlds (the park's `kernel_a` and `kernel_b`), the same records, every compared
    tensor of the park and the layer equal after every step (float64 by their bits): {"diffs": [(step, name)],
    "tensors": count compared, "steps"}."""
    T = len(cs[0]["steps"]) if steps is None else int(steps)
    a = CollideWorlds(cs, device, kernel=kernel_a, contact_slots=contact_slots)
    b = CollideWorlds(cs, device, kernel=kernel_b, contact_slots=contact_slots)
    diffs, n = [], 0
    for t in range(T):
        for bw in (a, b):
            bw.space.clear_events()
            bw.step(t)
        sa, sb = _park_state(a), _park_state(b)
        for k in sa:
            n += 1
            if not _same_tensor(sa[k], sb[k]):
                diffs.append((t, k))
    return {"diffs": diffs, "tensors": n, "steps": T}


# ================================================== Destiny's compiled code (the conformance kit's space harness), directly

def HARNESS_PATH():
    """The conformance kit's space harness (tests/space_cases.py's HARNESS)."""
    import space_cases as sc
    return sc.HARNESS


HARNESS_FIELDS = ("bubble", "old_bubble", "box_level", "box_ix", "box_iy", "box_iz", "mod_x", "mod_y", "mod_z", "px",
                  "py", "pz", "vx", "vy", "vz", "cx", "cy", "cz", "last_collision", "radius", "collisions",
                  "interactive", "cloaked")


def _as_record(sc, step: int, rec: tuple):
    """One layer record as tests/space_cases.py's record at `step` (None for the follower query: no ball follows)."""
    kind, a = rec[0], rec[1:]
    k = sc._k
    simple = {"goto": sc.GOTO_OP, "dir": sc.DIR_OP, "pos": sc.POS_OP, "vel": sc.VEL_OP}
    if kind in simple:
        return k(step, simple[kind], a[0], x=a[1], y=a[2], z=a[3])
    one = {"sf": sc.SF_OP, "radius": sc.RADIUS_OP, "maxvel": sc.MAXVEL_OP, "mass": sc.MASS_OP,
           "agility": sc.AGILITY_OP, "notify": sc.NOTIFY_OP}
    if kind in one:
        return k(step, one[kind], a[0], rr=a[1])
    if kind == "stop":
        return k(step, sc.STOP_OP, a[0])
    if kind == "interactive":
        return k(step, sc.INTERACTIVE_OP, a[0], w=int(a[1]))
    if kind == "keepalive":
        return k(step, sc.KEEPALIVE_OP, a[0])
    if kind == "init":
        return k(step, sc.INIT_BUBBLES_OP, a[0])
    if kind == "copy":
        return k(step, sc.COPY_BUBBLES_OP)
    if kind == "cloaked":
        return k(step, sc.CLOAK_OP, a[0], w=int(a[1]))
    if kind == "sensor":
        return k(step, sc.SENSOR_OP, a[0], b=0, w=int(a[3]), x=a[2], rr=a[1])
    if kind == "unsensor":
        return k(step, sc.UNSENSOR_OP, a[0])
    if kind == "track":
        return k(step, sc.TRACK_OP, a[0], b=int(a[1]), rr=a[2])
    if kind == "vis":
        return k(step, sc.VIS_Q, a[0], b=a[1])
    if kind in ("range_ball", "range_dist"):
        return k(step, sc.RANGE_BALL_Q if kind == "range_ball" else sc.RANGE_DIST_Q, a[0], w=int(a[2]), rr=a[1])
    if kind == "range_point":
        return k(step, sc.RANGE_POINT_Q, 0, w=int(a[4]), x=a[0], y=a[1], z=a[2], rr=a[3])
    if kind == "bubble_at":
        return k(step, sc.BUBBLE_AT_Q, 0, x=a[0], y=a[1], z=a[2])
    if kind == "remote":
        return None
    raise ValueError(kind)


def scenarios(cs: list) -> list:
    """The worlds as tests/space_cases.py's scenarios, for Destiny's compiled code (the conformance kit's space harness) and the
    master SpaceBallpark: AddBall at step 0 in AddBall's order, each step t's records at step t + 1 before its tick."""
    import space_cases as sc
    out = []
    for i, w in enumerate(cs):
        recs = [("B", 0, {"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                          "free": b["free"], "global": False, "massive": b["massive"], "interactive": b["interactive"],
                          "junk": False, "pos": list(b["pos"]), "vel": list(b["vel"]), "agility": b["agility"],
                          "sf": b["sf"]}) for b in w["balls"]]
        for t, step in enumerate(w["steps"]):
            for rec in step:
                k = _as_record(sc, t + 1, rec)
                if k is not None:
                    recs.append(k)
        out.append({"id": i + 1, "kind": "g5", "friction": 1.0e6, "tick": 1000, "steps": len(w["steps"]),
                    "new_orbit": False, "master": True, "seed": 1, "start": 0, "records": recs})
    return out


def compare_harness(cs: list, device: str, kernel: str = "torch", want: dict | None = None) -> dict:
    """The batched layer over the collision park against Destiny's compiled code directly: every ball's
    HARNESS_FIELDS after every step against the harness's rows (doubles by their bits); the worlds the layer flags
    left out. {"diffs", "fields", "rows", "missing"}."""
    import space_cases as sc
    want = sc.oracle(scenarios(cs)) if want is None else want
    idx = {f: i for i, f in enumerate(sc.S_FIELDS)}
    rows = {(r[0], r[1], r[2]): r for r in want["S"]}
    bw = CollideWorlds(cs, device, kernel=kernel)
    p = bw.real
    diffs, n, nrows, missing = [], 0, 0, []
    for t in range(len(cs[0]["steps"])):
        bw.space.clear_events()
        bw.step(t)
        flags = bw.space.st["flags"].cpu().tolist()
        tv = {k: v.cpu() for k, v in bw.view.tensors().items()}
        st = {k: v.cpu() for k, v in bw.space.st.items() if v is not None}
        lc, lcol = p.last_c.reshape(-1, 3).cpu(), p.last_collision.reshape(-1).cpu()
        hits, hn = p.hits.reshape(p.W * p.R, -1).cpu().tolist(), p.hits_n.reshape(-1).cpu().tolist()
        for (wi, bid), (r, j) in bw.index.items():
            if flags[wi] or not st["in_park"][wi, j]:
                continue
            want_row = rows.get((wi + 1, t + 1, bid))
            if want_row is None:
                missing.append((t, wi, bid))
                continue
            s = r - wi * p.N
            hr = wi * p.R + s
            mine = {"bubble": int(st["bubble"][wi, j]), "old_bubble": int(st["old_bubble"][wi, j]),
                    "box_level": int(st["lvl"][wi, j]), "radius": float(tv["radius"][r]),
                    "last_collision": float(lcol[r]), "interactive": int(tv["interactive"][r]),
                    "cloaked": int(tv["cloaked"][r]),
                    "collisions": tuple(int(o) for o in hits[hr][:hn[hr]]) if s < p.R else ()}
            for i, ax in enumerate("xyz"):
                mine["box_i" + ax] = int(st["key"][wi, j, i])
                mine["mod_" + ax] = int(st["mod"][wi, j, i])
                mine["p" + ax], mine["v" + ax] = float(tv["pos"][r, i]), float(tv["vel"][r, i])
                mine["c" + ax] = float(lc[r, i])
            nrows += 1
            for f in HARNESS_FIELDS:
                n += 1
                if not sc.same(want_row[idx[f]], mine[f]):
                    diffs.append((t, wi, bid, f, want_row[idx[f]], mine[f]))
    return {"diffs": diffs, "fields": n, "rows": nrows, "missing": missing}

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Worlds for Destiny's space on the master batched, generated from a seed, and the two
runners the tests compare bit for bit: the scalar reference (exact_ballpark.full.FullBallpark with is_master=True, one
park a world, its partition, bubbles and sensors those of exact_ballpark.space) and the batched layer
(exact_ballpark.vec_space.VecSpace over VecOrientPark on the CPU, or VecModesPark on a CUDA device).

A world is 2 to 6 balls around one to three anchors: the origin (a corner of eight top boxes and of every level's
boxes), a top box's face, a far point (its own bubble), a box face at some finer level. Each step applies 0 to 4
records per world, then the park's tick: commands that carry ships across top faces (GotoPoint, GotoDirection, Stop,
the speed fraction, the velocity), jumps (SetBallPosition: to another anchor, a far point, a new bubble), the re-filing
setters (SetBallRadius, SetMaxSpeed; refused values now and then), the mass and agility, interactive changes (a bubble
left without an interactive ball dies and its balls freeze), keep-alive balls, InitializeBubbles (its quirk) and
UpdateBallBubble, CopyBubbles, a raw cloak flag, immediate removal, proximity sensors (ranges, periods, only
interactives; added, removed), notification ranges and target tracking, and the queries (CheckVisibility with massive
stationary occluders, GetBallIdsInRange from a ball and a point, GetBallIdsAndDistInRange, GetBubbleAtCoordinates,
GetRemoteFollowers). On a CUDA device (VecModesPark) also follows and orbits and warps between far top boxes (a ship
in warp proper takes a new bubble at each tick; its drop-out insert). Free balls are not massive and massive balls
are stationary occluders apart from each other, so no collision is computed on either side (the parks under the layer
have none).

After every step the runners' records are compared: each ball's state (bubble, old bubble, filing: level, key, sides;
its place in the interactive and keep-alive sets; position and velocity bits; its sensor, tracking and notification
state and its sensor's list in order), each world's bubble dictionaries in insertion order, interactive and
keep-alive sets, bubble id counter, the top boxes that exist with their bubbles, the proximity balls in order, the
events (DoProximity, DoRange, DoTargetTracking, DoFinalCleanup) in Destiny's order, and every query's answer.
"""

from __future__ import annotations

import math
import random
import struct
from collections import Counter

import torch

from exact_ballpark import vec_space as VS
from exact_ballpark.full import FullBallpark
from exact_ballpark.modes import WARP, _float32
from exact_ballpark.space import LEVEL_WIDTH

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
TOP = LEVEL_WIDTH[0]                                        # 7,864,320 m
EVENT_NAMES = ("DoProximity", "DoRange", "DoTargetTracking", "DoFinalCleanup")
QUERY_KINDS = ("vis", "range_ball", "range_dist", "range_point", "bubble_at", "remote")


# ============================================================================================== the generator

def _f(r: random.Random, lo: float, hi: float) -> float:
    return float(r.uniform(lo, hi))


def _unit(r: random.Random) -> list:
    while True:
        u = [r.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        if n > 1e-3:
            return [c / n for c in u]


def _ids(r: random.Random, n: int) -> list:
    out: list = []
    while len(out) < n:
        k = r.choice([r.randrange(1, 40), r.randrange(100, 2000), r.randrange(10 ** 5, 10 ** 9)])
        if r.random() < 0.3 and out:
            k = out[-1] + 13 * r.randint(1, 3)              # the same unordered_map bucket (13 buckets)
        if k not in out:
            out.append(k)
    return out


def _anchors(r: random.Random) -> list:
    pool = [(0.0, 0.0, 0.0), (TOP, 0.0, 0.0), (0.0, -TOP, 0.0), (0.0, 0.0, TOP), (TOP, TOP, 0.0),
            (3 * TOP, -2 * TOP, TOP), (-5 * TOP, 4 * TOP, 2 * TOP)]
    lvl = r.randrange(1, 8)
    w = LEVEL_WIDTH[lvl]
    pool.append((r.randint(-3, 3) * w + 0.5 * TOP, r.randint(-3, 3) * w + 0.5 * TOP, 0.25 * TOP))
    return r.sample(pool, r.randint(1, 3))


def _near(r: random.Random, a, spread: float = 9000.0) -> list:
    out = []
    for c in a:
        k = r.random()
        if k < 0.15:
            out.append(c)                                   # on the face itself
        elif k < 0.3:
            out.append(c + r.choice([-1.0, 1.0]) * _f(r, 0.0, 2.0))
        else:
            out.append(c + _f(r, -spread, spread))
    return out


def _row_world(r: random.Random, wid: int, park: str) -> dict:
    """Three top boxes in a row along x: the outer two filed first (bubbles 0 and 1, not adjacent), then the middle
    one, which takes the first neighbour's bubble (i-j-k order: 0), so the middle and the right top are adjacent with
    different bubbles; two balls face each other across that face, 1 to 9 km apart."""
    n = r.randint(4, 6)
    ids = _ids(r, n)
    y, z = 0.3 * TOP + _f(r, -2e3, 2e3), -0.4 * TOP + _f(r, -2e3, 2e3)
    places = [(-0.5 * TOP, y, z), (1.5 * TOP, y, z), (TOP - _f(r, 500.0, 4500.0), y + _f(r, -2e3, 2e3), z),
              (TOP + _f(r, 500.0, 4500.0), y + _f(r, -2e3, 2e3), z)]
    while len(places) < n:
        places.append((TOP + _f(r, -6e3, 6e3), y + _f(r, -3e3, 3e3), z + _f(r, -3e3, 3e3)))
    balls = []
    for k, bid in enumerate(ids):
        balls.append({"id": bid, "mass": 10 ** r.uniform(5.5, 9.0), "radius": r.choice([_f(r, 1.0, 60.0),
                                                                                            _f(r, 60.0, 400.0)]),
                      "max_vel": r.choice([_f(r, 50.0, 400.0), _f(r, 1.0, 50.0)]), "agility": _f(r, 0.05, 2.0),
                      "sf": r.choice([1.0, _f(r, 0.1, 1.0)]), "pos": list(places[k]),
                      "vel": [0.0, 0.0, 0.0] if k < 2 else [_f(r, -60.0, 60.0) for _ in range(3)],
                      "interactive": k < 2 or r.random() < 0.6, "massive": False, "free": True, "occluder": False})
    anchors = [(TOP, y, z), (1.5 * TOP, y, z), (-0.5 * TOP, y, z)]
    return {"id": wid, "park": park, "balls": balls, "anchors": anchors, "steps": [], "pair": (ids[2], ids[3]),
            "row": (TOP, y, z)}


def _world(r: random.Random, wid: int, park: str) -> dict:
    if r.random() < 0.25:
        return _row_world(r, wid, park)
    n = r.randint(2, 6)
    ids = _ids(r, n)
    anchors = _anchors(r)
    balls = []
    occluders = 0
    for k, bid in enumerate(ids):
        a = r.choice(anchors)
        occ = k > 0 and occluders < 2 and r.random() < 0.25
        b = {"id": bid, "mass": 10 ** r.uniform(5.5, 9.0),
             "radius": r.choice([_f(r, 1.0, 60.0), _f(r, 60.0, 400.0), _f(r, 400.0, 3000.0)]),
             "max_vel": r.choice([_f(r, 50.0, 1500.0), _f(r, 300.0, 1500.0), _f(r, 1.0, 199.0)]),
             "agility": _f(r, 0.05, 2.0), "sf": r.choice([1.0, _f(r, 0.1, 1.0)]),
             "pos": _near(r, a), "vel": r.choice([[0.0, 0.0, 0.0], [_f(r, -300.0, 300.0) for _ in range(3)]]),
             "interactive": r.random() < 0.5, "massive": False, "free": True, "occluder": occ}
        if occ:                                             # a stationary massive ball, apart from the other
            occluders += 1
            b["vel"] = [0.0, 0.0, 0.0]
            b["massive"] = True
            b["radius"] = _f(r, 200.0, 2500.0)
            b["pos"] = [a[0] + occluders * 40000.0 * (1 if r.random() < 0.5 else -1), a[1] + _f(r, -3e3, 3e3),
                        a[2] + _f(r, -3e3, 3e3)]
            if park == "modes" and r.random() < 0.5:
                b["free"] = False                           # a fixed ball (VecModesPark has the flag)
        balls.append(b)
    if not any(b["interactive"] for b in balls):
        balls[0]["interactive"] = True
    pair = None
    occ = [b for b in balls if b["occluder"]]
    others = [b for b in balls if not b["occluder"]]
    if occ and len(others) >= 2 and r.random() < 0.8:      # two balls flanking an occluder, the segment through it
        o = occ[0]
        for sgn, b in ((1.0, others[0]), (-1.0, others[1])):
            d = o["radius"] + _f(r, 200.0, 15000.0)
            b["pos"] = [o["pos"][0] + sgn * d, o["pos"][1] + _f(r, -0.4, 0.4) * o["radius"],
                        o["pos"][2] + _f(r, -0.4, 0.4) * o["radius"]]
            b["vel"] = [0.0, 0.0, 0.0] if r.random() < 0.5 else [_f(r, -30.0, 30.0) for _ in range(3)]
        pair = (others[0]["id"], others[1]["id"])
    return {"id": wid, "park": park, "balls": balls, "anchors": anchors, "steps": [], "pair": pair}


def _record(r: random.Random, w: dict, alive: list, park: str, state: dict) -> tuple | None:
    movers = [b for b in w["balls"] if b["id"] in alive and not b["occluder"] and b["id"] not in state["warping"]]
    every = [b for b in w["balls"] if b["id"] in alive]
    if not every:
        return None
    b = r.choice(movers) if movers else r.choice(every)
    s = b["id"]
    a = r.choice(w["anchors"])
    row = w.get("row")
    if row and r.random() < 0.25:                           # across the face between two bubbles, and a query there
        x = r.choice([bb for bb in w["balls"] if bb["id"] in alive])
        if r.random() < 0.5:
            return ("pos", x["id"], row[0] + r.choice([-1.0, 1.0]) * _f(r, 200.0, 4000.0), row[1] + _f(r, -2e3, 2e3),
                    row[2] + _f(r, -2e3, 2e3))
        return ("range_ball", x["id"], _f(r, 3e3, 2e4), r.choice([0, 1]))
    k = r.random()
    if k < 0.14:
        target = _near(r, (a[0] + r.choice([-1.0, 1.0]) * 6000.0, a[1], a[2]), 4000.0)
        if r.random() < 0.3:
            target = [c + _f(r, -3000.0, 3000.0) for c in b["pos"]]
        if r.random() < 0.2:
            target = [r.choice([1.0, -1.0]) * 1e9, _f(r, -1e3, 1e3), 0.0]
        return ("goto", s, *target)
    if k < 0.20:
        return ("dir", s, *[_f(r, -1.0, 1.0) for _ in range(3)])
    if k < 0.24:
        return ("stop", s)
    if k < 0.28:
        return ("sf", s, r.choice([0.0, 1.0, _f(r, 0.1, 1.0), -0.5, 1.5]))
    if k < 0.31:
        return ("vel", s, *[_f(r, -400.0, 400.0) for _ in range(3)])
    if k < 0.37:
        dst = r.choice(w["anchors"] + [(r.randint(-9, 9) * TOP + 0.5 * TOP, r.randint(-9, 9) * TOP + 0.5 * TOP,
                                        r.randint(-9, 9) * TOP + 0.5 * TOP)])
        return ("pos", s, *_near(r, dst))
    if k < 0.41:
        return ("radius", s, r.choice([_f(r, 1.0, 60.0), _f(r, 60.0, 3000.0), -5.0]))
    if k < 0.45:
        return ("maxvel", s, r.choice([_f(r, 10.0, 1500.0), -1.0, 0.0]))
    if k < 0.47:
        return ("mass", s, r.choice([10 ** r.uniform(5.5, 9.0), -1.0]))
    if k < 0.49:
        return ("agility", s, r.choice([_f(r, 0.05, 2.0), 0.0]))
    if k < 0.55:
        return ("interactive", s, r.choice([0, 1]))
    if k < 0.57:
        return ("keepalive", s)
    if k < 0.585:
        return ("init", r.choice([0, 0, 0, s]))
    if k < 0.66:
        return ("copy",)
    if k < 0.675:
        o = r.choice(every)
        return ("cloaked", o["id"], r.choice([0, 1, 2]))
    if k < 0.685 and len(alive) > 2:
        return ("remove", s)
    if k < 0.74:
        return ("sensor", r.choice(every)["id"], r.choice([_f(r, 300.0, 3000.0), _f(r, 3000.0, 30000.0)]),
                r.choice([2.0, 1.0, 3.0, 0.5]), r.choice([0, 0, 1]))
    if k < 0.76:
        return ("unsensor", r.choice(every)["id"])
    if k < 0.79:
        if r.random() < 0.5:                                # a goto point near its start, and a range around it
            return ("notify", s, r.choice([_f(r, 1000.0, 20000.0), _f(r, 100.0, 3000.0)]))
        return ("notify", s, r.choice([0.0, _f(r, 100.0, 5000.0), -1.0]))
    if k < 0.82:
        o = r.choice(every)
        return ("track", s, r.choice([o["id"], o["id"], 0, 987654321]), r.choice([_f(r, 500.0, 30000.0), 0.0]))
    if park == "modes" and k < 0.86 and len(movers) > 1:
        o = r.choice([x for x in every if x["id"] != s])
        return (r.choice(["follow", "orbit"]), s, o["id"], _f(r, 100.0, 5000.0))
    if park == "modes" and k < 0.88 and not state["warping"]:
        # a warp that finishes within the run (warp factors as the cases: 1.5 to 10 AU/s): the ship aligned at
        # 90 % of its top speed first (activation at once, TRV-706); a top speed of 0.5 m/s drops out below
        # sqrt(0.1) m/s, so the drop-out re-files it (phase 3's insert)
        state["warping"].add(s)
        mv = r.choice([0.5, b["max_vel"]])
        u = _unit(r)
        dist = 10 ** r.uniform(8.0, 10.0)
        dest = [b["pos"][i] + dist * u[i] for i in range(3)]
        if r.random() < 0.3:
            return [("maxvel", s, mv), ("warpin", s, *dest, r.randint(1500, 6000))]
        return [("maxvel", s, mv), ("vel", s, *[u[i] * 0.9 * mv for i in range(3)]),
                ("warp", s, *dest, r.choice([0.0, 2e4]), r.randint(1500, 10000))]
    q = r.random()
    o = r.choice(every)
    if q < 0.3:
        pair = w.get("pair")
        if pair and r.random() < 0.6 and pair[0] in alive and pair[1] in alive:
            return ("vis",) + (pair if r.random() < 0.5 else pair[::-1])
        return ("vis", s, o["id"])
    if q < 0.5:
        return ("range_ball", r.choice([s, 777777777]), r.choice([_f(r, 1e3, 3e4), _f(r, 3e4, 3e5)]),
                r.choice([0, 1]))
    if q < 0.6:
        return ("range_dist", s, _f(r, 1e3, 3e5), r.choice([0, 1]))
    if q < 0.75:
        return ("range_point", *_near(r, a), _f(r, 1e3, 3e5), r.choice([0, 1]))
    if q < 0.9:
        return ("bubble_at", *_near(r, r.choice(w["anchors"] + [(5e9, 0.0, 0.0)])))
    return ("remote",)


def cases(seed: int, n: int, park: str = "orient", steps: int = 16) -> list:
    """n worlds for a park ("orient": VecOrientPark, CPU or GPU; "modes": VecModesPark, a CUDA device), each with
    `steps` steps of records."""
    r = random.Random(f"space-{park}-{seed}")
    out = []
    for wid in range(n):
        w = _world(r, wid, park)
        alive = [b["id"] for b in w["balls"]]
        state = {"warping": set()}
        for _ in range(steps):
            recs = []
            for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4])):
                rec = _record(r, w, alive, park, state)
                if rec is None:
                    continue
                for one in (rec if isinstance(rec, list) else [rec]):
                    if one[0] == "remove":
                        alive.remove(one[1])
                    recs.append(one)
            w["steps"].append(recs)
        out.append(w)
    return out


# ============================================================================================ the scalar side

def _bits(x: float) -> str:
    return "nan" if x != x else struct.pack("<d", float(x)).hex()


class ScalarWorld:
    """One world on a master FullBallpark: the balls added in order, the records applied, the tick."""

    def __init__(self, w: dict, friction: float = 1.0e6, tick_ms: int = 1000):
        self.park = p = FullBallpark(friction, tick_ms, is_master=True)
        p.trace = Counter()
        for b in w["balls"]:
            p.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], is_free=b["free"], is_massive=b["massive"],
                       is_interactive=b["interactive"], x=b["pos"][0], y=b["pos"][1], z=b["pos"][2],
                       vx=b["vel"][0], vy=b["vel"][1], vz=b["vel"][2], agility=b["agility"],
                       speed_fraction=b["sf"])
        self.answers: list = []

    def apply(self, rec: tuple) -> None:
        p, kind = self.park, rec[0]
        a = rec[1:]
        if kind == "goto":
            p.goto_point(*a)
        elif kind == "dir":
            p.goto_direction(*a)
        elif kind == "stop":
            p.stop(*a)
        elif kind == "sf":
            p.set_speed_fraction(*a)
        elif kind == "vel":
            p.set_velocity(*a)
        elif kind == "pos":
            p.set_position(*a)
        elif kind == "radius":
            p.set_radius(*a)
        elif kind == "maxvel":
            p.set_max_velocity(*a)
        elif kind == "mass":
            p.set_mass(*a)
        elif kind == "agility":
            p.set_agility(*a)
        elif kind == "interactive":
            p.set_interactive(a[0], bool(a[1]))
        elif kind == "keepalive":
            p.add_keep_alive_ball(a[0])
        elif kind == "init":
            p.initialize_bubbles(a[0])
        elif kind == "copy":
            p.copy_bubbles()
        elif kind == "cloaked":
            p.set_cloaked(a[0], a[1])
        elif kind == "remove":
            p.remove_ball(a[0])
        elif kind == "sensor":
            p.add_proximity_sensor(a[0], a[1], a[2], 0, bool(a[3]))
        elif kind == "unsensor":
            p.remove_proximity_sensor(a[0])
        elif kind == "notify":
            p.set_notification_range(a[0], a[1])
        elif kind == "track":
            p.set_target_tracking(a[0], a[1], a[2])
        elif kind == "follow":
            p.follow_ball(a[0], a[1], a[2])
        elif kind == "orbit":
            p.orbit(a[0], a[1], a[2])
        elif kind == "warp":
            p.warp_to(a[0], a[1], a[2], a[3], a[4], a[5])
        elif kind == "warpin":
            p.entity_warp_in(a[0], a[1], a[2], a[3], a[4])
        elif kind == "vis":
            self.answers.append(("vis", p.check_visibility(a[0], a[1])))
        elif kind == "range_ball":
            self.answers.append(("range", p.ball_ids_in_range(a[0], float(a[1]), int(a[2]))))
        elif kind == "range_dist":
            res = p.ball_ids_in_range(a[0], float(a[1]), int(a[2]), include_dist=True)
            self.answers.append(("dist", None if res is None else [(_bits(d), i) for d, i in res]))
        elif kind == "range_point":
            self.answers.append(("range", p.ball_ids_in_range(float(a[0]), float(a[1]), float(a[2]), float(a[3]),
                                                              int(a[4]))))
        elif kind == "bubble_at":
            self.answers.append(("at", p.bubble_at(*a)))
        elif kind == "remote":
            self.answers.append(("remote", p.remote_followers()))
        else:
            raise ValueError(kind)

    def evolve(self) -> None:
        self.park.evolve()


def scalar_state(sw: ScalarWorld) -> dict:
    p = sw.park
    balls = {}
    for b in p.balls.values():
        tr = b.tracking_ptr
        balls[b.id] = {
            "bubble": b.bubble, "old_bubble": b.old_bubble,
            "filing": None if b.my_box is None else (b.my_box.level, tuple(b.my_box.key), tuple(b.mod)),
            "in_ia": sorted(k for k, v in p.bubble_interactives.items() if b.id in v),
            "in_ka": sorted(k for k, v in p.bubble_keep_alives.items() if b.id in v),
            "ka_ball": b.id in p.bubble_keep_alive_balls,
            "pos": tuple(_bits(x) for x in b.pos), "vel": tuple(_bits(x) for x in b.vel), "radius": _bits(b.radius),
            "mode": b.mode,
            "sensor": (_bits(b.sensor_range), _bits(b.sensor_period), _bits(b.sensor_elapsed),
                       bool(b.sensor_only_interactives), bool(b.sensor_active), bool(b.has_proximity),
                       tuple(b.sensor_balls.keys())),
            "tracking": (None if tr is None else tr.id, _bits(b.tracking_range), bool(b.within_tracking_range)),
            "notify": (_bits(b.notification_range), bool(b.within_range)),
        }
    world = {
        "dicts": {k: list(v.items()) for k, v in p.bubbles.items()},
        "interactives": {k: set(v) for k, v in p.bubble_interactives.items()},
        "keep_alives": {k: set(v) for k, v in p.bubble_keep_alives.items()},
        "bubble_id": p.bubble_id,
        "tops": {tuple(k): box.bubble for k, box in p.levels[0].items()},
        "proximity": tuple(p.proximity_balls.keys()),
        "live_boxes": sum(len(lv) for lv in p.levels),
    }
    return {"balls": balls, "world": world}


def scalar_events(sw: ScalarWorld) -> list:
    out = []
    for e in sw.park.events:
        if e[1] in EVENT_NAMES:
            out.append(tuple(e[1:]))
    sw.park.events = []
    return out


# =========================================================================================== the batched side

class BatchedWorlds:
    """The worlds on one batched park with the VecSpace layer over it."""

    def __init__(self, cs: list, device: str, park: str = "orient", friction: float = 1.0e6, tick_ms: int = 1000,
                 hits: bool = True, lib: dict | None = None, umap_cap: int | None = None, events: int = 256):
        from exact_ballpark.vec_modes import VecModesPark
        from exact_ballpark.vec_orient import VecOrientPark
        self.cs = cs
        self.dev = torch.device(device)
        self.kind = park
        N = max(len(w["balls"]) for w in cs)
        rows, index, flat = [], {}, []
        for wi, w in enumerate(cs):
            rr = []
            for j, b in enumerate(w["balls"]):
                index[(wi, b["id"])] = (len(flat), j)
                rr.append(len(flat))
                flat.append((wi, b))
            rows.append(rr + [-1] * (N - len(rr)))
        self.index, self.flat = index, flat
        R = len(flat)
        if park == "modes":
            self.park = p = VecModesPark(R, friction=friction, tick_interval_ms=tick_ms, device=self.dev)
            p.load([{"id": b["id"], "world": wi, "time": 0, "mass": b["mass"], "radius": b["radius"],
                     "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                     "speed_fraction": b["sf"], "free": b["free"], "massive": b["massive"],
                     "interactive": b["interactive"], "cloaked": 0, "bubble": -1, "moribund": False}
                    for wi, b in flat])
            self.view = VS.ModesView(p)
        else:
            self.park = p = VecOrientPark(R, friction=friction, tick_interval_ms=tick_ms, device=self.dev)
            p.load([{"mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"], "pos": b["pos"],
                     "vel": b["vel"], "agility": b["agility"], "speed_fraction": b["sf"]} for wi, b in flat])
            self.view = VS.OrientView(p, [b["id"] for wi, b in flat])
            self.view.own["interactive"] = torch.tensor([b["interactive"] for wi, b in flat], dtype=BOOL,
                                                        device=self.dev)
            self.view.own["massive"] = torch.tensor([b["massive"] for wi, b in flat], dtype=BOOL, device=self.dev)
        self.space = VS.VecSpace(self.view, rows, events=events, hits=hits, lib=lib, umap_cap=umap_cap)
        self.space.load()
        self.answers = [[] for _ in cs]
        self.W = len(cs)

    # ---- the park's fields, by name ----

    def _t(self, name: str) -> torch.Tensor:
        return self.view.tensors()[name]

    def _set(self, name: str, value: torch.Tensor) -> None:
        if self.kind == "modes":
            attr = {"interactive": "interactive", "cloaked": "cloaked", "massive": "massive", "moribund": "moribund"}
            setattr(self.park, attr[name], value)
        elif name == "moribund":
            self.view.moribund = value
        else:
            self.view.set_flag(name, value)

    def row(self, wi: int, bid) -> int:
        x = self.index.get((wi, bid))
        return -1 if x is None else x[0]

    def slot(self, wi: int, bid) -> int:
        x = self.index.get((wi, bid))
        return -1 if x is None else x[1]

    def alive(self, wi: int, bid) -> bool:
        s = self.slot(wi, bid)
        return s >= 0 and bool(self.space.st["in_park"][wi, s])

    # ---- one step ----

    def step(self, t: int) -> None:
        width = max((len(w["steps"][t]) for w in self.cs), default=0)
        for j in range(width):
            recs = [w["steps"][t][j] if j < len(w["steps"][t]) else None for w in self.cs]
            self._apply(recs)
        self._evolve()

    def _mask(self, recs, kinds) -> tuple:
        R = self.park.n
        m = torch.zeros(R, dtype=BOOL)
        args = {}
        for wi, rec in enumerate(recs):
            if rec is None or rec[0] not in kinds:
                continue
            if not self.alive(wi, rec[1]):
                continue
            r = self.row(wi, rec[1])
            m[r] = True
            args[r] = rec
        return m.to(self.dev), args

    def _vec(self, args: dict, k: int, width: int = 1) -> torch.Tensor:
        R = self.park.n
        out = torch.zeros(R, width, dtype=F64)
        for r, rec in args.items():
            for i in range(width):
                out[r, i] = float(rec[k + i])
        return out.to(self.dev) if width > 1 else out[:, 0].to(self.dev)

    def _apply(self, recs: list) -> None:
        p, sp = self.park, self.space
        ops = [[] for _ in recs]
        # the park's commands and setters (each world's j-th record), then the layer's part, in that order
        m, args = self._mask(recs, ("goto",))
        if args:
            p.goto_point(m, self._vec(args, 2, 3))
        m, args = self._mask(recs, ("dir",))
        if args:
            p.goto_direction(m, self._vec(args, 2, 3))
        m, args = self._mask(recs, ("stop",))
        if args:
            p.stop(m)
        m, args = self._mask(recs, ("sf",))
        if args:
            p.set_speed_fraction(m, self._vec(args, 2))
        m, args = self._mask(recs, ("vel",))
        if args:
            p.set_velocity(m, self._vec(args, 2, 3))
        m, args = self._mask(recs, ("pos",))
        if args:
            xyz = self._vec(args, 2, 3)
            p.pos = torch.where(m[:, None], xyz, p.pos).contiguous()
            p.old_pos = torch.where(m[:, None], xyz, p.old_pos).contiguous()
        m, args = self._mask(recs, ("radius",))
        if args:
            v = self._vec(args, 2)
            p.set_radius(m, v)
            acc = {r: rec for r, rec in args.items() if not (_float32(float(rec[2])) < 0.0)}
        else:
            acc = {}
        m2, args2 = self._mask(recs, ("maxvel",))
        if args2:
            p.set_max_velocity(m2, self._vec(args2, 2))
            acc.update({r: rec for r, rec in args2.items() if not (_float32(float(rec[2])) < 0.0)})
        m, args = self._mask(recs, ("mass",))
        if args:
            p.set_mass(m, self._vec(args, 2))
        m, args = self._mask(recs, ("agility",))
        if args:
            p.set_agility(m, self._vec(args, 2))
        if self.kind == "modes":
            for kind, fn in (("follow", p.follow), ("orbit", p.orbit)):
                m, args = self._mask(recs, (kind,))
                if args:
                    tgt = torch.full((p.n,), -1, dtype=I64)
                    for r, rec in args.items():
                        wi = self.flat[r][0]
                        tgt[r] = self.row(wi, rec[2]) if self.alive(wi, rec[2]) else -1
                    fn(m, tgt.to(self.dev), self._vec(args, 3))
            m, args = self._mask(recs, ("warp",))
            if args:
                p.warp_to(m, self._vec(args, 2, 3), self._vec(args, 5),
                          torch.tensor([int(args[r][6]) if r in args else 1 for r in range(p.n)],
                                       dtype=I64).to(self.dev))
            m, args = self._mask(recs, ("warpin",))
            if args:
                p.entity_warp_in(m, self._vec(args, 2, 3),
                                 torch.tensor([int(args[r][5]) if r in args else 1 for r in range(p.n)],
                                              dtype=I64).to(self.dev))
        m, args = self._mask(recs, ("cloaked",))
        if args:
            v = torch.zeros(p.n, dtype=I64)
            for r, rec in args.items():
                v[r] = int(rec[2]) & 0xFF
            self._set("cloaked", torch.where(m, v.to(self.dev), self._t("cloaked")).contiguous())
        # the layer's operations, each world's in its record's place
        for wi, rec in enumerate(recs):
            if rec is None:
                continue
            kind = rec[0]
            if kind in ("init", "copy", "remote", "range_point", "bubble_at"):
                if kind == "init":
                    ops[wi].append(("INIT_BUBBLES", self.slot(wi, rec[1]) if rec[1] != 0 else -1))
                elif kind == "copy":
                    ops[wi].append(("COPY_BUBBLES",))
                continue
            if not self.alive(wi, rec[1]):
                continue
            s = self.slot(wi, rec[1])
            r = self.row(wi, rec[1])
            if kind == "pos" or ((kind in ("radius", "maxvel")) and r in acc):
                ops[wi].append(("REFILE", s))
            elif kind == "interactive":
                ops[wi].append(("SET_INTERACTIVE", s, int(rec[2])))
            elif kind == "keepalive":
                ops[wi].append(("KEEP_ALIVE", s))
            elif kind == "remove":
                ops[wi].append(("REMOVE_BALL", s))
            elif kind == "sensor":
                ops[wi].append(("ADD_SENSOR", s, int(rec[4]), (rec[2], rec[3])))
            elif kind == "unsensor":
                ops[wi].append(("REMOVE_SENSOR", s))
            elif kind == "notify":
                ops[wi].append(("NOTIFY_RANGE", s, 0, (rec[2],)))
            elif kind == "track":
                ops[wi].append(("TRACK", s, int(rec[2]), (rec[3],)))
        if any(ops):
            removed = [(wi, rec[1]) for wi, rec in enumerate(recs) if rec is not None and rec[0] == "remove"
                       and self.alive(wi, rec[1])]
            if removed:                                     # the park's side of RemoveBall: Stop, followers released
                m = torch.zeros(p.n, dtype=BOOL)
                for wi, bid in removed:
                    m[self.row(wi, bid)] = True
                m = m.to(self.dev)
                if self.kind == "modes":
                    p.stop(m)
                    p._release_followers(m)
                else:
                    p.stop(m)
            sp.ops(ops)
        # the queries
        self._queries(recs)

    def _queries(self, recs: list) -> None:
        sp, W = self.space, self.W
        for kind in QUERY_KINDS:
            sel = [wi for wi, rec in enumerate(recs) if rec is not None and rec[0] == kind]
            if not sel:
                continue
            a = torch.full((W,), -1, dtype=I64)
            b = torch.zeros(W, dtype=I64)
            f = torch.zeros(W, 5, dtype=F64)
            for wi in sel:
                rec = recs[wi]
                if kind == "vis":
                    a[wi] = self.slot(wi, rec[1]) if self.alive(wi, rec[1]) else -1
                    b[wi] = self.slot(wi, rec[2]) if self.alive(wi, rec[2]) else -1
                elif kind in ("range_ball", "range_dist"):
                    a[wi] = self.slot(wi, rec[1]) if self.alive(wi, rec[1]) else -1
                    f[wi, 3] = float(rec[2])
                    b[wi] = int(rec[3])
                elif kind == "range_point":
                    f[wi, :4] = torch.tensor([float(x) for x in rec[1:5]], dtype=F64)
                    b[wi] = int(rec[5])
                elif kind == "bubble_at":
                    f[wi, :3] = torch.tensor([float(x) for x in rec[1:4]], dtype=F64)
            qk = {"vis": "VISIBILITY", "range_ball": "RANGE_BALL", "range_dist": "RANGE_BALL",
                  "range_point": "RANGE_POINT", "bubble_at": "BUBBLE_AT", "remote": "REMOTE_FOLLOWERS"}[kind]
            out, outf = sp.query(qk, a, b, f)
            out, outf = out.cpu(), outf.cpu()
            for wi in sel:
                ids = sp.slot_ids(wi)
                o = out[wi].tolist()
                if kind == "vis":
                    self.answers[wi].append(("vis", o[0]))
                elif kind == "bubble_at":
                    self.answers[wi].append(("at", None if o[0] == VS.NO_TOP else o[0]))
                elif kind == "remote":
                    self.answers[wi].append(("remote", [ids[j] for j in o[1:1 + o[0]]]))
                elif kind == "range_dist":
                    self.answers[wi].append(("dist", None if o[0] < 0 else
                                             [(_bits(outf[wi, k].item()), ids[o[1 + k]]) for k in range(o[0])]))
                else:
                    self.answers[wi].append(("range", None if o[0] < 0 else [ids[j] for j in o[1:1 + o[0]]]))

    def _evolve(self) -> None:
        p, sp = self.park, self.space
        dead = sp.dead_rows()
        skip = dead | self._t("moribund")
        if self.kind == "modes":
            warp_proper = (p.mode == WARP) & (p.effect_stamp >= 0) & p.free & ~p.moribund & ~skip
            VS.evolve_skipping(p, dead)
            dropped = warp_proper & (p.mode != WARP)
        else:
            VS.evolve_skipping(p, skip)
            dropped = None
        sp.after_evolve(dropped)


def batched_state(bw: BatchedWorlds, wi: int) -> dict:
    sp = bw.space
    st = {k: (v[wi].cpu() if v is not None else None) for k, v in sp.st.items()}
    t = {k: v.cpu() for k, v in bw.view.tensors().items()}
    ids = sp.slot_ids(wi)
    ia, ka = sp.interactives(wi), sp.keep_alives(wi)
    balls = {}
    for j, bid in ids.items():
        if not st["in_park"][j]:
            continue
        r = int(sp.rows[wi, j])
        lvl = int(st["lvl"][j])
        trk = int(st["trk"][j])
        balls[bid] = {
            "bubble": int(st["bubble"][j]), "old_bubble": int(st["old_bubble"][j]),
            "filing": None if lvl < 0 else (lvl, tuple(st["key"][j].tolist()), tuple(st["mod"][j].tolist())),
            "in_ia": sorted(k for k, v in ia.items() if bid in v), "in_ka": sorted(k for k, v in ka.items() if bid in v),
            "ka_ball": bool(st["ka_ball"][j]),
            "pos": tuple(_bits(x) for x in t["pos"][r].tolist()), "vel": tuple(_bits(x) for x in t["vel"][r].tolist()),
            "radius": _bits(float(t["radius"][r])), "mode": int(t["mode"][r]),
            "sensor": (_bits(float(st["srange"][j])), _bits(float(st["speriod"][j])),
                       _bits(float(st["selapsed"][j])), bool(st["sonly"][j]), bool(t["sensor"][r]),
                       bool(st["has_prox"][j]), tuple(sp.sensor_list(wi, j))),
            "tracking": (None if trk < 0 else ids[trk], _bits(float(st["trk_range"][j])), bool(st["trk_in"][j])),
            "notify": (_bits(float(st["ntf_range"][j])), bool(st["ntf_in"][j])),
        }
    world = {"dicts": sp.dicts(wi), "interactives": sp.interactives(wi), "keep_alives": sp.keep_alives(wi),
             "bubble_id": int(st["bubble_id"]), "tops": sp.tops(wi), "proximity": tuple(sp.proximity_order(wi)),
             "box_bound": int(st["box_sum"])}
    return {"balls": balls, "world": world}


# ================================================================================================== compare

def compare(cs: list, device: str, park: str = "orient", steps: int | None = None, hits: bool = True,
            lib: dict | None = None) -> dict:
    """Both sides on the worlds cs, step by step: the differences (step, world, what), the volumes compared, the
    flagged worlds (refused: excluded from the comparison) and the branch counts."""
    T = len(cs[0]["steps"]) if steps is None else int(steps)
    scal = [ScalarWorld(w) for w in cs]
    bw = BatchedWorlds(cs, device, park, hits=hits, lib=lib)
    diffs, compared = [], Counter()
    for sw in scal:
        scalar_events(sw)                                   # the load's events are not compared (none expected)
    for t in range(T):
        for wi, w in enumerate(cs):
            for rec in w["steps"][t]:
                scal[wi].apply(rec)
        for wi in range(len(cs)):
            scal[wi].evolve()
        bw.space.clear_events()
        bw.step(t)
        flags = bw.space.st["flags"].cpu().tolist()
        for wi in range(len(cs)):
            if flags[wi]:
                continue
            want, got = scalar_state(scal[wi]), batched_state(bw, wi)
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
            if got["world"]["box_bound"] + 8 < want["world"]["live_boxes"]:
                diffs.append((t, wi, "box_bound", want["world"]["live_boxes"], got["world"]["box_bound"]))
            ev_w = scalar_events(scal[wi])
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
    flagged = [wi for wi, f in enumerate(bw.space.st["flags"].cpu().tolist()) if f]
    ref_hits = Counter()
    for sw in scal:
        ref_hits.update(sw.park.trace)
    return {"diffs": diffs, "compared": dict(compared), "flagged": flagged, "hits": bw.space.hit_counts(),
            "ref_hits": dict(ref_hits), "steps": T}

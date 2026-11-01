# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the space layer (partition, bubbles, proximity sensors, visibility, range queries, collisions,
removal), generated from a seed; run by the reference (exact_ballpark.space) and by Destiny's own code compiled
verbatim (conformance/build_space.py), and compared bit for bit.

A scenario is a ballpark (friction, tick, start time, orbit law, server or client view, the rand() seed) and a list of
records applied at given steps: balls added with AddBall, commands, setters and queries. Step 0 is the setup before the
first tick; after the records of each later step the park evolves one tick. After every step everything the slice
decides is recorded: each ball's state (mode, flags, follow and tracking fields, bubbles, sensor, position, velocity,
old position and velocity, goto point, behaviour and collision accelerations, the properties, its partition boxes as
Destiny files them, its collision list, its followers and its sensor's list in the order Destiny walks it), the
partition (a checksum of every box and the count per level), the proximity balls in Destiny's order, the bubble
dictionaries, and every query's answer; the events Destiny posts beside them. At the end of a scenario every box is
dumped.
"""

from __future__ import annotations

import math
import random
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

from exact_ballpark.ref import f32
from exact_ballpark.space import LEVEL_WIDTH, DestinyUndefined, SpaceBallpark

from kit import harness as kit_harness  # noqa: E402  (the conformance kit's paths: conformance/kit.toml)
HARNESS = kit_harness("space")
MANIFEST = HARNESS.with_name("manifest.json")

(GOTO_OP, DIR_OP, STOP_OP, FOLLOW_OP, ORBIT_OP, WARP_OP, WARPIN_OP, POS_OP, VEL_OP, RADIUS_OP, MAXVEL_OP, MASS_OP,
 AGILITY_OP, SF_OP, FREE_OP, MASSIVE_OP, GLOBAL_OP, INTERACTIVE_OP, SENSOR_OP, UNSENSOR_OP, NOTIFY_OP, TRACK_OP,
 HARMONIC_OP, CLOAK_OP, INIT_BUBBLES_OP, COPY_BUBBLES_OP, CLEAN_FOLLOWERS_OP, KEEPALIVE_OP, DEADS_OP,
 REMOVE_OP) = range(1, 31)
VIS_Q, RANGE_BALL_Q, RANGE_POINT_Q, RANGE_DIST_Q, BUBBLE_AT_Q = 31, 32, 33, 34, 35
TOP = LEVEL_WIDTH[0]                                        # 7,864,320 m

S_INT = ("case", "step", "id", "mode", "effect_stamp", "massive", "sensor", "follow_id", "owner_id", "follow_ptr",
         "free", "in_free_balls", "global", "interactive", "cloaked", "moribund", "junk", "bubble", "old_bubble",
         "has_proximity", "within_range", "within_tracking", "tracking_ptr", "in_dead_bubble", "mod_x", "mod_y",
         "mod_z", "active_boxes", "box_level", "box_ix", "box_iy", "box_iz", "harmonic", "corporation", "alliance",
         "only_interactives")
S_DBL = ("px", "py", "pz", "vx", "vy", "vz", "opx", "opy", "opz", "ovx", "ovy", "ovz", "gx", "gy", "gz", "ax", "ay",
         "az", "cx", "cy", "cz", "mass", "radius", "max_vel", "agility", "speed_fraction", "follow_range",
         "last_collision", "time_factor", "notification_range", "tracking_range", "sensor_range", "sensor_period",
         "sensor_elapsed")
S_LISTS = ("boxes", "collisions", "followers", "sensor_balls")
S_FIELDS = S_INT + S_DBL + S_LISTS
ARITY = {"DoModeChange": 2, "OnExitWarp": 1, "OnActivatingWarp": 1, "OnDeactivatingWarp": 1, "DoProximity": 2,
         "DoRange": 1, "DoTargetTracking": 1, "DoFinalCleanup": 0}


# ---- generation ----

def _unit(r: random.Random) -> list:
    while True:
        u = [r.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        if n > 1e-3:
            return [c / n for c in u]


def _ids(r: random.Random, n: int) -> list:
    ids: set = set()
    while len(ids) < n:
        ids.add(r.choice([r.randrange(1, 20), r.randrange(1, 1000), r.randrange(1, 2 ** 40)]))
    out = sorted(ids)
    r.shuffle(out)                                          # insertion order differs from id order
    return out


def _edge(r: random.Random, level: int | None = None) -> float:
    """A coordinate on or near a box face at some level (the grid is centred on the origin)."""
    level = r.randrange(0, 8) if level is None else level
    w = LEVEL_WIDTH[level]
    return r.randint(-3, 3) * w + r.choice([0.0, r.uniform(-30.0, 30.0), r.uniform(-1.0, 1.0) * w * 0.02,
                                            r.choice([-1.0, 1.0]) * r.uniform(0.0, 2.0)])


def _ball(r: random.Random, bid: int, step: int = 0, **kw) -> tuple:
    """An AddBall record: a random ship; keywords override."""
    b = {"id": bid, "mass": 10 ** r.uniform(5.5, 9.5),
         "radius": r.choice([0.0, r.uniform(1.0, 50.0), r.uniform(50.0, 500.0), r.uniform(500.0, 5000.0)]),
         "max_vel": r.choice([r.uniform(50.0, 1500.0), r.uniform(50.0, 1500.0), r.uniform(1.0, 199.0)]),
         "free": True, "global": False, "massive": r.random() < 0.6, "interactive": r.random() < 0.5,
         "junk": False, "pos": [r.uniform(-2e4, 2e4) for _ in range(3)],
         "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-300.0, 300.0) for _ in range(3)]]),
         "agility": r.uniform(0.05, 2.0), "sf": r.choice([1.0, r.uniform(0.1, 1.0)])}
    b.update(kw)
    return ("B", step, b)


def _k(step: int, op: int, a: int = 0, b: int = 0, w: int = 0, x: float = 0.0, y: float = 0.0, z: float = 0.0,
       rr: float = 0.0) -> tuple:
    return ("K", step, op, a, b, w, float(x), float(y), float(z), float(rr))


def _test_ship(bid: int, step: int = 0, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass: float = 13000000.0,
               max_vel: float = 10.0) -> list:
    """helpers.create_space_ball as records: AddBall with the POPO defaults, then the setters in the helper's order."""
    return [("B", step, {"id": bid, "mass": 0.0, "radius": 0.0, "max_vel": 0.0, "free": False, "global": False,
                         "massive": False, "interactive": False, "junk": False, "pos": list(pos), "vel": list(vel),
                         "agility": 0.0, "sf": 0.0}),
            _k(step, FREE_OP, bid, w=1), _k(step, MAXVEL_OP, bid, rr=max_vel), _k(step, AGILITY_OP, bid, rr=0.9),
            _k(step, MASS_OP, bid, rr=mass), _k(step, RADIUS_OP, bid, rr=2.0), _k(step, MASSIVE_OP, bid, w=1),
            _k(step, SF_OP, bid, rr=0.95)]


def cases(seed: int) -> list[dict]:
    r = random.Random(seed)
    out: list[dict] = []

    def case(kind: str, records: list, steps: int, *, master: bool = True, tick: int = 1000,
             friction: float = 1.0e6, new_orbit: bool = False, start: int | None = None) -> None:
        out.append({"id": len(out) + 1, "kind": kind, "friction": friction, "tick": tick, "steps": steps,
                    "new_orbit": new_orbit, "master": master, "seed": r.randrange(1, 2 ** 31),
                    "start": r.randrange(0, 100000) if start is None else start, "records": records})

    def tick_choice() -> int:
        return r.choice([250, 500, 1000, 1000, 1000, 2000])

    # the partition: balls on and near box faces at every level, every radius scale, moved, resized, fixed, freed
    for _ in range(170):
        n = r.randint(1, 5)
        ids = _ids(r, n)
        recs = []
        for bid in ids:
            radius = r.choice([0.0, r.uniform(0.0, 50.0), r.uniform(50.0, 2000.0), r.uniform(2e3, 2e5),
                               r.uniform(2e5, 3.8e6), r.uniform(3.9e6, 6e6)])
            pos = [_edge(r) for _ in range(3)] if r.random() < 0.7 else [r.uniform(-3e7, 3e7) for _ in range(3)]
            recs.append(_ball(r, bid, radius=radius, pos=pos, free=r.random() < 0.8,
                              interactive=r.random() < 0.7, massive=r.random() < 0.3))
        steps = r.randint(2, 25)
        for _ in range(r.randint(0, 6)):
            k, a = r.randint(1, steps), r.choice(ids)
            what = r.choice(["pos", "pos", "radius", "maxvel", "free", "goto", "vel", "mass", "agility", "sf"])
            if what == "pos":
                recs.append(_k(k, POS_OP, a, x=_edge(r), y=_edge(r), z=_edge(r)))
            elif what == "radius":
                recs.append(_k(k, RADIUS_OP, a, rr=r.choice([0.0, r.uniform(0, 500), r.uniform(1e4, 4e6), -1.0])))
            elif what == "maxvel":
                recs.append(_k(k, MAXVEL_OP, a, rr=r.choice([0.0, r.uniform(1, 2000), r.uniform(1e4, 1e6)])))
            elif what == "free":
                recs.append(_k(k, FREE_OP, a, w=r.randint(0, 1)))
            elif what == "goto":
                u = _unit(r)
                recs.append(_k(k, GOTO_OP, a, x=u[0] * 1e6, y=u[1] * 1e6, z=u[2] * 1e6))
            elif what == "vel":
                recs.append(_k(k, VEL_OP, a, x=r.uniform(-500, 500), y=r.uniform(-500, 500), z=r.uniform(-500, 500)))
            elif what == "mass":
                recs.append(_k(k, MASS_OP, a, rr=r.choice([10 ** r.uniform(4, 10), -1.0, 0.0])))
            elif what == "agility":
                recs.append(_k(k, AGILITY_OP, a, rr=r.choice([r.uniform(0.01, 3.0), -1.0])))
            else:
                recs.append(_k(k, SF_OP, a, rr=r.uniform(-0.2, 1.2)))
        case("partition", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.75, tick=tick_choice())

    # the grid's edges: positions beyond the clamped keys (|x| > 3.3e13 m)
    for _ in range(15):
        ids = _ids(r, r.randint(1, 3))
        recs = [_ball(r, bid, pos=[r.choice([-1, 1]) * r.uniform(3.2e13, 3.6e13) for _ in range(3)],
                      interactive=True) for bid in ids]
        recs.append(_k(r.randint(1, 3), GOTO_OP, ids[0], x=0.0, y=0.0, z=0.0))
        case("grid_edge", recs, r.randint(2, 8), master=r.random() < 0.7)

    # bubbles: balls spread over neighbouring and distant top boxes; interactive or not, keep-alive balls, moves across
    # top boxes, InitializeBubbles (all, or one ball), CopyBubbles, and the queries that read the bubble dictionaries
    for _ in range(150):
        n = r.randint(2, 7)
        ids = _ids(r, n)
        recs = []
        cells = [(r.randint(-3, 3), r.randint(-3, 3), r.randint(-3, 3)) for _ in range(r.randint(1, 3))]
        for bid in ids:
            c = r.choice(cells)
            pos = [(c[i] + r.uniform(0.0, 1.0)) * TOP for i in range(3)]
            recs.append(_ball(r, bid, pos=pos, interactive=r.random() < 0.5, massive=r.random() < 0.5,
                              radius=r.choice([0.0, r.uniform(1, 500)]), free=r.random() < 0.85))
        steps = r.randint(3, 20)
        for _ in range(r.randint(1, 10)):
            k, a, b = r.randint(0, steps), r.choice(ids), r.choice(ids)
            what = r.choice(["pos", "pos", "init", "init_one", "copy", "interactive", "keepalive", "range", "range",
                             "range_point", "range_dist", "bubble_at", "visibility", "goto", "global", "remote"])
            if what == "pos":
                c = r.choice(cells + [(r.randint(-4, 4), r.randint(-4, 4), r.randint(-4, 4))])
                recs.append(_k(k, POS_OP, a, x=(c[0] + r.uniform(0, 1)) * TOP, y=(c[1] + r.uniform(0, 1)) * TOP,
                               z=(c[2] + r.uniform(0, 1)) * TOP))
            elif what == "init":
                recs.append(_k(k, INIT_BUBBLES_OP))
            elif what == "init_one":
                recs.append(_k(k, INIT_BUBBLES_OP, a))
            elif what == "copy":
                recs.append(_k(k, COPY_BUBBLES_OP))
            elif what == "interactive":
                recs.append(_k(k, INTERACTIVE_OP, a, w=r.randint(0, 1)))
            elif what == "keepalive":
                recs.append(_k(k, KEEPALIVE_OP, a))
            elif what == "range":
                recs.append(_k(k, RANGE_BALL_Q, a, w=r.randint(0, 1), rr=r.choice([0.0, r.uniform(0, 5e6),
                                                                                   r.uniform(0, 2e7)])))
            elif what == "range_point":
                c = r.choice(cells)
                recs.append(_k(k, RANGE_POINT_Q, w=r.randint(0, 1), x=(c[0] + 0.5) * TOP, y=(c[1] + 0.5) * TOP,
                               z=(c[2] + 0.5) * TOP, rr=r.uniform(0, 2e7)))
            elif what == "range_dist":
                recs.append(_k(k, RANGE_DIST_Q, a, w=r.randint(0, 1), rr=r.uniform(0, 2e7)))
            elif what == "bubble_at":
                c = r.choice(cells)
                recs.append(_k(k, BUBBLE_AT_Q, x=(c[0] + r.uniform(-0.5, 1.5)) * TOP, y=(c[1] + 0.5) * TOP,
                               z=(c[2] + 0.5) * TOP))
            elif what == "visibility":
                recs.append(_k(k, VIS_Q, a, b))
            elif what == "goto":
                u = _unit(r)
                recs.append(_k(k, GOTO_OP, a, x=u[0] * 1e8, y=u[1] * 1e8, z=u[2] * 1e8))
            elif what == "global":
                recs.append(_k(k, GLOBAL_OP, a, w=r.randint(0, 1)))
            else:
                recs.append(_k(k, CLEAN_FOLLOWERS_OP))
        case("bubbles", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.85, tick=tick_choice())

    # collisions: overlapping, coincident, approaching and crossing pairs and groups; fixed walls; light and heavy;
    # space junk; cloaked, non-massive and global neighbours; the test ships of Destiny's helpers
    for _ in range(260):
        n = r.randint(2, 5)
        ids = _ids(r, n)
        recs = []
        centre = [r.choice([0.0, _edge(r, 7), r.uniform(-1e4, 1e4)]) for _ in range(3)]
        flavour = r.choice(["overlap", "coincident", "approach", "approach", "fixed", "light", "mixed", "test_ships",
                            "field"])
        for j, bid in enumerate(ids):
            radius = r.choice([2.0, r.uniform(1.0, 60.0), r.uniform(60.0, 400.0)])
            if flavour == "coincident":
                pos = list(centre)
            elif flavour in ("overlap", "light"):
                pos = [centre[i] + r.uniform(-1.0, 1.0) * radius for i in range(3)]
            else:
                pos = [centre[i] + r.uniform(-1.0, 1.0) * radius * r.uniform(1.0, 6.0) for i in range(3)]
            mass = 10 ** r.uniform(5.0, 9.0)
            if flavour == "light" and j == 0:
                mass = 10 ** r.uniform(3.0, 5.0)
            free = not (flavour == "fixed" and j == 0) and r.random() < 0.92
            if flavour == "test_ships":
                recs += _test_ship(bid, pos=pos, mass=r.choice([13000000.0, 1.0, 1e9]),
                                   max_vel=r.choice([10.0, 100.0, 1000.0]))
                recs.append(_k(0, INTERACTIVE_OP, bid, w=1))
                continue
            recs.append(_ball(r, bid, pos=pos, radius=radius, mass=mass, free=free,
                              massive=r.random() < 0.85, interactive=r.random() < 0.6,
                              junk=r.random() < 0.08, agility=r.uniform(0.05, 2.0),
                              vel=r.choice([[0.0, 0.0, 0.0], [r.uniform(-50, 50) for _ in range(3)]]),
                              max_vel=r.uniform(5.0, 400.0)))
            if r.random() < 0.06:
                recs.append(_k(0, CLOAK_OP, bid, w=r.choice([1, 2, 3])))
            if r.random() < 0.05:
                recs.append(_k(0, GLOBAL_OP, bid, w=1))
            if r.random() < (0.5 if flavour == "field" else 0.05):    # a force field, or a ship's harmonics
                recs.append(_k(0, HARMONIC_OP, bid, b=r.choice([-2, -1, 5, 7]), w=int(r.random() < 0.6),
                               x=r.choice([-1, 3]), y=r.choice([-1, 4])))
        steps = r.randint(3, 30)
        for bid in ids:
            if flavour in ("approach", "mixed", "fixed", "test_ships", "field") and r.random() < 0.8:
                target = [centre[i] + r.uniform(-20.0, 20.0) for i in range(3)]
                recs.append(_k(r.choice([1, 1, 2]), GOTO_OP, bid, x=target[0], y=target[1], z=target[2]))
            elif flavour == "mixed" and r.random() < 0.4:
                recs.append(_k(1, FOLLOW_OP if r.random() < 0.5 else ORBIT_OP, bid, r.choice(ids), rr=r.uniform(0, 50)))
        case("collision", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.7, tick=tick_choice(),
             friction=r.choice([1.0e6] * 5 + [10 ** r.uniform(5.0, 7.0)]), new_orbit=r.random() < 0.5)

    # force fields (FIELD through SetBallHarmonic): ships whose harmonic, corporation or alliance matches, or the
    # master pass (-2), fly through; the others collide (COL-703's clause)
    for _ in range(60):
        n = r.randint(2, 4)
        ids = _ids(r, n)
        fid = ids[0]
        h, corp, alliance = r.choice([5, 9]), r.choice([-1, 3]), r.choice([-1, 4])
        recs = [_ball(r, fid, pos=[0.0, 0.0, 0.0], radius=r.uniform(100.0, 300.0), massive=True, interactive=True,
                      free=r.random() < 0.7),
                _k(0, HARMONIC_OP, fid, b=h, w=1, x=corp, y=alliance)]
        steps = r.randint(5, 25)
        for bid in ids[1:]:
            u = _unit(r)
            d = r.uniform(400.0, 900.0)
            recs.append(_ball(r, bid, pos=[u[i] * d for i in range(3)], radius=r.uniform(1.0, 50.0), massive=True,
                              interactive=True, max_vel=r.uniform(100.0, 400.0), mass=10 ** r.uniform(6.0, 8.0)))
            match = r.choice(["none", "pass", "harmonic", "corporation", "alliance"])
            hb = {"none": 1, "pass": -2, "harmonic": h, "corporation": 2, "alliance": 2}[match]
            cb = corp if match == "corporation" else 7
            ab = alliance if match == "alliance" else 8
            recs.append(_k(0, HARMONIC_OP, bid, b=hb, w=0, x=cb, y=ab))
            recs.append(_k(1, GOTO_OP, bid, x=-u[0] * d, y=-u[1] * d, z=-u[2] * d))
        case("field", recs, steps, master=r.random() < 0.8)

    # proximity sensors, notification ranges and target tracking: balls crossing in and out, periods and shuffles,
    # only-interactive sensors, fixed and cloaked balls, sensors removed, dead bubbles with balls in the sensor
    for _ in range(220):
        n = r.randint(2, 6)
        ids = _ids(r, n)
        recs = []
        for bid in ids:
            pos = [r.uniform(-3000.0, 3000.0) for _ in range(3)]
            recs.append(_ball(r, bid, pos=pos, radius=r.choice([0.0, r.uniform(1, 100)]), free=r.random() < 0.85,
                              interactive=r.random() < 0.5, massive=r.random() < 0.3,
                              max_vel=r.uniform(50.0, 800.0)))
        steps = r.randint(3, 30)
        s = r.choice(ids)
        recs.append(_k(r.choice([0, 0, 1, 2]), SENSOR_OP, s, b=int(r.random() < 0.3), w=int(r.random() < 0.25),
                       x=r.choice([0.0, 1.0, 2.0, 2.0, 3.5]), rr=r.choice([r.uniform(100, 3000), r.uniform(0, 100),
                                                                          -r.uniform(0, 50)])))
        if r.random() < 0.4:
            other = r.choice(ids)
            recs.append(_k(r.choice([0, 1]), SENSOR_OP, other, b=int(r.random() < 0.5), x=r.choice([0.0, 2.0]),
                           rr=r.uniform(10, 2000)))
        for _ in range(r.randint(1, 8)):
            k, a, b = r.randint(0, steps), r.choice(ids), r.choice(ids)
            what = r.choice(["goto", "goto", "goto", "pos", "notify", "track", "track", "unsensor", "cloak", "free",
                             "interactive", "follow", "orbit", "stop", "massive"])
            if what == "goto":
                recs.append(_k(k, GOTO_OP, a, x=r.uniform(-4000, 4000), y=r.uniform(-4000, 4000),
                               z=r.uniform(-4000, 4000)))
            elif what == "pos":
                recs.append(_k(k, POS_OP, a, x=r.uniform(-4000, 4000), y=r.uniform(-4000, 4000),
                               z=r.uniform(-4000, 4000)))
            elif what == "notify":
                recs.append(_k(k, NOTIFY_OP, a, rr=r.choice([0.0, r.uniform(0, 3000), -1.0])))
            elif what == "track":
                recs.append(_k(k, TRACK_OP, a, r.choice([b, b, 0]), rr=r.choice([0.0, r.uniform(0, 3000), -1.0])))
            elif what == "unsensor":
                recs.append(_k(k, UNSENSOR_OP, a))
            elif what == "cloak":
                recs.append(_k(k, CLOAK_OP, a, w=r.choice([0, 1])))
            elif what == "free":
                recs.append(_k(k, FREE_OP, a, w=r.randint(0, 1)))
            elif what == "interactive":
                recs.append(_k(k, INTERACTIVE_OP, a, w=r.randint(0, 1)))
            elif what in ("follow", "orbit"):
                recs.append(_k(k, FOLLOW_OP if what == "follow" else ORBIT_OP, a, b, rr=r.uniform(0, 500)))
            elif what == "stop":
                recs.append(_k(k, STOP_OP, a))
            else:
                recs.append(_k(k, MASSIVE_OP, a, w=r.randint(0, 1)))
        case("sensor", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.8, tick=tick_choice())

    # visibility: occluders on, beside and behind the segment; massive or not, cloaked or not; other bubbles
    for _ in range(120):
        n = r.randint(2, 7)
        ids = _ids(r, n)
        src = [r.uniform(-1e4, 1e4) for _ in range(3)]
        u = _unit(r)
        length = r.uniform(10.0, 5e4)
        dst = [src[i] + u[i] * length for i in range(3)]
        recs = [_ball(r, ids[0], pos=src, interactive=True), _ball(r, ids[1], pos=dst, interactive=True)]
        for bid in ids[2:]:
            t = r.choice([r.uniform(-0.2, 1.2), r.uniform(0.0, 1.0), r.uniform(0.1, 0.9)])
            radius = r.uniform(0.0, 400.0)
            off = [c * r.choice([r.uniform(0.0, 300.0), r.uniform(0.0, 0.5) * radius]) for c in _unit(r)]
            pos = [src[i] + u[i] * length * t + off[i] for i in range(3)]
            recs.append(_ball(r, bid, pos=pos, radius=radius, massive=r.random() < 0.75, free=r.random() < 0.7))
            if r.random() < 0.3:
                recs.append(_k(0, CLOAK_OP, bid, w=r.choice([1, 2])))
        if r.random() < 0.15:                                # the destination in another bubble
            recs.append(_k(0, POS_OP, ids[1], x=src[0] + 5 * TOP, y=src[1], z=src[2]))
        steps = r.randint(1, 6)
        for k in range(steps + 1):
            recs.append(_k(k, VIS_Q, ids[0], ids[1]))
            if r.random() < 0.5:
                recs.append(_k(k, VIS_Q, ids[1], ids[0]))
            if r.random() < 0.3:
                recs.append(_k(k, VIS_Q, r.choice(ids), r.choice(ids + [999999])))
        if r.random() < 0.5:
            recs.append(_k(1, GOTO_OP, ids[-1], x=r.uniform(-1e4, 1e4), y=r.uniform(-1e4, 1e4), z=r.uniform(-1e4, 1e4)))
        case("visibility", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.8)

    # following across bubbles: the target moves to another bubble; the pre-tick clean-up stops the followers
    for _ in range(90):
        n = r.randint(2, 5)
        ids = _ids(r, n)
        target = ids[0]
        recs = [_ball(r, bid, interactive=True, pos=[r.uniform(-5e3, 5e3) for _ in range(3)]) for bid in ids]
        for bid in ids[1:]:
            recs.append(_k(1, r.choice([FOLLOW_OP, ORBIT_OP]), bid, target, rr=r.uniform(0, 1000)))
        steps = r.randint(3, 12)
        k = r.randint(1, steps)
        if r.random() < 0.7:
            recs.append(_k(k, POS_OP, target, x=r.choice([3, -4, 10]) * TOP, y=0.0, z=0.0))
        else:
            recs.append(_k(k, FREE_OP, r.choice(ids[1:]), w=0))
        for j in range(k, steps + 1):
            if r.random() < 0.7:
                recs.append(_k(j, CLEAN_FOLLOWERS_OP))
        case("remote_followers", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.85)

    # sensor events: notification ranges in GOTO, FOLLOW and ORBIT, target tracking, balls leaving the candidates
    # (moved away, fixed, no longer interactive), and a sensor whose bubble dies while it holds balls (TIME-714)
    for _ in range(170):
        n = r.randint(2, 5)
        ids = _ids(r, n)
        s_id, other = ids[0], ids[1]
        recs = [_ball(r, bid, pos=[r.uniform(-1500.0, 1500.0) for _ in range(3)], radius=r.uniform(0.0, 60.0),
                      interactive=True, max_vel=r.uniform(100.0, 600.0), free=r.random() < 0.9) for bid in ids]
        steps = r.randint(4, 25)
        flavour = r.choice(["notify_goto", "notify_follow", "track", "vanish", "dead", "dead", "only_interactives"])
        if flavour == "notify_goto":
            target = [r.uniform(-1500.0, 1500.0) for _ in range(3)]
            recs.append(_k(r.randint(0, 2), GOTO_OP, s_id, x=target[0], y=target[1], z=target[2]))
            recs.append(_k(r.randint(0, 3), NOTIFY_OP, s_id, rr=r.choice([r.uniform(50, 800), r.uniform(800, 3000)])))
            recs.append(_k(r.randint(4, steps), GOTO_OP, s_id, x=-target[0], y=-target[1], z=-target[2]))
        elif flavour == "notify_follow":
            recs.append(_k(1, r.choice([FOLLOW_OP, ORBIT_OP]), s_id, other, rr=r.uniform(0.0, 600.0)))
            recs.append(_k(r.randint(0, 3), NOTIFY_OP, s_id, rr=r.uniform(10, 1500)))
            if r.random() < 0.5:
                recs.append(_k(r.randint(2, steps), GOTO_OP, other, x=r.uniform(-9e3, 9e3), y=r.uniform(-9e3, 9e3),
                               z=r.uniform(-9e3, 9e3)))
        elif flavour == "track":
            recs.append(_k(r.randint(0, 2), TRACK_OP, s_id, other, rr=r.uniform(50.0, 2500.0)))
            recs.append(_k(r.randint(1, steps), GOTO_OP, other, x=r.uniform(-6e3, 6e3), y=r.uniform(-6e3, 6e3),
                           z=r.uniform(-6e3, 6e3)))
            if r.random() < 0.3:
                recs.append(_k(r.randint(1, steps), TRACK_OP, s_id, other, rr=0.0))
        else:
            recs.append(_k(0, SENSOR_OP, s_id, b=int(r.random() < 0.3), w=int(flavour == "only_interactives"),
                           x=r.choice([0.0, 1.0, 2.0]), rr=r.uniform(1000.0, 4000.0)))
            k = r.randint(2, steps)
            for bid in ids[1:]:
                what = r.choice(["far", "fixed", "uninteractive", "stay"])
                if flavour == "dead":
                    recs.append(_k(k, INTERACTIVE_OP, bid, w=0))
                elif what == "far":
                    recs.append(_k(k, POS_OP, bid, x=r.uniform(2e4, 5e4), y=0.0, z=0.0))
                elif what == "fixed":
                    recs.append(_k(k, FREE_OP, bid, w=0))
                elif what == "uninteractive":
                    recs.append(_k(k, INTERACTIVE_OP, bid, w=0))
            if flavour == "dead":
                recs.append(_k(k, INTERACTIVE_OP, s_id, w=0))
                if r.random() < 0.3:
                    recs.append(_k(k + 1, INTERACTIVE_OP, s_id, w=1))
        case("sensor_events", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.85, tick=tick_choice())

    # thresholds: a ball exactly at a sensor's reach as Destiny sums it in float, at the double sum, or one step
    # beyond either; a range query exactly at range + radius (SEN-702, SEN-710)
    for _ in range(60):
        s_id, o_id = _ids(r, 2)
        rng = f32(r.choice([10.0, r.uniform(1.0, 2000.0)]))
        rs, ro = f32(r.uniform(0.0, 60.0)), f32(r.uniform(0.0, 60.0))
        ck = f32(f32(rng + ro) + rs)
        d = r.choice([ck, math.nextafter(ck, math.inf), math.nextafter(ck, 0.0), rng + ro + rs,
                      math.nextafter(rng + ro + rs, math.inf)])
        recs = [_ball(r, s_id, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=rs, interactive=True),
                _ball(r, o_id, pos=[d, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=ro, interactive=True),
                _k(0, SENSOR_OP, s_id, x=0.0, rr=rng)]
        q = r.choice([rng, rng + 1e-9, math.nextafter(d - ro, 0.0), d - ro])
        recs.append(_k(1, RANGE_BALL_Q, s_id, rr=q))
        recs.append(_k(1, RANGE_DIST_Q, s_id, rr=q))
        recs.append(_k(1, RANGE_POINT_Q, x=0.0, y=0.0, z=0.0, rr=q))
        case("thresholds", recs, 2, master=r.random() < 0.7)

    # bubble edges: three top boxes in a row, the outer two with their own bubbles; balls moved into the middle one
    # take the first neighbour's bubble (NearbyBubbles scans from -1), so boxes on the middle/right face hold balls of
    # two bubbles; collisions, sensors, visibility, follows and range queries run across the face; global balls
    for _ in range(110):
        n = r.randint(3, 7)
        ids = _ids(r, n)
        cx, cy, cz = r.randint(-2, 2), r.randint(-2, 2), r.randint(-2, 2)
        face = (cx + 1) * TOP                               # between the middle box (cx) and the right one (cx + 1)
        left, right = ids[:max(2, n // 2)], ids[max(2, n // 2):]
        recs = []
        for bid in left:                                    # the left box, cx - 1
            recs.append(_ball(r, bid, pos=[(cx - 1 + r.uniform(0.2, 0.8)) * TOP, (cy + 0.5) * TOP + r.uniform(-50, 50),
                                           (cz + 0.5) * TOP + r.uniform(-50, 50)],
                              interactive=True, massive=r.random() < 0.85, radius=r.uniform(1.0, 80.0),
                              max_vel=r.uniform(20.0, 300.0)))
        for bid in right:                                   # the right box, beside the face
            recs.append(_ball(r, bid, pos=[face + r.uniform(1.0, 120.0), (cy + 0.5) * TOP + r.uniform(-40, 40),
                                           (cz + 0.5) * TOP + r.uniform(-40, 40)],
                              interactive=r.random() < 0.8, massive=r.random() < 0.9, radius=r.uniform(10.0, 150.0),
                              max_vel=r.uniform(20.0, 300.0)))
            if r.random() < 0.25:
                recs.append(_k(0, GLOBAL_OP, bid, w=1))
        movers = left[1:]                                   # the first stays: the left box keeps its bubble
        for bid in movers:
            recs.append(_k(r.choice([0, 0, 1]), POS_OP, bid, x=face - r.uniform(1.0, 120.0),
                           y=(cy + 0.5) * TOP + r.uniform(-40, 40), z=(cz + 0.5) * TOP + r.uniform(-40, 40)))
            if r.random() < 0.25:
                recs.append(_k(0, GLOBAL_OP, bid, w=1))
        steps = r.randint(3, 15)
        crossing = movers + right
        for bid in crossing:
            if r.random() < 0.5:
                recs.append(_k(r.randint(1, 3), GOTO_OP, bid, x=face, y=(cy + 0.5) * TOP, z=(cz + 0.5) * TOP))
        if r.random() < 0.6:
            recs.append(_k(r.randint(0, 2), SENSOR_OP, r.choice(crossing), x=r.choice([0.0, 2.0]),
                           rr=r.uniform(50.0, 500.0)))
        if r.random() < 0.4:
            recs.append(_k(r.randint(1, 3), FOLLOW_OP, r.choice(movers or right), r.choice(right or movers),
                           rr=r.uniform(0, 100)))
        for _ in range(r.randint(0, 3)):
            recs.append(_k(r.randint(0, steps), r.choice([INIT_BUBBLES_OP, COPY_BUBBLES_OP, CLEAN_FOLLOWERS_OP])))
        for _ in range(r.randint(0, 3)):
            recs.append(_k(r.randint(0, steps), RANGE_BALL_Q, r.choice(ids), w=r.randint(0, 1), rr=r.uniform(0, 500)))
            recs.append(_k(r.randint(0, steps), VIS_Q, r.choice(crossing), r.choice(crossing)))
        case("bubble_edges", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.9)

    # removal: at once, or moribund with a delay and brought out before a later tick (one ball due at a time);
    # followers released, sensors and range queries skipping the moribund, bubbles and partition left (TIME-713)
    for _ in range(80):
        n = r.randint(2, 6)
        ids = _ids(r, n)
        recs = [_ball(r, bid, interactive=r.random() < 0.7, massive=r.random() < 0.7,
                      pos=[r.uniform(-2e3, 2e3) for _ in range(3)], radius=r.uniform(0.0, 60.0)) for bid in ids]
        steps = r.randint(4, 20)
        victims = r.sample(ids, r.randint(1, max(1, n - 1)))
        for bid in ids:
            if bid not in victims and r.random() < 0.6:
                recs.append(_k(1, r.choice([FOLLOW_OP, ORBIT_OP]), bid, r.choice(victims), rr=r.uniform(0, 500)))
        if r.random() < 0.6:
            recs.append(_k(0, SENSOR_OP, r.choice(ids), x=r.choice([0.0, 1.0]), rr=r.uniform(500.0, 5000.0)))
        if r.random() < 0.4:
            recs.append(_k(0, TRACK_OP, r.choice(ids), r.choice(victims), rr=r.uniform(100.0, 3000.0)))
        due = 3 * r.randint(0, 3)
        for v in victims:
            k = r.randint(1, steps)
            delay = r.choice([0, 0, due + 3 * victims.index(v) + 1])
            recs.append(_k(k, REMOVE_OP, v, w=delay))
        for k in range(steps + 1):
            recs.append(_k(k, DEADS_OP))
            if r.random() < 0.3:
                recs.append(_k(k, RANGE_BALL_Q, r.choice(ids), w=1, rr=r.uniform(0, 5e3)))
            if r.random() < 0.2:
                recs.append(_k(k, VIS_Q, r.choice(ids), r.choice(ids)))
        case("removal", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.8, tick=tick_choice())

    # balls added again under an existing id (AddBall's other path): followers, flags, moribund state reset
    for _ in range(40):
        n = r.randint(2, 4)
        ids = _ids(r, n)
        recs = [_ball(r, bid, interactive=True, pos=[r.uniform(-3e3, 3e3) for _ in range(3)]) for bid in ids]
        for bid in ids[1:]:
            if r.random() < 0.6:
                recs.append(_k(1, r.choice([FOLLOW_OP, ORBIT_OP]), bid, ids[0], rr=r.uniform(0, 500)))
        steps = r.randint(3, 10)
        for _ in range(r.randint(1, 3)):
            again = r.choice(ids)
            recs.append(_ball(r, again, step=r.randint(1, steps), interactive=r.random() < 0.7,
                              free=r.random() < 0.8, pos=[r.uniform(-3e3, 3e3) for _ in range(3)]))
        case("readd", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.8)

    # keep-alive balls: a keep-alive ball that changes bubble carries its bubble's life with it (AddToBubble)
    for _ in range(50):
        n = r.randint(2, 5)
        ids = _ids(r, n)
        recs = [_ball(r, bid, pos=[(r.randint(-2, 2) + r.uniform(0, 1)) * TOP for _ in range(3)],
                      interactive=r.random() < 0.3) for bid in ids]
        keep = ids[:r.randint(1, len(ids))]
        for bid in keep:
            recs.append(_k(0, KEEPALIVE_OP, bid))
        steps = r.randint(3, 12)
        for _ in range(r.randint(2, 6)):
            k, a = r.randint(0, steps), r.choice(ids)
            what = r.choice(["pos", "pos", "pos", "copy", "init", "interactive", "goto"])
            if what == "pos":
                recs.append(_k(k, POS_OP, a, x=(r.randint(-3, 3) + r.uniform(0, 1)) * TOP,
                               y=(r.randint(-3, 3) + r.uniform(0, 1)) * TOP, z=(r.randint(-3, 3) + r.uniform(0, 1)) * TOP))
            elif what == "copy":
                recs.append(_k(k, COPY_BUBBLES_OP))
            elif what == "init":
                recs.append(_k(k, INIT_BUBBLES_OP))
            elif what == "interactive":
                recs.append(_k(k, INTERACTIVE_OP, a, w=r.randint(0, 1)))
            else:
                u = _unit(r)
                recs.append(_k(k, GOTO_OP, a, x=u[0] * 1e7, y=u[1] * 1e7, z=u[2] * 1e7))
        case("keepalive", sorted(recs, key=lambda c: c[1]), steps, master=True)

    # long warps through many top boxes: bubbles change in flight, followers are released, sensors switch off and on,
    # the drop-out re-files the ball; bystanders, EntityWarpIn
    for _ in range(80):
        n = r.randint(1, 4)
        ids = _ids(r, n)
        wid = ids[0]
        u = _unit(r)
        slow = r.random() < 0.3                              # a light, slow ship: at rest within the drop-out tick
        recs = [_ball(r, wid, sf=1.0, mass=10 ** (r.uniform(5.0, 6.0) if slow else r.uniform(5.5, 8.0)),
                      interactive=r.random() < 0.7, massive=r.random() < 0.8,
                      max_vel=r.uniform(0.1, 0.8) if slow else r.uniform(100.0, 1500.0),
                      agility=r.uniform(0.5, 1.5) if slow else r.uniform(0.05, 2.0))]
        recs[0][2]["vel"] = [u[i] * recs[0][2]["max_vel"] * r.uniform(0.8, 1.0) for i in range(3)]
        for bid in ids[1:]:
            recs.append(_ball(r, bid, interactive=r.random() < 0.6))
            if r.random() < 0.6:
                recs.append(_k(1, r.choice([FOLLOW_OP, ORBIT_OP]), bid, wid, rr=r.uniform(0, 3000)))
        if r.random() < 0.5:
            recs.append(_k(0, SENSOR_OP, wid, x=r.choice([0.0, 2.0]), rr=r.uniform(100, 5000)))
        dist = 10 ** (r.uniform(6.0, 8.0) if slow else r.uniform(7.0, 10.5))
        w = r.randint(6000, 20000) if slow else r.randint(1500, 20000)
        if r.random() < 0.85:
            recs.append(_k(r.choice([1, 2]), WARP_OP, wid, w=w, x=u[0] * dist, y=u[1] * dist, z=u[2] * dist,
                           rr=r.choice([0.0, 20000.0])))
        else:
            recs.append(_k(1, WARPIN_OP, wid, w=w, x=u[0] * dist, y=u[1] * dist, z=u[2] * dist))
        steps = r.randint(10, 70)
        for _ in range(r.randint(0, 4)):
            recs.append(_k(r.randint(1, steps), r.choice([CLEAN_FOLLOWERS_OP, COPY_BUBBLES_OP])))
        case("warp", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.8, tick=tick_choice())

    # the whole tick: slice 1's modes (follow, orbit, warp) with collisions, sensors and bubbles changing during warps
    for _ in range(160):
        n = r.randint(3, 7)
        ids = _ids(r, n)
        recs = [_ball(r, bid, pos=[r.uniform(-2e4, 2e4) for _ in range(3)], interactive=r.random() < 0.6,
                      massive=r.random() < 0.7) for bid in ids]
        if r.random() < 0.3:                                 # an inert mode (FIELD through SetBallHarmonic)
            recs.append(_k(0, HARMONIC_OP, r.choice(ids), b=r.choice([-1, 9]), w=1, x=-1, y=-1))
        steps = r.randint(10, 60)
        if r.random() < 0.6:
            recs.append(_k(r.randint(0, 3), SENSOR_OP, r.choice(ids), b=int(r.random() < 0.3),
                           x=r.choice([0.0, 2.0]), rr=r.uniform(100, 10000)))
        for _ in range(r.randint(2, 9)):
            k = r.randint(1, steps)
            a, b = r.choice(ids), r.choice(ids)
            op = r.choice([GOTO_OP, DIR_OP, STOP_OP, FOLLOW_OP, FOLLOW_OP, ORBIT_OP, ORBIT_OP, WARP_OP, WARP_OP,
                           CLEAN_FOLLOWERS_OP, VIS_Q, RANGE_BALL_Q])
            if op in (GOTO_OP, DIR_OP):
                recs.append(_k(k, op, a, x=r.uniform(-3e4, 3e4), y=r.uniform(-3e4, 3e4), z=r.uniform(-3e4, 3e4)))
            elif op == WARP_OP:
                v = _unit(r)
                d = r.choice([5e4, 10 ** r.uniform(6.5, 9.5)])
                recs.append(_k(k, op, a, w=r.randint(2000, 20000), x=v[0] * d, y=v[1] * d, z=v[2] * d,
                               rr=r.choice([0.0, 20000.0])))
            elif op in (STOP_OP, CLEAN_FOLLOWERS_OP):
                recs.append(_k(k, op, a))
            elif op == VIS_Q:
                recs.append(_k(k, op, a, b))
            elif op == RANGE_BALL_Q:
                recs.append(_k(k, op, a, w=r.randint(0, 1), rr=r.uniform(0, 1e5)))
            else:
                recs.append(_k(k, op, a, b, rr=r.uniform(-100, 3000)))
        case("mixed", sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.8, tick=tick_choice(),
             new_orbit=r.random() < 0.5, friction=r.choice([1.0e6] * 6 + [10 ** r.uniform(5.0, 7.0)]))
    return out


def iterative_cases(seed: int) -> list[dict]:
    """Scenarios with the iterative collision method switched on (COL-714 to COL-718): overlapping, coincident and
    approaching balls, fixed balls, chains and dense groups (several passes, sub-steps, invalidated impacts), light and
    heavy, slow and fast; with junk, cloaked, non-massive and other-bubble neighbours, warps, removal and dead
    bubbles; Destiny's test ships among them."""
    r = random.Random(1000 + seed)
    out: list[dict] = []

    def case(kind: str, records: list, steps: int, *, master: bool = True, tick: int = 1000,
             friction: float = 1.0e6) -> None:
        out.append({"id": len(out) + 1, "kind": kind, "friction": friction, "tick": tick, "steps": steps,
                    "new_orbit": r.random() < 0.3, "iterative": True, "master": master,
                    "seed": r.randrange(1, 2 ** 31), "start": r.randrange(0, 100000), "records": records})

    for _ in range(420):
        n = r.randint(2, 8)
        ids = _ids(r, n)
        flavour = r.choice(["overlap", "coincident", "approach", "approach", "chain", "dense", "fixed", "light",
                            "mixed", "test_ships", "fast", "limit"])
        centre = [r.choice([0.0, r.uniform(-1e4, 1e4)]) for _ in range(3)]
        if flavour == "limit":                              # a fast heavy ball into slow ones: |dv| > 5 maxVel
            u = _unit(r)
            recs = [_ball(r, ids[0], pos=[centre[i] - 400.0 * u[i] for i in range(3)], radius=r.uniform(5.0, 40.0),
                          mass=10 ** r.uniform(8.0, 9.5), massive=True, interactive=True,
                          vel=[u[i] * r.uniform(100.0, 600.0) for i in range(3)], max_vel=r.uniform(100.0, 600.0))]
            for bid in ids[1:]:
                recs.append(_ball(r, bid, pos=[centre[i] + r.uniform(-30.0, 30.0) for i in range(3)],
                                  radius=r.uniform(5.0, 40.0), mass=10 ** r.uniform(4.0, 7.0), massive=True,
                                  interactive=True, vel=[0.0, 0.0, 0.0], max_vel=r.uniform(0.5, 20.0)))
            case("iter_limit", recs, r.randint(2, 8), master=r.random() < 0.75)
            continue
        recs = []
        for j, bid in enumerate(ids):
            radius = r.choice([2.0, r.uniform(1.0, 50.0), r.uniform(50.0, 300.0)])
            if flavour == "coincident":
                pos = list(centre)
            elif flavour in ("overlap", "dense", "light"):
                pos = [centre[i] + r.uniform(-1.0, 1.0) * radius for i in range(3)]
            elif flavour == "chain":                         # a row along x, touching or nearly
                pos = [centre[0] + j * 2.0 * radius * r.uniform(0.95, 1.3), centre[1], centre[2]]
            else:
                pos = [centre[i] + r.uniform(-1.0, 1.0) * radius * r.uniform(1.0, 8.0) for i in range(3)]
            mass = 10 ** r.uniform(5.0, 9.0)
            if flavour == "light" and j == 0:
                mass = 10 ** r.uniform(2.0, 5.0)
            if flavour == "test_ships":
                recs += _test_ship(bid, pos=pos, mass=r.choice([13000000.0, 1e6, 1e9]),
                                   max_vel=r.choice([10.0, 20.0, 200.0]))
                recs.append(_k(0, INTERACTIVE_OP, bid, w=1))
                continue
            free = not (flavour == "fixed" and j == 0) and r.random() < 0.9
            vel = r.choice([[0.0, 0.0, 0.0], [r.uniform(-50, 50) for _ in range(3)]])
            if flavour == "fast":
                vel = [r.uniform(-2000.0, 2000.0) for _ in range(3)]
            recs.append(_ball(r, bid, pos=pos, radius=radius, mass=mass, free=free, massive=r.random() < 0.88,
                              interactive=r.random() < 0.7, junk=r.random() < 0.06, vel=vel,
                              agility=r.choice([r.uniform(0.05, 2.0), r.uniform(100.0, 1000.0)]),
                              max_vel=r.choice([r.uniform(5.0, 400.0), r.uniform(1000.0, 3000.0)])))
            if r.random() < 0.05:
                recs.append(_k(0, CLOAK_OP, bid, w=1))
            if r.random() < 0.05:
                recs.append(_k(0, GLOBAL_OP, bid, w=1))
        steps = r.randint(2, 25)
        if flavour == "chain":                               # the first pushes the row
            recs.append(_k(1, GOTO_OP, ids[0], x=centre[0] + 1e5, y=centre[1], z=centre[2]))
        for bid in ids:
            if flavour in ("approach", "mixed", "fixed", "test_ships", "fast") and r.random() < 0.75:
                target = [centre[i] + r.uniform(-20.0, 20.0) for i in range(3)]
                recs.append(_k(r.choice([1, 1, 2]), GOTO_OP, bid, x=target[0], y=target[1], z=target[2]))
        if flavour == "mixed":
            for _ in range(r.randint(1, 4)):
                k, a, b = r.randint(1, steps), r.choice(ids), r.choice(ids)
                what = r.choice(["warp", "remove", "follow", "free", "massive", "sensor", "interactive", "pos"])
                if what == "warp":
                    v = _unit(r)
                    recs.append(_k(k, WARP_OP, a, w=r.randint(2000, 20000), x=v[0] * 1e8, y=v[1] * 1e8,
                                   z=v[2] * 1e8))
                elif what == "remove":
                    recs.append(_k(k, REMOVE_OP, a, w=r.choice([0, 0, 3])))
                elif what == "follow":
                    recs.append(_k(k, r.choice([FOLLOW_OP, ORBIT_OP]), a, b, rr=r.uniform(0, 100)))
                elif what == "free":
                    recs.append(_k(k, FREE_OP, a, w=r.randint(0, 1)))
                elif what == "massive":
                    recs.append(_k(k, MASSIVE_OP, a, w=r.randint(0, 1)))
                elif what == "sensor":
                    recs.append(_k(k, SENSOR_OP, a, x=0.0, rr=r.uniform(10.0, 500.0)))
                elif what == "interactive":
                    recs.append(_k(k, INTERACTIVE_OP, a, w=0))
                else:
                    recs.append(_k(k, POS_OP, a, x=centre[0] + r.uniform(-50, 50), y=centre[1], z=centre[2]))
            for k in range(steps + 1):
                if r.random() < 0.5:
                    recs.append(_k(k, DEADS_OP))
        case("iter_" + flavour, sorted(recs, key=lambda c: c[1]), steps, master=r.random() < 0.75,
             tick=r.choice([250, 500, 1000, 1000, 1000, 2000]),
             friction=r.choice([1.0e6] * 5 + [10 ** r.uniform(-1.0, 2.0), 10 ** r.uniform(5.0, 7.0)]))
    return out


# ---- running ----

def serialize(cs: list[dict]) -> str:
    lines = []
    for c in cs:
        flags = int(c["new_orbit"]) | (2 if c.get("iterative") else 0)     # bit 1: the iterative collision method
        lines.append(f"C {c['id']} {c['friction']!r} {c['tick']} {c['steps']} {flags} {int(c['master'])} "
                     f"{c['start']} {c['seed']} {len(c['records'])}")
        for rec in c["records"]:
            if rec[0] == "B":
                b = rec[2]
                lines.append(f"B {rec[1]} {b['id']} "
                             + " ".join(repr(float(b[k])) for k in ("mass", "radius", "max_vel"))
                             + f" {int(b['free'])} {int(b['global'])} {int(b['massive'])} {int(b['interactive'])}"
                               f" {int(b['junk'])} " + " ".join(repr(float(v)) for v in (*b["pos"], *b["vel"]))
                             + f" {float(b['agility'])!r} {float(b['sf'])!r}")
            else:
                _, step, op, a, bb, w, x, y, z, rr = rec
                lines.append(f"K {step} {op} {a} {bb} {w} {x!r} {y!r} {z!r} {rr!r}")
    return "\n".join(lines) + "\n"


def _parse_args(tokens: list) -> tuple:
    out = []
    for t in tokens:
        if t[0] == "i":
            out.append(int(t[1:]))
        elif t[0] == "d":
            out.append(float(t[1:]))
        else:
            out.append(None)
    return tuple(out)


def _s_row(p: list) -> tuple:
    ni, nd = len(S_INT), len(S_DBL)                         # the tag is p[0]
    ints = tuple(int(x) for x in p[1:1 + ni])
    dbls = tuple(float(x) for x in p[1 + ni:1 + ni + nd])
    i = 1 + ni + nd
    nb = int(p[i])
    boxes = tuple(tuple(int(x) for x in p[i + 1 + 4 * j:i + 5 + 4 * j]) for j in range(nb))
    i += 1 + 4 * nb
    lists = []
    for _ in range(3):
        n = int(p[i])
        lists.append(tuple(int(x) for x in p[i + 1:i + 1 + n]))
        i += 1 + n
    return ints + dbls + (boxes, *lists)


def parse(text: str) -> dict:
    """The harness's (or the reference's) printed rows, by kind."""
    out = defaultdict(list)
    for line in text.splitlines():
        p = line.split()
        tag = p[0]
        if tag == "S":
            out["S"].append(_s_row(p))
        elif tag == "P":
            counts = tuple(int(x) for x in p[6:14])
            nprox = int(p[14])
            prox = tuple(int(x) for x in p[15:15 + nprox])
            out["P"].append((int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), counts, prox,
                             int(p[15 + nprox])))
        elif tag in "UIV":
            n = int(p[4])
            body = (tuple((int(p[5 + 2 * j]), int(p[6 + 2 * j])) for j in range(n)) if tag == "U"
                    else tuple(int(x) for x in p[5:5 + n]))
            out[tag].append((int(p[1]), int(p[2]), int(p[3]), body))
        elif tag == "W":
            n = int(p[3])
            out["W"].append((int(p[1]), int(p[2]), tuple(int(x) for x in p[4:4 + n])))
        elif tag == "D":
            nb = int(p[14])
            balls = tuple(int(x) for x in p[15:15 + nb])
            nc = int(p[15 + nb])
            kids = tuple(tuple(int(x) for x in p[16 + nb + 3 * j:19 + nb + 3 * j]) for j in range(nc))
            out["D"].append((int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]), int(p[7]),
                             *map(float, p[8:14]), balls, kids))
        elif tag == "Q":
            op, status, n = int(p[3]), int(p[6]), int(p[7])
            if op == RANGE_DIST_Q:
                items = tuple((float(p[8 + 2 * j]), int(p[9 + 2 * j])) for j in range(n))
            else:
                items = tuple(int(x) for x in p[8:8 + n])
            out["Q"].append((int(p[1]), int(p[2]), op, int(p[4]), int(p[5]), status, items))
        elif tag == "E":
            target, n = int(p[4]), int(p[5])
            args = _parse_args(p[6:6 + n])
            owner = args[0] if target == -1 else target
            if target == -1:
                args = args[1:]
            out["E"].append((int(p[1]), int(p[2]), p[3], owner, args))
        else:
            raise ValueError(f"unknown row {line[:40]!r}")
    return out


def oracle(cs: list[dict]) -> dict:
    """Destiny's compiled code."""
    res = subprocess.run([str(HARNESS)], input=serialize(cs), capture_output=True, text=True, check=True, timeout=240)
    return parse(res.stdout)


def _ball_row(cid: int, s: int, park: SpaceBallpark, b) -> tuple:
    mb = b.my_box
    active = sum(1 << i for i in range(27) if b.active[i])
    return (cid, s, b.id, b.mode, b.effect_stamp, int(b.is_massive), int(b.sensor_active), b.follow_id, b.owner_id,
            -1 if b.follow_ptr is None else b.follow_ptr, int(b.is_free), int(b.id in park.free_balls),
            int(b.is_global), int(b.is_interactive), b.is_cloaked, int(b.is_moribund), int(b.is_space_junk),
            b.bubble, b.old_bubble, int(b.has_proximity), int(b.within_range), int(b.within_tracking_range),
            -1 if b.tracking_ptr is None else b.tracking_ptr.id, int(b.in_dead_bubble), *b.mod, active,
            mb.level if mb else -1, *(mb.key if mb else (0, 0, 0)), b.harmonic, b.corporation_id, b.alliance_id,
            int(b.sensor_only_interactives),
            *b.pos, *b.vel, *b.old_pos, *b.old_vel, *b.goto, *b.last_g, *b.last_c, b.mass, b.radius, b.max_vel,
            b.agility, b.speed_fraction, b.follow_range, b.last_collision, b.time_factor, b.notification_range,
            b.tracking_range, b.sensor_range, b.sensor_period, b.sensor_elapsed,
            tuple((box.level, *box.key) for box in b.sorted_boxes()), tuple(b.collisions), tuple(sorted(b.followers)),
            tuple(b.sensor_balls.keys()))


def _state(out: dict, cid: int, s: int, park: SpaceBallpark) -> None:
    for b in park.balls.values():
        out["S"].append(_ball_row(cid, s, park, b))
    counts = tuple(len(level) for level in park.levels)
    out["P"].append((cid, s, park.current_time, park.bubble_id, park.partition_hash(), counts,
                     tuple(park.proximity_balls.keys()), len(park.free_balls)))
    for bubble, members in park.bubbles.items():
        out["U"].append((cid, s, bubble, tuple(members.items())))
    for bubble, members in park.bubble_interactives.items():
        out["I"].append((cid, s, bubble, tuple(sorted(members))))
    for bubble, members in park.bubble_keep_alives.items():
        out["V"].append((cid, s, bubble, tuple(sorted(members))))
    out["W"].append((cid, s, tuple(sorted(park.bubble_keep_alive_balls))))


def _dump(out: dict, cid: int, park: SpaceBallpark) -> None:
    for level in range(len(park.levels)):
        for key in sorted(park.levels[level]):
            box = park.levels[level][key]
            out["D"].append((cid, level, *key, box.bubble, int(box.handled), *box.lo, *box.hi,
                             tuple(sorted(box.balls)), tuple(sorted(box.children))))


def _apply(park: SpaceBallpark, rec: tuple, cid: int, s: int, out: dict) -> None:
    if rec[0] == "B":
        b = rec[2]
        park.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], b["free"], b["global"], b["massive"],
                      b["interactive"], b["junk"], *b["pos"], *b["vel"], b["agility"], b["sf"])
        return
    _, _, op, a, bb, w, x, y, z, rr = rec
    if op == GOTO_OP:
        park.goto_point(a, x, y, z)
    elif op == DIR_OP:
        park.goto_direction(a, x, y, z)
    elif op == STOP_OP:
        park.stop(a)
    elif op == FOLLOW_OP:
        park.follow_ball(a, bb, rr)
    elif op == ORBIT_OP:
        park.orbit(a, bb, rr)
    elif op == WARP_OP:
        park.warp_to(a, x, y, z, rr, w)
    elif op == WARPIN_OP:
        park.entity_warp_in(a, x, y, z, w)
    elif op == POS_OP:
        park.set_position(a, x, y, z)
    elif op == VEL_OP:
        park.set_velocity(a, x, y, z)
    elif op == RADIUS_OP:
        park.set_radius(a, rr)
    elif op == MAXVEL_OP:
        park.set_max_velocity(a, rr)
    elif op == MASS_OP:
        park.set_mass(a, rr)
    elif op == AGILITY_OP:
        park.set_agility(a, rr)
    elif op == SF_OP:
        park.set_speed_fraction(a, rr)
    elif op == FREE_OP:
        park.set_free(a, w != 0)
    elif op == MASSIVE_OP:
        park.set_massive(a, w != 0)
    elif op == GLOBAL_OP:
        park.set_global(a, w != 0)
    elif op == INTERACTIVE_OP:
        park.set_interactive(a, w != 0)
    elif op == SENSOR_OP:
        park.add_proximity_sensor(a, rr, x, bb, w != 0)
    elif op == UNSENSOR_OP:
        park.remove_proximity_sensor(a)
    elif op == NOTIFY_OP:
        park.set_notification_range(a, rr)
    elif op == TRACK_OP:
        park.set_target_tracking(a, bb, rr)
    elif op == HARMONIC_OP:
        park.set_harmonic(a, bb, int(x), int(y), w != 0)
    elif op == CLOAK_OP:
        park.set_cloaked(a, w)
    elif op == INIT_BUBBLES_OP:
        park.initialize_bubbles(a)
    elif op == COPY_BUBBLES_OP:
        park.copy_bubbles()
    elif op == CLEAN_FOLLOWERS_OP:
        out["Q"].append((cid, s, op, a, bb, 1, tuple(park.clean_up_followers())))
    elif op == KEEPALIVE_OP:
        park.add_keep_alive_ball(a)
    elif op == DEADS_OP:
        park.bring_out_the_deads()
    elif op == REMOVE_OP:
        park.remove_ball(a, w)
    elif op == VIS_Q:
        out["Q"].append((cid, s, op, a, bb, 1, (park.check_visibility(a, bb),)))
    elif op in (RANGE_BALL_Q, RANGE_POINT_Q, RANGE_DIST_Q):
        args = (a, rr, w) if op != RANGE_POINT_Q else (x, y, z, rr, w)
        res = park.ball_ids_in_range(*args, include_dist=op == RANGE_DIST_Q)
        out["Q"].append((cid, s, op, a, bb, 0, ()) if res is None else (cid, s, op, a, bb, 1, tuple(res)))
    elif op == BUBBLE_AT_Q:
        bub = park.bubble_at(x, y, z)
        out["Q"].append((cid, s, op, a, bb, 1, (-999 if bub is None else bub,)))
    else:
        raise ValueError(f"unknown op {op}")


def reference(c: dict, trace=None) -> dict:
    """The reference on one scenario, in the oracle's layout."""
    park = SpaceBallpark(friction=c["friction"], tick_interval_ms=c["tick"], use_new_orbit=c["new_orbit"],
                         is_master=c["master"], seed=c["seed"], iterative=c.get("iterative", False))
    park.current_time = c["start"]
    park.trace = trace
    out = defaultdict(list)
    by_step = defaultdict(list)
    for rec in c["records"]:
        by_step[rec[1]].append(rec)
    for s in range(c["steps"] + 1):
        for rec in by_step.get(s, ()):
            _apply(park, rec, c["id"], s, out)
        if s > 0:
            park.evolve()
        _state(out, c["id"], s, park)
    _dump(out, c["id"], park)
    for (t, name, bid, a, b) in park.events:
        out["E"].append((c["id"], t, name, bid, (a, b)[:ARITY[name]]))
    return out


def run_reference(cs: list[dict], trace=None) -> tuple[dict, dict]:
    """The reference on every scenario: (rows by kind, {case id: reason} for the scenarios where Destiny's result is
    undefined, which are left out of the rows)."""
    rows, undefined = defaultdict(list), {}
    for c in cs:
        try:
            out = reference(c, trace)
        except DestinyUndefined as exc:
            undefined[c["id"]] = str(exc)
            continue
        for k, v in out.items():
            rows[k].extend(v)
    return rows, undefined


def without(rows: dict, cases_out) -> dict:
    """Rows of the scenarios not in cases_out (every row's first field is its case)."""
    drop = set(cases_out)
    return {k: [row for row in v if row[0] not in drop] for k, v in rows.items()}


def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def same(a, b) -> bool:
    """Equal integers and tuples, or doubles with equal bits (two NaNs count as equal)."""
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, float) and isinstance(b, float) and (bits(a) == bits(b) or (a != a and b != b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def per_owner(events: list) -> dict:
    """Events grouped by (case, ball), each group in order."""
    out = defaultdict(list)
    for (cid, t, name, owner, args) in events:
        out[(cid, owner)].append((t, name, args))
    return dict(out)

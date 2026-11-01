# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Cases for more than 13 keys in the unordered containers Destiny walks, generated from a
seed. Three families, each compared bit for bit:

- tapes for libstdc++'s containers alone (conformance/hashtable_probe.cpp, compiled with the harness's g++,
  against exact_ballpark.space's StdHashMap and StdHashSet): keys inserted, set, erased, cleared, copied and erased
  during a walk, up to some hundreds a container, so the buckets grow through several primes of __prime_list;
- crowds for the compiled Destiny harness (tests/space_cases.py's records: conformance/build_space.py's
  binary against exact_ballpark.space): 14 to 70 balls in one bubble, many of them proximity balls (sensors,
  notification ranges, tracking), sensors whose lists pass 13, 29 and 59 balls, lists emptied and refilled (a sensor
  added again, a dead bubble), balls removed, a ball with more than 13 followers released;
- crowded worlds for the batched layer (tests/space_batch.py's worlds: exact_ballpark.vec_space's host path and kernel
  against the scalar reference): up to 64 balls a world, as many proximity balls, sensor lists past 13, 29 and 59.

Keys come from families that share buckets before and after a rehash (multiples of 13, 29, 59, 127 apart) and from
small and large ids, in an order unrelated to their values. Development seeds 1 and 3; held-out seeds 2 and 4, first
run after this file was frozen.
"""

from __future__ import annotations

import random

import space_cases as SC
from exact_ballpark.space import LEVEL_WIDTH

TOP = LEVEL_WIDTH[0]
MODULI = (13, 29, 59, 127, 257, 13 * 29, 29 * 59)


def keys(r: random.Random, n: int, big: bool = True) -> list:
    """n distinct positive ids: families sharing buckets (a base plus multiples of a bucket count), small ids, large
    ids (up to 2**40, or 2**62 with `big`), shuffled."""
    out: list = []
    seen: set = set()
    while len(out) < n:
        k = r.random()
        if k < 0.45 and out:
            m = r.choice(MODULI)
            x = r.choice(out) + m * r.randint(1, 40)
        elif k < 0.7:
            x = r.randrange(1, 400)
        elif k < 0.9 or not big:
            x = r.randrange(1, 2 ** 40)
        else:
            x = r.randrange(1, 2 ** 62)
        if x not in seen:
            seen.add(x)
            out.append(x)
    r.shuffle(out)
    return out


# ============================================================================================ container tapes

def walk_erases(key: int, d: int) -> bool:
    """The probe's walk (op w, argument d): the keys it erases while it walks, as CheckForProximity_Sensor erases."""
    return ((key & ((1 << 64) - 1)) // 7) % d == 0


def tapes(seed: int, n: int = 40) -> list:
    """n tapes of (kind, index, op, argument) lines for the probe: kinds m (MapOfLongLongs: operator[] and insert),
    d (DictOfBalls: insert(pair)), s (SetOfIDs); ops i, o, e, c, y and w (a walk erasing the keys walk_erases names)."""
    r = random.Random(f"g12-tapes-{seed}")
    out = []
    for _ in range(n):
        kinds = r.sample(["m", "d", "s"], r.randint(1, 2))
        target = r.choice([20, 40, 70, 140, 300, 700])
        pool = keys(r, 2 * target)
        live = {(kd, j): set() for kd in kinds for j in range(r.randint(1, 2))}
        lines = []
        for step in range(r.randint(2 * target, 4 * target)):
            kd, j = r.choice(sorted(live))
            cur = live[(kd, j)]
            if step < target and r.random() < 0.8:           # the growth: fresh keys, one by one
                fresh = [k for k in pool if k not in cur]
                if fresh:
                    key = fresh[0] if r.random() < 0.5 else r.choice(fresh)
                    lines.append((kd, j, "o" if kd == "m" and r.random() < 0.5 else "i", key))
                    cur.add(key)
                    continue
            q = r.random()
            if q < 0.7 or not cur:
                fresh = [k for k in pool[:len(cur) + 8] if k not in cur]
                key = r.choice(fresh) if fresh and r.random() < 0.8 else r.choice(pool)
                op = "o" if kd == "m" and r.random() < 0.5 else "i"
                lines.append((kd, j, op, key))
                cur.add(key)
            elif q < 0.88:
                key = r.choice(sorted(cur)) if r.random() < 0.85 else r.choice(pool)
                lines.append((kd, j, "e", key))
                cur.discard(key)
            elif q < 0.94:
                lines.append((kd, j, "y", 0))
            elif q < 0.985:
                d = r.choice([2, 3, 5, 7])
                lines.append((kd, j, "w", d))
                cur.difference_update({k for k in cur if walk_erases(k, d)})
            else:
                lines.append((kd, j, "c", 0))
                cur.clear()
        out.append(lines)
    return out


def tape_text(lines: list) -> str:
    return "".join(f"{kd} {j} {op} {arg}\n" for kd, j, op, arg in lines)


# ============================================================================== crowds for the Destiny harness

def _crowd_ball(r: random.Random, bid: int, centre, spread: float, **kw) -> tuple:
    """A ball of a crowd: a coarse filing (radius 40 to 160 km, or 170 to 600 km), placed around centre."""
    radius = r.choice([r.uniform(4.2e4, 1.6e5), r.uniform(4.2e4, 1.6e5), r.uniform(1.7e5, 6.0e5)])
    pos = [centre[q] + r.uniform(-spread, spread) for q in range(3)]
    b = dict(radius=radius, pos=pos, free=r.random() < 0.85, interactive=r.random() < 0.6,
             massive=False, max_vel=r.uniform(50.0, 900.0), vel=r.choice([[0.0, 0.0, 0.0],
                                                                          [r.uniform(-200, 200) for _ in range(3)]]))
    b.update(kw)
    return SC._ball(r, bid, **b)


def crowds(seed: int, n: int = 24) -> list:
    """n scenarios in tests/space_cases.py's format (one id a scenario, records at steps), each a crowd in one top box:
    14 to 70 balls, proximity balls by sensors, notification ranges and tracking, sensor lists past 13, 29 and 59."""
    r = random.Random(f"g12-crowds-{seed}")
    out: list = []
    for cid in range(n):
        nb = r.choice([14, 16, 20, 31, 33, 45, 61, 64, 70, 70, 70])
        ids = keys(r, nb, big=False)
        w1 = LEVEL_WIDTH[1]
        centre = [(r.randint(-2, 1) + 0.5) * w1 for _ in range(3)]
        spread = r.choice([2.0e5, 4.0e5, 6.5e5])
        recs = [_crowd_ball(r, bid, centre, spread) for bid in ids]
        if not any(rec[2]["interactive"] for rec in recs):
            recs[0][2]["interactive"] = True
        steps = r.randint(6, 14)
        sensors = r.sample(ids, r.randint(1, min(len(ids), 6)))
        for s in sensors:                                   # wide sensors: lists of most of the crowd
            recs.append(SC._k(r.choice([0, 0, 1]), SC.SENSOR_OP, s, b=0, w=int(r.random() < 0.2),
                              x=r.choice([0.0, 1.0, 2.0]), rr=r.choice([r.uniform(1.5e6, 3e6), r.uniform(4e5, 2.5e6),
                                                                       r.uniform(1e5, 6e5)])))
        prox = ids if r.random() < 0.5 else r.sample(ids, r.randint(min(len(ids), 12), len(ids)))
        for a in prox:                                      # many proximity balls
            if r.random() < 0.7:
                recs.append(SC._k(r.choice([0, 0, 1, 2]), SC.NOTIFY_OP, a, rr=r.uniform(1e3, 4e5)))
            else:
                recs.append(SC._k(r.choice([0, 1]), SC.TRACK_OP, a, r.choice(ids), rr=r.uniform(1e4, 9e5)))
        if r.random() < 0.5:                                # a ball with many followers, released at a step
            target = r.choice(ids)
            for f in r.sample([x for x in ids if x != target], min(len(ids) - 1, r.randint(14, 40))):
                recs.append(SC._k(1, r.choice([SC.FOLLOW_OP, SC.ORBIT_OP]), f, target, rr=r.uniform(0, 5e5)))
            k = r.randint(2, steps)
            recs.append(SC._k(k, r.choice([SC.REMOVE_OP, SC.POS_OP, SC.STOP_OP]), target,
                              x=centre[0] + r.uniform(-1e5, 1e5), y=centre[1], z=centre[2]))
        for _ in range(r.randint(4, 16)):                   # lists emptied, refilled, balls out of and into range
            k, a = r.randint(1, steps), r.choice(ids)
            what = r.choice(["away", "away", "back", "resensor", "unsensor", "notify0", "track0", "interactive",
                             "remove", "goto"])
            if what == "away":
                recs.append(SC._k(k, SC.POS_OP, a, x=centre[0] + r.choice([-1, 1]) * r.uniform(1.2e6, 1.8e6),
                                  y=centre[1] + r.uniform(-3e5, 3e5), z=centre[2] + r.uniform(-3e5, 3e5)))
            elif what == "back":
                recs.append(SC._k(k, SC.POS_OP, a, x=centre[0] + r.uniform(-spread, spread),
                                  y=centre[1] + r.uniform(-spread, spread), z=centre[2] + r.uniform(-spread, spread)))
            elif what == "resensor":
                recs.append(SC._k(k, SC.SENSOR_OP, r.choice(sensors), x=r.choice([0.0, 1.0]),
                                  rr=r.uniform(3e5, 2.5e6)))
            elif what == "unsensor":
                recs.append(SC._k(k, SC.UNSENSOR_OP, r.choice(sensors)))
            elif what == "notify0":
                recs.append(SC._k(k, SC.NOTIFY_OP, a, rr=0.0))
            elif what == "track0":
                recs.append(SC._k(k, SC.TRACK_OP, a, 0, rr=0.0))
            elif what == "interactive":
                recs.append(SC._k(k, SC.INTERACTIVE_OP, a, w=r.randint(0, 1)))
            elif what == "remove":
                recs.append(SC._k(k, SC.REMOVE_OP, a, w=r.choice([0, 0, 2])))
            else:
                recs.append(SC._k(k, SC.GOTO_OP, a, x=centre[0] + r.uniform(-spread, spread),
                                  y=centre[1] + r.uniform(-spread, spread), z=centre[2] + r.uniform(-spread, spread)))
        if r.random() < 0.3:
            recs.append(SC._k(r.randint(2, steps), SC.DEADS_OP))
        out.append({"id": cid + 1, "kind": "crowd", "friction": 1.0e6, "tick": r.choice([500, 1000, 2000]),
                    "steps": steps, "new_orbit": False, "master": True, "seed": r.randrange(1, 2 ** 31),
                    "start": r.randrange(0, 100000), "records": sorted(recs, key=lambda c: c[1])})
    return out


# ================================================================= crowded worlds for the batched layer

def _world_ball(r: random.Random, bid: int, centre, spread: float) -> dict:
    radius = r.choice([r.uniform(4.2e4, 1.6e5), r.uniform(1.7e5, 6.0e5), r.uniform(1.7e5, 6.0e5)])
    return {"id": bid, "mass": 10 ** r.uniform(5.5, 9.0), "radius": radius, "max_vel": r.uniform(50.0, 900.0),
            "agility": r.uniform(0.05, 2.0), "sf": r.choice([1.0, r.uniform(0.1, 1.0)]),
            "pos": [centre[q] + r.uniform(-spread, spread) for q in range(3)],
            "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-200.0, 200.0) for _ in range(3)]]),
            "interactive": r.random() < 0.6, "massive": False, "free": True, "occluder": False}


def worlds(seed: int, n: int, park: str = "orient", steps: int = 12, sizes=(14, 20, 30, 31, 45, 60, 64)) -> list:
    """n worlds in tests/space_batch.py's format: a crowd of 14 to 64 balls in one top box, wide sensors, notification
    ranges and tracking on most balls, then records that empty and refill the lists and the proximity map (balls
    jumping out of range and back, sensors added again and removed, ranges cleared, interactive changes, removal),
    and the layer's queries now and then. On VecModesPark ("modes") also follows."""
    r = random.Random(f"g12-worlds-{park}-{seed}")
    out = []
    w1 = LEVEL_WIDTH[1]
    for wid in range(n):
        nb = r.choice(sizes)
        ids = keys(r, nb, big=False)
        centre = [(r.randint(-2, 1) + 0.5) * w1 for _ in range(3)]
        spread = r.choice([2.0e5, 4.0e5, 6.5e5])
        balls = [_world_ball(r, bid, centre, spread) for bid in ids]
        if not any(b["interactive"] for b in balls):
            balls[0]["interactive"] = True
        w = {"id": wid, "park": park, "balls": balls, "anchors": [tuple(centre)], "steps": [], "pair": None}
        sensors = r.sample(ids, r.randint(1, min(nb, 6)))
        alive = list(ids)
        first = []
        for s in sensors:
            first.append(("sensor", s, r.choice([r.uniform(1.5e6, 3e6), r.uniform(4e5, 2.5e6), r.uniform(1e5, 6e5)]),
                          r.choice([2.0, 1.0, 0.5]), int(r.random() < 0.2)))
        for a in (ids if r.random() < 0.5 else r.sample(ids, r.randint(min(nb, 12), nb))):
            if r.random() < 0.7:
                first.append(("notify", a, r.uniform(1e3, 4e5)))
            else:
                first.append(("track", a, r.choice(ids), r.uniform(1e4, 9e5)))
        r.shuffle(first)
        w["steps"].append(first)
        for _ in range(steps - 1):
            recs = []
            for _ in range(r.choice([0, 1, 2, 3, 4, 6])):
                if not alive:
                    break
                a = r.choice(alive)
                k = r.random()
                if k < 0.22:
                    recs.append(("pos", a, centre[0] + r.choice([-1, 1]) * r.uniform(1.2e6, 1.8e6),
                                 centre[1] + r.uniform(-3e5, 3e5), centre[2] + r.uniform(-3e5, 3e5)))
                elif k < 0.40:
                    recs.append(("pos", a, *[centre[q] + r.uniform(-spread, spread) for q in range(3)]))
                elif k < 0.50:
                    recs.append(("sensor", r.choice(sensors), r.uniform(3e5, 2.5e6), r.choice([2.0, 1.0, 0.5]),
                                 int(r.random() < 0.2)))
                elif k < 0.55:
                    recs.append(("unsensor", r.choice(sensors)))
                elif k < 0.62:
                    recs.append(("notify", a, r.choice([0.0, r.uniform(1e3, 4e5)])))
                elif k < 0.68:
                    recs.append(("track", a, r.choice([r.choice(ids), 0]), r.choice([0.0, r.uniform(1e4, 9e5)])))
                elif k < 0.74:
                    recs.append(("interactive", a, r.randint(0, 1)))
                elif k < 0.77 and len(alive) > 2:
                    recs.append(("remove", a))
                    alive.remove(a)
                elif k < 0.83:
                    recs.append(("goto", a, *[centre[q] + r.uniform(-spread, spread) for q in range(3)]))
                elif park == "modes" and k < 0.90 and len(alive) > 1:
                    o = r.choice([x for x in alive if x != a])
                    recs.append((r.choice(["follow", "orbit"]), a, o, r.uniform(1e3, 3e5)))
                elif k < 0.93:
                    recs.append(("copy",))
                elif k < 0.96:
                    recs.append(("range_ball", a, r.uniform(1e5, 2e6), r.choice([0, 1])))
                else:
                    recs.append(("remote",))
            w["steps"].append(recs)
        out.append(w)
    return out

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the composed park (exact_ballpark.vec_composed.VecComposedPark): flight modes, missiles
and ship-to-ship collisions together, generated from a seed, run by the scalar reference
(exact_ballpark.full.FullBallpark, one park a world) and by the batch (one park for a batch of worlds), and compared bit
for bit after every step: every ball's state (tests/destiny_combat_cases.py's fields, with the collision acceleration
mLastC, the collision list and the partition's filing: level, key, adjacent side), which balls exist, the answers of
ReserveFormationSlot and the DoCollision events.

A world is a cluster of ships (ids 1 to 6) a few hundred metres across, most of them massive, some fixed spheres
(ids 7 and 8), a missile pool (101 to 104, tests/destiny_combat_cases.py's) and records: the ships GOTO through the
cluster, follow and orbit each other at short range, stop, change their radius, maximum speed, speed fraction and
mass, are made massive or not, kicked, cloaked; a ship warps into the cluster from far away (WarpTo with a small
minimum range, or EntityWarpIn) and drops out among the others, or warps out of it; missiles are launched at ships
(massive or not, aimed or not, or with no target, flying on massive by GotoDirection) and time out; formations are
flown. On the server the cluster stays inside one top box (one bubble), and a normal cloak (a proximity sensor, not
modelled by the batch) is not generated. The reference is held to Destiny's compiled code by
tests/test_movement_full_oracle.py and tests/test_movement_space_oracle.py.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import torch

import destiny_combat_cases as dc
from exact_ballpark.full import FullBallpark
from exact_ballpark.space import LEVEL_WIDTH, DestinyUndefined

POOL = dc.POOL
FORMATIONS = dc.FORMATIONS
EXTRA = ("last_c", "collisions", "level", "key", "mod")
EXTRA_INTS = ("collisions", "level", "key", "mod")
SHIPS = 6                                                   # ids 1..6; fixed spheres 7 and 8


def _point(r: random.Random, centre, spread: float) -> list:
    u = dc._unit(r)
    d = r.uniform(0.0, spread)
    return [centre[i] + u[i] * d for i in range(3)]


def _ship(r: random.Random, bid: int, centre, spread: float, **kw) -> dict:
    b = {"id": bid, "mass": 10 ** r.uniform(5.0, 9.0),
         "radius": r.choice([r.uniform(5.0, 40.0), r.uniform(40.0, 150.0)]),
         "max_vel": r.choice([150.0, r.uniform(40.0, 450.0)]), "free": True, "massive": r.random() < 0.85,
         "interactive": r.random() < 0.5, "pos": _point(r, centre, spread),
         "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-60.0, 60.0) for _ in range(3)]]),
         "agility": r.uniform(0.2, 1.5), "speed_fraction": r.choice([1.0, r.uniform(0.2, 1.0)]),
         "max_ang_vel": r.choice([1.0, r.uniform(0.1, 2.0)]), "rot_agility": r.choice([0.9, r.uniform(0.1, 1.5)])}
    b.update(kw)
    return b


def _warp_time(w: int, dist: float) -> float:
    """A rough warp duration in seconds (tests/mode_cases.py's estimate; the generator's, never the reference)."""
    from exact_ballpark.modes import AU
    s = w * 0.001 * AU
    acc, dec = w / 1000.0, min(w / 3000.0, 2.0)
    s = min(s, (dist + 1) * acc * dec / (acc + dec))
    return (math.log(max(s / acc, 1.0)) / acc + max(0.0, dist / s - 1 / acc - 1 / dec)
            + math.log(max(s / dec, 1.0)) / dec + 1.0)


def world(r: random.Random, steps: int, master: bool, tick: int = 1000) -> dict:
    """One world: ships, fixed spheres, a missile pool, and its records [(step, op, *args)]."""
    if master:                                              # inside one top box, away from its faces: one bubble
        centre = [r.uniform(1.5e6, LEVEL_WIDTH[0] - 1.5e6) for _ in range(3)]
    else:                                                   # anywhere near the origin (every level's faces)
        centre = [r.uniform(-3e5, 3e5) for _ in range(3)]
    spread = r.choice([60.0, 200.0, 600.0])
    n = r.randint(2, SHIPS)
    ships = [_ship(r, 1, centre, spread, interactive=True, massive=True)]
    ships += [_ship(r, k, centre, spread) for k in range(2, n + 1)]
    for k in range(r.choice([0, 0, 1, 2])):                # fixed spheres: obstacles
        ships.append({"id": SHIPS + 1 + k, "mass": 10 ** r.uniform(6.0, 10.0), "radius": r.uniform(20.0, 300.0),
                      "max_vel": 0.0, "free": False, "massive": True, "interactive": False,
                      "pos": _point(r, centre, spread), "vel": [0.0, 0.0, 0.0], "agility": 1.0,
                      "speed_fraction": 0.0, "max_ang_vel": 1.0, "rot_agility": 0.9})
    free_ids = [s["id"] for s in ships if s["free"]]
    recs: list = []

    def near():
        return _point(r, centre, spread)

    warper = None
    if n >= 2 and r.random() < 0.7:                         # a ship far away warps into the cluster and drops out
        warper = r.choice(free_ids[1:])
        s = next(s for s in ships if s["id"] == warper)
        u = dc._unit(r)
        far = r.uniform(1.2e5, 9.0e5)
        s["pos"] = [centre[i] + u[i] * far for i in range(3)]
        if r.random() < 0.25:                               # nearly no speed: the drop-out past the deceleration, at
            s["max_vel"] = r.choice([0.0, 0.5, 1.0])         # rest (re-filed there)
        k = r.choice([0, 0, 1, 2])
        dest = near()
        if r.random() < 0.5:                                # on a cluster ship: the drop-out touches it
            on = r.choice([o for o in ships if o['id'] != warper])
            dest = _point(r, on['pos'], on['radius'])
        warpin = r.random() < 0.5
        if not warpin:                                      # aligned and fast: warp proper at its first tick
            d = [dest[i] - s["pos"][i] for i in range(3)]
            nd = math.sqrt(sum(c * c for c in d))
            s["vel"] = [d[i] / nd * s["max_vel"] * r.uniform(0.8, 1.0) for i in range(3)]
        left = max(1.0, (steps - k - 2) * 0.001 * tick)       # the warp's time, from its estimate (never the reference)
        w = next((w for w in (1500, 3000, 6000, 10000, 20000) if _warp_time(w, far) < left), 20000)
        if warpin:
            recs.append((k, "warpin", warper, *dest, w))
        else:
            recs.append((k, "warp", warper, *dest, r.choice([0.0, 0.0, 50.0, 500.0]), w))
    for s in ships:                                          # the cluster's flight
        bid = s["id"]
        if not s["free"] or bid == warper:
            continue
        for _ in range(r.choice([1, 1, 2, 3])):
            k = r.choice([0, 0, 1, r.randint(1, steps)])
            others = [o for o in free_ids if o != bid] or [bid]
            how = r.choice(["goto", "goto", "dir", "stop", "follow", "follow", "orbit", "orbit"])
            if how == "goto":
                recs.append((k, "goto", bid, *near()))
            elif how == "dir":
                recs.append((k, "dir", bid, *dc._unit(r)))
            elif how == "stop":
                recs.append((k, "stop", bid))
            elif how == "follow":
                recs.append((k, "follow", bid, r.choice(others), r.choice([0.0, 0.0, 5.0, 50.0])))
            else:
                recs.append((k, "orbit", bid, r.choice(others), r.choice([0.0, 20.0, 150.0, 400.0])))
    warpers = {warper}
    if r.random() < 0.3 and len(free_ids) > 1:              # a ship warps out of the cluster (aligning while massive)
        who = r.choice(free_ids)
        warpers.add(who)
        u = dc._unit(r)
        recs.append((r.randint(0, steps), "warp", who, *[centre[i] + u[i] * r.uniform(2e5, 9e5) for i in range(3)],
                     r.choice([0.0, 2e4]), r.choice([1500, 3000, 20000])))
    for _ in range(r.randint(0, 4)):                         # the setters, kicks, massiveness, cloaks
        k = r.randint(1, steps)
        who = r.choice(free_ids)
        what = r.choice(["radius", "maxvel", "sf", "mass", "massive", "vel", "cloak"])
        if what == "massive" and who in warpers:            # SetBallMassive in warp proper: outside the domain
            what = "vel"
        if what == "radius":
            recs.append((k, "radius", who, r.uniform(2.0, 200.0)))
        elif what == "maxvel":
            recs.append((k, "maxvel", who, r.uniform(20.0, 500.0)))
        elif what == "sf":
            recs.append((k, "sf", who, r.uniform(0.0, 1.0)))
        elif what == "mass":
            recs.append((k, "mass", who, 10 ** r.uniform(5.0, 9.0)))
        elif what == "massive":
            recs.append((k, "massive", who, r.choice([0, 1, 1])))
        elif what == "vel":
            recs.append((k, "vel", who, *[r.uniform(-120.0, 120.0) for _ in range(3)]))
        else:
            mode = r.choice([2, 3, 0, 129] if master else [1, 1, 2, 3, 0, 129])
            recs.append((k, "cloak", who, mode))
            if r.random() < 0.6:
                recs.append((r.randint(k, steps), "uncloak", who))
    if r.random() < 0.3 and len(free_ids) > 1:               # a formation
        leader = r.choice(free_ids)
        recs.append((0, "setform", leader, r.choice([0, 0, 1, 2])))
        for bid in free_ids:
            if bid == leader or r.random() < 0.4:
                continue
            k = r.choice([0, 1, r.randint(1, steps)])
            if r.random() < 0.6:
                recs.append((k, "reserve_follow", bid, leader))
            else:
                recs.append((k, "formfollow", bid, leader, r.choice([0, 1, 2, 5])))
    missiles = list(POOL[:r.choice([0, 0, 1, 2, 4])])
    for mid in missiles:                                     # missiles at the cluster's ships
        k = r.randint(1, max(1, steps - 4))
        m = dc._missile(r, mid, near())
        recs.append((k, "add", m))
        owner = r.choice(free_ids)
        target = r.choice([t for t in free_ids if t != owner] or free_ids)
        how = r.choice(["launch", "launch", "launch", "follow", "no_target"])
        massive = r.choice([0, 1, 1, 1])
        if how == "follow":
            recs.append((k, "vel", mid, *[c * r.uniform(50.0, 300.0) for c in dc._unit(r)]))
            recs.append((k, "massive", mid, massive))
            recs.append((k, "mfollow", mid, target, owner))
        elif how == "no_target":
            recs.append((k, "launch", mid, 999, owner, r.choice([0, 1]), massive))
        else:
            recs.append((k, "launch", mid, target, owner, r.choice([0, 1, 1, 2]), massive))
        if r.random() < 0.25:
            recs.append((r.randint(k + 1, steps), "remove", mid))
    recs.sort(key=lambda rec: rec[0])
    return {"ships": ships, "pool": list(POOL), "records": recs, "start": r.randrange(0, 1000)}


def batches(seed: int, n_batches: int = 16, worlds: int = 16) -> list:
    """The scenarios of a seed: batches of worlds, each batch one configuration (even batches without dynamical
    orientation, odd ones with it; the server's view or a client's; the orbit law; the tick and friction)."""
    r = random.Random(seed)
    out = []
    for b in range(n_batches):
        do = b % 2 == 1
        master = r.random() < 0.5
        cfg = {"do": do, "do_off": do and r.random() < 0.4, "master": master,
               "tick": r.choice([250, 500, 1000, 1000, 1000, 2000]),
               "friction": r.choice([1.0e6] * 5 + [10 ** r.uniform(5.0, 7.0)]),
               "new_orbit": r.random() < 0.5, "steps": r.randint(12, 30)}
        cfg["worlds"] = [world(r, cfg["steps"], master, cfg["tick"]) for _ in range(worlds)]
        out.append(cfg)
    return out


# ---- the reference ----

def _ref_extra(b) -> dict:
    mb = b.my_box
    return {"last_c": tuple(float(c) for c in b.last_c), "collisions": tuple(int(c) for c in b.collisions),
            "level": mb.level if mb is not None else -1, "key": tuple(mb.key) if mb is not None else (0, 0, 0),
            "mod": tuple(int(c) for c in b.mod)}


def _apply_ref(park: FullBallpark, rec: tuple, answers: list) -> None:
    op, a = rec[1], rec[2:]
    if op == "follow":
        park.follow_ball(a[0], a[1], a[2])
    elif op == "orbit":
        park.orbit(a[0], a[1], a[2])
    elif op == "warpin":
        park.entity_warp_in(a[0], *a[1:4], a[4])
    elif op == "radius":
        park.set_radius(a[0], a[1])
    elif op == "maxvel":
        park.set_max_velocity(a[0], a[1])
    elif op == "sf":
        park.set_speed_fraction(a[0], a[1])
    elif op == "mass":
        park.set_mass(a[0], a[1])
    else:
        dc._apply_ref(park, rec, answers)


def reference(cfg: dict, w: dict, trace=None) -> dict:
    """One world on FullBallpark: tests/destiny_combat_cases.py's layout, each row with EXTRA."""
    park = FullBallpark(friction=cfg["friction"], tick_interval_ms=cfg["tick"], use_new_orbit=cfg["new_orbit"],
                        is_master=cfg["master"], dynamical_orientation=cfg["do"],
                        disable_do_for_missiles=cfg["do_off"])
    park.current_time = w["start"]
    park.trace = trace
    for s in w["ships"]:
        dc._add_ref(park, s)
    park.load_formations(FORMATIONS)
    by_step = defaultdict(list)
    for rec in w["records"]:
        by_step[rec[0]].append(rec)
    out = {"S": {}, "E": [], "A": [], "B": [], "X": None}
    try:
        for s in range(cfg["steps"] + 1):
            for rec in by_step.get(s, ()):
                _apply_ref(park, rec, out["A"])
            if s > 0:
                park.evolve()
            out["S"][s] = {b.id: {**dc._ref_row(b, cfg["do"]), **_ref_extra(b)} for b in park.balls.values()}
            if cfg["master"] and len({b.bubble for b in park.balls.values()}) > 1:
                out["B"].append(s)
    except DestinyUndefined as exc:
        out["X"] = str(exc)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:     # where Python raises and C computes on
        out["X"] = f"the reference raised {type(exc).__name__}: {exc}"
    out["E"] = [(t, bid, args[0]) for (t, name, bid, *args) in park.events if name == "DoCollision"]
    return out


# ---- the batch ----

def batched(cfg: dict, device: str = "cpu", park_cls=None, **kw) -> tuple:
    """The batch on VecComposedPark (or park_cls): (per world the layout of `reference`, the park)."""
    from exact_ballpark.vec_composed import VecComposedPark
    cls = park_cls or VecComposedPark
    rows, where = dc.rows_for(cfg)
    p = cls(len(rows), cfg["friction"], cfg["tick"], device, cfg["do"], cfg["new_orbit"], master=cfg["master"],
            disable_do_for_missiles=cfg["do_off"], **kw)
    p.load(rows)
    p.load_formations(FORMATIONS)
    W = len(cfg["worlds"])
    outs = [{"S": {}, "E": [], "A": [], "B": [], "X": None} for _ in range(W)]
    by_step = [defaultdict(list) for _ in range(W)]
    for wi, w in enumerate(cfg["worlds"]):
        for rec in w["records"]:
            by_step[wi][rec[0]].append(rec)
    n = len(rows)
    names = dc.fields(cfg["do"])
    for s in range(cfg["steps"] + 1):
        slots = max(len(by_step[wi].get(s, ())) for wi in range(W))
        for j in range(slots):
            recs = {wi: by_step[wi][s][j] for wi in range(W) if j < len(by_step[wi].get(s, ()))}
            apply_batch(p, recs, where, n, outs)
        if s > 0:
            p.evolve_torch()
            for (wl, t, mid, col) in p.events():
                outs[wl]["E"].append((t, mid, col))
        cache = {k: getattr(p, k).cpu().tolist() for k in names}
        last_c, level, key, mod = (getattr(p, k).cpu().tolist() for k in ("last_c", "level", "key", "mod"))
        lists = p.collision_lists()
        exists = p.exists.cpu().tolist()
        for wi in range(W):
            outs[wi]["S"][s] = {}
        for (wi, bid), r in where.items():
            if exists[r]:
                row = {k: dc._value(cache[k][r], k) for k in names}
                row.update(last_c=tuple(float(c) for c in last_c[r]), collisions=lists[r], level=int(level[r]),
                           key=tuple(int(c) for c in key[r]), mod=tuple(int(c) for c in mod[r]))
                outs[wi]["S"][s][bid] = row
    return outs, p


def apply_batch(p, recs: dict, where: dict, n: int, outs: list) -> None:
    """The j-th record of each world's step as masked commands: the new kinds here, the others by
    tests/destiny_combat_cases.py's apply_batch."""
    I64, F64, BOOL = torch.int64, torch.float64, torch.bool
    dev = p.dev
    mine = {"follow", "orbit", "warpin", "radius", "maxvel", "sf", "mass"}
    rest = {wi: rec for wi, rec in recs.items() if rec[1] not in mine}
    if rest:
        dc.apply_batch(p, rest, where, n, outs)
    kinds = defaultdict(dict)
    for wi, rec in recs.items():
        if rec[1] in mine:
            kinds[rec[1]][wi] = rec
    for op, group in kinds.items():
        mask = [False] * n
        ia = [0] * n
        f1 = [0.0] * n
        f3 = [[0.0, 0.0, 0.0] for _ in range(n)]
        for wi, rec in group.items():
            a = rec[2:]
            if (wi, a[0]) not in where:
                continue
            r = where[(wi, a[0])]
            mask[r] = True
            if op in ("follow", "orbit"):
                ia[r] = where.get((wi, a[1]), -1)
                f1[r] = a[2]
            elif op == "warpin":
                f3[r], ia[r] = list(a[1:4]), a[4]
            else:
                f1[r] = a[1]
        m = torch.tensor(mask, dtype=BOOL, device=dev) & p.exists
        A = torch.tensor(ia, dtype=I64, device=dev)
        X1 = torch.tensor(f1, dtype=F64, device=dev)
        X3 = torch.tensor(f3, dtype=F64, device=dev)
        if op == "follow":
            p.follow(m, A, X1)
        elif op == "orbit":
            p.orbit(m, A, X1)
        elif op == "warpin":
            p.entity_warp_in(m, X3, A)
        elif op == "radius":
            p.set_radius(m, X1)
        elif op == "maxvel":
            p.set_max_velocity(m, X1)
        elif op == "sf":
            p.set_speed_fraction(m, X1)
        elif op == "mass":
            p.set_mass(m, X1)


# ---- the comparison ----

def differences(want: dict, got: dict, limit: int = 20) -> list:
    """Every difference of one world (tests/destiny_combat_cases.py's, the EXTRA fields included)."""
    return dc.differences(want, got, limit)


def in_domain(want: dict) -> bool:
    return dc.in_domain(want)

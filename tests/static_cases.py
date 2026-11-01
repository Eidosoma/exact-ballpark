# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the batched static collidables (capsules and oriented boxes in
exact_ballpark.vec_collide's park, its torch path and the kernel), generated from a seed, in the records of
tests/full_cases.py (AddBall, AddCapsule and AddOrientedBox at step 0, commands at later steps), so that one scenario
runs in the scalar reference (exact_ballpark.full's FullBallpark, a client's view), in the batched park and in Destiny's
own code compiled verbatim (conformance/build_full.py, full_cases.HARNESS).

Flavours: ships flown through a capsule's cylinder, at its caps and along its axis, grazing it, starting inside it
(slowly, fast, at rest), resting beside it (the end velocity parallel to the axis: the caps alone), axis-aligned
capsules flown along their axis, zero-length capsules and capsules with no volume, ships backing into one (moving in,
thrust out: mLastG pointing out of it at the impact); ships flown at an oriented box's faces, edges and corners
(aligned, rotated, skewed or flat boxes), starting inside it (slowly, fast, at rest, at its centre), backing into it,
a box wider than a top box (filed in the 27 top boxes); fields of capsules, boxes and fixed spheres with
several racers flying GOTO through them with throttle and stop; a ship touching two static collidables at once (the
order and the equal-time rule); static collidables on partition box faces at several levels (the walk's order across
their boxes); clusters of a coarse capsule and fine static collidables on a box face with fast ships among them (several
responses in one tick at different times and in different boxes of the walk). Ticks of 0.25 to 2 s, Destiny's friction and others.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import torch

import full_cases as fc
from collide_cases import _collect, differences  # noqa: F401  (differences: the batched rows' comparison)
from exact_ballpark.space import LEVEL_WIDTH

FLAVOURS = ("capsule_through", "capsule_through", "capsule_cap", "capsule_graze", "capsule_inside", "capsule_rest",
            "capsule_axis", "capsule_degenerate", "capsule_backing", "box_face", "box_face", "box_edge", "box_corner",
            "box_inside", "box_centre", "box_huge", "box_backing", "field", "field", "two_statics", "two_statics",
            "three_statics", "faces", "ball_and_static", "cluster", "cluster", "cluster")


def _unit(r: random.Random) -> list:
    while True:
        u = [r.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        if n > 1e-3:
            return [c / n for c in u]


def _perp(r: random.Random, axis: list) -> list:
    """A unit vector perpendicular to `axis`."""
    while True:
        u = _unit(r)
        d = sum(u[i] * axis[i] for i in range(3))
        v = [u[i] - d * axis[i] for i in range(3)]
        n = math.sqrt(sum(c * c for c in v))
        if n > 1e-3:
            return [c / n for c in v]


def _ids(r: random.Random, n: int, low: int = 1) -> list:
    ids: set = set()
    while len(ids) < n:
        ids.add(r.choice([r.randrange(low, 40), r.randrange(low, 1000), r.randrange(low, 2 ** 40)]))
    out = sorted(ids)
    r.shuffle(out)                                          # insertion order differs from id order
    return out


def _ship(r: random.Random, bid: int, pos, vel=(0.0, 0.0, 0.0), **kw) -> tuple:
    """A Python AddBall record of a free, massive ship (keywords override)."""
    b = {"id": bid, "mass": 10 ** r.uniform(6.0, 8.5), "radius": r.choice([2.0, r.uniform(1.0, 15.0),
                                                                           r.uniform(15.0, 60.0)]),
         "max_vel": r.choice([10.0, r.uniform(5.0, 150.0), r.uniform(150.0, 400.0)]), "free": True, "global": False,
         "massive": True, "interactive": r.random() < 0.6, "junk": False, "pos": [float(x) for x in pos],
         "vel": [float(x) for x in vel], "agility": r.uniform(0.2, 1.5), "sf": r.choice([0.95, 1.0, r.uniform(0.3, 1.0)]),
         "max_ang_vel": 1.0, "rot_agility": 0.9}
    b.update(kw)
    return ("B", 0, b)


def _sphere(r: random.Random, bid: int, pos, radius: float) -> tuple:
    """A fixed, massive sphere (an obstacle)."""
    return _ship(r, bid, pos, free=False, radius=radius, mass=10 ** r.uniform(8.0, 11.0), max_vel=0.0)


def _capsule(sid: int, parent: int, a, b, radius: float) -> tuple:
    return fc._k(0, fc.CAPSULE_OP, sid, parent, d=(*a, *b, radius))


def _box(sid: int, parent: int, corner, edges) -> tuple:
    return fc._k(0, fc.OBOX_OP, sid, parent, d=(*corner, *edges[0], *edges[1], *edges[2]))


def _capsule_shape(r: random.Random, center, axis=None) -> tuple:
    axis = axis or _unit(r)
    length = r.choice([r.uniform(10.0, 120.0), r.uniform(120.0, 400.0)])
    radius = r.choice([r.uniform(1.0, 20.0), r.uniform(20.0, 80.0)])
    radius = max(radius, length / 40.0)                   # Capsule::InsertInBoxes' 2r steps: at most about 20
    a = [center[i] - 0.5 * length * axis[i] for i in range(3)]
    b = [center[i] + 0.5 * length * axis[i] for i in range(3)]
    return a, b, radius, axis


def _box_shape(r: random.Random, center, big: bool = False, flat: bool = False) -> tuple:
    """A corner and three edge vectors around a centre: axis-aligned, rotated, skewed or (rarely, or when asked) flat."""
    sizes = [r.uniform(4.0e6, 6.0e6) if big else r.choice([r.uniform(5.0, 60.0), r.uniform(60.0, 400.0)])
             for _ in range(3)]
    kind = "flat" if flat else r.choice(["aligned", "aligned", "rotated", "rotated", "rotated", "skew", "flat"])
    if kind == "aligned":
        axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    elif kind == "skew":
        axes = [_unit(r) for _ in range(3)]
    else:
        u, t = _unit(r), _unit(r)
        v = [u[1] * t[2] - u[2] * t[1], u[2] * t[0] - u[0] * t[2], u[0] * t[1] - u[1] * t[0]]
        nv = math.sqrt(sum(c * c for c in v)) or 1.0
        v = [c / nv for c in v]
        w = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        axes = [u, v, w]
    if kind == "flat":
        sizes[r.randrange(3)] = 0.0
    edges = [[axes[i][j] * sizes[i] for j in range(3)] for i in range(3)]
    corner = [center[j] - 0.5 * (edges[0][j] + edges[1][j] + edges[2][j]) for j in range(3)]
    return corner, edges


def _edge_coord(r: random.Random, level: int) -> float:
    """A coordinate on or near a partition box face at `level` (the grid is centred on the origin)."""
    w = LEVEL_WIDTH[level]
    return r.randint(-3, 3) * w + r.choice([0.0, r.uniform(-30.0, 30.0), r.choice([-1.0, 1.0]) * r.uniform(0.0, 2.0)])


def _goto(step: int, bid: int, point) -> tuple:
    return fc._k(step, fc.GOTO_OP, bid, d=tuple(float(x) for x in point))


def cases(seed: int, n: int = 60) -> list[dict]:
    """`n` scenarios from `seed`: full_cases' scenario dicts (a client's view, the gradient method)."""
    r = random.Random(81000 + seed)
    out: list[dict] = []
    for _ in range(n):
        flavour = r.choice(FLAVOURS)
        centre = [r.choice([0.0, r.uniform(-2.0e4, 2.0e4)]) for _ in range(3)]
        recs: list = []
        steps = r.randint(6, 16)
        ids = _ids(r, 6)
        sids = _ids(r, 5, low=1)
        ship_r = r.choice([2.0, r.uniform(1.0, 10.0), r.uniform(10.0, 40.0)])
        speed = r.choice([10.0, r.uniform(5.0, 60.0), r.uniform(60.0, 300.0)])
        parent = r.choice([0, 7, ids[-1]])

        def ship_at(bid, start, goal, moving=None, **kw):
            moving = r.random() < 0.3 if moving is None else moving
            vel = [0.0, 0.0, 0.0]
            if moving:
                d0 = [goal[i] - start[i] for i in range(3)]
                nd = math.sqrt(sum(c * c for c in d0)) or 1.0
                vel = [d0[i] / nd * speed for i in range(3)]
            kw.setdefault("radius", ship_r)
            kw.setdefault("max_vel", speed)
            recs.append(_ship(r, bid, start, vel, **kw))
            recs.append(_goto(r.choice([1, 1, 2]), bid, goal))

        if flavour.startswith("capsule"):
            a, b, rad, axis = _capsule_shape(r, centre)
            if flavour == "capsule_axis":                    # along a coordinate axis, a power-of-two length
                q = r.randrange(3)
                axis = [1.0 if i == q else 0.0 for i in range(3)]
                length = float(2 ** r.randint(4, 8))
                a = [centre[i] - 0.5 * length * axis[i] for i in range(3)]
                b = [centre[i] + 0.5 * length * axis[i] for i in range(3)]
                rad = max(rad, length / 40.0)
            if flavour == "capsule_degenerate":
                if r.random() < 0.5:
                    b = list(a)                              # zero length: a sphere at A
                else:
                    recs.append(_capsule(sids[1], parent, [centre[i] + 30.0 for i in range(3)],
                                         [centre[i] + 90.0 for i in range(3)], 0.0))    # no volume: never filed
            recs.append(_capsule(sids[0], parent, a, b, rad))
            reach = rad + ship_r
            for k, bid in enumerate(ids[:r.randint(1, 2)]):
                t_ax = r.choice([r.uniform(0.0, 1.0), r.uniform(-0.4, 1.4)])
                p_ax = [a[i] + (b[i] - a[i]) * t_ax for i in range(3)]
                nrm = _perp(r, axis)
                if flavour in ("capsule_through", "capsule_degenerate"):
                    start = [p_ax[i] + nrm[i] * (reach + r.uniform(1.0, 80.0)) for i in range(3)]
                    ship_at(bid, start, [p_ax[i] - nrm[i] * 200.0 for i in range(3)])
                elif flavour == "capsule_cap":
                    end, sgn = (b, 1.0) if r.random() < 0.5 else (a, -1.0)
                    start = [end[i] + sgn * axis[i] * (reach + r.uniform(2.0, 60.0)) + nrm[i] * r.uniform(-reach, reach)
                             * r.choice([0.0, 0.5]) for i in range(3)]
                    ship_at(bid, start, [end[i] - sgn * axis[i] * 300.0 for i in range(3)])
                elif flavour == "capsule_graze":
                    start = [p_ax[i] + nrm[i] * (reach + r.uniform(20.0, 100.0)) for i in range(3)]
                    side = _perp(r, nrm)
                    ship_at(bid, start, [p_ax[i] + side[i] * reach * r.uniform(0.9, 1.1) - nrm[i] * 200.0
                                         for i in range(3)])
                elif flavour == "capsule_inside":
                    start = [p_ax[i] + nrm[i] * reach * r.uniform(0.0, 0.95) for i in range(3)]
                    ship_at(bid, start, [start[i] + nrm[i] * r.choice([5.0, 500.0]) for i in range(3)])
                elif flavour == "capsule_backing":             # moving in, thrust out
                    start = [p_ax[i] + nrm[i] * (reach + r.uniform(0.5, 1.0) * speed) for i in range(3)]
                    vel = [-nrm[i] * speed * r.uniform(1.0, 3.0) for i in range(3)]
                    recs.append(_ship(r, bid, start, vel, radius=ship_r, max_vel=speed))
                    recs.append(_goto(1, bid, [start[i] + nrm[i] * 1000.0 for i in range(3)]))
                elif flavour == "capsule_rest":                # no command: at rest beside or inside it
                    start = [p_ax[i] + nrm[i] * reach * r.choice([r.uniform(0.0, 0.95), r.uniform(1.0, 1.3)])
                             for i in range(3)]
                    recs.append(_ship(r, bid, start, radius=ship_r))
                else:                                          # capsule_axis: along the axis, at a cap
                    start = [b[i] + axis[i] * (reach + r.uniform(1.0, 30.0)) * r.choice([1.0, -1.0])
                             for i in range(3)]
                    ship_at(bid, start, [a[i] - axis[i] * 200.0 for i in range(3)], moving=r.random() < 0.5)
        elif flavour.startswith("box"):
            corner, edges = _box_shape(r, centre, big=flavour == "box_huge",
                                       flat=flavour == "box_centre" and r.random() < 0.4)
            recs.append(_box(sids[0], parent, corner, edges))
            half = 0.5 * math.sqrt(sum(sum(e[j] for e in edges) ** 2 for j in range(3)))
            for k, bid in enumerate(ids[:r.randint(1, 2)]):
                out_dir = _unit(r)
                if flavour == "box_face":
                    ax = r.randrange(3)
                    sgn = r.choice([1.0, -1.0])
                    e = edges[ax]
                    ne = math.sqrt(sum(c * c for c in e)) or 1.0
                    out_dir = [sgn * c / ne for c in e] if ne > 1.0 else out_dir
                    target = [centre[i] + 0.5 * sgn * e[i] for i in range(3)]
                    start = [target[i] + out_dir[i] * (ship_r + r.uniform(1.0, 100.0)) for i in range(3)]
                    ship_at(bid, start, [target[i] - out_dir[i] * 300.0 for i in range(3)])
                elif flavour in ("box_edge", "box_corner"):
                    sg = [r.choice([1.0, -1.0]) for _ in range(3)]
                    if flavour == "box_edge":
                        sg[r.randrange(3)] = 0.0
                    target = [centre[i] + 0.5 * sum(sg[q] * edges[q][i] for q in range(3)) for i in range(3)]
                    d0 = [target[i] - centre[i] for i in range(3)]
                    nd = math.sqrt(sum(c * c for c in d0)) or 1.0
                    od = [d0[i] / nd for i in range(3)]
                    start = [target[i] + od[i] * (ship_r + r.uniform(1.0, 80.0)) for i in range(3)]
                    ship_at(bid, start, [target[i] - od[i] * 200.0 + r.uniform(-20.0, 20.0) for i in range(3)])
                elif flavour == "box_inside":
                    start = [centre[i] + r.uniform(-0.3, 0.3) * sum(e[i] for e in edges) for i in range(3)]
                    goal = [start[i] + out_dir[i] * r.choice([5.0, 600.0]) for i in range(3)]
                    if r.random() < 0.3:
                        recs.append(_ship(r, bid, start, radius=ship_r))       # at rest inside
                    else:
                        ship_at(bid, start, goal)
                elif flavour == "box_backing":                 # moving in, thrust out
                    start = [centre[i] + out_dir[i] * (half + ship_r + r.uniform(0.5, 1.0) * speed) for i in range(3)]
                    vel = [-out_dir[i] * speed * r.uniform(1.0, 3.0) for i in range(3)]
                    recs.append(_ship(r, bid, start, vel, radius=ship_r, max_vel=speed))
                    recs.append(_goto(1, bid, [start[i] + out_dir[i] * 1000.0 for i in range(3)]))
                elif flavour == "box_centre":                  # at the centre: the extrication's deep branches
                    c0 = [corner[i] + 0.5 * (edges[0][i] + edges[1][i] + edges[2][i]) for i in range(3)]
                    if r.random() < 0.4:
                        recs.append(_ship(r, bid, c0, radius=ship_r))
                    else:
                        ship_at(bid, c0, [c0[i] + out_dir[i] * 300.0 for i in range(3)], moving=r.random() < 0.6)
                else:                                          # box_huge: near a face of a box wider than a top box
                    start = [centre[i] + out_dir[i] * (half + ship_r + r.uniform(1.0, 100.0)) for i in range(3)]
                    ship_at(bid, start, [centre[i] for i in range(3)], moving=True)
        elif flavour == "cluster":                             # a coarse capsule and fine ones on a face, fast ships
            lv = r.choice([6, 7])
            centre = [_edge_coord(r, lv) if q == r.randrange(3) else _edge_coord(r, lv) + r.uniform(-20.0, 20.0)
                      for q in range(3)]
            axis = _unit(r)
            big = r.uniform(150.0, 500.0)
            off = _perp(r, axis)
            mid = [centre[i] + off[i] * (big + r.uniform(-30.0, 10.0)) for i in range(3)]
            half_len = r.uniform(50.0, 400.0)
            recs.append(_capsule(sids[0], r.choice([parent, 0]), [mid[i] - axis[i] * half_len for i in range(3)],
                                 [mid[i] + axis[i] * half_len for i in range(3)], big))
            for j in range(1, r.randint(2, 4)):
                c = [centre[i] + r.uniform(-40.0, 40.0) for i in range(3)]
                if r.random() < 0.5:
                    a, b, rad, _ = _capsule_shape(r, c)
                    recs.append(_capsule(sids[j], r.choice([parent, 0, sids[0]]), a, b, min(rad, 20.0)))
                else:
                    corner, edges = _box_shape(r, c)
                    recs.append(_box(sids[j], r.choice([parent, 0, sids[0]]), corner,
                                     [[x * 0.25 for x in e] for e in edges]))
            for bid in ids[:r.randint(1, 2)]:
                start = [centre[i] + r.uniform(-30.0, 30.0) for i in range(3)]
                fast = r.uniform(80.0, 300.0)
                d0 = _unit(r)
                recs.append(_ship(r, bid, start, [d0[i] * fast for i in range(3)], radius=r.uniform(2.0, 30.0),
                                  max_vel=fast))
                recs.append(_goto(1, bid, [start[i] + d0[i] * 2000.0 for i in range(3)]))
        elif flavour in ("field", "faces", "ball_and_static"):
            if flavour == "faces":
                lv = r.randrange(4, 8)
                centre = [_edge_coord(r, lv) for _ in range(3)]
            nstat = r.randint(2, 4) if flavour != "ball_and_static" else 1
            for j in range(nstat):
                c = [centre[i] + r.uniform(-250.0, 250.0) for i in range(3)]
                if r.random() < 0.5:
                    a, b, rad, _ = _capsule_shape(r, c)
                    recs.append(_capsule(sids[j], r.choice([parent, 0, sids[(j + 1) % 5]]), a, b, rad))
                else:
                    corner, edges = _box_shape(r, c)
                    recs.append(_box(sids[j], r.choice([parent, 0, sids[(j + 1) % 5]]), corner, edges))
            for j in range(r.randint(0, 2) if flavour != "ball_and_static" else 1):
                recs.append(_sphere(r, ids[5 - j], [centre[i] + r.uniform(-200.0, 200.0) for i in range(3)],
                                    r.uniform(5.0, 80.0)))
            for bid in ids[:r.randint(1, 3)]:
                start = [centre[i] + r.uniform(-450.0, 450.0) for i in range(3)]
                goal = [2.0 * centre[i] - start[i] + r.uniform(-60.0, 60.0) for i in range(3)]
                ship_at(bid, start, goal, radius=r.choice([2.0, r.uniform(1.0, 30.0)]))
                for _ in range(r.randint(0, 2)):
                    k = r.randint(2, steps)
                    recs.append(r.choice([fc._k(k, fc.SF_OP, bid, d=fc._xyz(rr=r.uniform(0.0, 1.0))),
                                          fc._k(k, fc.STOP_OP, bid),
                                          fc._k(k, fc.DIR_OP, bid, d=tuple(_unit(r)))]))
        else:                                                  # two_statics, three_statics: a ship touching several
            a, b, rad, axis = _capsule_shape(r, centre)
            recs.append(_capsule(sids[0], parent, a, b, rad))
            nrm = _perp(r, axis)
            gap = rad * r.uniform(0.3, 1.0)
            for j in range(1, 3 if flavour == "three_statics" else 2):
                side = nrm if j == 1 else [-x for x in nrm]
                other = [centre[i] + side[i] * gap * 2.0 for i in range(3)]
                if r.random() < 0.5:
                    a2, b2, rad2, _ = _capsule_shape(r, other)
                    recs.append(_capsule(sids[j], r.choice([parent, 0, sids[0]]), a2, b2, max(rad2, gap * 1.2)))
                else:
                    corner, edges = _box_shape(r, other)
                    recs.append(_box(sids[j], r.choice([parent, 0, sids[0]]), corner, edges))
            for bid in ids[:r.randint(1, 2)]:
                start = [centre[i] + nrm[i] * gap * r.uniform(0.5, 1.5) + r.uniform(-0.2, 0.2) * rad for i in range(3)]
                if r.random() < 0.5:
                    recs.append(_ship(r, bid, start, radius=max(ship_r, gap)))
                else:
                    ship_at(bid, start, [start[i] + _unit(r)[i] * 300.0 for i in range(3)], radius=max(ship_r, gap))
        fc.case(out, r, flavour, recs, steps, master=False, tick=fc._tick(r), friction=fc._friction(r))
    return out


# ---- the batched run ----

def _ball_dict(b: dict) -> dict:
    return {"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"], "agility": b["agility"],
            "speed_fraction": b["sf"], "pos": b["pos"], "vel": b["vel"], "free": b["free"], "massive": b["massive"],
            "junk": b["junk"]}


def statics_of(c: dict) -> list:
    """A scenario's static collidables as VecCollidePark.load_statics takes them (AddCapsule's and AddOrientedBox's
    arguments; the id as given, a client's view keeping it)."""
    out = []
    for rec in c["records"]:
        if rec[0] != "K":
            continue
        _, step, op, a, bb, cc, w, d = rec
        if op == fc.CAPSULE_OP:
            out.append({"kind": "capsule", "id": a, "parent": bb, "a": d[0:3], "b": d[3:6], "radius": d[6]})
        elif op == fc.OBOX_OP:
            d = list(d) + [0.0] * 12
            out.append({"kind": "box", "id": a, "parent": bb, "corner": d[0:3], "lx": d[3:6], "ly": d[6:9],
                        "lz": d[9:12]})
    return out


def _commands(park, slot: list, recs: list) -> None:
    """A step's commands (full_cases' records), each its own masked call, in record order per world."""
    dev = park.dev
    for w, rec in recs:
        _, _, op, a, bb, cc, ww, d = rec
        d = list(d) + [0.0] * 4
        mask = torch.zeros(park.W, park.N, dtype=torch.bool, device=dev)
        mask[w, slot[w][a]] = True
        v = torch.zeros(park.W, park.N, 3, dtype=torch.float64, device=dev)
        v[w, slot[w][a]] = torch.tensor(d[:3], dtype=torch.float64)
        if op == fc.GOTO_OP:
            park.goto_point(mask, v)
        elif op == fc.DIR_OP:
            park.goto_direction(mask, v)
        elif op == fc.STOP_OP:
            park.stop(mask)
        elif op == fc.SF_OP:
            park.set_speed_fraction(mask, torch.full((park.W, park.N), d[3], dtype=torch.float64, device=dev))
        else:
            raise ValueError(f"op {op} is not a batched command")


def batched(cs: list[dict], device: str, contact_slots=None, kernel: str = "torch", rows=None, after_tick=None,
            parks: list | None = None, **kw) -> dict:
    """Run the scenarios batched with their static collidables, one park per (friction, tick; with rows="free", per
    free-ball count); rows {(case, step, id): state} in collide_cases' terms. `parks` collects the parks (the tests read
    their static records)."""
    from exact_ballpark.vec_collide import VecCollidePark
    groups = defaultdict(list)
    nfree = lambda c: sum(1 for rec in c["records"] if rec[0] == "B" and rec[2]["free"])      # noqa: E731
    for c in cs:
        groups[(c["friction"], c["tick"], nfree(c) if rows == "free" else None)].append(c)
    out = {}
    for (friction, tick, free), g in groups.items():
        n = max(sum(1 for rec in c["records"] if rec[0] == "B") for c in g)
        s = max(len(statics_of(c)) for c in g)
        if rows is not None:
            kw["rows"] = max(1, free) if free is not None else rows
        park = VecCollidePark(len(g), n, friction=friction, tick_interval_ms=tick, contact_slots=contact_slots,
                              device=device, kernel=kernel, statics=s, **kw)
        worlds = [list({rec[2]["id"]: _ball_dict(rec[2]) for rec in c["records"] if rec[0] == "B"}.values())
                  for c in g]
        park.load(worlds, [statics_of(c) for c in g])
        slot = [{int(i): j for j, i in enumerate(ids) if i} for ids in park.id.cpu().tolist()]
        by_step = defaultdict(list)
        for w, c in enumerate(g):
            for rec in c["records"]:
                if rec[0] == "K" and rec[1] > 0:
                    by_step[rec[1]].append((w, rec))
        _collect(out, g, park, slot, 0)
        for st in range(1, max(c["steps"] for c in g) + 1):
            _commands(park, slot, by_step.get(st, []))
            park.evolve()
            if after_tick is not None:
                after_tick(park, st)
            _collect(out, g, park, slot, st)
        park.check()
        if parks is not None:
            parks.append((g, park))
    return out


# ---- the reference and the compiled oracle, in the batched rows' terms ----

def rows_of(s_rows: list) -> dict:
    """Ball rows in full_cases' S layout (the reference's or the compiled oracle's) as {(case, step, id): state}."""
    idx = {f: i for i, f in enumerate(fc.S_FIELDS)}
    out = {}
    for row in s_rows:
        g = lambda *names: [row[idx[nm]] for nm in names]      # noqa: E731
        out[(row[0], row[1], row[2])] = {
            "pos": g("px", "py", "pz"), "vel": g("vx", "vy", "vz"), "old_pos": g("opx", "opy", "opz"),
            "old_vel": g("ovx", "ovy", "ovz"), "goto": g("gx", "gy", "gz"), "last_g": g("ax", "ay", "az"),
            "last_c": g("cx", "cy", "cz"), "last_collision": row[idx["last_collision"]], "mode": row[idx["mode"]],
            "level": row[idx["box_level"]], "key": g("box_ix", "box_iy", "box_iz"),
            "mod": g("mod_x", "mod_y", "mod_z"), "hits": tuple(row[idx["collisions"]]),
            "speed_fraction": row[idx["speed_fraction"]], "boxes": row[idx["boxes"]]}
    return out


def reference(cs: list[dict]) -> tuple:
    """The reference (full_cases.reference: FullBallpark) on every scenario: (ball rows, the static collidables' rows
    {(case, id): (level, boxes)} at step 0, every row kind as full_cases keeps them, the undefined cases)."""
    rows, undefined = fc.run_reference(cs)
    statics = {}
    idx = {f: i for i, f in enumerate(fc.T_FIELDS)}
    for row in rows.get("T", []):
        if row[1] == 0:
            statics[(row[0], row[idx["id"]])] = tuple(row[idx["boxes"]])
    return rows_of(rows.get("S", [])), statics, rows, set(undefined)


def park_statics(parks: list) -> dict:
    """The batched parks' static filings {(case, id): boxes (level, ix, iy, iz), sorted}."""
    out = {}
    for g, park in parks:
        ints, keys, ex = park.s_int.cpu().tolist(), park.s_keys.cpu().tolist(), park.s_exists.cpu().tolist()
        for w, c in enumerate(g):
            for j in range(park.S):
                if ex[w][j]:
                    i = ints[w][j]
                    out[(c["id"], i[1])] = tuple((i[3], *keys[w][j][b]) for b in range(i[4]))
    return out


def counts(rows: dict) -> dict:
    """What a run reached: ball-steps with a static collidable on the collision list, with two or more entries, and
    rows in all."""
    with_static = sum(1 for r in rows.values() if r["hits"])
    return {"rows": len(rows), "ball_steps_with_hits": with_static,
            "ball_steps_with_two_or_more": sum(1 for r in rows.values() if len(r["hits"]) > 1)}

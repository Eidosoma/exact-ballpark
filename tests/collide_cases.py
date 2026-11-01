# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the batched collisions (exact_ballpark.vec_collide), generated from a seed, in the records of
tests/space_cases.py (AddBall at step 0, commands at later steps), so that one scenario runs in the scalar reference
(exact_ballpark.space, a client's view), in the batched park and in Destiny's own code compiled verbatim
(conformance/build_space.py).

Flavours: approaching pairs and groups (head-on, glancing, crossing), overlapping and coincident balls (Potential's
recursion and its light-ball rule), fixed spheres as walls and as a field of obstacles, several contacts of one ball in
one tick, near misses just outside the contact distance, chains, fast balls, balls on and near partition box faces at
several levels (the candidates' order), space junk, cloaked and non-massive neighbours, and racers: ships flying GOTO
through a field of fixed spheres with throttle and stop commands. Ticks of 0.25 to 2 s, Destiny's friction and others.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import torch

from space_cases import CLOAK_OP, DIR_OP, GOTO_OP, SF_OP, STOP_OP, _ball, _edge, _ids, _k, _unit
from exact_ballpark.space import LEVEL_WIDTH

FLAVOURS = ("approach", "approach", "glancing", "overlap", "coincident", "light", "fixed", "field", "multi",
            "near_miss", "chain", "fast", "faces", "junk", "racers", "racers")


def _point_near(r: random.Random, c: list, spread: float) -> list:
    return [c[i] + r.uniform(-spread, spread) for i in range(3)]


def cases(seed: int, n: int = 240, max_balls: int = 8, iterative: bool = False) -> list[dict]:
    r = random.Random(7000 + seed + (500 if iterative else 0))
    out: list[dict] = []

    def case(kind: str, records: list, steps: int, tick: int, friction: float) -> None:
        out.append({"id": len(out) + 1, "kind": kind, "friction": friction, "tick": tick, "steps": steps,
                    "new_orbit": False, "iterative": iterative, "master": False, "seed": r.randrange(1, 2 ** 31),
                    "start": r.randrange(0, 100000), "records": sorted(records, key=lambda c: (c[1], c[0] != "B"))})

    for _ in range(n):
        flavour = r.choice(FLAVOURS)
        nb = r.randint(2, max_balls)
        ids = _ids(r, nb)
        centre = [r.choice([0.0, r.uniform(-2e4, 2e4), _edge(r, r.randrange(3, 8))]) for _ in range(3)]
        recs: list = []
        steps = r.randint(3, 16)
        tick = r.choice([250, 500, 1000, 1000, 1000, 2000])
        friction = r.choice([1.0e6] * 5 + [10 ** r.uniform(5.0, 7.0)])
        for j, bid in enumerate(ids):
            radius = r.choice([2.0, r.uniform(1.0, 60.0), r.uniform(60.0, 400.0)])
            mass = 10 ** r.uniform(5.0, 9.0)
            free, massive = True, r.random() < 0.92
            vel = [0.0, 0.0, 0.0]
            if flavour in ("overlap", "light", "multi"):
                pos = _point_near(r, centre, radius)
            elif flavour == "coincident":
                pos = list(centre) if j < 3 else _point_near(r, centre, radius * 3)
            elif flavour == "chain":
                pos = [centre[0] + j * 2.0 * radius * r.uniform(0.9, 1.2), centre[1], centre[2]]
            elif flavour == "faces":
                level = r.randrange(4, 8)
                pos = [_edge(r, level) if r.random() < 0.7 else centre[i] for i in range(3)]
                radius = r.choice([r.uniform(1.0, 60.0), r.uniform(60.0, 900.0), r.uniform(900.0, 9000.0)])
            elif flavour in ("field", "racers") and j >= 2:
                pos = _point_near(r, centre, 3000.0)
            else:
                pos = _point_near(r, centre, radius * r.uniform(1.5, 8.0))
            if flavour == "light" and j == 0:
                mass = 10 ** r.uniform(2.0, 4.5)
            if flavour in ("fixed", "field", "racers") and j >= (1 if flavour == "fixed" else 2):
                free = r.random() < (0.5 if flavour == "fixed" else 0.15)
                radius = r.uniform(50.0, 900.0) if not free else radius
            if flavour == "fast":
                vel = [r.uniform(-1500.0, 1500.0) for _ in range(3)]
            elif flavour == "glancing":
                vel = [r.uniform(-80.0, 80.0) for _ in range(3)]
            elif r.random() < 0.4:
                vel = [r.uniform(-50.0, 50.0) for _ in range(3)]
            junk = flavour == "junk" and r.random() < 0.5
            recs.append(_ball(r, bid, pos=pos, radius=radius, mass=mass, free=free, massive=massive, junk=junk,
                              interactive=True, agility=r.uniform(0.05, 2.0), vel=vel,
                              max_vel=r.choice([r.uniform(5.0, 400.0), r.uniform(50.0, 1500.0)])))
            if r.random() < 0.04:
                recs.append(_k(0, CLOAK_OP, bid, w=1))
        # commands: aim at the group, at each other, through the field; throttle and stop
        for j, bid in enumerate(ids):
            if flavour in ("racers", "field") and j < 2:
                u = _unit(r)
                recs.append(_k(1, GOTO_OP, bid, x=centre[0] + u[0] * 6000.0, y=centre[1] + u[1] * 6000.0,
                               z=centre[2] + u[2] * 6000.0))
                for _ in range(r.randint(0, 3)):
                    k = r.randint(2, steps)
                    what = r.choice(["goto", "dir", "sf", "stop"])
                    if what == "goto":
                        t = _point_near(r, centre, 4000.0)
                        recs.append(_k(k, GOTO_OP, bid, x=t[0], y=t[1], z=t[2]))
                    elif what == "dir":
                        d = _unit(r)
                        recs.append(_k(k, DIR_OP, bid, x=d[0], y=d[1], z=d[2]))
                    elif what == "sf":
                        recs.append(_k(k, SF_OP, bid, rr=r.choice([0.0, 0.3, 1.0, r.uniform(0.0, 1.2)])))
                    else:
                        recs.append(_k(k, STOP_OP, bid))
            elif flavour == "near_miss" and j == 0:
                # aim past the second ball by a hair more than the two radii
                recs.append(_k(1, GOTO_OP, bid, x=centre[0] + 1e5, y=centre[1], z=centre[2]))
            elif flavour == "chain" and j == 0:
                recs.append(_k(1, GOTO_OP, bid, x=centre[0] + 1e5, y=centre[1], z=centre[2]))
            elif r.random() < 0.7:
                t = _point_near(r, centre, 20.0)
                recs.append(_k(r.choice([1, 1, 2]), GOTO_OP, bid, x=t[0], y=t[1], z=t[2]))
            if r.random() < 0.15:
                recs.append(_k(r.randint(2, steps), STOP_OP, bid))
            if r.random() < 0.1:
                recs.append(_k(r.randint(1, steps), SF_OP, bid, rr=r.uniform(0.0, 1.0)))
        if flavour in ("fixed", "field", "racers"):
            _aim_at_obstacles(r, recs, ids, flavour, steps)
        if flavour == "near_miss":
            # place the second ball just off the first's line, at a gap of a few millimetres to metres
            a = next(rec for rec in recs if rec[0] == "B" and rec[2]["id"] == ids[0])[2]
            b = next(rec for rec in recs if rec[0] == "B" and rec[2]["id"] == ids[1])[2]
            gap = r.choice([1e-3, 0.01, 0.1, 1.0, r.uniform(0.0, 5.0)])
            off = f32_sum(a["radius"], b["radius"]) + gap
            b["pos"] = [a["pos"][0] + r.uniform(200.0, 2000.0), a["pos"][1] + off, a["pos"][2]]
            b["free"] = r.random() < 0.5
        case("collide_" + flavour, recs, steps, tick, friction)
    return out


def _aim_at_obstacles(r: random.Random, recs: list, ids: list, flavour: str, steps: int) -> None:
    """Fixed spheres as obstacles: the free balls start outside them and are sent at one (its centre, or off it by up
    to its radius), from a few metres to a few kilometres away."""
    balls = {rec[2]["id"]: rec[2] for rec in recs if rec[0] == "B"}
    fixed = [b for b in balls.values() if not b["free"]]
    if not fixed:
        return
    for bid in ids:
        b = balls[bid]
        if not b["free"] or r.random() < 0.2:
            continue
        o = r.choice(fixed)
        u = _unit(r)
        far = r.choice([r.uniform(1.0, 60.0), r.uniform(60.0, 900.0), r.uniform(900.0, 5000.0)])
        d = o["radius"] + b["radius"] + far
        b["pos"] = [o["pos"][i] + u[i] * d for i in range(3)]
        b["vel"] = r.choice([[0.0, 0.0, 0.0], [-u[i] * r.uniform(10.0, 600.0) for i in range(3)]])
        b["max_vel"] = r.uniform(50.0, 1500.0) if flavour == "racers" else b["max_vel"]
        off = [r.uniform(-1.0, 1.0) * o["radius"] for _ in range(3)] if r.random() < 0.5 else [0.0, 0.0, 0.0]
        recs[:] = [rec for rec in recs if not (rec[0] == "K" and rec[1] == 1 and rec[3] == bid and rec[2] == GOTO_OP)]
        recs.append(_k(1, GOTO_OP, bid, x=o["pos"][0] + off[0], y=o["pos"][1] + off[1], z=o["pos"][2] + off[2]))


def f32_sum(a: float, b: float) -> float:
    """Two radii as AddBall stores them (floats), summed in double, as Potential does."""
    import struct
    f = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    return f(a) + f(b)


# ---- the batched run ----

def _ball_dict(b: dict) -> dict:
    return {"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"], "agility": b["agility"],
            "speed_fraction": b["sf"], "pos": b["pos"], "vel": b["vel"], "free": b["free"], "massive": b["massive"],
            "junk": b["junk"]}


def batched(cs: list[dict], device: str, contact_slots=None, park_cls=None, undefined: set | None = None,
            after_tick=None, rows=None, **kw) -> dict:
    """Run the scenarios batched, one park per (friction, tick, method; with rows="free", per free-ball count, the
    park's rows that many: the lever); rows {(case, step, id): state}. The cases whose world the park marks
    undefined (the iterative method's CalculateEnd on a full vector) are added to `undefined` when given."""
    from exact_ballpark.vec_collide import VecCollidePark
    park_cls = park_cls or VecCollidePark
    groups = defaultdict(list)
    nfree = lambda c: sum(1 for rec in c["records"] if rec[0] == "B" and rec[2]["free"])      # noqa: E731
    for c in cs:
        groups[(c["friction"], c["tick"], bool(c.get("iterative")), nfree(c) if rows == "free" else None)].append(c)
    out = {}
    for (friction, tick, iterative, free), g in groups.items():
        n = max(sum(1 for rec in c["records"] if rec[0] == "B") for c in g)
        if rows is not None:
            kw["rows"] = max(1, free) if free is not None else rows
        park = park_cls(len(g), n, friction=friction, tick_interval_ms=tick, iterative=iterative,
                        contact_slots=contact_slots, device=device, **kw)
        worlds = []
        for c in g:
            balls = {rec[2]["id"]: _ball_dict(rec[2]) for rec in c["records"] if rec[0] == "B"}
            for rec in c["records"]:
                if rec[0] == "K" and rec[1] == 0 and rec[2] == CLOAK_OP:
                    balls[rec[3]]["cloaked"] = rec[5]
            worlds.append(list(balls.values()))
        park.load(worlds)
        slot = [{int(i): j for j, i in enumerate(ids) if i} for ids in park.id.cpu().tolist()]
        by_step = defaultdict(list)
        for w, c in enumerate(g):
            for rec in c["records"]:
                if rec[0] == "K" and rec[1] > 0:
                    by_step[rec[1]].append((w, rec))
        _collect(out, g, park, slot, 0)
        for s in range(1, max(c["steps"] for c in g) + 1):
            _commands(park, slot, by_step.get(s, []))
            park.evolve()
            if after_tick is not None:
                after_tick(park, s)
            _collect(out, g, park, slot, s)
        if undefined is not None:
            undefined.update(c["id"] for c, u in zip(g, park.undefined.cpu().tolist()) if u)
        park.check()
    return out


def _commands(park, slot: list, recs: list) -> None:
    """The step's commands, in record order per world (each command its own masked call, as the scalar code applies
    them one by one)."""
    dev = park.dev
    for w, rec in recs:
        _, _, op, a, bb, ww, x, y, z, rr = rec
        mask = torch.zeros(park.W, park.N, dtype=torch.bool, device=dev)
        mask[w, slot[w][a]] = True
        v = torch.zeros(park.W, park.N, 3, dtype=torch.float64, device=dev)
        v[w, slot[w][a]] = torch.tensor([x, y, z], dtype=torch.float64)
        if op == GOTO_OP:
            park.goto_point(mask, v)
        elif op == DIR_OP:
            park.goto_direction(mask, v)
        elif op == STOP_OP:
            park.stop(mask)
        elif op == SF_OP:
            park.set_speed_fraction(mask, torch.full((park.W, park.N), rr, dtype=torch.float64, device=dev))
        else:
            raise ValueError(f"op {op} is not a batched command")


def _collect(out: dict, g: list, park, slot: list, s: int) -> None:
    host = {k: getattr(park, k).cpu().tolist() for k in ("pos", "vel", "old_pos", "old_vel", "goto", "last_g",
                                                         "last_c", "last_collision", "mode", "level", "key", "mod",
                                                         "hits", "hits_n", "speed_fraction")}
    for w, c in enumerate(g):
        if s > c["steps"]:
            continue
        for bid, j in slot[w].items():
            rows = j < len(host["hits"][w])                     # the collision lists are the park's rows' (fixed: none)
            row = {k: host[k][w][j] for k in host if k not in ("hits", "hits_n") or rows}
            row["hits"] = tuple(row["hits"][:row.pop("hits_n")]) if rows else ()
            out[(c["id"], s, bid)] = row


def boxes_of(level: int, key: list, mod: list) -> tuple:
    """A filing's boxes as the reference lists them (level, ix, iy, iz), sorted by key."""
    opts = [sorted({key[q], key[q] + mod[q]}) for q in range(3)]
    return tuple(sorted((level, x, y, z) for x in opts[0] for y in opts[1] for z in opts[2]))


def rows_of(s_rows: list) -> dict:
    """Ball rows in space_cases' S layout (the reference's or the compiled oracle's) as {(case, step, id): state} in
    the batched rows' terms."""
    import space_cases as sc
    idx = {f: i for i, f in enumerate(sc.S_FIELDS)}
    out = {}
    for row in s_rows:
        g = lambda *names: [row[idx[n]] for n in names]
        out[(row[0], row[1], row[2])] = {
            "pos": g("px", "py", "pz"), "vel": g("vx", "vy", "vz"), "old_pos": g("opx", "opy", "opz"),
            "old_vel": g("ovx", "ovy", "ovz"), "goto": g("gx", "gy", "gz"), "last_g": g("ax", "ay", "az"),
            "last_c": g("cx", "cy", "cz"), "last_collision": row[idx["last_collision"]], "mode": row[idx["mode"]],
            "level": row[idx["box_level"]], "key": g("box_ix", "box_iy", "box_iz"), "mod": g("mod_x", "mod_y",
                                                                                             "mod_z"),
            "hits": tuple(row[idx["collisions"]]), "speed_fraction": row[idx["speed_fraction"]],
            "boxes": row[idx["boxes"]]}
    return out


def reference_rows(cs: list[dict]) -> tuple:
    """The scalar reference on every scenario (space_cases.reference): (rows, the cases where Destiny's result is
    undefined, which the reference refuses and which are left out)."""
    import space_cases as sc
    rows, undefined = sc.run_reference(cs)
    return rows_of(rows.get("S", [])), set(undefined)


def oracle_rows(cs: list[dict]) -> dict:
    """Destiny's own code compiled verbatim (conformance/build_space.py) on the same scenarios."""
    import space_cases as sc
    return rows_of(sc.oracle(cs).get("S", []))


def _norm(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


FLOAT_FIELDS = ("pos", "vel", "old_pos", "old_vel", "goto", "last_g", "last_c", "last_collision", "speed_fraction")
INT_FIELDS = ("mode", "level", "key", "mod", "hits")


def differences(want: dict, got: dict) -> list:
    """(case, step, id, field) for every value that differs (doubles by their bits, NaNs equal)."""
    import space_cases as sc
    bad = []
    if set(want) != set(got):
        return [("rows", len(want), len(got), sorted(set(want) ^ set(got))[:5])]
    for key, w in want.items():
        gv = got[key]
        for f in FLOAT_FIELDS:
            a, b = w[f], gv[f]
            a, b = (a, b) if isinstance(a, list) else ([a], [b])
            if not all(sc.same(float(x), float(y)) for x, y in zip(a, b)):
                bad.append((*key, f))
        for f in INT_FIELDS:
            if _norm(w[f]) != _norm(gv[f]):
                bad.append((*key, f))
        if "boxes" in w and tuple(w["boxes"]) != boxes_of(gv["level"], gv["key"], gv["mod"]):
            bad.append((*key, "boxes"))
    return bad

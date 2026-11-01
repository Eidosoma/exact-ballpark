# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the batched minis and force fields (the collision park's leftovers, COL-713's minis
and MOV-737's force fields in exact_ballpark.vec_collide's park, its torch path and the kernel), generated from a seed
in the records of tests/full_cases.py (AddBall, AddCapsule, AddOrientedBox, AddMiniBall, AddMiniCapsule, AddMiniBox
and SetBallHarmonic at step 0; GOTO, direction, stop, speed fraction and SetBallHarmonic at later steps), so that one
scenario runs in the scalar reference (exact_ballpark.full's FullBallpark, a client's view), in the batched park and in
Destiny's own code compiled verbatim (conformance/build_full.py, full_cases.HARNESS).

Flavours: fixed parents (massive or not) with miniballs, their calls interleaved between parents so that the local ids
interleave, ships flown through them; two parents whose miniballs are packed in one box with the parents' ids against
the calls' order, a fast wide ship through them (BallSortComparer's owner-first order is what the collision list
shows); parents with minicapsules and miniboxes (shapes as tests/static_cases.py's, offsets from the parent), a
minicapsule of no positive radius (refused, no id) and a free parent (its minis unmade, no id) among them; mixes of
minis, explicit static collidables and fixed spheres with racers flying GOTO, throttle, stop and direction commands; a
force field (SetBallHarmonic, fixed or drifting free) with ships whose harmonic, corporation or alliance matches it, or
which pass every field (-2), or match nothing, or carry -1 against a field of -1; fields turned on, off and changed
during a run, a field sent to a point (leaving FIELD); two fields; a field among minis; a field around a cluster of
miniballs and fixed spheres at the ships' level (a passed field between two contacts in the walk's order). Ticks of 0.25
to 2 s, Destiny's friction and others.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import torch

import collide_cases as cc
import full_cases as fc
import static_cases as stc

FLAVOURS = ("miniballs", "miniballs", "minis_order", "minis_order", "minis_order", "minicapsules", "minicapsules",
            "miniboxes", "miniboxes", "minis_mixed", "minis_mixed", "field_fixed", "field_fixed", "field_free",
            "field_toggle", "field_toggle", "two_fields", "field_and_minis", "field_shell", "field_shell")
MINI_OPS = (fc.MINIBALL_OP, fc.MINICAPSULE_OP, fc.MINIBOX_OP)


def _harmonic(step: int, bid: int, h: int, field: bool, corp: int, alliance: int) -> tuple:
    """SetBallHarmonic(id, harmonic, corporation, alliance, isField) as full_cases' record."""
    return fc._k(step, fc.HARMONIC_OP, bid, h, w=int(field), d=(float(corp), float(alliance)))


def _through(r: random.Random, centre, near: float, far: float) -> tuple:
    """A start `near` to `far` metres from the centre and the point opposite it (a flight through the centre), and the
    distance."""
    u = stc._unit(r)
    d = r.uniform(near, far)
    start = [centre[i] + u[i] * d for i in range(3)]
    goal = [centre[i] - u[i] * (d + 100.0) + r.uniform(-30.0, 30.0) for i in range(3)]
    return start, goal, d


def cases(seed: int, n: int = 80) -> list[dict]:
    """`n` scenarios from `seed`: full_cases' scenario dicts (a client's view, the gradient method)."""
    r = random.Random(91000 + seed)
    out: list[dict] = []
    for _ in range(n):
        flavour = r.choice(FLAVOURS)
        centre = [r.choice([0.0, r.uniform(-2.0e4, 2.0e4)]) for _ in range(3)]
        recs: list = []
        steps = r.randint(6, 20)
        tick = fc._tick(r)
        ids = stc._ids(r, 10)
        speed = r.choice([10.0, r.uniform(20.0, 120.0), r.uniform(120.0, 400.0)])

        def ship(bid, start, goal, radius=None, step=None, dist=None, **kw):
            if dist is not None:                         # fast enough to get there in about half the run
                kw.setdefault("max_vel", max(speed, dist / (0.5 * steps * tick * 0.001)))
            kw.setdefault("max_vel", speed)
            moving = r.random() < 0.7
            vel = [0.0, 0.0, 0.0]
            if moving:                                   # already on its way (a heavy ship accelerates slowly)
                d0 = [goal[i] - start[i] for i in range(3)]
                nd = math.sqrt(sum(c * c for c in d0)) or 1.0
                vel = [d0[i] / nd * kw["max_vel"] * r.uniform(0.5, 1.0) for i in range(3)]
            radius = r.choice([2.0, r.uniform(1.0, 20.0), r.uniform(20.0, 70.0)]) if radius is None else radius
            recs.append(stc._ship(r, bid, start, vel, radius=radius, **kw))
            recs.append(stc._goto(r.choice([1, 1, 2]) if step is None else step, bid, goal))

        def parent(pid, pos, free=False):
            if r.random() < 0.5:
                recs.append(stc._sphere(r, pid, pos, r.uniform(20.0, 200.0)))         # fixed, massive
            else:
                recs.append(fc._parent(pid, pos=pos, radius=r.uniform(20.0, 300.0), mass=10 ** r.uniform(6.0, 10.0),
                                       free=free))                                  # fixed, not massive (or free)

        def mini(pid, kind, off_centre):
            if kind == "ball":
                return fc._k(0, fc.MINIBALL_OP, pid, d=(*off_centre, r.choice([r.uniform(2.0, 60.0), 25.0])))
            if kind == "capsule":
                a, b, rad, _ = stc._capsule_shape(r, off_centre)
                if r.random() < 0.08:
                    rad = r.choice([0.0, -3.0, 1e-46])                              # refused: no positive float
                return fc._k(0, fc.MINICAPSULE_OP, pid, d=(*a, *b, rad))
            corner, edges = stc._box_shape(r, off_centre)
            return fc._k(0, fc.MINIBOX_OP, pid, d=(*corner, *edges[0], *edges[1], *edges[2]))

        def harmonics(bid, field_h, field_c, field_a, step=0):
            """A ship's SetBallHarmonic against a field's values: no call (all -1), none matching, the pass, or a
            match by harmonic, corporation or alliance."""
            match = r.choice(["default", "none", "pass", "harmonic", "corporation", "alliance", "minus_one"])
            if match == "default":
                return
            hb = {"none": 1, "pass": -2, "harmonic": field_h, "corporation": 2, "alliance": 2,
                  "minus_one": -1}[match]
            cb = field_c if match == "corporation" else (-1 if match == "minus_one" else 7)
            ab = field_a if match == "alliance" else (-1 if match == "minus_one" else 8)
            recs.append(_harmonic(step, bid, hb, False, cb, ab))

        if flavour in ("miniballs", "minis_order", "minicapsules", "miniboxes", "minis_mixed", "field_and_minis"):
            n_par = 2 if flavour == "minis_order" else r.randint(1, 2 if flavour != "minis_mixed" else 3)
            pids = ids[:n_par]
            if flavour == "minis_order":                     # the parents' ids against the calls' order
                pids = sorted(pids, reverse=r.random() < 0.5)
            calls = []
            targets = [list(centre)]                             # what the ships fly through: the minis, the centre
            for k, pid in enumerate(pids):
                spread = 60.0 if flavour == "minis_order" else 200.0
                ppos = [centre[i] + r.uniform(-spread, spread) for i in range(3)]
                free = flavour == "minis_mixed" and r.random() < 0.2
                parent(pid, ppos, free=free)
                if flavour in ("miniballs", "field_and_minis"):
                    kinds = ["ball"] * r.randint(1, 4)
                elif flavour == "minis_order":
                    kinds = ["ball"] * r.randint(2, 3)
                elif flavour == "minicapsules":
                    kinds = ["capsule"] * r.randint(1, 3)
                elif flavour == "miniboxes":
                    kinds = ["box"] * r.randint(1, 3)
                else:
                    kinds = [r.choice(["ball", "capsule", "box"]) for _ in range(r.randint(1, 4))]
                for kind in kinds:
                    if flavour == "minis_order":                 # packed around the centre, one radius
                        off = [centre[i] - ppos[i] + r.uniform(-25.0, 25.0) for i in range(3)]
                    else:
                        off = [r.uniform(-150.0, 150.0) for _ in range(3)]
                    calls.append((k, mini(pid, kind, off)))
                    targets.append([ppos[i] + off[i] for i in range(3)])
            if flavour == "minis_order":                         # alternate between the parents
                seen: dict = defaultdict(int)
                keyed = []
                for kc in calls:
                    keyed.append((seen[kc[0]], kc[0], kc))
                    seen[kc[0]] += 1
                calls = [kc for _, _, kc in sorted(keyed, key=lambda t: (t[0], t[1]))]
            else:
                r.shuffle(calls)                                 # interleaved: the local ids interleave
            recs.extend(rec for _, rec in calls)
            if flavour == "minis_mixed" and r.random() < 0.3:    # a call for a ball not in the park: nothing
                recs.append(fc._k(0, fc.MINIBALL_OP, ids[9], d=(1.0, 2.0, 3.0, 10.0)))
            if flavour == "minis_mixed":
                for j in range(r.randint(0, 2)):                 # explicit static collidables and fixed spheres
                    c = [centre[i] + r.uniform(-250.0, 250.0) for i in range(3)]
                    sid = 5000 + j
                    if r.random() < 0.5:
                        a, b, rad, _ = stc._capsule_shape(r, c)
                        recs.append(stc._capsule(sid, r.choice([0, pids[0]]), a, b, rad))
                    else:
                        corner, edges = stc._box_shape(r, c)
                        recs.append(stc._box(sid + 100, r.choice([0, pids[0]]), corner, edges))
                for j in range(r.randint(0, 2)):
                    recs.append(stc._sphere(r, ids[4 + j], [centre[i] + r.uniform(-250.0, 250.0) for i in range(3)],
                                            r.uniform(10.0, 80.0)))
            if flavour == "field_and_minis":
                h, cp, al = r.choice([5, 9]), r.choice([-1, 3, 3]), r.choice([-1, 4, 4])
                fpos = [centre[i] + r.uniform(-150.0, 150.0) for i in range(3)]
                recs.append(stc._ship(r, ids[4], fpos, radius=r.uniform(60.0, 200.0), free=r.random() < 0.5))
                recs.append(_harmonic(0, ids[4], h, True, cp, al))
            n_ships = r.randint(1, 3)
            for s_i, bid in enumerate(ids[6:6 + n_ships]):
                if flavour == "minis_order":                     # at the cluster, or through it
                    near = r.choice([0.0, 120.0])
                    start, goal, d = _through(r, centre, near, near + r.choice([40.0, 200.0]))
                    ship(bid, start, goal, radius=r.uniform(30.0, 90.0), dist=d)
                else:
                    start, goal, d = _through(r, r.choice(targets), 60.0, 350.0)
                    ship(bid, start, goal, dist=d)
                if flavour == "field_and_minis":
                    harmonics(bid, h, cp, al)
                if flavour == "minis_mixed":
                    for _ in range(r.randint(0, 2)):
                        k = r.randint(2, steps)
                        recs.append(r.choice([fc._k(k, fc.SF_OP, bid, d=fc._xyz(rr=r.uniform(0.0, 1.0))),
                                              fc._k(k, fc.STOP_OP, bid),
                                              fc._k(k, fc.DIR_OP, bid, d=tuple(stc._unit(r)))]))
        elif flavour == "field_shell":                           # a field around a cluster, at the ships' level
            fid = ids[0]
            h, cp, al = r.choice([5, 9]), r.choice([-1, 3, 3]), r.choice([-1, 4, 4])
            recs.append(stc._ship(r, fid, centre, radius=r.uniform(60.0, 150.0), free=False))
            recs.append(_harmonic(0, fid, h, True, cp, al))
            pid = ids[1]
            parent(pid, [centre[i] + r.uniform(-40.0, 40.0) for i in range(3)])
            for _ in range(r.randint(1, 3)):
                recs.append(fc._k(0, fc.MINIBALL_OP, pid, d=(*[r.uniform(-40.0, 40.0) for _ in range(3)],
                                                             r.uniform(5.0, 30.0))))
            for j in range(r.randint(1, 3)):
                recs.append(stc._sphere(r, ids[2 + j], [centre[i] + r.uniform(-50.0, 50.0) for i in range(3)],
                                        r.uniform(5.0, 30.0)))
            for bid in ids[6:6 + r.randint(1, 3)]:
                start, goal, d = _through(r, centre, 120.0, 300.0)
                ship(bid, start, goal, radius=r.uniform(2.0, 40.0), dist=d, mass=10 ** r.uniform(6.0, 7.5))
                harmonics(bid, h, cp, al)
        else:                                                    # the force fields
            n_fields = 2 if flavour == "two_fields" else 1
            fields = []
            for k in range(n_fields):
                fid = ids[k]
                h, cp, al = r.choice([5, 9, 5, 9, -1]), r.choice([-1, 3, 3]), r.choice([-1, 4, 4])
                fpos = [centre[i] + (r.uniform(-200.0, 200.0) if n_fields > 1 else 0.0) for i in range(3)]
                free = flavour == "field_free" or (flavour != "field_fixed" and r.random() < 0.3)
                vel = [r.uniform(-30.0, 30.0) for _ in range(3)] if free and r.random() < 0.7 else [0.0] * 3
                recs.append(stc._ship(r, fid, fpos, vel, radius=r.uniform(80.0, 260.0), free=free,
                                      massive=r.random() < 0.95))
                on = 0 if flavour != "field_toggle" else r.choice([0, 0, r.randint(1, steps)])
                recs.append(_harmonic(on, fid, h, True, cp, al))
                fields.append((fid, h, cp, al))
                if flavour == "field_toggle":
                    for _ in range(r.randint(1, 2)):
                        k2 = r.randint(1, steps)
                        recs.append(r.choice([_harmonic(k2, fid, h, False, cp, al),                  # cleared: STOP
                                              _harmonic(k2, fid, r.choice([5, 9, -1]), True, r.choice([-1, 3]),
                                                        r.choice([-1, 4])),                           # changed
                                              stc._goto(k2, fid, [centre[i] + 50.0 for i in range(3)])]))
            for bid in ids[4:4 + r.randint(1, 3)]:
                start, goal, d = _through(r, centre, 300.0, 800.0)
                ship(bid, start, goal, dist=d, mass=10 ** r.uniform(6.0, 7.5))
                fid, h, cp, al = r.choice(fields)
                harmonics(bid, h, cp, al)
                if flavour == "field_toggle" and r.random() < 0.5:
                    harmonics(bid, h, cp, al, step=r.randint(1, steps))
        fc.case(out, r, flavour, recs, steps, master=False, tick=tick, friction=fc._friction(r))
    return out


# ---- the batched run ----

def minis_of(c: dict) -> list:
    """A scenario's mini calls at step 0, in order, as vec_collide.make_minis takes them."""
    out = []
    for rec in c["records"]:
        if rec[0] != "K" or rec[1] != 0 or rec[2] not in MINI_OPS:
            continue
        _, _, op, a, bb, ccc, w, d = rec
        d = list(d) + [0.0] * 12
        if op == fc.MINIBALL_OP:
            out.append({"kind": "ball", "parent": a, "pos": d[0:3], "radius": d[3]})
        elif op == fc.MINICAPSULE_OP:
            out.append({"kind": "capsule", "parent": a, "a": d[0:3], "b": d[3:6], "radius": d[6]})
        else:
            out.append({"kind": "box", "parent": a, "corner": d[0:3], "lx": d[3:6], "ly": d[6:9], "lz": d[9:12]})
    return out


def world_of(c: dict) -> tuple:
    """(balls, static collidables) of a scenario as VecCollidePark.load takes them, its minis made by
    vec_collide.make_minis (a new park's local ids)."""
    from exact_ballpark.vec_collide import make_minis
    balls = list({rec[2]["id"]: stc._ball_dict(rec[2]) for rec in c["records"] if rec[0] == "B"}.values())
    mb, ms, _ = make_minis(balls, minis_of(c))
    return balls + mb, stc.statics_of(c) + ms


def needs(c: dict) -> tuple:
    """(minis, fields): whether a scenario calls a mini's thunker or SetBallHarmonic."""
    ops = {rec[2] for rec in c["records"] if rec[0] == "K"}
    return bool(ops & set(MINI_OPS)), fc.HARMONIC_OP in ops


def _commands(park, slot: list, recs: list) -> None:
    """A step's commands (full_cases' records), each its own masked call, in record order per world; SetBallHarmonic
    among them; the mini calls (made at the load) skipped."""
    dev = park.dev
    for w, rec in recs:
        _, _, op, a, bb, ccc, ww, d = rec
        if op in MINI_OPS or op in (fc.CAPSULE_OP, fc.OBOX_OP):
            continue
        if op != fc.HARMONIC_OP:
            stc._commands(park, slot, [(w, rec)])
            continue
        d = list(d) + [0.0] * 4
        mask = torch.zeros(park.W, park.N, dtype=torch.bool, device=dev)
        mask[w, slot[w][a]] = True
        full = lambda v: torch.full((park.W, park.N), v, dtype=torch.int64, device=dev)      # noqa: E731
        park.set_harmonic(mask, full(int(bb)), full(int(d[0])), full(int(d[1])),
                          torch.full((park.W, park.N), ww != 0, dtype=torch.bool, device=dev))


EXTRA_FLOAT = ("mass", "radius", "max_vel", "agility", "time_factor")
EXTRA_INT = ("owner_id", "harmonic", "corporation", "alliance")


def _collect(out: dict, g: list, park, slot: list, s: int) -> None:
    """collide_cases' rows, with each ball's mass, radius, maximum speed, agility, time factor, owner, harmonic,
    corporation and alliance (a park without minis: owner 0; without fields: -1, AddBall's)."""
    cc._collect(out, g, park, slot, s)
    host = {k: getattr(park, k).cpu().tolist() for k in EXTRA_FLOAT}
    own = park.owner.cpu().tolist() if park.minis else None
    fl = {k: getattr(park, k).cpu().tolist() for k in ("harmonic", "corporation", "alliance")} if park.fields else None
    for w, c in enumerate(g):
        if s > c["steps"]:
            continue
        for bid, j in slot[w].items():
            row = out[(c["id"], s, bid)]
            for k in EXTRA_FLOAT:
                row[k] = host[k][w][j]
            row["owner_id"] = own[w][j] if own is not None else 0
            for k in ("harmonic", "corporation", "alliance"):
                row[k] = fl[k][w][j] if fl is not None else -1


def batched(cs: list[dict], device: str, contact_slots=None, kernel: str = "torch", rows=None, after_tick=None,
            parks: list | None = None, flags: str = "auto", after_load=None, **kw) -> dict:
    """Run the scenarios batched, one park per (friction, tick, with rows="free" the free-ball count, and with flags
    "auto" the scenario's needs: minis, fields); flags "both" makes every park with minis and fields. Rows
    {(case, step, id): state}; `parks` collects the parks; `after_load(park)` runs after each load (planted faults)."""
    from exact_ballpark.vec_collide import VecCollidePark
    groups = defaultdict(list)
    built = {c["id"]: world_of(c) for c in cs}
    nfree = lambda c: sum(1 for b in built[c["id"]][0] if b["free"])      # noqa: E731
    for c in cs:
        need = needs(c) if flags == "auto" else (True, True)
        groups[(c["friction"], c["tick"], nfree(c) if rows == "free" else None, need)].append(c)
    out = {}
    for (friction, tick, free, (minis, fields)), g in groups.items():
        n = max(len(built[c["id"]][0]) for c in g)
        s = max(len(built[c["id"]][1]) for c in g)
        if rows is not None:
            kw["rows"] = max(1, free) if free is not None else rows
        park = VecCollidePark(len(g), n, friction=friction, tick_interval_ms=tick, contact_slots=contact_slots,
                              device=device, kernel=kernel, statics=s, minis=minis, fields=fields, **kw)
        park.load([built[c["id"]][0] for c in g], [built[c["id"]][1] for c in g])
        if after_load is not None:
            after_load(park)
        slot = [{int(i): j for j, i in enumerate(ids) if i} for ids in park.id.cpu().tolist()]
        by_step = defaultdict(list)
        for w, c in enumerate(g):
            for rec in c["records"]:
                if rec[0] == "K":
                    by_step[rec[1]].append((w, rec))
        _commands(park, slot, by_step.get(0, []))
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
    """Ball rows in full_cases' S layout (the reference's or the compiled oracle's) as {(case, step, id): state}, with
    the extra fields."""
    idx = {f: i for i, f in enumerate(fc.S_FIELDS)}
    out = stc.rows_of(s_rows)
    for row in s_rows:
        rec = out[(row[0], row[1], row[2])]
        for k in EXTRA_FLOAT + EXTRA_INT:
            rec[k] = row[idx[k]]
    return out


def differences(want: dict, got: dict) -> list:
    """collide_cases' differences, and the extra fields' (doubles by their bits, NaNs equal; integers equal)."""
    import space_cases as spc
    bad = cc.differences(want, got)
    if bad and bad[0][0] == "rows":
        return bad
    for key, w in want.items():
        gv = got[key]
        bad.extend((*key, f) for f in EXTRA_FLOAT if not spc.same(float(w[f]), float(gv[f])))
        bad.extend((*key, f) for f in EXTRA_INT if int(w[f]) != int(gv[f]))
    return bad


def reference(cs: list[dict]) -> tuple:
    """The reference (full_cases.run_reference: FullBallpark) on every scenario: (ball rows, the static collidables'
    filings {(case, id): boxes} at step 0, every row kind as full_cases keeps them, the undefined cases)."""
    rows, undefined = fc.run_reference(cs)
    statics = {}
    idx = {f: i for i, f in enumerate(fc.T_FIELDS)}
    for row in rows.get("T", []):
        if row[1] == 0:
            statics[(row[0], row[idx["id"]])] = tuple(row[idx["boxes"]])
    return rows_of(rows.get("S", [])), statics, rows, set(undefined)


def reach(rows: dict) -> dict:
    """What a run reached, from the reference's rows: ball-steps with a miniball on the collision list, with two
    miniballs of different owners on it, and with two of them in an order that BallSortComparer gives and the ids do
    not (the owner-first order seen); with a mini static collidable (local id) on it; with a FIELD ball on it."""
    from exact_ballpark.modes import FIELD, MINIBALL
    owner = {(cid, bid): r["owner_id"] for (cid, st, bid), r in rows.items() if st == 0}
    miniball = {k for k, r in rows.items() if r["mode"] == MINIBALL and k[1] == 0}
    mb_ids = {(cid, bid) for (cid, _, bid) in miniball}
    field_ids = {(cid, bid) for (cid, st, bid), r in rows.items() if r["mode"] == FIELD}
    out = {"rows": len(rows), "miniball": 0, "miniball_two_owners": 0, "owner_order_seen": 0, "mini_static": 0,
           "field_hit": 0}
    for (cid, st, bid), r in rows.items():
        hits = r["hits"]
        mbs = [h for h in hits if (cid, h) in mb_ids]
        out["miniball"] += bool(mbs)
        owners = [owner[(cid, h)] for h in mbs]
        out["miniball_two_owners"] += len(set(owners)) > 1
        out["owner_order_seen"] += any(mbs[i] > mbs[j] and owners[i] != owners[j]
                                       for i in range(len(mbs)) for j in range(i + 1, len(mbs)))
        out["mini_static"] += any(h < 0 and (cid, h) not in mb_ids for h in hits)
        out["field_hit"] += any((cid, h) in field_ids for h in hits)
    return out

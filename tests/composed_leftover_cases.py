# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the pieces of the composed park (exact_ballpark.vec_composed.VecComposedPark): the commands (SetBallFree, SetBallRigid, the Python SetBoid, SetBallTroll, SetBallInteractive) and the
mushrooms (AddMushroom, the growth by glibc's powf with its re-filing, the removal in a tick, BringOutTheDeads) among the
composed park's collisions, warps, follows, orbits, missiles and formations (tests/park_cases.py's cluster worlds),
generated from a seed, run by the scalar reference (exact_ballpark.full.FullBallpark, one park a world) and by the batch
(one composed park for a batch of worlds), compared bit for bit after every step: every ball's state (park_cases' fields:
destiny_combat_cases' with the collision acceleration mLastC, the collision list and the partition's filing; and the
free, interactive and moribund flags), which balls exist, the ReserveFormationSlot answers and the DoCollision events.

A world is a park_cases cluster (ships 1 to 6 a few hundred metres across, most massive, fixed spheres 7 and 8, a warp
in or out, follows, orbits, setters, cloaks, missiles, formations) with two empty rows for mushrooms and, by family:
"mush" (a mushroom at a ship, its range of the cluster's size so that it grows into the others, freed to grow and be
removed while followers, orbiters, formation followers and missiles hold it, members SetBoid took out of a follow mode
and trolls among them, fixed again now and then; on a client sometimes a ship -5, a member whose turn comes before the
mushroom's id 0; BringOutTheDeads between ticks; now and then a second mushroom), "cmd" (SetBallFree both ways, SetBallRigid,
SetBoid, SetBallTroll and SetBallInteractive on the cluster's ships, the fixed spheres freed, followers set to BOID before
their target warps out of the cluster or cloaks). Ship 1 stays interactive (the master's bubble alive). At most one
negative id a world (the finding: the reference's partition is held to Destiny's there).

The batch maps its rows to balls by its own ids at every step (a mushroom's id is the park's: 0 on a client, the world's
next local id on the master). Frozen before any held-out seed is drawn.
"""

from __future__ import annotations

import random
from collections import defaultdict

import torch

import combat_leftover_cases as cl
import destiny_combat_cases as dc
import park_cases as pc
from exact_ballpark.full import FullBallpark
from exact_ballpark.space import DestinyUndefined

POOL = dc.POOL
MUSH_POOL = cl.MUSH_POOL
FORMATIONS = pc.FORMATIONS
FLAGS = cl.FLAGS
EXTRA = pc.EXTRA
PC_OPS = {"warpin", "radius", "maxvel", "sf", "mass"}       # park_cases' own records (the rest: combat_leftover_cases')


def fields(do: bool) -> tuple:
    return dc.fields(do) + FLAGS


def _centre(ships: list) -> list:
    """The cluster's centre (ship 1's position: park_cases puts the others around the same centre)."""
    return list(ships[0]["pos"])


def world(r: random.Random, steps: int, master: bool, do: bool, tick: int, kind: str) -> dict:
    """One world: park_cases' cluster world, two mushroom rows, and the records of its family."""
    w = pc.world(r, steps, master, tick)
    ships, recs = w["ships"], list(w["records"])
    centre = _centre(ships)
    ids = [s["id"] for s in ships]
    free_ids = [s["id"] for s in ships if s["free"]]
    others = [i for i in free_ids if i != 1]
    total = steps * tick * 0.001                             # the run's seconds
    warpers = {rec[2] for rec in recs if rec[1] in ("warp", "warpin")}
    if kind == "mush":
        mid = cl._mush_id(master, do, 0)
        owner = r.choice(free_ids)
        rng = r.choice([r.uniform(20.0, 150.0)] * 2 + [r.uniform(150.0, 900.0)] * 3 + [r.uniform(900.0, 3000.0)]
                       + [-5.0, 0.0][:1 if r.random() < 0.9 else 2])
        life = r.choice([r.uniform(0.2, 0.7) * total] * 6 + [r.uniform(1.0, 2.0) * total] * 2 + [0.0, -1.0])
        k0 = r.choice([0, 0, 1, 2])
        recs.append((k0, "mushroom", r.choice([owner] * 9 + [999999]), rng, life))
        recs.append((r.choice([k0, k0, k0 + 1, k0 + 3]), "setfree", mid, 1))
        if r.random() < 0.25:
            recs.append((r.randint(k0 + 2, steps), "setfree", mid, 0))   # fixed again: re-filed at rest, no growth
        if r.random() < 0.3:
            recs.append((k0, "setform", mid, r.choice([0, 0, 1, 2])))     # a mushroom as a formation leader
        holders = [i for i in others if i not in warpers and i != owner]
        r.shuffle(holders)
        for bid in holders[:r.randint(0, len(holders))]:   # ships that hold it: followers, orbiters, formation
            k = r.choice([k0, k0, k0 + 1, k0 + 2])
            how = r.choice(["follow", "orbit", "formfollow", "reserve_follow"])
            if how in ("follow", "orbit"):
                recs.append((k, how, bid, mid, r.choice([0.0, 10.0, 200.0, 1500.0])))
            elif how == "formfollow":
                recs.append((k, "formfollow", bid, mid, r.choice([0, 1, 2, 5])))
            else:
                recs.append((k, "reserve_follow", bid, mid))
            if r.random() < 0.35:
                kb = k + r.randint(0, 3)
                recs.append((kb, "boid", bid, 1))                       # a member out of the follow modes ...
                if r.random() < 0.5:                                      # ... made a troll, still a member
                    recs.append((kb + r.randint(0, 2), "troll", bid, r.choice([2, 6, 30, -2])))
            elif r.random() < 0.2:
                recs.append((k + r.randint(1, 4), "troll", bid, r.choice([1, 3, 30, -2])))
        if not master and r.random() < 0.5:                   # a client's ship whose turn comes before id 0's
            s = pc._ship(r, -5, centre, 300.0, interactive=r.random() < 0.5)
            ships.append(s)
            how = r.choice(["follow", "orbit", "formfollow", "reserve_follow", "follow"])
            k = r.choice([k0, k0, k0 + 1])
            if how in ("follow", "orbit"):
                recs.append((k, how, -5, mid, r.choice([0.0, 10.0, 200.0, 1500.0])))
            elif how == "formfollow":
                recs.append((k, "formfollow", -5, mid, r.choice([0, 1, 2, 5])))
            else:
                recs.append((k, "reserve_follow", -5, mid))
            if r.random() < 0.3:
                recs.append((k + r.randint(0, 3), "boid", -5, 1))
        for mi in [m for m in POOL if not any(rec[1] == "add" and rec[2]["id"] == m for rec in recs)][:r.randint(0, 2)]:
            k = r.randint(k0, max(k0, steps - 4))                # a missile at the mushroom
            recs.append((k, "add", dc._missile(r, mi, centre)))
            recs.append((k, "launch", mi, mid, r.choice(free_ids), r.choice([0, 1, 1]), r.choice([0, 1, 1])))
            if r.random() < 0.2:
                recs.append((k + r.randint(0, 3), "boid", mi, 1))
        for k in range(1, steps + 1):
            if r.random() < 0.25:
                recs.append((k, "deads"))
        if r.random() < 0.3:                                  # a second mushroom
            k2 = r.randint(2, steps)
            recs.append((k2, "deads"))
            recs.append((k2, "mushroom", r.choice(free_ids), r.uniform(20.0, 600.0), r.uniform(0.1, 0.5) * total))
            recs.append((k2, "setfree", cl._mush_id(master, do, 1), 1))
            recs.append((k2, "setfree", mid, 1))
    else:
        warp_out = [i for i in others if i not in warpers]
        if warp_out and r.random() < 0.7:                    # a target warps out; its followers set to BOID first
            x = r.choice(warp_out)
            kw = r.randint(2, max(2, steps // 2))
            u = dc._unit(r)
            if r.random() < 0.7:                              # aligned and fast: warp proper at its next tick
                vmax = next(s["max_vel"] for s in ships if s["id"] == x)
                recs.append((kw, "vel", x, *[u[i] * vmax * r.uniform(0.8, 1.0) for i in range(3)]))
            recs.append((kw, "warp", x, *[centre[i] + u[i] * r.uniform(2e5, 9e5) for i in range(3)],
                         r.choice([0.0, 2e4]), r.choice([1500, 3000, 20000])))
            warpers.add(x)
            for f in [i for i in free_ids if i not in warpers and r.random() < 0.6]:
                k = r.randint(0, kw - 1)
                recs.append((k, r.choice(["follow", "orbit"]), f, x, r.choice([0.0, 50.0, 500.0])))
                if r.random() < 0.7:
                    recs.append((r.randint(k, kw), "boid", f, 1))
                    if r.random() < 0.3 and f != 1:
                        recs.append((r.randint(k, kw), "troll", f, r.choice([2, 8, 40])))
        for s in ships:
            if r.random() < 0.6:
                bid = s["id"]
                k = r.randint(0, steps)
                what = r.choice(["setfree", "setfree", "setfree", "rigid", "boid", "troll", "troll", "inter"])
                if bid == 1 and what in ("troll", "inter"):        # ship 1 keeps its bubble alive
                    what = "boid"
                if what == "setfree":
                    recs.append((k, "setfree", bid, r.choice([0, 0, 1, 2])))
                    if r.random() < 0.5:
                        recs.append((r.randint(k, steps), "setfree", bid, r.choice([0, 1])))
                elif what == "rigid":
                    recs.append((k, "rigid", bid))
                elif what == "boid":
                    recs.append((k, "boid", bid, r.choice([1, 1, 0])))
                elif what == "troll":
                    recs.append((k, "troll", bid, r.choice([-3, 0, 1, 2, 5, 12, 25])))
                else:
                    recs.append((k, "inter", bid, r.choice([0, 1])))
    recs.sort(key=lambda rec: rec[0])
    return {"ships": ships, "pool": list(POOL), "records": recs, "start": w["start"], "kind": kind}


def batches(seed: int, n_batches: int = 12, worlds: int = 12) -> list:
    """The scenarios of a seed: batches of worlds, each one configuration (even batches without dynamical orientation,
    odd ones with it; the server's view or a client's; the orbit law; the tick and friction), the families mixed."""
    r = random.Random(f"compose3-{seed}")
    out = []
    for b in range(n_batches):
        do = b % 2 == 1
        master = r.random() < 0.5
        cfg = {"do": do, "do_off": do and r.random() < 0.4, "master": master,
               "tick": r.choice([250, 500, 1000, 1000, 1000, 2000]),
               "friction": r.choice([1.0e6] * 5 + [10 ** r.uniform(5.0, 7.0)]),
               "new_orbit": r.random() < 0.5, "steps": r.randint(12, 26)}
        cfg["worlds"] = [world(r, cfg["steps"], master, do, cfg["tick"], "mush" if r.random() < 0.6 else "cmd")
                         for _ in range(worlds)]
        out.append(cfg)
    return out


# ---- the reference ----

def _apply_ref(park: FullBallpark, rec: tuple, answers: list) -> None:
    if rec[1] in PC_OPS:
        pc._apply_ref(park, rec, answers)
    else:
        cl._apply_ref(park, rec, answers)


def reference(cfg: dict, w: dict, trace=None) -> dict:
    """One world on FullBallpark: park_cases' layout (destiny_combat_cases' rows with EXTRA) with the FLAGS."""
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
            out["S"][s] = {b.id: {**cl._ref_row(b, cfg["do"]), **pc._ref_extra(b)} for b in park.balls.values()}
            if cfg["master"] and len({b.bubble for b in park.balls.values()}) > 1:
                out["B"].append(s)
    except DestinyUndefined as exc:
        out["X"] = str(exc)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:     # where Python raises and C computes on
        out["X"] = f"the reference raised {type(exc).__name__}: {exc}"
    out["E"] = [(t, bid, args[0]) for (t, name, bid, *args) in park.events if name == "DoCollision"]
    return out


# ---- the batch ----

def apply_batch(p, recs: dict, outs: list) -> None:
    """The j-th record of each world's step: park_cases' own kinds by its apply_batch, the rest by
    combat_leftover_cases' (the kinds, then destiny_combat_cases'), the rows found by the park's ids."""
    mine = {wi: rec for wi, rec in recs.items() if rec[1] in PC_OPS}
    rest = {wi: rec for wi, rec in recs.items() if rec[1] not in PC_OPS}
    if rest:
        cl.apply_batch(p, rest, outs)
    if mine:
        pc.apply_batch(p, mine, cl._rows_by_id(p), p.n, outs)


def batched(cfg: dict, device: str = "cpu", path: str = "torch", graph: bool = False, park_cls=None,
            libs: dict | None = None) -> tuple:
    """The batch on VecComposedPark (or `park_cls`): (per world the layout of `reference`, the park). `path` "torch"
    (evolve_torch) or "kernel" (evolve, with `libs` the kernels' libraries); `graph` a ComposedGraph's replays."""
    from exact_ballpark.vec_composed import VecComposedPark
    rows = cl.rows_for(cfg)
    cls = park_cls or VecComposedPark
    p = cls(len(rows), cfg["friction"], cfg["tick"], device, cfg["do"], cfg["new_orbit"], master=cfg["master"],
            disable_do_for_missiles=cfg["do_off"])
    p.load(rows)
    p.load_formations(FORMATIONS)
    W = len(cfg["worlds"])
    outs = [{"S": {}, "E": [], "A": [], "B": [], "X": None} for _ in range(W)]
    by_step = [defaultdict(list) for _ in range(W)]
    for wi, w in enumerate(cfg["worlds"]):
        for rec in w["records"]:
            by_step[wi][rec[0]].append(rec)
    names = fields(cfg["do"])
    ids0, world0 = p.ids.tolist(), p.world.tolist()
    p.mush_rows = {wi: [r for r in range(p.n) if world0[r] == wi and ids0[r] in MUSH_POOL] for wi in range(W)}
    runner = None
    for s in range(cfg["steps"] + 1):
        slots = max(len(by_step[wi].get(s, ())) for wi in range(W))
        for j in range(slots):
            recs = {wi: by_step[wi][s][j] for wi in range(W) if j < len(by_step[wi].get(s, ()))}
            apply_batch(p, recs, outs)
        if s > 0:
            if graph:
                if runner is None:
                    from exact_ballpark.vec_composed_cuda import ComposedGraph
                    runner = ComposedGraph(p, libs=libs)
                runner.tick()
            elif path == "kernel":
                p.evolve(libs)
            else:
                p.evolve_torch()
            for (wl, t, mi, col) in p.events():
                outs[wl]["E"].append((t, mi, col))
        cache = {k: getattr(p, k).cpu().tolist() for k in names}
        last_c, level, key, mod = (getattr(p, k).cpu().tolist() for k in ("last_c", "level", "key", "mod"))
        lists = p.collision_lists()
        for wi in range(W):
            outs[wi]["S"][s] = {}
        for (wi, bid), r in cl._rows_by_id(p).items():
            row = {k: (int(cache[k][r]) if k in FLAGS else dc._value(cache[k][r], k)) for k in names}
            row.update(last_c=tuple(float(c) for c in last_c[r]), collisions=lists[r], level=int(level[r]),
                       key=tuple(int(c) for c in key[r]), mod=tuple(int(c) for c in mod[r]))
            outs[wi]["S"][s][bid] = row
    return outs, p


# ---- the comparison ----

def differences(want: dict, got: dict, limit: int = 20) -> list:
    return dc.differences(want, got, limit)


def in_domain(want: dict) -> bool:
    return dc.in_domain(want)


# ---- the same worlds on Destiny's own code (the conformance kit's compiled full harness: tests/full_cases.py) ----

def harness_case(cfg: dict, w: dict, cid: int, answers: list) -> dict:
    """A world as the full harness takes it (full_cases' records), in the world's own record order: park_cases' setters,
    EntityWarpIn and SetBallMassive mapped here, every other record by combat_leftover_cases.harness_case (the mapping, one record at a time; ReserveFormationSlot-then-FormationFollow with the slot the reference
    answered, `answers` the reference's "A" in order)."""
    import full_cases as fc
    k = fc._k
    setters = {"radius": fc.RADIUS_OP, "maxvel": fc.MAXVEL_OP, "sf": fc.SF_OP, "mass": fc.MASS_OP}
    base = cl.harness_case(cfg, {**w, "records": []}, cid, [])          # the ships, then the formations' table
    recs, ans = list(base["records"]), list(answers)
    for rec in w["records"]:
        s, op, a = rec[0], rec[1], rec[2:]
        if op in setters:
            recs.append(k(s, setters[op], a[0], d=(0.0, 0.0, 0.0, a[1])))
        elif op == "warpin":
            recs.append(k(s, fc.WARPIN_OP, a[0], w=a[4], d=tuple(a[1:4])))
        elif op == "massive":
            recs.append(k(s, fc.MASSIVE_OP, a[0], w=int(a[1])))
        else:
            one = cl.harness_case(cfg, {**w, "ships": [], "records": [rec]}, cid, ans)
            recs.extend(one["records"][1:])                                 # past the formations' table
            if op == "reserve_follow" and ans:
                ans = ans[1:]
    return {**base, "kind": "compose3", "records": recs}

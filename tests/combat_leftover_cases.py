# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the pieces of the combat park (exact_ballpark.vec_combat.VecCombatPark): the
commands (SetBallFree, SetBallRigid, the Python SetBoid, SetBallTroll, SetBallInteractive) among missiles and
formations, and mushrooms (AddMushroom, the growth by glibc's powf, the removal in a tick, BringOutTheDeads), generated
from a seed, run by the scalar reference (exact_ballpark.full.FullBallpark, one park a world) and by the batch (one park
for a batch of worlds), compared bit for bit after every step: every ball's state the batch holds (with its free,
interactive and moribund flags), which balls exist, the ReserveFormationSlot answers and the DoCollision events.

A batch is a configuration (as tests/destiny_combat_cases.py's: friction, tick, dynamical orientation and its missile
exclusion, the orbit law, server or client view) and its worlds. A world is its ships (ids 1 to 5; on a client
sometimes a ship -5, whose turn comes before a mushroom's), a pool of missile ids (101 to 104; in half of a client's
mushroom worlds the first is -7, also before it) with an empty row each, two empty rows for mushrooms, and records applied at given steps (step 0 the setup; after each later step's records every
park evolves one tick). Two families: "cmd" (the commands on ships, missiles and formation followers, with releases of
members SetBoid took out of a follow mode: a cloak, a removal, a formation left, a warp) and "mush" (a mushroom freed to
grow and be removed while missiles, followers, formation followers and orbiters hold it as their target, members with
ids on either side of its own, trolls and BOID members among them; BringOutTheDeads between ticks).

The batch maps its rows to balls by its own ids at every step (a mushroom's id is the park's: 0 on a client, the
world's next local id on the master). The domain is VecCombatPark's (at most one massive ball outside MISSILE mode a
world: the mushroom takes the place, every ship non-massive in a "mush" world). Frozen before any held-out seed is
drawn.
"""

from __future__ import annotations

import random
from collections import defaultdict

import torch

import destiny_combat_cases as dc
from exact_ballpark.full import FullBallpark
from exact_ballpark.space import DestinyUndefined

POOL = dc.POOL
MUSH_POOL = (900001, 900002)              # placeholder ids of the two empty rows a mushroom fills (never a ball's)
# the tables a mid-run LoadFormations loads ("loadform", index): the park's own table, so every world of a batch at once
RELOADS = (dc.FORMATIONS, [dc.FORMATIONS[2], dc.FORMATIONS[0][:3]])
FLAGS = ("free", "interactive", "moribund")
_REF_FLAG = {"free": "is_free", "interactive": "is_interactive", "moribund": "is_moribund"}


def fields(do: bool) -> tuple:
    return dc.fields(do) + FLAGS


def _ref_row(b, do: bool) -> dict:
    row = dc._ref_row(b, do)
    for k in FLAGS:
        row[k] = int(bool(getattr(b, _REF_FLAG[k])))
    return row


def _mush_id(master: bool, do: bool, k: int) -> int:
    """The id of the k-th mushroom a world accepts (0 on a client; the master's local ids from -1 down, two a ball with
    dynamical orientation)."""
    return -(1 + (k + 1) * (2 if do else 1)) if master else 0


def world(r: random.Random, steps: int, master: bool, do: bool, kind: str) -> dict:
    """One world: ships, a missile pool, two mushroom rows and its records [(step, op, *args)]."""
    centre = [r.uniform(-2e5, 2e5) for _ in range(3)]
    n = r.randint(2, 5)
    ids = list(range(1, n + 1))
    if not master and r.random() < 0.5:
        ids[-1] = -5                                         # a client ship whose turn comes before id 0's
    ships = [dc._ship(r, ids[0], centre, interactive=True)] + [dc._ship(r, k, centre) for k in ids[1:]]
    if r.random() < 0.2:
        ships[-1].update(free=False, vel=[0.0, 0.0, 0.0])
    heavy = r.choice([None] + ids) if kind == "cmd" else None
    for s in ships:
        s["massive"] = s["id"] == heavy
    recs, free_ids = [], [s["id"] for s in ships if s["free"]]
    for s in ships:                                          # the ships' flight
        if s["free"] and r.random() < 0.7:
            k = r.choice([0, 0, 1, r.randint(1, steps)])
            how = r.choice(["goto", "dir", "dir", "stop"])
            if how == "goto":
                recs.append((k, "goto", s["id"], *[centre[i] + r.uniform(-2e4, 2e4) for i in range(3)]))
            elif how == "dir":
                recs.append((k, "dir", s["id"], *dc._unit(r)))
            else:
                recs.append((k, "stop", s["id"]))
    pool = list(POOL)
    if kind == "mush" and not master and r.random() < 0.5:   # a client's missile whose turn comes before id 0's
        pool = [-7] + list(POOL[1:])
    missiles = pool[:r.randint(1 if kind == "mush" else 0, len(pool))]
    mid = _mush_id(master, do, 0)
    targets = ids + ([mid] * 3 if kind == "mush" else [])
    if kind == "mush":                                       # a mushroom at a ship, freed to grow
        owner = r.choice(ids)
        rng = r.choice([r.uniform(50.0, 400.0)] * 4 + [r.uniform(400.0, 3000.0)] * 3 + [-5.0, 0.0])
        life = r.choice([r.uniform(0.5, 6.0)] * 5 + [r.uniform(6.0, 20.0)] * 2 + [0.0, -1.0])
        recs.append((0, "mushroom", r.choice([owner] * 9 + [999999]), rng, life))
        recs.append((r.choice([0, 1, 2, 3]), "setfree", mid, 1))
        if r.random() < 0.3:
            recs.append((r.randint(1, steps), "setfree", mid, 0))        # fixed again: it stops growing
        if r.random() < 0.4:
            recs.append((0, "setform", mid, r.choice([0, 0, 1, 2])))     # a mushroom as a formation leader
        for s in ships[1:]:                                  # ships that hold it: followers, orbiters, formation
            if not s["free"] or r.random() < 0.3:
                continue
            k = r.choice([0, 1, 1, 2])
            how = r.choice(["follow", "orbit", "formfollow", "reserve_follow", "reserve_follow"])
            if how in ("follow", "orbit"):
                recs.append((k, how, s["id"], mid, r.choice([10.0, 200.0, 1500.0])))
            elif how == "formfollow":
                recs.append((k, "formfollow", s["id"], mid, r.choice([0, 1, 2, 5])))
            else:
                recs.append((k, "reserve_follow", s["id"], mid))
            if r.random() < 0.35:
                kb = k + r.randint(0, 3)
                recs.append((kb, "boid", s["id"], 1))                   # a member out of the follow modes ...
                if r.random() < 0.5:                                      # ... made a troll, still a member
                    recs.append((kb + r.randint(0, 2), "troll", s["id"], r.choice([2, 6, 30, -2])))
            elif r.random() < 0.2 and s["id"] != ids[0]:
                recs.append((k + r.randint(1, 4), "troll", s["id"], r.choice([1, 3, 30, -2])))
        for k in range(1, steps + 1):
            if r.random() < 0.25:
                recs.append((k, "deads"))
        if r.random() < 0.35:                                # a second mushroom
            k2 = r.randint(2, steps)
            recs.append((k2, "deads"))
            recs.append((k2, "mushroom", r.choice(ids), r.uniform(20.0, 600.0), r.uniform(0.5, 4.0)))
            recs.append((k2, "setfree", _mush_id(master, do, 1), 1))
            recs.append((k2, "setfree", mid, 1))
    else:                                                    # the commands on ships, missiles, formations
        if r.random() < 0.5:
            leader = r.choice(free_ids or ids)
            recs.append((0, "setform", leader, r.choice([0, 0, 1, 2])))
            for s in ships:
                if s["id"] != leader and s["free"] and r.random() < 0.6:
                    k = r.choice([0, 1, r.randint(1, steps)])
                    recs.append((k, "reserve_follow", s["id"], leader))
                    if r.random() < 0.4:
                        recs.append((k + r.randint(0, 3), "boid", s["id"], 1))
            for _ in range(r.randint(0, 2)):
                what = r.choice(["setform", "setfree", "warp"])
                if what == "warp" and heavy not in (None, leader):
                    what = "setform"
                recs.append((r.randint(2, steps), what, leader))
        for s in ships:
            if r.random() < 0.6:
                k = r.randint(0, steps)
                what = r.choice(["setfree", "setfree", "rigid", "boid", "boid", "troll", "troll", "inter", "orbit"])
                if s["id"] == ids[0] and what in ("troll", "inter"):     # ship 1 keeps its bubble alive
                    what = "boid"
                if what == "setfree":
                    recs.append((k, "setfree", s["id"], r.choice([0, 0, 1, 2])))
                    if r.random() < 0.5:
                        recs.append((r.randint(k, steps), "setfree", s["id"], 1))
                elif what == "rigid":
                    recs.append((k, "rigid", s["id"]))
                elif what == "boid":
                    recs.append((k, "boid", s["id"], r.choice([1, 1, 0])))
                elif what == "troll":
                    recs.append((k, "troll", s["id"], r.choice([-3, 0, 1, 2, 5, 12, 25])))
                elif what == "inter":
                    recs.append((k, "inter", s["id"], r.choice([0, 1])))
                else:
                    other = r.choice([t for t in ids if t != s["id"]] or ids)
                    recs.append((k, r.choice(["orbit", "follow"]), s["id"], other, r.choice([50.0, 500.0, 3000.0])))
                    if r.random() < 0.5:
                        recs.append((k + r.randint(0, 3), "boid", s["id"], 1))
        for _ in range(r.randint(0, 2)):                     # what happens to targets and leaders
            k = r.randint(1, steps)
            who = r.choice(ids[1:] or ids)
            what = r.choice(["cloak", "remove", "warp"])
            if what == "warp" and heavy not in (None, who):  # a drop-out is massive: the world's one place
                what = "cloak"
            recs.append((k, what, who))
    for mi in missiles:                                      # missiles at ships or the mushroom; some set to BOID
        k = r.randint(0 if kind == "mush" else 1, max(1, steps - 6))
        recs.append((k, "add", dc._missile(r, mi, centre)))
        owner = r.choice([i for i in ids if i > 0])          # a negative owner is a defender: not this park's
        target = r.choice([t for t in targets if t != owner] or targets)
        if mi < 0:                                           # the client's missile -7 holds the mushroom
            target = mid
        boid_it, fix_it = r.random() < 0.3, r.random() < 0.2
        massive = 0 if (boid_it or fix_it) else r.choice([0, 1, 1])     # a massive non-missile: the world's place
        if r.random() < 0.6:
            recs.append((k, "launch", mi, target, owner, r.choice([0, 1, 1]), massive))
        else:
            recs.append((k, "vel", mi, *[c * r.uniform(50.0, 300.0) for c in dc._unit(r)]))
            recs.append((k, "mfollow", mi, target, owner))
        if boid_it:
            recs.append((k + r.randint(0, 4), "boid", mi, 1))
        if fix_it:
            recs.append((k + r.randint(0, 4), "setfree", mi, r.choice([0, 1])))
    out = []
    for rec in recs:                                         # expand the target-and-leader events
        k, op = rec[0], rec[1]
        if op == "warp" and len(rec) == 3:
            u = dc._unit(r)
            out.append((k, "warp", rec[2], *[u[i] * 1e9 for i in range(3)], r.choice([0.0, 2e4]), r.randint(1, 30)))
        elif op == "cloak":
            out.append((k, "cloak", rec[2], 1))
        elif op == "uncloak_ok":
            out.append((k, "cloak", rec[2], 1))
            out.append((min(steps, k + 2), "uncloak", rec[2]))
        elif op == "setform" and len(rec) == 3:
            out.append((k, "setform", rec[2], r.choice([1, -1, 3])))
        elif op == "setfree" and len(rec) == 3:
            out.append((k, "setfree", rec[2], r.choice([0, 1])))
        else:
            out.append(rec)
    out = [rec for rec in out if not (kind == "mush" and rec[1] in ("warp", "uncloak"))]
    out.sort(key=lambda rec: rec[0])
    return {"ships": ships, "pool": pool, "records": out, "start": r.randrange(0, 1000), "kind": kind}


def batches(seed: int, n_batches: int = 12, worlds: int = 12) -> list:
    """The scenarios of a seed: batches of worlds, each one configuration (even batches without dynamical orientation,
    odd ones with it), the families mixed in each batch."""
    r = random.Random(seed)
    out = []
    for b in range(n_batches):
        do = b % 2 == 1
        cfg = {"do": do, "do_off": do and r.random() < 0.4, "master": r.random() < 0.55,
               "tick": r.choice([250, 500, 1000, 1000, 1000, 2000]),
               "friction": r.choice([1.0e6] * 5 + [10 ** r.uniform(5.0, 7.0)]),
               "new_orbit": r.random() < 0.3, "steps": r.randint(10, 22)}
        cfg["worlds"] = [world(r, cfg["steps"], cfg["master"], do, "mush" if r.random() < 0.55 else "cmd")
                         for _ in range(worlds)]
        out.append(cfg)
    return out


def with_reloads(cfg: dict) -> dict:
    """cfg with a mid-run LoadFormations in every world (the park's own table, so every world at once, first in its
    step): the second table (RELOADS[1]: fewer formations, so a leader's index past the end) at a third of the steps,
    the first again at two thirds."""
    k1, k2 = max(1, cfg["steps"] // 3), max(2, 2 * cfg["steps"] // 3)
    worlds = []
    for w in cfg["worlds"]:
        recs = [(k1, "loadform", 1), (k2, "loadform", 0)] + list(w["records"])
        recs.sort(key=lambda rec: (rec[0], rec[1] != "loadform"))
        worlds.append({**w, "records": recs})
    return {**cfg, "worlds": worlds}


# ---- the reference ----

def _apply_ref(park: FullBallpark, rec: tuple, answers: list) -> None:
    op, a = rec[1], rec[2:]
    if op == "setfree":
        park.set_free(a[0], bool(a[1]))
    elif op == "rigid":
        park.set_ball_rigid(a[0])
    elif op == "boid":
        park.set_boid(a[0], a[1])
    elif op == "troll":
        park.set_ball_troll(a[0], a[1])
    elif op == "inter":
        if a[0] in park.balls:
            park.set_interactive(a[0], bool(a[1]))
    elif op == "follow":
        park.follow_ball(a[0], a[1], a[2])
    elif op == "orbit":
        park.orbit(a[0], a[1], a[2])
    elif op == "mushroom":
        park.add_mushroom(a[0], a[1], a[2])
    elif op == "deads":
        park.bring_out_the_deads()
    elif op == "loadform":
        park.load_formations(RELOADS[a[0]])
    else:
        dc._apply_ref(park, rec, answers)


def reference(cfg: dict, w: dict, trace=None) -> dict:
    """One world on FullBallpark: as destiny_combat_cases.reference, every ball's row with its flags."""
    park = FullBallpark(friction=cfg["friction"], tick_interval_ms=cfg["tick"], use_new_orbit=cfg["new_orbit"],
                        is_master=cfg["master"], dynamical_orientation=cfg["do"],
                        disable_do_for_missiles=cfg["do_off"])
    park.current_time = w["start"]
    park.trace = trace
    for s in w["ships"]:
        dc._add_ref(park, s)
    park.load_formations(dc.FORMATIONS)
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
            out["S"][s] = {b.id: _ref_row(b, cfg["do"]) for b in park.balls.values()}
            if cfg["master"] and len({b.bubble for b in park.balls.values()}) > 1:
                out["B"].append(s)
    except DestinyUndefined as exc:
        out["X"] = str(exc)
    out["E"] = [(t, bid, args[0]) for (t, name, bid, *args) in park.events if name == "DoCollision"]
    return out


# ---- the batch ----

def rows_for(cfg: dict) -> list:
    """The batch's rows: each world's ships, its missile pool's empty rows, its two mushroom rows."""
    rows = []
    bubble = 0 if cfg["master"] else -1
    empty = {"exists": False, "mass": 1.0, "radius": 1.0, "max_vel": 1.0, "pos": [0.0, 0.0, 0.0],
             "vel": [0.0, 0.0, 0.0], "agility": 1.0, "speed_fraction": 1.0, "bubble": bubble}
    for wi, w in enumerate(cfg["worlds"]):
        for s in w["ships"]:
            rows.append({**s, "world": wi, "time": w["start"], "bubble": bubble})
        for mid in list(w["pool"]) + list(MUSH_POOL):
            rows.append({**empty, "id": mid, "world": wi, "time": w["start"]})
    return rows


def _rows_by_id(p, empty: bool = False) -> dict:
    """{(world, id): row} of the existing balls, from the park's own ids; with `empty`, the empty rows too by the id
    they hold (a missile's pool row by its id, for destiny_combat_cases.apply_batch's "add"), an existing ball first."""
    ids, world, exists = p.ids.tolist(), p.world.tolist(), p.exists.tolist()
    out = {(world[r], ids[r]): r for r in range(p.n) if empty and not exists[r]}
    out.update({(world[r], ids[r]): r for r in range(p.n) if exists[r]})
    return out


def batched(cfg: dict, device: str = "cuda", graph: bool = False, path: str = "kernel", park_cls=None) -> tuple:
    """The batch on VecCombatPark (or `park_cls`, a planted fault's subclass): (per world the layout of `reference`,
    the park)."""
    import exact_ballpark.vec_combat as vc
    rows = rows_for(cfg)
    cls = park_cls or vc.VecCombatPark
    p = cls(len(rows), cfg["friction"], cfg["tick"], device, cfg["do"], cfg["new_orbit"], master=cfg["master"],
            disable_do_for_missiles=cfg["do_off"])
    p.load(rows)
    p.load_formations(dc.FORMATIONS)
    W = len(cfg["worlds"])
    outs = [{"S": {}, "E": [], "A": [], "B": [], "X": None} for _ in range(W)]
    by_step = [defaultdict(list) for _ in range(W)]
    for wi, w in enumerate(cfg["worlds"]):
        for rec in w["records"]:
            by_step[wi][rec[0]].append(rec)
    runner = None
    names = fields(cfg["do"])
    ids0, world0 = p.ids.tolist(), p.world.tolist()
    p.mush_rows = {wi: [r for r in range(p.n) if world0[r] == wi and ids0[r] in MUSH_POOL] for wi in range(W)}
    for s in range(cfg["steps"] + 1):
        slots = max(len(by_step[wi].get(s, ())) for wi in range(W))
        for j in range(slots):
            recs = {wi: by_step[wi][s][j] for wi in range(W) if j < len(by_step[wi].get(s, ()))}
            apply_batch(p, recs, outs)
        if s > 0:
            if graph:
                if runner is None:
                    from exact_ballpark.vec_combat_cuda import CombatGraph
                    runner = CombatGraph(p)
                runner.tick()
                p.after_tick()
            else:
                p.evolve() if path == "kernel" else p.evolve_torch()
            for (wl, t, mi, col) in p.events():
                outs[wl]["E"].append((t, mi, col))
        cache = {k: getattr(p, k).cpu().tolist() for k in names}
        for wi in range(W):
            outs[wi]["S"][s] = {}
        for (wi, bid), r in _rows_by_id(p).items():
            outs[wi]["S"][s][bid] = {k: (int(cache[k][r]) if k in FLAGS else dc._value(cache[k][r], k))
                                     for k in names}
    return outs, p


def apply_batch(p, recs: dict, outs: list) -> None:
    """The j-th record of each world's step as masked commands, one kind at a time; the rows found by the park's ids."""
    I64, F64, BOOL = torch.int64, torch.float64, torch.bool
    dev, n = p.dev, p.n
    where = _rows_by_id(p)
    kinds = defaultdict(dict)
    for wi, rec in recs.items():
        kinds[rec[1]][wi] = rec
    for op, group in kinds.items():
        if op == "loadform":                                 # the park's table: every world's record the same
            if len(group) != len(outs) or len({rec[2] for rec in group.values()}) != 1:
                raise ValueError("a mid-run LoadFormations is the park's: every world at once, one table")
            p.load_formations(RELOADS[next(iter(group.values()))[2]])
            continue
        if op in ("setfree", "rigid", "boid", "troll", "inter", "follow", "orbit", "mushroom", "deads"):
            mask, ia, ib = [False] * n, [0] * n, [0] * n
            fa, fb = [0.0] * n, [0.0] * n
            exists = p.exists.tolist()
            for wi, rec in group.items():
                a = rec[2:]
                if op == "deads":                            # every row of the worlds that call it
                    for r in range(n):
                        mask[r] = mask[r] or int(p.world[r]) == wi
                    continue
                if op == "mushroom":                         # the world's first empty mushroom row
                    r = next(q for q in p.mush_rows[wi] if not exists[q])
                    mask[r], ia[r], fa[r], fb[r] = True, a[0], a[1], a[2]
                    continue
                if (wi, a[0]) not in where:
                    continue
                r = where[(wi, a[0])]
                mask[r] = True
                if op in ("setfree", "boid", "troll", "inter"):
                    ia[r] = a[1]
                elif op in ("follow", "orbit"):
                    ib[r], fa[r] = where.get((wi, a[1]), -1), a[2]
            m = torch.tensor(mask, dtype=BOOL, device=dev)
            A, B = torch.tensor(ia, dtype=I64, device=dev), torch.tensor(ib, dtype=I64, device=dev)
            FA, FB = torch.tensor(fa, dtype=F64, device=dev), torch.tensor(fb, dtype=F64, device=dev)
            if op == "setfree":
                p.set_free(m, A)
            elif op == "rigid":
                p.set_ball_rigid(m)
            elif op == "boid":
                p.set_boid(m, A)
            elif op == "troll":
                p.set_ball_troll(m, A)
            elif op == "inter":
                p.set_interactive(m, A)
            elif op == "follow":
                p.follow(m, B, FA)
            elif op == "orbit":
                p.orbit(m, B, FA)
            elif op == "mushroom":
                p.add_mushroom(m, A, FA, FB)
            else:
                p.bring_out_the_deads(m)
            where = _rows_by_id(p)
            continue
        dc.apply_batch(p, group, _rows_by_id(p, empty=True), n, outs)
        where = _rows_by_id(p)


# ---- the comparison ----

def differences(want: dict, got: dict, limit: int = 20) -> list:
    return dc.differences(want, got, limit)


def in_domain(want: dict) -> bool:
    return dc.in_domain(want)


def torch_domain(w: dict) -> bool:
    """A world VecCombatPark.evolve_torch holds (GOTO, STOP, MISSILE, FORMATION and the inert modes: no warp, follow
    or orbit)."""
    return not any(rec[1] in ("warp", "follow", "orbit") for rec in w["records"])


# ---- the same worlds on Destiny's own code (the conformance kit's compiled full harness: tests/full_cases.py) ----

def harness_case(cfg: dict, w: dict, cid: int, answers: list) -> dict:
    """A world as the full harness takes it (full_cases' records), as test_movement_vec_combat's _harness_case does for
    the worlds, with this module's records: SetBallFree, SetBallRigid, SetBoid, SetBallTroll,
    SetBallInteractive, FollowBall, Orbit, AddMushroom and BringOutTheDeads; ReserveFormationSlot-then-FormationFollow
    as the query and a FormationFollow with the slot the reference answered (`answers`, the reference's "A")."""
    import full_cases as fc
    char = lambda v: ((int(v) + 128) & 0xFF) - 128
    ans = iter(answers)

    def ball(b, step):
        return ("B", step, {"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                            "free": b["free"], "global": False, "massive": b["massive"],
                            "interactive": b["interactive"], "junk": False, "pos": list(b["pos"]),
                            "vel": list(b["vel"]), "agility": b["agility"], "sf": b["speed_fraction"],
                            "max_ang_vel": b["max_ang_vel"], "rot_agility": b["rot_agility"]})
    recs = [ball(s, 0) for s in w["ships"]]
    flat = [float(len(dc.FORMATIONS))]
    for vectors in dc.FORMATIONS:
        flat.append(float(len(vectors)))
        for v in vectors:
            flat.extend(float(c) for c in v)
    k = fc._k
    recs.append(k(0, fc.FORMATIONS_OP, d=flat))
    simple = {"goto": fc.GOTO_OP, "dir": fc.DIR_OP, "vel": fc.VEL_OP}
    for rec in w["records"]:
        s, op, a = rec[0], rec[1], rec[2:]
        if op in simple:
            recs.append(k(s, simple[op], a[0], d=a[1:4]))
        elif op == "stop":
            recs.append(k(s, fc.STOP_OP, a[0]))
        elif op == "warp":
            recs.append(k(s, fc.WARP_OP, a[0], w=a[5], d=(*a[1:4], a[4])))
        elif op == "add":
            recs.append(ball(a[0], s))
        elif op == "launch":
            recs.append(k(s, fc.LAUNCH_OP, a[0], a[1], a[2], w=(char(a[3]) == 1) | ((char(a[4]) != 0) << 1)))
        elif op == "mfollow":
            recs.append(k(s, fc.MISSILE_FOLLOW_OP, a[0], a[1], a[2]))
        elif op == "cloak":
            recs.append(k(s, fc.CLOAK_OP, a[0], w=a[1]))
        elif op == "uncloak":
            recs.append(k(s, fc.UNCLOAK_OP, a[0]))
        elif op == "remove":
            recs.append(k(s, fc.REMOVE_OP, a[0], w=0))
        elif op == "setform":
            recs.append(k(s, fc.SET_FORMATION_OP, a[0], w=a[1]))
        elif op == "formfollow":
            recs.append(k(s, fc.FORMATION_FOLLOW_OP, a[0], a[1], w=a[2]))
        elif op == "reserve_follow":
            slot = next(ans, -1)
            recs.append(k(s, fc.RESERVE_SLOT_Q, a[1]))
            if slot >= 0:
                recs.append(k(s, fc.FORMATION_FOLLOW_OP, a[0], a[1], w=slot))
        elif op == "setfree":
            recs.append(k(s, fc.FREE_OP, a[0], w=int(a[1] != 0)))
        elif op == "rigid":
            recs.append(k(s, fc.RIGID_OP, a[0]))
        elif op == "boid":
            recs.append(k(s, fc.BOID_OP, a[0], w=a[1]))
        elif op == "troll":
            recs.append(k(s, fc.TROLL_OP, a[0], w=a[1]))
        elif op == "inter":
            recs.append(k(s, fc.INTERACTIVE_OP, a[0], w=a[1]))
        elif op in ("follow", "orbit"):
            recs.append(k(s, fc.FOLLOW_OP if op == "follow" else fc.ORBIT_OP, a[0], a[1], d=(0.0, 0.0, 0.0, a[2])))
        elif op == "mushroom":
            recs.append(k(s, fc.MUSHROOM_OP, a[0], d=(a[1], a[2])))
        elif op == "deads":
            recs.append(k(s, fc.DEADS_OP))
        elif op == "loadform":
            flat = [float(len(RELOADS[a[0]]))]
            for vectors in RELOADS[a[0]]:
                flat.append(float(len(vectors)))
                for v in vectors:
                    flat.extend(float(c) for c in v)
            recs.append(k(s, fc.FORMATIONS_OP, d=flat))
        else:
            raise ValueError(op)
    return {"id": cid, "kind": "combat2", "friction": cfg["friction"], "tick": cfg["tick"], "steps": cfg["steps"],
            "new_orbit": cfg["new_orbit"], "iterative": False, "do": cfg["do"], "do_missiles_off": cfg["do_off"],
            "manual": False, "master": cfg["master"], "seed": 1, "start": w["start"], "records": recs}

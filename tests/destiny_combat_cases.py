# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the batched combat pieces (exact_ballpark.vec_combat.VecCombatPark), generated from a
seed, run by the scalar reference (exact_ballpark.full.FullBallpark, one park a world) and by the batch (one park for a
batch of worlds), and compared bit for bit after every step: every ball's state the batch holds, which balls exist,
the answers of ReserveFormationSlot, and the DoCollision events.

A batch is a configuration (friction, tick, dynamical orientation and its missile exclusion, the orbit law, server or
client view) and its worlds; a world is its ships (ids 1 to 5), a pool of missile ids (101 to 104) with an empty row
each, and records applied at given steps: step 0 is the setup, and after the records of each later step every park
evolves one tick. The formation table is the batch's (FORMATIONS), loaded in every world at step 0.

The domain the batch holds (exact_ballpark.vec_combat): at most one massive ball a world besides missiles (so no two
balls meet but a missile and its target), normal missiles, the gradient method; on the server one bubble a world,
kept alive by ship 1 (interactive, never removed). The reference is held to Destiny's compiled code by
tests/test_movement_full_oracle.py.
"""

from __future__ import annotations

import math
import random
import struct
from collections import defaultdict

import torch

from exact_ballpark.full import FullBallpark
from exact_ballpark.space import DestinyUndefined

POOL = (101, 102, 103, 104)
FORMATIONS = [
    [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0], [0.0, 1.0, -2.0], [0.0, -1.0, -2.0], [2.0, 0.5, 0.0]],
    [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.5, -0.25, 4.0], [10.0, 10.0, 10.0]],
    [[float(i), 0.0, -float(i) * 0.5] for i in range(17)],                    # 17 slots: ReserveFormationSlot refuses
]


def _unit(r: random.Random) -> list:
    while True:
        v = [r.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(sum(c * c for c in v))
        if n > 1e-6:
            return [c / n for c in v]


def _ship(r: random.Random, bid: int, centre, **kw) -> dict:
    u = _unit(r)
    d = r.uniform(500.0, 6000.0)
    b = {"id": bid, "mass": 10 ** r.uniform(6.0, 8.5), "radius": r.choice([2.0, r.uniform(10.0, 80.0),
                                                                          r.uniform(80.0, 400.0)]),
         "max_vel": r.choice([150.0, r.uniform(50.0, 450.0)]), "free": True, "massive": False,
         "interactive": r.random() < 0.5, "pos": [centre[i] + u[i] * d for i in range(3)],
         "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-120.0, 120.0) for _ in range(3)]]),
         "agility": r.uniform(0.2, 1.5), "speed_fraction": r.choice([1.0, r.uniform(0.2, 1.0)]),
         "max_ang_vel": r.choice([1.0, r.uniform(0.1, 2.0)]), "rot_agility": r.choice([0.9, r.uniform(0.1, 1.5)])}
    b.update(kw)
    return b


def _missile(r: random.Random, bid: int, centre) -> dict:
    return {"id": bid, "mass": 10 ** r.uniform(3.0, 5.0), "radius": r.uniform(1.0, 6.0),
            "max_vel": r.uniform(600.0, 4000.0), "free": True, "massive": False, "interactive": False,
            "pos": list(centre), "vel": [0.0, 0.0, 0.0], "agility": r.uniform(0.01, 0.2), "speed_fraction": 1.0,
            "max_ang_vel": r.uniform(0.5, 3.0), "rot_agility": r.uniform(0.2, 1.0)}


def world(r: random.Random, steps: int) -> dict:
    """One world: ships, a missile pool, and its records [(step, op, *args)]."""
    centre = [r.uniform(-2e5, 2e5) for _ in range(3)]
    n = r.randint(2, 5)
    ships = [_ship(r, 1, centre, interactive=True)]
    ships += [_ship(r, k, centre) for k in range(2, n + 1)]
    if r.random() < 0.25:                                   # a fixed launcher or leader
        ships[-1].update(free=False, vel=[0.0, 0.0, 0.0])
    heavy = r.choice([None] + [s["id"] for s in ships])     # at most one massive ship: no two balls meet
    for s in ships:
        s["massive"] = s["id"] == heavy
    recs = []
    ids = [s["id"] for s in ships]
    for s in ships:                                          # the ships' flight
        if s["free"] and r.random() < 0.7:
            k = r.choice([0, 0, 1, r.randint(1, steps)])
            how = r.choice(["goto", "dir", "dir", "stop"])
            if how == "goto":
                recs.append((k, "goto", s["id"], *[centre[i] + r.uniform(-2e4, 2e4) for i in range(3)]))
            elif how == "dir":
                recs.append((k, "dir", s["id"], *_unit(r)))
            else:
                recs.append((k, "stop", s["id"]))
    if r.random() < 0.45:                                    # formations
        leader = r.choice([s["id"] for s in ships if s["free"]])
        recs.append((0, "setform", leader, r.choice([0, 0, 1, 2, 3])))
        for s in ships:
            if s["id"] == leader or not s["free"] or r.random() < 0.3:
                continue
            k = r.choice([0, 1, r.randint(1, steps)])
            if r.random() < 0.6:
                recs.append((k, "reserve_follow", s["id"], leader))
            else:
                recs.append((k, "formfollow", s["id"], leader, r.choice([0, 1, 2, 5, 9, 16, 300])))
        if r.random() < 0.6:
            recs.append((0, "goto", leader, *[centre[i] + r.uniform(-3e4, 3e4) for i in range(3)]))
        for _ in range(r.randint(0, 2)):
            k = r.randint(2, steps)
            what = r.choice(["setform", "setform", "leave", "free", "reserve"])
            if what == "setform":
                recs.append((k, "setform", leader, r.choice([0, 1, 2, 3, 5, -1])))
            elif what == "leave":
                recs.append((k, "setform", leader, -1))
            elif what == "free":
                recs.append((k, "free", leader, r.choice([0, 1, 2, 7, 16, -1])))
            else:
                recs.append((k, "reserve", leader))
    # The batch's domain: at most one massive ball outside MISSILE mode a world. An uncloak, SetBallMassive, a warp's
    # drop-out and a massive missile that does not follow (no target, a cloaked or removed one) each make one: they
    # spend the world's one place, which the heavy ship takes if there is one.
    budget = [0 if heavy is not None else 1]

    def spend(want: bool) -> bool:
        if want and budget[0] > 0:
            budget[0] -= 1
            return True
        return False

    gone = {}                                                # the first step a ship is cloaked or removed
    if r.random() < 0.6:                                     # what happens to targets, owners and leaders
        for _ in range(r.randint(1, 2)):
            k = r.randint(1, steps)
            who = r.choice(ids[1:] or ids)
            what = r.choice(["cloak", "cloak", "uncloak_later", "remove", "warp", "massive", "vel"])
            if what == "cloak":
                recs.append((k, "cloak", who, r.choice([1, 1, 2, 3, 0, 129])))
                gone[who] = min(gone.get(who, k), k)
            elif what == "uncloak_later":
                recs.append((k, "cloak", who, 1))
                gone[who] = min(gone.get(who, k), k)
                if who == heavy or spend(True):
                    recs.append((r.randint(k, steps), "uncloak", who))
            elif what == "remove":
                recs.append((k, "remove", who))
                gone[who] = min(gone.get(who, k), k)
            elif what == "warp" and (who == heavy or spend(True)):
                u = _unit(r)
                recs.append((k, "warp", who, *[u[i] * 1e9 for i in range(3)], r.choice([0.0, 2e4]),
                             r.randint(1, 30)))
            elif what == "massive" and spend(True):
                recs.append((k, "massive", who, 1))
            else:
                recs.append((k, "vel", who, *[r.uniform(-300.0, 300.0) for _ in range(3)]))
    if r.random() < 0.3 and (heavy == 1 or spend(True)):     # ship 1, the keep-alive, cloaks and uncloaks
        recs.append((r.randint(1, steps), "cloak", 1, 1))
        recs.append((r.randint(1, steps), "uncloak", 1))
    missiles = list(POOL[:r.randint(0, len(POOL))])
    for mid in missiles:                                     # missiles: added, then launched or made to follow
        k = r.randint(1, max(1, steps - 6))
        recs.append((k, "add", _missile(r, mid, centre)))
        owner = r.choice(ids)
        target = r.choice([t for t in ids if t != owner] or ids)
        how = r.choice(["launch", "launch", "launch", "launch", "follow", "no_target", "self", "missile"])
        follows = how == "launch" and gone.get(target, steps + 1) > k
        massive = r.choice([0, 1, 1, 1, 256]) if follows else (1 if spend(r.random() < 0.6) else r.choice([0, 256]))
        if how == "follow":
            recs.append((k, "vel", mid, *[c * r.uniform(50.0, 300.0) for c in _unit(r)]))
            recs.append((k, "mfollow", mid, target, r.choice([owner, owner, owner, mid, target, 0])))
        elif how == "missile" and missiles.index(mid) > 0:
            other = missiles[missiles.index(mid) - 1]
            recs.append((k, "launch", mid, other, owner, r.randrange(3), massive))
        else:
            dst = {"launch": target, "no_target": 999, "self": owner}.get(how, target)
            recs.append((k, "launch", mid, dst, owner, r.choice([0, 1, 1, 2, 257]), massive))
        if r.random() < 0.25:
            recs.append((r.randint(k + 1, steps), "remove", mid))       # the game's time-out
    recs.sort(key=lambda rec: rec[0])
    return {"ships": ships, "pool": list(POOL), "records": recs, "start": r.randrange(0, 1000)}


def batches(seed: int, n_batches: int = 24, worlds: int = 24) -> list:
    """The scenarios of a seed: batches of worlds, each batch one configuration (even batches without dynamical
    orientation, odd ones with it)."""
    r = random.Random(seed)
    out = []
    for b in range(n_batches):
        do = b % 2 == 1
        cfg = {"do": do, "do_off": do and r.random() < 0.4, "master": r.random() < 0.6,
               "tick": r.choice([250, 500, 1000, 1000, 1000, 2000]),
               "friction": r.choice([1.0e6] * 5 + [10 ** r.uniform(5.0, 7.0)]),
               "new_orbit": r.random() < 0.3, "steps": r.randint(10, 24)}
        cfg["worlds"] = [world(r, cfg["steps"]) for _ in range(worlds)]
        out.append(cfg)
    return out


# ---- the reference ----

STATE = ("pos", "vel", "old_pos", "old_vel", "goto", "last_g", "mode", "effect_stamp", "follow_id", "owner_id",
         "massive", "cloaked", "speed_fraction", "follow_range", "last_collision", "formation_id", "formation_slots",
         "mass", "radius", "max_vel", "agility", "time_factor")
DO_STATE = ("rot", "old_rot", "ang_vel", "old_ang_vel", "torque", "rot_roll")
YPR_STATE = ("yaw", "old_yaw", "pitch", "old_pitch", "roll", "old_roll", "roll_speed", "old_roll_speed", "yaw_delta")
INTS = ("mode", "effect_stamp", "follow_id", "owner_id", "massive", "cloaked", "formation_id", "formation_slots")
_REF_NAME = {"massive": "is_massive", "cloaked": "is_cloaked", "roll": "roll_angle"}


def fields(do: bool) -> tuple:
    return STATE + (DO_STATE if do else YPR_STATE)


def _value(v, k: str):
    if isinstance(v, (list, tuple)):
        return tuple(float(c) for c in v)
    return int(v) if k in INTS else float(v)


def _ref_row(b, do: bool) -> dict:
    return {k: _value(getattr(b, _REF_NAME.get(k, k)), k) for k in fields(do)}


def _add_ref(park: FullBallpark, b: dict) -> None:
    park.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], b["free"], False, b["massive"], b["interactive"],
                  False, *b["pos"], *b["vel"], b["agility"], b["speed_fraction"], b["max_ang_vel"], b["rot_agility"])


def _apply_ref(park: FullBallpark, rec: tuple, answers: list) -> None:
    op, a = rec[1], rec[2:]
    if op == "goto":
        park.goto_point(a[0], *a[1:4])
    elif op == "dir":
        park.goto_direction(a[0], *a[1:4])
    elif op == "stop":
        park.stop(a[0])
    elif op == "warp":
        park.warp_to(a[0], *a[1:4], a[4], a[5])
    elif op == "vel":
        park.set_velocity(a[0], *a[1:4])
    elif op == "massive":
        park.set_massive(a[0], bool(a[1]))
    elif op == "add":
        _add_ref(park, a[0])
    elif op == "launch":
        park.launch_missile(*a[:5])
    elif op == "mfollow":
        park.missile_follow(*a[:3])
    elif op == "cloak":
        park.cloak_ball(a[0], a[1])
    elif op == "uncloak":
        park.uncloak_ball(a[0])
    elif op == "remove":
        if a[0] in park.balls:
            park.remove_ball(a[0])
    elif op == "setform":
        park.set_ball_formation(a[0], a[1])
    elif op == "formfollow":
        park.formation_follow(a[0], a[1], a[2])
    elif op == "reserve_follow":
        slot = park.reserve_formation_slot(a[1]) if a[1] in park.balls else -1
        answers.append(slot)
        if slot >= 0:
            park.formation_follow(a[0], a[1], slot)
    elif op == "reserve":
        answers.append(park.reserve_formation_slot(a[0]) if a[0] in park.balls else -1)
    elif op == "free":
        park.free_formation_slot(a[0], a[1])
    else:
        raise ValueError(op)


def reference(cfg: dict, w: dict, trace=None) -> dict:
    """One world on FullBallpark: {"S": {step: {id: row}}, "E": [(time, missile, collider)], "A": answers,
    "B": the steps whose balls were not all in one bubble (outside the batch's domain), "X": DestinyUndefined's text
    or None}."""
    park = FullBallpark(friction=cfg["friction"], tick_interval_ms=cfg["tick"], use_new_orbit=cfg["new_orbit"],
                        is_master=cfg["master"], dynamical_orientation=cfg["do"],
                        disable_do_for_missiles=cfg["do_off"])
    park.current_time = w["start"]
    park.trace = trace
    for s in w["ships"]:
        _add_ref(park, s)
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
            out["S"][s] = {b.id: _ref_row(b, cfg["do"]) for b in park.balls.values()}
            if cfg["master"] and len({b.bubble for b in park.balls.values()}) > 1:
                out["B"].append(s)
    except DestinyUndefined as exc:
        out["X"] = str(exc)
    ev = [(t, bid, args[0], tuple(args[1:4])) for (t, name, bid, *args) in park.events if name == "DoCollision"]
    out["E"] = [(t, bid, c) for t, bid, c, _ in ev]
    out["Z"] = all(bits(x) == 0 for *_, lc in ev for x in lc)        # each event's mLastC is +0 (a normal missile's)
    return out


# ---- the batch ----

def rows_for(cfg: dict) -> tuple:
    """The batch's rows: each world's ships, then its pool's empty rows; (rows, {(world, id): row})."""
    rows, where = [], {}
    bubble = 0 if cfg["master"] else -1
    for wi, w in enumerate(cfg["worlds"]):
        for s in w["ships"]:
            where[(wi, s["id"])] = len(rows)
            rows.append({**s, "world": wi, "time": w["start"], "bubble": bubble})
        for mid in w["pool"]:
            where[(wi, mid)] = len(rows)
            rows.append({"id": mid, "world": wi, "time": w["start"], "exists": False, "mass": 1.0, "radius": 1.0,
                         "max_vel": 1.0, "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "agility": 1.0,
                         "speed_fraction": 1.0, "bubble": bubble})
    return rows, where


def batched(cfg: dict, device: str = "cuda", graph: bool = False, path: str = "kernel") -> tuple:
    """The batch on VecCombatPark: (per world the layout of `reference`, the park)."""
    import exact_ballpark.vec_combat as vc
    rows, where = rows_for(cfg)
    p = vc.VecCombatPark(len(rows), cfg["friction"], cfg["tick"], device, cfg["do"], cfg["new_orbit"],
                         master=cfg["master"], disable_do_for_missiles=cfg["do_off"])
    p.load(rows)
    p.load_formations(FORMATIONS)
    W = len(cfg["worlds"])
    outs = [{"S": {}, "E": [], "A": [], "B": [], "X": None} for _ in range(W)]
    by_step = [defaultdict(list) for _ in range(W)]
    for wi, w in enumerate(cfg["worlds"]):
        for rec in w["records"]:
            by_step[wi][rec[0]].append(rec)
    n = len(rows)
    runner = None
    names = fields(cfg["do"])
    for s in range(cfg["steps"] + 1):
        slots = max(len(by_step[wi].get(s, ())) for wi in range(W))
        for j in range(slots):
            recs = {wi: by_step[wi][s][j] for wi in range(W) if j < len(by_step[wi].get(s, ()))}
            apply_batch(p, recs, where, n, outs)
        if s > 0:
            if graph:
                if runner is None:
                    from exact_ballpark.vec_combat_cuda import CombatGraph
                    runner = CombatGraph(p)
                runner.tick()
                p.after_tick()
            else:
                p.evolve() if path == "kernel" else p.evolve_torch()
            for (wl, t, mid, col) in p.events():
                outs[wl]["E"].append((t, mid, col))
        cache = {k: getattr(p, k).cpu().tolist() for k in names}
        exists = p.exists.cpu().tolist()
        for wi in range(W):
            outs[wi]["S"][s] = {}
        for (wi, bid), r in where.items():
            if exists[r]:
                outs[wi]["S"][s][bid] = {k: _value(cache[k][r], k) for k in names}
    return outs, p


def apply_batch(p, recs: dict, where: dict, n: int, outs: list) -> None:
    """The j-th record of each world's step, as masked commands, one kind at a time (the worlds are independent)."""
    I64, F64, BOOL = torch.int64, torch.float64, torch.bool
    dev = p.dev
    kinds = defaultdict(dict)
    for wi, rec in recs.items():
        kinds[rec[1]][wi] = rec
    for op, group in kinds.items():
        mask = [False] * n
        ia, ib, ic, idd = [0] * n, [0] * n, [0] * n, [0] * n
        f3 = [[0.0, 0.0, 0.0] for _ in range(n)]
        f1 = [0.0] * n
        adds = []
        for wi, rec in group.items():
            a = rec[2:]
            if op == "add":
                adds.append((where[(wi, a[0]["id"])], a[0]))
                continue
            if (wi, a[0]) not in where:
                continue
            r = where[(wi, a[0])]
            mask[r] = True
            if op in ("goto", "dir", "vel"):
                f3[r] = list(a[1:4])
            elif op == "warp":
                f3[r], f1[r], ia[r] = list(a[1:4]), a[4], a[5]
            elif op in ("massive", "cloak", "setform", "free", "reserve_follow"):
                ia[r] = a[1]
            elif op == "launch":
                ia[r], ib[r], ic[r], idd[r] = a[1], a[2], a[3], a[4]
            elif op in ("mfollow", "formfollow"):
                ia[r], ib[r] = a[1], a[2]
        m = torch.tensor(mask, dtype=BOOL, device=dev)
        A, B, C, D = (torch.tensor(x, dtype=I64, device=dev) for x in (ia, ib, ic, idd))
        X3 = torch.tensor(f3, dtype=F64, device=dev)
        live = m & p.exists
        if op == "add":
            adds.sort(key=lambda x: x[0])
            mm = torch.zeros(n, dtype=BOOL, device=dev)
            mm[[r for r, _ in adds]] = True
            p.add_balls(mm, [b for _, b in adds])
        elif op == "goto":
            p.goto_point(live, X3)
        elif op == "dir":
            p.goto_direction(live, X3)
        elif op == "stop":
            p.stop(live)
        elif op == "warp":
            p.warp_to(live, X3, torch.tensor(f1, dtype=F64, device=dev), A)
        elif op == "vel":
            p.set_velocity(live, X3)
        elif op == "massive":
            p.massive = torch.where(live, A != 0, p.massive)
        elif op == "launch":
            p.launch_missile(m, A, B, C, D)
        elif op == "mfollow":
            p.missile_follow(m, A, B)
        elif op == "cloak":
            p.cloak(m, A)
        elif op == "uncloak":
            p.uncloak(m)
        elif op == "remove":
            p.remove(m)
        elif op == "setform":
            p.set_ball_formation(m, A)
        elif op == "formfollow":
            p.formation_follow(m, A, B)
        elif op in ("reserve_follow", "reserve", "free"):
            for wi, rec in sorted(group.items()):
                a = rec[2:]
                if (wi, a[0]) not in where:
                    continue
                r = where[(wi, a[0])]
                if op == "reserve_follow":
                    lr = where.get((wi, a[1]))
                    slot = p.reserve_formation_slot(lr) if lr is not None and bool(p.exists[lr]) else -1
                    outs[wi]["A"].append(slot)
                    if slot >= 0:
                        one = torch.zeros(n, dtype=BOOL, device=dev)
                        one[r] = True
                        p.formation_follow(one, torch.full((n,), a[1], dtype=I64, device=dev),
                                           torch.full((n,), slot, dtype=I64, device=dev))
                elif op == "reserve":
                    outs[wi]["A"].append(p.reserve_formation_slot(r) if bool(p.exists[r]) else -1)
                elif bool(p.exists[r]):
                    p.free_formation_slot(r, a[1])
        else:
            raise ValueError(op)


# ---- the comparison ----

def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, float) and isinstance(b, float) and (bits(a) == bits(b) or (a != a and b != b))
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def differences(want: dict, got: dict, limit: int = 20) -> list:
    """Every difference of one world: (kind, step, ball, field, want, got)."""
    bad = []
    for s, balls in want["S"].items():
        g = got["S"].get(s, {})
        if set(balls) != set(g):
            bad.append(("balls", s, None, None, sorted(balls), sorted(g)))
        for bid, row in balls.items():
            if bid not in g:
                continue
            for k, v in row.items():
                if not same(v, g[bid][k]):
                    bad.append(("field", s, bid, k, v, g[bid][k]))
                    if len(bad) >= limit:
                        return bad
    if want["E"] != got["E"]:
        bad.append(("events", None, None, None, want["E"], got["E"]))
    if want["A"] != got["A"]:
        bad.append(("answers", None, None, None, want["A"], got["A"]))
    return bad


def in_domain(want: dict) -> bool:
    """A world the batch is meant to hold: Destiny's result defined, one bubble at every step."""
    return want["X"] is None and not want["B"]


def torch_domain(w: dict) -> bool:
    """A world VecCombatPark.evolve_torch holds (GOTO, STOP, MISSILE and FORMATION: no warp)."""
    return not any(rec[1] == "warp" for rec in w["records"])

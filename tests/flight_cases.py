# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for Destiny's flight modes on the GPU, generated from a seed, in
tests/full_cases.py's records: run by the reference (exact_ballpark.full, FullBallpark), by Destiny's own code compiled
verbatim (conformance/build_full.py, full_cases.HARNESS), and by the batched modes
(exact_ballpark.vec_modes, one CUDA kernel a tick), all compared bit for bit.

Every scenario is a client's view (every bubble live, as the batched park has no bubbles) of free, non-massive balls
(they never collide: the batched park has no collisions), at most one lone massive ball among them (massive balls
collide only with massive balls), with and without dynamical orientation. Families:

- "follow" and "orbit": followers and orbiters (both laws; on the target's centre, at, inside and far outside the
  range; heavy ships for OrbitThrust; zero cruise speed) of a target that turns, stops, goes on or is fixed, ids below
  and above the target's, chains; re-commanded in flight (follow, orbit, stop, goto, a new target), and every setter
  in flight on any of them (speed fraction, mass, agility, maximum speed, radius, velocity, maximum angular speed,
  rotational agility, angular velocity, rotation), now and then with the values Destiny refuses.
- "warp": a ship warping (short, long and very long, slow and forced starts, warp factors, minimum ranges), with
  followers and orbiters of ids below and above it (interactive orbiters fly on by GotoDirection, the others stop),
  commands during alignment and warp proper (Stop: the public code's mid-warp stop; a new or repeated WarpTo, GotoPoint,
  FollowBall, Orbit, EntityWarpIn) and the setters during warp (the maximum speed moves the drop-out threshold).
- "warpin": EntityWarpIn from far, near and the origin, early in the park's life, with followers.
- "mixed": three to six balls under random commands and setters at random steps.

`from_mode_case` turns a scenario of tests/mode_cases.py (ModeBallpark's) into these records (its balls non-massive,
its guard family left out), so mode_cases' families also run with dynamical orientation through FullBallpark.
"""

from __future__ import annotations

import math
import random

import full_cases as fc
import mode_cases as mc

FAMILY_SIZES = {"follow": 140, "orbit": 220, "warp": 260, "warpin": 50, "mixed": 130}
SETTER_OPS = (fc.SF_OP, fc.MASS_OP, fc.AGILITY_OP, fc.MAXVEL_OP, fc.RADIUS_OP, fc.VEL_OP, fc.MAX_ANGVEL_OP,
              fc.ROT_AGILITY_OP, fc.ANGVEL_OP, fc.ROTATION_OP)
REFUSED = (-1.0, 0.0, math.nan, math.inf, -math.inf)
# a NaN or infinite mass or agility is stored (their guards are `<= 0`) and makes the positions NaN, whose old-style
# orbit then converts a NaN to an integer, which C leaves undefined (and the reference refuses): left out here
REFUSED_MASS = (-1.0, 0.0, -math.inf)


def _ship(r: random.Random, bid: int, **kw) -> tuple:
    """A free, non-massive ship with a random hull, as an AddBall record; keywords override."""
    b = {"pos": [r.uniform(-2e4, 2e4) for _ in range(3)], "mass": 10 ** r.uniform(5.5, 9.5),
         "radius": r.choice([0.0, r.uniform(1.0, 50.0), r.uniform(50.0, 500.0)]),
         "max_vel": r.choice([r.uniform(50.0, 1500.0), r.uniform(50.0, 1500.0), r.uniform(1.0, 199.0)]),
         "agility": r.choice([r.uniform(0.05, 0.5), r.uniform(0.5, 2.0)]), "sf": r.choice([1.0, r.uniform(0.1, 1.0)]),
         "max_ang_vel": r.choice([1.0, r.uniform(0.02, 4.0), 0.0]), "rot_agility": r.choice([0.9, r.uniform(0.02, 3.0)]),
         "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-300, 300) for _ in range(3)]]), "free": True, "massive": False,
         "interactive": r.random() < 0.5}
    b.update(kw)
    return fc._ball(r, bid, **b)


def _value(r: random.Random, op: int) -> tuple:
    """A setter's record values for op, refused once in a while."""
    refuse = r.random() < 0.12
    if op == fc.SF_OP:
        v = r.choice(REFUSED + (1.5, -0.2)) if refuse else r.choice([r.uniform(0.0, 1.0), 1.0, 0.0])
    elif op == fc.MASS_OP:
        v = r.choice(REFUSED_MASS) if refuse else 10 ** r.uniform(5.0, 9.8)
    elif op == fc.AGILITY_OP:
        v = r.choice(REFUSED_MASS) if refuse else r.uniform(0.03, 3.0)
    elif op == fc.MAXVEL_OP:
        v = r.choice((-5.0, -0.0, -math.inf)) if refuse else r.choice([r.uniform(1.0, 1500.0), 0.0, r.uniform(1.0, 60.0)])
    elif op == fc.RADIUS_OP:
        v = r.choice((-5.0, -0.0)) if refuse else r.uniform(0.0, 800.0)
    elif op == fc.MAX_ANGVEL_OP:
        return (r.choice((-1.0, -0.0)) if refuse else r.choice([r.uniform(0.01, 5.0), 0.0]),)
    elif op == fc.ROT_AGILITY_OP:
        return (r.choice((-1.0, 0.0)) if refuse else r.uniform(0.01, 4.0),)
    elif op == fc.ANGVEL_OP:
        return tuple(r.uniform(-2.0, 2.0) for _ in range(3))
    elif op == fc.ROTATION_OP:
        return (0.0, 0.0, 0.0, 0.0) if refuse else tuple(r.gauss(0, 1) for _ in range(4))
    elif op == fc.VEL_OP:
        return tuple(r.choice([[0.0, 0.0, 0.0], [r.uniform(-500, 500) for _ in range(3)]]))
    else:
        raise ValueError(op)
    return fc._xyz(rr=v)


def _setters(r: random.Random, recs: list, ids: list, steps: int, n: int) -> None:
    """n setters at random steps on random balls."""
    for _ in range(n):
        op = r.choice(SETTER_OPS)
        recs.append(fc._k(r.randint(1, steps), op, r.choice(ids), d=_value(r, op)))


def _range(r: random.Random) -> float:
    return r.choice([0.0, 1.0, r.uniform(0.0, 500.0), r.uniform(0.0, 5000.0), r.uniform(500.0, 30000.0),
                     -r.uniform(0.0, 300.0)])


def _near(r: random.Random, centre: list, rr: float) -> list:
    kind = r.choice(["inside", "at", "outside", "far", "on", "axis"])
    if kind == "on":
        return list(centre)
    if kind == "axis":
        return [centre[0] + abs(rr), centre[1], centre[2]]
    d = {"inside": r.uniform(0.0, 1.0) * abs(rr), "at": abs(rr) * r.uniform(0.98, 1.02),
         "outside": abs(rr) * r.uniform(1.0, 3.0) + r.uniform(0.0, 2000.0), "far": r.uniform(1e4, 2e5)}[kind]
    u = fc._unit(r)
    return [centre[i] + d * u[i] for i in range(3)]


def _orbit_range(r: random.Random, b: dict, t: dict) -> float:
    """A range never making r exactly 0 (an old-style orbiter on its target's centre with r = 0 is undefined)."""
    rng = _range(r)
    f = lambda x: fc.f32(x)
    return 1.0 if f(rng) + f(b["radius"]) + f(t["radius"]) == 0.0 else rng


def _case(out: list, r: random.Random, kind: str, recs: list, steps: int, do: bool, new_orbit: bool) -> None:
    fc.case(out, r, kind, recs, steps, master=False, tick=r.choice([250, 500, 1000, 1000, 1000, 2000]), do=do,
            new_orbit=new_orbit, friction=r.choice([1.0e6] * 6 + [10 ** r.uniform(-1.0, 0.0), 10 ** r.uniform(5.0, 7.0)]))


def follow_cases(r: random.Random, out: list, n: int, orbit: bool) -> None:
    for _ in range(n):
        do, new = r.random() < 0.55, r.random() < 0.5
        ids = fc._ids(r, r.randint(2, 4))
        tid = ids[0]
        target = _ship(r, tid, free=r.random() < 0.85, vel=r.choice([[0.0] * 3, [r.uniform(-60, 60) for _ in range(3)]]))
        recs = [target]
        steps = r.randint(8, 50)
        tb = target[2]
        if tb["free"]:
            for k in range(r.choice([0, 1]), steps, r.choice([3, 5, 8, 50])):   # the target turns
                recs.append(fc._k(k, fc.GOTO_OP, tid, d=[r.uniform(-3e4, 3e4) for _ in range(3)]))
            if r.random() < 0.2:
                recs.append(fc._k(r.randint(1, steps), fc.STOP_OP, tid))
        for bid in ids[1:]:
            rec = _ship(r, bid)
            b = rec[2]
            rng = _orbit_range(r, b, tb) if orbit else _range(r)
            if orbit and r.random() < 0.15:
                b["sf"] = 0.0
            if orbit and r.random() < 0.3:
                b["mass"], b["agility"] = 10 ** r.uniform(8.5, 9.5), r.uniform(1.0, 2.0)
            b["pos"] = _near(r, tb["pos"], fc.f32(rng) + fc.f32(b["radius"]) + fc.f32(tb["radius"]))
            recs.append(rec)
            op = fc.ORBIT_OP if orbit else fc.FOLLOW_OP
            recs.append(fc._k(r.choice([0, 1, 1, 3]), op, bid, tid, d=fc._xyz(rr=rng)))
            if r.random() < 0.35:                                               # re-commanded in flight
                k = r.randint(2, steps)
                what = r.choice([fc.FOLLOW_OP, fc.ORBIT_OP, fc.STOP_OP, fc.GOTO_OP, fc.DIR_OP])
                if what in (fc.FOLLOW_OP, fc.ORBIT_OP):
                    other = r.choice([x for x in ids if x != bid])
                    recs.append(fc._k(k, what, bid, other, d=fc._xyz(rr=_orbit_range(r, b, tb))))
                elif what == fc.STOP_OP:
                    recs.append(fc._k(k, what, bid))
                else:
                    recs.append(fc._k(k, what, bid, d=[r.uniform(-3e4, 3e4) for _ in range(3)]))
        if len(ids) > 2 and r.random() < 0.3:                                   # a chain
            recs.append(fc._k(1, fc.FOLLOW_OP, ids[1], ids[-1], d=fc._xyz(rr=_range(r))))
        _setters(r, recs, ids, steps, r.randint(0, 8))
        _case(out, r, "orbit" if orbit else "follow", recs, steps, do, new)


def _warp_record(r: random.Random, step: int, bid: int, pos: list, dist: float) -> tuple:
    u = fc._unit(r)
    w = r.choice([r.randint(1500, 6000), r.randint(1500, 20000), 3000, 10000, r.randint(100, 1500)])
    mr = r.choice([0.0, 20000.0, r.uniform(0.0, 1e6), -r.uniform(1.0, 100.0), dist * r.uniform(1.0, 1.5)])
    return fc._k(step, fc.WARP_OP, bid, w=w, d=(*(pos[i] + dist * u[i] for i in range(3)), mr)), u


def warp_cases(r: random.Random, out: list, n: int) -> None:
    for _ in range(n):
        do, new = r.random() < 0.55, r.random() < 0.5
        ids = fc._ids(r, r.randint(1, 4))
        wid = r.choice(ids)
        kind = r.choice(["long", "medium", "medium", "short", "slow", "still", "very_long"])
        dist = {"long": 10 ** r.uniform(9.0, 11.5), "very_long": 10 ** r.uniform(11.5, 12.5),
                "medium": 10 ** r.uniform(5.0, 8.5), "short": r.uniform(0.0, 99999.0), "slow": 10 ** r.uniform(6.0, 9.0),
                "still": 10 ** r.uniform(6.0, 9.0)}[kind]
        rec = _ship(r, wid, mass=10 ** r.uniform(5.5, 8.5), massive=r.random() < 0.3)
        w = rec[2]
        steps = r.randint(10, 60)
        k0 = r.choice([0, 1, 1, 2])
        wrec, u = _warp_record(r, k0, wid, w["pos"], dist)
        if kind == "slow":
            w["sf"] = r.uniform(0.05, 0.7)
            steps = r.randint(150, 200)
        elif kind == "still":
            w["max_vel"], w["vel"] = 0.0, [0.0, 0.0, 0.0]
            steps = r.randint(150, 200)
        elif r.random() < 0.5:                                                  # aligned and fast: at once
            w["vel"] = [u[i] * w["max_vel"] * r.uniform(0.76, 1.0) for i in range(3)]
        recs = [rec, wrec]
        for bid in ids:                                                         # its followers, ids below and above
            if bid == wid:
                continue
            f = _ship(r, bid, interactive=r.random() < 0.6)
            rng = _orbit_range(r, f[2], w)
            f[2]["pos"] = _near(r, w["pos"], fc.f32(rng) + fc.f32(f[2]["radius"]) + fc.f32(w["radius"]))
            recs.append(f)
            recs.append(fc._k(r.choice([0, 1]), r.choice([fc.FOLLOW_OP, fc.ORBIT_OP, fc.ORBIT_OP]), bid, wid,
                              d=fc._xyz(rr=rng)))
        if r.random() < 0.6:                                                    # a command during the warp
            k = r.randint(k0 + 1, max(k0 + 1, steps - 1))
            what = r.choice(["stop", "stop", "repeat", "new", "goto", "follow", "orbit", "warpin", "dir"])
            if what == "stop":
                recs.append(fc._k(k, fc.STOP_OP, wid))
            elif what == "repeat":
                d = list(wrec[7])
                if r.random() < 0.5:
                    d[3] = r.choice([d[3], 5000.0])
                recs.append(fc._k(k, fc.WARP_OP, wid, w=r.choice([wrec[6], wrec[6] + 1]), d=tuple(d)))
            elif what == "new":
                recs.append(_warp_record(r, k, wid, w["pos"], 10 ** r.uniform(6, 11))[0])
            elif what == "goto":
                recs.append(fc._k(k, fc.GOTO_OP, wid, d=wrec[7][:3]))
            elif what == "dir":
                recs.append(fc._k(k, fc.DIR_OP, wid, d=fc._unit(r)))
            elif what in ("follow", "orbit") and len(ids) > 1:
                other = r.choice([x for x in ids if x != wid])
                recs.append(fc._k(k, fc.FOLLOW_OP if what == "follow" else fc.ORBIT_OP, wid, other,
                                  d=fc._xyz(rr=_range(r))))
            elif what == "warpin":
                recs.append(fc._k(k, fc.WARPIN_OP, wid, w=r.randint(1, 20000),
                                  d=[r.uniform(-3e11, 3e11) for _ in range(3)]))
        _setters(r, recs, ids, steps, r.randint(0, 6))
        if r.random() < 0.3:                                                    # the maximum speed during the warp
            recs.append(fc._k(r.randint(k0 + 1, steps), fc.MAXVEL_OP, wid, d=_value(r, fc.MAXVEL_OP)))
        _case(out, r, "warp", recs, steps, do, new)


def warpin_cases(r: random.Random, out: list, n: int) -> None:
    for _ in range(n):
        do = r.random() < 0.55
        ids = fc._ids(r, r.randint(1, 3))
        wid = ids[0]
        rec = _ship(r, wid, pos=r.choice([[0.0, 0.0, 0.0], [r.uniform(-1e7, 1e7) for _ in range(3)]]))
        kind = r.choice(["far", "far", "origin", "close"])
        dest = {"far": [r.uniform(-3e11, 3e11) for _ in range(3)], "origin": [0.0, 0.0, 0.0],
                "close": [rec[2]["pos"][i] + r.uniform(-5e4, 5e4) for i in range(3)]}[kind]
        if kind == "origin" and math.dist(rec[2]["pos"], dest) < 1e5:
            rec[2]["pos"] = [2e6, 0.0, 0.0]
        recs = [rec]
        for bid in ids[1:]:
            f = _ship(r, bid, pos=[rec[2]["pos"][i] + r.uniform(-3e3, 3e3) for i in range(3)])
            recs += [f, fc._k(0, r.choice([fc.FOLLOW_OP, fc.ORBIT_OP]), bid, wid, d=fc._xyz(rr=r.uniform(0, 2000)))]
        steps = r.randint(3, 40)
        recs.append(fc._k(r.randint(1, 3), fc.WARPIN_OP, wid, w=r.randint(1, 20000), d=dest))
        _setters(r, recs, ids, steps, r.randint(0, 4))
        fc.case(out, r, "warpin", recs, steps, master=False, tick=r.choice([250, 1000, 1000, 2000]), do=do,
                new_orbit=r.random() < 0.5, start=r.choice([0, 3, 5, 7, r.randrange(0, 100000)]))


def mixed_cases(r: random.Random, out: list, n: int) -> None:
    for _ in range(n):
        do, new = r.random() < 0.55, r.random() < 0.5
        ids = fc._ids(r, r.randint(3, 6))
        recs = [_ship(r, bid, pos=[r.uniform(-2e4, 2e4) for _ in range(3)], free=r.random() < 0.9) for bid in ids]
        steps = r.randint(10, 70)
        for _ in range(r.randint(2, 10)):
            k = r.randint(0, steps)
            a, b = r.choice(ids), r.choice(ids)
            op = r.choice([fc.GOTO_OP, fc.DIR_OP, fc.STOP_OP, fc.FOLLOW_OP, fc.FOLLOW_OP, fc.ORBIT_OP, fc.ORBIT_OP,
                           fc.WARP_OP])
            if op in (fc.GOTO_OP, fc.DIR_OP):
                recs.append(fc._k(k, op, a, d=[r.uniform(-3e4, 3e4) for _ in range(3)]))
            elif op == fc.STOP_OP:
                recs.append(fc._k(k, op, a))
            elif op == fc.WARP_OP:
                v = fc._unit(r)
                d = r.choice([5e4, 10 ** r.uniform(5.5, 9)])
                recs.append(fc._k(k, op, a, w=r.randint(2000, 20000),
                                  d=(v[0] * d, v[1] * d, v[2] * d, r.choice([0.0, 20000.0]))))
            else:
                recs.append(fc._k(k, op, a, b, d=fc._xyz(rr=_range(r))))
        _setters(r, recs, ids, steps, r.randint(2, 12))
        _case(out, r, "mixed", recs, steps, do, new)


def cases(seed: int, sizes: dict | None = None) -> list:
    sizes = dict(FAMILY_SIZES, **(sizes or {}))
    r = random.Random(9000 + seed)
    out: list = []
    follow_cases(r, out, sizes["follow"], orbit=False)
    follow_cases(r, out, sizes["orbit"], orbit=True)
    warp_cases(r, out, sizes["warp"])
    warpin_cases(r, out, sizes["warpin"])
    mixed_cases(r, out, sizes["mixed"])
    return out


# ============================================================================== mode_cases with dynamical orientation

_MC_OPS = {mc.GOTO_OP: fc.GOTO_OP, mc.DIR_OP: fc.DIR_OP, mc.STOP_OP: fc.STOP_OP, mc.FOLLOW_OP: fc.FOLLOW_OP,
           mc.ORBIT_OP: fc.ORBIT_OP, mc.WARP_OP: fc.WARP_OP, mc.WARPIN_OP: fc.WARPIN_OP}


def from_mode_case(c: dict, do: bool) -> dict | None:
    """A mode_cases scenario as full_cases records for FullBallpark (a client's view; its balls non-massive, so they
    never collide; Destiny's turning defaults), or None for one the batched park and FullBallpark cannot share: the
    guard family (moribund, cloaked and other-bubble targets need the server's bookkeeping) and inert modes."""
    if c["kind"] == "guards" or any(b["mode"] != mc.STOP or b["moribund"] or b["cloaked"] for b in c["balls"]):
        return None
    recs = [("B", 0, {"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                      "free": b["free"], "global": False, "massive": False, "interactive": b["interactive"],
                      "junk": False, "pos": list(b["pos"]), "vel": list(b["vel"]), "agility": b["agility"],
                      "sf": b["sf"], "max_ang_vel": 1.0, "rot_agility": 0.9}) for b in c["balls"]]
    for (step, op, a, b, x, y, z, rr, w) in c["cmds"]:
        recs.append(fc._k(step, _MC_OPS[op], a, b, w=w, d=(x, y, z, rr)))
    return {"id": c["id"], "kind": "mc_" + c["kind"], "friction": c["friction"], "tick": c["tick"],
            "steps": c["steps"], "new_orbit": c["new_orbit"], "iterative": False, "do": do, "do_missiles_off": False,
            "manual": False, "master": False, "seed": 1, "start": c["start"],
            "records": sorted(recs, key=lambda rec: rec[1])}

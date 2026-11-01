# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for Destiny's turning and ship values set in flight, generated from a seed,
in tests/full_cases.py's records: run by the reference (exact_ballpark.full), by Destiny's own code compiled verbatim with
DirectXMath (conformance/build_full.py, full_cases.HARNESS) and, for the batch-compatible families, by the
batched turning (exact_ballpark.vec_orient), all compared bit for bit.

Families:
- "vec" and "vec_do" (batch-compatible): one to six free, non-massive ships turning hard in GOTO, GotoDirection and STOP
  (points behind them, beside them, far and near), with and without dynamical orientation, and every setter in flight
  (speed fraction, mass, agility, maximum speed, radius, velocity, maximum angular speed, rotational agility, angular
  velocity, rotation), each also with the values Destiny refuses (negative, zero, NaN, infinite).
- "turned_modes": turned ships through every flight mode the reference covers: FOLLOW and ORBIT (both laws) behind a
  turning leader, WARP aligning while turned and dropping out, a missile launched from a turned launcher at rest
  (MOV-725), formation followers of a turned leader (MOV-726), and GetRotatedVector queries of turned balls; with and
  without dynamical orientation (without it every turned rotation goes through DirectXMath).
- "setters": massive ships close together, both collision methods, server and client views, with the setters in flight
  (SetMaxSpeed and SetBallRadius re-file the ball, mass and agility refresh the decay factor) while they meet.
"""

from __future__ import annotations

import math
import random

import full_cases as fc

FAMILY_SIZES = {"vec": 260, "vec_do": 360, "turned_modes": 240, "setters": 200}
VEC_OPS = (fc.GOTO_OP, fc.DIR_OP, fc.STOP_OP, fc.SF_OP, fc.MASS_OP, fc.AGILITY_OP, fc.MAXVEL_OP, fc.RADIUS_OP,
           fc.VEL_OP, fc.MAX_ANGVEL_OP, fc.ROT_AGILITY_OP, fc.ANGVEL_OP, fc.ROTATION_OP)
REFUSED = (-1.0, 0.0, math.nan, math.inf, -math.inf)


def _ship(r: random.Random, bid: int, **kw) -> tuple:
    """A free, non-massive ship (it meets nothing) with a random hull, as an AddBall record; keywords override."""
    b = {"pos": [r.uniform(-3e4, 3e4) for _ in range(3)], "mass": 10 ** r.uniform(5.5, 9.5),
         "radius": r.uniform(5.0, 400.0), "max_vel": r.choice([r.uniform(5.0, 120.0), r.uniform(120.0, 900.0)]),
         "agility": r.choice([r.uniform(0.05, 0.5), r.uniform(0.5, 3.0)]),
         "sf": r.choice([1.0, r.uniform(0.05, 1.0), 0.0]), "max_ang_vel": r.choice([1.0, r.uniform(0.02, 4.0), 0.0]),
         "rot_agility": r.choice([0.9, r.uniform(0.02, 3.0)]),
         "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-300, 300) for _ in range(3)]]), "free": True, "massive": False,
         "interactive": False}
    b.update(kw)
    return fc._ball(r, bid, **b)


def _point(r: random.Random, pos: list, vel: list) -> list:
    """A goto point that makes a ship turn: behind it, beside it, ahead, far or near."""
    speed = math.sqrt(sum(c * c for c in vel)) or 1.0
    ahead = [c / speed for c in vel]
    side = fc._unit(r)
    how = r.choice(["behind", "side", "ahead", "far", "near", "random"])
    d = r.choice([r.uniform(50.0, 2e3), r.uniform(2e3, 5e4)])
    if how == "behind":
        u = [-a + 0.1 * s for a, s in zip(ahead, side)]
    elif how == "side":
        u = side
    elif how == "ahead":
        u = [a + 0.05 * s for a, s in zip(ahead, side)]
    elif how == "far":
        u, d = side, r.uniform(1e5, 1e7)
    elif how == "near":
        u, d = side, r.uniform(0.0, 30.0)
    else:
        return [r.uniform(-1e5, 1e5) for _ in range(3)]
    return [p + c * d for p, c in zip(pos, u)]


def _value(r: random.Random, op: int) -> tuple:
    """A setter's value for op, refused once in a while: the record's d."""
    refuse = r.random() < 0.12
    if op == fc.SF_OP:
        v = r.choice(REFUSED + (1.5, -0.2)) if refuse else r.choice([r.uniform(0.0, 1.0), 1.0, 0.0])
    elif op == fc.MASS_OP:
        v = r.choice(REFUSED) if refuse else 10 ** r.uniform(5.0, 9.8)
    elif op == fc.AGILITY_OP:
        v = r.choice(REFUSED) if refuse else r.uniform(0.03, 3.0)
    elif op == fc.MAXVEL_OP:
        v = r.choice((-5.0, -0.0, math.inf)) if refuse else r.choice([r.uniform(1.0, 1500.0), 0.0])
    elif op == fc.RADIUS_OP:
        v = r.choice((-5.0, -0.0)) if refuse else r.uniform(1.0, 800.0)
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


def vec_cases(r: random.Random, out: list, n: int, do: bool) -> None:
    """Batch-compatible worlds: free, non-massive ships at step 0, GOTO/GotoDirection/STOP and every setter later."""
    for _ in range(n):
        ids = fc._ids(r, r.randint(1, 6))
        recs, state = [], {}
        for bid in ids:
            rec = _ship(r, bid)
            recs.append(rec)
            state[bid] = (rec[2]["pos"], rec[2]["vel"])
        steps = r.randint(15, 60)
        for bid in ids:
            pos, vel = state[bid]
            recs.append(fc._k(r.randint(0, 2), fc.GOTO_OP, bid, d=_point(r, pos, vel)))
            k = 0
            while True:
                k += r.choice([1, 1, 2, 3, 5, 8])
                if k > steps:
                    break
                op = r.choice((fc.GOTO_OP,) * 4 + (fc.DIR_OP, fc.STOP_OP) + VEC_OPS[3:])
                if op == fc.GOTO_OP:
                    recs.append(fc._k(k, op, bid, d=_point(r, pos, vel)))
                elif op == fc.DIR_OP:
                    d = r.choice([fc._unit(r), [-c for c in vel] if any(vel) else fc._unit(r), [0.0, 0.0, 0.0],
                                  [math.nan, 0.0, 1.0]])
                    recs.append(fc._k(k, op, bid, d=d))
                elif op == fc.STOP_OP:
                    recs.append(fc._k(k, op, bid))
                else:
                    recs.append(fc._k(k, op, bid, d=_value(r, op)))
        fc.case(out, r, "vec_do" if do else "vec", recs, steps, master=False, tick=fc._tick(r), do=do,
                friction=fc._friction(r))


def turned_mode_cases(r: random.Random, out: list, n: int) -> None:
    """Turned ships in every flight mode: follow and orbit behind a turning leader, warp aligning while turned, a
    missile from a turned launcher at rest, formation followers of a turned leader, GetRotatedVector of turned balls."""
    for _ in range(n):
        do = r.random() < 0.45
        ids = fc._ids(r, 7)
        leader, follower, orbiter, warper, launcher, missile, wingman = ids
        recs = [_ship(r, leader, pos=[0.0, 0.0, 0.0], vel=[r.uniform(-80, 80) for _ in range(3)],
                      max_vel=r.uniform(50.0, 400.0), interactive=True),
                _ship(r, follower, pos=[r.uniform(-3e3, 3e3) for _ in range(3)], interactive=True),
                _ship(r, orbiter, pos=[r.uniform(-3e3, 3e3) for _ in range(3)], interactive=r.random() < 0.5),
                _ship(r, warper, pos=[r.uniform(-1e4, 1e4) for _ in range(3)], max_vel=r.uniform(100.0, 900.0)),
                _ship(r, launcher, pos=[r.uniform(-5e3, 5e3) for _ in range(3)], interactive=True),
                fc._missile_ball(r, missile, [0.0, 0.0, 0.0]),
                _ship(r, wingman, pos=[r.uniform(-2e3, 2e3) for _ in range(3)], interactive=True)]
        forms = [[[r.uniform(-400, 400) for _ in range(3)] for _ in range(r.randint(1, 4))]]
        data = [1.0, float(len(forms[0]))] + [c for v in forms[0] for c in v]
        recs += [fc._k(0, fc.FORMATIONS_OP, d=data), fc._k(0, fc.SET_FORMATION_OP, leader, w=0)]
        steps = r.randint(12, 40)
        for k in range(1, steps, r.choice([3, 4, 6])):      # the leader zigzags: it keeps turning
            recs.append(fc._k(k, fc.GOTO_OP, leader, d=[r.uniform(-2e4, 2e4) for _ in range(3)]))
        recs.append(fc._k(r.randint(0, 3), fc.GOTO_OP, launcher, d=[r.uniform(-3e4, 3e4) for _ in range(3)]))
        recs.append(fc._k(r.randint(1, 4), fc.FOLLOW_OP, follower, leader, d=fc._xyz(rr=r.uniform(0, 1500))))
        recs.append(fc._k(r.randint(1, 4), fc.ORBIT_OP, orbiter, leader, d=fc._xyz(rr=r.uniform(100, 3000))))
        v = fc._unit(r)
        dd = r.choice([3e5, 10 ** r.uniform(6.5, 8.5)])
        recs.append(fc._k(r.randint(0, 2), fc.GOTO_OP, warper, d=[r.uniform(-2e4, 2e4) for _ in range(3)]))
        recs.append(fc._k(r.randint(3, 6), fc.WARP_OP, warper, w=r.randint(2000, 30000),
                          d=(v[0] * dd, v[1] * dd, v[2] * dd, r.choice([0.0, 20000.0]))))
        k = r.randint(4, max(5, steps - 3))
        recs.append(fc._k(k, fc.VEL_OP, launcher, d=(0.0, 0.0, 0.0)))        # at rest: its turned (0, 0, -1)
        recs.append(fc._k(k, fc.LAUNCH_OP, missile, r.choice([0, follower]), launcher, w=r.randrange(4)))
        recs.append(fc._k(r.randint(1, 3), fc.FORMATION_FOLLOW_OP, wingman, leader, w=r.choice([0, 0, 1, 5])))
        for k in range(1, steps + 1, 2):
            recs.append(fc._k(k, fc.ROTATED_Q, r.choice([leader, launcher, warper, orbiter]), d=fc._unit(r)))
        fc.case(out, r, "turned_modes", recs, steps, master=r.random() < 0.6, tick=fc._tick(r), do=do,
                new_orbit=r.random() < 0.5, friction=fc._friction(r))


def setter_world_cases(r: random.Random, out: list, n: int) -> None:
    """Massive ships converging, the setters in flight while they meet (re-filing, decay factors), both collision
    methods, both views, with and without dynamical orientation."""
    for _ in range(n):
        ids = fc._ids(r, r.randint(2, 6))
        recs = []
        for bid in ids:
            recs.append(_ship(r, bid, pos=[r.uniform(-800, 800) for _ in range(3)], massive=True,
                              radius=r.uniform(10.0, 300.0), interactive=r.random() < 0.6,
                              vel=[r.uniform(-50, 50) for _ in range(3)]))
            recs.append(fc._k(0, fc.GOTO_OP, bid, d=[r.uniform(-200, 200) for _ in range(3)]))
        steps = r.randint(10, 35)
        for _ in range(r.randint(4, 14)):
            k = r.randint(1, steps)
            bid = r.choice(ids)
            op = r.choice(VEC_OPS[3:] + (fc.GOTO_OP, fc.STOP_OP))
            if op == fc.GOTO_OP:
                recs.append(fc._k(k, op, bid, d=[r.uniform(-500, 500) for _ in range(3)]))
            elif op == fc.STOP_OP:
                recs.append(fc._k(k, op, bid))
            else:
                recs.append(fc._k(k, op, bid, d=_value(r, op)))
        fc.case(out, r, "setters", recs, steps, master=r.random() < 0.5, tick=fc._tick(r), do=r.random() < 0.4,
                iterative=r.random() < 0.3, friction=fc._friction(r))


def cases(seed: int, sizes: dict | None = None) -> list:
    sizes = dict(FAMILY_SIZES, **(sizes or {}))
    r = random.Random(7000 + seed)
    out: list = []
    vec_cases(r, out, sizes["vec"], do=False)
    vec_cases(r, out, sizes["vec_do"], do=True)
    turned_mode_cases(r, out, sizes["turned_modes"])
    setter_world_cases(r, out, sizes["setters"])
    return out

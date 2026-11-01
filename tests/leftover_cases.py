# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the leftovers on the GPU (31 October 2026), generated from a seed, in
tests/full_cases.py's records: SetBallFree, SetBallRigid, the Python SetBoid, SetBallTroll with PetrifyTroll,
SetBallInteractive, and GetCenterDist and GetSurfaceDist, among the flight modes (FollowBall, both orbits, WarpTo,
EntityWarpIn, GotoPoint, GotoDirection, Stop) and the setters. Run by the reference (exact_ballpark.full,
FullBallpark), by Destiny's own code compiled verbatim (conformance/build_full.py, full_cases.HARNESS), and
by the batched park (exact_ballpark.vec_modes.VecModesPark: its commands on tensors, its CUDA kernel a tick;
tests/leftover_batch.py), all compared bit for bit.

As tests/flight_cases.py: a client's view (every bubble live, the batched park has no bubbles) of free, non-massive
ships (they never collide: the batched park has no collisions), some fixed at the start, with and without dynamical
orientation, under both orbit laws. Families:

- "boid": followers and orbiters (ids below and above their target's) set to BOID without a Stop, so they stay in the
  target's follower set; then re-commanded or not (GotoPoint, Stop, SetBoid off, a new follow, WarpTo, SetBallRigid,
  SetBallTroll), while the target starts warp proper (aligned and fast: at once) or warps in (EntityWarpIn): the members
  it then releases stop, whatever their mode.
- "rigid": SetBallRigid on ships in every mode (following, followed, aligning, in warp proper, at rest), then commands.
- "free": SetBallFree off and on (fixing stops a ship, zeroes its velocity and its thrust; freeing makes it evolve)
  on ships in every mode, ships fixed from the start freed, followers of fixed targets, commands to fixed ships.
- "troll": SetBallTroll (delays below 1 become 1, due at once or later) on free and fixed ships in every mode, trolls as
  targets, trolls re-commanded before they are due (they then never petrify), re-trolled, freed again once rigid, and
  trolls that were members of a set (SetBoid first) released by their target's warp before they are due.
- "mixed": three to six ships under every command above, the flight commands and the setters at random steps.

Every scenario asks GetCenterDist and GetSurfaceDist (DIST_Q) for every ordered pair of its balls, and for a missing
ball, at a few steps.
"""

from __future__ import annotations

import random

import flight_cases as flc
import full_cases as fc

FAMILY_SIZES = {"boid": 70, "rigid": 50, "free": 60, "troll": 80, "mixed": 60}
MISSING = 987654321987                                  # an id no scenario adds (_ids draws below 2^40)
TROLL_DELAYS = (-3, -1, 0, 1, 1, 2, 3, 5, 9, 25)


def _far(r: random.Random, pos: list) -> list:
    u = fc._unit(r)
    d = r.uniform(1e3, 4e4)
    return [pos[i] + d * u[i] for i in range(3)]


def _queries(r: random.Random, recs: list, ids: list, steps: int) -> None:
    """GetCenterDist/GetSurfaceDist (DIST_Q) for every ordered pair and a missing ball, at one to three steps."""
    for k in sorted(set(r.randint(0, steps) for _ in range(r.randint(1, 3)))):
        for a in ids:
            for b in ids:
                recs.append(fc._k(k, fc.DIST_Q, a, b))
        recs.append(fc._k(k, fc.DIST_Q, r.choice(ids), MISSING))
        recs.append(fc._k(k, fc.DIST_Q, MISSING, r.choice(ids)))


def _warp_now(r: random.Random, step: int, rec: tuple, recs: list) -> None:
    """A WarpTo at step far along a direction u the ship flies from the start (its velocity set along u above 75 % of
    its maximum speed, and a GotoDirection along u at step 0 when the WarpTo comes later), so that warp proper starts
    when it is given, unless other commands turn the ship first."""
    b = rec[2]
    dist = 10 ** r.uniform(8.0, 11.0)
    k, u = flc._warp_record(r, step, b["id"], b["pos"], dist)
    b["max_vel"] = max(b["max_vel"], 50.0)
    b["vel"] = [u[i] * b["max_vel"] * r.uniform(0.8, 0.99) for i in range(3)]
    if step > 0:
        recs.append(fc._k(0, fc.DIR_OP, b["id"], d=u))
    recs.append(k)


def _command(r: random.Random, step: int, bid: int, ids: list, recs: list, balls: dict) -> None:
    """One random command of every kind to ball bid at step."""
    what = r.choice(["goto", "dir", "stop", "follow", "orbit", "warp", "warpin", "rigid", "boid", "boid_off", "free",
                     "fix", "troll", "interactive", "setter"])
    b = balls[bid]
    others = [x for x in ids if x != bid]
    if what == "goto":
        recs.append(fc._k(step, fc.GOTO_OP, bid, d=_far(r, b["pos"])))
    elif what == "dir":
        recs.append(fc._k(step, fc.DIR_OP, bid, d=fc._unit(r)))
    elif what == "stop":
        recs.append(fc._k(step, fc.STOP_OP, bid))
    elif what in ("follow", "orbit") and others:
        t = r.choice(others)
        recs.append(fc._k(step, fc.FOLLOW_OP if what == "follow" else fc.ORBIT_OP, bid, t,
                          d=fc._xyz(rr=flc._orbit_range(r, b, balls[t]))))
    elif what == "warp":
        recs.append(flc._warp_record(r, step, bid, b["pos"], 10 ** r.uniform(5.5, 11.0))[0])
    elif what == "warpin":
        u = fc._unit(r)
        d = 10 ** r.uniform(6.0, 10.0)
        recs.append(fc._k(step, fc.WARPIN_OP, bid, w=r.choice([3000, 10000, r.randint(1, 20000)]),
                          d=(u[0] * d, u[1] * d, u[2] * d)))
    elif what == "rigid":
        recs.append(fc._k(step, fc.RIGID_OP, bid))
    elif what == "boid":
        recs.append(fc._k(step, fc.BOID_OP, bid, w=r.choice([1, 1, 7, -1])))
    elif what == "boid_off":
        recs.append(fc._k(step, fc.BOID_OP, bid, w=0))
    elif what == "free":
        recs.append(fc._k(step, fc.FREE_OP, bid, w=1))
    elif what == "fix":
        recs.append(fc._k(step, fc.FREE_OP, bid, w=0))
    elif what == "troll":
        recs.append(fc._k(step, fc.TROLL_OP, bid, w=r.choice(TROLL_DELAYS)))
    elif what == "interactive":
        recs.append(fc._k(step, fc.INTERACTIVE_OP, bid, w=r.choice([0, 1])))
    else:
        op = r.choice(flc.SETTER_OPS)
        recs.append(fc._k(step, op, bid, d=flc._value(r, op)))


def _followers(r: random.Random, recs: list, ids: list, tid: int, balls: dict, k: int = None) -> None:
    """Every ball but tid follows or orbits it from step 0 or 1, placed around it."""
    tb = balls[tid]
    for bid in ids:
        if bid == tid:
            continue
        b = balls[bid]
        rng = flc._orbit_range(r, b, tb)
        b["pos"] = flc._near(r, tb["pos"], fc.f32(rng) + fc.f32(b["radius"]) + fc.f32(tb["radius"]))
        recs.append(fc._k(r.choice([0, 1]) if k is None else k, r.choice([fc.FOLLOW_OP, fc.ORBIT_OP, fc.ORBIT_OP]),
                          bid, tid, d=fc._xyz(rr=rng)))


def _world(r: random.Random, n: int, fixed: float = 0.0) -> tuple:
    ids = fc._ids(r, n)
    recs = [flc._ship(r, bid, interactive=r.random() < 0.6, free=r.random() >= fixed) for bid in ids]
    balls = {rec[2]["id"]: rec[2] for rec in recs}
    return ids, recs, balls


def boid_cases(r: random.Random, out: list, n: int) -> None:
    for _ in range(n):
        do, new = r.random() < 0.5, r.random() < 0.5
        ids, recs, balls = _world(r, r.randint(3, 5))
        tid = ids[0]
        steps = r.randint(8, 36)
        _followers(r, recs, ids, tid, balls)
        kb = r.randint(1, max(1, steps // 2))
        for bid in ids[1:]:
            if r.random() < 0.75:
                recs.append(fc._k(kb + r.choice([0, 0, 1]), fc.BOID_OP, bid, w=1))
                if r.random() < 0.6:
                    k2 = kb + r.randint(1, 4)
                    what = r.choice(["goto", "stop", "boid_off", "refollow", "other", "warp", "rigid", "troll", "dir"])
                    b = balls[bid]
                    if what == "goto":
                        recs.append(fc._k(k2, fc.GOTO_OP, bid, d=_far(r, b["pos"])))
                    elif what == "dir":
                        recs.append(fc._k(k2, fc.DIR_OP, bid, d=fc._unit(r)))
                    elif what == "stop":
                        recs.append(fc._k(k2, fc.STOP_OP, bid))
                    elif what == "boid_off":
                        recs.append(fc._k(k2, fc.BOID_OP, bid, w=0))
                    elif what == "refollow":
                        recs.append(fc._k(k2, fc.ORBIT_OP, bid, tid, d=fc._xyz(rr=flc._orbit_range(r, b, balls[tid]))))
                    elif what == "other":
                        o = r.choice([x for x in ids if x not in (bid, tid)])
                        recs.append(fc._k(k2, fc.FOLLOW_OP, bid, o, d=fc._xyz(rr=flc._orbit_range(r, b, balls[o]))))
                    elif what == "warp":
                        recs.append(flc._warp_record(r, k2, bid, b["pos"], 10 ** r.uniform(5.5, 10.0))[0])
                    elif what == "rigid":
                        recs.append(fc._k(k2, fc.RIGID_OP, bid))
                    else:
                        recs.append(fc._k(k2, fc.TROLL_OP, bid, w=r.choice(TROLL_DELAYS)))
        kw = r.randint(kb, steps)
        if r.random() < 0.7:                                                # the target starts warp proper at once
            _warp_now(r, kw, recs[0], recs)                                 # recs[0] is tid's AddBall
        else:                                                               # or warps in (the release on the host)
            u = fc._unit(r)
            d = 10 ** r.uniform(6.0, 10.0)
            recs.append(fc._k(kw, fc.WARPIN_OP, tid, w=r.choice([3000, 10000]), d=(u[0] * d, u[1] * d, u[2] * d)))
        if r.random() < 0.3:
            recs.append(fc._k(r.randint(0, steps), fc.BOID_OP, r.choice(ids), w=1))
        flc._setters(r, recs, ids, steps, r.randint(0, 3))
        _queries(r, recs, ids, steps)
        flc._case(out, r, "boid", recs, steps, do, new)


def rigid_cases(r: random.Random, out: list, n: int) -> None:
    for _ in range(n):
        do, new = r.random() < 0.5, r.random() < 0.5
        ids, recs, balls = _world(r, r.randint(2, 5), fixed=0.15)
        steps = r.randint(6, 30)
        if len(ids) > 2 and r.random() < 0.7:
            _followers(r, recs, ids[1:], ids[1], balls)
        for bid in ids:
            if r.random() < 0.4:
                _command(r, r.randint(0, steps), bid, ids, recs, balls)
            if r.random() < 0.7:
                recs.append(fc._k(r.randint(0, steps), fc.RIGID_OP, bid))
        if r.random() < 0.5:
            _warp_now(r, r.randint(0, steps), r.choice([x for x in recs if x[0] == "B"]), recs)
        flc._setters(r, recs, ids, steps, r.randint(0, 3))
        _queries(r, recs, ids, steps)
        flc._case(out, r, "rigid", recs, steps, do, new)


def free_cases(r: random.Random, out: list, n: int) -> None:
    for _ in range(n):
        do, new = r.random() < 0.5, r.random() < 0.5
        ids, recs, balls = _world(r, r.randint(2, 5), fixed=0.35)
        steps = r.randint(6, 30)
        if len(ids) > 1 and r.random() < 0.7:
            _followers(r, recs, ids, ids[0], balls)
        for bid in ids:
            k = r.randint(0, steps)
            before = r.choice(["none", "goto", "warp", "warp", "boid", "stop"])     # the mode it is fixed in
            if before == "goto":
                recs.append(fc._k(max(0, k - r.randint(0, 3)), fc.GOTO_OP, bid, d=_far(r, balls[bid]["pos"])))
            elif before == "warp" and bid != ids[0]:
                _warp_now(r, max(0, k - r.randint(0, 3)), next(x for x in recs if x[0] == "B" and x[2]["id"] == bid),
                          recs)
            elif before == "boid":
                recs.append(fc._k(max(0, k - 1), fc.BOID_OP, bid, w=1))
            elif before == "stop":
                recs.append(fc._k(max(0, k - 1), fc.STOP_OP, bid))
            recs.append(fc._k(k, fc.FREE_OP, bid, w=0))
            for _ in range(r.randint(0, 3)):
                recs.append(fc._k(r.randint(k, steps), fc.FREE_OP, bid, w=r.choice([0, 1, 1, 5])))
            if r.random() < 0.4:
                _command(r, r.randint(0, steps), bid, ids, recs, balls)
        flc._setters(r, recs, ids, steps, r.randint(0, 3))
        _queries(r, recs, ids, steps)
        flc._case(out, r, "free", recs, steps, do, new)


def troll_cases(r: random.Random, out: list, n: int) -> None:
    for _ in range(n):
        do, new = r.random() < 0.5, r.random() < 0.5
        ids, recs, balls = _world(r, r.randint(2, 5), fixed=0.25)
        steps = r.randint(6, 34)
        tid = ids[0]
        if len(ids) > 1 and r.random() < 0.6:
            _followers(r, recs, ids, tid, balls)
        for bid in ids:
            if r.random() < 0.8:
                k = r.randint(0, steps)
                if bid != tid and r.random() < 0.25:                        # a member set to BOID first
                    recs.append(fc._k(max(0, k - 1), fc.BOID_OP, bid, w=1))
                recs.append(fc._k(k, fc.TROLL_OP, bid, w=r.choice(TROLL_DELAYS)))
                if r.random() < 0.35:
                    _command(r, k + r.randint(0, 6), bid, ids, recs, balls)
                if r.random() < 0.2:
                    recs.append(fc._k(k + r.randint(1, 12), fc.TROLL_OP, bid, w=r.choice(TROLL_DELAYS)))
                if r.random() < 0.25:
                    recs.append(fc._k(k + r.randint(2, 20), fc.FREE_OP, bid, w=1))
        if r.random() < 0.4:
            _warp_now(r, r.randint(0, steps), recs[0], recs)               # recs[0] is tid's AddBall
        flc._setters(r, recs, ids, steps, r.randint(0, 3))
        _queries(r, recs, ids, steps)
        flc._case(out, r, "troll", recs, steps, do, new)


def mixed_cases(r: random.Random, out: list, n: int) -> None:
    for _ in range(n):
        do, new = r.random() < 0.5, r.random() < 0.5
        ids, recs, balls = _world(r, r.randint(3, 6), fixed=0.15)
        steps = r.randint(10, 40)
        if r.random() < 0.6:
            _followers(r, recs, ids, ids[0], balls)
        for _ in range(r.randint(4, 14)):
            _command(r, r.randint(0, steps), r.choice(ids), ids, recs, balls)
        if r.random() < 0.4:
            _warp_now(r, r.randint(0, steps), r.choice([x for x in recs if x[0] == "B"]), recs)
        _queries(r, recs, ids, steps)
        flc._case(out, r, "mixed", recs, steps, do, new)


def cases(seed: int, sizes: dict | None = None) -> list:
    """The scenarios of one seed (development: 1; held out after the freeze: 2 and on), in family order."""
    r = random.Random(f"modes2-leftovers-{seed}")
    sizes = dict(FAMILY_SIZES, **(sizes or {}))
    out: list = []
    boid_cases(r, out, sizes["boid"])
    rigid_cases(r, out, sizes["rigid"])
    free_cases(r, out, sizes["free"])
    troll_cases(r, out, sizes["troll"])
    mixed_cases(r, out, sizes["mixed"])
    return out

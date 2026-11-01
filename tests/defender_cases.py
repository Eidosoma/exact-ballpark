# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for defender missiles (a missile whose owner id is negative, the GPU parity report's G9)
in the composed park (exact_ballpark.vec_composed.VecComposedPark with collisions), generated from a seed:
tests/park_cases.py's worlds (a cluster of ships, most of them massive, fixed spheres, warps, setters, cloaks,
formations, normal missiles from the pool 101 to 104) with one more normal missile (105) and up to four defender
missiles (201 to 204) added, each with an empty row of its world.

A defender is added near the cluster and then launched by a cluster ship with the owner -launcher (PyLaunchMissile
finds the launcher by |owner|, Thunkers.cpp:857): aimed or not, massive or not, at a normal missile in flight, at
another defender, at a ship, at no target (it flies on by GotoDirection); or made to follow by MissileFollow with a
negative owner; or launched with a launcher missing from the park (refused). Launched at its launcher's position, a
massive defender starts inside a massive launcher's sphere: Gradient's owner skip compares the signed id
(Ballpark.cpp:2745), so the two meet. Some defenders time out (RemoveBall), and missile 105 is at times launched at a
defender (a normal missile's one candidate is then a defender). A defender sees every existing, massive, uncloaked ball
of a related partition box, missiles included (Partition::GetCollisionCandidates, Partition.cpp:301).

The reference, the batch and the comparison are tests/park_cases.py's (every field, mLastC, the collision lists, the
filings, the DoCollision events, the answers); tests/test_movement_vec_defenders.py also runs these worlds on the conformance kit's full harness (Destiny's own code compiled verbatim). Development seed 21; held-out seed 22, first run after this file
was frozen.
"""

from __future__ import annotations

import random

import destiny_combat_cases as dc
import park_cases as pc

POOL = pc.POOL
EXTRA_MISSILE = 105                                      # a normal missile the defenders can intercept
DEFENDERS = (201, 202, 203, 204)
FORMATIONS = pc.FORMATIONS


def world(r: random.Random, steps: int, master: bool, tick: int = 1000) -> dict:
    """One world: park_cases' world, then the extra normal missile and the defenders, with their records."""
    w = pc.world(r, steps, master, tick)
    ships = w["ships"]
    free_ids = [s["id"] for s in ships if s["free"]]
    recs = list(w["records"])
    one = next(s for s in ships if s["id"] == 1)                 # ship 1: never park_cases' far warper
    near = lambda: pc._point(r, one["pos"], 300.0)
    added = {rec[2]["id"]: rec[0] for rec in recs if rec[1] == "add"}   # a pool missile's first step
    mine: list = []
    if r.random() < 0.6:                                         # a normal missile at a cluster ship, massive
        k = r.randint(1, max(1, steps // 2))
        mine.append((k, "add", dc._missile(r, EXTRA_MISSILE, near())))
        owner = r.choice(free_ids)
        target = r.choice([t for t in free_ids if t != owner] or free_ids)
        mine.append((k, "launch", EXTRA_MISSILE, target, owner, r.choice([0, 1, 1]), r.choice([1, 1, 0])))
        added[EXTRA_MISSILE] = k
    defenders = list(DEFENDERS[:r.choice([1, 2, 2, 3, 4])])
    for d in defenders:
        k = r.randint(1, max(1, steps - 3))
        mine.append((k, "add", dc._missile(r, d, near())))
        launcher = r.choice(free_ids)
        inflight = [m for m, km in added.items() if km <= k and m != d]
        ship_targets = [t for t in free_ids if t != launcher] or free_ids
        target = r.choice(inflight) if inflight and r.random() < 0.6 else r.choice(ship_targets)
        massive = r.choice([1, 1, 1, 0])
        how = r.choice(["launch", "launch", "launch", "launch", "follow", "no_target", "missing"])
        if how == "follow":
            mine.append((k, "vel", d, *[c * r.uniform(50.0, 300.0) for c in dc._unit(r)]))
            mine.append((k, "massive", d, massive))
            mine.append((k, "mfollow", d, target, -launcher))
        elif how == "no_target":
            mine.append((k, "launch", d, 999, -launcher, r.choice([0, 1]), massive))
        elif how == "missing":
            mine.append((k, "launch", d, target, -99, r.choice([0, 1]), massive))
        else:
            mine.append((k, "launch", d, target, -launcher, r.choice([0, 1, 1, 2]), massive))
        added[d] = k
        if r.random() < 0.2:
            mine.append((r.randint(k + 1, steps), "remove", d))
    if EXTRA_MISSILE not in added and r.random() < 0.5 and defenders:   # a normal missile at a defender
        k = max(added[defenders[0]], r.randint(1, max(1, steps - 2)))
        mine.append((k, "add", dc._missile(r, EXTRA_MISSILE, near())))
        owner = r.choice(free_ids)
        mine.append((k, "launch", EXTRA_MISSILE, defenders[0], owner, r.choice([0, 1]), 1))
    recs.extend(mine)
    recs.sort(key=lambda rec: rec[0])
    return {"ships": ships, "pool": list(POOL) + [EXTRA_MISSILE] + list(DEFENDERS), "records": recs,
            "start": w["start"]}


def batches(seed: int, n_batches: int = 16, worlds: int = 16) -> list:
    """The scenarios of a seed, as park_cases' batches (even batches without dynamical orientation, odd ones with it;
    the server's view or a client's; the orbit law; the tick and friction), each world this module's."""
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


# the reference, the batch and the comparison: park_cases'
reference = pc.reference
batched = pc.batched
differences = pc.differences
in_domain = pc.in_domain

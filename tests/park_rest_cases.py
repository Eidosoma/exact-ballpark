# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the composed park in which a ball dropping out of warp comes to rest where its last filing was not
(the note E20_DESTINY_COMPOSED_PARK_PARK1.md, section 7, item 4): long ticks and short
warp-ins. EntityWarpIn begins its warp five ticks back (TRV-715; modes.py's entity_warp_in), so with ticks of 2 to 5 s
a warp of one to a few hundred kilometres (Destiny warps no shorter than 100 km: WarpTo's MIN_WARP_DISTANCE_SQ) is
past its deceleration at its first tick and drops out at once, at rest on its goto point (WarpDistance's last
branch), re-filed there (modes.py:711, the drop-out's re-filing at rest), far from the filing of its last commit; the
longer ones drop out a tick or two later, after a long last step. In the generator (tests/park_cases.py)
the rest point and the warper's last commit are filed alike, so the re-filing at rest changes no generated world there.

Each world is a park_cases world (the cluster: ships flying through, following and orbiting each other, missiles,
formations, cloaks, setters, the far warper) with a long tick, and one to three of its ships (never ship 1, nor the
far warper) moved 100 to 450 km out, each warping in (EntityWarpIn) at a step onto or beside a cluster ship. The
reference, the batch and the comparison are park_cases' (`reference`, `batched`, `differences`). The development seed
is 11; seed 3 is held out (first run once this file is frozen).
"""

from __future__ import annotations

import random

import destiny_combat_cases as dc
import park_cases as pc

TICKS = (2000, 2000, 3000, 4000, 5000)                      # ms: five ticks back is 10 to 25 s of warp
FACTORS = (3000, 6000, 10000, 20000, 20000)                 # Destiny's warp factors (pc.world's set, the faster)
NEAR, FAR = 1.05e5, 4.5e5                                   # the warp-ins' lengths: past WarpTo's 100 km minimum


def _centre(ships: list) -> list:
    """The cluster's centre: the mean position of its free ships (the far warper left out by the caller)."""
    xs = [s["pos"] for s in ships]
    return [sum(p[i] for p in xs) / len(xs) for i in range(3)]


def world(r: random.Random, steps: int, master: bool, tick: int) -> dict:
    """A park_cases world with one to three short warp-ins: each chosen ship starts 100 to 450 km out and warps in
    (EntityWarpIn) at a step onto or beside a cluster ship (or a point in the cluster)."""
    w = pc.world(r, steps, master, tick)
    ships = w["ships"]
    taken = {rec[2] for rec in w["records"] if rec[1] in ("warp", "warpin")}
    cluster = [s for s in ships if s["free"] and s["id"] not in taken]
    if len(cluster) < 2:
        return w
    centre = _centre(cluster)
    cands = [s for s in cluster if s["id"] != 1]
    k = min(len(cands), r.choice([1, 1, 2, 3]))
    movers = r.sample(cands, k)
    stay = [s for s in ships if s not in movers and s["id"] not in taken]
    recs = list(w["records"])
    for s in movers:
        u = dc._unit(r)
        far = r.uniform(NEAR, FAR)
        s["pos"] = [centre[i] + u[i] * far for i in range(3)]
        if stay and r.random() < 0.5:                       # onto a ship of the cluster (a contact at rest)
            on = r.choice(stay)
            dest = pc._point(r, on["pos"], on["radius"])
        else:
            dest = pc._point(r, centre, r.choice([60.0, 200.0, 600.0]))
        step = r.choice([0, 0, 1, r.randint(1, max(1, steps // 2))])
        recs.append((step, "warpin", s["id"], *dest, r.choice(FACTORS)))
    recs.sort(key=lambda rec: rec[0])
    w["records"] = recs
    return w


def batches(seed: int, n_batches: int = 16, worlds: int = 16) -> list:
    """The scenarios of a seed: batches of worlds, each batch one configuration (even batches without dynamical
    orientation, odd ones with it; the server's view or a client's; the orbit law; a long tick; the friction)."""
    r = random.Random(seed)
    out = []
    for b in range(n_batches):
        do = b % 2 == 1
        master = r.random() < 0.5
        cfg = {"do": do, "do_off": do and r.random() < 0.4, "master": master, "tick": r.choice(TICKS),
               "friction": r.choice([1.0e6] * 5 + [10 ** r.uniform(5.0, 7.0)]),
               "new_orbit": r.random() < 0.5, "steps": r.randint(8, 16)}
        cfg["worlds"] = [world(r, cfg["steps"], master, cfg["tick"]) for _ in range(worlds)]
        out.append(cfg)
    return out


reference = pc.reference
batched = pc.batched
differences = pc.differences
in_domain = pc.in_domain

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Shared movement test cases (conformance rows 2 and 3): generated from a seed, one ball per case."""

from __future__ import annotations

import random

from exact_ballpark.ref import GOTO, STOP, Ballpark


def cases(seed: int) -> list[tuple]:
    r = random.Random(seed)
    out = []

    def one(kind: str):
        friction = 1.0e6 if kind != "taylor" else 10 ** r.uniform(-6, -2)
        tick = r.choice([250, 500, 1000, 1000, 2000])
        mass = 10 ** r.uniform(5, 9)
        agility, maxvel, sf = r.uniform(0.05, 2.0), r.uniform(1.0, 5000.0), r.uniform(0.05, 1.0)
        p = [r.uniform(-1e5, 1e5) for _ in range(3)]
        v = [r.uniform(-500, 500) for _ in range(3)]
        mode = STOP if kind == "stop" else GOTO
        if kind == "homing":                     # a goto point inside the one-tick distance
            d = sf * maxvel * tick * 0.001 * r.uniform(0.05, 0.95)
            u = [r.gauss(0, 1) for _ in range(3)]
            n = sum(c * c for c in u) ** 0.5
            g = [p[i] + d * u[i] / n for i in range(3)]
        elif kind == "zero":                     # the goto point is the position
            g = list(p)
        else:
            g = [r.uniform(-1e6, 1e6) for _ in range(3)]
        out.append((len(out) + 1, friction, tick, r.randint(1, 40), mode, mass, agility, maxvel, sf, *p, *v, *g))

    for kind, n in (("goto", 300), ("stop", 100), ("taylor", 60), ("homing", 40), ("zero", 5)):
        for _ in range(n):
            one(kind)
    return out


def reference(case: tuple) -> list[tuple]:
    cid, friction, tick, steps, mode, mass, agility, maxvel, sf = case[:9]
    p, v, g = list(case[9:12]), list(case[12:15]), list(case[15:18])
    park = Ballpark(friction=friction, tick_interval_ms=tick)
    b = park.add_ball(1, mass, 0.0, maxvel, p, v, agility, sf)
    b.mode, b.goto = mode, g
    out = []
    for s in range(1, steps + 1):
        park.evolve()
        out.append((cid, s, *b.pos, *b.vel))
    return out

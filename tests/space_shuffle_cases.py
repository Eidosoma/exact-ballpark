# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Shuffled proximity sensors on the master batched: worlds of the frozen generator
(tests/space_batch.py, unchanged) with shuffled sensors added, each world its own srand seed, and the comparison of the
batched layer (exact_ballpark.vec_space over VecOrientPark on the CPU, or VecModesPark on a CUDA device) with the scalar
reference (exact_ballpark.full.FullBallpark with is_master=True, one park a world), bit for bit, after every step.

Destiny draws a shuffled sensor's first period from the C library's rand() (AddProximitySensor, Ball.cpp:244-263;
floatrand, Ball.cpp:19): 0 + rand() / RAND_MAX x (period - 0). The reference transcribes glibc's generator
(exact_ballpark.space.GlibcRandom, srand(seed) per park) and is held to compiled Destiny with shuffled sensors by
tests/test_movement_space_oracle.py; the batched layer keeps each world's generator as state (`rnd`, seeded by
`VecSpace(seed=...)`) and draws on the device.

On top of the records each world gets: an srand seed (edge values among them: 0, which glibc takes as 1, 2**31 - 1,
2**31, 2**32 - 1, -1 and values past 32 bits, which srand's unsigned int truncates); about a third of the own
sensor records turned shuffled (a nonzero shuffle: 1, 2 or -3); and extra shuffled sensors (periods 2, 1, 3, 0.5,
7.25 and 1e-3 s; only-interactive or not), some re-added on a ball that already senses (a fresh draw each time), some
refused (a range below minus the radius: no draw), some on a removed ball (no draw). Every comparison of
tests/space_batch.py is made (each ball's sensor state includes its elapsed time's bits), and each world's generator
state (the table, fptr, rptr) is compared with the reference's after every step.
"""

from __future__ import annotations

import random
from collections import Counter

import torch

import space_batch as SB
from exact_ballpark import vec_space as V
from exact_ballpark.space import GlibcRandom

SEED_EDGES = (0, 1, 2, 7, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1, -1, 2 ** 32 + 5, 2 ** 40 + 12345)
PERIODS = (2.0, 1.0, 3.0, 0.5, 7.25, 1e-3)


def cases(seed: int, n: int, park: str = "orient", steps: int = 16) -> list:
    """n worlds: the generator's worlds for (seed, park), each given an srand seed and shuffled sensors by this
    module's own stream (`shuffle-{park}-{seed}`); the stream is not touched."""
    cs = SB.cases(seed, n, park, steps)
    r = random.Random(f"shuffle-{park}-{seed}")
    for w in cs:
        w["seed"] = r.choice(SEED_EDGES) if r.random() < 0.3 else r.randrange(0, 2 ** 32)
        alive = [b["id"] for b in w["balls"]]
        removed: list = []
        for t, recs in enumerate(w["steps"]):
            out = []
            for rec in recs:
                if rec[0] == "sensor" and r.random() < 0.35:
                    rec = tuple(rec) + (r.choice((1, 1, 2, -3)),)
                out.append(rec)
                if rec[0] == "remove":
                    alive.remove(rec[1])
                    removed.append(rec[1])
            for _ in range(r.choice((0, 0, 1, 1, 2))):
                if not alive:
                    break
                k = r.random()
                bid = r.choice(alive)
                if k < 0.12 and removed:
                    bid = r.choice(removed)                 # a removed ball: nothing happens, nothing drawn
                rng = r.choice((r.uniform(300.0, 3000.0), r.uniform(3000.0, 30000.0)))
                if 0.12 <= k < 0.2:
                    rng = -1.0e9                            # refused: float(radius) + float(range) < 0, no draw
                rec = ("sensor", bid, rng, r.choice(PERIODS), r.choice((0, 0, 1)), r.choice((1, 1, 1, 2, -3)))
                out.insert(r.randrange(len(out) + 1), rec)
            w["steps"][t] = out
    return cs


class ShuffleScalarWorld(SB.ScalarWorld):
    """the scalar world, its park's generator seeded with the world's seed (srand(seed): no draw happens before
    the first shuffled sensor), and the shuffled sensor records applied with their shuffle."""

    def __init__(self, w: dict, friction: float = 1.0e6, tick_ms: int = 1000):
        super().__init__(w, friction, tick_ms)
        self.park.rng = GlibcRandom(w["seed"])

    def apply(self, rec: tuple) -> None:
        if rec[0] == "sensor" and len(rec) > 5:
            self.park.add_proximity_sensor(rec[1], rec[2], rec[3], rec[5], bool(rec[4]))
        else:
            super().apply(rec)

    def rand_state(self) -> list:
        g = self.park.rng
        return list(g.s) + [g.f, g.r]


class ShuffleBatchedWorlds(SB.BatchedWorlds):
    """the batched worlds with each world's generator seeded (`rnd` set as VecSpace(seed=...) sets it, from
    vec_space.glibc_srand), and the shuffled sensor records applied as ADD_SENSOR with f[2] nonzero."""

    def __init__(self, cs: list, device: str, park: str = "orient", **kw):
        super().__init__(cs, device, park, **kw)
        self.space.st["rnd"] = V.glibc_srand([w["seed"] for w in cs]).to(self.dev).contiguous()

    def _apply(self, recs: list) -> None:
        shuffled = [rec is not None and rec[0] == "sensor" and len(rec) > 5 for rec in recs]
        super()._apply([None if s else rec for rec, s in zip(recs, shuffled)])
        ops = [[] for _ in recs]
        for wi, (rec, s) in enumerate(zip(recs, shuffled)):
            if s and self.alive(wi, rec[1]):
                ops[wi].append(("ADD_SENSOR", self.slot(wi, rec[1]), int(rec[4]), (rec[2], rec[3], float(rec[5]))))
        if any(ops):
            self.space.ops(ops)

    def rand_state(self, wi: int) -> list:
        return self.space.st["rnd"][wi].cpu().tolist()


def compare(cs: list, device: str, park: str = "orient", hits: bool = True, lib: dict | None = None) -> dict:
    """tests/space_batch.py's comparison on these worlds, plus each world's generator state after every step: the
    differences (step, world, what), the volumes compared, the flagged worlds, the branch counts on both sides."""
    T = len(cs[0]["steps"])
    scal = [ShuffleScalarWorld(w) for w in cs]
    bw = ShuffleBatchedWorlds(cs, device, park, hits=hits, lib=lib)
    diffs, compared = [], Counter()
    for sw in scal:
        SB.scalar_events(sw)
    for t in range(T):
        for wi, w in enumerate(cs):
            for rec in w["steps"][t]:
                scal[wi].apply(rec)
        for wi in range(len(cs)):
            scal[wi].evolve()
        bw.space.clear_events()
        bw.step(t)
        flags = bw.space.st["flags"].cpu().tolist()
        for wi, w in enumerate(cs):
            for rec in w["steps"][t]:
                if rec[0] == "sensor" and len(rec) > 5:
                    compared["shuffled_records"] += 1
            if flags[wi]:
                continue
            want, got = SB.scalar_state(scal[wi]), SB.batched_state(bw, wi)
            if set(want["balls"]) != set(got["balls"]):
                diffs.append((t, wi, "balls", sorted(set(want["balls"]) ^ set(got["balls"]))))
            for bid in want["balls"]:
                if bid not in got["balls"]:
                    continue
                for key, val in want["balls"][bid].items():
                    compared["ball_fields"] += 1
                    if got["balls"][bid][key] != val:
                        diffs.append((t, wi, bid, key, val, got["balls"][bid][key]))
            for key in ("dicts", "interactives", "keep_alives", "bubble_id", "tops", "proximity"):
                compared["world_fields"] += 1
                if want["world"][key] != got["world"][key]:
                    diffs.append((t, wi, key, want["world"][key], got["world"][key]))
            ev_w, ev_g = SB.scalar_events(scal[wi]), bw.space.events(wi)
            compared["events"] += len(ev_w)
            if ev_w != ev_g:
                diffs.append((t, wi, "events", ev_w, ev_g))
            aw, ag = scal[wi].answers, bw.answers[wi]
            compared["answers"] += len(aw)
            if aw != ag:
                diffs.append((t, wi, "answers", aw, ag))
            scal[wi].answers, bw.answers[wi] = [], []
            rw, rg = scal[wi].rand_state(), bw.rand_state(wi)
            compared["rand_states"] += 1
            if rw != rg:
                diffs.append((t, wi, "rand_state", rw, rg))
            compared["ball_steps"] += len(want["balls"])
    flagged = [wi for wi, f in enumerate(bw.space.st["flags"].cpu().tolist()) if f]
    ref_hits = Counter()
    for sw in scal:
        ref_hits.update(sw.park.trace)
    return {"diffs": diffs, "compared": dict(compared), "flagged": flagged, "hits": bw.space.hit_counts(),
            "ref_hits": dict(ref_hits), "steps": T}

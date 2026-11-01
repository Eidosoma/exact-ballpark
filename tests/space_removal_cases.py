# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Worlds for Destiny's delayed removal on the master batched (exact_ballpark.vec_space_removal),
generated from a seed, and the two runners compared bit for bit: the scalar reference (tests/space_batch.py's
ScalarWorld: exact_ballpark.full.FullBallpark with is_master=True, BringOutTheDeads before each tick as OnTick runs it)
and the batched layer (space_batch's BatchedWorlds: exact_ballpark.vec_space.VecSpace over VecOrientPark on the CPU or
VecModesPark on a CUDA device, with DelayedRemoval).

A world is space_batch's (2 to 6 balls around one to three anchors, or a row of three top boxes), half of them grown
to 9 or 10 balls so that more than 7 can wait in the 2-tick buffer at once. Each step: 0 to 3 of space_batch's
records among the balls not waiting (flights across faces, jumps, the re-filing setters, interactive changes,
keep-alive balls, InitializeBubbles, CopyBubbles, sensors, tracking, the queries; follows, orbits and warps on
VecModesPark), then the removals: RemoveBall with a delay of 1 to 5 ticks (or 0: at once), now and then a burst of
8 or more with one delay (the kill count of 7 binds), and commands given to waiting balls (a Stop, a GotoPoint, a
FollowBall or Orbit, a WarpTo, a second RemoveBall: the removal tick lives in mEffectStamp, so some move it), and
follows of a waiting ball (refused). Compared after every step: space_batch's whole state (every ball's bubble, filing,
sets, position and velocity bits, mode, sensor, tracking and notification state; the dictionaries, sets, top boxes,
proximity balls; the events, DoFinalCleanup's included, in order; the queries' answers) and, for this module, every
ball's moribund and massive flags and removal tick (mEffectStamp), and each world's moribundBalls.
"""

from __future__ import annotations

import random
from collections import Counter

import torch

import space_batch as SB
from exact_ballpark.vec_space_removal import DelayedRemoval

I64, BOOL = torch.int64, torch.bool
KINDS = ("remove_later", "remove")


def _grow(r: random.Random, w: dict) -> None:
    """Grow a world to 9 or 10 balls near its anchors (free, not massive, not occluders; large and fast, so filed
    coarsely: the layer's live-box bound stays below 256)."""
    n = r.randint(9, 10)
    used = {b["id"] for b in w["balls"]}
    while len(w["balls"]) < n:
        bid = r.choice([r.randrange(1, 40), r.randrange(100, 2000), r.randrange(10 ** 5, 10 ** 9)])
        if bid in used:
            continue
        used.add(bid)
        a = r.choice(w["anchors"])
        w["balls"].append({"id": bid, "mass": 10 ** r.uniform(5.5, 9.0), "radius": SB._f(r, 50.0, 3000.0),
                           "max_vel": SB._f(r, 300.0, 1500.0),
                           "agility": SB._f(r, 0.05, 2.0), "sf": r.choice([1.0, SB._f(r, 0.1, 1.0)]),
                           "pos": SB._near(r, a), "vel": r.choice([[0.0, 0.0, 0.0],
                                                                    [SB._f(r, -300.0, 300.0) for _ in range(3)]]),
                           "interactive": r.random() < 0.5, "massive": False, "free": True, "occluder": False})


def _waiting_record(r: random.Random, w: dict, alive: list, waiting: list, park: str) -> tuple:
    """A command given to a waiting ball, or a follow of one."""
    s = r.choice(waiting)
    k = r.random()
    if k < 0.25:
        return ("stop", s)
    if k < 0.45:
        return ("goto", s, *SB._near(r, r.choice(w["anchors"]), 6000.0))
    if k < 0.6:
        return ("remove_later", s, r.choice([1, 2, 4, 6]))
    if k < 0.65:
        return ("remove", s)
    if park == "modes" and alive and k < 0.8:
        o = r.choice(alive)
        return (r.choice(["follow", "orbit"]), s, o, SB._f(r, 100.0, 5000.0))
    if park == "modes" and alive and k < 0.9:
        return (r.choice(["follow", "orbit"]), r.choice(alive), s, SB._f(r, 100.0, 5000.0))
    if park == "modes":
        u = SB._unit(r)
        b = next(x for x in w["balls"] if x["id"] == s)
        return ("warp", s, *[b["pos"][i] + 1e9 * u[i] for i in range(3)], 0.0, r.randint(1500, 6000))
    return ("stop", s)


def cases(seed: int, n: int, park: str = "orient", steps: int = 16) -> list:
    """n worlds for a park ("orient": VecOrientPark, CPU or GPU; "modes": VecModesPark, a CUDA device)."""
    r = random.Random(f"space-removal-{park}-{seed}")
    out = []
    for wid in range(n):
        w = SB._world(r, wid, park)
        if r.random() < 0.5:
            _grow(r, w)
        alive = [b["id"] for b in w["balls"]]                # occluders too: a massive ball removed with a delay
        waiting: list = []
        state = {"warping": set()}
        for _ in range(steps):
            recs = []
            for _ in range(r.choice([0, 1, 1, 2, 3])):
                rec = SB._record(r, w, alive, park, state)
                if rec is None:
                    continue
                for one in (rec if isinstance(rec, list) else [rec]):
                    if one[0] == "remove":
                        alive.remove(one[1])
                    recs.append(one)
            k = r.random()
            if k < 0.12 and len(alive) >= 9:                # a burst: 8 or more with one delay
                d = r.choice([1, 2, 3])
                for s in r.sample(alive, r.randint(8, len(alive) - 1)):
                    recs.append(("remove_later", s, d))
                    alive.remove(s)
                    waiting.append(s)
            elif k < 0.45 and len(alive) > 1:
                s = r.choice(alive)
                d = r.choice([1, 1, 2, 3, 5, 0])
                recs.append(("remove_later", s, d))
                alive.remove(s)
                if d > 0:
                    waiting.append(s)
            if waiting and r.random() < 0.35:
                rec = _waiting_record(r, w, alive, waiting, park)
                recs.append(rec)
                if rec[0] == "remove":
                    waiting.remove(rec[1])
            w["steps"].append(recs)
        out.append(w)
    return out


class ScalarWorld(SB.ScalarWorld):
    """space_batch's ScalarWorld with RemoveBall's delay and BringOutTheDeads before each tick (OnTick's order); it
    counts the bring-outs whose buffer held more than 7 balls (the kill count binding) and the commands that moved a
    waiting ball's removal tick."""

    def __init__(self, w: dict, **kw):
        super().__init__(w, **kw)
        self.limit_binds = 0
        self.stamp_moved = 0

    def apply(self, rec: tuple) -> None:
        p = self.park
        waiting = [b for bid in sorted(p.moribund) if (b := p.balls.get(bid)) is not None]
        before = {b.id: b.effect_stamp for b in waiting}
        if rec[0] == "remove_later":
            p.remove_ball(rec[1], rec[2])
        else:
            super().apply(rec)
        if rec[0] not in KINDS:
            self.stamp_moved += sum(1 for bid, st in before.items() if bid in p.balls and
                                    p.balls[bid].effect_stamp != st)

    def evolve(self) -> None:
        p = self.park
        cand = [b for bid in sorted(p.moribund) if (b := p.balls[bid]).is_moribund
                and not (b.effect_stamp - p.current_time < 0) and b.effect_stamp - p.current_time - 2 < 0]
        if len(cand) > 7:
            self.limit_binds += 1
        p.bring_out_the_deads()
        p.evolve()


class BatchedWorlds(SB.BatchedWorlds):
    """space_batch's BatchedWorlds with DelayedRemoval: the removals in their records' places, BringOutTheDeads before
    the park's tick."""

    def __init__(self, cs: list, device: str, park: str = "orient", removal_cls=DelayedRemoval, **kw):
        super().__init__(cs, device, park, **kw)
        self.removal = removal_cls(self.space)

    def _apply(self, recs: list) -> None:
        rest = [None if (rec is not None and rec[0] in KINDS) else rec for rec in recs]
        if any(rec is not None for rec in rest):
            super()._apply(rest)
        rem = [(wi, rec) for wi, rec in enumerate(recs) if rec is not None and rec[0] in KINDS]
        if rem:
            W = self.W
            mask = torch.zeros(W, dtype=BOOL)
            ball = torch.zeros(W, dtype=I64)
            delay = torch.zeros(W, dtype=I64)
            for wi, rec in rem:
                mask[wi] = True
                ball[wi] = int(rec[1])
                delay[wi] = int(rec[2]) if rec[0] == "remove_later" else 0
            self.removal.remove_ball(mask, ball, delay)

    def _evolve(self) -> None:
        self.removal.bring_out_the_deads()
        super()._evolve()


def _removal_state_scalar(sw: ScalarWorld) -> dict:
    p = sw.park
    return {"balls": {b.id: (bool(b.is_moribund), bool(b.is_massive), int(b.effect_stamp)) for b in p.balls.values()},
            "pending": sorted(p.moribund)}


def _removal_state_batched(bw: BatchedWorlds, wi: int) -> dict:
    sp = bw.space
    t = {k: v.cpu() for k, v in bw.view.tensors().items()}
    ids = sp.slot_ids(wi)
    in_park = sp.st["in_park"][wi].cpu()
    pend = bw.removal.pending[wi].cpu()
    balls, pending = {}, []
    for j, bid in ids.items():
        if not in_park[j]:
            continue
        r = int(sp.rows[wi, j])
        balls[bid] = (bool(t["moribund"][r]), bool(t["massive"][r]), int(t["effect_stamp"][r]))
        if pend[j]:
            pending.append(bid)
    return {"balls": balls, "pending": sorted(pending)}


def compare(cs: list, device: str, park: str = "orient", steps: int | None = None, removal_cls=DelayedRemoval) -> dict:
    """space_batch.compare with the delayed removal: the differences (step, world, what), the volumes, the flagged
    worlds (excluded), the reference's branch counts and this module's coverage."""
    T = len(cs[0]["steps"]) if steps is None else int(steps)
    scal = [ScalarWorld(w) for w in cs]
    bw = BatchedWorlds(cs, device, park, removal_cls=removal_cls, hits=False)
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
        for wi in range(len(cs)):
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
            rw, rg = _removal_state_scalar(scal[wi]), _removal_state_batched(bw, wi)
            compared["removal_fields"] += 3 * len(rw["balls"]) + 1
            compared["waiting_ball_steps"] += len(rw["pending"])
            if rw != rg:
                bad = [bid for bid in rw["balls"] if rw["balls"][bid] != rg["balls"].get(bid)]
                diffs.append((t, wi, "removal", bad[:3], rw["pending"], rg["pending"]))
            ev_w = SB.scalar_events(scal[wi])
            ev_g = bw.space.events(wi)
            compared["events"] += len(ev_w)
            compared["final_cleanups"] += sum(1 for e in ev_w if e[0] == "DoFinalCleanup")
            if ev_w != ev_g:
                diffs.append((t, wi, "events", ev_w, ev_g))
            aw, ag = scal[wi].answers, bw.answers[wi]
            compared["answers"] += len(aw)
            if aw != ag:
                diffs.append((t, wi, "answers", aw, ag))
            scal[wi].answers, bw.answers[wi] = [], []
            compared["ball_steps"] += len(want["balls"])
    flagged = [wi for wi, f in enumerate(bw.space.st["flags"].cpu().tolist()) if f]
    ref_hits = Counter()
    for sw in scal:
        ref_hits.update(sw.park.trace)
    cov = {"remove.delayed": ref_hits.get("remove.delayed", 0), "remove.now": ref_hits.get("remove.now", 0),
           "remove.buffer": ref_hits.get("remove.buffer", 0),
           "limit_binds": sum(sw.limit_binds for sw in scal), "stamp_moved": sum(sw.stamp_moved for sw in scal)}
    return {"diffs": diffs, "compared": dict(compared), "flagged": flagged, "coverage": cov, "steps": T}

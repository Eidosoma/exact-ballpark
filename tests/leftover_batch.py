# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The harness (31 October 2026): the batched park (exact_ballpark.vec_modes.VecModesPark, its commands on
tensors and its CUDA kernel a tick) on tests/leftover_cases.py's scenarios (full_cases' records) and the scalar
reference (exact_ballpark.full.FullBallpark) on the same, in one row layout, so the two compare field by field, every ball
every step, and GetCenterDist/GetSurfaceDist (exact_ballpark.vec_distance) query by query.

As tests/modes_batch.py's batched_full (whose op dispatch it extends with SetBallFree, SetBallRigid, SetBoid,
SetBallTroll, SetBallInteractive and DIST_Q): each scenario a world, its balls rows, its start time their clock; each
step's records applied in their order (the j-th record of every world together, op by op), then the park evolves.

The rows are modes_batch.FIELDS_FULL with `free` and `interactive` added. The follower set: the park's (the balls whose
follow_ptr is the row and whose follow id is its id: VecModesPark.followers) against the reference's set less its stale
entries (a follower whose follow_ptr or follow id no longer names the target: Destiny's StopAllFollowers erases them
without effect); `stale_entries` counts what was left out.
"""

from __future__ import annotations

from collections import defaultdict

import torch

import full_cases as fc
import modes_batch as mb
from exact_ballpark.vec_distance import park_dist
from exact_ballpark.vec_modes import VecModesPark

F64, I64 = torch.float64, torch.int64
FIELDS = tuple(f for f in mb.FIELDS_FULL if f != "followers") + ("free", "interactive", "followers")
NEW_OPS = (fc.FREE_OP, fc.RIGID_OP, fc.BOID_OP, fc.TROLL_OP, fc.INTERACTIVE_OP)


def _columns(park: VecModesPark) -> dict:
    cols = mb._columns(park)
    ids, ptr, fid = park.ids.tolist(), park.follow_ptr.tolist(), park.follow_id.tolist()
    fol = defaultdict(list)
    for row, (p, f) in enumerate(zip(ptr, fid)):
        if p >= 0 and f == ids[p]:
            fol[p].append(ids[row])
    cols["followers"] = [tuple(sorted(fol.get(row, ()))) for row in range(park.n)]
    cols["free"] = [int(v) for v in park.free.tolist()]
    cols["interactive"] = [int(v) for v in park.interactive.tolist()]
    return cols


def _apply(park: VecModesPark, op: int, items: list, index: dict, step: int, queries: dict) -> None:
    """One op of full_cases' records on its rows: items are (case id, record)."""
    if op not in NEW_OPS and op != fc.DIST_Q:
        mb._apply_full(park, op, items, index)
        return
    n, dev = park.n, park.dev
    if op == fc.DIST_Q:
        a = torch.tensor([index.get((cid, rec[3]), -1) for cid, rec in items], dtype=I64, device=dev)
        b = torch.tensor([index.get((cid, rec[4]), -1) for cid, rec in items], dtype=I64, device=dev)
        center, surface, found = park_dist(park, a, b)
        for (cid, rec), c, s, ok in zip(items, center.tolist(), surface.tolist(), found.tolist()):
            queries[(cid, step, fc.DIST_Q, rec[3], rec[4])].append((1, (c, s)) if ok else (0, ()))
        return
    mask = torch.zeros(n, dtype=torch.bool)
    w = torch.zeros(n, dtype=I64)
    for cid, (_, _, _, a, _, _, ww, _) in items:
        row = index.get((cid, a))
        if row is None:
            continue
        mask[row] = True
        w[row] = ww
    mask, w = mask.to(dev), w.to(dev)
    if op == fc.FREE_OP:
        park.set_free(mask, w)
    elif op == fc.RIGID_OP:
        park.set_ball_rigid(mask)
    elif op == fc.BOID_OP:
        park.set_boid(mask, w)
    elif op == fc.TROLL_OP:
        park.set_ball_troll(mask, w)
    else:
        park.set_interactive(mask, w)


def batched(cs: list, device: str = "cuda", lib: dict | None = None, block: int = 128, park_hook=None,
            ticks: bool = True) -> tuple:
    """({(case, step, ball id): the FIELDS values}, {(case, step, op, a, b): [(status, (centre, surface))]}) for the
    scenarios cs (balls added at step 0, a client's view), steps 0 to the scenario's last. `park_hook(park)` may
    return the tick to use (a CUDA graph's). With ticks False no tick runs (the commands alone, on any device: the
    scenarios must be marked manual, so the reference does not evolve either)."""
    groups = defaultdict(list)
    for c in cs:
        assert not c["master"] and not c.get("iterative") and bool(c.get("manual")) != ticks, c["id"]
        groups[(c["friction"], c["tick"], bool(c["new_orbit"]), bool(c["do"]))].append(c)
    out, queries = {}, defaultdict(list)
    for (friction, tick, new_orbit, do), members in groups.items():
        balls = []
        for c in members:
            for rec in c["records"]:
                if rec[0] == "B":
                    assert rec[1] == 0, (c["id"], rec)
                    balls.append((c, rec[2]))
        park = VecModesPark(len(balls), friction=friction, tick_interval_ms=tick, device=device,
                            dynamical_orientation=do, use_new_orbit=new_orbit, block=block)
        park.load([{"id": b["id"], "world": c["id"], "time": c["start"], "mass": b["mass"], "radius": b["radius"],
                    "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                    "speed_fraction": b["sf"], "free": b["free"], "massive": b["massive"],
                    "interactive": b["interactive"], "max_ang_vel": b["max_ang_vel"],
                    "rot_agility": b["rot_agility"]} for c, b in balls])
        index = {(c["id"], b["id"]): i for i, (c, b) in enumerate(balls)}
        per_step = defaultdict(lambda: defaultdict(list))
        for c in members:
            for rec in c["records"]:
                if rec[0] == "K":
                    per_step[rec[1]][c["id"]].append(rec)
        last = max(c["steps"] for c in members)
        tick_fn = (park_hook(park) if park_hook is not None else None) or (lambda: park.evolve(lib))
        for s in range(last + 1):
            todo = per_step.get(s, {})
            for j in range(max((len(v) for v in todo.values()), default=0)):
                by_op = defaultdict(list)
                for cid, recs in todo.items():
                    if j < len(recs):
                        by_op[recs[j][2]].append((cid, recs[j]))
                for op, items in by_op.items():
                    _apply(park, op, items, index, s, queries)
            if s > 0 and ticks:
                tick_fn()
            cols = _columns(park)
            for i, (c, b) in enumerate(balls):
                if s <= c["steps"]:
                    out[(c["id"], s, b["id"])] = tuple(cols[f][i] for f in FIELDS)
    return out, dict(queries)


def reference(cs: list) -> tuple:
    """(rows, queries, stale entries, undefined case ids) of the reference (full_cases.reference) in batched's layout:
    the follower set less its stale entries (counted). A scenario where Destiny's result is undefined (the reference
    raises DestinyUndefined) is left out and named."""
    idx = [fc.S_FIELDS.index(f) for f in FIELDS]
    i_fp, i_fid, i_fol = fc.S_FIELDS.index("follow_ptr"), fc.S_FIELDS.index("follow_id"), fc.S_FIELDS.index("followers")
    rows, queries, stale, undefined = {}, defaultdict(list), 0, []
    for c in cs:
        try:
            res = fc.reference(c)
        except fc.DestinyUndefined:
            undefined.append(c["id"])
            continue
        by_step = defaultdict(dict)
        for row in res["S"]:
            by_step[row[1]][row[2]] = row
        for s, balls in by_step.items():
            for bid, row in balls.items():
                keep = tuple(f for f in row[i_fol] if f in balls and balls[f][i_fp] == bid and balls[f][i_fid] == bid)
                stale += len(row[i_fol]) - len(keep)
                vals = [row[i] for i in idx]
                vals[-1] = keep
                rows[(row[0], s, bid)] = tuple(vals)
        for (cid, s, op, a, b, status, items) in res["Q"]:
            queries[(cid, s, op, a, b)].append((status, tuple(items)))
    return rows, dict(queries), stale, undefined


def query_differences(want: dict, got: dict) -> list:
    """The queries whose status or values differ (doubles by their bits, two NaNs equal)."""
    if set(want) != set(got):
        return [("keys", len(want), len(got), sorted(set(want) ^ set(got))[:5])]
    bad = []
    for key, w in want.items():
        g = got[key]
        if len(w) != len(g):
            bad.append((key, "count"))
            continue
        for (sw, vw), (sg, vg) in zip(w, g):
            if sw != sg or len(vw) != len(vg) or not all(fc.same(float(x), float(y)) for x, y in zip(vw, vg)):
                bad.append(key)
    return bad

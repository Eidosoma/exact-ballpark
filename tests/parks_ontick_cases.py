# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""OnTick's intervals in the four batched parks: every world driven by its own
simulation clock through `on_tick` (exact_ballpark.vec_orient.VecOrientPark.on_tick, inherited by VecModesPark,
VecCombatPark and VecComposedPark), against the scalar reference driven the same way, one FullBallpark a world
(exact_ballpark.full.FullBallpark.on_tick, Ballpark.cpp:219-330, TIME-702 to TIME-705), compared bit for bit after every
call: every field the batch holds of every ball, which balls exist, the DoCollision events, and each world's clocks
(mTime, mFirstTime and, where the park keeps it, mCurrentTime).

**The clocks.** Each world starts at a simulation time of 0, under one interval, one interval or several (a first call
too early, or one with a backlog), and each call advances it by its own step: nothing, under one interval (too early:
no tick, the time kept), one interval and a little more (the remainder carried), two to four intervals (as many ticks in
one call), or more than five (a master's resynchronisation: one tick; a client catches every interval up); a world's
first call that holds an interval evolves once. **The scenarios** are the earlier generators, frozen before
this file: tests/flight_cases.py's (VecModesPark) and tests/turn_cases.py's batch kinds (VecOrientPark), in
tests/full_cases.py's records, made OnTick's here (`with_ontick`: manual, an ONTICK_OP record closing each step, so
Destiny's compiled code runs them too, `fc.oracle`); tests/destiny_combat_cases.py's worlds (VecCombatPark) and
tests/park_cases.py's (VecComposedPark), each world's tick replaced by its OnTick call."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import torch

import combat_leftover_cases as clb
import destiny_combat_cases as dc
import full_cases as fc
import modes_batch as mb
import orient_batch as ob
import park_cases as pc
from exact_ballpark.full import FullBallpark
from exact_ballpark.space import DestinyUndefined

STEPS = (0.0, 0.4, 0.9999, 1.0, 1.0, 1.0, 1.0, 1.25, 1.6, 2.0, 2.3, 3.0, 4.2, 5.5, 6.0, 7.4)


def clocks(r: random.Random, tick_ms: int, calls: int) -> list:
    """A world's simulation times (100 ns units) at its `calls` OnTick calls: a start of 0, under one interval, one
    interval or a backlog of up to 20, then each call's advance from STEPS (intervals) with a jitter under 0.1 ms."""
    interval = int(tick_ms) * 10000
    t = r.choice([0, r.randrange(1, interval), interval, r.randrange(interval, 20 * interval)])
    out = []
    for _ in range(calls):
        k = r.choice(STEPS)
        t += int(k * interval) + (r.randint(0, 999) if k else 0)
        out.append(t)
    return out


# ============================================================ full_cases' scenarios (VecModesPark, VecOrientPark)

def with_ontick(c: dict, r: random.Random, master: bool | None = None) -> dict:
    """A full_cases scenario made OnTick's: manual (no tick of its own), each step s >= 1 closed by an ONTICK_OP record
    at the world's simulation time for call s, after the step's other records; `master` overrides the view."""
    sims = clocks(r, c["tick"], c["steps"])
    recs = list(c["records"]) + [fc._k(s, fc.ONTICK_OP, sims[s - 1]) for s in range(1, c["steps"] + 1)]
    out = {**c, "manual": True, "records": sorted(recs, key=lambda rec: rec[1]), "sims": sims}
    if master is not None:
        out["master"] = bool(master)
    return out


def reference_full(c: dict, trace=None) -> tuple:
    """full_cases.reference's run of one OnTick scenario, keeping the park: (its rows by kind, {step: (mTime,
    mFirstTime, mCurrentTime)}, the park's hit counts)."""
    park = FullBallpark(friction=c["friction"], tick_interval_ms=c["tick"], use_new_orbit=c["new_orbit"],
                        is_master=c["master"], seed=c["seed"], iterative=c.get("iterative", False),
                        dynamical_orientation=c.get("do", False),
                        disable_do_for_missiles=c.get("do_missiles_off", False))
    park.current_time = c["start"]
    park.trace = Counter() if trace is None else trace
    out = defaultdict(list)
    by_step = defaultdict(list)
    for rec in c["records"]:
        by_step[rec[1]].append(rec)
    clk = {}
    for s in range(c["steps"] + 1):
        for rec in by_step.get(s, ()):
            fc._apply(park, rec, c["id"], s, out)
        fc._state(out, c["id"], s, park)
        clk[s] = (int(park.time), bool(park.first_time), int(park.current_time))
    return out, clk, dict(park.trace)


def references_full(cs: list, fields: tuple) -> tuple:
    """Every scenario's reference: ({(case, step, ball): row in `fields`' order}, {(case, step): clocks}, hit counts
    summed)."""
    idx = [fc.S_FIELDS.index(f) for f in fields]
    rows, clk, hits = {}, {}, defaultdict(int)
    for c in cs:
        out, ck, h = reference_full(c)
        for row in out["S"]:
            rows[(row[0], row[1], row[2])] = tuple(row[i] for i in idx)
        for s, v in ck.items():
            clk[(c["id"], s)] = v
        for k, v in h.items():
            hits[k] += v
    return rows, clk, dict(hits)


def _per_step(cs: list) -> tuple:
    """{step: {case: its records but the OnTick, in order}} and {(case, step): the OnTick's simulation time}."""
    per_step = defaultdict(lambda: defaultdict(list))
    sims = {}
    for c in cs:
        for rec in c["records"]:
            if rec[0] != "K":
                continue
            if rec[2] == fc.ONTICK_OP:
                sims[(c["id"], rec[1])] = rec[3]
            else:
                per_step[rec[1]][c["id"]].append(rec)
    return per_step, sims


def _sim_rows(members: list, sims: dict, s: int, last: dict) -> torch.Tensor:
    """Each row's world's simulation time at step s (a scenario past its last step keeps its last time: too early)."""
    out = []
    for c, _ in members:
        t = sims.get((c["id"], s), last.get(c["id"], 0))
        last[c["id"]] = t
        out.append(t)
    return torch.tensor(out, dtype=torch.int64)


def batched_modes(cs: list, device: str = "cuda", graph: bool = False, lib: dict | None = None) -> tuple:
    """VecModesPark driven by `on_tick` on OnTick scenarios of flight_cases (modes_batch.batched_full's grouping and
    layout): ({(case, step, ball): mb.FIELDS_FULL row}, {(case, step): clocks}, OnTick's masks: (rows ticked, rows
    out) summed). `graph`: the ticks a CUDA graph's replays (vec_modes_cuda.ModesGraph with OnTick's mask)."""
    from exact_ballpark.vec_modes import VecModesPark
    groups = defaultdict(list)
    for c in cs:
        assert not c["master"] and c["manual"], c["id"]
        groups[(c["friction"], c["tick"], bool(c["new_orbit"]), bool(c["do"]))].append(c)
    out, clk, seen = {}, {}, [0, 0]
    for (friction, tick, new_orbit, do), group in groups.items():
        members = [(c, rec[2]) for c in group for rec in c["records"] if rec[0] == "B"]
        assert all(rec[1] == 0 for c in group for rec in c["records"] if rec[0] == "B")
        park = VecModesPark(len(members), friction=friction, tick_interval_ms=tick, device=device,
                            dynamical_orientation=do, use_new_orbit=new_orbit)
        park.load([{"id": b["id"], "world": c["id"], "time": c["start"], "mass": b["mass"], "radius": b["radius"],
                    "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                    "speed_fraction": b["sf"], "free": b["free"], "massive": b["massive"],
                    "interactive": b["interactive"], "max_ang_vel": b["max_ang_vel"],
                    "rot_agility": b["rot_agility"]} for c, b in members])
        index = {(c["id"], b["id"]): i for i, (c, b) in enumerate(members)}
        per_step, sims = _per_step(group)
        runner = None
        if graph:
            from exact_ballpark.vec_modes_cuda import ModesGraph
            runner = ModesGraph(park, lib=lib, masked=True)
        tick_fn = (lambda m: runner.tick(m)) if runner else (lambda m: park.evolve(lib, active=m))
        last = {}
        for s in range(max(c["steps"] for c in group) + 1):
            todo = per_step.get(s, {})
            for j in range(max((len(v) for v in todo.values()), default=0)):
                by_op = defaultdict(list)
                for cid, recs in todo.items():
                    if j < len(recs):
                        by_op[recs[j][2]].append((cid, recs[j]))
                for op, items in by_op.items():
                    mb._apply_full(park, op, items, index)
            if s > 0:
                for m in park.on_tick(_sim_rows(members, sims, s, last).to(park.dev), tick=tick_fn):
                    k = int(m.sum())
                    seen[0] += k
                    seen[1] += park.n - k
            cols = mb._columns(park)
            t, f, ck = park.time.tolist() if s > 0 else None, None, park.clock.tolist()
            if s > 0:
                f = park.first_time.tolist()
            for i, (c, b) in enumerate(members):
                if s <= c["steps"]:
                    out[(c["id"], s, b["id"])] = tuple(cols[name][i] for name in mb.FIELDS_FULL)
                    clk[(c["id"], s)] = (t[i], f[i], ck[i]) if s > 0 else (0, True, ck[i])
    return out, clk, tuple(seen)


def batched_orient(cs: list, device: str = "cpu", kernel: str = "torch", ypr: str = "host",
                   graph: bool = False) -> tuple:
    """VecOrientPark driven by `on_tick` on OnTick scenarios of turn_cases' batch kinds (orient_batch.batched's
    grouping and layout; every ball a row, a scenario's rows one world): ({(case, step, ball): ob.FIELDS row},
    {(case, step): (mTime, mFirstTime, None)}, the masks' rows (ticked, out)). `graph`: vec_orient_cuda.TickGraph
    with OnTick's mask (dynamical orientation with kernel="cuda", or ypr="cuda")."""
    from exact_ballpark.vec_orient import VecOrientPark
    groups = defaultdict(list)
    for c in cs:
        assert c["kind"] in ob.BATCH_KINDS and not c["master"] and c["manual"], c["id"]
        for rec in c["records"]:
            if rec[0] == "B":
                groups[(c["friction"], c["tick"], bool(c["do"]))].append((c, rec[2]))
    out, clk, seen = {}, {}, [0, 0]
    for (friction, tick, do), members in groups.items():
        park = VecOrientPark(len(members), friction=friction, tick_interval_ms=tick, device=device,
                             dynamical_orientation=do, kernel=kernel if do else "torch",
                             ypr=ypr if not do else "host")
        park.load([{"mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"], "pos": b["pos"],
                    "vel": b["vel"], "agility": b["agility"], "speed_fraction": b["sf"],
                    "max_ang_vel": b["max_ang_vel"], "rot_agility": b["rot_agility"]} for _, b in members])
        index = {(c["id"], b["id"]): i for i, (c, b) in enumerate(members)}
        cases = list({id(c): c for c, _ in members}.values())
        per_step = defaultdict(lambda: defaultdict(list))
        sims = {}
        for c in cases:
            for rec in c["records"]:
                if rec[0] != "K":
                    continue
                if rec[2] == fc.ONTICK_OP:
                    sims[(c["id"], rec[1])] = rec[3]
                else:
                    per_step[rec[1]][index[(c["id"], rec[3])]].append(rec)
        runner = None
        if graph:
            from exact_ballpark.vec_orient_cuda import TickGraph
            runner = TickGraph(park, masked=True)
        tick_fn = (lambda m: runner.tick(m)) if runner else (lambda m: park.evolve(active=m))
        last = {}
        for s in range(max(c["steps"] for c in cases) + 1):
            todo = per_step.get(s, {})
            for j in range(max((len(v) for v in todo.values()), default=0)):
                by_op = defaultdict(list)
                for row, recs in todo.items():
                    if j < len(recs):
                        by_op[recs[j][2]].append((row, recs[j][7]))
                for op, items in by_op.items():
                    mask = torch.zeros(park.n, dtype=torch.bool)
                    vals = torch.zeros(park.n, 4, dtype=torch.float64)
                    for row, d in items:
                        mask[row] = True
                        vals[row] = torch.tensor((list(d) + [0.0] * 4)[:4], dtype=torch.float64)
                    ob._apply(park, op, mask.to(park.dev), vals.to(park.dev))
            if s > 0:
                for m in park.on_tick(_sim_rows(members, sims, s, last).to(park.dev), tick=tick_fn):
                    k = int(m.sum())
                    seen[0] += k
                    seen[1] += park.n - k
            rows = ob._rows(park)
            t = park.time.tolist() if s > 0 else None
            f = park.first_time.tolist() if s > 0 else None
            for i, (c, b) in enumerate(members):
                if s <= c["steps"]:
                    out[(c["id"], s, b["id"])] = rows[i]
                    clk[(c["id"], s)] = (t[i], f[i], None) if s > 0 else (0, True, None)
    return out, clk, tuple(seen)


def flight(seed: int, n: int = 60) -> list:
    """OnTick scenarios for VecModesPark: n of flight_cases' scenarios of the seed (a client's view each), with clocks
    drawn from the seed."""
    import flight_cases as flc
    r = random.Random(f"parks-ontick-flight-{seed}")
    cs = [c for c in flc.cases(seed) if not c["master"] and not c.get("iterative") and not c.get("manual")]
    cs = r.sample(cs, min(n, len(cs)))
    return [with_ontick(c, r) for c in sorted(cs, key=lambda c: c["id"])]


def turning(seed: int, n: int = 60) -> list:
    """OnTick scenarios for VecOrientPark: n of turn_cases' batch-kind scenarios of the seed, with clocks."""
    import turn_cases as tc
    r = random.Random(f"parks-ontick-turn-{seed}")
    cs = [c for c in tc.cases(seed) if c["kind"] in ob.BATCH_KINDS]
    cs = r.sample(cs, min(n, len(cs)))
    return [with_ontick(c, r) for c in sorted(cs, key=lambda c: c["id"])]


# ========================================================== the combat and composed worlds (VecCombatPark, ...)

FAMILIES = ("combat", "leftovers", "composed")      # destiny_combat_cases', combat_leftover_cases', park_cases' worlds


def worlds(family: str, seed: int, nb: int = 4, nw: int = 8) -> list:
    """A family's batches of the seed (its generator's), each world with its OnTick clocks (`w["sims"]`, one a step):
    "combat" VecCombatPark's, "leftovers" its mushrooms, trolls and the rest, "composed"
    VecComposedPark's."""
    gen = {"combat": dc.batches, "leftovers": clb.batches, "composed": pc.batches}[family]
    r = random.Random(f"parks-ontick-{family}-{seed}")
    out = gen(seed, nb, nw)
    for cfg in out:
        for w in cfg["worlds"]:
            w["sims"] = clocks(r, cfg["tick"], cfg["steps"])
    return out


def _ref_row(family: str, b, do: bool) -> dict:
    if family == "composed":
        return {**dc._ref_row(b, do), **pc._ref_extra(b)}
    if family == "leftovers":
        return clb._ref_row(b, do)
    return dc._ref_row(b, do)


def reference_world(family: str, cfg: dict, w: dict) -> dict:
    """One world on FullBallpark driven by OnTick (its family's reference run, each tick replaced by the world's OnTick
    call): the family's layout, with "C": {step: (mTime, mFirstTime, mCurrentTime)} and "H": the park's hit counts."""
    park = FullBallpark(friction=cfg["friction"], tick_interval_ms=cfg["tick"], use_new_orbit=cfg["new_orbit"],
                        is_master=cfg["master"], dynamical_orientation=cfg["do"],
                        disable_do_for_missiles=cfg["do_off"])
    park.current_time = w["start"]
    park.trace = Counter()
    for s in w["ships"]:
        dc._add_ref(park, s)
    park.load_formations(dc.FORMATIONS)
    by_step = defaultdict(list)
    for rec in w["records"]:
        by_step[rec[0]].append(rec)
    out = {"S": {}, "E": [], "A": [], "B": [], "X": None, "C": {}}
    apply = {"combat": dc._apply_ref, "leftovers": clb._apply_ref, "composed": pc._apply_ref}[family]
    try:
        for s in range(cfg["steps"] + 1):
            for rec in by_step.get(s, ()):
                apply(park, rec, out["A"])
            if s > 0:
                park.on_tick(w["sims"][s - 1])
            out["S"][s] = {b.id: _ref_row(family, b, cfg["do"]) for b in park.balls.values()}
            out["C"][s] = (int(park.time), bool(park.first_time), int(park.current_time))
            if cfg["master"] and len({b.bubble for b in park.balls.values()}) > 1:
                out["B"].append(s)
    except DestinyUndefined as exc:
        out["X"] = str(exc)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:     # where Python raises and C computes on
        if family != "composed":
            raise
        out["X"] = f"the reference raised {type(exc).__name__}: {exc}"
    out["E"] = [(t, bid, args[0]) for (t, name, bid, *args) in park.events if name == "DoCollision"]
    out["H"] = dict(park.trace)
    return out


def batched_worlds(family: str, cfg: dict, device: str, path: str = "torch", graph: bool = False) -> tuple:
    """A batch of the family on VecCombatPark (VecComposedPark for "composed") driven by `on_tick`, the layout of
    `reference_world` per world (with "C"): (outs, park, OnTick's masks' rows (ticked, out)). path "torch"
    (evolve_torch) or "kernel" (evolve); `graph`: the kernels' CUDA graph with OnTick's mask."""
    import exact_ballpark.vec_combat as vc
    if family == "leftovers":
        rows, where = clb.rows_for(cfg), None
    else:
        rows, where = dc.rows_for(cfg)
    if family == "composed":
        from exact_ballpark.vec_composed import VecComposedPark
        cls = VecComposedPark
    else:
        cls = vc.VecCombatPark
    p = cls(len(rows), cfg["friction"], cfg["tick"], device, cfg["do"], cfg["new_orbit"], master=cfg["master"],
            disable_do_for_missiles=cfg["do_off"])
    p.load(rows)
    p.load_formations(dc.FORMATIONS)
    W = len(cfg["worlds"])
    outs = [{"S": {}, "E": [], "A": [], "B": [], "X": None, "C": {}} for _ in range(W)]
    by_step = [defaultdict(list) for _ in range(W)]
    for wi, w in enumerate(cfg["worlds"]):
        for rec in w["records"]:
            by_step[wi][rec[0]].append(rec)
    n = len(rows)
    names = clb.fields(cfg["do"]) if family == "leftovers" else dc.fields(cfg["do"])
    world_of = p.world.tolist()
    first_row = {}
    for r, wi in enumerate(world_of):
        first_row.setdefault(wi, r)
    if family == "leftovers":
        ids0 = p.ids.tolist()
        p.mush_rows = {wi: [r for r in range(n) if world_of[r] == wi and ids0[r] in clb.MUSH_POOL] for wi in range(W)}
    runner = None
    if graph:
        if family == "composed":
            from exact_ballpark.vec_composed_cuda import ComposedGraph
            runner = ComposedGraph(p, masked=True)
        else:
            from exact_ballpark.vec_combat_cuda import CombatGraph
            runner = CombatGraph(p, masked=True)

    def tick_fn(m):
        if runner is not None:
            # the eager tick's domain watch at its start and end (VecCombatPark.evolve; a graph's replay holds the
            # kernels only), the rows outside the mask given back their flags as a masked tick gives them
            f0 = p.flags.clone()
            p.watch()
            runner.tick(m)
            p.after_tick()
            p.watch()
            if m is not None:
                p.flags = torch.where(m, p.flags, f0)
        elif path == "kernel":
            p.evolve(active=m)
        else:
            p.evolve_torch(active=m)

    def collect(m):
        for (wl, t, mid, col) in p.events():
            outs[wl]["E"].append((t, mid, col))

    seen = [0, 0]
    for s in range(cfg["steps"] + 1):
        slots = max(len(by_step[wi].get(s, ())) for wi in range(W))
        for j in range(slots):
            recs = {wi: by_step[wi][s][j] for wi in range(W) if j < len(by_step[wi].get(s, ()))}
            if family == "leftovers":
                clb.apply_batch(p, recs, outs)
            elif family == "composed":
                pc.apply_batch(p, recs, where, n, outs)
            else:
                dc.apply_batch(p, recs, where, n, outs)
        if s > 0:
            sim = torch.tensor([cfg["worlds"][world_of[r]]["sims"][s - 1] for r in range(n)], dtype=torch.int64)
            for m in p.on_tick(sim.to(p.dev), tick=tick_fn, after=collect):
                k = int(m.sum())
                seen[0] += k
                seen[1] += n - k
        cache = {k: getattr(p, k).cpu().tolist() for k in names}
        if family == "composed":
            last_c, level, key, mod = (getattr(p, k).cpu().tolist() for k in ("last_c", "level", "key", "mod"))
            lists = p.collision_lists()
        t = p.time.tolist() if s > 0 else None
        f = p.first_time.tolist() if s > 0 else None
        ck = p.clock.tolist()
        for wi in range(W):
            outs[wi]["S"][s] = {}
            r0 = first_row[wi]
            outs[wi]["C"][s] = (t[r0], f[r0], ck[r0]) if s > 0 else (0, True, ck[r0])
        if family == "leftovers":
            held = clb._rows_by_id(p).items()
        else:
            exists = p.exists.cpu().tolist()
            held = [(k, r) for k, r in where.items() if exists[r]]
        for (wi, bid), r in held:
            if family == "leftovers":
                row = {k: (int(cache[k][r]) if k in clb.FLAGS else dc._value(cache[k][r], k)) for k in names}
            else:
                row = {k: dc._value(cache[k][r], k) for k in names}
            if family == "composed":
                row.update(last_c=tuple(float(c) for c in last_c[r]), collisions=lists[r], level=int(level[r]),
                           key=tuple(int(c) for c in key[r]), mod=tuple(int(c) for c in mod[r]))
            outs[wi]["S"][s][bid] = row
    return outs, p, tuple(seen)


def torch_domain(family: str, w: dict) -> bool:
    """A world VecCombatPark.evolve_torch holds (no FOLLOW, ORBIT or WARP); VecComposedPark's holds every mode."""
    if family == "composed":
        return True
    return (clb.torch_domain if family == "leftovers" else dc.torch_domain)(w)


def compare_worlds(family: str, cfgs: list, refs: list, device: str, path: str = "torch", graph: bool = False) -> dict:
    """Every batch of `cfgs` on the batched park against its references, world by world: counts and the first
    differences (rows, events, answers, clocks); worlds outside the reference's domain or flagged by the batch are
    counted, not compared; on the torch path only the worlds it holds (`torch_domain`)."""
    out = Counter()
    bad = []
    for bi, cfg in enumerate(cfgs):
        if graph and not cfg["do"]:
            continue
        idx = [wi for wi, w in enumerate(cfg["worlds"]) if path != "torch" or torch_domain(family, w)]
        if not idx:
            continue
        sub = {**cfg, "worlds": [cfg["worlds"][wi] for wi in idx]}
        got, park, seen = batched_worlds(family, sub, device, path, graph)
        out["masked rows ticked"] += seen[0]
        out["masked rows out"] += seen[1]
        flags = park.world_flags()
        for k, wi in enumerate(idx):
            want = refs[bi][wi]
            if want["X"] is not None or want["B"]:
                out["outside the reference's domain"] += 1
                continue
            if flags[k]:
                out["refused"] += 1
                continue
            out["worlds"] += 1
            for name, v in want["H"].items():
                if name.startswith("tick.") or name.startswith("mushroom."):
                    out[name] += v
            g = got[k]
            d = list(dc.differences(want, g))
            if g["C"] != want["C"]:
                d.append(("clocks", [(s, want["C"][s], g["C"].get(s)) for s in want["C"]
                                     if want["C"][s] != g["C"].get(s)][:3]))
            out["ball_steps"] += sum(len(v) for v in want["S"].values())
            out["events"] += len(want["E"])
            if d:
                out["differing"] += 1
                bad.append((bi, wi, d[:3]))
    return {"counts": dict(out), "bad": bad[:10], "n_bad": len(bad)}

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The batched flight modes (exact_ballpark.vec_modes.VecModesPark, one CUDA kernel a tick) on the scalar reference's
scenarios, in the reference's own row layout, so the two can be compared field by field:

- tests/mode_cases.py's scenarios (FOLLOW, ORBIT under both laws, WARP with its transitions, EntityWarpIn, the guards,
  mixed worlds; exact_ballpark.modes.ModeBallpark), every ball every step: mode_cases' state rows (mode, effect stamp,
  flags, follow fields, position, velocity, goto point, thrust, follow range, warp length, follower set);
- tests/flight_cases.py's scenarios in tests/full_cases.py's records (exact_ballpark.full.FullBallpark: the modes with
  dynamical orientation on and off and every setter in flight), every ball every step: the fields of full_cases' S rows
  the park holds (FIELDS_FULL).

Every scenario is a world of one park per (friction, tick, orbit law, dynamical orientation): its balls are rows, its
start time the rows' clock; each step's records are applied in their order (the j-th record of every world together,
op by op), then the park evolves.
"""

from __future__ import annotations

from collections import defaultdict

import torch

import full_cases as fc
import mode_cases as mc
from exact_ballpark.modes import FOLLOW, ORBIT
from exact_ballpark.vec_modes import VecModesPark

F64, I64 = torch.float64, torch.int64
FIELDS_MODES = mc.FIELDS[3:]          # every field of mode_cases' rows after (case, step, id)
FIELDS_FULL = ("mode", "effect_stamp", "massive", "sensor", "follow_id", "owner_id", "follow_ptr", "px", "py", "pz",
               "vx", "vy", "vz", "opx", "opy", "opz", "ovx", "ovy", "ovz", "gx", "gy", "gz", "ax", "ay", "az", "mass",
               "radius", "max_vel", "agility", "speed_fraction", "follow_range", "last_collision", "time_factor",
               "max_ang_vel", "rot_roll", "rot_agility", "rx", "ry", "rz", "rw", "orx", "ory", "orz", "orw", "wx", "wy",
               "wz", "owx", "owy", "owz", "tx", "ty", "tz", "followers")
_TURNING = ("max_ang_vel", "rot_roll", "rot_agility", "rx", "ry", "rz", "rw", "orx", "ory", "orz", "orw", "wx", "wy",
            "wz", "owx", "owy", "owz", "tx", "ty", "tz")


_VECS = (("p", "pos", "xyz"), ("v", "vel", "xyz"), ("op", "old_pos", "xyz"), ("ov", "old_vel", "xyz"),
         ("g", "goto", "xyz"), ("a", "last_g", "xyz"), ("w", "ang_vel", "xyz"), ("ow", "old_ang_vel", "xyz"),
         ("t", "torque", "xyz"), ("r", "rot", "xyzw"), ("or", "old_rot", "xyzw"))
_DOUBLES = ("mass", "radius", "max_vel", "agility", "speed_fraction", "follow_range", "last_collision", "time_factor",
            "max_ang_vel", "rot_roll", "rot_agility")
_INTS = ("mode", "effect_stamp", "follow_id", "owner_id", "ids", "follow_ptr", "massive", "sensor")


def _columns(park: VecModesPark) -> dict:
    """Every column the rows need, as Python lists (two transfers: the doubles and the integers)."""
    dd = torch.cat([getattr(park, name) for _, name, _ in _VECS] + [getattr(park, n)[:, None] for n in _DOUBLES],
                   dim=1).cpu().t().tolist()
    ii = torch.stack([getattr(park, n).to(I64) for n in _INTS], dim=1).cpu().t().tolist()
    cols, k = {}, 0
    for prefix, _, axes in _VECS:
        for axis in axes:
            cols[prefix + axis] = dd[k]
            k += 1
    for name in _DOUBLES:
        cols[name] = dd[k]
        k += 1
    for j, name in enumerate(_INTS):
        cols[name] = ii[j]
    ptr = cols["follow_ptr"]
    ids = cols["ids"]
    cols["follow_ptr"] = [-1 if p < 0 else ids[p] for p in ptr]
    fol = defaultdict(list)
    for row, (m, p) in enumerate(zip(cols["mode"], ptr)):
        if m in (FOLLOW, ORBIT) and p >= 0:
            fol[p].append(ids[row])
    cols["followers"] = [tuple(sorted(fol.get(row, ()))) for row in range(park.n)]
    return cols


# ============================================================================== mode_cases (ModeBallpark)

def _apply_modes(park: VecModesPark, op: int, items: list, index: dict) -> None:
    """One mode_cases op on its rows: items are (case id, command record)."""
    n, dev = park.n, park.dev
    mask = torch.zeros(n, dtype=torch.bool)
    target = torch.full((n,), -1, dtype=I64)
    xyz = torch.zeros(n, 3, dtype=F64)
    rr = torch.zeros(n, dtype=F64)
    w = torch.zeros(n, dtype=I64)
    for cid, (_, _, a, b, x, y, z, r, ww) in items:
        row = index.get((cid, a))
        if row is None:                                   # a missing ball: Destiny's lookup fails, nothing happens
            continue
        mask[row] = True
        target[row] = index.get((cid, b), -1)
        xyz[row] = torch.tensor((x, y, z), dtype=F64)
        rr[row], w[row] = r, ww
    mask, target, xyz, rr, w = (t.to(dev) for t in (mask, target, xyz, rr, w))
    if op == mc.GOTO_OP:
        park.goto_point(mask, xyz)
    elif op == mc.DIR_OP:
        park.goto_direction(mask, xyz)
    elif op == mc.STOP_OP:
        park.stop(mask)
    elif op == mc.FOLLOW_OP:
        park.follow(mask, target, rr)
    elif op == mc.ORBIT_OP:
        park.orbit(mask, target, rr)
    elif op == mc.WARP_OP:
        park.warp_to(mask, xyz, rr, w)
    elif op == mc.WARPIN_OP:
        park.entity_warp_in(mask, xyz, w)
    else:
        raise ValueError(f"op {op}")


def batched_modes(cs: list, device: str = "cuda", lib: dict | None = None, block: int = 128, park_hook=None,
                  record=None) -> dict:
    """{(case, step, ball id): mode_cases' row fields after (case, step, id)} for the scenarios cs. `park_hook(park)`
    may return the tick to use (a CUDA graph's); `record`, a set of steps, limits the rows to those steps."""
    groups = defaultdict(list)
    for c in cs:
        groups[(c["friction"], c["tick"], bool(c["new_orbit"]))].append(c)
    out = {}
    for (friction, tick, new_orbit), members in groups.items():
        balls = [(c, b) for c in members for b in c["balls"]]
        park = VecModesPark(len(balls), friction=friction, tick_interval_ms=tick, device=device,
                            use_new_orbit=new_orbit, block=block)
        park.load([{"id": b["id"], "world": c["id"], "time": c["start"], "mass": b["mass"], "radius": b["radius"],
                    "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                    "speed_fraction": b["sf"], "free": b["free"], "massive": b["massive"],
                    "interactive": b["interactive"], "cloaked": b["cloaked"], "bubble": b["bubble"],
                    "moribund": b["moribund"], "mode": b["mode"]} for c, b in balls])
        index = {(c["id"], b["id"]): i for i, (c, b) in enumerate(balls)}
        per_step = defaultdict(lambda: defaultdict(list))            # step -> case -> its commands in order
        for c in members:
            for k in c["cmds"]:
                per_step[k[0]][c["id"]].append(k)
        last = max(c["steps"] for c in members)
        tick = (park_hook(park) if park_hook is not None else None) or (lambda: park.evolve(lib))
        for s in range(1, last + 1):
            todo = per_step.get(s, {})
            for j in range(max((len(v) for v in todo.values()), default=0)):
                by_op = defaultdict(list)
                for cid, cmds in todo.items():
                    if j < len(cmds):
                        by_op[cmds[j][1]].append((cid, cmds[j]))
                for op, items in by_op.items():
                    _apply_modes(park, op, items, index)
            tick()
            if record is not None and s not in record:
                continue
            cols = _columns(park)
            for i, (c, b) in enumerate(balls):
                if s <= c["steps"]:
                    out[(c["id"], s, b["id"])] = tuple(cols[f][i] for f in FIELDS_MODES)
    return out


def reference_modes(cs: list) -> dict:
    """The reference's rows (mode_cases.reference) by (case, step, ball id)."""
    out = {}
    for c in cs:
        rows, _ = mc.reference(c)
        for row in rows:
            out[(row[0], row[1], row[2])] = tuple(row[3:])
    return out


def differences(want: dict, got: dict, fields: tuple) -> list:
    """(case, step, ball, field) for every value that differs (doubles by their bits, two NaNs equal)."""
    if set(want) != set(got):
        return [("rows", len(want), len(got), sorted(set(want) ^ set(got))[:5])]
    bad = []
    for key, w in want.items():
        g = got[key]
        for f, a, b in zip(fields, w, g):
            if isinstance(a, float) or isinstance(b, float):
                if not fc.same(float(a), float(b)):
                    bad.append((*key, f))
            elif a != b:
                bad.append((*key, f))
    return bad


# ============================================================================= full_cases' records (FullBallpark)

def _apply_full(park: VecModesPark, op: int, items: list, index: dict) -> None:
    """One full_cases op on its rows: items are (case id, record)."""
    n, dev = park.n, park.dev
    mask = torch.zeros(n, dtype=torch.bool)
    target = torch.full((n,), -1, dtype=I64)
    vals = torch.zeros(n, 4, dtype=F64)
    w = torch.zeros(n, dtype=I64)
    for cid, (_, _, _, a, bb, _, ww, d) in items:
        row = index.get((cid, a))
        if row is None:
            continue
        mask[row] = True
        target[row] = index.get((cid, bb), -1)
        vals[row] = torch.tensor((list(d) + [0.0] * 4)[:4], dtype=F64)
        w[row] = ww
    mask, target, vals, w = (t.to(dev) for t in (mask, target, vals, w))
    xyz, rr = vals[:, :3].contiguous(), vals[:, 3].contiguous()
    if op == fc.GOTO_OP:
        park.goto_point(mask, xyz)
    elif op == fc.DIR_OP:
        park.goto_direction(mask, xyz)
    elif op == fc.STOP_OP:
        park.stop(mask)
    elif op == fc.FOLLOW_OP:
        park.follow(mask, target, rr)
    elif op == fc.ORBIT_OP:
        park.orbit(mask, target, rr)
    elif op == fc.WARP_OP:
        park.warp_to(mask, xyz, rr, w)
    elif op == fc.WARPIN_OP:
        park.entity_warp_in(mask, xyz, w)
    elif op == fc.SF_OP:
        park.set_speed_fraction(mask, rr)
    elif op == fc.MASS_OP:
        park.set_mass(mask, rr)
    elif op == fc.AGILITY_OP:
        park.set_agility(mask, rr)
    elif op == fc.MAXVEL_OP:
        park.set_max_velocity(mask, rr)
    elif op == fc.RADIUS_OP:
        park.set_radius(mask, rr)
    elif op == fc.VEL_OP:
        park.set_velocity(mask, xyz)
    elif op == fc.MAX_ANGVEL_OP:
        park.set_max_angular_speed(mask, vals[:, 0].contiguous())
    elif op == fc.ROT_AGILITY_OP:
        park.set_rot_agility(mask, vals[:, 0].contiguous())
    elif op == fc.ANGVEL_OP:
        park.set_angular_velocity(mask, xyz)
    elif op == fc.ROTATION_OP:
        park.set_rotation(mask, vals)
    else:
        raise ValueError(f"op {op} is not batched")


def batched_full(cs: list, device: str = "cuda", lib: dict | None = None, block: int = 128, park_hook=None) -> dict:
    """{(case, step, ball id): the FIELDS_FULL values} for scenarios in full_cases' records (balls added at step 0, a
    client's view), steps 0 to the scenario's last. `park_hook(park)` may return the tick to use."""
    groups = defaultdict(list)
    for c in cs:
        assert not c["master"] and not c.get("iterative") and not c.get("manual"), c["id"]
        groups[(c["friction"], c["tick"], bool(c["new_orbit"]), bool(c["do"]))].append(c)
    out = {}
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
        tick = (park_hook(park) if park_hook is not None else None) or (lambda: park.evolve(lib))
        for s in range(last + 1):
            todo = per_step.get(s, {})
            for j in range(max((len(v) for v in todo.values()), default=0)):
                by_op = defaultdict(list)
                for cid, recs in todo.items():
                    if j < len(recs):
                        by_op[recs[j][2]].append((cid, recs[j]))
                for op, items in by_op.items():
                    _apply_full(park, op, items, index)
            if s > 0:
                tick()
            cols = _columns(park)
            for i, (c, b) in enumerate(balls):
                if s <= c["steps"]:
                    out[(c["id"], s, b["id"])] = tuple(cols[f][i] for f in FIELDS_FULL)
    return out


def reference_full(cs: list) -> dict:
    """The reference's ball states (full_cases.reference) by (case, step, ball id), in FIELDS_FULL's order."""
    idx = [fc.S_FIELDS.index(f) for f in FIELDS_FULL]
    out = {}
    for c in cs:
        for row in fc.reference(c)["S"]:
            out[(row[0], row[1], row[2])] = tuple(row[i] for i in idx)
    return out

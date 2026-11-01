# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The batched turning (exact_ballpark.vec_orient.VecOrientPark) on tests/turn_cases.py's batch-compatible scenarios
("vec", "vec_do"), in the reference's ball-state layout (full_cases' S rows), so the two can be compared field by
field: every ball a row of one park per (friction, tick, dynamical orientation), each step's records applied in
their order (a ball's j-th records of a step together, op by op), then the tick."""

from __future__ import annotations

from collections import defaultdict

import torch

import full_cases as fc
from exact_ballpark.vec_orient import VecOrientPark

FIELDS = ("mode", "px", "py", "pz", "vx", "vy", "vz", "opx", "opy", "opz", "ovx", "ovy", "ovz", "gx", "gy", "gz",
          "ax", "ay", "az", "mass", "radius", "max_vel", "agility", "speed_fraction", "time_factor", "yaw", "old_yaw",
          "pitch", "old_pitch", "roll", "old_roll", "roll_speed", "old_roll_speed", "yaw_delta", "max_ang_vel",
          "rot_roll", "rot_agility", "rx", "ry", "rz", "rw", "orx", "ory", "orz", "orw", "wx", "wy", "wz", "owx", "owy",
          "owz", "tx", "ty", "tz")
_VEC3 = {"p": "pos", "v": "vel", "op": "old_pos", "ov": "old_vel", "g": "goto", "a": "last_g", "w": "ang_vel",
         "ow": "old_ang_vel", "t": "torque"}
_SCALAR = {"mass": "mass", "radius": "radius", "max_vel": "max_vel", "agility": "agility",
           "speed_fraction": "speed_fraction", "time_factor": "time_factor", "yaw": "yaw", "old_yaw": "old_yaw",
           "pitch": "pitch", "old_pitch": "old_pitch", "roll": "roll", "old_roll": "old_roll",
           "roll_speed": "roll_speed", "old_roll_speed": "old_roll_speed", "yaw_delta": "yaw_delta",
           "max_ang_vel": "max_ang_vel", "rot_roll": "rot_roll", "rot_agility": "rot_agility"}
BATCH_KINDS = ("vec", "vec_do")


def _apply(park: VecOrientPark, op: int, mask: torch.Tensor, vals: torch.Tensor) -> None:
    """One op on the masked rows; vals (B, 4) holds each row's record's d, padded as full_cases pads it."""
    rr = vals[:, 3]
    if op == fc.GOTO_OP:
        park.goto_point(mask, vals[:, :3])
    elif op == fc.DIR_OP:
        park.goto_direction(mask, vals[:, :3])
    elif op == fc.STOP_OP:
        park.stop(mask)
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
        park.set_velocity(mask, vals[:, :3])
    elif op == fc.MAX_ANGVEL_OP:
        park.set_max_angular_speed(mask, vals[:, 0])
    elif op == fc.ROT_AGILITY_OP:
        park.set_rot_agility(mask, vals[:, 0])
    elif op == fc.ANGVEL_OP:
        park.set_angular_velocity(mask, vals[:, :3])
    elif op == fc.ROTATION_OP:
        park.set_rotation(mask, vals[:, :4])
    else:
        raise ValueError(f"op {op} is not batched")


def _rows(park: VecOrientPark) -> list:
    """Every row's state in FIELDS' order, as Python numbers."""
    cols = {"mode": park.mode.cpu().tolist()}
    for prefix, name in _VEC3.items():
        t = getattr(park, name).cpu().tolist()
        for j, axis in enumerate("xyz"):
            cols[prefix + axis] = [row[j] for row in t]
    for f, name in _SCALAR.items():
        cols[f] = getattr(park, name).cpu().tolist()
    for prefix, name in (("r", "rot"), ("or", "old_rot")):
        t = getattr(park, name).cpu().tolist()
        for j, axis in enumerate("xyzw"):
            cols[prefix + axis] = [row[j] for row in t]
    return [tuple(cols[f][i] for f in FIELDS) for i in range(park.n)]


def batched(cs: list, device: str, park_hook=None) -> dict:
    """{(case, step, ball id): state tuple in FIELDS' order} for the batch-compatible scenarios of cs."""
    groups = defaultdict(list)
    for c in cs:
        if c["kind"] in BATCH_KINDS:
            for rec in c["records"]:
                if rec[0] == "B":
                    groups[(c["friction"], c["tick"], bool(c["do"]))].append((c, rec[2]))
    out = {}
    for (friction, tick, do), members in groups.items():
        park = VecOrientPark(len(members), friction=friction, tick_interval_ms=tick, device=device,
                             dynamical_orientation=do)
        if park_hook is not None:
            park_hook(park)
        park.load([{"mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"], "pos": b["pos"],
                    "vel": b["vel"], "agility": b["agility"], "speed_fraction": b["sf"],
                    "max_ang_vel": b["max_ang_vel"], "rot_agility": b["rot_agility"]} for _, b in members])
        index = {(c["id"], b["id"]): i for i, (c, b) in enumerate(members)}
        per_step = defaultdict(lambda: defaultdict(list))            # step -> row -> its records in order
        for c in {id(c): c for c, _ in members}.values():
            for rec in c["records"]:
                if rec[0] == "K":
                    per_step[rec[1]][index[(c["id"], rec[3])]].append(rec)
        last = max(c["steps"] for c, _ in members)
        for s in range(last + 1):
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
                    _apply(park, op, mask.to(park.dev), vals.to(park.dev))
            if s > 0:
                park.evolve()
            rows = _rows(park)
            for (c, b), i in zip(members, range(park.n)):
                if s <= c["steps"]:
                    out[(c["id"], s, b["id"])] = rows[i]
    return out


def reference_rows(cs: list) -> dict:
    """The reference's ball states (full_cases.reference) of the batch-compatible scenarios, in FIELDS' order."""
    idx = [fc.S_FIELDS.index(f) for f in FIELDS]
    out = {}
    for c in cs:
        if c["kind"] in BATCH_KINDS:
            for row in fc.reference(c)["S"]:
                out[(row[0], row[1], row[2])] = tuple(row[i] for i in idx)
    return out


def differences(want: dict, got: dict) -> list:
    """(case, step, ball, field) for every value that differs (doubles by their bits, two NaNs equal)."""
    bad = []
    if set(want) != set(got):
        return [("rows", len(want), len(got), sorted(set(want) ^ set(got))[:5])]
    for key, w in want.items():
        g = got[key]
        for f, a, b in zip(FIELDS, w, g):
            if not fc.same(float(a) if f != "mode" else a, float(b) if f != "mode" else b):
                bad.append((*key, f))
    return bad

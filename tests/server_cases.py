# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Command streams for Destiny's server layer batched, generated from a seed, and the two
runners the tests compare bit for bit: the scalar server layer (exact_ballpark.server: Actions, Ticker, BubbleUpdater
and ParkUpdateBatcher on a master DestinyPark over FullBallpark, one stack a world) and the batched layer
(exact_ballpark.vec_server over VecModesPark, VecOrientPark or VecCollidePark), both driven as Destiny's own test helper
drives the layer (test_ticker.py:172-182): each tick the world's Actions calls in order, the Ticker's tick, the park's
Evolve (or none: the pre-tick alone), the batcher's send_batch.

A world is one bubble (2 to 5 ships within 15 km of the origin, so the master files them all in bubble 0) with its
pilots: ship 0 is interactive and flown by pilot 0 and never made non-interactive (a dead bubble is refused); the other
ships are interactive or not, with a pilot or not. One "special" ship a world does what can make a ship massive or move
it far (cloak and uncloak, SetBallMassive, WarpTo, EntityWarpIn; the parks without collisions hold only worlds whose
massive ships cannot meet); in collision worlds (VecCollidePark) every ship is massive and any may cloak with modes 2
and 3 (the normal cloak's sensor is not modelled there).

Each tick a world issues 0 to 6 calls: Destiny's commands and setters (refused values now and then), several for one
ship (conflicting commands, a Stop after a command), the same call twice in a row (TIME-710 drops the second), the
same call with another between (kept), a local and a non-local SetBallPosition, cloaks and uncloaks in both orders in
one tick, follows and orbits of the special ship around its cloaks, warps and warp-ins with followers, and with
dynamical orientation the turning setters.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import torch

import full_cases as fc
import modes_batch as mb
from exact_ballpark import server as srv
from exact_ballpark import vec_server as VS
from exact_ballpark.client import PilotInterests, RecordingNetwork

F64, I64 = torch.float64, torch.int64
PARKS = ("modes", "orient", "collide")
FIELDS_MODES = mb.FIELDS_FULL + ("cloaked", "interactive")
FIELDS_ORIENT = ("mode", "px", "py", "pz", "vx", "vy", "vz", "opx", "opy", "opz", "ovx", "ovy", "ovz", "gx", "gy", "gz",
                 "ax", "ay", "az", "mass", "radius", "max_vel", "agility", "speed_fraction", "time_factor",
                 "massive", "cloaked", "interactive", "sensor")
FIELDS_ORIENT_OLD = ("yaw", "old_yaw", "pitch", "old_pitch", "roll", "old_roll", "roll_speed", "old_roll_speed",
                     "yaw_delta")
FIELDS_ORIENT_DO = ("max_ang_vel", "rot_roll", "rot_agility", "rx", "ry", "rz", "rw", "orx", "ory", "orz", "orw",
                    "wx", "wy", "wz", "owx", "owy", "owz", "tx", "ty", "tz")
FIELDS_COLLIDE = ("mode", "px", "py", "pz", "vx", "vy", "vz", "opx", "opy", "opz", "ovx", "ovy", "ovz", "gx", "gy",
                  "gz", "ax", "ay", "az", "cx", "cy", "cz", "last_collision", "speed_fraction", "massive", "cloaked",
                  "interactive", "box_level", "box_ix", "box_iy", "box_iz", "mod_x", "mod_y", "mod_z", "collisions")


# ============================================================================================== the generator

def _f(r: random.Random, lo: float, hi: float) -> float:
    return float(r.uniform(lo, hi))


def _nan() -> float:
    return float("nan")                                     # a new object each time (no identity shortcut)


def _world(r: random.Random, wid: int, park: str, do: bool) -> dict:
    n = r.randint(2, 5)
    ids = r.sample(range(101, 999), n)
    if park == "collide":
        ids = sorted(ids)                                   # the park's slots are in ascending id
    balls, pilots, p = [], {}, 0
    for k, bid in enumerate(ids):
        inter = k == 0 or r.random() < 0.6
        b = {"id": bid, "mass": 10 ** r.uniform(6.0, 9.0), "radius": r.choice([_f(r, 5.0, 60.0), _f(r, 60.0, 300.0)]),
             "max_vel": r.choice([_f(r, 50.0, 500.0), _f(r, 5.0, 150.0)]), "agility": _f(r, 0.1, 2.0),
             "sf": r.choice([1.0, _f(r, 0.1, 1.0), 0.0]),
             "pos": [_f(r, -1.5e4, 1.5e4) for _ in range(3)],
             "vel": r.choice([[0.0, 0.0, 0.0], [_f(r, -150.0, 150.0) for _ in range(3)]]),
             "interactive": inter, "massive": park == "collide",
             "max_ang_vel": r.choice([1.0, _f(r, 0.05, 3.0)]), "rot_agility": r.choice([0.9, _f(r, 0.05, 2.0)])}
        if park == "collide":
            b["pos"] = [_f(r, -3e3, 3e3) for _ in range(3)]  # near each other: they meet
        balls.append(b)
        if inter and (k == 0 or r.random() < 0.7):
            pilots[bid] = p
            p += 1
    order = list(range(n))
    if park != "collide":
        r.shuffle(order)                                    # the bubble's order is the order of AddBall
        order.remove(0)
        order.insert(r.randint(0, len(order)), 0)
    special = ids[1] if n > 1 else None
    return {"id": wid, "park": park, "do": do, "balls": [balls[k] for k in order], "anchor": ids[0],
            "special": special, "pilots": pilots, "ticks": []}


def _call(r: random.Random, w: dict, park: str, do: bool) -> tuple:
    """One Actions call (name, args) for a world."""
    ids = [b["id"] for b in w["balls"]]
    s = r.choice(ids)
    other = r.choice([i for i in ids if i != s])
    sp = w["special"]
    refuse = r.random() < 0.08
    xyz = lambda lo, hi: [_f(r, lo, hi) for _ in range(3)]          # noqa: E731
    if park == "collide":
        k = r.random()
        if k < 0.25:
            return "go_to_direction", (s, *([_nan(), 0.0, 1.0] if refuse else xyz(-1.0, 1.0)))
        if k < 0.45:
            return "go_to_point", (s, *xyz(-6e3, 6e3))
        if k < 0.60:
            return "set_speed_fraction", (s, r.choice([-0.5, 1.5, _nan()]) if refuse else r.choice([0.0, 1.0,
                                                                                                     _f(r, 0, 1)]))
        if k < 0.70:
            return "stop", (s,)
        if k < 0.78:
            return "set_ball_massive", (s, r.choice([0, 1]))
        if k < 0.84 and s != w["anchor"]:
            return "set_ball_interactive", (s, r.choice([0, 1]))
        if k < 0.93:
            return "cloak_ball", (s, r.choice([2, 3, 0, 200]), _f(r, 100.0, 3000.0))
        return "uncloak_ball", (s,)
    k = r.random()
    if k < 0.12:
        return "go_to_direction", (s, *([0.0, _nan(), 0.0] if refuse else xyz(-1.0, 1.0)))
    if k < 0.20:
        return "go_to_point", (s, *xyz(-2e4, 2e4))
    if k < 0.28:
        return "set_speed_fraction", (s, r.choice([-0.5, 1.5, _nan()]) if refuse else r.choice([0.0, 1.0, _f(r, 0, 1)]))
    if k < 0.33:
        return "stop", (s,)
    if park == "modes" and k < 0.39:
        return "follow_ball", (s, r.choice([other, sp or other, 4242]), _f(r, 50.0, 3000.0))
    if park == "modes" and k < 0.45:
        return "orbit", (s, r.choice([other, sp or other]), r.choice([_f(r, 300.0, 5000.0), math.inf])
                         if refuse else _f(r, 300.0, 5000.0))
    if park == "modes" and k < 0.50 and sp is not None:
        d = r.choice([1.0, -1.0]) * _f(r, 1.5e5, 6e5)
        if r.random() < 0.15:
            d = _f(r, 1e3, 5e4)                             # within 1e5 m: a GotoPoint
        return "warp_to", (sp, d, _f(r, -1e4, 1e4), _f(r, -1e4, 1e4), r.choice([0.0, 2e4, _f(r, 0, 5e4)]),
                           r.choice([1, 3, 10]))
    if park == "modes" and k < 0.55 and sp is not None:
        return "entity_warp_in", (sp, r.choice([1.0, -1.0]) * _f(r, 2e5, 6e5), _f(r, -1e4, 1e4), 0.0,
                                  r.choice([1, 5]))
    if k < 0.58:
        return "set_max_speed", (s, -5.0 if refuse else _f(r, 10.0, 400.0))
    if k < 0.62:
        return "set_ball_mass", (s, -1.0 if refuse else 10 ** r.uniform(6.0, 9.0))
    if k < 0.66:
        return "set_ball_agility", (s, 0.0 if refuse else _f(r, 0.1, 2.0))
    if k < 0.69:
        return "set_ball_radius", (s, -5.0 if refuse else _f(r, 5.0, 300.0))
    if k < 0.73:
        return "set_ball_velocity", (s, *xyz(-200.0, 200.0))
    if k < 0.76 and s != w["anchor"]:
        return "set_ball_interactive", (s, r.choice([0, 1]))
    if k < 0.79:
        return "set_ball_position", (s, *xyz(-1.5e4, 1.5e4), r.choice([True, False]))
    if k < 0.82 and sp is not None:
        return "set_ball_massive", (sp, r.choice([0, 1]))
    if k < 0.88 and sp is not None:
        return "cloak_ball", (sp, r.choice([1, 2, 3, 0]), _f(r, 100.0, 3000.0))
    if k < 0.93 and sp is not None:
        return "uncloak_ball", (sp,)
    if do:
        j = r.random()
        if j < 0.3:
            return "set_ball_angular_velocity", (s, *xyz(-1.0, 1.0))
        if j < 0.5:
            q = (0.0, 0.0, 0.0, 0.0) if refuse else tuple(r.gauss(0, 1) for _ in range(4))
            return "set_ball_rotation", (s, *q)
        if j < 0.7:
            return "set_max_angular_speed", (s, -1.0 if refuse else _f(r, 0.05, 3.0))
        if j < 0.85:
            return "set_ball_angular_agility", (s, 0.0 if refuse else _f(r, 0.05, 2.0))
        return "set_max_angular_velocity", (s, *xyz(0.1, 2.0))
    return "go_to_direction", (s, *xyz(-1.0, 1.0))


# what interrupts a warp: a command that stops the ship (or re-aims a warp) in warp proper leaves it flying at warp
# speed in normal flight (the public code's mid-warp stop; EntityWarpIn is in warp proper at once), an AU a tick, so
# the master files it in a new bubble; SetBallMassive(1) in warp proper makes Evolve reset its warp length to -1 (the
# gradient's reset of a massive ball), and the warp constants then take the log of zero (the reference refuses; C
# computes NaN positions). Once the special ship has warped it gets none of these (nor a new velocity or position,
# which its warp path would carry on from)
_INTERRUPTS = ("stop", "go_to_point", "go_to_direction", "follow_ball", "orbit", "warp_to", "entity_warp_in",
               "set_ball_velocity", "set_ball_position", "set_ball_massive")


def _tick_calls(r: random.Random, w: dict, park: str, do: bool) -> list:
    calls = []
    warped_before = w.get("warped", False)
    for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4])):
        c = _call(r, w, park, do)
        if c[1][0] == w["special"] and c[0] in _INTERRUPTS and (warped_before or w.get("warped_in")):
            c = ("set_max_speed", (w["special"], _f(r, 50.0, 400.0)))
        if c[0] == "warp_to" and c[1][0] == w["special"]:
            w["warped"] = True                              # aligning this tick: interruptible until the next
        if c[0] == "entity_warp_in" and c[1][0] == w["special"]:
            w["warped"] = w["warped_in"] = True             # in warp proper at once
        calls.append(c)
        x = r.random()
        nan = any(isinstance(v, float) and v != v for v in c[1])
        if x < 0.10 and not nan:                            # (one NaN object twice: Python's identity shortcut)
            calls.append((c[0], tuple(c[1])))               # the same call twice in a row: the second dropped
        elif x < 0.16 and not (c[1][0] == w["special"] and (warped_before or w.get("warped_in"))):
            calls.append(("stop", (c[1][0],)))              # a Stop after a command, same tick
        elif c[0] == "set_ball_position" and x < 0.45:      # the same move with the other locality: not a duplicate
            calls.append((c[0], (*c[1][:4], not c[1][4])))
        elif x < 0.20 and len(calls) >= 2 and not any(isinstance(v, float) and v != v for v in calls[-2][1]):
            again = calls[-2]
            if not (again[1][0] == w["special"] and again[0] in _INTERRUPTS and (warped_before or w.get("warped_in"))):
                calls.append((again[0], tuple(again[1])))   # the same call with another between: kept
    w.pop("warped_in", None)
    return calls


def cases(seed: int, n: int, park: str, do: bool = False, ticks: int = 12) -> list:
    """n worlds for a park ('modes', 'orient', 'collide'), each with `ticks` ticks of calls."""
    r = random.Random(f"server-{park}-{int(do)}-{seed}")
    out = []
    for wid in range(n):
        w = _world(r, wid, park, do)
        w["ticks"] = [_tick_calls(r, w, park, do) for _ in range(ticks)]
        w.pop("warped", None)
        out.append(w)
    return out


# ============================================================================================ the scalar side

class ScalarWorld:
    """One world's scalar server layer on a master DestinyPark (FullBallpark), as Destiny's test helper wires it."""

    def __init__(self, w: dict, friction: float = 1.0e6, tick_ms: int = 1000):
        if w["do"]:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()
        self.park = park = srv.DestinyPark(is_master=True)
        assert (park.lane.friction, park.lane.tick_interval) == (friction, tick_ms), "the parks' defaults"
        for b in w["balls"]:
            args = (b["id"], b["mass"], b["radius"], b["max_vel"], 1, 0, int(b["massive"]), int(b["interactive"]), 0,
                    *b["pos"], *b["vel"], b["agility"], b["sf"])
            if w["do"]:
                args += (b["max_ang_vel"], b["rot_agility"])
            park.AddBall(*args)
        self.interests = PilotInterests(w["pilots"])
        self.network = RecordingNetwork()
        self.actions = srv.Actions(park)
        self.batcher = srv.ParkUpdateBatcher(park, self.network, self.interests, self.interests,
                                             srv.BubbleUpdater(park))
        self.ticker = srv.Ticker(park, self.interests, self.actions, self.batcher)

    def tick(self, calls: list, evolve: bool = True) -> dict:
        for name, args in calls:
            getattr(self.actions, name)(*args)
        self.ticker.tick()
        if evolve:
            self.park.Evolve()
        self.batcher.send_batch()
        return per_client(self.network.take())


def per_client(sent: list) -> dict:
    """The network's sends as each client receives them: {client: [(kind, method, state, wait, count)]}."""
    out = defaultdict(list)
    for kind, to, method, state, wait, count in sent:
        for c in ([to] if kind == "single" else to):
            out[c].append((kind, method, state, wait, count))
    return dict(out)


def scalar_rows(world: ScalarWorld, park: str, do: bool) -> dict:
    """{ball id: the compared fields} of a scalar world, in the batched park's field order."""
    lane = world.park.lane
    out = {}
    for b in lane.balls.values():
        row = dict(zip(fc.S_FIELDS, fc._ball_row(0, 0, lane, b)))
        if park == "modes":
            names = FIELDS_MODES
        elif park == "orient":
            names = FIELDS_ORIENT + (FIELDS_ORIENT_DO if do else FIELDS_ORIENT_OLD)
        else:
            names = FIELDS_COLLIDE
        vals = []
        for f in names:
            v = row[f]
            if f in ("massive", "interactive", "sensor"):
                v = int(v)
            vals.append(v)
        out[b.id] = tuple(vals)
    return out


# =========================================================================================== the batched side

def build(cs: list, park: str, device: str, do: bool = False, friction: float = 1.0e6, tick_ms: int = 1000,
          kernel: str = "torch", contact_slots: int | None = None, server_kernel: str = "torch") -> tuple:
    """(VecServer, index {(world, ball id): row}) for the worlds cs on one batched park; `kernel` the park's,
    `server_kernel` the server layer's (exact_ballpark.vec_server_cuda for "cuda")."""
    from exact_ballpark.vec_collide import VecCollidePark
    from exact_ballpark.vec_modes import VecModesPark
    from exact_ballpark.vec_orient import VecOrientPark
    rows = [(w, b) for w in cs for b in w["balls"]]
    if park == "modes":
        p = VecModesPark(len(rows), friction=friction, tick_interval_ms=tick_ms, device=device,
                         dynamical_orientation=do)
        p.load([{"id": b["id"], "world": w["id"], "time": 0, "mass": b["mass"], "radius": b["radius"],
                 "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                 "speed_fraction": b["sf"], "free": True, "massive": b["massive"], "interactive": b["interactive"],
                 "cloaked": 0, "bubble": 0, "moribund": False, "max_ang_vel": b["max_ang_vel"],
                 "rot_agility": b["rot_agility"]} for w, b in rows])
        a = VS.ModesAdapter(p)
        index = {(w["id"], b["id"]): i for i, (w, b) in enumerate(rows)}
    elif park == "orient":
        p = VecOrientPark(len(rows), friction=friction, tick_interval_ms=tick_ms, device=device,
                          dynamical_orientation=do, kernel=kernel)
        p.load([{"mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"],
                 "agility": b["agility"], "speed_fraction": b["sf"], "max_ang_vel": b["max_ang_vel"],
                 "rot_agility": b["rot_agility"]} for w, b in rows])
        a = VS.OrientAdapter(p, world=[w["id"] for w, b in rows], ids=[b["id"] for w, b in rows],
                             interactive=[b["interactive"] for w, b in rows], massive=[b["massive"] for w, b in rows])
        index = {(w["id"], b["id"]): i for i, (w, b) in enumerate(rows)}
    else:
        slots = max(len(w["balls"]) for w in cs)
        p = VecCollidePark(len(cs), slots, friction=friction, tick_interval_ms=tick_ms, device=device,
                           contact_slots=contact_slots, kernel=kernel)
        p.load([[{"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                  "agility": b["agility"], "speed_fraction": b["sf"], "pos": b["pos"], "vel": b["vel"], "free": True,
                  "massive": b["massive"]} for b in w["balls"]] for w in cs])
        inter = []
        index = {}
        for wi, w in enumerate(cs):
            by_id = {b["id"]: b for b in w["balls"]}
            for s, bid in enumerate(sorted(by_id)):
                index[(w["id"], bid)] = wi * slots + s
            row_inter = [False] * slots
            for s, bid in enumerate(sorted(by_id)):
                row_inter[s] = by_id[bid]["interactive"]
            inter += row_inter
        a = VS.CollideAdapter(p, interactive=inter)
    pilots = [-1] * a.R
    for w in cs:
        for bid, pp in w["pilots"].items():
            pilots[index[(w["id"], bid)]] = pp
    return VS.VecServer(a, pilots, kernel=server_kernel), index


def pushes(cs: list, t: int, device) -> list:
    """Tick t's calls of every world as pushes: the j-th calls of kind K of all worlds together, each world's in its
    own order: [(kind, mask, ball, target, f, n, loc)]."""
    W = len(cs)
    out = []
    width = max((len(w["ticks"][t]) for w in cs), default=0)
    for j in range(width):
        by_kind = defaultdict(list)
        for wi, w in enumerate(cs):
            calls = w["ticks"][t]
            if j < len(calls):
                by_kind[VS.ACTION_KINDS[calls[j][0]]].append((wi, calls[j][1]))
        for kind, items in by_kind.items():
            mask = torch.zeros(W, dtype=torch.bool)
            ball = torch.zeros(W, dtype=I64)
            target = torch.zeros(W, dtype=I64)
            f = torch.zeros(W, 4, dtype=F64)
            n = torch.zeros(W, dtype=I64)
            loc = torch.zeros(W, dtype=I64)
            for wi, args in items:
                mask[wi] = True
                ball[wi] = int(args[0])
                rest = list(args[1:])
                if kind in ("FollowBall", "Orbit"):
                    target[wi] = int(rest[0])
                    f[wi, 0] = float(rest[1])
                elif kind == "WarpTo":
                    f[wi, :4] = torch.tensor([float(x) for x in rest[:4]], dtype=F64)
                    n[wi] = int(rest[4])
                elif kind == "EntityWarpIn":
                    f[wi, :3] = torch.tensor([float(x) for x in rest[:3]], dtype=F64)
                    n[wi] = int(rest[3])
                elif kind in ("SetBallMassive", "SetBallInteractive", "SetBallTroll", "SetBallFree"):
                    n[wi] = int(rest[0])                # (SetBallTroll's delay, SetBallFree's flag)
                elif kind == "SetBallPosition":
                    f[wi, :3] = torch.tensor([float(x) for x in rest[:3]], dtype=F64)
                    loc[wi] = 1 if rest[3] is True else 0
                elif kind == "CloakBall":
                    n[wi] = int(rest[0])
                    f[wi, 0] = float(rest[1])
                elif rest:
                    f[wi, :len(rest)] = torch.tensor([float(x) for x in rest], dtype=F64)
            out.append((kind, *(x.to(device) for x in (mask, ball, target, f, n, loc))))
    return out


def batched_rows(server: VS.VecServer, cs: list, index: dict, park: str, do: bool) -> dict:
    """{(world, ball id): the compared fields} of the batched park."""
    a, p = server.a, server.a.park
    out = {}
    if park == "modes":
        cols = mb._columns(p)
        cols["cloaked"] = p.cloaked.tolist()
        cols["interactive"] = p.interactive.to(I64).tolist()
        names = FIELDS_MODES
    elif park == "orient":
        names = FIELDS_ORIENT + (FIELDS_ORIENT_DO if do else FIELDS_ORIENT_OLD)
        cols = {}
        vecs = (("p", "pos", "xyz"), ("v", "vel", "xyz"), ("op", "old_pos", "xyz"), ("ov", "old_vel", "xyz"),
                ("g", "goto", "xyz"), ("a", "last_g", "xyz"), ("w", "ang_vel", "xyz"), ("ow", "old_ang_vel", "xyz"),
                ("t", "torque", "xyz"), ("r", "rot", "xyzw"), ("or", "old_rot", "xyzw"))
        for pre, attr, axes in vecs:
            t = getattr(p, attr).cpu()
            for i, ax in enumerate(axes):
                cols[pre + ax] = t[:, i].tolist()
        for attr in ("mass", "radius", "max_vel", "agility", "speed_fraction", "time_factor", "max_ang_vel",
                     "rot_roll", "rot_agility", "yaw", "old_yaw", "pitch", "old_pitch", "roll", "old_roll",
                     "roll_speed", "old_roll_speed", "yaw_delta"):
            cols[attr] = getattr(p, attr).cpu().tolist()
        cols["mode"] = p.mode.cpu().tolist()
        for attr in ("massive", "cloaked", "interactive", "sensor"):
            cols[attr] = getattr(a, attr).to(I64).cpu().tolist()
    else:
        names = FIELDS_COLLIDE
        R = a.R
        flat = lambda t: t.reshape(R, *t.shape[2:]).cpu()          # noqa: E731
        cols = {}
        for pre, attr in (("p", "pos"), ("v", "vel"), ("op", "old_pos"), ("ov", "old_vel"), ("g", "goto"),
                          ("a", "last_g"), ("c", "last_c")):
            t = flat(getattr(p, attr))
            for i, ax in enumerate("xyz"):
                cols[pre + ax] = t[:, i].tolist()
        for attr in ("last_collision", "speed_fraction"):
            cols[attr] = flat(getattr(p, attr)).tolist()
        cols["mode"] = flat(p.mode).tolist()
        cols["massive"] = a.massive.to(I64).cpu().tolist()
        cols["cloaked"] = a.cloaked.cpu().tolist()
        cols["interactive"] = a.interactive.to(I64).cpu().tolist()
        cols["box_level"] = flat(p.level).tolist()
        key, mod = flat(p.key), flat(p.mod)
        for i, ax in enumerate("xyz"):
            cols["box_i" + ax] = key[:, i].tolist()
            cols["mod_" + ax] = mod[:, i].tolist()
        hits, hn = p.hits.cpu(), p.hits_n.cpu()
        cols["collisions"] = []
        for w in range(p.W):
            for s in range(p.N):
                if s < p.R:                     # the entries are the other balls' ids, as Potential appends them
                    cols["collisions"].append(tuple(int(o) for o in hits[w, s, :int(hn[w, s])].tolist()))
                else:                           # (VecCollidePark._update, the update(..., ot.id); the
                    cols["collisions"].append(())   # fix of the finding: they were read as slots)
    for (wid, bid), r in index.items():
        out[(wid, bid)] = tuple(cols[f][r] for f in names)
    return out


def canon(x):
    """A value as the tests compare it: floats by their bits, ints and bools apart, bytes, Marshalled by value,
    sequences by type and item."""
    if isinstance(x, bool):
        return ("b", x)
    if isinstance(x, int):
        return ("i", int(x))
    if isinstance(x, float):
        return ("f", "nan" if x != x else x.hex())
    if isinstance(x, (bytes, bytearray)):
        return ("B", bytes(x))
    if isinstance(x, srv.Marshalled):
        return ("M", canon(x.value))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(canon(v) for v in x))
    if isinstance(x, str):
        return ("s", x)
    raise TypeError(type(x))


def same_rows(a: tuple, b: tuple) -> bool:
    return fc.same(tuple(a), tuple(b))


def run(cs: list, park: str, device: str, do: bool = False, evolve: bool = True, ticks: int | None = None,
        kernel: str = "torch", contact_slots: int | None = None, record_every: bool = True,
        server_kernel: str = "torch") -> dict:
    """Both sides on the worlds cs, tick by tick: {"rows": [(tick, world, ball, field)] that differ, "sends": [(tick,
    world, client)] whose received updates differ, "ticks", "compared"} (doubles by their bits)."""
    T = len(cs[0]["ticks"]) if ticks is None else int(ticks)
    old = srv.is_dynamical_orientation_enabled()
    try:
        scal = [ScalarWorld(w) for w in cs]
        server, index = build(cs, park, device, do=do, kernel=kernel, contact_slots=contact_slots,
                              server_kernel=server_kernel)
        bad_rows, bad_sends, compared, sends = [], [], 0, 0
        for t in range(T):
            want_sends = [sw.tick(w["ticks"][t], evolve=evolve) for sw, w in zip(scal, cs)]
            for kind, *data in pushes(cs, t, server.dev):
                server.push(kind, *data)
            server.pre_tick(snapshot=True)
            if evolve:
                server.evolve()
            server.post_tick()
            got_sends = VS.decode(server)
            for wi, w in enumerate(cs):
                if canon(_listify(want_sends[wi])) != canon(_listify(got_sends[wi])):
                    bad_sends.append((t, w["id"], sorted(set(want_sends[wi]) ^ set(got_sends[wi]))
                                      or _first_diff(want_sends[wi], got_sends[wi])))
                sends += len(want_sends[wi])
            if record_every or t == T - 1:
                got = batched_rows(server, cs, index, park, do)
                for wi, w in enumerate(cs):
                    for bid, row in scalar_rows(scal[wi], park, do).items():
                        g = got[(w["id"], bid)]
                        compared += 1
                        if not same_rows(row, g):
                            names = (FIELDS_MODES if park == "modes" else FIELDS_COLLIDE if park == "collide" else
                                     FIELDS_ORIENT + (FIELDS_ORIENT_DO if do else FIELDS_ORIENT_OLD))
                            diff = [f for f, x, y in zip(names, row, g) if not fc.same(x, y)]
                            bad_rows.append((t, w["id"], bid, diff))
                    lane = scal[wi].park.lane
                    assert all(b.bubble == 0 for b in lane.balls.values()), "a ball left bubble 0: invalid world"
        server.check() if evolve else None
        return {"rows": bad_rows, "sends": bad_sends, "ticks": T, "compared": compared, "updates": sends}
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()


def _listify(per: dict) -> list:
    return sorted((c, [list(u) for u in v]) for c, v in per.items())


def _first_diff(a: dict, b: dict):
    for c in sorted(set(a) | set(b)):
        if canon([list(u) for u in a.get(c, [])]) != canon([list(u) for u in b.get(c, [])]):
            return (c, a.get(c), b.get(c))
    return None

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Command streams for the combat server's kernel (exact_ballpark.vec_server_combat_cuda), generated from
a seed, and the runners that hold the kernel to the torch path (CombatServer, kernel="torch") every tick, and to the
scalar server layer (tests/server_combat_cases.py's worlds and ScalarWorld).

The kernel's worlds are tests/server_combat_cases.py's (3 to 5 ships, ship 1 interactive and flown by pilot 0, 1 to 3
missile balls at rest, ship 2 leading a formation followed by ships 3 and 4 in reserved slots), with or without
dynamical orientation, and with 0 to 2 empty rows (VecCombatPark's `exists` false: no ball) placed before the world's
balls: one may hold the id of a ball of the world (a lookup that ignores `exists` would find it first), the other a
fresh id. Every Actions call the park applies, on any ball: flights, Stop (ships, formation followers, missiles),
FollowBall and Orbit (of and by missiles too; a NaN range), WarpTo (short and long) and EntityWarpIn (releasing
missiles and formation followers), SetBallVelocity (the yaw, pitch and roll's snap without dynamical orientation, the
rotation's with it; at rest: the direction toward mGoto), SetBallPosition (local or not), the setters (negative values
refused), the angular setters with dynamical orientation, CloakBall (of a formation leader, a missile's target, a
missile; modes 0, 1, 2, -1, 257) and UncloakBall, SetBallInteractive (not ship 1), SetBallMassive, LaunchMissile
(relaunches, a ship launched, launchers at rest and moving, aimed or not, at ships, missiles, the launcher, the
missile itself, an empty row's id, a missing ball, a negative id; owners negative (defenders, flagged) and missing;
aimed and massive chars past a byte), calls on empty rows' ids and on missing balls, duplicates. The kernel against the
torch path holds outside the park's domain too (both run the same park kernels at Evolve), so these worlds are not
kept inside it; the scalar comparisons use tests/server_combat_cases.py's worlds, which are.
"""

from __future__ import annotations

import random
import struct

import torch

import server_cases as S
import server_combat_cases as SC
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_combat as VSC

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
NAN = float("nan")


def _f(r: random.Random, lo: float, hi: float) -> float:
    return float(r.uniform(lo, hi))


def _world(r: random.Random, wid: int, do: bool, empty: bool) -> dict:
    w = SC._world(r, wid, "cuda")
    w["do"] = bool(do)
    w["empty"] = []
    if empty:
        for k in range(r.choice([0, 1, 1, 2])):
            bid = r.choice([b["id"] for b in w["balls"]]) if k == 0 and r.random() < 0.7 else 201 + k
            w["empty"].append({"id": bid, "mass": 1.0e6, "radius": _f(r, 20.0, 200.0), "max_vel": 100.0,
                               "agility": 1.0, "sf": 1.0, "pos": [_f(r, -6e3, 6e3) for _ in range(3)],
                               "vel": [0.0, 0.0, 0.0], "interactive": False, "massive": False, "max_ang_vel": 1.0,
                               "rot_agility": 0.9})
    return w


def _vec(r: random.Random, scale: float) -> tuple:
    return tuple(_f(r, -scale, scale) for _ in range(3))


def _call(r: random.Random, w: dict, do: bool) -> tuple:
    ships, missiles, sp = w["ships"], w["missiles"], w["special"]
    ghosts = [b["id"] for b in w["empty"]] + [4242]
    k = r.random()
    who = r.choice(ships * 3 + missiles * 2 + ghosts)            # the commanded ball (an empty row's or missing id)
    any_ball = r.choice(ships + missiles + ghosts + [-3])
    if k < 0.07:
        return "go_to_point", (who, *_vec(r, 8e3))
    if k < 0.14:
        return "go_to_direction", (who, *r.choice([_vec(r, 1.0), (0.0, 0.0, 0.0), (NAN, 0.0, 1.0)]))
    if k < 0.22:
        return "stop", (r.choice(w["formation"]["followers"] + missiles + [who]),)
    if k < 0.25:
        return "set_speed_fraction", (who, r.choice([0.0, 1.0, _f(r, -0.5, 1.5)]))
    if k < 0.41:
        m = r.choice(missiles * 4 + ships)                     # a missile (relaunched too) or a ship
        launcher = r.choice(ships * 3 + missiles + ghosts)
        owner = launcher if r.random() < 0.85 else -launcher   # a negative owner: a defender (flagged)
        target = r.choice(ships * 3 + missiles + [launcher, m, any_ball, 4242, -3])
        aimed = r.choice([0, 1, 1, 2, 257, -1])
        massive = r.choice([0, 0, 1, 256, -255])
        return "launch_missile", (m, target, owner, aimed, massive)
    if k < 0.49:
        return "cloak_ball", (r.choice([sp, sp, 2, *missiles, who]), r.choice([1, 1, 2, 0, -1, 257]),
                              _f(r, 100.0, 6000.0))
    if k < 0.54:
        return "uncloak_ball", (r.choice([sp, sp, 2, *missiles, who]),)
    if k < 0.57:
        return "set_ball_interactive", (r.choice(ships[1:] + missiles), r.choice([0, 1]))
    if k < 0.60:
        return "set_ball_massive", (who, r.choice([0, 1]))
    if k < 0.70:
        rng = r.choice([_f(r, 200.0, 3000.0), _f(r, 200.0, 3000.0), NAN])
        return r.choice(["follow_ball", "orbit"]), (who, any_ball, rng)
    if k < 0.74:
        far = r.random() < 0.6
        pt = _vec(r, 5e6) if far else tuple(x + _f(r, -2e4, 2e4) for x in w["balls"][0]["pos"])
        return "warp_to", (who, *pt, r.choice([0.0, _f(r, 1e3, 1e5), -5.0]), r.choice([1, 3, 0, -2]))
    if k < 0.77:
        return "entity_warp_in", (who, *_vec(r, 5e6), r.choice([1, 3, 0]))
    if k < 0.84:
        return "set_ball_velocity", (who, *r.choice([_vec(r, 300.0), _vec(r, 300.0), (0.0, 0.0, 0.0)]))
    if k < 0.87:
        return "set_ball_position", (who, *_vec(r, 6e3), r.random() < 0.3)
    if k < 0.91:
        name = r.choice(["set_max_speed", "set_ball_mass", "set_ball_agility", "set_ball_radius"])
        lo, hi = {"set_max_speed": (-50.0, 500.0), "set_ball_mass": (-1e6, 1e8), "set_ball_agility": (-0.5, 2.0),
                  "set_ball_radius": (-20.0, 300.0)}[name]
        return name, (who, _f(r, lo, hi))
    if do and k < 0.96:
        name = r.choice(["set_max_angular_speed", "set_ball_angular_velocity", "set_ball_rotation",
                         "set_ball_angular_agility", "set_max_angular_velocity"])
        if name in ("set_ball_angular_velocity", "set_max_angular_velocity"):
            return name, (who, *_vec(r, 0.5))
        if name == "set_ball_rotation":
            return name, (who, *_vec(r, 1.0), _f(r, -1.0, 1.0))
        return name, (who, _f(r, -0.5, 2.0))
    return "go_to_direction", (who, *_vec(r, 1.0))


def cases(seed: int, n: int, do: bool = False, ticks: int = 12, empty: bool = True) -> list:
    r = random.Random(f"server-combat-kernel-{int(do)}-{int(empty)}-{seed}")
    out = []
    for wid in range(n):
        w = _world(r, wid, do, empty)
        for _ in range(ticks):
            calls = []
            for _ in range(r.choice([0, 1, 2, 2, 3, 4, 5])):
                c = _call(r, w, do)
                calls.append(c)
                if r.random() < 0.08:
                    calls.append((c[0], tuple(c[1])))       # the same call twice in a row: the second dropped
            w["ticks"].append(calls)
        out.append(w)
    return out


# =========================================================================================== the batched side

def build(cs: list, device: str, kernel: str = "torch", block: int | None = None, fault: str | None = None) -> tuple:
    """(CombatServer, index {(world, ball id): row}) on one VecCombatPark (master), each world's empty rows first, the
    formations set as tests/server_combat_cases.build sets them; `kernel` the server layer's."""
    from exact_ballpark.vec_combat import VecCombatPark
    do = bool(cs[0].get("do", False))
    rows = []
    for w in cs:
        rows += [(w, b, False) for b in w.get("empty", [])] + [(w, b, True) for b in w["balls"]]
    p = VecCombatPark(len(rows), device=device, master=True, dynamical_orientation=do)
    p.load([{"id": b["id"], "world": w["id"], "time": 0, "mass": b["mass"], "radius": b["radius"],
             "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
             "speed_fraction": b["sf"], "free": True, "massive": b["massive"], "interactive": b["interactive"],
             "cloaked": 0, "bubble": 0, "moribund": False, "max_ang_vel": b["max_ang_vel"],
             "rot_agility": b["rot_agility"], "exists": ex} for w, b, ex in rows])
    index = {(w["id"], b["id"]): i for i, (w, b, ex) in enumerate(rows) if ex}
    p.load_formations(SC.FORMATIONS)
    R, dev = len(rows), p.dev
    lead = torch.zeros(R, dtype=BOOL)
    fid = torch.zeros(R, dtype=I64)
    for w in cs:
        lead[index[(w["id"], w["formation"]["leader"])]] = True
        fid[index[(w["id"], w["formation"]["leader"])]] = w["formation"]["formation"]
    p.set_ball_formation(lead.to(dev), fid.to(dev))
    depth = max(len(w["formation"]["followers"]) for w in cs)
    for k in range(depth):
        m = torch.zeros(R, dtype=BOOL)
        leader = torch.zeros(R, dtype=I64)
        slot = torch.zeros(R, dtype=I64)
        for w in cs:
            fs = w["formation"]["followers"]
            if k < len(fs):
                r = index[(w["id"], fs[k])]
                m[r] = True
                leader[r] = w["formation"]["leader"]
                slot[r] = p.reserve_formation_slot(index[(w["id"], w["formation"]["leader"])])
        p.formation_follow(m.to(dev), leader.to(dev), slot.to(dev))
    a = VSC.CombatAdapter(p)
    pilots = [-1] * a.R
    for w in cs:
        for bid, pp in w["pilots"].items():
            pilots[index[(w["id"], bid)]] = pp
    return VSC.CombatServer(a, pilots, kernel=kernel, block=block, fault=fault), index


def pushes(cs: list, t: int, device) -> list:
    return SC.pushes(cs, t, device)


# ================================================================================================ the comparisons

def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal by the bits (two NaNs equal), integers and flags exactly; dtype and shape too."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        ia = a.contiguous().view(torch.int64 if a.dtype == F64 else torch.int32)
        ib = b.contiguous().view(torch.int64 if b.dtype == F64 else torch.int32)
        return bool(((ia == ib) | (torch.isnan(a) & torch.isnan(b))).all())
    return torch.equal(a, b)


def tensors_differing(x, y) -> list:
    """The names of the tensor attributes of x and y (vars) that differ, or that one of them lacks."""
    tx = {k: v for k, v in vars(x).items() if torch.is_tensor(v)}
    ty = {k: v for k, v in vars(y).items() if torch.is_tensor(v)}
    return sorted(k for k in set(tx) | set(ty) if k not in tx or k not in ty or not _bits_equal(tx[k], ty[k]))


def batch_view(server) -> dict:
    """Every field of the last Batch, on the host, each slot's data where it was queued (the _batch_view)."""
    b = server.batch
    out = {"stamp": b.stamp.tolist(), "clock": b.clock.tolist(), "route": b.route.tolist(),
           "has": b.has_character.tolist(), "gets": b.gets_bubble.tolist(),
           "fresh": b.additions[0].tolist(), "listed": b.additions[1].tolist(),
           "notices": [(j, rec.tolist()) for j, rec, _ in b.notices]}
    for j, s in enumerate(b.slots):
        okw = s.ok.tolist()
        rows = [r if o else None for r, o in zip(s.row.tolist(), okw)]
        trows = [r if o else None for r, o in zip(s.trow.tolist(), okw)]
        locs = [r if o else None for r, o in zip(s.loc.tolist(), okw)]
        out[f"slot{j}"] = (s.kind, okw, rows, trows, locs)
    return out


SERVER_STATE = ("stamp_for_system", "settled", "dead", "_bubble_time", "_fresh", "_listed", "pilot")


def kernel_against_torch(cs: list, snapshot: bool = True, block: int | None = None, fault: str | None = None,
                         ticks: int | None = None) -> tuple:
    """Both servers on their own VecCombatPark (CUDA), the same pushes every tick: ([(tick, what differs)], volumes).
    Compared every tick: every tensor of the two parks and adapters, the park's time, the servers' state, every field
    of the Batch, and (with snapshots) the decoded updates."""
    T = len(cs[0]["ticks"]) if ticks is None else ticks
    ts, index = build(cs, "cuda")
    ks, _ = build(cs, "cuda", kernel="cuda", block=block, fault=fault)
    bad, vol = [], {"ticks": 0, "worlds": len(cs), "queued": {}, "bubble_routes": 0, "character_routes": 0,
                    "decoded_updates": 0, "launches_on_a_ball": 0, "ball_ticks": 0, "in_MISSILE": 0,
                    "in_FORMATION": 0, "missiles_released": 0, "slots_freed": 0, "flagged_rows": 0}
    prev_mode = prev_slots = None
    for t in range(T):
        for kind, *data in pushes(cs, t, "cuda"):
            ts.push(kind, *data)
            ks.push(kind, *data)
        for srv in (ts, ks):
            srv.pre_tick(snapshot=snapshot)
            srv.evolve()
            srv.post_tick()
        for name in tensors_differing(ts.a.park, ks.a.park):
            bad.append((t, "park", name))
        for name in tensors_differing(ts.a, ks.a):
            bad.append((t, "adapter", name))
        if ts.a.park.current_time != ks.a.park.current_time:
            bad.append((t, "current_time"))
        for k in SERVER_STATE:
            if not _bits_equal(getattr(ts, k), getattr(ks, k)):
                bad.append((t, "server", k))
        va, vb = batch_view(ts), batch_view(ks)
        diff = [k for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)]
        if diff:
            bad.append((t, "batch", diff[:5]))
        if snapshot:
            da, db = VS.decode(ts), VS.decode(ks)
            if S.canon([S._listify(x) for x in da]) != S.canon([S._listify(x) for x in db]):
                bad.append((t, "decode"))
            vol["decoded_updates"] += sum(len(v) for per in da for v in per.values())
        # the volumes, from the torch path
        b, p = ts.batch, ts.a.park
        route = b.route.cpu()
        vol["bubble_routes"] += int((route == VS.BUBBLE).sum())
        vol["character_routes"] += int((route == VS.CHARACTER).sum())
        for s in b.slots:
            name = VS.KINDS[s.kind]
            vol["queued"][name] = vol["queued"].get(name, 0) + int(s.ok.sum())
            if s.kind == VS.LAUNCH:
                vol["launches_on_a_ball"] += int((s.ok & (s.row >= 0)).sum())
        ex = p.exists.cpu()
        mode, slots = p.mode.cpu(), p.formation_slots.cpu()
        vol["ball_ticks"] += int(ex.sum())
        vol["in_MISSILE"] += int((ex & (mode == 5)).sum())
        vol["in_FORMATION"] += int((ex & (mode == 12)).sum())
        if prev_mode is not None:
            vol["missiles_released"] += int((ex & (prev_mode == 5) & (mode != 5)).sum())
            vol["slots_freed"] += int(sum(bin(int(x)).count("1") for x in (prev_slots & ~slots).tolist()))
        prev_mode, prev_slots = mode, slots
        vol["ticks"] += 1
    vol["flagged_rows"] = int((ts.a.park.flags != 0).sum())
    return bad, vol


# ================================================================================ against the scalar server layer

def run_scalar(cs: list, kernel: str = "cuda", ticks: int | None = None) -> dict:
    """tests/server_combat_cases.run with the server built here (its `kernel`): every client's update list and every
    ball's state against the scalar server layer, tick by tick; the worlds are server_combat_cases' (no empty rows)."""
    T = len(cs[0]["ticks"]) if ticks is None else int(ticks)
    scal = [SC.ScalarWorld(w) for w in cs]
    server, index = build(cs, "cuda", kernel=kernel)
    bad_sends, bad_rows, sends, compared, reached = [], [], 0, 0, {}
    for t in range(T):
        want = [sw.tick(w["ticks"][t]) for sw, w in zip(scal, cs)]
        for kind, *data in pushes(cs, t, server.dev):
            server.push(kind, *data)
        server.pre_tick(snapshot=True)
        server.evolve()
        server.post_tick()
        got = VS.decode(server)
        p = server.a.park
        cpu = type("P", (), {k: v.cpu() for k, v in vars(p).items() if torch.is_tensor(v)})
        flagged = {wi for wi, f in enumerate(p.world_flags()) if f}
        for wi, w in enumerate(cs):
            if wi in flagged:
                continue
            if S.canon(S._listify(want[wi])) != S.canon(S._listify(got[wi])):
                bad_sends.append((t, w["id"], S._first_diff(want[wi], got[wi])))
            sends += sum(len(v) for v in want[wi].values())
            for per in want[wi].values():
                for _, _, st, _, _ in per:
                    for _, (name, _) in st:
                        reached[name] = reached.get(name, 0) + 1
            for b in scal[wi].park.lane.balls.values():
                r = index[(w["id"], b.id)]
                compared += 1
                mine, theirs = SC.ball_state(cpu, r), SC.scalar_ball_state(b)
                if mine != theirs:
                    bad_rows.append((t, w["id"], b.id, theirs, mine))
    return {"sends": bad_sends, "rows": bad_rows, "updates": sends, "compared": compared, "reached": reached,
            "flagged": [wi for wi, f in enumerate(server.a.park.world_flags()) if f], "ticks": T}


def bits(x: float) -> str:
    return "nan" if x != x else struct.pack("<d", float(x)).hex()

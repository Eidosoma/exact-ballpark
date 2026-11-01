# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Command streams for Destiny's server layer over the combat park (exact_ballpark.vec_server_combat),
generated from a seed, and the two runners compared bit for bit: the scalar server layer (tests/server_cases.py's
ScalarWorld: Actions, Ticker, BubbleUpdater, ParkUpdateBatcher on a master DestinyPark over FullBallpark, its
formations set up on the park before the first tick) and the batched layer (CombatServer over VecCombatPark with
master=True, its formations set up the same way: its torch tick on the CPU, its kernels on a CUDA device).

A world is one bubble near the origin: 3 to 5 ships (ship 1 interactive and flown by pilot 0, never cloaked: it keeps
the bubble alive; the others interactive or not, flown or not), 1 to 3 missile balls in the park from the load, at
rest; ship 2 leads a formation (FORMATIONS) with ships 3 and, in a world of 5, 4 following in reserved slots. The
special ship (the last ship) is the only one that can turn massive besides the missiles, a missile is launched massive
only at a ship, and missiles are not commanded otherwise (the park's domain: no two balls meet but a missile and its
target). Each tick 0 to 4 Actions calls: GotoPoint, GotoDirection, Stop (of a formation
follower too: on the master its slot is freed), the speed fraction, LaunchMissile (aimed or not, massive or not, at a
ship, at its launcher, at a missing ball, at a negative id: refused), the special ship's cloaks (releasing the
missiles chasing it and the formation's followers when it leads) and uncloaks, SetBallInteractive (not ship 1),
SetBallMassive (the special ship), duplicates; on a CUDA device also FollowBall and Orbit of a ship. Compared after
every tick: every client's update list (stamps, events, argument types, AddBallsToPark streams byte for byte with a
missile's and a formation follower's tails, PackagedActions, wait_for_bubble, counts) and every ball's position,
velocity, mode, massive flag, follow and owner ids, formation id and slots.
"""

from __future__ import annotations

import random
import struct

import torch

import server_cases as S
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_combat as VSC

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
FORMATIONS = [[[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0]], [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]]


def _bits(x: float) -> str:
    return "nan" if x != x else struct.pack("<d", float(x)).hex()


def _f(r: random.Random, lo: float, hi: float) -> float:
    return float(r.uniform(lo, hi))


def _world(r: random.Random, wid: int, device: str) -> dict:
    k = r.randint(3, 5)
    m = r.randint(1, 3)
    balls = []
    for i in range(1, k + 1):
        balls.append({"id": i, "mass": 10 ** r.uniform(6.0, 8.0), "radius": _f(r, 20.0, 200.0),
                      "max_vel": _f(r, 50.0, 400.0), "agility": _f(r, 0.2, 2.0), "sf": r.choice([1.0, 0.5, 0.0]),
                      "pos": [_f(r, -6e3, 6e3), _f(r, -6e3, 6e3), _f(r, -6e3, 6e3)],
                      "vel": r.choice([[0.0, 0.0, 0.0], [_f(r, -80.0, 80.0) for _ in range(3)]]),
                      "interactive": i == 1 or r.random() < 0.6, "massive": False, "max_ang_vel": 1.0,
                      "rot_agility": 0.9})
    for i in range(m):
        balls.append({"id": 101 + i, "mass": 10 ** r.uniform(3.0, 4.0), "radius": _f(r, 1.0, 5.0),
                      "max_vel": _f(r, 2000.0, 4000.0), "agility": _f(r, 0.05, 0.3), "sf": 1.0,
                      "pos": [_f(r, -6e3, 6e3), _f(r, -6e3, 6e3), _f(r, -6e3, 6e3)], "vel": [0.0, 0.0, 0.0],
                      "interactive": False, "massive": False, "max_ang_vel": 1.0, "rot_agility": 0.9})
    pilots, p = {1: 0}, 1
    for b in balls[1:k]:
        if b["interactive"] and r.random() < 0.7:
            pilots[b["id"]] = p
            p += 1
    form = {"leader": 2, "formation": r.randint(0, 1), "followers": [3] + ([4] if k == 5 else [])}
    return {"id": wid, "park": "combat", "do": False, "balls": balls, "ships": list(range(1, k + 1)),
            "missiles": [101 + i for i in range(m)], "special": k, "pilots": pilots, "formation": form, "ticks": [],
            "device": device}


def _call(r: random.Random, w: dict, device: str) -> tuple:
    ships, missiles, sp = w["ships"], w["missiles"], w["special"]
    s = r.choice(ships)                                     # (a missile commanded off MISSILE while massive would
                                                            # meet the special ship outside the park's domain)
    k = r.random()
    if k < 0.12:
        return "go_to_point", (s, _f(r, -8e3, 8e3), _f(r, -8e3, 8e3), _f(r, -8e3, 8e3))
    if k < 0.20:
        return "go_to_direction", (s, _f(r, -1.0, 1.0), _f(r, -1.0, 1.0), _f(r, -1.0, 1.0))
    if k < 0.28:
        return "stop", (r.choice(w["formation"]["followers"] + [s]),)
    if k < 0.32:
        return "set_speed_fraction", (s, r.choice([0.0, 1.0, _f(r, 0.0, 1.0)]))
    if k < 0.52:
        ready = [x for x in missiles if x not in w["launched"]]
        if ready:                                           # each missile launched once (a massive one relaunched
            launcher = r.choice(ships)                      # without a target would leave MISSILE massive)
            target = r.choice([x for x in ships if x != launcher] * 4 + [launcher, 4242, -3])
            massive = r.choice([0, 0, 1]) if target in ships and target not in (launcher, sp) else 0   # (a cloaked
            # target leaves a massive missile outside MISSILE: only the special ship cloaks)
            m = r.choice(ready)
            w["launched"].add(m)
            return "launch_missile", (m, target, launcher, r.choice([0, 1, 1]), massive)
    if k < 0.60:
        return "cloak_ball", (sp, r.choice([1, 1, 2, 0]), _f(r, 100.0, 6000.0))
    if k < 0.66:
        return "uncloak_ball", (sp,)
    if k < 0.72:
        return "set_ball_interactive", (r.choice(ships[1:]), r.choice([0, 1]))
    if k < 0.75:
        return "set_ball_massive", (sp, r.choice([0, 1]))
    if device == "cuda" and k < 0.85:
        o = r.choice([x for x in ships if x != s] or ships)
        return r.choice(["follow_ball", "orbit"]), (r.choice(ships), o, _f(r, 200.0, 3000.0))
    return "go_to_direction", (s, _f(r, -1.0, 1.0), _f(r, -1.0, 1.0), _f(r, -1.0, 1.0))


def cases(seed: int, n: int, device: str = "cpu", ticks: int = 12) -> list:
    r = random.Random(f"server-combat-{device}-{seed}")
    out = []
    for wid in range(n):
        w = _world(r, wid, device)
        w["launched"] = set()
        for _ in range(ticks):
            calls = []
            for _ in range(r.choice([0, 1, 1, 2, 3, 4])):
                c = _call(r, w, device)
                calls.append(c)
                if r.random() < 0.08:
                    calls.append((c[0], tuple(c[1])))       # the same call twice in a row: the second dropped
            w["ticks"].append(calls)
        w.pop("launched")
        out.append(w)
    return out


class ScalarWorld(S.ScalarWorld):
    """server_cases' ScalarWorld with the formation loaded and set on the master park before the first tick."""

    def __init__(self, w: dict, **kw):
        super().__init__(w, **kw)
        lane = self.park.lane
        f = w["formation"]
        lane.load_formations(FORMATIONS)
        lane.set_ball_formation(f["leader"], f["formation"])
        for fol in f["followers"]:
            lane.formation_follow(fol, f["leader"], lane.reserve_formation_slot(f["leader"]))


def build(cs: list, device: str) -> tuple:
    """(CombatServer, index {(world, ball id): row}) on one VecCombatPark (master), the formations set as the
    scalar side sets them."""
    from exact_ballpark.vec_combat import VecCombatPark
    rows = [(w, b) for w in cs for b in w["balls"]]
    p = VecCombatPark(len(rows), device=device, master=True)
    p.load([{"id": b["id"], "world": w["id"], "time": 0, "mass": b["mass"], "radius": b["radius"],
             "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
             "speed_fraction": b["sf"], "free": True, "massive": b["massive"], "interactive": b["interactive"],
             "cloaked": 0, "bubble": 0, "moribund": False, "max_ang_vel": b["max_ang_vel"],
             "rot_agility": b["rot_agility"]} for w, b in rows])
    index = {(w["id"], b["id"]): i for i, (w, b) in enumerate(rows)}
    p.load_formations(FORMATIONS)
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
    return VSC.CombatServer(a, pilots), index


def pushes(cs: list, t: int, device) -> list:
    """server_cases.pushes, LaunchMissile's arguments as CombatServer takes them (the target's id as `target`, the
    owner's as `n`, aimed and massive as f[0] and f[1])."""
    out = []
    for kind, mask, ball, target, f, n, loc in S.pushes(cs, t, device):
        if kind == "LaunchMissile":
            target = f[:, 0].to(I64)
            n = f[:, 1].to(I64)
            ff = torch.zeros_like(f)
            ff[:, 0], ff[:, 1] = f[:, 2], f[:, 3]
            f = ff
        out.append((kind, mask, ball, target, f, n, loc))
    return out


def ball_state(p, r: int) -> tuple:
    return (tuple(_bits(x) for x in p.pos[r].tolist()), tuple(_bits(x) for x in p.vel[r].tolist()), int(p.mode[r]),
            bool(p.massive[r]), int(p.follow_id[r]), int(p.owner_id[r]), int(p.formation_id[r]),
            int(p.formation_slots[r]))


def scalar_ball_state(b) -> tuple:
    return (tuple(_bits(x) for x in b.pos), tuple(_bits(x) for x in b.vel), int(b.mode), bool(b.is_massive),
            int(b.follow_id), int(b.owner_id), int(b.formation_id), int(b.formation_slots))


def run(cs: list, device: str, ticks: int | None = None) -> dict:
    """Both sides tick by tick: the client update lists and the balls that differ, the volumes, the kinds reached."""
    T = len(cs[0]["ticks"]) if ticks is None else int(ticks)
    scal = [ScalarWorld(w) for w in cs]
    server, index = build(cs, device)
    bad_sends, bad_rows, sends, compared, reached = [], [], 0, 0, {}
    prev = {}
    for t in range(T):
        want = [sw.tick(w["ticks"][t]) for sw, w in zip(scal, cs)]
        for kind, *data in pushes(cs, t, server.dev):
            server.push(kind, *data)
        server.pre_tick(snapshot=True)
        server.evolve()
        server.post_tick()
        got = VS.decode(server)
        p = server.a.park
        state = {k: v for k, v in vars(p).items() if torch.is_tensor(v)}
        cpu = type("P", (), {k: v.cpu() for k, v in state.items()})
        flagged = {wi for wi, f in enumerate(server.a.park.world_flags()) if f}
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
                mine, theirs = ball_state(cpu, r), scalar_ball_state(b)
                if mine != theirs:
                    bad_rows.append((t, w["id"], b.id, theirs, mine))
                if b.mode == 5:
                    reached["ticks_in_MISSILE"] = reached.get("ticks_in_MISSILE", 0) + 1
                if b.mode == 12:
                    reached["ticks_in_FORMATION"] = reached.get("ticks_in_FORMATION", 0) + 1
                was = prev.get((w["id"], b.id))
                if was is not None and was[0] == 5 and b.mode != 5:
                    reached["missiles_released"] = reached.get("missiles_released", 0) + 1
                if was is not None and (was[1] & ~int(b.formation_slots)):
                    reached["formation_slots_freed"] = reached.get("formation_slots_freed", 0) + 1
                prev[(w["id"], b.id)] = (b.mode, int(b.formation_slots))
    return {"sends": bad_sends, "rows": bad_rows, "updates": sends, "compared": compared, "reached": reached,
            "flagged": [wi for wi, f in enumerate(server.a.park.world_flags()) if f], "ticks": T}

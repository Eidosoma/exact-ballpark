# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the flight modes (FOLLOW, ORBIT, WARP), generated from a seed; run by the reference (exact_ballpark.modes)
and by Destiny's own code compiled verbatim (conformance/build_modes.py), and compared bit for bit.

A scenario is a ballpark (friction, tick, start time, orbit law, server or client view), a few balls and commands
applied before given ticks. After every tick every ball's state is recorded: mode, effect stamp, flags, follow fields,
position, velocity, goto point, behaviour acceleration, follow range, warp length and follower set; beside them the
events Destiny posts (DoModeChange, OnExitWarp, OnActivatingWarp, OnDeactivatingWarp, partition insertions).
"""

from __future__ import annotations

import math
import random
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

from exact_ballpark.modes import AU, BOID, FIELD, MINIBALL, RIGID, ModeBallpark
from exact_ballpark.ref import STOP

from kit import harness as kit_harness  # noqa: E402  (the conformance kit's paths: conformance/kit.toml)
HARNESS = kit_harness("modes")
MANIFEST = HARNESS.with_name("manifest.json")

GOTO_OP, DIR_OP, STOP_OP, FOLLOW_OP, ORBIT_OP, WARP_OP, WARPIN_OP = 1, 2, 3, 4, 5, 6, 7
NAN, INF = float("nan"), float("inf")


# ---- generation ----

def _unit(r: random.Random) -> list:
    while True:
        u = [r.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        if n > 1e-3:
            return [c / n for c in u]


def _ball(r: random.Random, bid: int, **kw) -> dict:
    """A random ship; keywords override."""
    b = {"id": bid, "mass": 10 ** r.uniform(5.5, 9.5), "radius": r.choice([0.0, r.uniform(1.0, 50.0),
                                                                          r.uniform(50.0, 500.0)]),
         "max_vel": r.choice([r.uniform(50.0, 1500.0), r.uniform(50.0, 1500.0), r.uniform(1.0, 199.0)]),
         "agility": r.uniform(0.05, 2.0), "sf": r.choice([1.0, r.uniform(0.1, 1.0)]),
         "pos": [r.uniform(-1e5, 1e5) for _ in range(3)],
         "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-300.0, 300.0) for _ in range(3)]]),
         "free": True, "massive": r.random() < 0.5, "interactive": r.random() < 0.5, "cloaked": 0, "bubble": 7,
         "moribund": False, "mode": STOP}
    b.update(kw)
    return b


def _cmd(step: int, op: int, a: int, b: int = 0, x: float = 0.0, y: float = 0.0, z: float = 0.0, rr: float = 0.0,
         w: int = 0) -> tuple:
    return (step, op, a, b, x, y, z, rr, w)


def _at(step: int, op: int, a: int, point, **kw) -> tuple:
    """A command with a point (or direction) argument."""
    return _cmd(step, op, a, x=point[0], y=point[1], z=point[2], **kw)


def _range(r: random.Random) -> float:
    return r.choice([0.0, 1.0, r.uniform(0.0, 500.0), r.uniform(0.0, 5000.0), r.uniform(500.0, 30000.0),
                     -r.uniform(0.0, 300.0)])


def _r_of(rng: float, b1: dict, b2: dict) -> float:
    """Destiny's r = (double)followRange + (double)radius + (double)other radius, the floats rounded first."""
    f = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    return f(rng) + f(b1["radius"]) + f(b2["radius"])


def _orbit_range(r: random.Random, b1: dict, b2: dict) -> float:
    """A follow range for an orbiter, never one making r exactly 0: an old-style orbiter on its target's centre with
    r = 0 divides 0 by 0 and then converts the NaN to an integer, which C leaves undefined (Ballpark.cpp:1290, 1301)."""
    rng = _range(r)
    return 1.0 if _r_of(rng, b1, b2) == 0.0 else rng


def _near(r: random.Random, centre: list, rr: float) -> list:
    """A start point relative to a target: inside, at, outside or far outside the range, or on its centre."""
    kind = r.choice(["inside", "at", "outside", "far", "on", "axis"])
    if kind == "on":
        return list(centre)
    if kind == "axis":                                    # exactly at r along +x: dist == r to the last bit
        return [centre[0] + abs(rr), centre[1], centre[2]]
    d = {"inside": r.uniform(0.0, 1.0) * abs(rr), "at": abs(rr) * r.uniform(0.98, 1.02),
         "outside": abs(rr) * r.uniform(1.0, 3.0) + r.uniform(0.0, 2000.0), "far": r.uniform(1e4, 2e5)}[kind]
    u = _unit(r)
    return [centre[i] + d * u[i] for i in range(3)]


def _ids(r: random.Random, n: int, big: bool) -> list:
    ids: set = set()
    while len(ids) < n:
        ids.add(r.randrange(1, 2 ** 40) if big else r.randrange(1, 1000))
    return sorted(ids) if r.random() < 0.5 else list(ids)


def _warp_steps(w: int, dist: float, tick: int, align: int) -> int:
    """A rough count of ticks to finish a warp (the generator's estimate, never the reference)."""
    s = w * 0.001 * AU
    acc, dec = w / 1000.0, min(w / 3000.0, 2.0)
    s = min(s, (dist + 1) * acc * dec / (acc + dec))
    t = math.log(max(s / acc, 1.0)) / acc + max(0.0, dist / s - 1 / acc - 1 / dec) + math.log(max(s / dec, 1.0)) / dec
    return align + int(t / (tick * 0.001)) + 4


def cases(seed: int) -> list[dict]:
    r = random.Random(seed)
    out: list[dict] = []

    def case(kind: str, balls: list, cmds: list, steps: int, *, tick: int = 1000, new_orbit: bool = False,
             friction: float = 1.0e6, master: bool = True, start: int | None = None) -> None:
        out.append({"id": len(out) + 1, "kind": kind, "friction": friction, "tick": tick, "steps": steps,
                    "new_orbit": new_orbit, "master": master,
                    "start": r.randrange(0, 2_000_000) if start is None else start, "balls": balls, "cmds": cmds})

    def friction_choice() -> float:                       # mostly Destiny's default; low (the Taylor branch) or high
        return r.choice([1.0e6] * 6 + [10 ** r.uniform(-2.0, 0.0), 10 ** r.uniform(3.0, 7.0)])

    def tick_choice(long_ok: bool = False) -> int:
        ticks = [250, 500, 1000, 1000, 1000, 1000, 2000]
        return r.choice(ticks + ([5000, 10000, 20000] if long_ok and r.random() < 0.3 else []))

    # FOLLOW: stationary, drifting, moving, fixed or following targets; one to three followers (MOV-714, MOV-715)
    for _ in range(160):
        ids = _ids(r, r.randint(2, 4), big=False)
        target = _ball(r, ids[0], free=r.random() < 0.85)
        balls, cmds = [target], []
        if target["free"] and r.random() < 0.5:
            cmds.append(_cmd(1, GOTO_OP, target["id"], x=r.uniform(-2e5, 2e5), y=r.uniform(-2e5, 2e5),
                             z=r.uniform(-2e5, 2e5)))
        for bid in ids[1:]:
            rng = _range(r)
            f = _ball(r, bid)
            f["pos"] = _near(r, target["pos"], _r_of(rng, f, target))
            balls.append(f)
            cmds.append(_cmd(r.choice([1, 1, 1, 2, 5]), FOLLOW_OP, bid, target["id"], rr=rng))
        if len(ids) > 2 and r.random() < 0.3:                # a chain: the last follower is followed in turn
            cmds.append(_cmd(1, FOLLOW_OP, ids[1], ids[-1], rr=_range(r)))
        case("follow", balls, cmds, r.randint(1, 60), tick=tick_choice(), master=r.random() < 0.9,
             friction=friction_choice())

    # ORBIT, both laws: near, at, far and on-centre starts, negative ranges, slow and stopped ships (MOV-716 to 722)
    for i in range(320):
        new = i % 2 == 1
        ids = _ids(r, r.randint(2, 4), big=True)
        target = _ball(r, ids[0], free=r.random() < 0.85, vel=r.choice([[0.0, 0.0, 0.0],
                                                                       [r.uniform(-50, 50) for _ in range(3)]]))
        balls, cmds = [target], []
        if target["free"] and r.random() < 0.4:
            cmds.append(_at(1, DIR_OP, target["id"], _unit(r)))
        for bid in ids[1:]:
            o = _ball(r, bid)
            rng = _orbit_range(r, o, target)
            if r.random() < 0.15:
                o["sf"] = 0.0                                # zero cruise speed (OrbitThrust off, no acceleration)
            if r.random() < 0.3:
                o["mass"], o["agility"] = 10 ** r.uniform(8.5, 9.5), r.uniform(1.0, 2.0)   # heavy: OrbitThrust
            o["pos"] = _near(r, target["pos"], _r_of(rng, o, target))
            balls.append(o)
            cmds.append(_cmd(r.choice([1, 1, 1, 3]), ORBIT_OP, bid, target["id"], rr=rng))
        case("orbit_new" if new else "orbit_old", balls, cmds, r.randint(1, 80), tick=tick_choice(long_ok=new),
             new_orbit=new, master=r.random() < 0.9, friction=friction_choice())

    # the new orbit's steepest spiral (factor < -0.999): negative ranges close in, or ticks of 10 s and more (MOV-720)
    for _ in range(48):
        ids = _ids(r, 2, big=True)
        target = _ball(r, ids[0], vel=[0.0, 0.0, 0.0])
        o = _ball(r, ids[1], mass=10 ** r.uniform(5.5, 7.5), sf=1.0, max_vel=r.uniform(100.0, 1500.0))
        long_tick = r.random() < 0.4
        rng = r.uniform(0.0, 200.0) if long_tick else -r.uniform(1000.0, 20000.0)
        u = _unit(r)
        d = r.uniform(2000.0, 20000.0) if long_tick else r.uniform(1.0, 200.0)
        o["pos"] = [target["pos"][i] + d * u[i] for i in range(3)]
        case("orbit_new", [target, o], [_cmd(1, ORBIT_OP, ids[1], ids[0], rr=rng)], r.randint(1, 30),
             tick=r.choice([10000, 20000]) if long_tick else 1000, new_orbit=True)

    # WARP: short and long, warp factors, minimum ranges, aligned and unaligned starts, slow ships (TRV-701 to 714)
    for _ in range(220):
        bid = r.randrange(1, 1000)
        ship = _ball(r, bid, interactive=r.random() < 0.5)
        tick = tick_choice(long_ok=True)
        kind = r.choice(["long", "long", "very_long", "medium", "medium", "short", "slow", "still"])
        dist = {"long": 10 ** r.uniform(9.0, 12.5), "very_long": 10 ** r.uniform(12.0, 13.3),
                "medium": 10 ** r.uniform(5.0, 9.0),
                "short": r.uniform(0.0, 99999.0), "slow": 10 ** r.uniform(6.0, 10.0),
                "still": 10 ** r.uniform(6.0, 10.0)}[kind]
        u = _unit(r)
        dest = [ship["pos"][i] + dist * u[i] for i in range(3)]
        w = r.choice([r.randint(1500, 6000), r.randint(1500, 20000), 3000, 10000])
        if kind == "very_long":                              # slower warps: a cruise phase of many ticks
            w = r.randint(1500, 4000)
        min_range = r.choice([0.0, 20000.0, r.uniform(0.0, 1e6), -r.uniform(1.0, 100.0), dist * r.uniform(1.0, 1.5)])
        align = 2
        if kind == "slow":                                   # below 75% cruise: the 180-tick forced start
            ship["sf"] = r.uniform(0.05, 0.7)
            align = 182
        elif kind == "still":                                # no thrust at all: forced start, then the overrun
            ship["max_vel"], ship["vel"] = 0.0, [0.0, 0.0, 0.0]
            align = 182
        elif r.random() < 0.5:                               # already aligned and fast: activates at once
            ship["vel"] = [u[i] * ship["max_vel"] * r.uniform(0.76, 1.0) for i in range(3)]
        else:
            align = 60
        steps = min(_warp_steps(w, max(dist - max(min_range, 0.0), 1.0), tick, align), 420)
        balls = [ship]
        if r.random() < 0.3:                                 # a bystander in STOP
            balls.append(_ball(r, bid + 1000))
        case("warp", balls, [_cmd(r.choice([1, 1, 2]), WARP_OP, bid, x=dest[0], y=dest[1], z=dest[2], rr=min_range,
                                  w=w)], max(steps, 3), tick=tick, master=r.random() < 0.8)

    # followers of a warping ball, with ids below and above it, released at activation (MOV-730)
    for _ in range(100):
        ids = _ids(r, r.randint(2, 5), big=True)
        wid = r.choice(ids)
        warper = _ball(r, wid, mass=10 ** r.uniform(5.5, 8.0), sf=1.0)
        u = _unit(r)
        warper["vel"] = [u[i] * warper["max_vel"] * r.uniform(0.6, 1.0) for i in range(3)]
        dest = [warper["pos"][i] + 10 ** r.uniform(8, 11) * u[i] for i in range(3)]
        balls, cmds = [warper], []
        new = r.random() < 0.5
        for bid in ids:
            if bid == wid:
                continue
            f = _ball(r, bid, interactive=r.random() < 0.6)
            rng = _orbit_range(r, f, warper)
            f["pos"] = _near(r, warper["pos"], _r_of(rng, f, warper))
            balls.append(f)
            cmds.append(_cmd(1, r.choice([FOLLOW_OP, ORBIT_OP, ORBIT_OP]), bid, wid, rr=rng))
        cmds.append(_cmd(r.choice([1, 2, 4]), WARP_OP, wid, x=dest[0], y=dest[1], z=dest[2],
                         rr=r.choice([0.0, 20000.0]), w=r.randint(1500, 20000)))
        case("warp_followers", balls, cmds, r.randint(5, 45), tick=tick_choice(), new_orbit=new,
             master=r.random() < 0.85)

    # commands during alignment and during warp proper (TRV-702, TRV-703, TRV-713)
    for _ in range(140):
        ids = _ids(r, 2, big=False)
        bid, other = ids
        ship = _ball(r, bid)
        u = _unit(r)
        if r.random() < 0.5:
            ship["sf"] = 1.0
            ship["vel"] = [u[i] * ship["max_vel"] * r.uniform(0.8, 1.0) for i in range(3)]
        else:
            ship["vel"] = [u[i] * ship["max_vel"] * r.uniform(0.0, 1.0) for i in range(3)]
        dist = 10 ** r.uniform(8, 11)
        dest = [ship["pos"][i] + dist * u[i] for i in range(3)]
        w = r.randint(1500, 20000)
        min_range = r.choice([0.0, 0.0, 20000.0, r.uniform(0.0, 1e5)])
        partner = _ball(r, other, pos=[ship["pos"][i] + r.uniform(-5e3, 5e3) for i in range(3)])
        cmds = [_cmd(1, WARP_OP, bid, x=dest[0], y=dest[1], z=dest[2], rr=min_range, w=w)]
        k = r.randint(2, 20)
        what = r.choice(["repeat", "repeat", "new", "stop", "goto", "follow", "orbit", "short", "warpin", "partner"])
        if what == "repeat":                                  # the same point: a no-op unless the exit point moved
            cmds.append(_cmd(k, WARP_OP, bid, x=dest[0], y=dest[1], z=dest[2], rr=r.choice([min_range, 5000.0]),
                             w=r.choice([w, w + 1])))
        elif what == "new":
            v = _unit(r)
            cmds.append(_at(k, WARP_OP, bid, [ship["pos"][i] + 10 ** r.uniform(8, 11) * v[i] for i in range(3)],
                             rr=0.0, w=r.randint(1500, 20000)))
        elif what == "stop":
            cmds.append(_cmd(k, STOP_OP, bid))
        elif what == "goto":
            cmds.append(_at(k, GOTO_OP, bid, dest))
        elif what in ("follow", "orbit"):
            cmds.append(_cmd(k, FOLLOW_OP if what == "follow" else ORBIT_OP, bid, other, rr=_range(r)))
        elif what == "short":
            cmds.append(_at(k, WARP_OP, bid, [ship["pos"][i] + 5e4 * u[i] for i in range(3)], rr=0.0, w=w))
        elif what == "warpin":
            cmds.append(_at(k, WARPIN_OP, bid, dest, w=w))
        else:                                                 # the partner follows or orbits the warper first
            cmds.insert(0, _cmd(1, r.choice([FOLLOW_OP, ORBIT_OP]), other, bid, rr=_range(r)))
        case("warp_commands", [ship, partner], cmds, k + r.randint(1, 60), tick=tick_choice(),
             new_orbit=r.random() < 0.5, master=r.random() < 0.8)

    # EntityWarpIn: far, at the origin, too close, early in the park's life (TRV-715)
    for _ in range(60):
        bid = r.randrange(1, 1000)
        ship = _ball(r, bid, pos=r.choice([[0.0, 0.0, 0.0], [r.uniform(-1e7, 1e7) for _ in range(3)]]))
        kind = r.choice(["far", "far", "origin", "close"])
        dest = {"far": [r.uniform(-3e11, 3e11) for _ in range(3)], "origin": [0.0, 0.0, 0.0],
                "close": [ship["pos"][i] + r.uniform(-5e4, 5e4) for i in range(3)]}[kind]
        if kind == "origin" and math.dist(ship["pos"], dest) < 1e5:
            ship["pos"] = [2e6, 0.0, 0.0]
        case("warpin", [ship], [_at(1, WARPIN_OP, bid, dest, w=r.randint(1, 20000))], r.randint(1, 40),
             tick=tick_choice(), start=r.choice([0, 3, 5, 7, r.randrange(0, 100000)]), master=r.random() < 0.7)

    # guards: non-finite ranges and points, self, missing, moribund, cloaked and other-bubble targets (MOV-715, TRV-701)
    for _ in range(80):
        ids = _ids(r, 3, big=False)
        balls = [_ball(r, bid) for bid in ids]
        t = balls[1]
        flavour = r.choice(["nan", "inf", "self", "missing_src", "missing_dst", "moribund", "cloaked", "bubble",
                            "warp_nan", "warp_missing", "warp_factor", "fixed"])
        if flavour == "moribund":
            t["moribund"] = True
        elif flavour == "cloaked":
            t["cloaked"] = r.choice([1, 2])
        elif flavour == "bubble":
            t["bubble"] = 9
        elif flavour == "fixed":
            t["free"] = False
        op = r.choice([FOLLOW_OP, ORBIT_OP])
        src, dst, rng = ids[0], ids[1], _range(r)
        if flavour == "nan":
            rng = NAN
        elif flavour == "inf":
            rng = r.choice([INF, -INF])
        elif flavour == "self":
            dst = src
        elif flavour == "missing_src":
            src = 5000
        elif flavour == "missing_dst":
            dst = 5000
        cmds = [_cmd(1, op, src, dst, rr=rng)]
        if flavour == "warp_nan":
            cmds = [_cmd(1, WARP_OP, src, x=NAN, y=1e9, z=0.0, rr=0.0, w=3000)]
        elif flavour == "warp_missing":
            cmds = [_cmd(1, WARP_OP, 5000, x=1e9, y=1e9, z=0.0, rr=0.0, w=3000)]
        elif flavour == "warp_factor":                        # clamped to 1: a slow warp, the first ticks only
            cmds = [_cmd(1, WARP_OP, src, x=1e9, y=0.0, z=0.0, rr=r.choice([0.0, -5.0]), w=r.choice([0, -3]))]
        cmds.append(_cmd(2, r.choice([FOLLOW_OP, ORBIT_OP]), ids[2], src, rr=_range(r)))
        case("guards", balls, cmds, r.randint(1, 6), new_orbit=r.random() < 0.5, master=r.random() < 0.8)

    # a soup: several balls, random commands at random ticks, both laws; inert modes now and then (MOV-734)
    for _ in range(140):
        n = r.randint(3, 6)
        ids = _ids(r, n, big=r.random() < 0.5)
        balls = [_ball(r, bid, mode=r.choice([STOP] * 8 + [BOID, MINIBALL, FIELD, RIGID]),
                       pos=[r.uniform(-2e4, 2e4) for _ in range(3)], free=r.random() < 0.9) for bid in ids]
        steps = r.randint(10, 90)
        cmds = []
        for _ in range(r.randint(1, 8)):
            k = r.randint(1, steps)
            a, b = r.choice(ids), r.choice(ids)
            op = r.choice([GOTO_OP, DIR_OP, STOP_OP, FOLLOW_OP, FOLLOW_OP, ORBIT_OP, ORBIT_OP, WARP_OP])
            if op in (GOTO_OP, DIR_OP):
                cmds.append(_cmd(k, op, a, x=r.uniform(-3e4, 3e4), y=r.uniform(-3e4, 3e4), z=r.uniform(-3e4, 3e4)))
            elif op == STOP_OP:
                cmds.append(_cmd(k, op, a))
            elif op == WARP_OP:
                v = _unit(r)
                d = r.choice([5e4, 10 ** r.uniform(5.5, 9)])
                cmds.append(_cmd(k, op, a, x=v[0] * d, y=v[1] * d, z=v[2] * d, rr=r.choice([0.0, 20000.0]),
                                 w=r.randint(2000, 20000)))
            else:
                cmds.append(_cmd(k, op, a, b, rr=_range(r)))
        case("mixed", balls, sorted(cmds, key=lambda c: c[0]), steps, tick=tick_choice(), new_orbit=r.random() < 0.5,
             master=r.random() < 0.85, friction=friction_choice())
    return out


# ---- running ----

def serialize(cs: list[dict]) -> str:
    lines = []
    for c in cs:
        lines.append(f"C {c['id']} {c['friction']!r} {c['tick']} {c['steps']} {int(c['new_orbit'])} {int(c['master'])} "
                     f"{c['start']} {len(c['balls'])} {len(c['cmds'])}")
        for b in c["balls"]:
            lines.append("B " + " ".join(repr(v) for v in (b["id"], float(b["mass"]), float(b["radius"]),
                                                            float(b["max_vel"]), float(b["agility"]), float(b["sf"]),
                                                            *map(float, b["pos"]), *map(float, b["vel"])))
                         + f" {int(b['free'])} {int(b['massive'])} {int(b['interactive'])} {b['cloaked']} {b['bubble']}"
                           f" {int(b['moribund'])} {b['mode']}")
        for (step, op, a, bb, x, y, z, rr, w) in c["cmds"]:
            lines.append(f"K {step} {op} {a} {bb} {float(x)!r} {float(y)!r} {float(z)!r} {float(rr)!r} {w}")
    return "\n".join(lines) + "\n"


def oracle(cs: list[dict]) -> tuple[list, list]:
    """Destiny's compiled code: (state rows, events)."""
    res = subprocess.run([str(HARNESS)], input=serialize(cs), capture_output=True, text=True, check=True, timeout=240)
    rows, events = [], []
    for line in res.stdout.splitlines():
        p = line.split()
        if p[0] == "S":
            nf = int(p[25])
            rows.append((*map(int, p[1:11]), *map(float, p[11:25]), tuple(int(x) for x in p[26:26 + nf])))
        elif p[0] == "E":
            events.append((int(p[1]), int(p[2]), p[3], int(p[4]), int(p[5]), int(p[6])))
    return rows, events


def reference(c: dict, trace=None) -> tuple[list, list]:
    """The reference on one scenario: (state rows, events), in the oracle's layout."""
    park = ModeBallpark(friction=c["friction"], tick_interval_ms=c["tick"], use_new_orbit=c["new_orbit"],
                        is_master=c["master"])
    park.current_time = c["start"]
    park.trace = trace
    for b in c["balls"]:
        park.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], b["pos"], b["vel"], b["agility"], b["sf"],
                      is_free=b["free"], is_massive=b["massive"], is_interactive=b["interactive"],
                      is_cloaked=b["cloaked"], bubble=b["bubble"], is_moribund=b["moribund"], mode=b["mode"])
    by_step = defaultdict(list)
    for k in c["cmds"]:
        by_step[k[0]].append(k)
    rows = []
    for s in range(1, c["steps"] + 1):
        for (_, op, a, bb, x, y, z, rr, w) in by_step.get(s, ()):
            if op == GOTO_OP:
                park.goto_point(a, x, y, z)
            elif op == DIR_OP:
                park.goto_direction(a, x, y, z)
            elif op == STOP_OP:
                park.stop(a)
            elif op == FOLLOW_OP:
                park.follow_ball(a, bb, rr)
            elif op == ORBIT_OP:
                park.orbit(a, bb, rr)
            elif op == WARP_OP:
                park.warp_to(a, x, y, z, rr, w)
            elif op == WARPIN_OP:
                park.entity_warp_in(a, x, y, z, w)
        park.evolve()
        for bid in sorted(park.balls):
            m = park.balls[bid]
            rows.append((c["id"], s, bid, m.mode, m.effect_stamp, int(m.is_massive), int(m.sensor_active),
                         m.follow_id, m.owner_id, -1 if m.follow_ptr is None else m.follow_ptr, *m.pos, *m.vel,
                         *m.goto, *m.last_g, m.follow_range, m.last_collision, tuple(sorted(m.followers))))
    return rows, [(c["id"], *e) for e in park.events]


def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def same(a, b) -> bool:
    """Equal integers, or doubles with equal bits (two NaNs count as equal)."""
    if isinstance(a, float) or isinstance(b, float):
        return bits(a) == bits(b) or (a != a and b != b)
    return a == b


def per_ball(events: list) -> dict:
    """Events grouped by (case, ball), each group in order: across followers released in one call Destiny's order
    is an unordered_set's, so only each ball's own sequence is defined."""
    out = defaultdict(list)
    for (cid, t, name, bid, a, b) in events:
        out[(cid, bid)].append((t, name, a, b))
    return dict(out)


FIELDS = ("case", "step", "id", "mode", "effect_stamp", "massive", "sensor", "follow_id", "owner_id", "follow_ptr",
          "px", "py", "pz", "vx", "vy", "vz", "gx", "gy", "gz", "ax", "ay", "az", "follow_range", "last_collision",
          "followers")

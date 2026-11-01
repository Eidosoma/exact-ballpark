# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Scenarios for the full ballpark (missiles, static collidables, cloaking, orientation dynamical or not, mushrooms, trolls,
formations, the accuracy helpers, OnTick), generated from a seed; run by the reference (exact_ballpark.full) and by
Destiny's own code compiled verbatim (conformance/build_full.py), and compared bit for bit.

A scenario is a ballpark (friction, tick, start time, orbit law, server or client view, the iterative collision
method, dynamical orientation and its missile exclusion, the rand() seed, and whether ticks come from OnTick) and a
list of records applied at given steps: balls added with the Python AddBall, commands, setters and queries. Step 0 is
the setup; after the records of each later step the park evolves one tick, unless the scenario drives its own ticks
(OnTick or Evolve records). After every step everything the slices decide is recorded: each ball's state (slice 2's
fields, with the orientation, rotation, angular velocity, torque, roll, formation, time stamps and minis), each static
collidable (its bubbles, boxes and position), the partition (a checksum per box, with its static collidables, and the
count per level), the proximity balls in Destiny's order, the park's clock, local id counters and the sizes of its
moribund sets and static maps, the bubble dictionaries, and every query's answer; the events Destiny posts beside
them. At the end every box is dumped. Until 23 October 2026 a scenario that needed DirectXMath (carbonengine/math's
RotationQuaternion(yaw, pitch, roll) away from zero) stopped with an "X" row in both; since then the oracle compiles
DirectXMath's scalar path and the reference transcribes it, so no scenario stops. Only HARNESS and
these docstrings changed then: the generator is the one frozen on 16 October.
"""

from __future__ import annotations

import math
import random
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

from exact_ballpark.full import DestinyUnavailable, FullBallpark
from exact_ballpark.ref import f32
from exact_ballpark.space import LEVEL_WIDTH, DestinyUndefined

from kit import harness as kit_harness, path as kit_path  # noqa: E402  (conformance/kit.toml)
HARNESS = kit_harness("full")   # DirectXMath
MANIFEST = HARNESS.with_name("manifest.json")
DXM_PROBE = HARNESS.with_name("dxm_probe")                  # RotationQuaternion(yaw, pitch, roll) alone
DXM_MANIFEST = kit_path("directxmath_manifest")

(GOTO_OP, DIR_OP, STOP_OP, FOLLOW_OP, ORBIT_OP, WARP_OP, WARPIN_OP, POS_OP, VEL_OP, RADIUS_OP, MAXVEL_OP, MASS_OP,
 AGILITY_OP, SF_OP, FREE_OP, MASSIVE_OP, GLOBAL_OP, INTERACTIVE_OP, SENSOR_OP, UNSENSOR_OP, NOTIFY_OP, TRACK_OP,
 HARMONIC_OP, CLOAK_ATTR_OP, INIT_BUBBLES_OP, COPY_BUBBLES_OP, CLEAN_FOLLOWERS_OP, KEEPALIVE_OP, DEADS_OP,
 REMOVE_OP) = range(1, 31)
VIS_Q, RANGE_BALL_Q, RANGE_POINT_Q, RANGE_DIST_Q, BUBBLE_AT_Q = 31, 32, 33, 34, 35
(MISSILE_FOLLOW_OP, LAUNCH_OP, CLOAK_OP, UNCLOAK_OP, CAPSULE_OP, OBOX_OP, MINIBALL_OP, MINICAPSULE_OP, MINIBOX_OP,
 REMOVE_CAPSULE_OP, REMOVE_OBOX_OP, ROTATION_OP, ANGVEL_OP, MAX_ANGVEL_OP, ROT_AGILITY_OP, TROLL_OP, RIGID_OP,
 BOID_OP, FORMATIONS_OP, SET_FORMATION_OP, FORMATION_FOLLOW_OP, RESERVE_SLOT_Q, FREE_SLOT_OP, MUSHROOM_OP,
 ACCURACY_Q, RELVEL_Q, ONTICK_OP, SHIFTED_Q, ROTATED_Q, EVOLVE_OP, DIST_Q) = range(36, 67)
TOP = LEVEL_WIDTH[0]

S_INT = ("case", "step", "id", "mode", "effect_stamp", "massive", "sensor", "follow_id", "owner_id", "follow_ptr",
         "free", "in_free_balls", "global", "interactive", "cloaked", "moribund", "junk", "bubble", "old_bubble",
         "has_proximity", "within_range", "within_tracking", "tracking_ptr", "in_dead_bubble", "mod_x", "mod_y",
         "mod_z", "active_boxes", "box_level", "box_ix", "box_iy", "box_iz", "harmonic", "corporation", "alliance",
         "only_interactives", "formation_id", "formation_slots", "new_time", "old_time")
S_DBL = ("px", "py", "pz", "vx", "vy", "vz", "opx", "opy", "opz", "ovx", "ovy", "ovz", "gx", "gy", "gz", "ax", "ay",
         "az", "cx", "cy", "cz", "mass", "radius", "max_vel", "agility", "speed_fraction", "follow_range",
         "last_collision", "time_factor", "notification_range", "tracking_range", "sensor_range", "sensor_period",
         "sensor_elapsed", "yaw", "old_yaw", "pitch", "old_pitch", "roll", "old_roll", "roll_speed", "old_roll_speed",
         "yaw_delta", "max_ang_vel", "rot_roll", "rot_agility", "rx", "ry", "rz", "rw", "orx", "ory", "orz", "orw",
         "wx", "wy", "wz", "owx", "owy", "owz", "tx", "ty", "tz")
S_LISTS = ("boxes", "collisions", "followers", "sensor_balls", "mini_balls", "mini_capsules", "mini_boxes")
S_FIELDS = S_INT + S_DBL + S_LISTS
T_INT = ("case", "step", "kind", "id", "parent", "moribund", "effect_stamp", "bubble", "old_bubble", "interactive",
         "handled", "box_level", "box_ix", "box_iy", "box_iz")
T_DBL = ("px", "py", "pz", "opx", "opy", "opz", "bounding_radius")
T_FIELDS = T_INT + T_DBL + ("boxes",)


# ---- generation helpers ----

def _unit(r: random.Random) -> list:
    while True:
        u = [r.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        if n > 1e-3:
            return [c / n for c in u]


def _ids(r: random.Random, n: int, low: int = 1) -> list:
    ids: set = set()
    while len(ids) < n:
        ids.add(r.choice([r.randrange(low, 40), r.randrange(low, 1000), r.randrange(low, 2 ** 40)]))
    out = sorted(ids)
    r.shuffle(out)                                          # insertion order differs from id order
    return out


def _ball(r: random.Random, bid: int, step: int = 0, **kw) -> tuple:
    """A Python AddBall record: a random ship; keywords override."""
    b = {"id": bid, "mass": 10 ** r.uniform(5.5, 9.0),
         "radius": r.choice([r.uniform(1.0, 50.0), r.uniform(50.0, 300.0)]),
         "max_vel": r.choice([r.uniform(50.0, 800.0), r.uniform(5.0, 199.0)]),
         "free": True, "global": False, "massive": r.random() < 0.7, "interactive": r.random() < 0.6,
         "junk": False, "pos": [r.uniform(-5e3, 5e3) for _ in range(3)],
         "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-200.0, 200.0) for _ in range(3)]]),
         "agility": r.uniform(0.05, 2.0), "sf": r.choice([1.0, r.uniform(0.1, 1.0)]),
         "max_ang_vel": r.choice([1.0, r.uniform(0.05, 3.0)]), "rot_agility": r.choice([0.9, r.uniform(0.05, 2.0)])}
    b.update(kw)
    return ("B", step, b)


def _k(step: int, op: int, a: int = 0, b: int = 0, c: int = 0, w: int = 0, d=()) -> tuple:
    return ("K", step, op, int(a), int(b), int(c), int(w), tuple(float(x) for x in d))


def _xyz(x=0.0, y=0.0, z=0.0, rr=0.0) -> tuple:
    return (x, y, z, rr)


def _test_ship(bid: int, step: int = 0, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass: float = 13000000.0,
               max_vel: float = 10.0, radius: float = 2.0) -> list:
    """helpers.create_space_ball as records: AddBall with the POPO defaults, then the setters in the helper's order."""
    return [("B", step, {"id": bid, "mass": 0.0, "radius": 0.0, "max_vel": 0.0, "free": False, "global": False,
                         "massive": False, "interactive": False, "junk": False, "pos": list(pos), "vel": list(vel),
                         "agility": 0.0, "sf": 0.0, "max_ang_vel": 1.0, "rot_agility": 0.9}),
            _k(step, FREE_OP, bid, w=1), _k(step, MAXVEL_OP, bid, d=_xyz(rr=max_vel)),
            _k(step, AGILITY_OP, bid, d=_xyz(rr=0.9)), _k(step, MASS_OP, bid, d=_xyz(rr=mass)),
            _k(step, RADIUS_OP, bid, d=_xyz(rr=radius)), _k(step, MASSIVE_OP, bid, w=1),
            _k(step, SF_OP, bid, d=_xyz(rr=0.95))]


def _parent(bid: int, step: int = 0, pos=(0.0, 0.0, 0.0), radius: float = 500.0, mass: float = 1e9,
            free: bool = False) -> tuple:
    """helpers.add_ball_to_park: a fixed, non-massive parent for minis."""
    return ("B", step, {"id": bid, "mass": mass, "radius": radius, "max_vel": 0.0, "free": free, "global": False,
                        "massive": False, "interactive": False, "junk": False, "pos": list(pos), "vel": [0.0] * 3,
                        "agility": 0.0, "sf": 0.0, "max_ang_vel": 1.0, "rot_agility": 0.9})


def case(out: list, r: random.Random, kind: str, records: list, steps: int, *, master: bool = True, tick: int = 1000,
         friction: float = 1.0e6, new_orbit: bool = False, iterative: bool = False, do: bool = False,
         do_missiles_off: bool = False, manual: bool = False, start: int | None = None) -> None:
    out.append({"id": len(out) + 1, "kind": kind, "friction": friction, "tick": tick, "steps": steps,
                "new_orbit": new_orbit, "iterative": iterative, "do": do, "do_missiles_off": do_missiles_off,
                "manual": manual, "master": master, "seed": r.randrange(1, 2 ** 31),
                "start": r.randrange(0, 100000) if start is None else start,
                "records": sorted(records, key=lambda rec: rec[1])})


# ---- running ----

def serialize(cs: list) -> str:
    lines = []
    for c in cs:
        flags = (int(c["new_orbit"]) | (2 if c.get("iterative") else 0) | (4 if c.get("do") else 0)
                 | (8 if c.get("do_missiles_off") else 0) | (16 if c.get("manual") else 0))
        lines.append(f"C {c['id']} {c['friction']!r} {c['tick']} {c['steps']} {flags} {int(c['master'])} "
                     f"{c['start']} {c['seed']} {len(c['records'])}")
        for rec in c["records"]:
            if rec[0] == "B":
                b = rec[2]
                lines.append(f"B {rec[1]} {b['id']} "
                             + " ".join(repr(float(b[k])) for k in ("mass", "radius", "max_vel"))
                             + f" {int(b['free'])} {int(b['global'])} {int(b['massive'])} {int(b['interactive'])}"
                               f" {int(b['junk'])} " + " ".join(repr(float(v)) for v in (*b["pos"], *b["vel"]))
                             + f" {float(b['agility'])!r} {float(b['sf'])!r} {float(b['max_ang_vel'])!r}"
                               f" {float(b['rot_agility'])!r}")
            else:
                _, step, op, a, bb, cc, w, d = rec
                lines.append(f"K {step} {op} {a} {bb} {cc} {w} {len(d)}" + "".join(f" {x!r}" for x in d))
    return "\n".join(lines) + "\n"


def _typed(tokens: list) -> tuple:
    out = []
    for t in tokens:
        if t[0] == "i":
            out.append(int(t[1:]))
        elif t[0] == "d":
            out.append(float(t[1:]))
        else:
            out.append(None)
    return tuple(out)


def _boxes(p: list, i: int) -> tuple:
    n = int(p[i])
    return tuple(tuple(int(x) for x in p[i + 1 + 4 * j:i + 5 + 4 * j]) for j in range(n)), i + 1 + 4 * n


def _ints(p: list, i: int) -> tuple:
    n = int(p[i])
    return tuple(int(x) for x in p[i + 1:i + 1 + n]), i + 1 + n


def _s_row(p: list) -> tuple:
    ni, nd = len(S_INT), len(S_DBL)
    ints = tuple(int(x) for x in p[1:1 + ni])
    dbls = tuple(float(x) for x in p[1 + ni:1 + ni + nd])
    i = 1 + ni + nd
    boxes, i = _boxes(p, i)
    lists = []
    for _ in range(3):
        lst, i = _ints(p, i)
        lists.append(lst)
    n = int(p[i])
    mb = tuple((int(p[i + 1 + 5 * j]), *map(float, p[i + 2 + 5 * j:i + 6 + 5 * j])) for j in range(n))
    i += 1 + 5 * n
    n = int(p[i])
    mc = tuple((int(p[i + 1 + 9 * j]), int(p[i + 2 + 9 * j]), *map(float, p[i + 3 + 9 * j:i + 10 + 9 * j]))
               for j in range(n))
    i += 1 + 9 * n
    n = int(p[i])
    mx = tuple((int(p[i + 1 + 14 * j]), int(p[i + 2 + 14 * j]), *map(float, p[i + 3 + 14 * j:i + 15 + 14 * j]))
               for j in range(n))
    return ints + dbls + (boxes, *lists, mb, mc, mx)


def parse(text: str) -> dict:
    """The harness's (or the reference's) printed rows, by kind."""
    out = defaultdict(list)
    for line in text.splitlines():
        p = line.split()
        tag = p[0]
        if tag == "S":
            out["S"].append(_s_row(p))
        elif tag == "T":
            ni, nd = len(T_INT), len(T_DBL)
            ints = tuple(int(x) for x in p[1:1 + ni])
            dbls = tuple(float(x) for x in p[1 + ni:1 + ni + nd])
            boxes, _ = _boxes(p, 1 + ni + nd)
            out["T"].append(ints + dbls + (boxes,))
        elif tag == "P":
            counts = tuple(int(x) for x in p[6:14])
            nprox = int(p[14])
            prox = tuple(int(x) for x in p[15:15 + nprox])
            rest = tuple(int(x) for x in p[15 + nprox:])
            out["P"].append((int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), counts, prox, *rest))
        elif tag in "UIV":
            n = int(p[4])
            body = (tuple((int(p[5 + 2 * j]), int(p[6 + 2 * j])) for j in range(n)) if tag == "U"
                    else tuple(int(x) for x in p[5:5 + n]))
            out[tag].append((int(p[1]), int(p[2]), int(p[3]), body))
        elif tag == "W":
            n = int(p[3])
            out["W"].append((int(p[1]), int(p[2]), tuple(int(x) for x in p[4:4 + n])))
        elif tag == "D":
            nb = int(p[14])
            balls = tuple(int(x) for x in p[15:15 + nb])
            i = 15 + nb
            statics, i = _ints(p, i)
            nc = int(p[i])
            kids = tuple(tuple(int(x) for x in p[i + 1 + 3 * j:i + 4 + 3 * j]) for j in range(nc))
            out["D"].append((int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]), int(p[7]),
                             *map(float, p[8:14]), balls, statics, kids))
        elif tag == "Q":
            n = int(p[7])
            out["Q"].append((int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]), _typed(p[8:8 + n])))
        elif tag == "E":
            target, n = int(p[4]), int(p[5])
            args = _typed(p[6:6 + n])
            owner = args[0] if target == -1 else target
            if target == -1:
                args = args[1:]
            out["E"].append((int(p[1]), int(p[2]), p[3], owner, args))
        elif tag == "X":
            out["X"].append((int(p[1]), int(p[2])))
        else:
            raise ValueError(f"unknown row {line[:40]!r}")
    return out


def oracle(cs: list, timeout: int = 270) -> dict:
    """Destiny's compiled code."""
    res = subprocess.run([str(HARNESS)], input=serialize(cs), capture_output=True, text=True, check=True,
                         timeout=timeout)
    return parse(res.stdout)


def _ball_row(cid: int, s: int, park: FullBallpark, b) -> tuple:
    mb = b.my_box
    active = sum(1 << i for i in range(27) if b.active[i])
    return (cid, s, b.id, b.mode, b.effect_stamp, int(b.is_massive), int(b.sensor_active), b.follow_id, b.owner_id,
            -1 if b.follow_ptr is None else b.follow_ptr, int(b.is_free), int(b.id in park.free_balls),
            int(b.is_global), int(b.is_interactive), b.is_cloaked, int(b.is_moribund), int(b.is_space_junk),
            b.bubble, b.old_bubble, int(b.has_proximity), int(b.within_range), int(b.within_tracking_range),
            -1 if b.tracking_ptr is None else b.tracking_ptr.id, int(b.in_dead_bubble), *b.mod, active,
            mb.level if mb else -1, *(mb.key if mb else (0, 0, 0)), b.harmonic, b.corporation_id, b.alliance_id,
            int(b.sensor_only_interactives), b.formation_id, b.formation_slots, b.new_time, b.old_time,
            *b.pos, *b.vel, *b.old_pos, *b.old_vel, *b.goto, *b.last_g, *b.last_c, b.mass, b.radius, b.max_vel,
            b.agility, b.speed_fraction, b.follow_range, b.last_collision, b.time_factor, b.notification_range,
            b.tracking_range, b.sensor_range, b.sensor_period, b.sensor_elapsed,
            b.yaw, b.old_yaw, b.pitch, b.old_pitch, b.roll_angle, b.old_roll, b.roll_speed, b.old_roll_speed,
            b.yaw_delta, b.max_ang_vel, b.rot_roll, b.rot_agility, *b.rot, *b.old_rot, *b.ang_vel, *b.old_ang_vel,
            *b.torque,
            tuple((box.level, *box.key) for box in b.sorted_boxes()), tuple(b.collisions), tuple(sorted(b.followers)),
            tuple(b.sensor_balls.keys()),
            tuple((m.id, m.radius, *m.pos) for m in b.mini_balls),
            tuple((m.id, m.parent, *m.a, *m.b, m.radius) for m in b.mini_capsules),
            tuple((m.id, m.parent, *m.corner, *m.lx, *m.ly, *m.lz) for m in b.mini_boxes))


def _static_row(cid: int, s: int, st) -> tuple:
    mb = st.my_box
    return (cid, s, st.kind, st.id, st.parent_id, int(st.is_moribund), st.effect_stamp, st.bubble, st.old_bubble,
            int(st.is_interactive), int(st.handled), mb.level if mb else -1, *(mb.key if mb else (0, 0, 0)),
            *st.pos, *st.old_pos, st.bounding_radius(), tuple((box.level, *box.key) for box in st.sorted_boxes()))


def _state(out: dict, cid: int, s: int, park: FullBallpark) -> None:
    for b in park.balls.values():
        out["S"].append(_ball_row(cid, s, park, b))
    for table in (park.capsules, park.oriented_boxes):
        for sid in sorted(k for k, v in table.items() if v is not None):
            out["T"].append(_static_row(cid, s, table[sid]))
    counts = tuple(len(level) for level in park.levels)
    out["P"].append((cid, s, park.current_time, park.bubble_id, park.partition_hash(), counts,
                     tuple(park.proximity_balls.keys()), len(park.free_balls), park.time, int(park.first_time),
                     park.local_cnt, park.local_hi_cnt, len(park.formations), len(park.moribund),
                     len(park.moribund_capsules), len(park.moribund_boxes), len(park.capsules),
                     len(park.oriented_boxes)))
    for bubble, members in park.bubbles.items():
        out["U"].append((cid, s, bubble, tuple(members.items())))
    for bubble, members in park.bubble_interactives.items():
        out["I"].append((cid, s, bubble, tuple(sorted(members))))
    for bubble, members in park.bubble_keep_alives.items():
        out["V"].append((cid, s, bubble, tuple(sorted(members))))
    out["W"].append((cid, s, tuple(sorted(park.bubble_keep_alive_balls))))


def _dump(out: dict, cid: int, park: FullBallpark) -> None:
    for level in range(len(park.levels)):
        for key in sorted(park.levels[level]):
            box = park.levels[level][key]
            out["D"].append((cid, level, *key, box.bubble, int(box.handled), *box.lo, *box.hi,
                             tuple(b.id for b in box.sorted_balls()), tuple(s.id for s in box.sorted_statics()),
                             tuple(sorted(box.children))))


def _q(out: dict, cid: int, s: int, op: int, a: int, b: int, status: int, items) -> None:
    out["Q"].append((cid, s, op, a, b, status, tuple(items)))


def _apply(park: FullBallpark, rec: tuple, cid: int, s: int, out: dict) -> None:
    if rec[0] == "B":
        b = rec[2]
        park.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], b["free"], b["global"], b["massive"],
                      b["interactive"], b["junk"], *b["pos"], *b["vel"], b["agility"], b["sf"], b["max_ang_vel"],
                      b["rot_agility"])
        return
    _, _, op, a, bb, cc, w, d = rec
    d = list(d) + [0.0] * max(0, 4 - len(d))
    x, y, z, rr = d[0], d[1], d[2], d[3]
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
    elif op == POS_OP:
        park.set_position(a, x, y, z)
    elif op == VEL_OP:
        park.set_velocity(a, x, y, z)
    elif op == RADIUS_OP:
        park.set_radius(a, rr)
    elif op == MAXVEL_OP:
        park.set_max_velocity(a, rr)
    elif op == MASS_OP:
        park.set_mass(a, rr)
    elif op == AGILITY_OP:
        park.set_agility(a, rr)
    elif op == SF_OP:
        park.set_speed_fraction(a, rr)
    elif op == FREE_OP:
        park.set_free(a, w != 0)
    elif op == MASSIVE_OP:
        park.set_massive(a, w != 0)
    elif op == GLOBAL_OP:
        park.set_global(a, w != 0)
    elif op == INTERACTIVE_OP:
        park.set_interactive(a, w != 0)
    elif op == SENSOR_OP:
        park.add_proximity_sensor(a, rr, x, bb, w != 0)
    elif op == UNSENSOR_OP:
        park.remove_proximity_sensor(a)
    elif op == NOTIFY_OP:
        park.set_notification_range(a, rr)
    elif op == TRACK_OP:
        park.set_target_tracking(a, bb, rr)
    elif op == HARMONIC_OP:
        park.set_harmonic(a, bb, int(x), int(y), w != 0)
    elif op == CLOAK_ATTR_OP:
        park.set_cloaked(a, w)
    elif op == INIT_BUBBLES_OP:
        park.initialize_bubbles(a)
    elif op == COPY_BUBBLES_OP:
        park.copy_bubbles()
    elif op == CLEAN_FOLLOWERS_OP:
        _q(out, cid, s, op, a, bb, 1, park.clean_up_followers())
    elif op == KEEPALIVE_OP:
        park.add_keep_alive_ball(a)
    elif op == DEADS_OP:
        park.bring_out_the_deads()
    elif op == REMOVE_OP:
        park.remove_ball(a, w)
    elif op == VIS_Q:
        _q(out, cid, s, op, a, bb, 1, (park.check_visibility(a, bb),))
    elif op in (RANGE_BALL_Q, RANGE_POINT_Q, RANGE_DIST_Q):
        args = (a, rr, w) if op != RANGE_POINT_Q else (x, y, z, rr, w)
        res = park.ball_ids_in_range(*args, include_dist=op == RANGE_DIST_Q)
        if res is None:
            _q(out, cid, s, op, a, bb, 0, ())
        elif op == RANGE_DIST_Q:
            _q(out, cid, s, op, a, bb, 1, tuple(v for pair in res for v in pair))
        else:
            _q(out, cid, s, op, a, bb, 1, tuple(res))
    elif op == BUBBLE_AT_Q:
        bub = park.bubble_at(x, y, z)
        _q(out, cid, s, op, a, bb, 1, (-999 if bub is None else bub,))
    elif op == MISSILE_FOLLOW_OP:
        park.missile_follow(a, bb, cc)
    elif op == LAUNCH_OP:
        park.launch_missile(a, bb, cc, w & 1, (w >> 1) & 1)
    elif op == CLOAK_OP:
        if len(rec[7]) >= 1:
            park.cloak_ball(a, w, rec[7][0])
        else:
            park.cloak_ball(a, w)
    elif op == UNCLOAK_OP:
        park.uncloak_ball(a)
    elif op == CAPSULE_OP:
        park.add_capsule(a, bb, *d[:7])
    elif op == OBOX_OP:
        park.add_oriented_box(a, bb, *(d + [0.0] * 12)[:12])
    elif op == MINIBALL_OP:
        park.add_mini_ball(a, *d[:4])
    elif op == MINICAPSULE_OP:
        try:
            park.add_mini_capsule(a, *(d + [0.0] * 7)[:7])
        except ValueError:                                  # the driver clears Python's error (as Destiny's caller)
            pass
    elif op == MINIBOX_OP:
        park.add_mini_box(a, *(d + [0.0] * 12)[:12])
    elif op == REMOVE_CAPSULE_OP:
        park.remove_capsule(a, w)
    elif op == REMOVE_OBOX_OP:
        park.remove_oriented_box(a, w)
    elif op == ROTATION_OP:
        park.set_ball_rotation(a, *d[:4])
    elif op == ANGVEL_OP:
        park.set_ball_angular_velocity(a, *d[:3])
    elif op == MAX_ANGVEL_OP:
        park.set_max_angular_speed(a, d[0])
    elif op == ROT_AGILITY_OP:
        park.set_ball_angular_agility(a, d[0])
    elif op == TROLL_OP:
        park.set_ball_troll(a, w)
    elif op == RIGID_OP:
        park.set_ball_rigid(a)
    elif op == BOID_OP:
        park.set_boid(a, w)
    elif op == FORMATIONS_OP:
        forms, i = [], 0
        nf = int(rec[7][i])
        i += 1
        for _ in range(nf):
            nv = int(rec[7][i])
            i += 1
            forms.append([list(rec[7][i + 3 * j:i + 3 * j + 3]) for j in range(nv)])
            i += 3 * nv
        park.load_formations(forms)
    elif op == SET_FORMATION_OP:
        park.set_ball_formation(a, w)
    elif op == FORMATION_FOLLOW_OP:
        park.formation_follow(a, bb, w)
    elif op == RESERVE_SLOT_Q:
        ball = park.balls.get(a)
        _q(out, cid, s, op, a, bb, 1 if ball else 0, (park.reserve_formation_slot(a) if ball else 0,))
    elif op == FREE_SLOT_OP:
        park.free_formation_slot(a, w)
    elif op == MUSHROOM_OP:
        park.add_mushroom(a, d[0], d[1])
    elif op == ACCURACY_Q:
        _q(out, cid, s, op, a, bb, 1, park.get_accuracy(a, bb, *d[:5]))
    elif op == RELVEL_Q:
        rel, rad, short, trans, ang = park.relative_velocities(a, bb, w)
        _q(out, cid, s, op, a, bb, 1 if a in park.balls else 0, (*rel, rad, short, trans, ang))
    elif op == ONTICK_OP:
        park.on_tick(a)
    elif op == SHIFTED_Q:
        ok = not park.is_master and a in park.balls
        _q(out, cid, s, op, a, bb, 1 if ok else 0, (park.shifted_time(a, bb) if ok else 0,))
    elif op == ROTATED_Q:
        ball = park.balls.get(a)
        v = (f32(d[0]), f32(d[1]), f32(d[2]))
        if ball is not None:
            v = park._rotated_vector(ball, v)
        _q(out, cid, s, op, a, bb, 1 if ball else 0, v)
    elif op == EVOLVE_OP:
        park.evolve(0)
    elif op == DIST_Q:
        res = park.center_dist(a, bb)
        _q(out, cid, s, op, a, bb, 0 if res is None else 1, () if res is None else res)
    else:
        raise ValueError(f"unknown op {op}")


def reference(c: dict, trace=None) -> dict:
    """The reference on one scenario, in the oracle's layout."""
    park = FullBallpark(friction=c["friction"], tick_interval_ms=c["tick"], use_new_orbit=c["new_orbit"],
                        is_master=c["master"], seed=c["seed"], iterative=c.get("iterative", False),
                        dynamical_orientation=c.get("do", False),
                        disable_do_for_missiles=c.get("do_missiles_off", False))
    park.current_time = c["start"]
    park.trace = trace
    out = defaultdict(list)
    by_step = defaultdict(list)
    for rec in c["records"]:
        by_step[rec[1]].append(rec)
    s = 0
    try:
        for s in range(c["steps"] + 1):
            for rec in by_step.get(s, ()):
                _apply(park, rec, c["id"], s, out)
            if s > 0 and not c.get("manual"):
                park.evolve()
            _state(out, c["id"], s, park)
        _dump(out, c["id"], park)
    except DestinyUnavailable:
        out["X"].append((c["id"], s))
    for (t, name, bid, *args) in park.events:
        out["E"].append((c["id"], t, name, bid, tuple(args)))
    return out


def run_reference(cs: list, trace=None) -> tuple:
    """The reference on every scenario: (rows by kind, {case id: reason} where Destiny's result is undefined, left out
    of the rows)."""
    rows, undefined = defaultdict(list), {}
    for c in cs:
        try:
            out = reference(c, trace)
        except DestinyUndefined as exc:
            undefined[c["id"]] = str(exc)
            continue
        for k, v in out.items():
            rows[k].extend(v)
    return rows, undefined


def without(rows: dict, cases_out) -> dict:
    drop = set(cases_out)
    return {k: [row for row in v if row[0] not in drop] for k, v in rows.items()}


def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def same(a, b) -> bool:
    """Equal integers and tuples, or doubles with equal bits (two NaNs count as equal)."""
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, float) and isinstance(b, float) and (bits(a) == bits(b) or (a != a and b != b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def canonical_events(events: list) -> list:
    """The events in order, except that consecutive final clean-ups at one tick are sorted by id: BringOutDeadBalls and
    BringOutDeadCapsules walk unordered_sets of pointers (heap order, Ballpark.cpp:5716, 5772)."""
    out, i = [], 0
    while i < len(events):
        j = i
        while (j < len(events) and events[j][2] == "DoFinalCleanup" and events[j][0] == events[i][0]
               and events[j][1] == events[i][1]):
            j += 1
        if j > i + 1:
            out.extend(sorted(events[i:j], key=lambda e: e[3]))
            i = j
        else:
            out.append(events[i])
            i += 1
    return out


KINDS_ROWS = ("S", "T", "P", "U", "I", "V", "W", "D", "Q", "X")


def differences(want: dict, got: dict, undefined) -> dict:
    """Per row kind, the differing rows as (case, step, field or kind); events per case, compared as sequences of
    (time, name, owner, args) with doubles by their bits."""
    want = without(want, undefined)
    out = {}
    for kind in KINDS_ROWS:
        w, g = want.get(kind, []), got.get(kind, [])
        if len(w) != len(g):
            out[kind] = [("row count", len(w), len(g))]
            continue
        if kind == "S":
            bad = [(a[0], a[1], a[2], S_FIELDS[i]) for a, b in zip(w, g) for i in range(len(S_FIELDS))
                   if not same(a[i], b[i])]
        elif kind == "T":
            bad = [(a[0], a[1], a[3], T_FIELDS[i]) for a, b in zip(w, g) for i in range(len(T_FIELDS))
                   if not same(a[i], b[i])]
        else:
            bad = [(a[0], a[1], kind) for a, b in zip(w, g) if not same(a, b)]
        if bad:
            out[kind] = bad
    we, ge = defaultdict(list), defaultdict(list)
    for e in canonical_events(want.get("E", [])):
        we[e[0]].append(e[1:])
    for e in canonical_events(got.get("E", [])):
        ge[e[0]].append(e[1:])
    bad_events = sorted(c for c in set(we) | set(ge) if not same(tuple(we.get(c, ())), tuple(ge.get(c, ()))))
    if bad_events:
        out["E"] = bad_events
    return out


# ---- the scenario families ----

def _tick(r: random.Random) -> int:
    return r.choice([250, 500, 1000, 1000, 1000, 2000])


def _friction(r: random.Random) -> float:
    return r.choice([1.0e6] * 6 + [10 ** r.uniform(5.0, 7.0)])


def _missile_ball(r: random.Random, bid: int, pos, step: int = 0, **kw) -> tuple:
    b = {"radius": r.uniform(1.0, 5.0), "mass": 10 ** r.uniform(3.0, 5.0), "max_vel": r.uniform(300.0, 3000.0),
         "agility": r.uniform(0.01, 0.2), "sf": 1.0, "massive": r.random() < 0.5, "interactive": False,
         "free": True, "pos": list(pos), "vel": [0.0, 0.0, 0.0], "max_ang_vel": r.uniform(0.5, 3.0),
         "rot_agility": r.uniform(0.2, 1.0)}
    b.update(kw)
    return _ball(r, bid, step, **b)


def _frame(r: random.Random) -> list:
    """Three orthogonal edge directions (a rotated frame)."""
    u = _unit(r)
    t = _unit(r)
    v = [u[1] * t[2] - u[2] * t[1], u[2] * t[0] - u[0] * t[2], u[0] * t[1] - u[1] * t[0]]
    nv = math.sqrt(sum(c * c for c in v)) or 1.0
    v = [c / nv for c in v]
    w = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    return [u, v, w]


def _box_shape(r: random.Random, center, big: bool = False) -> tuple:
    """A corner and three edge vectors around a center: axis-aligned, rotated, or (rarely) skewed or flat."""
    sizes = [r.uniform(4e6, 6e6) if big else r.choice([r.uniform(5.0, 60.0), r.uniform(60.0, 400.0)])
             for _ in range(3)]
    kind = r.choice(["aligned", "aligned", "rotated", "rotated", "rotated", "skew", "flat"])
    if kind == "aligned":
        axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    elif kind == "skew":
        axes = [_unit(r) for _ in range(3)]
    else:
        axes = _frame(r)
    if kind == "flat":
        sizes[r.randrange(3)] = 0.0
    edges = [[axes[i][j] * sizes[i] for j in range(3)] for i in range(3)]
    corner = [center[j] - 0.5 * (edges[0][j] + edges[1][j] + edges[2][j]) for j in range(3)]
    return corner, edges


def _capsule_shape(r: random.Random, center) -> tuple:
    """A segment A-B around a center and a radius (zero length or zero radius sometimes)."""
    u = _unit(r)
    length = r.choice([0.0, r.uniform(10.0, 120.0), r.uniform(120.0, 400.0)])
    radius = r.choice([0.0, r.uniform(1.0, 20.0), r.uniform(20.0, 80.0)]) if r.random() < 0.97 else 0.0
    if radius > 0.0 and length > 0.0:
        radius = max(radius, length / 300.0)               # keeps Capsule::InsertInBoxes' 2r steps bounded
    a = [center[i] - 0.5 * length * u[i] for i in range(3)]
    b = [center[i] + 0.5 * length * u[i] for i in range(3)]
    return a, b, radius


def _ship_at(r: random.Random, bid: int, pos, step: int = 0, **kw) -> tuple:
    b = {"radius": r.choice([2.0, r.uniform(1.0, 15.0), r.uniform(15.0, 60.0)]), "mass": 10 ** r.uniform(6.0, 8.5),
         "max_vel": r.choice([10.0, r.uniform(5.0, 150.0)]), "agility": r.uniform(0.2, 1.5), "free": True,
         "massive": True, "interactive": r.random() < 0.6, "pos": list(pos),
         "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-20.0, 20.0) for _ in range(3)]]), "sf": r.choice([0.95, 1.0])}
    b.update(kw)
    return _ball(r, bid, step, **b)


def missile_cases(r: random.Random, out: list, n: int) -> None:
    """Missiles (MOV-723 to MOV-725, COL-719, COL-703/704's missile parts): launches aimed or not, massive or not,
    from moving, stationary (fixed) or defending owners, at targets near or cloaked or missing; MissileFollow
    directly; the targets warping, cloaking, removed or moved to another bubble; owners removed or cloaked; colliders
    on the path; both collision methods; dynamical orientation, with or without the missile exclusion."""
    for _ in range(n):
        master, iterative, do = r.random() < 0.8, r.random() < 0.3, r.random() < 0.3
        ids = _ids(r, r.randint(4, 8))
        owner, target = ids[0], ids[1]
        missiles = ids[2:2 + r.randint(1, min(3, len(ids) - 2))]
        rest = ids[2 + len(missiles):]
        moving = r.random() < 0.6 or (do and r.random() < 0.5)
        recs = [_ball(r, owner, pos=[r.uniform(-500, 500) for _ in range(3)], free=moving or r.random() < 0.3,
                      vel=[r.uniform(-100, 100) for _ in range(3)] if moving else [0.0, 0.0, 0.0],
                      interactive=True, massive=True)]
        u = _unit(r)
        d = r.uniform(2e3, 3e4)
        recs.append(_ball(r, target, pos=[u[i] * d for i in range(3)], interactive=True, massive=r.random() < 0.85,
                          radius=r.uniform(20.0, 300.0)))
        if r.random() < 0.5:
            v = _unit(r)
            recs.append(_k(r.randint(0, 3), GOTO_OP, target, d=(v[0] * 5e4, v[1] * 5e4, v[2] * 5e4)))
        for mid in missiles:
            recs.append(_missile_ball(r, mid, [0.0, 0.0, 0.0]))
        for bid in rest:                                    # colliders and bystanders on the way
            t = r.uniform(0.1, 0.9)
            recs.append(_ball(r, bid, pos=[u[i] * d * t + r.uniform(-50, 50) for i in range(3)],
                              radius=r.uniform(10.0, 500.0), massive=True, free=r.random() < 0.6))
        steps = r.randint(6, 25)
        for j, mid in enumerate(missiles):
            k = r.randint(1, 3)
            how = r.choice(["launch", "launch", "launch", "follow", "no_target", "self", "defender"])
            if how == "follow":
                v = _unit(r)
                recs.append(_k(k, VEL_OP, mid, d=(v[0] * 100.0, v[1] * 100.0, v[2] * 100.0)))
                recs.append(_k(k, MISSILE_FOLLOW_OP, mid, target, r.choice([owner, owner, -owner, mid, target])))
            else:
                dst = {"launch": target, "no_target": 999999937, "self": owner, "defender": target}[how]
                own = -owner if how == "defender" else owner
                recs.append(_k(k, LAUNCH_OP, mid, dst, own, w=r.randrange(4)))
        if r.random() < 0.5:
            k = r.randint(2, steps)
            what = r.choice(["cloak_target", "remove_target", "warp_target", "far_target", "remove_owner",
                             "cloak_owner", "uncloak_target"])
            if what == "cloak_target":
                recs.append(_k(k, CLOAK_OP, target, w=r.choice([1, 2])))
            elif what == "uncloak_target":
                recs.append(_k(1, CLOAK_OP, target, w=1))
                recs.append(_k(k, UNCLOAK_OP, target))
            elif what == "remove_target":
                recs.append(_k(k, REMOVE_OP, target, w=r.choice([0, 0, 2])))
            elif what == "warp_target":
                v = _unit(r)
                recs.append(_k(k, WARP_OP, target, w=r.randint(1000, 20000), d=(v[0] * 1e9, v[1] * 1e9, v[2] * 1e9,
                                                                                 0.0)))
            elif what == "far_target":
                recs.append(_k(k, POS_OP, target, d=(5 * TOP, 0.0, 0.0)))
                recs.append(_k(k + 1, CLEAN_FOLLOWERS_OP))
            elif what == "remove_owner":
                recs.append(_k(k, REMOVE_OP, owner, w=r.choice([0, 0, 3])))
            else:
                recs.append(_k(k, CLOAK_OP, owner, w=1))
        for k in range(steps + 1):
            if r.random() < 0.3:
                recs.append(_k(k, DEADS_OP))
        case(out, r, "missile", recs, steps, master=master, tick=_tick(r), iterative=iterative, do=do,
             do_missiles_off=do and r.random() < 0.4, friction=_friction(r), new_orbit=r.random() < 0.3)


def static_cases(r: random.Random, out: list, n: int) -> None:
    """Static collidables (COL-712, COL-713, MOV-732, and their parts of COL-703, 711, 715, 716, 718): capsules and
    oriented boxes made directly or as minis of fixed parents, miniballs; ships flying into them, along them, starting
    inside, in both collision methods; parents freed and fixed again, removed; static collidables removed at once, with
    a delay (brought out later) or twice (a stale id), re-added under their id; across bubbles on the server."""
    for _ in range(n):
        master, iterative, do = r.random() < 0.6, r.random() < 0.45, r.random() < 0.2
        center = [r.choice([0.0, r.uniform(-1e4, 1e4)]) for _ in range(3)]
        if r.random() < 0.08:                               # near a top box face: shapes in two bubbles
            center[0] = r.randint(-2, 2) * TOP + r.uniform(-200.0, 200.0)
        recs: list = []
        statics_ids: list = []
        parents = []
        n_parents = r.randint(0, 2)
        for pi in range(n_parents):
            pid = 9000 + pi
            parents.append(pid)
            recs.append(_parent(pid, pos=[center[i] + r.uniform(-100, 100) for i in range(3)],
                                radius=r.uniform(50.0, 600.0), mass=10 ** r.uniform(6.0, 10.0),
                                free=r.random() < 0.15))
            for _ in range(r.randint(1, 3)):
                off = [r.uniform(-300, 300) for _ in range(3)]
                kind = r.choice(["capsule", "box", "ball"])
                if kind == "capsule":
                    a, b, rad = _capsule_shape(r, off)
                    if rad <= 0.0 and r.random() < 0.7:
                        rad = r.uniform(1.0, 30.0)
                    recs.append(_k(0, MINICAPSULE_OP, pid, d=(*a, *b, rad)))
                elif kind == "box":
                    corner, edges = _box_shape(r, off)
                    recs.append(_k(0, MINIBOX_OP, pid, d=(*corner, *edges[0], *edges[1], *edges[2])))
                else:
                    recs.append(_k(0, MINIBALL_OP, pid, d=(*off, r.uniform(1.0, 80.0))))
        for j in range(r.randint(0 if n_parents else 1, 3)):
            sid = 5000 + j
            statics_ids.append(sid)
            c = [center[i] + r.uniform(-300, 300) for i in range(3)]
            if r.random() < 0.55:
                a, b, rad = _capsule_shape(r, c)
                recs.append(_k(0, CAPSULE_OP, sid, r.choice([0, 7, 9000]), d=(*a, *b, rad)))
            else:
                corner, edges = _box_shape(r, c, big=r.random() < 0.08)
                recs.append(_k(0, OBOX_OP, sid + 100, r.choice([0, 7, 9000]),
                               d=(*corner, *edges[0], *edges[1], *edges[2])))
        ships = _ids(r, r.randint(1, 4))
        for bid in ships:
            start = [center[i] + r.uniform(-450, 450) for i in range(3)]
            if r.random() < 0.15:
                recs += _test_ship(bid, pos=start, max_vel=r.choice([10.0, 50.0]), mass=r.choice([13000000.0, 1e9]))
            else:
                recs.append(_ship_at(r, bid, start))
            target = [center[i] + r.uniform(-300, 300) for i in range(3)]
            if r.random() < 0.85:
                recs.append(_k(r.choice([0, 1, 1]), GOTO_OP, bid, d=target))
        steps = r.randint(6, 30)
        for _ in range(r.randint(0, 4)):
            k = r.randint(1, steps)
            what = r.choice(["free", "fix", "remove_static", "remove_static_delay", "remove_parent", "readd",
                             "deads", "remove_twice", "pos"])
            if what in ("free", "fix") and parents:
                recs.append(_k(k, FREE_OP, r.choice(parents), w=int(what == "free")))
            elif what == "remove_static" and statics_ids:
                recs.append(_k(k, REMOVE_CAPSULE_OP, r.choice(statics_ids), w=0))
                recs.append(_k(k, REMOVE_OBOX_OP, r.choice(statics_ids) + 100, w=0))
            elif what == "remove_static_delay" and statics_ids:
                recs.append(_k(k, REMOVE_CAPSULE_OP, r.choice(statics_ids), w=r.randint(1, 4)))
                recs.append(_k(k, REMOVE_OBOX_OP, r.choice(statics_ids) + 100, w=r.randint(1, 4)))
            elif what == "remove_parent" and parents:
                recs.append(_k(k, REMOVE_OP, r.choice(parents), w=r.choice([0, 0, 2])))
            elif what == "readd" and statics_ids:
                sid = r.choice(statics_ids)
                a, b, rad = _capsule_shape(r, center)
                recs.append(_k(k, CAPSULE_OP, sid, 7, d=(*a, *b, rad)))
            elif what == "remove_twice":
                recs.append(_k(k, REMOVE_CAPSULE_OP, r.choice([-1073741825, -1073741826, 4999]), w=0))
            elif what == "pos" and parents:
                recs.append(_k(k, POS_OP, r.choice(parents), d=[center[i] + r.uniform(-50, 50) for i in range(3)]))
            else:
                recs.append(_k(k, DEADS_OP))
        for k in range(steps + 1):
            if r.random() < 0.3:
                recs.append(_k(k, DEADS_OP))
        case(out, r, "static", recs, steps, master=master, tick=_tick(r), iterative=iterative, do=do,
             friction=r.choice([1.0e6] * 6 + [10 ** r.uniform(5.0, 7.0)]))


def cloak_cases(r: random.Random, out: list, n: int) -> None:
    """Cloaking (SEN-711, MOV-730's cloak trigger): modes normal, restore, GM, zero, negative and above 127; the
    master's 2 s sensor; followers (follow, orbit, missiles) released; uncloaking in and out of warp proper, and
    leaving other sensors' lists; cloaked balls in sensors, range queries, visibility and collisions."""
    for _ in range(n):
        master = r.random() < 0.7
        ids = _ids(r, r.randint(3, 7))
        tgt = ids[0]
        recs = [_ball(r, bid, pos=[r.uniform(-2e3, 2e3) for _ in range(3)], interactive=True,
                      massive=r.random() < 0.8, free=r.random() < 0.9) for bid in ids]
        for bid in ids[1:]:
            how = r.choice(["follow", "orbit", "missile", "sensor", "none"])
            if how == "follow":
                recs.append(_k(1, FOLLOW_OP, bid, tgt, d=_xyz(rr=r.uniform(0, 500))))
            elif how == "orbit":
                recs.append(_k(1, ORBIT_OP, bid, tgt, d=_xyz(rr=r.uniform(0, 500))))
            elif how == "missile":
                v = _unit(r)
                recs.append(_k(1, VEL_OP, bid, d=(v[0] * 50, v[1] * 50, v[2] * 50)))
                recs.append(_k(1, MISSILE_FOLLOW_OP, bid, tgt, ids[-1] if ids[-1] != bid else 0))
            elif how == "sensor":
                recs.append(_k(0, SENSOR_OP, bid, d=(r.choice([0.0, 1.0]), 0.0, 0.0, r.uniform(500, 5000))))
        steps = r.randint(4, 20)
        k = r.randint(1, steps)
        mode = r.choice([1, 1, 1, 2, 3, 0, -1, 200])
        recs.append(_k(k, CLOAK_OP, tgt, w=mode, d=(r.uniform(100, 5000),) if r.random() < 0.6 else ()))
        if r.random() < 0.6:
            k2 = r.randint(k, steps)
            if r.random() < 0.4:                            # uncloak in warp proper
                v = _unit(r)
                recs.append(_k(max(1, k2 - 3), WARP_OP, tgt, w=r.randint(2000, 20000),
                               d=(v[0] * 1e9, v[1] * 1e9, v[2] * 1e9, 0.0)))
            recs.append(_k(k2, UNCLOAK_OP, tgt))
        for _ in range(r.randint(0, 3)):
            kk = r.randint(0, steps)
            recs.append(r.choice([_k(kk, RANGE_BALL_Q, ids[1], w=r.randint(0, 1), d=_xyz(rr=r.uniform(0, 5e3))),
                                  _k(kk, VIS_Q, ids[1], ids[-1]), _k(kk, GOTO_OP, tgt, d=(0.0, 0.0, 0.0)),
                                  _k(kk, CLOAK_ATTR_OP, r.choice(ids), w=r.randint(0, 1))]))
        case(out, r, "cloak", recs, steps, master=master, tick=_tick(r), iterative=r.random() < 0.2)


def orientation_cases(r: random.Random, out: list, n: int, do: bool) -> None:
    """Orientation (MOV-729 without dynamical orientation, MOV-728 with it): ships in every mode, turning, stopping,
    warping (aligning against obstacles; heavy ships in warp proper), bumped, given rotations and angular velocities,
    their limits and agilities; missiles excluded or not."""
    for _ in range(n):
        ids = _ids(r, r.randint(2, 6))
        recs = []
        for bid in ids:
            heavy = r.random() < 0.15
            recs.append(_ball(r, bid, pos=[r.uniform(-3e3, 3e3) for _ in range(3)],
                              mass=10 ** r.uniform(8.3, 9.5) if heavy else 10 ** r.uniform(5.5, 8.0),
                              massive=r.random() < 0.7))
        if r.random() < 0.3:                                # an obstacle a warping ship aligns against
            recs.append(_k(0, POS_OP, ids[-1], d=[c + 1.0 for c in recs[0][2]["pos"]]))
        steps = r.randint(8, 40)
        for _ in range(r.randint(2, 8)):
            k = r.randint(1, steps)
            a, b = r.choice(ids), r.choice(ids)
            op = r.choice([GOTO_OP, DIR_OP, STOP_OP, FOLLOW_OP, ORBIT_OP, WARP_OP, VEL_OP, ROTATION_OP, ANGVEL_OP,
                           MAX_ANGVEL_OP, ROT_AGILITY_OP, ROTATED_Q])
            if op in (GOTO_OP, DIR_OP):
                recs.append(_k(k, op, a, d=[r.uniform(-1e4, 1e4) for _ in range(3)]))
            elif op == STOP_OP:
                recs.append(_k(k, op, a))
            elif op in (FOLLOW_OP, ORBIT_OP):
                recs.append(_k(k, op, a, b, d=_xyz(rr=r.uniform(0, 2000))))
            elif op == WARP_OP:
                v = _unit(r)
                dd = r.choice([5e4, 10 ** r.uniform(6.5, 9.0)])
                recs.append(_k(k, op, a, w=r.randint(2000, 20000), d=(v[0] * dd, v[1] * dd, v[2] * dd,
                                                                     r.choice([0.0, 20000.0]))))
            elif op == VEL_OP:
                recs.append(_k(k, op, a, d=r.choice([[0.0, 0.0, 0.0], [r.uniform(-200, 200) for _ in range(3)]])))
            elif op == ROTATION_OP:
                q = [r.gauss(0, 1) for _ in range(4)]
                if r.random() < 0.1:
                    q = [0.0, 0.0, 0.0, 0.0]
                recs.append(_k(k, op, a, d=q))
            elif op == ANGVEL_OP:
                recs.append(_k(k, op, a, d=[r.uniform(-1, 1) for _ in range(3)]))
            elif op == MAX_ANGVEL_OP:
                recs.append(_k(k, op, a, d=(r.choice([r.uniform(0.0, 3.0), -1.0]),)))
            elif op == ROT_AGILITY_OP:
                recs.append(_k(k, op, a, d=(r.choice([r.uniform(0.05, 3.0), -1.0, 0.0]),)))
            else:
                recs.append(_k(k, op, a, d=_unit(r)))
        case(out, r, "orient_do" if do else "orient", recs, steps, master=r.random() < 0.7, tick=_tick(r),
             do=do, iterative=r.random() < 0.25, new_orbit=r.random() < 0.5, friction=_friction(r))


def mushroom_cases(r: random.Random, out: list, n: int) -> None:
    """Mushrooms (MOV-734): AddMushroom (refused for a non-positive range or time, or a missing owner), freed to grow
    as range x (t / T)^0.25 until removed; owners ignoring theirs; balls bumped; on the server (a local id) and a
    client (id 0)."""
    for _ in range(n):
        master, do = r.random() < 0.6, r.random() < 0.2
        ids = _ids(r, r.randint(2, 5))
        owner = ids[0]
        recs = [_ball(r, bid, pos=[r.uniform(-500, 500) for _ in range(3)], massive=True,
                      interactive=r.random() < 0.6) for bid in ids]
        rng = r.choice([r.uniform(50.0, 400.0), r.uniform(400.0, 3000.0), -5.0, 0.0])
        life = r.choice([r.uniform(1.0, 8.0), r.uniform(8.0, 25.0), 0.0])
        recs.append(_k(0, MUSHROOM_OP, r.choice([owner, owner, owner, 999999]), d=(rng, life)))
        mid = (-3 if do else -2) if master else 0
        steps = r.randint(4, 30)
        if r.random() < 0.85:
            recs.append(_k(r.randint(0, 2), FREE_OP, mid, w=1))
        for k in range(steps + 1):
            if r.random() < 0.6:
                recs.append(_k(k, DEADS_OP))
        for bid in ids[1:]:
            if r.random() < 0.5:
                recs.append(_k(1, GOTO_OP, bid, d=recs[0][2]["pos"]))
        case(out, r, "mushroom", recs, steps, master=master, tick=_tick(r), do=do, iterative=r.random() < 0.3)


def troll_cases(r: random.Random, out: list, n: int) -> None:
    """Trolls, rigid balls and boids (MOV-736, MOV-734): SetBallTroll (delays below 1 become 1) frees and makes
    interactive; petrified at the end of the tick it is due (fixed, non-interactive, RIGID), its minis made inside
    Evolve; SetBallRigid; SetBoid on and off."""
    for _ in range(n):
        ids = _ids(r, r.randint(2, 5))
        recs = [_ball(r, bid, pos=[r.uniform(-1e3, 1e3) for _ in range(3)], free=r.random() < 0.6,
                      massive=r.random() < 0.8, interactive=r.random() < 0.5) for bid in ids]
        steps = r.randint(4, 20)
        for bid in ids:
            if r.random() < 0.3:
                recs.append(_k(0, MINIBALL_OP, bid, d=(r.uniform(-50, 50), 0.0, 0.0, r.uniform(1.0, 30.0))))
            if r.random() < 0.2:
                a, b, rad = _capsule_shape(r, [0.0, 0.0, 0.0])
                recs.append(_k(0, MINICAPSULE_OP, bid, d=(*a, *b, max(rad, 2.0))))
            what = r.choice(["troll", "troll", "rigid", "boid", "boid_off", "goto"])
            k = r.randint(0, steps)
            if what == "troll":
                recs.append(_k(k, TROLL_OP, bid, w=r.choice([-1, 0, 1, 2, 5])))
            elif what == "rigid":
                recs.append(_k(k, RIGID_OP, bid))
            elif what == "boid":
                recs.append(_k(k, BOID_OP, bid, w=1))
            elif what == "boid_off":
                recs.append(_k(max(0, k - 2), BOID_OP, bid, w=1))
                recs.append(_k(k, BOID_OP, bid, w=0))
            else:
                recs.append(_k(k, GOTO_OP, bid, d=[r.uniform(-1e3, 1e3) for _ in range(3)]))
        case(out, r, "troll", recs, steps, master=r.random() < 0.7, tick=_tick(r), iterative=r.random() < 0.2)


def formation_cases(r: random.Random, out: list, n: int) -> None:
    """Formations (MOV-726): LoadFormations; a leader's formation set, changed (a smaller one stops the followers in
    the slots it lacks), left (-1 stops them all), or out of range; FormationFollow with slots in and out of range;
    slots reserved (the first free of at most 16) and freed; the leader turning (dynamical orientation), fixed or at
    zero angles otherwise, and once in a while turned without it (DirectXMath's rotation; until 23 October 2026 the
    scenario stopped there)."""
    for _ in range(n):
        do = r.random() < 0.6
        ids = _ids(r, r.randint(3, 8))
        leader, followers = ids[0], ids[1:]
        forms = []
        for _ in range(r.randint(1, 3)):
            size = r.choice([r.randint(1, 6), r.randint(6, 16), 17])
            forms.append([[r.choice([0.0, r.uniform(-500, 500)]) for _ in range(3)] for _ in range(size)])
        data = [float(len(forms))]
        for f in forms:
            data.append(float(len(f)))
            for v in f:
                data.extend(v)
        turned = do or r.random() < 0.08
        recs = [_k(0, FORMATIONS_OP, d=data),
                _ball(r, leader, pos=[0.0, 0.0, 0.0], free=turned or r.random() < 0.3, interactive=True,
                      vel=[r.uniform(-50, 50) for _ in range(3)] if turned else [0.0, 0.0, 0.0])]
        recs.append(_k(0, SET_FORMATION_OP, leader, w=r.choice([0, 0, len(forms) - 1, -1, len(forms)])))
        for bid in followers:
            recs.append(_ball(r, bid, pos=[r.uniform(-2e3, 2e3) for _ in range(3)], interactive=True))
            recs.append(_k(1, FORMATION_FOLLOW_OP, bid, leader, w=r.choice([0, 1, 2, 3, 5, 15, 17])))
        steps = r.randint(5, 25)
        if turned and r.random() < 0.6:
            recs.append(_k(r.randint(1, 5), GOTO_OP, leader, d=[r.uniform(-1e4, 1e4) for _ in range(3)]))
        for _ in range(r.randint(0, 5)):
            k = r.randint(1, steps)
            what = r.choice(["set", "reserve", "free_slot", "leave", "warp", "reload", "shrink"])
            if what == "shrink":                            # to the smallest formation: followers past it stop
                recs.append(_k(k, SET_FORMATION_OP, leader, w=min(range(len(forms)), key=lambda i: len(forms[i]))))
            elif what == "reload":                          # LoadFormations again, as many but smaller: no stops
                small = [[[r.uniform(-300, 300) for _ in range(3)] for _ in range(r.randint(1, 3))]
                         for _ in range(len(forms))]
                data2 = [float(len(small))]
                for f in small:
                    data2.append(float(len(f)))
                    for v in f:
                        data2.extend(v)
                recs.append(_k(k, FORMATIONS_OP, d=data2))
            elif what == "set":
                recs.append(_k(k, SET_FORMATION_OP, leader, w=r.randrange(len(forms))))
            elif what == "reserve":
                recs.append(_k(k, RESERVE_SLOT_Q, r.choice([leader, leader, followers[0]])))
            elif what == "free_slot":
                recs.append(_k(k, FREE_SLOT_OP, leader, w=r.choice([0, 1, 3, 20, -1])))
            elif what == "leave":
                recs.append(_k(k, SET_FORMATION_OP, leader, w=-1))
            else:
                v = _unit(r)
                recs.append(_k(k, WARP_OP, leader, w=5000, d=(v[0] * 1e9, v[1] * 1e9, v[2] * 1e9, 0.0)))
        case(out, r, "formation", recs, steps, master=r.random() < 0.7, tick=_tick(r), do=do)


def accuracy_cases(r: random.Random, out: list, n: int) -> None:
    """The accuracy helpers (CMB-701, CMB-702): GetAccuracy with in and out of range, zero and negative parameters,
    the same ball, a missing ball; CalculateRelativeVelocities with every flag combination, with and without
    dynamical orientation; GetCenterDist and GetSurfaceDist."""
    for _ in range(n):
        do = r.random() < 0.4
        ids = _ids(r, r.randint(2, 4))
        recs = [_ball(r, bid, pos=[r.uniform(-2e4, 2e4) for _ in range(3)],
                      vel=[r.uniform(-500, 500) for _ in range(3)]) for bid in ids]
        if r.random() < 0.15:
            recs.append(_k(0, POS_OP, ids[1], d=recs[0][2]["pos"]))
        steps = r.randint(1, 5)
        for k in range(steps + 1):
            for _ in range(r.randint(1, 4)):
                a, b = r.choice(ids), r.choice(ids + [987654321])
                what = r.choice(["acc", "acc", "rel", "dist", "angvel"])
                if what == "acc":
                    recs.append(_k(k, ACCURACY_Q, a, b, d=(r.choice([r.uniform(0, 2e4), -1.0, 0.0]),
                                                          r.choice([r.uniform(1, 2e4), 0.0]),
                                                          r.choice([r.uniform(0.001, 2.0), 0.0]),
                                                          r.choice([r.uniform(10, 500), 0.0]),
                                                          r.uniform(10, 500))))
                elif what == "rel":
                    recs.append(_k(k, RELVEL_Q, a, b, w=r.randrange(16)))
                elif what == "dist":
                    recs.append(_k(k, DIST_Q, a, b))
                else:
                    recs.append(_k(k, ANGVEL_OP, a, d=[r.uniform(-0.5, 0.5) for _ in range(3)]))
            if r.random() < 0.5:
                recs.append(_k(k, GOTO_OP, r.choice(ids), d=[r.uniform(-2e4, 2e4) for _ in range(3)]))
        case(out, r, "accuracy", recs, steps, master=r.random() < 0.6, do=do)


def ontick_cases(r: random.Random, out: list, n: int) -> None:
    """OnTick (TIME-702 to TIME-705) and the client's shifted time (TIME-708): ticks driven by sim times that advance
    by fractions and multiples of the interval (the first call evolving once; more than 5 intervals resynchronised on
    the server, caught up on a client); moribund balls brought out before the elapsed-time check; balls freed after a
    tick (mNewTime a tick back); Evolve between OnTicks; GetShiftedTime on a client's balls."""
    for _ in range(n):
        master = r.random() < 0.5
        tick = _tick(r)
        interval = tick * 10000
        ids = _ids(r, r.randint(1, 4))
        recs = [_ball(r, bid, pos=[r.uniform(-1e3, 1e3) for _ in range(3)], free=r.random() < 0.8) for bid in ids]
        recs.append(_k(0, GOTO_OP, ids[0], d=[r.uniform(-1e4, 1e4) for _ in range(3)]))
        steps = r.randint(4, 16)
        sim = r.choice([0, r.randrange(1, interval), interval, r.randrange(interval, 20 * interval)])
        for k in range(1, steps + 1):
            sim += r.choice([0, r.randrange(1, interval), interval, interval, 2 * interval + r.randrange(interval),
                             r.randint(3, 9) * interval + r.randrange(interval)])
            recs.append(_k(k, ONTICK_OP, sim))
            if r.random() < 0.2:
                recs.append(_k(k, EVOLVE_OP))
            if r.random() < 0.2:
                recs.append(_k(k, REMOVE_OP, r.choice(ids), w=r.choice([0, 1, 3])))
            if r.random() < 0.2:
                recs.append(_k(k, FREE_OP, r.choice(ids), w=r.randint(0, 1)))
            if r.random() < 0.4:
                recs.append(_k(k, SHIFTED_Q, r.choice(ids), r.randrange(0, 10 ** 9)))
        case(out, r, "ontick", recs, steps, master=master, tick=tick, manual=True)


def mixed_cases(r: random.Random, out: list, n: int) -> None:
    """Everything together: slices 1 and 2's commands with missiles, static collidables, cloaks, trolls, formations
    and mushrooms, with and without dynamical orientation and the iterative method."""
    for _ in range(n):
        do, iterative = r.random() < 0.35, r.random() < 0.35
        ids = _ids(r, r.randint(3, 7))
        recs = [_ball(r, bid, pos=[r.uniform(-3e3, 3e3) for _ in range(3)], interactive=r.random() < 0.7,
                      massive=r.random() < 0.75) for bid in ids]
        if r.random() < 0.5:
            a, b, rad = _capsule_shape(r, [r.uniform(-500, 500) for _ in range(3)])
            recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*a, *b, max(rad, 5.0))))
        if r.random() < 0.4:
            corner, edges = _box_shape(r, [r.uniform(-500, 500) for _ in range(3)])
            recs.append(_k(0, OBOX_OP, 5100, 0, d=(*corner, *edges[0], *edges[1], *edges[2])))
        if r.random() < 0.5:
            recs.append(_k(r.randint(0, 2), SENSOR_OP, r.choice(ids), d=(r.choice([0.0, 2.0]), 0.0, 0.0,
                                                                         r.uniform(100, 5000))))
        steps = r.randint(8, 40)
        for _ in range(r.randint(3, 10)):
            k = r.randint(1, steps)
            a, b = r.choice(ids), r.choice(ids)
            op = r.choice([GOTO_OP, DIR_OP, STOP_OP, FOLLOW_OP, ORBIT_OP, WARP_OP, LAUNCH_OP, MISSILE_FOLLOW_OP,
                           CLOAK_OP, UNCLOAK_OP, TROLL_OP, BOID_OP, RIGID_OP, REMOVE_OP, FREE_OP, VIS_Q, RANGE_BALL_Q,
                           CLEAN_FOLLOWERS_OP, ACCURACY_Q, ROTATION_OP])
            if op in (GOTO_OP, DIR_OP):
                recs.append(_k(k, op, a, d=[r.uniform(-1e4, 1e4) for _ in range(3)]))
            elif op in (STOP_OP, UNCLOAK_OP, RIGID_OP, CLEAN_FOLLOWERS_OP):
                recs.append(_k(k, op, a))
            elif op in (FOLLOW_OP, ORBIT_OP):
                recs.append(_k(k, op, a, b, d=_xyz(rr=r.uniform(-100, 2000))))
            elif op == WARP_OP:
                v = _unit(r)
                dd = r.choice([5e4, 10 ** r.uniform(6.5, 9.0)])
                recs.append(_k(k, op, a, w=r.randint(2000, 20000), d=(v[0] * dd, v[1] * dd, v[2] * dd, 0.0)))
            elif op == LAUNCH_OP:
                c = r.choice(ids)
                recs.append(_k(k, op, a, b, c, w=r.randrange(4)))
            elif op == MISSILE_FOLLOW_OP:
                recs.append(_k(k, op, a, b, r.choice(ids)))
            elif op == CLOAK_OP:
                recs.append(_k(k, op, a, w=r.choice([1, 2, 3])))
            elif op == TROLL_OP:
                recs.append(_k(k, op, a, w=r.randint(0, 4)))
            elif op == BOID_OP:
                recs.append(_k(k, op, a, w=r.randint(0, 1)))
            elif op == REMOVE_OP:
                recs.append(_k(k, op, a, w=r.choice([0, 0, 2])))
            elif op == FREE_OP:
                recs.append(_k(k, op, a, w=r.randint(0, 1)))
            elif op == VIS_Q:
                recs.append(_k(k, op, a, b))
            elif op == RANGE_BALL_Q:
                recs.append(_k(k, op, a, w=r.randint(0, 1), d=_xyz(rr=r.uniform(0, 1e4))))
            elif op == ACCURACY_Q:
                recs.append(_k(k, op, a, b, d=(r.uniform(0, 5e3), r.uniform(1, 5e3), r.uniform(0.01, 1.0),
                                               r.uniform(10, 300), r.uniform(10, 300))))
            else:
                recs.append(_k(k, op, a, d=[r.gauss(0, 1) for _ in range(4)]))
        for k in range(steps + 1):
            if r.random() < 0.3:
                recs.append(_k(k, DEADS_OP))
        case(out, r, "mixed", recs, steps, master=r.random() < 0.75, tick=_tick(r), do=do, iterative=iterative,
             do_missiles_off=do and r.random() < 0.3, new_orbit=r.random() < 0.5, friction=_friction(r))


def _perp(r: random.Random, axis) -> list:
    """A unit vector perpendicular to a unit axis."""
    while True:
        u = _unit(r)
        d = sum(u[i] * axis[i] for i in range(3))
        v = [u[i] - d * axis[i] for i in range(3)]
        n = math.sqrt(sum(c * c for c in v))
        if n > 1e-3:
            return [c / n for c in v]


def approach_cases(r: random.Random, out: list, n: int) -> None:
    """Ships flown at one static shape (COL-712, COL-715, COL-716): at a capsule's cylinder or caps, along its axis, at
    a zero-length capsule, grazing, starting inside the cylinder or a cap (slow or fast), backing out (velocity in,
    thrust out); at an oriented box's faces, edges and corners, starting inside it; in both collision methods, with
    dynamical orientation sometimes; the shape made directly or as a parent's mini."""
    for _ in range(n):
        iterative, do, master = r.random() < 0.5, r.random() < 0.2, r.random() < 0.5
        center = [r.uniform(-3e3, 3e3) for _ in range(3)]
        recs: list = []
        capsule = r.random() < 0.55
        mini = r.random() < 0.4
        ship_r = r.choice([2.0, r.uniform(1.0, 10.0), r.uniform(10.0, 40.0)])
        if capsule:
            axis = _unit(r)
            length = r.choice([0.0, r.uniform(20.0, 150.0), r.uniform(150.0, 400.0)])
            rad = r.choice([r.uniform(1.0, 10.0), r.uniform(10.0, 60.0)])
            rad = max(rad, length / 300.0)
            a = [center[i] - 0.5 * length * axis[i] for i in range(3)]
            b = [center[i] + 0.5 * length * axis[i] for i in range(3)]
            if mini:
                ppos = [center[i] + r.uniform(-50, 50) for i in range(3)]
                recs.append(_parent(9000, pos=ppos, radius=r.uniform(50.0, 500.0)))
                recs.append(_k(0, MINICAPSULE_OP, 9000, d=(*[a[i] - ppos[i] for i in range(3)],
                                                             *[b[i] - ppos[i] for i in range(3)], rad)))
            else:
                recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*a, *b, rad)))
        else:
            corner, edges = _box_shape(r, center)
            if mini:
                ppos = [center[i] + r.uniform(-50, 50) for i in range(3)]
                recs.append(_parent(9000, pos=ppos, radius=r.uniform(50.0, 500.0)))
                recs.append(_k(0, MINIBOX_OP, 9000, d=(*[corner[i] - ppos[i] for i in range(3)],
                                                         *edges[0], *edges[1], *edges[2])))
            else:
                recs.append(_k(0, OBOX_OP, 5100, 0, d=(*corner, *edges[0], *edges[1], *edges[2])))
        ships = _ids(r, r.randint(1, 2))
        steps = r.randint(8, 30)
        for bid in ships:
            flavour = r.choice(["through", "through", "graze", "inside_slow", "inside_fast", "along", "backing",
                                "corner"])
            speed = r.choice([10.0, r.uniform(5.0, 60.0), r.uniform(60.0, 300.0)])
            if capsule:
                t_ax = r.choice([r.uniform(0.0, 1.0), r.uniform(-0.4, 1.4)])
                p_ax = [a[i] + (b[i] - a[i]) * t_ax for i in range(3)]
                nrm = _perp(r, axis)
                reach = rad + ship_r
                if flavour == "along":
                    start = [a[i] - axis[i] * (reach + r.uniform(5.0, 60.0)) + nrm[i] * r.uniform(-reach, reach)
                             * r.choice([0.0, 0.0, 0.5]) for i in range(3)]
                    goal = [b[i] + axis[i] * 200.0 for i in range(3)]
                elif flavour in ("inside_slow", "inside_fast"):
                    start = [p_ax[i] + nrm[i] * reach * r.uniform(0.0, 0.95) for i in range(3)]
                    goal = [start[i] + nrm[i] * (5.0 if flavour == "inside_slow" else 500.0) for i in range(3)]
                elif flavour == "graze":
                    start = [p_ax[i] + nrm[i] * (reach + r.uniform(20.0, 100.0)) for i in range(3)]
                    side = _perp(r, nrm)
                    goal = [p_ax[i] + side[i] * reach * r.uniform(0.9, 1.1) - nrm[i] * 200.0 for i in range(3)]
                else:
                    start = [p_ax[i] + nrm[i] * (reach + r.uniform(1.0, 80.0)) for i in range(3)]
                    goal = [p_ax[i] - nrm[i] * 200.0 for i in range(3)]
                if flavour == "corner":
                    start = [b[i] + axis[i] * (reach + r.uniform(5.0, 50.0)) for i in range(3)]
                    goal = [b[i] - axis[i] * 50.0 + nrm[i] * r.uniform(-reach, reach) for i in range(3)]
            else:
                out_dir = _unit(r)
                half = 0.5 * math.sqrt(sum(sum(e[j] for e in edges) ** 2 for j in range(3)))
                if flavour in ("inside_slow", "inside_fast"):
                    start = [center[i] + r.uniform(-0.3, 0.3) * sum(e[i] for e in edges) for i in range(3)]
                    goal = [start[i] + out_dir[i] * (5.0 if flavour == "inside_slow" else 600.0) for i in range(3)]
                else:
                    start = [center[i] + out_dir[i] * (half + ship_r + r.uniform(1.0, 100.0)) for i in range(3)]
                    goal = [center[i] - out_dir[i] * 300.0 + (r.uniform(-30, 30) if flavour == "graze" else 0.0)
                            for i in range(3)]
            vel = [0.0, 0.0, 0.0]
            if flavour == "backing" or r.random() < 0.3:
                d0 = [goal[i] - start[i] for i in range(3)]
                nd = math.sqrt(sum(c * c for c in d0)) or 1.0
                vel = [d0[i] / nd * speed for i in range(3)]
                if flavour == "backing":
                    goal = [start[i] - d0[i] for i in range(3)]
            if r.random() < 0.15:
                recs += _test_ship(bid, pos=start, max_vel=speed, radius=ship_r, vel=vel)
            else:
                recs.append(_ship_at(r, bid, start, radius=ship_r, max_vel=speed, vel=vel))
            recs.append(_k(r.choice([0, 1]), GOTO_OP, bid, d=goal))
        for _ in range(r.randint(0, 2)):
            k = r.randint(1, steps)
            what = r.choice(["readd_box", "readd_capsule", "free_parent", "deads", "remove_later", "remove_later"])
            if what == "remove_later" and not mini:         # moribund until brought out: it still collides
                recs.append(_k(r.randint(0, 2), REMOVE_CAPSULE_OP if capsule else REMOVE_OBOX_OP,
                               5000 if capsule else 5100, w=r.randint(2, 40)))
            elif what == "readd_box" and not capsule and not mini:
                recs.append(_k(k, OBOX_OP, 5100, 0, d=(*corner, *edges[0], *edges[1], *edges[2])))
            elif what == "readd_capsule" and capsule and not mini:
                recs.append(_k(k, CAPSULE_OP, 5000, 0, d=(*a, *b, rad)))
            elif what == "free_parent" and mini:
                recs.append(_k(k, FREE_OP, 9000, w=1))
                recs.append(_k(min(steps, k + r.randint(1, 3)), FREE_OP, 9000, w=0))
            else:
                recs.append(_k(k, DEADS_OP))
        case(out, r, "approach", recs, steps, master=master, tick=_tick(r), iterative=iterative, do=do,
             friction=r.choice([1.0e6] * 6 + [10 ** r.uniform(5.5, 6.5)]))


def corner_cases(r: random.Random, out: list, n: int) -> None:
    """Rarely reached corners: balls added again under their id; heavy ships put into warp proper at once
    (EntityWarpIn) turning faster as they speed up; formations reloaded under a leader; a leader added again; a
    missile on its target's centre; a defender missile next to a static collidable."""
    for _ in range(n):
        flavour = r.choice(["readd", "heavy_warp", "formation_reload", "on_target", "defender_static",
                            "static_bubble_edge", "equal_time", "missile_far", "mini_radius", "parallel",
                            "missile_static", "cloaked_target", "capsule_deads", "uncloak_warp", "slam",
                            "setters", "backing_capsule", "cap_hit", "formation_refused"])
        ids = _ids(r, r.randint(2, 5))
        recs = [_ball(r, bid, pos=[r.uniform(-2e3, 2e3) for _ in range(3)], interactive=True) for bid in ids]
        steps = r.randint(4, 20)
        do = r.random() < 0.3
        if flavour == "readd":
            for _ in range(r.randint(1, 3)):
                recs.append(_ball(r, r.choice(ids), step=r.randint(1, steps), free=r.random() < 0.8))
            recs.append(_k(1, FOLLOW_OP, ids[1], ids[0], d=_xyz(rr=100.0)))
        elif flavour == "heavy_warp":
            recs.append(_ball(r, 77, pos=[0.0, 0.0, 0.0], mass=10 ** r.uniform(8.3, 9.5), sf=1.0,
                              vel=[r.uniform(10, 50) for _ in range(3)]))
            v = _unit(r)
            recs.append(_k(1, WARPIN_OP, 77, w=r.randint(500, 3000), d=(v[0] * 1e10, v[1] * 1e10, v[2] * 1e10)))
            do = False
        elif flavour == "formation_reload":
            recs.append(_k(0, FORMATIONS_OP, d=(2.0, 3.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0,
                                                1.0, 50.0, 50.0, 0.0)))
            recs.append(_ball(r, 88, pos=[0.0, 0.0, 0.0], free=False))
            recs.append(_k(0, SET_FORMATION_OP, 88, w=r.choice([0, 0, 1])))
            for j, bid in enumerate(ids[:3]):
                recs.append(_k(1, FORMATION_FOLLOW_OP, bid, 88, w=j))
            k = r.randint(2, steps)
            if r.random() < 0.7:                            # fewer formations: the leader's or the slots may be gone
                recs.append(_k(k, FORMATIONS_OP, d=(1.0, 1.0, 10.0, 0.0, 0.0)))
            else:
                recs.append(_ball(r, 88, step=k, pos=[0.0, 0.0, 0.0], free=False))
        elif flavour == "on_target":
            recs.append(_ball(r, 90, pos=[0.0, 0.0, 0.0], free=False, massive=True, radius=50.0))
            recs.append(_missile_ball(r, 91, [100.0, 0.0, 0.0]))
            recs.append(_k(1, VEL_OP, 91, d=(-10.0, 0.0, 0.0)))
            recs.append(_k(1, MISSILE_FOLLOW_OP, 91, 90, ids[0]))
            for k in range(3, steps + 1):
                recs.append(_k(k, POS_OP, 91, d=(0.0, 0.0, 0.0)))
        elif flavour == "defender_static":
            a, b, rad = _capsule_shape(r, [0.0, 0.0, 0.0])
            recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*a, *b, max(rad, 10.0))))
            recs.append(_missile_ball(r, 92, [0.0, 0.0, 0.0]))
            recs.append(_k(1, LAUNCH_OP, 92, ids[1], -ids[0], w=0))
        elif flavour == "static_bubble_edge":                # a static collidable in another bubble, next door
            cx = r.randint(-2, 2)
            face = (cx + 1) * TOP
            recs = [_ball(r, ids[0], pos=[(cx - 1 + 0.5) * TOP, 0.5 * TOP, 0.5 * TOP], interactive=True)]
            c = [face + r.uniform(30.0, 150.0), 0.5 * TOP + r.uniform(-20, 20), 0.5 * TOP + r.uniform(-20, 20)]
            if r.random() < 0.5:
                aa, bb, rad = _capsule_shape(r, c)
                recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*aa, *bb, max(rad, 5.0))))
            else:
                corner, edges = _box_shape(r, c)
                recs.append(_k(0, OBOX_OP, 5100, 0, d=(*corner, *edges[0], *edges[1], *edges[2])))
            recs.append(_ship_at(r, ids[1], [face - r.uniform(5.0, 60.0), c[1], c[2]], step=0))
            recs.append(_k(1, GOTO_OP, ids[1], d=(face + 300.0, c[1], c[2])))
            do = False
        elif flavour == "equal_time":                       # inside two overlapping shapes at once
            c = [r.uniform(-500, 500) for _ in range(3)]
            aa, bb, rad = _capsule_shape(r, c)
            recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*aa, *bb, max(rad, 30.0))))
            corner, edges = _box_shape(r, c)
            recs.append(_k(0, OBOX_OP, 5100, 0, d=(*corner, *edges[0], *edges[1], *edges[2])))
            recs.append(_k(0, CAPSULE_OP, 5001, 0, d=(*c, *[c[i] + 1.0 for i in range(3)], 40.0)))
            if r.random() < 0.7:                            # and a fixed ball there (COL-715's order)
                recs.append(_ball(r, 97, pos=[c[i] + r.uniform(-5, 5) for i in range(3)], free=False, massive=True,
                                  radius=r.uniform(20.0, 60.0)))
            recs.append(_ship_at(r, ids[1], c))
            recs.append(_k(1, GOTO_OP, ids[1], d=[c[i] + r.uniform(-100, 100) for i in range(3)]))
            case(out, r, "corner", recs, steps, master=r.random() < 0.7, tick=_tick(r), do=do,
                 iterative=r.random() < 0.6)
            continue
        elif flavour == "mini_radius":                      # minicapsule radii at float's zero (a ValueError)
            recs.append(_parent(9000, pos=[0.0, 0.0, 0.0]))
            for rad in (0.0, -0.0, -1.0, 1e-46, 0.05, 2.0):
                recs.append(_k(0, MINICAPSULE_OP, 9000, d=(-20.0, 0.0, 0.0, 20.0, 0.0, 0.0, rad)))
            recs.append(_ship_at(r, ids[1], [0.0, 60.0, 0.0]))
            recs.append(_k(1, GOTO_OP, ids[1], d=(0.0, -200.0, 0.0)))
        elif flavour == "parallel":                         # velocities exactly along an axis-aligned capsule
            c = [float(r.randint(-500, 500)) for _ in range(3)]
            half = r.choice([16.0, 32.0, 64.0])
            rad = r.choice([4.0, 10.0, r.uniform(2.0, 30.0)])
            recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(c[0] - half, c[1], c[2], c[0] + half, c[1], c[2], rad)))
            for j, bid in enumerate(ids[:2]):
                off = r.choice([0.0, 0.0, 0.5 * rad, 0.9 * rad])
                side = r.choice([-1.0, 1.0])
                start = [c[0] + side * (half + rad + r.uniform(5.0, 60.0)), c[1] + off, c[2]]
                if r.random() < 0.3:                        # at rest against the capsule
                    start = [c[0] + r.uniform(-half, half), c[1] + rad + r.uniform(0.0, 3.0), c[2]]
                    recs.append(_ship_at(r, bid, start, vel=[0.0, 0.0, 0.0]))
                    continue
                recs.append(_ship_at(r, bid, start, vel=[-side * r.uniform(5.0, 50.0), 0.0, 0.0]))
                recs.append(_k(1, GOTO_OP, bid, d=(c[0] - side * 500.0, start[1], start[2])))
            do = False
        elif flavour == "missile_static":                   # missiles flying past static shapes (gradient method)
            c = [r.uniform(-300, 300) for _ in range(3)]
            if r.random() < 0.5:
                aa, bb, rad = _capsule_shape(r, c)
                recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*aa, *bb, max(rad, 20.0))))
            else:
                corner, edges = _box_shape(r, c)
                recs.append(_k(0, OBOX_OP, 5100, 0, d=(*corner, *edges[0], *edges[1], *edges[2])))
            recs[0] = _ball(r, ids[0], pos=[c[0] + 800.0, c[1], c[2]], interactive=True, free=False)
            recs.append(_missile_ball(r, 94, [c[0] - 300.0, c[1], c[2]], massive=True))
            recs.append(_k(1, LAUNCH_OP, 94, ids[0], r.choice([ids[1], -ids[1], -ids[1], -ids[1]]),
                           w=r.choice([0, 1, 2, 3])))
            if r.random() < 0.5:                            # a defender (negative owner) sees static shapes
                recs.append(_missile_ball(r, 95, [c[0], c[1] - 300.0, c[2]]))
                recs.append(_k(1, VEL_OP, 95, d=(0.0, 50.0, 0.0)))
                recs.append(_k(1, MISSILE_FOLLOW_OP, 95, ids[0], r.choice([ids[1], -ids[1]])))
            for iterative in (False, True):                 # the same scenario in both methods
                case(out, r, "corner", list(recs), steps, master=r.random() < 0.7, tick=_tick(r), do=do,
                     iterative=iterative)
            continue
        elif flavour == "cloaked_target":                   # missiles sent at a cloaked ball
            recs.append(_k(1, CLOAK_OP, ids[0], w=r.choice([1, 1, 2, 3])))
            recs.append(_missile_ball(r, 96, [r.uniform(-500, 500) for _ in range(3)]))
            recs.append(_k(r.choice([1, 2]), r.choice([MISSILE_FOLLOW_OP, LAUNCH_OP]), 96, ids[0], ids[1],
                           w=r.choice([0, 1])))
            recs.append(_k(r.randint(2, steps), UNCLOAK_OP, ids[0]))
            recs.append(_k(r.randint(2, steps), MISSILE_FOLLOW_OP, 96, ids[0], ids[1]))
            k = r.randint(2, steps)                         # MissileFollow's refusals (MOV-724)
            how = r.choice(["same_src_dst", "same_src_owner", "missing_target", "missing_missile", "negative_target",
                            "moribund_target"])
            if how == "same_src_dst":
                recs.append(_k(k, MISSILE_FOLLOW_OP, 96, 96, ids[1]))
            elif how == "same_src_owner":
                recs.append(_k(k, MISSILE_FOLLOW_OP, 96, ids[0], 96))
            elif how == "missing_target":
                recs.append(_k(k, MISSILE_FOLLOW_OP, 96, 999999937, ids[1]))
            elif how == "missing_missile":
                recs.append(_k(k, MISSILE_FOLLOW_OP, 999999937, ids[0], ids[1]))
            elif how == "negative_target":
                recs.append(_k(k, MISSILE_FOLLOW_OP, 96, -ids[0], ids[1]))
            else:
                recs.append(_k(k, REMOVE_OP, ids[0], w=3))
                recs.append(_k(k, MISSILE_FOLLOW_OP, 96, ids[0], ids[1]))
        elif flavour == "capsule_deads":                    # capsules removed later, then brought out (at most 7:
            for j in range(r.randint(2, 7)):                # which of more would go first is Destiny's heap order)
                cj = [r.uniform(-1500, 1500) for _ in range(3)]
                aa, bb, rad = _capsule_shape(r, cj)
                recs.append(_k(0, CAPSULE_OP, 5000 + j, 0, d=(*aa, *bb, max(rad, 2.0))))
                recs.append(_k(r.randint(1, 3), REMOVE_CAPSULE_OP, 5000 + j, w=r.choice([1, 2, 2, 3, 4])))
            for k in range(2, steps + 1):
                if r.random() < 0.6:
                    recs.append(_k(k, DEADS_OP))
        elif flavour == "uncloak_warp":                     # a cloak lifted in warp proper
            recs.append(_ball(r, 78, pos=[0.0, 0.0, 0.0], mass=10 ** r.uniform(6.0, 7.5), sf=1.0, interactive=True))
            v = _unit(r)
            recs.append(_k(1, WARPIN_OP, 78, w=r.randint(500, 3000), d=(v[0] * 1e10, v[1] * 1e10, v[2] * 1e10)))
            k = r.randint(1, max(1, steps - 2))
            recs.append(_k(k, CLOAK_OP, 78, w=r.choice([1, 2])))
            recs.append(_k(k + r.randint(0, 2), UNCLOAK_OP, 78))
            do = False
        elif flavour == "slam":                             # a heavy fast ship into slow light ones (iterative)
            c = [r.uniform(-300, 300) for _ in range(3)]
            recs = []
            for j, bid in enumerate(ids):
                recs.append(_ship_at(r, bid, [c[0] + 25.0 * j, c[1] + r.uniform(-3, 3), c[2] + r.uniform(-3, 3)],
                                     max_vel=r.choice([0.5, 1.0, 2.0]), mass=1e6, radius=10.0,
                                     vel=[0.0, 0.0, 0.0]))
            recs.append(_ship_at(r, 79, [c[0] - 200.0, c[1], c[2]], mass=1e10, max_vel=400.0, radius=20.0,
                                 vel=[300.0, 0.0, 0.0]))
            recs.append(_k(1, GOTO_OP, 79, d=(c[0] + 2000.0, c[1], c[2])))
            if r.random() < 0.4:
                aa, bb, rad = _capsule_shape(r, [c[0] + 25.0 * len(ids) + 40.0, c[1], c[2]])
                recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*aa, *bb, max(rad, 10.0))))
            case(out, r, "corner", recs, steps, master=r.random() < 0.7, tick=_tick(r), do=do, iterative=True)
            continue
        elif flavour == "backing_capsule":                  # moving onto a capsule while thrusting away from it
            c = [r.uniform(-300, 300) for _ in range(3)]
            axis = _unit(r)
            half, rad = r.uniform(10.0, 100.0), r.uniform(3.0, 40.0)
            recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*[c[i] - half * axis[i] for i in range(3)],
                                                      *[c[i] + half * axis[i] for i in range(3)], rad)))
            nrm = _perp(r, axis)
            for bid in ids[:2]:
                ship_r = r.uniform(1.0, 20.0)
                start = [c[i] + axis[i] * r.uniform(-half, half) + nrm[i] * (rad + ship_r + r.uniform(0.5, 15.0))
                         for i in range(3)]
                speed = r.uniform(10.0, 80.0)
                recs.append(_ship_at(r, bid, start, radius=ship_r, max_vel=speed,
                                     vel=[-nrm[i] * speed for i in range(3)]))
                recs.append(_k(1, GOTO_OP, bid, d=[start[i] + nrm[i] * 1000.0 for i in range(3)]))
                nrm = [-x for x in nrm]
            case(out, r, "corner", recs, steps, master=r.random() < 0.7, tick=_tick(r), do=do, iterative=False)
            continue
        elif flavour == "cap_hit":                          # across the axis line beyond an end: the cap is hit
            c = [r.uniform(-300, 300) for _ in range(3)]
            axis = _unit(r)
            half, rad = r.uniform(10.0, 100.0), r.uniform(3.0, 40.0)
            recs.append(_k(0, CAPSULE_OP, 5000, 0, d=(*[c[i] - half * axis[i] for i in range(3)],
                                                      *[c[i] + half * axis[i] for i in range(3)], rad)))
            for bid in ids[:2]:
                ship_r = r.uniform(1.0, 20.0)
                nrm, end = _perp(r, axis), r.choice([-1.0, 1.0])
                beyond = r.uniform(0.05, 0.9) * (rad + ship_r)
                start = [c[i] + axis[i] * end * (half + beyond) + nrm[i] * (rad + ship_r + r.uniform(1.0, 40.0))
                         for i in range(3)]
                speed = r.uniform(20.0, 120.0)
                recs.append(_ship_at(r, bid, start, radius=ship_r, max_vel=speed,
                                     vel=[-nrm[i] * speed for i in range(3)]))
                recs.append(_k(1, GOTO_OP, bid, d=[start[i] - nrm[i] * 1000.0 for i in range(3)]))
            case(out, r, "corner", recs, steps, master=r.random() < 0.7, tick=_tick(r), do=do,
                 iterative=r.random() < 0.7)
            continue
        elif flavour == "formation_refused":                # a leader in another bubble or cloaked (SEN-708)
            recs.append(_k(0, FORMATIONS_OP, d=(1.0, 2.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0)))
            far = r.random() < 0.5
            recs.append(_ball(r, 89, pos=[5 * TOP, 0.0, 0.0] if far else [0.0, 0.0, 0.0], free=False))
            recs.append(_k(0, SET_FORMATION_OP, 89, w=0))
            if not far:
                recs.append(_k(0, CLOAK_OP, 89, w=r.choice([1, 2, 3])))
            for j, bid in enumerate(ids[:2]):
                recs.append(_k(1, FORMATION_FOLLOW_OP, bid, 89, w=j))
            if not far and r.random() < 0.5:
                recs.append(_k(2, UNCLOAK_OP, 89))
                recs.append(_k(3, FORMATION_FOLLOW_OP, ids[0], 89, w=0))
            do = True
        elif flavour == "setters":                          # AddBall's clamps and the setters' refusals (MOV-711/713)
            recs = []
            for bid in ids:
                free = r.random() < 0.7
                recs.append(_ball(r, bid, pos=[r.uniform(-1e3, 1e3) for _ in range(3)], free=free,
                                  radius=r.choice([-5.0, -0.0, 0.0, 3.0, 40.0]),
                                  mass=r.choice([-1.0, 1e7, 3e8] + ([] if free else [0.0])),
                                  max_vel=r.choice([-10.0, 0.0, 50.0, 200.0]),
                                  sf=r.choice([-0.5, 0.0, 1.7, 1.0, 0.4]), agility=r.choice([-1.0, 0.0, 0.7]),
                                  vel=r.choice([[0.0, 0.0, 0.0], [r.uniform(-30, 30) for _ in range(3)]])))
                recs.append(_k(1, GOTO_OP, bid, d=[r.uniform(-2e3, 2e3) for _ in range(3)]))
            for _ in range(r.randint(3, 10)):
                k, bid = r.randint(1, steps), r.choice(ids)
                op = r.choice([RADIUS_OP, MAXVEL_OP, MASS_OP, AGILITY_OP, SF_OP, SF_OP, POS_OP, VEL_OP])
                if op == RADIUS_OP:
                    v = r.choice([-1.0, -0.0, 0.0, 7.0])
                elif op == MAXVEL_OP:
                    v = r.choice([-5.0, 0.0, 80.0])
                elif op == MASS_OP:
                    v = r.choice([-1e6, 0.0, 2e7])
                elif op == AGILITY_OP:
                    v = r.choice([-0.5, 0.0, 0.3])
                elif op == SF_OP:
                    v = r.choice([math.nan, math.inf, -math.inf, -0.2, 1.5, 0.6])
                if op in (POS_OP, VEL_OP):
                    recs.append(_k(k, op, bid, d=[r.uniform(-50, 50) for _ in range(3)]))
                else:
                    recs.append(_k(k, op, bid, d=_xyz(rr=v)))
            do = r.random() < 0.2
        else:                                               # a missile told to follow a target in another bubble
            recs.append(_missile_ball(r, 93, [5 * TOP, 0.0, 0.0]))
            recs.append(_k(1, VEL_OP, 93, d=(-50.0, 0.0, 0.0)))
            recs.append(_k(1, MISSILE_FOLLOW_OP, 93, ids[1], ids[0]))
        case(out, r, "corner", recs, steps, master=r.random() < 0.7, tick=_tick(r), do=do,
             iterative=r.random() < 0.3)


FAMILY_SIZES = {"missile": 220, "static": 200, "approach": 300, "cloak": 120, "orient": 90, "orient_do": 150,
                "mushroom": 70, "troll": 80, "formation": 200, "accuracy": 70, "ontick": 110, "corner": 440,
                "mixed": 170}


def cases(seed: int, sizes: dict | None = None) -> list:
    sizes = dict(FAMILY_SIZES, **(sizes or {}))
    r = random.Random(3000 + seed)
    out: list = []
    missile_cases(r, out, sizes["missile"])
    static_cases(r, out, sizes["static"])
    approach_cases(r, out, sizes["approach"])
    cloak_cases(r, out, sizes["cloak"])
    orientation_cases(r, out, sizes["orient"], do=False)
    orientation_cases(r, out, sizes["orient_do"], do=True)
    mushroom_cases(r, out, sizes["mushroom"])
    troll_cases(r, out, sizes["troll"])
    formation_cases(r, out, sizes["formation"])
    accuracy_cases(r, out, sizes["accuracy"])
    ontick_cases(r, out, sizes["ontick"])
    corner_cases(r, out, sizes["corner"])
    mixed_cases(r, out, sizes["mixed"])
    return out

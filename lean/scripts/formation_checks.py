# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Compare formation proofs with the scalar command and slot APIs."""
from exact_ballpark.full import DestinyUndefined, FullBallpark
from exact_ballpark.modes import FORMATION, STOP


def ship(park, bid):
    return park.add_ball(bid, 1, 1, 1, True, agility=1)


def layout(size):
    return [(0, 0, 0)] * size


def allocate(kind, fields):
    size, arg, *expected = map(int, fields)
    park = FullBallpark()
    leader = ship(park, 1)
    park.load_formations([layout(size)])
    park.set_ball_formation(1, 0)
    if kind == "reserve":
        leader.formation_slots = arg
        idx = park.reserve_formation_slot(1)
        result = [idx, leader.formation_slots]
        if idx >= 0:
            park.free_formation_slot(1, idx)
            if leader.formation_slots != arg:
                raise SystemExit(f"Reservation did not round-trip: size {size}, mask {arg}, slot {idx}")
    else:
        leader.formation_slots = 65535
        try:
            park.free_formation_slot(1, arg)
        except DestinyUndefined as error:
            if "bitset<16>::reset out of range" not in str(error) or leader.formation_slots != 65535:
                raise
            result = [-1]
        else:
            result = [leader.formation_slots]
    return result, expected


def join(fields):
    scenario, n, raw, same, cloak = map(int, fields[:5])
    mode, target, stamp = fields[5:]
    park = FullBallpark(is_master=True)
    follower, leader, previous = [ship(park, i) for i in range(1, 4)]
    park.load_formations([layout(n), layout(1)])
    park.set_ball_formation(previous.id, 1)
    park.formation_follow(follower.id, previous.id, park.reserve_formation_slot(previous.id))
    if scenario != 4:
        park.set_ball_formation(leader.id, 0)
    leader.bubble = follower.bubble + (not same)
    leader.is_cloaked = cloak
    src = 9 if scenario == 2 else follower.id
    dst = {1: follower.id, 3: 9}.get(scenario, leader.id)
    park.formation_follow(src, dst, raw)
    result = (follower.mode, follower.follow_ptr, follower.follow_id, follower.effect_stamp,
              follower.id in previous.followers, previous.formation_slots,
              follower.id in leader.followers, leader.formation_slots)
    target = int(target)
    expected = (STOP if mode == "stop" else FORMATION, target or None, target, int(stamp),
                target == previous.id, int(target == previous.id), target == leader.id, 0)
    return result, expected


def resize(fields):
    master, size, word, *expected = map(int, fields)
    park = FullBallpark(is_master=bool(master))
    leader, *followers = [ship(park, i) for i in range(1, 6)]
    park.load_formations([layout(4), layout(size)])
    park.set_ball_formation(leader.id, 0)
    for ball, slot in zip(followers, (3, 0, 3)):
        park.formation_follow(ball.id, leader.id, slot)
    park.follow_ball(followers[-1].id, leader.id)
    leader.formation_slots = word
    park.set_ball_formation(leader.id, 1)
    stopped = sum(1 << f.id for f in followers if f.mode == STOP)
    for f in followers:
        if (f.id in leader.followers) != (f.mode != STOP):
            raise SystemExit(f"Resize left an inconsistent follow link for ship {f.id}")
    bits = leader.formation_slots
    park.set_ball_formation(leader.id, 0)
    back = leader.formation_slots
    allocated = park.reserve_formation_slot(leader.id)
    return [stopped, bits, back, allocated, leader.formation_slots], expected


def compare(run_lake):
    rows = run_lake("env", "lean", "--run", "scripts/FormationProbe.lean", capture=True).splitlines()
    counts = {"reserve": 0, "free": 0, "join": 0, "resize": 0}
    for row in rows:
        kind, *fields = row.split()
        counts[kind] += 1
        if kind in ("reserve", "free"):
            result, expected = allocate(kind, fields)
        elif kind == "join":
            result, expected = join(fields)
        else:
            result, expected = resize(fields)
        if result != expected:
            raise SystemExit(f"Formation mismatch on {row}: scalar {result}, Lean {expected}")
    if not all(counts.values()):
        raise SystemExit(f"Missing formation probe cases: {counts}")
    return counts

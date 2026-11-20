# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Compare registered sensor and distribution projections with scalar engine calls."""
from collections import Counter
from types import SimpleNamespace


def check_uncloak():
    from exact_ballpark.full import FullBallpark
    from exact_ballpark.modes import WARP

    for master in (False, True):
        for mode in (1, 2):
            for stamp in (-1, 0):  # aligning and active warp
                park = FullBallpark(is_master=master)
                ship, live, distant, dying = [
                    park.add_ball(i, 1, 1, 1, True, agility=1) for i in range(1, 5)]
                for observer in (ship, live, distant, dying):
                    park.add_proximity_sensor(observer.id, 10, 1)
                    observer.sensor_balls.set(ship.id, 0)
                park.cloak_ball(ship.id, mode)
                ship.mode, ship.effect_stamp = WARP, stamp
                distant.bubble = ship.bubble + 1
                dying.is_moribund = True
                park.uncloak_ball(ship.id)
                observed = (ship.is_cloaked, ship.is_massive, ship.sensor_active,
                            ship.id in live.sensor_balls, ship.id in distant.sensor_balls,
                            ship.id in dying.sensor_balls)
                expected = (0, stamp < 0, not (master and mode == 1), stamp >= 0, True, True)
                if observed != expected:
                    raise SystemExit(f"Uncloak master={master}, mode={mode}, stamp={stamp}: {observed} != {expected}")
                if stamp < 0:
                    park._check_for_proximity(live)
                    if (park.current_time, "DoProximity", live.id, ship.id, 1) not in park.events:
                        raise SystemExit("Uncloaked ship did not re-enter the cleared observer")


def pilot_transition(bid, flags, previous, current):
    from exact_ballpark.server import DestinyPark

    pilot = SimpleNamespace(id=1, old_bubble=11 if previous else -1,
                            bubble=12 if current else -1, mode=0)
    balls = {1: pilot}
    if flags:
        balls[bid] = SimpleNamespace(is_global=flags in (3, 4), is_cloaked=flags in (2, 4))
    bubbles = {}
    for bubble, code in [(11, previous), (12, current)]:
        if code:
            bubbles[bubble] = {} if code == 1 else {bid: code - 3}
    park = DestinyPark(SimpleNamespace(bubbles=bubbles, balls=balls, bubble_interactives={}))
    added, removed = {}, {}
    park.AdditionsAndDeletions(added, removed, {}, {}, [1])
    return added[1], removed[1]


def history_route(bid, flags, known, interactive, character):
    from exact_ballpark.server import Ticker

    balls = {bid: SimpleNamespace(newBubbleId=12, isCloaked=flags == 2)} if flags else {}
    park = SimpleNamespace(currentTime=1, balls=balls, bubbles={11, 12} if known else {11},
                           bubbleInteractives={11, 12} if interactive else {11})
    info = SimpleNamespace(get_character_for_ball=lambda _: character if character >= 0 else None)
    sent = []
    batcher = SimpleNamespace(add_to_bubble_history=lambda b, e: sent.append((f"bubble/{b}", e)),
                              add_to_character_history=lambda c, e: sent.append((f"character/{c}", e)))
    ticker = Ticker(park, info, None, batcher)
    event = ("RemoveBall", (bid, 11))
    ticker.current_system_history = [(bid, 1, event)]
    ticker._apply_history()
    return sent


def compare(run_lake):
    from exact_ballpark.full import FullBallpark

    check_uncloak()
    park = FullBallpark()
    owner = park.add_ball(3, 1, 1, 1, True, agility=1)
    park.add_ball(4, 1, 1, 1, True, agility=1, x=12)
    park.add_proximity_sensor(3, 10, 1.5)
    rows = run_lake("env", "lean", "--run", "scripts/VisibilityProbe.lean", capture=True).splitlines()
    count = {"sensor": 0, "route": 0, "pilot": 0}
    for row in rows:
        kind, *fields = row.split()
        count[kind] += 1
        if kind == "sensor":
            x, member, event = fields
            park.set_position(4, int(x), 0, 0)
            park.events.clear()
            park.trace = Counter()
            park._check_for_proximity(owner)
            if owner.sensor_elapsed != 1 or park.events:
                raise SystemExit("Sensor scanned before its due time or carried overshoot")
            park._check_for_proximity(owner)
            observed = [e[4] for e in park.events if e[:4] == (park.current_time, "DoProximity", 3, 4)]
            expected = [] if event == "quiet" else [int(event)]
            if event == "0":
                branch = "sensor.vanished" if int(x) == 1073741824 else "sensor.left"
                if park.trace[branch] != 1:
                    raise SystemExit(f"Expected target exit through {branch} at x={x}")
            if observed != expected or (4 in owner.sensor_balls) != bool(int(member)) or owner.sensor_elapsed != 0:
                raise SystemExit(f"Sensor mismatch: {row}, observed events {observed}")
        elif kind == "pilot":
            bid, flags, old, new, add, remove = map(int, fields)
            actual = pilot_transition(bid, flags, old, new)
            if actual != ([bid] * add, [bid] * remove):
                raise SystemExit(f"Bubble transition mismatch: {row}, observed {actual}")
        else:
            bid, flags, known, interactive, character = map(int, fields[:-1])
            actual = history_route(bid, flags, known, interactive, character)
            expected = [] if fields[-1] == "drop" else [(fields[-1], (1, ("RemoveBall", (bid, 11))))]
            if actual != expected:
                raise SystemExit(f"History routing mismatch: {row}, observed {actual}")
    if count != {"sensor": 6, "pilot": 375, "route": 108}:
        raise SystemExit(f"Incomplete visibility probe: {count}")
    return count

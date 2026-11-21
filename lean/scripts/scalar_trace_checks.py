# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Compare every observable scalar field after each command and tick of short API traces."""
import math
import random
import struct
import subprocess
from pathlib import Path

from exact_ballpark.ref import Ballpark


def bits(x):
    return int.from_bytes(struct.pack(">d", x), "big")


def is_nan_word(w):
    return (w & 0x7fff_ffff_ffff_ffff) > 0x7ff0_0000_0000_0000


FIELDS = ("mass", "radius", "max_vel", "agility", "speed_fraction", "time_factor",
          *(f"{name}.{axis}" for name in ("pos", "vel", "goto", "last_g") for axis in "xyz"))


def snapshot(park):
    balls = []
    for key in sorted(park.balls):
        b = park.balls[key]
        if key != b.id:
            raise SystemExit(f"Scalar ball ID {b.id} differs from its lookup key {key}")
        values = (b.mass, b.radius, b.max_vel, b.agility, b.speed_fraction, b.time_factor,
                  *b.pos, *b.vel, *b.goto, *b.last_g)
        balls.append((b.id, b.mode, tuple(map(bits, values))))
    return park.current_time, balls


def compare_row(where, row, expected):
    if expected is None:
        if row != "outside":
            raise SystemExit(f"{where}: Lean admitted a trace where the scalar call raised")
        return
    if row == "outside":
        raise SystemExit(f"{where}: Lean rejected an admitted scalar trace")
    stamp, *encoded = row.split("|")
    clock, balls = expected
    if int(stamp) != clock or len(encoded) != len(balls):
        raise SystemExit(f"{where}: clock or number of balls differs from scalar {clock}, {len(balls)}")
    for raw, (bid, mode, words) in zip(encoded, balls, strict=True):
        observed_id, observed_mode, *actual = map(int, raw.split(","))
        if (observed_id, observed_mode) != (bid, mode) or len(actual) != len(FIELDS):
            raise SystemExit(f"{where}: expected ball {bid}, mode {mode}, {len(FIELDS)} scalar fields")
        for name, got, want in zip(FIELDS, actual, words, strict=True):
            if got != want and not (is_nan_word(got) and is_nan_word(want)):
                raise SystemExit(f"{where}: ball {bid} {name}, Lean {got:#018x}, scalar {want:#018x}")


def add(bid, *, mass=2e6, radius=7., max_vel=81.5, pos=(5., 1., -3.),
        vel=(13., -8., 0.5), agility=1.25, fraction=0.75):
    return ("add", bid, mass, radius, max_vel, *pos, *vel, agility, fraction)


TICK = ("tick",)


def coast_events():
    ball = add(1, mass=1., radius=1., max_vel=100., pos=(0., 0., 0.), vel=(100., 0., 0.),
               agility=1., fraction=1.)
    return [ball] + [TICK] * 5


def scenarios():
    nan, inf = math.nan, math.inf
    big = 2**72 + 3
    actors = [add(big, fraction=-0.), add(-7), add(0, fraction=1.75)]
    flight = [
        ("point", -7, 11., -5., 9.), ("direction", big, -3., 4., -0.), TICK, TICK,
        ("agility", big, 2.5), ("mass", -7, 1e6), ("max_velocity", 0, 13.125),
        ("speed_fraction", 0, 4.), ("radius", 0, 12.25), ("point", 0, 21., -1., 0.),
        TICK, ("stop", -7), TICK, ("point", big, nan, 0., 0.), TICK,
        ("agility", -7, -2.), ("mass", big, -0.), ("radius", 0, -1.),
        ("max_velocity", -7, -1.), ("speed_fraction", 0, inf),
        ("speed_fraction", -7, -1.), ("point", -7, -0., 0., 5.),
        TICK, add(big, max_vel=0., pos=(-0., 2., -1.)), TICK, TICK,
        ("direction", 0, 0., 0., -0.), ("direction", 0, inf, 1., 2.), TICK,
    ]
    yield "interleaved", 820000., 137, [TICK, *actors, *flight], None
    yield "reversed insertion", 820000., 137, [TICK, *reversed(actors), *flight], None
    yield "five-second coast", 1., 1000, coast_events(), None
    yield "zero duration", 1., 0, [TICK, add(1, pos=(-0., 0., 0.), vel=(-0., -18., 3.)),
        TICK, TICK, ("point", 1, 0., -1., -0.), TICK, ("stop", 1), TICK], None
    yield "special branch", 1., 1000, [
        add(3, mass=2.**40, agility=1., fraction=1., max_vel=2.**60,
            pos=(0., 0., 0.), vel=(1., -0., 0.)), ("point", 3, 2.**65, 0., -0.),
        TICK, TICK, ("mass", 3, 1.), ("stop", 3), TICK], None
    halfway = 1. + 2.**-24
    yield "storage and guards", 1., 250, [
        add(9, mass=-0., radius=-0., max_vel=1e39, agility=-1., fraction=2.), TICK,
        ("agility", 9, 2.**-150), ("agility", 9, math.nextafter(2.**-150, inf)),
        ("mass", 9, 5.), ("max_velocity", 9, halfway),
        ("radius", 9, math.nextafter(halfway, inf)), ("speed_fraction", 9, -0.),
        ("point", 9, -0., 0., -0.), TICK, ("agility", 9, nan), TICK,
        ("agility", 9, 1.), ("mass", 9, nan), ("speed_fraction", 9, nan), TICK,
        add(9, mass=0., radius=nan, max_vel=-0., agility=-0., fraction=nan), TICK], None
    rng = random.Random(701)
    cmds = [add(bid) for bid in (2, 0, -1)]
    for _ in range(45):
        bid = rng.choice((2, 0, -1))
        op = rng.choice(("point", "direction", "mass", "agility", "speed_fraction", "stop", "tick"))
        if op in ("point", "direction"):
            cmds.append((op, bid, *(rng.uniform(-95., 95.) for _ in range(3))))
        elif op == "tick":
            cmds.append(TICK)
        elif op == "stop":
            cmds.append((op, bid))
        else:
            cmds.append((op, bid, rng.uniform(0.2, 3.) * (1e6 if op == "mass" else 1.)))
        cmds.append(TICK)
    yield "seeded flight", 1e6, 1000, cmds, None
    yield "missing ID", 1., 1000, [TICK, ("stop", 42)], KeyError
    yield "zero GOTO denominator", 1., 1000, [add(4, mass=0.), ("point", 4, 1., 2., 3.), TICK], ZeroDivisionError
    yield "setter overflow", 1., 1000, [add(1), ("agility", 1, 1e39)], OverflowError


def apply(park, event):
    if event == TICK:
        park.evolve()
        return
    op, bid, *args = event
    if op == "add":
        m, r, v, px, py, pz, vx, vy, vz, a, f = args
        park.add_ball(bid, m, r, v, [px, py, pz], [vx, vy, vz], a, f)
    elif op == "point":
        park.goto_point(bid, *args)
    elif op == "direction":
        park.goto_direction(bid, *args)
    elif op == "stop":
        park.stop(bid)
    else:
        getattr(park, "set_" + op)(bid, *args)


def compare(run_lake):
    inputs, observations = [], []
    coast = None
    prefixes = rejected = 0
    for name, friction, milliseconds, events, error in scenarios():
        park = Ballpark(friction, milliseconds)
        inputs.append(f"park,{bits(friction)},{bits(park.dt)}")
        observations.append((name + " at creation", snapshot(park)))
        for i, event in enumerate(events, 1):
            inputs.append(",".join([*map(str, event[:2]), *(str(bits(x)) for x in event[2:])]))
            expected_error = error if i == len(events) else None
            where = f"{name}, action {i} ({event[0]})"
            try:
                apply(park, event)
            except (KeyError, ZeroDivisionError, OverflowError) as exc:
                if type(exc) is not expected_error:
                    raise SystemExit(f"{where}: unexpected scalar {type(exc).__name__}") from exc
                observations.append((where, None))
                rejected += 1
            else:
                if expected_error is not None:
                    raise SystemExit(f"{where}: expected scalar {expected_error.__name__}")
                observations.append((where, snapshot(park)))
                prefixes += 1
        if name == "five-second coast":
            b = park.balls[1]
            coast = (bits(b.time_factor), *b.pos[0].as_integer_ratio(), *b.vel[0].as_integer_ratio())
    rows = run_lake("env", "lean", "--run", "scripts/ScalarTraceProbe.lean", *inputs, "coast",
                    capture=True).splitlines()
    if not rows or tuple(map(int, rows.pop().split(","))) != coast:
        raise SystemExit("The scalar coast differs from the certified decay or five-second endpoint")
    if len(rows) != len(observations):
        raise SystemExit(f"Expected {len(observations)} scalar trace rows, received {len(rows)}")
    for row, (where, expected) in zip(rows, observations, strict=True):
        compare_row(where, row, expected)
    return prefixes, rejected


def compare_coast_harness(harness):
    park = Ballpark(1., 1000)
    create, *ticks = coast_events()
    apply(park, create)
    b = park.balls[1]
    fields = (b.id, park.friction, park.tick_interval, len(ticks), b.mode, b.mass, b.agility,
              b.max_vel, b.speed_fraction, *b.pos, *b.vel, *b.goto)
    result = subprocess.run([harness], input=" ".join(map(str, fields)) + "\n",
                            capture_output=True, text=True, timeout=10)
    rows = result.stdout.splitlines()
    if result.returncode or len(rows) != len(ticks):
        raise SystemExit(f"Movement harness returned {len(rows)} ticks with exit code {result.returncode}")
    for stamp, (action, row) in enumerate(zip(ticks, rows, strict=True), 1):
        apply(park, action)
        bid, step, *motion = row.split()
        if ((int(bid), int(step)) != (b.id, stamp) or
                [bits(float(x)) for x in motion] != [bits(x) for x in (*b.pos, *b.vel)]):
            raise SystemExit(f"Compiled coast differs from scalar at tick {stamp}")
    print(f"All six motion words match the compiled movement harness at each of {len(ticks)} ticks.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check the five-second coast with a built Destiny movement harness.")
    parser.add_argument("harness", type=Path)
    compare_coast_harness(parser.parse_args().harness.resolve())

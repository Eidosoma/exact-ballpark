# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Check the curve-clearance CLI against the scalar GOTO tick and exact rational chord geometry."""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import struct

from exact_ballpark.ref import Ballpark
from exact_ballpark.space import SpaceBallpark


def bits(x):
    return int.from_bytes(struct.pack(">d", x), "big")


def turning_ship():
    park = Ballpark(friction=1.0, tick_interval_ms=1000)
    start = [0.0, 0.0, 0.0]
    ship = park.add_ball(1, 1.0, 0.25, 10.0, start, [10.0, 0.0, 0.0], 1.0, 1.0)
    park.goto_point(1, 0.0, 100.0, 0.0)
    acceleration = park.goto_thrust(ship)
    return park, ship, start, acceleration


def partial(park, ship, start, acceleration, time):
    return park.integrate(start, ship.vel, acceleration, ship.mass * ship.agility,
                          park.friction, ship.time_factor, time)[0]


def turning_sweep():
    park, ship, start, acceleration = turning_ship()
    mid = partial(park, ship, start, acceleration, 0.5)
    park.evolve()
    targets = [(4.0, y, 0.0) for y in (1.0, -4.0)]
    times = [SpaceBallpark.collide_two_spheres(start, ship.pos, target, target, 0.5)
             for target in targets]
    return ship.time_factor, acceleration, mid, ship.pos, times


def turning_edges():
    state = turning_ship()
    return [partial(*state, t) for t in (0.435661, 0.435662)]


def second_turn():
    """The other moving sphere has a two-second drag time and downward thrust."""
    park = Ballpark(friction=1., tick_interval_ms=1000)
    start, velocity, acceleration = [4., 1., 0.], [-.5, -.25, 0.], [0., -1., 0.]
    ball = park.add_ball(2, 2., .25, 10., start, velocity, 1., 1.)
    finish = partial(park, ball, start, acceleration, 1.)
    return start, finish, velocity, acceleration


def turning_bracket():
    """Independent high-precision root estimate and scalar partial-step comparisons."""
    with localcontext() as ctx:
        ctx.prec = 60
        def gap(t):
            x = 10 * (1 - (-t).exp())
            y = 10*t - x
            return (x-4)**2 + (y-1)**2 - Decimal("0.25")

        lo, hi = Decimal("0.435661"), Decimal("0.435662")
        for t, position, sign in zip((lo, hi), turning_edges(), (1, -1), strict=True):
            measured = sum((Decimal.from_float(v)-c)**2 for v, c in zip(position, (4, 1, 0), strict=True))
            if gap(t)*sign <= 0 or abs(measured-Decimal("0.25")-gap(t)) > Decimal("1e-13"):
                raise SystemExit(f"Scalar partial tick differs from the continuous turning fixture at {t} s")
        for _ in range(100):
            mid = (lo+hi)/2
            if gap(mid) > 0:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2


def closest_squared(p0, p1, q0, q1):
    """Exact squared distance at the clipped projection onto the relative endpoint chord."""
    start = [Fraction(x) - Fraction(y) for x, y in zip(p0, q0, strict=True)]
    finish = [Fraction(x) - Fraction(y) for x, y in zip(p1, q1, strict=True)]
    delta = [y-x for x, y in zip(start, finish, strict=True)]
    norm_sq = sum(w*w for w in delta)
    u = min(1, max(0, -sum(x*w for x, w in zip(start, delta, strict=True)) / norm_sq)) if norm_sq else 0
    return sum((x+u*w)**2 for x, w in zip(start, delta, strict=True))


def lower_metres(squared):
    return Fraction(math.isqrt(squared.numerator * 10**12 // squared.denominator), 10**6)


def log_measurements():
    _, _, mid, end, _ = turning_sweep()
    start, near, clear = (0., 0., 0.), (4., 1., 0.), (4., -4., 0.)
    margin = lower_metres(closest_squared(start, end, clear, clear)) - Fraction(1, 2) - Fraction(15, 8) - Fraction(1, 10**12)
    return [f"{math.dist(mid, near):.6f} m", f"{math.sqrt(closest_squared(start, end, near, near)):.6f} m",
            f"{float(margin):.12f} m", f"{turning_bracket():.12f} s"]


def compare(run_lake):
    cache, acceleration, mid, end, times = turning_sweep()
    if acceleration != [0., 10., 0.] or math.dist(mid, [4., 1., 0.]) >= 0.1:
        raise SystemExit("The scalar GOTO fixture no longer passes within 0.1 m at half a tick")
    demo = run_lake("env", "lean", "--run", "scripts/CurveClearance.lean", "--demo", capture=True)
    for index, (row, time) in enumerate(zip(demo.splitlines(), times, strict=True)):
        entry = json.loads(row)
        if (entry["cacheBits"], entry["endBits"], entry["sweepTimeBits"]) != (bits(cache), list(map(bits, end)), bits(time)):
            raise SystemExit("Lean and the scalar turning sweep disagree on a cache, endpoint or contact word")
        margin = entry.get("marginMetres")
        if (index == 0 and margin is not None) or (index == 1 and (margin is None or Fraction(margin) <= 3)):
            raise SystemExit("The scalar turning fixture or its clear control lost its expected certificate")

    def body(start, finish, accel=0, start_error=0, end_error=0):
        return dict(start=start, end=finish, accelerationBound=accel, startError=start_error, endError=end_error)

    def fixed(location):
        return body(location, location)

    def encoded(b):
        return {**b, **{k: [{"bits": bits(x)} for x in b[k]] for k in ("start", "end")}}

    def exact_margin(first, second, radius, duration, time):
        bounds = (radius, duration, *(b[k] for b in (first, second)
                                     for k in ("accelerationBound", "startError", "endError")))
        if time != -1.0 or min(bounds) < 0:
            return None
        lower = lower_metres(closest_squared(first["start"], first["end"], second["start"], second["end"]))
        allowance = sum(Fraction(str(b["accelerationBound"])) * Fraction(str(duration))**2 / 8
                        + max(Fraction(str(b["startError"])), Fraction(str(b["endError"]))) for b in (first, second))
        margin = lower - Fraction(radius) - allowance
        return margin if margin > 0 else None

    origin = (0., 0., 0.)
    moving = body(origin, end, 15, end_error=1e-12)
    # Interior and endpoint minima, zero relative speed, both bodies moving, strict touching,
    # bad input, the micrometre floor and a large-coordinate false positive in the binary64 solver.
    cases = [
        (moving, fixed((4., 1., 0.)), .5, 1),
        (moving, fixed((4., -4., 0.)), .5, 1),
        (body((3., 0., 0.), (6., 0., 0.)), fixed(origin), 1., 1),
        (body((6., 0., 0.), (3., 0., 0.)), fixed(origin), 1., 1),
        (body(origin, (6., 0., 0.), 2, .125, .25), body((0., 3., 0.), (6., 3., 0.), 4, .5, 0), .5, 1),
        (fixed(origin), fixed((3., 4., 0.)), 4., 0),
        (body(origin, origin, 0, 0, 1), fixed((2., 0., 0.)), 1., 1),
        (fixed(origin), fixed((1., 0., 0.)), 1., 1),
        (body(origin, origin, -1), fixed((3., 0., 0.)), .5, 1),
        (fixed(origin), fixed((3., 0., 0.)), .5, -1),
        (fixed(origin), fixed((1.0000001, 0., 0.)), 1., 1),
        (body((-2.**28, 2., 0.), (2.**28, 2., 0.)), fixed(origin), 1., 1),
    ]
    requests, expected = [], []
    for first, second, radius, duration in cases:
        time = SpaceBallpark.collide_two_spheres(first["start"], first["end"], second["start"], second["end"], radius)
        expected.append((bits(time), exact_margin(first, second, radius, duration, time)))
        requests.append(dict(first=encoded(first), second=encoded(second), radius=radius, duration=duration))
    # Submit a NaN by word. The chord decoder must reject it independently of the solver's output.
    nonfinite = dict(requests[0], radius={"bits": bits(float("nan"))})
    nan_time = SpaceBallpark.collide_two_spheres(origin, end, (4., 1., 0.), (4., 1., 0.), float("nan"))
    expected.append((bits(nan_time), None))
    requests.append(nonfinite)
    rows = run_lake("env", "lean", "--run", "scripts/CurveClearance.lean",
                    *(json.dumps(x) for x in requests), capture=True).splitlines()
    for row, (time, margin) in zip(rows, expected, strict=True):
        result = json.loads(row)
        status = "insufficient clearance" if margin is None else "certified clear"
        value = Fraction(result["marginMetres"]) if "marginMetres" in result else None
        if (result["status"], result["sweepTimeBits"], value) != (status, time, margin):
            raise SystemExit(f"Curve clearance differs from the scalar sweep or exact chord minimum: {row}")
    return len(requests)


def compare_contacts(run_lake):
    _, acceleration, _, turn_end, _ = turning_sweep()
    zero = (0., 0., 0.)

    def flight(start, end, velocity=zero, applied=zero):
        return dict(start=start, end=end, velocity=velocity, applied=applied, effectiveMass=1, friction=1)

    def fixed(x, y=0.):
        return flight((x, y, 0.), (x, y, 0.))

    def straight(x, speed, duration=1):
        v = (speed, 0., 0.)
        return flight((x, 0., 0.), (x+duration*speed, 0., 0.), v, v)

    def window(start, end):
        return "first contact in window", (Fraction(start, 10**6), Fraction(end, 10**6))

    turn = flight(zero, turn_end, (10., 0., 0.), acceleration)
    def coasting(rate):
        park = Ballpark(friction=rate, tick_interval_ms=1000)
        v = (10., 0., 0.)
        end = park.integrate(zero, v, zero, 1., rate, math.exp(-rate), 1.)[0]
        return dict(flight(zero, end, v), friction=rate)

    unresolved, clear = ("unresolved", None), ("certified clear", None)
    cases = [
        ("turn", turn, fixed(4, 1), .5, 1, 10**6, window(435661, 435662)),
        ("clear control", turn, fixed(4, -4), .5, 1, 10**6, clear),
        ("head-on moving pair", straight(0, 2), straight(3, -2), 1, 1, 10**6, window(499999, 500000)),
        ("two turning bodies", turn, dict(flight(*second_turn()), effectiveMass=2),
         .5, 1, 10**6, window(404889, 404890)),
        ("initial penetration", straight(0, 4), fixed(.5), 1, 1, 10, window(0, 0)),
        ("initial touch", straight(0, -1), fixed(1), 1, 1, 10, window(0, 0)),
        ("zero duration clear", fixed(0), fixed(2), 1, 0, 1, clear),
        ("zero duration contact", fixed(0), fixed(0), 0, 0, 1, window(0, 0)),
        ("off-grid graze", straight(0, 3), fixed(1, 1), 1, 1, 1000, unresolved),
        ("on-grid graze", straight(0, 2), fixed(1, 1), 1, 1, 1000, window(499000, 500000)),
        ("parallel clear by subdivision", straight(0, 8), straight(2, 8), 1, 1, 100, clear),
        ("near obstacle", straight(0, 16, .5), fixed(3), 1, .5, 500000, window(124999, 125000)),
        ("far obstacle", straight(0, 16, .5), fixed(5), 1, .5, 500000, window(249999, 250000)),
        ("zero mass", dict(turn, effectiveMass=0), fixed(4, 1), .5, 1, 10, unresolved),
        ("special branch", dict(turn, effectiveMass=2**40), fixed(4, 1), .5, 1, 10, unresolved),
        ("negative duration", turn, fixed(4, 1), .5, -1, 10, unresolved),
        ("negative radius", turn, fixed(4, 1), -.5, 1, 10, unresolved),
        ("unknown acceleration", dict(turn, applied=[{"bits": bits(math.nan)}, 0, 0]),
         fixed(4, 1), .5, 1, 10, unresolved),
        ("coarse turning grid", turn, fixed(4, 1), .5, 1, 1, unresolved),
        ("different supplied chord", dict(turn, end=(4, 1, 0)), fixed(4, 1), .5, 1, 10**6, window(435661, 435662)),
        ("first hit after one decay time", coasting(2.), fixed(4.5), .5, 1, 10**6, window(804718, 804719)),
        ("32 decay times", coasting(32.), fixed(2), 1, 1, 100, clear),
        ("128 decay times", coasting(128.), fixed(2), 1, 1, 100, clear),
    ]
    requests = [dict(mode="firstContact", first=p, second=q, radius=r, duration=t, cells=n)
                for _, p, q, r, t, n, _ in cases]
    # Use the exact cache-derived endpoint words and the default grid on the turning case.
    requests[0] = {k: v for k, v in requests[0].items() if k != "cells"}
    requests[0]["first"] = dict(turn, end=[{"bits": bits(x)} for x in turn_end])
    rows = run_lake("env", "lean", "--run", "scripts/CurveClearance.lean",
                    *(json.dumps(request) for request in requests), capture=True).splitlines()
    windows = {}
    for (name, p, q, radius, _, _, expected), row in zip(cases, rows, strict=True):
        out = json.loads(row)
        actual = (Fraction(out["loSeconds"]), Fraction(out["hiSeconds"])) if "loSeconds" in out else None
        if (out["status"], actual) != expected:
            raise SystemExit(f"First-contact checker disagrees on {name}: {out}")
        chord = SpaceBallpark.collide_two_spheres(p["start"], p["end"], q["start"], q["end"], radius)
        decision = "undefined" if radius < 0 else "miss" if chord == -1.0 else "hit"
        if (out["sweepDecision"], out["sweepTimeBits"]) != (decision, bits(chord)):
            raise SystemExit(f"First-contact chord disagrees with the scalar engine on {name}: {out}")
        if name == "clear control" and Fraction(out["marginMetres"]) != Fraction(7, 2):
            raise SystemExit("The clear control lost its 3.5 m surface bound")
        if actual is not None:
            windows[name] = actual
    root = Fraction(turning_bracket())
    if not windows["turn"][0] < root < windows["turn"][1]:
        raise SystemExit("The high-precision turning root is outside the first-contact certificate")
    if not windows["near obstacle"][1] < windows["far obstacle"][0]:
        raise SystemExit("The certified obstacle windows overlap")
    documented = []
    for name in ("turn", "two turning bodies"):
        lo, hi = windows[name]
        documented.append(f"[{float(lo):.6f}, {float(hi):.6f}] s")
    for filename in ("DISCREPANCIES.md", "README.md"):
        page = (Path(__file__).resolve().parents[1] / filename).read_text(encoding="utf-8")
        if not all(interval in page for interval in documented):
            raise SystemExit(f"Update the certified first-contact windows in {filename}")
    return len(cases)

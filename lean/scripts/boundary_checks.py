# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Focused CPU comparisons for STOP, orbit, bounded contacts, boxes, callbacks, cleanup and accuracy."""
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import math
from unittest.mock import patch

from exact_ballpark import modes
from exact_ballpark.full import FullBallpark, OrientedBox
from exact_ballpark.ref import Ballpark
from exact_ballpark.space import CollisionItem, SpaceBallpark


def same(label, actual, expected):
    if actual != expected:
        raise SystemExit(f"{label}: got {actual!r}, expected {expected!r}")


def near(label, actual, expected, tolerance=2e-13):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise SystemExit(f"{label}: got {actual!r}, expected {expected!r} within {tolerance}")


def stop_traces():
    # Use 70-digit closed forms, including the exact two stored STOP literals.
    with localcontext() as ctx:
        ctx.prec = 70
        vertical = (1 - Decimal(1261007895663739) / 2**54) * Decimal(8417943228729899) / 2**53
        starts, speeds = [8, -4, 2], [7, 11, -3]
        endpoints = {}
        for milliseconds in (250, 500, 1000):
            p = Ballpark(friction=1, tick_interval_ms=milliseconds)
            b = p.add_ball(3, 2, 1, 12, starts, speeds, 1, 1)
            q = (-Decimal(milliseconds)/2000).exp()
            for n in range(1, 13):
                p.evolve()
                for i, d in enumerate((1, vertical, 1)):
                    gain = q*d
                    travel = 2*(1-q)*d
                    predicted = starts[i] + speeds[i]*travel*(1-gain**n)/(1-gain)
                    near(f"STOP position at {milliseconds} ms, tick {n}, axis {i}", b.pos[i], float(predicted))
                    near(f"STOP velocity at {milliseconds} ms, tick {n}, axis {i}", b.vel[i], float(speeds[i]*gain**n))
                if n*milliseconds == 1000:
                    endpoints[milliseconds] = tuple(b.vel)
        near("STOP horizontal split", endpoints[500][0], endpoints[1000][0])
        near("STOP vertical split", endpoints[500][1], float(vertical)*endpoints[1000][1])


def orbit_steering():
    p = modes.ModeBallpark(friction=1, is_master=False)
    p.trace = Counter()
    b = p.add_ball(3, 1, 1, 8, (0, 0, 0), (1, 0, 0), 1, 1)
    other = p.add_ball(4, 1, 1, 8, (100, 0, 0), (0, 0, 0), 1, 1)
    p.orbit(3, 4, 98)
    original = modes._trunc7
    seen = []
    def observe(x):
        y = original(x)
        seen.append(y)
        return y

    with localcontext() as ctx, patch.object(modes, "_trunc7", observe):
        ctx.prec = 70
        for offset in (1, 40, 400, 802, 803, 1200):
            other.pos[0] = 100 + offset
            seen.clear()
            thrust = p._evolve_old_orbit(b, 5)
            near(f"outer orbit thrust at offset {offset}", math.hypot(*thrust), 8.)
            weight = int((-Decimal(offset**2)/40000).exp()*10**7) / 10**7
            same(f"truncated orbit weight at {offset}", (len(seen), seen[-1]), (4, weight))
    same("outer-orbit branch", p.trace["orbit_old.tangent"], 6)

    # m = k = dt = 1, so free position depends on acceleration with coefficient q.
    p.use_new_orbit = True
    q = b.time_factor
    for target, w in [((1, 1, 0), 0.), ((1, 1, 0), 1.), ((1, 1, 0), 3.), ((100, 40, 0), 1.)]:
        desired = (1., .5, 0.)
        thrust = p._goto_thrust_follow(b, target, desired, w)
        reach = [(x-(1-q)*v)/q for x, v in zip(target, b.vel, strict=True)]
        match = [(x-q*v)/(1-q) for x, v in zip(desired, b.vel, strict=True)]
        raw = [(x+w*y)/(1+w) for x, y in zip(reach, match, strict=True)]
        scale = min(1., 8 / math.hypot(*raw))
        for got, value in zip(thrust, raw, strict=True):
            near("new-orbit clipped acceleration", got, scale*value)
        if scale == 1.:
            pos, vel = p.integrate(b.pos, b.vel, thrust, 1, 1, q, 1)
            for i in range(3):
                near("blended position residual", pos[i]-target[i], w/(1+w)*((1-q)*b.vel[i]+q*match[i]-target[i]))
                near("blended velocity residual", vel[i]-desired[i], (q*b.vel[i]+(1-q)*reach[i]-desired[i])/(1+w))
    same("new-orbit free and capped branches", (p.trace["gtf.free"], p.trace["gtf.capped"]), (3, 1))


def limited_contact():
    """Two touching equal-mass bodies. Read velocities actually stored by AddImpact."""
    p = SpaceBallpark(friction=1, is_master=False, iterative=True)
    a = p.add_ball(1, 1, .5, 1, True, is_massive=True, x=1, vx=-100, agility=1)
    b = p.add_ball(2, 1, .5, 2, True, is_massive=True, x=0, vx=-90, agility=1)
    ap, bp = p._properties(a), p._properties(b)
    hit = CollisionItem()
    p._next_collision_free(ap, hit, [b])
    same("initial free contact", (hit.time, hit.collider), (0., b))
    other = p._impact_free(ap, bp, hit)
    same("unlimited normal kicks", (hit.impact.dv[0], other.impact.dv[0]), (10., -10.))
    p._add_impact(other.impact, b)
    p._add_impact(hit.impact, a)
    u, v = (Fraction.from_float(prop.locations[-1].v[0]) for prop in (ap, bp))
    return [u, v, u+v-(-190), (u*u+v*v-100**2-90**2)/2]


def solid_boxes():
    layouts = [
        ((4, -2, 0), (0, 0, 6), (2, 4, 0), True),
        ((-3, 0, 0), (1, 4, 0), (2, -1, 2), False),
    ]
    for x, y, z, rectangular in layouts:
        a = (8, -7, 2)
        ob = OrientedBox()
        ob.initialize(10, 3, a, x, y, z)
        for i in range(3):
            low = a[i] + sum(min(0, arm[i]) for arm in (x, y, z))
            high = a[i] + sum(max(0, arm[i]) for arm in (x, y, z))
            same("corner bounds", (ob.aabb.low[i], ob.aabb.high[i]), (low, high))
            for face, direction in ((low, -1), (high, 1)):
                touching = list(ob.pos)
                touching[i] = face + direction*.5
                beyond = touching.copy()
                beyond[i] = math.nextafter(touching[i], math.copysign(math.inf, direction))
                same("sphere on closed face", ob.aabb.can_exclude(touching, beyond, .5), False)
                same("sphere strictly beyond face", ob.aabb.can_exclude(beyond, beyond, .5), True)
        for s, t, u in product((0, .5, 1), repeat=3):
            pt = [a[i]+s*x[i]+t*y[i]+u*z[i] for i in range(3)]
            same("solid point inside coordinate extrema", ob.aabb.can_exclude(pt, pt, 0.), False)
            if rectangular and math.dist(pt, ob.pos) > ob.shape.bounding_radius():
                raise SystemExit(f"Rectangular box point escaped the unrounded diagonal bound: {pt}")


def range_alerts(rows):
    p = FullBallpark(is_master=False)
    b = p.add_ball(3, 1, 1, 10, True)
    p.add_ball(4, 1, 2, 10, True)
    p.set_notification_range(3, 10)
    p.set_target_tracking(3, 4, 10)
    def sample():
        p.events.clear()
        p._handle_proximities()
        codes = {"DoRange": 1, "DoTargetTracking": 2}
        return [int(b.within_range), int(b.within_tracking_range),
                *(codes[name]*(1 if flag else -1) for _, name, bid, flag, *_ in p.events if bid == 3 and name in codes)]

    observed = []
    for x in (math.nextafter(11., 0.), 11., math.nextafter(11., math.inf)):
        p.set_position(4, x, 0, 0)
        p.goto_point(3, x, 0, 0)
        observed.append(sample())
    p.follow_ball(3, 4, 4)
    p.set_notification_range(3, 5)
    p.set_target_tracking(3, 0, 0)
    for x in (2., 3., 12.):
        p.set_position(4, x, 0, 0)
        observed.append(sample())
    p.set_radius(4, 0)
    p.set_position(4, 0, 0, 0)
    p.orbit(3, 4, 0)
    p.set_notification_range(3, 10)
    p.set_target_tracking(3, 4, math.nextafter(1e-5, 0.))
    observed.append(sample())
    p.stop(3)
    p.set_target_tracking(3, 4, 1e-5)
    observed.append(sample())
    same("ordered range and tracking trace", observed, rows)
    p.goto_point(3, 0, 0, 0)
    same("new range entry", sample(), [1, 1, 1])
    p.set_notification_range(3, 10)
    same("notification setter re-arms entry", sample(), [1, 1, 1])
    p.stop(3)
    same("STOP exits once", (sample(), sample()), ([0, 1, -1], [0, 1]))


def cleanup(cohorts, polls, masks):
    ids = list(range(3, 20))
    def pending(age):
        p = FullBallpark(is_master=False)
        for bid in reversed(ids):
            p.add_ball(bid, 1, 1, 1, True)
            p.remove_ball(bid, 2)
        p.current_time += age
        return p

    for left, *survivors in cohorts:
        p = pending(2-left)
        p.bring_out_the_deads()
        same(f"survivors with {left} ticks left", sorted(p.balls), survivors)
    p = pending(1)
    stamp = (p.current_time, p.time)
    for survivors in polls:
        p.on_tick(p.time)
        same("early poll preserves both clocks", (p.current_time, p.time), stamp)
        same("early poll survivor IDs", sorted(p.balls), survivors)
    p = FullBallpark(is_master=False)
    # Mix stale, overdue and early entries in the same iteration, with a late ID overdue.
    for i in reversed(range(len(masks))):
        b = p.add_ball(i+3, 1, 1, 1, True)
        if i:
            p.remove_ball(b.id, 1)
            b.effect_stamp = p.current_time + (-1 if i == 16 else 2 if i % 4 == 3 else i % 2)
            if i == 2:
                b.is_moribund = False
    p.bring_out_the_deads()
    for i, (clear, killed) in enumerate(masks):
        same(f"mixed cleanup row {i}", (bool(i) and i+3 not in p.moribund, i+3 not in p.balls),
             (bool(clear), bool(killed)))


def falloff():
    p = FullBallpark(is_master=False)
    p.add_ball(3, 1, 1, 10)
    p.add_ball(4, 1, 2, 10)
    for angular, radial in product((0., 1., 2.), (0., 3.)):
        for d in (2., 7., 8., 9., 11.):
            p.set_position(4, d, 0, 0)
            p.set_velocity(4, radial, angular*d, 0)
            chance, short, speed = p.get_accuracy(3, 4, 4, 2, 1, 1, 1)
            expected_short = max(0., d-3)
            cost = angular**2+(max(0., expected_short-4)/2)**2
            same("surface and angular terms", (short, speed), (expected_short, angular))
            near("range and tracking probability", chance, 2**(-cost), 1e-15)
            same("half chance threshold", chance >= .5, cost <= 1)


def compare(run_lake):
    records = defaultdict(list)
    for line in run_lake("env", "lean", "--run", "scripts/BoundaryReplay.lean", capture=True).splitlines():
        name, *values = line.split()
        records[name].append([Fraction(x) if name == "impact" else int(x) for x in values])
    same("boundary replay row counts", {k: len(v) for k, v in records.items()},
         {"impact": 1, "alerts": 8, "cohort": 4, "poll": 3, "mask": 19})
    stop_traces()
    orbit_steering()
    same("limited pair against the rational Lean witness", limited_contact(), records["impact"][0])
    solid_boxes()
    range_alerts(records["alerts"])
    cleanup(records["cohort"], records["poll"], records["mask"])
    falloff()

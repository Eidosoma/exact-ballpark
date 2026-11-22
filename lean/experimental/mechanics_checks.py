# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Numerical controls for the ten mechanics experiments in the experiment register."""
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import math
import struct
from unittest.mock import patch

from boundary_checks import limited_contact
from exact_ballpark import full
from exact_ballpark.modes import ModeBallpark
from exact_ballpark.ref import Ballpark, STOP_DAMP
from exact_ballpark.space import CollisionItem, FUZZY_DELTA, SpaceBallpark


def bits(value):
    """Compare finite words and signed zero exactly, with one encoding for all NaNs."""
    return int.from_bytes(struct.pack(">d", math.nan if math.isnan(value) else value), "big")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, label, *, atol=2e-12, rtol=3e-14):
    check(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
          f"{label}: got {actual!r}, expected {expected!r}")


def warp_units():
    # Rebuild the profile in kilometres from independent decimal quantities.
    metre = Decimal("0.001")
    a, b = Decimal(3), Decimal(1)
    cruise_speed = 3 * Decimal(149597870700) * metre
    for metres, short in [(1000, True), (10**14, False)]:
        d = metres * metre
        s = min(cruise_speed, (d + metre)*a*b/(a+b))
        check((s < cruise_speed) == short, "Warp controls stopped spanning both peak branches")
        tc = d/s - 1/a - 1/b + metre/s
        control = ((s/(metre*a)).ln()/a, tc, (s/(metre*b)).ln()/b, s/a, s*tc, s/b-metre, a, b, s)
        engine = ModeBallpark.warp_constants(3000, metres)
        for i, (actual, reference) in enumerate(zip(engine, control, strict=True)):
            converted = actual / 1000 if i in (3, 4, 5, 8) else actual
            near(converted, float(reference), f"Warp distance {metres}, field {i}")
        if short:
            short_peak = engine[-1]
            bare = float((d + 1)*a*b/(a+b))
            check(bare != float(s), "Bare kilometre offset unexpectedly agrees with the unit-aware control")
    return (f"All nine converted outputs agree to the replay tolerance at 1,000 m and 10¹⁴ m. The short peak is "
            f"{short_peak:g} m/s. Using a bare 1 in kilometres gives {bare:g} km/s.")


def goto_clearance():
    def turn(p, ship, distance, expected_thrust):
        p.goto_point(ship.id, 0, distance, 0)
        a = p.goto_thrust(ship)
        near(a[1], expected_thrust, "GOTO thrust on full/tapered command")
        mass = ship.mass * ship.agility
        rate = p.friction / mass
        bound = rate * (ship.max_vel * ship.speed_fraction + math.hypot(*ship.vel)) * p.dt**2 / 8
        def position(t):
            return p.integrate(ship.pos, ship.vel, a, mass, p.friction, ship.time_factor, t)[0]
        end = position(p.dt)
        amplitude = math.hypot(*[v-g/rate for v, g in zip(ship.vel, a, strict=True)]) / rate
        deviations = []
        for i in range(17):
            u = i/16
            point = position(u*p.dt)
            chord = [x*u for x in end]
            delta = math.dist(point, chord)
            error = amplitude * abs(-math.expm1(-rate*u*p.dt) + u*math.expm1(-rate*p.dt))
            near(delta, error, "GOTO chord distance")
            check(delta <= bound, "Ship-parameter chord bound exceeded")
            deviations.append(delta)
        return bound, deviations[8]

    p = Ballpark(friction=1)
    ship = p.add_ball(1, 1, 0, 10, (0, 0, 0), (10, 0, 0), 1, 1)
    bound, full_turn = turn(p, ship, 100, 10.)
    _, tapered = turn(p, ship, 5, .625)
    half = Ballpark(friction=2, tick_interval_ms=500)
    other = half.add_ball(2, 2, 0, 8, (0, 0, 0), (6, 0, 0), 2, .5)
    other_bound, _ = turn(half, other, 10, 2.)
    turn(half, other, 1, .125)
    return (f"The 1 s ship has a {bound:g} m bound. Full and tapered turns deviate "
            f"{full_turn:.6f} m and {tapered:.6f} m at mid-tick. The 0.5 s ship has a {other_bound:g} m bound.")


def history_travel():
    near(FUZZY_DELTA, .0001, "History time tolerance", atol=0, rtol=0)
    inside = math.nextafter(FUZZY_DELTA, 0.)
    for second_speed in (-250., 750.):
        p = SpaceBallpark(friction=1, is_master=False, iterative=True)
        paths = []
        for i, v in enumerate((750., second_speed)):
            b = p.add_ball(i+1, 1, 0, abs(v), True, agility=1, x=10*i, vx=v)
            b.last_g = [v, 0., 0.]  # Applied force balances drag throughout this interval.
            paths.append(p._properties(b))
        def separation(t):
            return paths[1].position(t)[0][0] - paths[0].position(t)[0][0]
        check(all(path.position(inside) == path.position(0.) for path in paths),
              "History stopped holding the stored state inside the tolerance")
        closed = separation(0) - separation(FUZZY_DELTA)
        near(closed, (750-second_speed)*FUZZY_DELTA, "Relative travel at the integration threshold")
        if second_speed < 0:
            opposed = closed
        else:
            together = closed
    return (f"At 100 µs the opposed 750 and 250 m/s bodies close {opposed:.3f} m. "
            f"Two co-moving bodies close {together:.3f} m. One binary64 step below, history returns the stored positions.")


def elastic_impulse():
    impulses = []
    for m, n, u, v in [(1, 1, -3, 5), (2, 6, -3, 5), (6, 2, 5, 9)]:
        p = SpaceBallpark(friction=1, is_master=False, iterative=True)
        first = p.add_ball(1, m, .5, 100, True, is_massive=True, agility=2, x=1, vx=u)
        second = p.add_ball(2, n, .5, 100, True, is_massive=True, agility=.5, x=0, vx=v)
        ap, bp = p._properties(first), p._properties(second)
        hit = CollisionItem()
        p._next_collision_free(ap, hit, [second])
        check(hit.collider is second and hit.time == 0., "Missing initial pair contact")
        other = p._impact_free(ap, bp, hit)
        j = Fraction(2*m*n*(v-u), m+n)
        w = u + Fraction(hit.impact.dv[0])
        z = v + Fraction(other.impact.dv[0])
        check(m*(w-u) == j == n*(v-z) and w-z == v-u, "Raw pair impulse or restitution changed")
        check(m*w*w+n*z*z == m*u*u+n*v*v, "Uncapped contact energy changed")
        impulses.append(str(j))
    return f"Three approaching pairs give impulses {', '.join(impulses)} kg·m/s. Each leaves with restitution 1."


def impact_energy():
    # Reuse the actual stored velocities from the existing one-sided limiter replay.
    w, z, momentum_change, lab = limited_contact()
    u, v = Fraction(-100), Fraction(-90)
    shifted = ((w+100)**2 + (z+100)**2 - (u+100)**2 - (v+100)**2)/2
    internal = ((w-z)**2 - (u-v)**2)/4
    check(shifted == lab + 100*momentum_change and
          (lab, shifted, internal) == (Fraction(5625, 8), Fraction(-375, 8), Fraction(-375, 16)),
          "Capped pair energy changed in one of the three comparisons")
    return (f"Lab ΔK {float(lab):+.3f} J, velocities shifted by +100 m/s {float(shifted):+.3f} J, "
            f"and internal ΔK {float(internal):+.4f} J.")


def tracking_frames(lean_rows, cpp_rows):
    def observe(r, v, offset=(0, 0, 0), boost=(0, 0, 0)):
        p = full.FullBallpark()
        for i, (pos, vel) in enumerate([((0, 0, 0), (0, 0, 0)), (r, v)]):
            x, y, z = [a+b for a, b in zip(pos, offset, strict=True)]
            p.add_ball(i+1, 1, 1, 10, True, x=x, y=y, z=z)
            p.set_velocity(i+1, *(a+b for a, b in zip(vel, boost, strict=True)))
        with patch.object(full, "_sqrt", wraps=full._sqrt) as root:
            output = p.get_accuracy(1, 2, 100, 1, 1, 1, 1)
        check(root.call_count == 1, "Tracking query missed its ordinary branch")
        return output, root.call_args.args[0]

    r, v = (3, 4, 12), (2, -3, 1)
    base = observe(r, v)
    transformed = observe(r[1:]+r[:1], v[1:]+v[:1], (32, 16, -8), (-4, 8, 2))
    check(transformed == base and base[0][2] > 0, "Dyadic tracking control changed under the frame transform")
    v = (.03, .04, .12)
    for name, axes in [("xyz", (0, 1, 2)), ("yzx", (1, 2, 0))]:
        sight, velocity = tuple(r[i] for i in axes), tuple(v[i] for i in axes)
        result, square = observe(sight, velocity)
        word = bits(square)
        check(result[:2] == (1., 11.), f"Changed tracking chance or range in {name}")
        check((math.isnan(result[2]) and square < 0) if name == "xyz" else
              (word == 0 and bits(result[2]) == 0), f"Changed tracking projection in {name}")
        lean_rows.append((f"A,{name}", str(word)))
        cpp_rows.append(("A " + " ".join(map(str, (*sight, *velocity))), (word, *map(bits, result))))
    return ("The translated, boosted and rotated dyadic control agrees exactly. Cycling the near-radial "
            "input from xyz to yzx changes the reported angular speed from NaN to +0 rad/s.")


def drag_frame(step):
    m, k, v, a, boost = 2., 1., 3., .75, 8.
    cached = math.exp(-k/m)
    for t in (.25, 1., 2.):
        q = math.exp(-k*t/m)
        base = step(2., v, a, m, k, cached, t)
        fixed = step(2., v+boost, a, m, k, cached, t)
        moving = step(2., v+boost, a+k/m*boost, m, k, cached, t)
        near(fixed[1]-base[1], q*boost, "Velocity offset in the fixed medium")
        near(fixed[0]-base[0], m/k*(1-q)*boost, "Position offset in the fixed medium")
        near(moving[1]-base[1], boost, "Velocity offset in the moving-medium control")
        near(moving[0]-base[0], boost*t, "Position offset in the moving-medium control")
        if t == 1:
            fixed_speed, moving_speed = fixed[1]-base[1], moving[1]-base[1]
    return (f"An 8 m/s boost leaves {fixed_speed:.6f} m/s after one second in the fixed medium. "
            f"The moving-medium force control retains {moving_speed:g} m/s.")


def orbit_budget():
    measurements = []
    for radius in (10, 40, 100):
        p = ModeBallpark(friction=1, is_master=False)
        p.use_new_orbit, p.trace = True, Counter()
        ship = p.add_ball(1, 2, 1, 20, (radius, 0, 0), (0, 8, 0), 1, 1)
        p.add_ball(2, 2, 1, 0, (0, 0, 0), (0, 0, 0), 1, 1)
        p.orbit(1, 2, radius-2)
        rate = p.friction/(ship.mass*ship.agility)
        budget = rate*ship.max_vel*ship.speed_fraction
        # Rationalized positive quadratic root in speed squared.
        ceiling = math.sqrt(2*budget**2/(rate**2 + math.hypot(rate**2, 2*budget/radius)))
        with patch.object(p, "_goto_thrust_follow", wraps=p._goto_thrust_follow) as guide:
            p.evolve()
        check(p.trace["orbit_new.circle"] == 1 and guide.call_count == 1, "Orbit missed the circular guidance branch")
        requested = math.hypot(*guide.call_args.args[2])
        required = math.hypot(requested**2/radius, rate*requested)
        near(math.hypot(ceiling**2/radius, rate*ceiling), budget, "Orbit force balance at the ceiling")
        near(requested, min(20., math.sqrt(radius*budget)), "Circular guidance target speed")
        check(requested > ceiling and required > budget and p.trace["gtf.capped"] == 1,
              "Orbit reference-speed and actual thrust controls changed")
        near(math.hypot(*ship.last_g), budget, "Clipped circular steering command")
        measurements.append((ceiling, requested, required))
    ceiling, requested, required = measurements[0]
    return (f"At radius 10 m the steady ceiling is {ceiling:.6f} m/s. The controller requests "
            f"{requested:g} m/s, which would require {required:.6f} m/s² on a circle with a 10 m/s² budget.")


def stop_rate(cpp_rows):
    vertical = (1-.07)*STOP_DAMP
    fitted = -math.log(vertical)  # Per second, fitted to the extra multiplier at a 1 s tick.
    velocities = {}
    for tick in (1000, 500, 250):
        p = Ballpark(friction=1, tick_interval_ms=tick)
        ship = p.add_ball(1, 2, 0, 10, (0, 0, 0), (10, 10, 0), 1, 1)
        for _ in range(1000//tick):
            p.evolve()
        near(ship.vel[0], 10*math.exp(-.5), "STOP horizontal control")
        near(ship.vel[1], ship.vel[0]*vertical**(1000//tick), "Vertical loss per STOP command tick")
        velocities[tick] = ship.vel[1]
        cpp_rows.append((f"S {tick} {1000//tick}", tuple(map(bits, (*ship.pos[:2], *ship.vel[:2])))))
    near(velocities[1000], 10*math.exp(-(.5+fitted)), "Continuous STOP fit at one second")
    check(velocities[250] < velocities[500] < velocities[1000], "STOP subdivision lost its extra damping")
    return (f"After one second the vertical speeds are {velocities[1000]:.6f}, {velocities[500]:.6f} "
            f"and {velocities[250]:.6f} m/s for 1, 2 and 4 ticks. The one-tick fit is γ = {fitted:.6f} s⁻¹.")


def resistance(data):
    products = []
    for sphere in data["sheffield"]["spheres"]:
        re = Decimal(sphere["reported_reynolds"])
        cd = Decimal(sphere["reported_drag_coefficient"])
        # Half a unit of the last printed digit, including trailing zeros.
        half_re = Decimal(1).scaleb(re.as_tuple().exponent)/2
        half_cd = Decimal(1).scaleb(cd.as_tuple().exponent)/2
        low, high = (re-half_re)*(cd-half_cd), (re+half_re)*(cd+half_cd)
        check(low > 24 and low <= re*cd <= high, f"Stokes printed-precision comparison changed for {sphere['id']}")
        products.append(f"{sphere['id']} {re*cd:.2f} ({low:f}–{high:f})")
    return (f"Stokes predicts Cᴅ Re = 24. Sheffield's printed products are {' and '.join(products)}. "
            "Parentheses give the endpoints from printed rounding.")


def experiments(data, step, lean_rows, cpp_rows):
    cases = [
        ("Warp length units", warp_units()),
        ("GOTO sweep clearance", goto_clearance()),
        ("Travel in a history tolerance", history_travel()),
        ("Unique elastic impulse", elastic_impulse()),
        ("Capped impact energy", impact_energy()),
        ("Tracking across frames", tracking_frames(lean_rows, cpp_rows)),
        ("Drag in a moving medium", drag_frame(step)),
        ("Sustainable circular speed", orbit_budget()),
        ("STOP damping per second", stop_rate(cpp_rows)),
        ("Sphere resistance", resistance(data)),
    ]
    return [f"| {name} | {finding} |" for name, finding in cases]
